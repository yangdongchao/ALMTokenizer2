''' Audio Diffusion model based on flow-matching
    Part of the code is based on MuCodec
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
import random
import inspect
import numpy as np
from tqdm import tqdm
import typing as tp
from abc import ABC
import os
import torchaudio
from einops import repeat
import diffusers
from diffusers.utils.torch_utils import randn_tensor
from diffusers import DDPMScheduler
from models.processor import Feature2DProcessor 
from models.transformer_1d_flow import Transformer1DModel, ProjectLayer
from transformers import AutoFeatureExtractor, Wav2Vec2BertModel,HubertModel
from transformers import AutoModel
from torch.cuda.amp import autocast
from modules.our_MERT_BESTRQ.test import load_best_rq_model
from models.PretrainedModel import BESTRQ_Model
from vector_quantize_pytorch import ResidualVQ
from whisper.audio import log_mel_spectrogram
import whisper
from models.modeling_whisper import WhisperModel
from typing import Dict, Iterable, Optional, List
from transformers import WhisperFeatureExtractor
from models.vocos import VocosBackbone
from models.semantic_decoder import Decoder
from modules.transformer import TransformerBlock
from models.gptc import GPTC, GPTCConfig


def GPTC_XXS(n_embd):
    return GPTC(GPTCConfig(n_layer=6, n_head=4, n_embd=n_embd)) # 5.0M


class BASECFM(torch.nn.Module, ABC):
    def __init__(
        self,
        estimator,
    ):
        super().__init__()
        self.sigma_min = 1e-4
        self.estimator = estimator

    @torch.inference_mode()
    def forward(self, mu, n_timesteps, temperature=1.0):
        """Forward diffusion

        Args:
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            n_timesteps (int): number of diffusion steps
            temperature (float, optional): temperature for scaling noise. Defaults to 1.0.

        Returns:
            sample: generated mel-spectrogram
                shape: (batch_size, n_feats, mel_timesteps)
        """
        z = torch.randn_like(mu) * temperature
        t_span = torch.linspace(0, 1, n_timesteps + 1, device=mu.device)
        return self.solve_euler(z, t_span=t_span)

    def solve_euler(self, x, incontext_x, incontext_length, t_span, mu, added_cond_kwargs, guidance_scale):
        """
        Fixed euler solver for ODEs.
        Args:
            x (torch.Tensor): random noise
            t_span (torch.Tensor): n_timesteps interpolated
                shape: (n_timesteps + 1,)
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
        """
        t, _, dt = t_span[0], t_span[-1], t_span[1] - t_span[0]
        noise = x.clone()

        # I am storing this because I can later plot it by putting a debugger here and saving it to a file
        # Or in future might add like a return_all_steps flag
        sol = []
        print('x ', x.shape, incontext_x.shape, mu.shape)
        for step in tqdm(range(1, len(t_span))):
            x[:,0:incontext_length,:] = (1 - (1 - self.sigma_min) * t) * noise[:,0:incontext_length,:] + t * incontext_x[:,0:incontext_length,:]
            if(guidance_scale > 1.0):
                dphi_dt = self.estimator( \
                    torch.cat([ \
                        torch.cat([x, x], 0), \
                        torch.cat([incontext_x, incontext_x], 0), \
                        torch.cat([torch.zeros_like(mu), mu], 0), \
                        ], 2), \
                timestep = t.unsqueeze(-1).repeat(2), \
                added_cond_kwargs={k:torch.cat([v,v],0) for k,v in added_cond_kwargs.items()}).sample
                dphi_dt_uncond, dhpi_dt_cond = dphi_dt.chunk(2,0)
                dphi_dt = dphi_dt_uncond + guidance_scale * (dhpi_dt_cond - dphi_dt_uncond)
            else:
                dphi_dt = self.estimator(torch.cat([x, incontext_x, mu], 1), \
                timestep = t.unsqueeze(-1),
                added_cond_kwargs=added_cond_kwargs).sample

            x = x + dt * dphi_dt
            t = t + dt
            sol.append(x)
            if step < len(t_span) - 1:
                dt = t_span[step + 1] - t

        return sol[-1]

    def compute_loss(self, x1, mu, added_cond_kwargs, latent_masks, validation_mode=False):
        """Computes diffusion loss

        Args:
            x1 (torch.Tensor): Target
                shape: (batch_size, n_feats , T)
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, T)

        Returns:
            loss: conditional flow matching loss
            y: conditional flow
                shape: (batch_size, n_feats, T)
        """
        b = mu[0].shape[0]

        # random timestep
        if(validation_mode):
            t = torch.ones([b, 1, 1], device=mu[0].device, dtype=mu[0].dtype) * 0.5
        else:
            t = torch.rand([b, 1, 1], device=mu[0].device, dtype=mu[0].dtype)
        # sample noise p(x_0)
        z = torch.randn_like(x1)

        y = (1 - (1 - self.sigma_min) * t) * z + t * x1
        u = x1 - (1 - self.sigma_min) * z
        # print('y ', y.shape)
        # print('t ', t.shape)
        out = self.estimator(
            torch.cat([y, *mu],2), 
            timestep = t.squeeze(-1).squeeze(-1),
            added_cond_kwargs=added_cond_kwargs,
        ).sample

        weight = (latent_masks > 1.5).unsqueeze(-1).repeat(1, 1, out.shape[-1]).float() + (latent_masks < 0.5).unsqueeze(-1).repeat(1, 1, out.shape[-1]).float() * 0.01
        loss = F.mse_loss(out * weight, u * weight, reduction="sum") / weight.sum()
        return loss

class AudioDiffusion1D(nn.Module):
    def __init__(
        self,
        num_channels,
        pre_trained_model_name = 'whisper&bestrq',
        features_type = 'continuous',
        statistical_prior = '/home/ydc/music/AnyToken2Audio2/stats_20k.pth',
        vq_training = True,
        unet_model_name=None,
        unet_model_config_path=None,
        whisper_path=None,
        snr_gamma=None,
        uncondition=True,
        out_paint=False,
        fine_decoder=False,
        use_gpt_loss = False
    ):
        super().__init__()
        self.unet_model_name = unet_model_name
        self.unet_model_config_path = unet_model_config_path
        self.snr_gamma = snr_gamma
        self.uncondition = uncondition
        self.num_channels = num_channels
        self.features_type = features_type
        self.max_t_len = 30*50 # set the max audio seqence as 30 seconds
        self.codec_dim = 768
        self.sample_rate = 24000 # the sample rate for mu_encoder is 24k hz
        self.whisper_encoder = WhisperModel.from_pretrained(whisper_path).encoder
        self.whisper_fea_dim = self.whisper_encoder.config.d_model
        self.wavlm_fea_dim = 768 # ?
        for param in self.whisper_encoder.parameters():
            param.requires_grad = False
        self.wavlm_encoder = AutoModel.from_pretrained("/turing_music_fs/music_data/ckpts/ckpts/wavlm")
        self.wavlm_transfer = torchaudio.transforms.Resample(24000, 16000)
        for param in self.wavlm_encoder.parameters():
            param.requires_grad = False
        self.pretrained_model = BESTRQ_Model(model_dir = 'modules/our_MERT_BESTRQ/mert_fairseq', 
                                checkpoint_dir = '/turing_music_fs/music_data/ckpts/ckpts/ssl.pt', 
                                output_features = features_type, layers = [4, 11])
        for v in self.pretrained_model.parameters():
            v.requires_grad = False 
        self.d_conv_whisper = nn.Conv1d(in_channels=self.whisper_fea_dim, out_channels=self.whisper_fea_dim, kernel_size=2,
                                        stride=2, padding=0, bias=True)
        self.d_conv_wavlm = nn.Conv1d(in_channels=self.wavlm_fea_dim, out_channels=self.wavlm_fea_dim, kernel_size=2,
                                        stride=2, padding=0, bias=True)
        self.pronunciation_decoder = Decoder(code_dim=self.codec_dim, output_channels=self.wavlm_fea_dim, decode_channels=self.wavlm_fea_dim, strides=[2, 2])
        
        if fine_decoder:
            for v in self.d_conv.parameters():
                v.requires_grad = False

        if features_type == 'continuous':
            if use_gpt_loss:
                from vector_quantize_pytorch_core import MultiScaledResidualVQ
                # if we decide to use GPT loss to optimize the codec, we must use MultiScaledResidualVQ
                self.vq_embed = MultiScaledResidualVQ(
                    dim = self.codec_dim,
                    codebook_size = [8192]*8, # codebook size
                    decay = 0.9, # the exponential moving average decay, lower means the dictionary will change faster
                    commitment_weight = 1.,   # the weight on the commitment loss
                    rotation_trick = True,
                    use_cosine_sim = False,
                    codebook_dim = 32,
                    implicit_neural_codebook=False,
                    num_quantizers= 8,
                    vq_strides= [1]*8)
            else:
                self.vq_embed = ResidualVQ(
                    dim = self.codec_dim,
                    codebook_size = 8192, # codebook size
                    decay = 0.9, # the exponential moving average decay, lower means the dictionary will change faster
                    commitment_weight = 1.,   # the weight on the commitment loss
                    threshold_ema_dead_code = 2,
                    use_cosine_sim = False,
                    codebook_dim = 32,
                    num_quantizers= 8,
                )

            if fine_decoder:
                for v in self.vq_embed.parameters():
                    v.requires_grad = False 
        else:
            # for discrete tokens, we use a nn.Embedding to map the token into features
            # not implement now. 2025. 2.15
            self.vq_embed = None
        if unet_model_config_path:
            #self.phone_decoder = DecoderV5(code_dim=2048, output_channels=1536, decode_channels=2048, strides=[2, 2])
            self.cond_fusion_layer_semantic = nn.Linear(1024, self.codec_dim)
            self.cond_fusion_layer_acoustic = nn.Linear(1024+self.whisper_fea_dim, self.codec_dim)
            self.cond_fusion_layer_phone = nn.Linear(self.wavlm_fea_dim, self.codec_dim)
            self.feature_proj = nn.Linear(self.codec_dim*3, self.codec_dim)
            self.cond_feature_emb = nn.Linear(self.codec_dim, self.codec_dim)
            #self.cond_feature_emb = ProjectLayer(hidden_size=512, filter_size=512, kernel_size=5)
            self.zero_cond_embedding1 = nn.Parameter(torch.randn(self.codec_dim))
            unet = Transformer1DModel.from_config(unet_model_config_path)
            self.set_from = "random"
            self.cfm_wrapper = BASECFM(unet)
            print("Transformer initialized from pretrain.")

        self.cls_token = nn.Parameter(torch.randn(1, self.codec_dim), requires_grad=True) # init the query token
        self.mask_token = nn.Parameter(torch.randn(1, self.codec_dim), requires_grad=True)
        self.interval = 2 # we fix it. also, we can use dynamic interval
        encoder_depth = 6 # small version
        decoder_depth = 8 # small version. using large decoder can bring better performance
        en_transformers = []
        power_normalized = True
        for _ in range(encoder_depth):
            en_transformers.append(TransformerBlock(self.codec_dim, dim_heads = 128, causal = False, zero_init_branch_outputs = False if power_normalized else True, 
                                                 remove_norms = False, power_normalized = power_normalized, conformer = False, layer_scale = True, 
                                                 add_rope = True, attn_kwargs={'qk_norm': True},  ff_kwargs={'mult': 4, 'no_bias': False}, norm_kwargs = {'eps': 1e-2}))
        self.encoder_transformers = nn.Sequential(*en_transformers)

        de_transformers = []
        power_normalized = True
        for _ in range(decoder_depth):
            de_transformers.append(TransformerBlock(self.codec_dim, dim_heads = 128, causal = False, zero_init_branch_outputs = False if power_normalized else True, 
                                                 remove_norms = False, power_normalized = power_normalized, conformer = False, layer_scale = True, 
                                                 add_rope = True, attn_kwargs={'qk_norm': True},  ff_kwargs={'mult': 4, 'no_bias': False}, norm_kwargs = {'eps': 1e-2}))
        self.decoder_transformers = nn.Sequential(*de_transformers)
        self.decoder_pred = nn.Linear(self.codec_dim, self.codec_dim, bias=True) 
        if use_gpt_loss:
            self.depth_gpt = GPTC_XXS(self.codec_dim)
            #gpt_loss = self.depth_gpt.compute_prior_loss(cat_quantized)


    def get_fsdp_wrap_module_list(self) -> List[nn.Module]:
        return list(self.cfm_wrapper.estimator.transformer_blocks) 

    def get_whisper_feature(self, mels, n_len):
        with torch.no_grad():
            # print('mel ', mel.shape)
            n_len = int((n_len/24000)*50)
            whisper_embeds = self.whisper_encoder(mels, return_dict=True).last_hidden_state # get whisper features
            return whisper_embeds[:,:n_len,:].transpose(1,2) # b, t, 512

    def get_wavlm_feature(self, wav_24k):
        ''' wav: (B, 1, T)
            return: B, T//320, self.wavlm_dim*2
        '''
        wav_16k = self.wavlm_transfer(wav_24k).squeeze(1)
        wav_16k = torch.cat([wav_16k, torch.zeros(wav_16k.shape[0], 160).to(wav_16k.device)], dim=-1)
        target = self.wavlm_encoder(wav_16k, output_hidden_states=True).hidden_states
        target = torch.stack(target, dim=1)
        target = target[:,6:10,:].mean(1).transpose(1,2) # we choose the 6~9 layers as the conidtional phone-level information
        return target
    
    def L2_loss(self, feature, target_feature):
        """
        feature: B, T, D
        target_feature: B, T ,D
        return the mse loss
        """
        n = min(feature.size(2), target_feature.size(2))
        return F.mse_loss(target_feature[:,:,:n], feature[:,:,:n])

    def set_masking(self, x):
        """
        Perform per-sample random masking 
        x: [N, T, D], sequence, we put mask token 
        we set the mask rate as 0.8, so we put a mask every 5 frames. And we sure the total frame is 5 times
        """
        B, T, D = x.shape  # batch, length, dim
        # print('x ', x.shape, self.audio_thinking.interval)
        cls_token = self.cls_token.repeat(1, (T//self.interval), 1) # repeat 
        mask = cls_token.repeat(B, 1, 1)  # 扩展mask token以匹配batch size
        new_T = T + (T // self.interval)
        x_reshaped = x.reshape(B, T // self.interval, self.interval, D)
        
        mask_tokens = mask.unsqueeze(2) #.repeat(1, T // self.interval, 1, 1) # B, a,1,D

        x_with_masks = torch.cat([x_reshaped, mask_tokens], dim=2)

        new_x = x_with_masks.reshape(B, -1, D)

        return new_x

    def extract_mask_positions(self, new_x):
        B, new_T, D = new_x.shape
        original_T = new_T - new_T // (self.interval + 1)
        
        mask_indices = [(i + 1) * (self.interval + 1) - 1 for i in range(original_T // self.interval)]
        mask_positions = new_x[:, mask_indices, :]

        return mask_positions

    def extract_non_mask_positions(self, new_x):
        B, new_T, D = new_x.shape
        num_masks = new_T // (self.interval + 1)
        original_T = new_T - num_masks

        # 生成一个索引列表，排除 mask 的位置
        mask_indices = [(i + 1) * (self.interval + 1) - 1 for i in range(num_masks)]
        all_indices = list(range(new_T))
        non_mask_indices = [i for i in all_indices if i not in mask_indices]

        non_mask_positions = new_x[:, non_mask_indices, :]

        return non_mask_positions
    
    def replace_mask(self, seq):
        B, T, D = seq.size()
        min_replace = int(0.1 * T)
        max_replace = int(0.2 * T)
        num_to_replace = torch.randint(min_replace, max_replace + 1, (1,)).item()
        indices = torch.randint(0, T, (num_to_replace,))
        for b in range(B):
            seq[b, indices] = self.cls_mask.squeeze(0)
        return seq

    def set_decoder_mask(self, en_token):
        """
        en_token is B, T, D
        """
        B, T, D = en_token.shape
        x = self.mask_token.repeat(B, 1, 1) # B, 1, D
        new_T = en_token.shape[1]*self.interval + en_token.shape[1]
        x = x.repeat(1, en_token.shape[1]*self.interval, 1) # B, n, D
        x = x.reshape(B, -1, self.interval, D) # B, -1, interval, D
        en_token = en_token.unsqueeze(2) # B, interval_num, 1 D
        x_with_masks = torch.cat([x, en_token], dim=2)
        new_x = x_with_masks.reshape(B, -1, D) # 将新的tensor reshape回 (B, new_T, D)
        return new_x

    def forward_decoder(self, x):
        # embed tokens
        de_mask = self.set_decoder_mask(x) # 
        de_mask = self.decoder_transformers(de_mask)
        de_non_mask = self.extract_non_mask_positions(de_mask) # get the pure features
        de_non_mask = self.decoder_pred(de_non_mask)
        return de_non_mask

    def forward(self, input_audios, spectrograms, latents, latent_masks, validation_mode=False, additional_feats = ['no'], train_rvq=True, train_ssl=False):
        ''' make sure input_audios is single-channel audio
        '''
        if not hasattr(self,"device"):
            self.device = input_audios.device
        if not hasattr(self,"dtype"):
            self.dtype = input_audios.dtype
        device = self.device

        with torch.no_grad():
            bestrq_emb_acoustic, bestrq_emb_semantic = self.pretrained_model.extract_continous_embeds_multiple(input_audios.clone())
            bestrq_emb_acoustic = bestrq_emb_acoustic.detach()
            bestrq_emb_semantic = bestrq_emb_semantic.detach()
            semantic_target = bestrq_emb_semantic.clone()
            whisper_embeds = self.get_whisper_feature(spectrograms, input_audios.shape[-1])
            whisper_embeds = whisper_embeds.detach()
            wavlm_embeds = self.get_wavlm_feature(input_audios) # 
            wavlm_embeds = wavlm_embeds.detach()
            
            

        wavlm_target = wavlm_embeds.clone()
        whisper_embeds = self.d_conv_whisper(whisper_embeds) #.transpose(1,2)
        wavlm_encoder_features = self.d_conv_wavlm(wavlm_embeds) 
        features_emb_semantic = bestrq_emb_semantic
        features_emb_acoustic = bestrq_emb_acoustic
        features_emb_phone = self.cond_fusion_layer_phone(wavlm_encoder_features.transpose(1,2)).transpose(1,2)
        features_emb_semantic = self.cond_fusion_layer_semantic(features_emb_semantic.transpose(1, 2)).transpose(1, 2)
        min_f_len = min(features_emb_acoustic.shape[-1], whisper_embeds.shape[-1])
        features_emb_acoustic = torch.cat([features_emb_acoustic[:, :, :min_f_len], whisper_embeds[:, :, :min_f_len]], dim=1)  # concat 
        features_emb_acoustic = self.cond_fusion_layer_acoustic(features_emb_acoustic.transpose(1, 2)).transpose(1, 2)
        #features_emb_semantic = features_emb_semantic[:,:,:min_f_len]
        total_feature = torch.cat([features_emb_phone, features_emb_semantic, features_emb_acoustic], dim=1)
        total_feature = self.feature_proj(total_feature.transpose(1, 2))

        # add query-based strategy
        query_embeds_llm = self.set_masking(total_feature) # add the query token
        query_embeds_llm = self.encoder_transformers(query_embeds_llm)
        query_tokens = self.extract_mask_positions(query_embeds_llm)
        # we should first mapping, then we quantize it
        quantized_features, indices, commitment_loss = self.vq_embed(query_tokens)
        '''Add pronunciation reconstruction loss'''
        pre_phone_feature = self.pronunciation_decoder(quantized_features.clone().transpose(1, 2))
        phone_rec_loss = self.L2_loss(pre_phone_feature, wavlm_target)
        quantized_feature_emb = self.forward_decoder(quantized_features)

        quantized_feature_emb = self.cond_feature_emb(quantized_feature_emb)

        B, T, D = quantized_feature_emb.shape
        # latents: B, 2*T, 64:  b, 2*T, 512/2 
        #quantized_feature_emb = F.interpolate(quantized_feature_emb.permute(0, 2, 1), scale_factor=2, mode='nearest').permute(0, 2, 1)
        scenario = np.random.choice(['start_seg', 'other_seg'])
        if(scenario == 'other_seg'):
            for b_idx in range(input_audios.shape[0]):
                # randomly choose some frames to be seen in the training. 
                # following voicebox. It will make use of in-context learning to realize segment-level inference
                latent_masks[b_idx, 0:random.randint(32, 64)] = 1
        
        quantized_feature_emb = (latent_masks > 0.5).unsqueeze(-1) * quantized_feature_emb \
            + (latent_masks < 0.5).unsqueeze(-1) * self.zero_cond_embedding1.unsqueeze(0) # add set the non-mask part as zeros
        bsz, T, dim = latents.shape
        resolution = torch.tensor([T, 1]).repeat(bsz, 1) # ?
        aspect_ratio = torch.tensor([float(T/self.max_t_len)]).repeat(bsz, 1)
        resolution = resolution.to(dtype=features_emb_acoustic.dtype, device=device)
        aspect_ratio = aspect_ratio.to(dtype=features_emb_acoustic.dtype, device=device)
        added_cond_kwargs = {"resolution": resolution, "aspect_ratio": aspect_ratio}
        if self.uncondition:
            mask_indices = [k for k in range(quantized_feature_emb.shape[0]) if random.random() < 0.1] # choose 10% samples as zero condition
            if len(mask_indices) > 0:
                quantized_feature_emb[mask_indices] = 0
        incontext_latents = latents * ((latent_masks > 0.5) * (latent_masks < 1.5)).unsqueeze(-1).float() # the length of latent_mask == 1
        loss = self.cfm_wrapper.compute_loss(latents, [incontext_latents, quantized_feature_emb], added_cond_kwargs, latent_masks, validation_mode=validation_mode)
        return loss, phone_rec_loss, commitment_loss.mean() #codebook_loss.mean()

    def init_device_dtype(self, device, dtype):
        self.device = device
        self.dtype = dtype

    @torch.no_grad()
    def fetch_codes_batch(self, input_audios, spectrograms,  additional_feats,layer):
        input_audio = input_audios[:,0,:]
    
        self.pretrained_model.eval()
        self.wavlm_encoder.eval()
        self.whisper_encoder.eval()
        self.vq_embed.eval()

        bestrq_emb_acoustic, bestrq_emb_semantic = self.pretrained_model.extract_continous_embeds_multiple(input_audios.clone())
        bestrq_emb_acoustic = bestrq_emb_acoustic.detach()
        bestrq_emb_semantic = bestrq_emb_semantic.detach()
        semantic_target = bestrq_emb_semantic.clone()
        whisper_embeds = self.get_whisper_feature(spectrograms, input_audios.shape[-1])
        whisper_embeds = whisper_embeds.detach()
        wavlm_embeds = self.get_wavlm_feature(input_audios) # 
        wavlm_embeds = wavlm_embeds.detach()

        whisper_embeds = self.d_conv_whisper(whisper_embeds) #.transpose(1,2)
        wavlm_encoder_features = self.d_conv_wavlm(wavlm_embeds) 
        features_emb_semantic = bestrq_emb_semantic
        features_emb_acoustic = bestrq_emb_acoustic
        features_emb_phone = self.cond_fusion_layer_phone(wavlm_encoder_features.transpose(1,2)).transpose(1,2)
        features_emb_semantic = self.cond_fusion_layer_semantic(features_emb_semantic.transpose(1, 2)).transpose(1, 2)
        min_f_len = min(features_emb_acoustic.shape[-1], whisper_embeds.shape[-1])
        features_emb_acoustic = torch.cat([features_emb_acoustic[:, :, :min_f_len], whisper_embeds[:, :, :min_f_len]], dim=1)  # concat 
        features_emb_acoustic = self.cond_fusion_layer_acoustic(features_emb_acoustic.transpose(1, 2)).transpose(1, 2)
        #features_emb_semantic = features_emb_semantic[:,:,:min_f_len]
        total_feature = torch.cat([features_emb_phone, features_emb_semantic, features_emb_acoustic], dim=1)
        total_feature = self.feature_proj(total_feature.transpose(1, 2))

        # add query-based strategy
        query_embeds_llm = self.set_masking(total_feature) # add the query token
        query_embeds_llm = self.encoder_transformers(query_embeds_llm)
        query_tokens = self.extract_mask_positions(query_embeds_llm)
        # we should first mapping, then we quantize it
        quantized_features, indices, commitment_loss = self.vq_embed(query_tokens)
        '''Add pronunciation reconstruction loss'''

        return indices, quantized_features

    @torch.no_grad()
    def inference_codes(self, codes, spk_embeds, true_latents, latent_length, incontext_length, additional_feats, 
                  guidance_scale=2, num_steps=20, disable_progress=True, scenario='start_seg'):
        classifier_free_guidance = guidance_scale > 1.0
        device = self.device
        dtype = self.dtype

        codes_semantic = codes[0] # reconstruction tokens
        batch_size = codes_semantic.shape[0]

        self.vq_embed.eval()

        quantized_feature_emb = self.vq_embed.get_output_from_indices(codes_semantic.transpose(1, 2))

        quantized_feature_emb = self.forward_decoder(quantized_feature_emb)
        quantized_feature_emb = self.cond_feature_emb(quantized_feature_emb) # b t 512
        # latents: B, 2*T, 64:  b, 2*T, 512/2 
        B, T, D = quantized_feature_emb.shape
        num_frames = quantized_feature_emb.shape[1] # 

        latents = self.prepare_latents(batch_size, num_frames, dtype, device) # prepapre the latent shape
        bsz, T, dim = latents.shape
        resolution = torch.tensor([T, 1]).repeat(bsz, 1) # ?
        aspect_ratio = torch.tensor([float(T/self.max_t_len)]).repeat(bsz, 1)
        resolution = resolution.to(dtype=quantized_feature_emb.dtype, device=device)
        aspect_ratio = aspect_ratio.to(dtype=quantized_feature_emb.dtype, device=device)
        if classifier_free_guidance:
            resolution = torch.cat([resolution, resolution], 0)
            aspect_ratio = torch.cat([aspect_ratio, aspect_ratio], 0)
        added_cond_kwargs = {"resolution": resolution, "aspect_ratio": aspect_ratio}

        latent_masks = torch.zeros(latents.shape[0], latents.shape[1], dtype=torch.int64, device=latents.device)
        latent_masks[:,0:latent_length] = 2
        if(scenario=='other_seg'):
            latent_masks[:,0:incontext_length] = 1

        quantized_feature_emb = (latent_masks > 0.5).unsqueeze(-1) * quantized_feature_emb \
            + (latent_masks < 0.5).unsqueeze(-1) * self.zero_cond_embedding1.unsqueeze(0)
        
        incontext_latents = true_latents * ((latent_masks > 0.5) * (latent_masks < 1.5)).unsqueeze(-1).float()
        incontext_length = ((latent_masks > 0.5) * (latent_masks < 1.5)).sum(-1)[0]

        if('spk' in additional_feats):
            additional_model_input = torch.cat([quantized_feature_emb, spk_embeds],1)
        else:
            additional_model_input = torch.cat([quantized_feature_emb],1)
        temperature = 1.0
        t_span = torch.linspace(0, 1, num_steps + 1, device=quantized_feature_emb.device)
        latents = self.cfm_wrapper.solve_euler(latents * temperature, incontext_latents, incontext_length, t_span, additional_model_input, added_cond_kwargs, guidance_scale)

        latents[:,0:incontext_length,:] = incontext_latents[:,0:incontext_length,:] # B, T, dim
        return latents

    
    def prepare_latents(self, batch_size, num_frames, dtype, device):
        shape = (batch_size, num_frames, 136)
        latents = randn_tensor(shape, generator=None, device=device, dtype=dtype)
        return latents


