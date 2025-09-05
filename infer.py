# Dongchao Yang

"""Command-line for audio compression."""
''' the first version of music semantic tokenizer
'''
import argparse
from pathlib import Path
import sys
import torchaudio
import os
import torch
import typing as tp
import torch.distributed as dist
from collections import OrderedDict
from omegaconf import OmegaConf
import logging
import json
import torch
from tqdm import tqdm
from models.AudioDiffusion1D import AudioDiffusion1D
import librosa
import os
import math
import numpy as np
from utils import torch_tools
import torch.nn as nn
import torch.nn.functional as F
from models.scalar24k import ScalarAE
from transformers import WhisperFeatureExtractor
import yaml
from torchaudio import transforms as T

class VolumeNorm(nn.Module):
    "Volume normalization and augmentation of a signal [LUFS standard]"
    def __init__(self, params=[-16, 3], sample_rate=24000, energy_threshold=1e-6):
        super().__init__()
        self.loudness = T.Loudness(sample_rate)
        self.value = params[0]
        self.gain_range = [-params[1], params[1]]
        self.energy_threshold = energy_threshold

    def __call__(self, signal):
        """
        signal: torch.Tensor [channels, time]
        """
        # avoid do normalisation for silence
        energy = torch.mean(signal**2)
        if energy < self.energy_threshold:
            return signal
        
        input_loudness = self.loudness(signal)
        # Generate a random target loudness within the specified range
        target_loudness = self.value + (torch.rand(1).item() * (self.gain_range[1] - self.gain_range[0]) + self.gain_range[0])
        delta_loudness = target_loudness - input_loudness
        gain = torch.pow(10.0, delta_loudness / 20.0)
        output = gain * signal

        # Check for potentially clipped samples
        if torch.max(torch.abs(output)) >= 1.0:
            output = self.declip(output)

        return output

    def declip(self, signal):
        """
        Declip the signal by scaling down if any samples are clipped
        """
        max_val = torch.max(torch.abs(signal))
        if max_val > 1.0:
            signal = signal / max_val
            signal *= 0.95
        return signal


class ALMTokenizer2(nn.Module):
    def __init__(self, model_path='', train_config=None, layer_num=6, load_main_model=True, device="cpu"):
        super(ALMTokenizer2, self).__init__()
        self.layer_num = layer_num 
        self.sample_rate = 24000
        self.device = device
        self.MAX_DURATION = 360
        self.dim_codebook = 10000 # 
        self.sq_codec_hz = 25
        self.n_codebook = 4
        self.bw = 2 # bw=2---> 4 codebooks
        self.freq = int(self.n_codebook * 12.5)
        self.SQCodec = ScalarAE(scalar_config=train_config.sq_config, resume_path=train_config.sq_resume)
        self.SQCodec = self.SQCodec.eval().to(device)
        self.wav_processor = WhisperFeatureExtractor.from_pretrained(train_config.whisper_path)
        self.transfer16k = torchaudio.transforms.Resample(24000, 16000).to(device)

        self.model = AudioDiffusion1D(
            num_channels= train_config.num_channels,
            statistical_prior = train_config.statistical_prior_path,
            pre_trained_model_name = 'whisper&bestrq',
            features_type = 'continuous',
            vq_training = True,
            unet_model_name = train_config.unet_model_name,
            unet_model_config_path = train_config.transformer_diffusion_config,
            whisper_path = train_config.whisper_path,
            snr_gamma = None,
            uncondition = True,
            out_paint = False,
        )
        #main_weights = torch.load(model_path, map_location='cpu')
        state_dict = torch.load(model_path, map_location='cpu')['model']
        state_dict = { k.split("module.")[-1] if k.startswith("module.") else k: v
                    for k, v in state_dict.items() }
        self.model.load_state_dict(state_dict)
        self.model = self.model.to(device)
        print ("Successfully loaded checkpoint from:", model_path)
        
        self.model.eval()
        self.model.init_device_dtype(torch.device(device), torch.float32)
        self.volume_norm = VolumeNorm(params=[-16, 3], sample_rate=24000, energy_threshold=1e-6)

    def get_whisper_features(self, audio, sr):
        if sr != 16000:
            audio = self.transfer16k(audio)
            sr = 16000
        spectrogram = self.wav_processor(audio.detach().cpu().numpy(), sampling_rate=16000, return_tensors="pt")["input_features"] # B, 80, t
        return spectrogram

    def file2code(self, fname):
        orig_samples, fs = torchaudio.load(fname)
        if(fs!=self.sample_rate):
            orig_samples = torchaudio.functional.resample(orig_samples, fs, self.sample_rate)
            fs = self.sample_rate
        if orig_samples.shape[0] == 1:
            orig_samples = torch.cat([orig_samples, orig_samples], 0)
        return self.sound2code(orig_samples)

    @torch.no_grad()
    @torch.autocast(device_type="cuda", dtype=torch.float32)
    def sound2code(self, orig_samples, sr, min_duration=20, batch_size=6):
        if(orig_samples.ndim == 2):
            audios = orig_samples.unsqueeze(0).to(self.device)
        elif(orig_samples.ndim == 3):
            audios = orig_samples.to(self.device)
        else:
            assert orig_samples.ndim in (2,3), orig_samples.shape
        
        audios = audios.squeeze(0) # change to 2-dim. (channel, time)
        print('audios org', audios.shape)
        orig_length = audios.shape[-1]
        min_samples = int(min_duration * self.sample_rate) # the min_segment
        print('min_samples ', min_samples)
        # 40.96秒对应1024个token
        output_len = int(orig_length / float(self.sample_rate) * 12.5) + 1
        
        while(audios.shape[-1] < min_samples + 240): # add 240, to make sure the last frame is completed
            audios = torch.cat([audios, audios], -1)
        print('audios org2 ', audios.shape)
        int_max_len = audios.shape[-1]//min_samples+1 # the max segment number
        # print("int_max_len: ", int_max_len)
        audios = torch.cat([audios, audios], -1)
        print('audios org3 ', audios.shape)
        audios=audios[:,:int(int_max_len*(min_samples+240))] 
        print('audios org4 ', audios.shape)
        semantic_codes_list = []
        audio_input = audios.reshape(1, -1, min_samples+240).permute(1, 0, 2).reshape(-1, 1, min_samples+240)
        print('audio_input ', audio_input.shape) # B,1 segment
        for audio_inx in range(0, audio_input.shape[0], batch_size):
            mels = self.get_whisper_features(audio_input[audio_inx:audio_inx+batch_size,0,:], 24000).to(self.device)
            print('mels ', mels.shape, mels.dtype)
            codes_semantic, quantized_feature_emb = self.model.fetch_codes_batch((audio_input[audio_inx:audio_inx+batch_size]), mels, additional_feats=[],layer=self.layer_num)
            print('codes_semantic ', codes_semantic.shape)
            semantic_codes_list.append(codes_semantic)
            
        # assert 1==2
        semantic_codec = torch.cat(semantic_codes_list, 0)
        semantic_codec = semantic_codec.reshape(-1, 8).unsqueeze(0)
        semantic_codec = semantic_codec[:,:output_len,:].transpose(1, 2)
        
        return semantic_codec 


    @torch.no_grad()
    def code2sound(self, codes, prompt=None, duration=20, guidance_scale=1.5, num_steps=20, disable_progress=False):
        ''' codes: (B, 4, T)
        '''
        semantic_codec = codes[0]
        first_latent = torch.randn(semantic_codec.shape[0], int(duration*self.sq_codec_hz), 136).to(self.device) # B, T, 64
        first_latent_length = 0
        first_latent_codes_length = 0

        min_samples = int(duration*12.5) # 20s ---> 500 frames
        hop_samples = min_samples // 4 * 3
        ovlp_samples = min_samples - hop_samples
        hop_frames = hop_samples // 2
        ovlp_frames = ovlp_samples // 2
        
        codes_len= semantic_codec.shape[-1] #
        print('codes_len ', codes_len)
        target_len = int((codes_len - first_latent_codes_length) / 12.5 * self.sample_rate)
        print('target_len ', target_len)
        # code repeat
        if(codes_len < min_samples):
            while(semantic_codec.shape[-1] < min_samples):
                semantic_codec = torch.cat([semantic_codec, semantic_codec], -1)
            semantic_codec = semantic_codec[:,:,0:min_samples]
        
        codes_len = semantic_codec.shape[-1]
        if((codes_len - ovlp_frames) % hop_samples > 0):
            len_codes=math.ceil((codes_len - ovlp_samples) / float(hop_samples)) * hop_samples + ovlp_samples
            while(semantic_codec.shape[-1] < len_codes):
                semantic_codec = torch.cat([semantic_codec, semantic_codec], -1)
                
            semantic_codec = semantic_codec[:,:,0:len_codes]
        
        latent_length = int(duration*self.sq_codec_hz)
        latent_list = []
        spk_embeds = None
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            for sinx in range(0, semantic_codec.shape[-1]-hop_samples, hop_samples):
                codes_input=[]
                codes_input.append(semantic_codec[:,:,sinx:sinx+min_samples])
                if(sinx == 0):
                    # print("Processing {} to {}".format(sinx/self.sample_rate, (sinx + min_samples)/self.sample_rate))
                    incontext_length = first_latent_length
                    latents = self.model.inference_codes(codes_input, spk_embeds, first_latent, latent_length, incontext_length, additional_feats=[], guidance_scale=1.5, num_steps = num_steps, disable_progress=disable_progress, scenario='other_seg')
                    latent_list.append(latents)
                else:
                    # print("Processing {} to {}".format(sinx/self.sample_rate, (sinx + min_samples)/self.sample_rate))
                    true_latent = latent_list[-1][:,-ovlp_frames:,:]
                    len_add_to_latent = latent_length - true_latent.shape[1] # 
                    incontext_length = true_latent.shape[1]
                    true_latent = torch.cat([true_latent, torch.randn(true_latent.shape[0], len_add_to_latent, true_latent.shape[-1]).to(self.device)], 1)
                    latents = self.model.inference_codes(codes_input, spk_embeds, true_latent, latent_length, incontext_length,  additional_feats=[], guidance_scale=1.5, num_steps = num_steps, disable_progress=disable_progress, scenario='other_seg')
                    latent_list.append(latents)

        latent_list = [l.float() for l in latent_list]
        latent_list[0] = latent_list[0][:,first_latent_length:,:]
        min_samples =  int(duration * self.sample_rate)
        hop_samples = min_samples // 4 * 3
        ovlp_samples = min_samples - hop_samples
        with torch.no_grad():
            output = None
            for i in range(len(latent_list)):
                latent = latent_list[i]
                bsz , t, f = latent.shape
                # latent = latent.reshape(bsz, ch//2, t, f)
                cur_output = self.SQCodec.decode(latent.transpose(1,2)).squeeze(0)
                cur_output = cur_output[:, 0:min_samples].detach().cpu() # B, T
                if output is None:
                    output = cur_output
                else:
                    ov_win = torch.from_numpy(np.linspace(0, 1, ovlp_samples)[None, :])
                    ov_win = torch.cat([ov_win, 1 - ov_win], -1)
                    output[:, -ovlp_samples:] = output[:, -ovlp_samples:] * ov_win[:, -ovlp_samples:] + cur_output[:, 0:ovlp_samples] * ov_win[:, 0:ovlp_samples]
                    output = torch.cat([output, cur_output[:, ovlp_samples:]], -1)
            output = output[:, 0:target_len]
        return output

    
    def encode_segment(self, orig_samples, sr):
        if(orig_samples.ndim == 2):
            audios = orig_samples.unsqueeze(0).to(self.device)
        elif(orig_samples.ndim == 3):
            audios = orig_samples.to(self.device)
        else:
            assert orig_samples.ndim in (2,3), orig_samples.shape
        #audios = audios.squeeze(0) # change to 2-dim. (channel, time)
        orig_length = audios.shape[-1]
        output_len = int(orig_length / float(self.sample_rate) * 25) + 1
        mels = self.get_whisper_features(audios.squeeze(0), 24000).to(self.device)
        print('mels ', mels.shape, mels.dtype)
        codes, _ = self.model.fetch_codes_batch(audios, mels, additional_feats=[],layer=self.layer_num)
        # print('codes ', codes[0].shape)
        # assert 1==2
        return codes[0].transpose(1, 2)

    def decode_segment(self, codes, prompt=None, duration=20, guidance_scale=1.5, num_steps=50, disable_progress=False):
        codes = codes.to(self.device)
        first_latent = torch.randn(codes.shape[0], 1000, 64).to(self.device) # B, T, 64
        first_latent_length = 0
        first_latent_codes_length = 0
        spk_embeds = None
        latent_length = 1000
        incontext_length = 0
        latents = self.model.inference_codes([codes], spk_embeds, first_latent, latent_length, incontext_length, additional_feats=[], guidance_scale=1.5, num_steps = num_steps, disable_progress=disable_progress, scenario='other_seg')
        # print('latents ', latents.shape)
        # assert 1==2
        cur_output = self.SQCodec.decode(latents.transpose(1,2)).squeeze(0)
        cur_output = cur_output.detach().cpu() # B, T
        return cur_output

    @property
    def is_discrete(self):
        return True

    def _flatten_codebooks(self, arr, offset_size=1000):
        assert len(arr.shape) == 2
        arr = arr.copy()
        if offset_size is not None:
            for n in range(arr.shape[0]):
                arr[n, :] += (offset_size * n)
        flat_arr = arr.ravel("F")
        return flat_arr


    def tokenize(self, wav, min_duration=20):
        ''' tokenize 时，只支持wav_path
        '''
        if isinstance(wav, str):
            prompt_audio, fs = torchaudio.load(wav) # read the file
            print('prompt_audio, fs ', wav, prompt_audio.shape, fs)
            if prompt_audio.shape[0] == 2:
                prompt_audio = prompt_audio.mean(0, keepdim=True)
            if(fs!=self.sample_rate):
                prompt_audio = torchaudio.functional.resample(prompt_audio, fs, self.sample_rate)
                fs = self.sample_rate
            semantic_codec = self.sound2code(prompt_audio, fs, min_duration=min_duration) # [1, 4, T]
            return semantic_codec
        elif isinstance(wav, torch.Tensor):
            return wav
        else:
            raise NotImplementedError
    
    def tokenize2(self, data):
        # we will use it in the training process
        token = data.cpu().numpy()
        token = self._flatten_codebooks(token, offset_size=self.dim_codebook)
        token = torch.tensor(token).to(torch.int64)
        return token

    @property
    def codebook_length(self):
        return self.dim_codebook * self.n_codebook 

    def find_length(self, x):
        return self.tokenize2(x).shape[0] // self.n_codebook

    def detokenize(self, codes, prompt=None, min_duration=20.48, steps=50, disable_progress=False):
        ''' 1, 4, T
        '''
        wave = self.code2sound(codes, prompt, duration=min_duration, guidance_scale=1.5, num_steps=steps, disable_progress=disable_progress)
        return wave 


def get_parser():
    parser = argparse.ArgumentParser()

    # model related: use the resume model if provided; otherwise use the latest in exp_dir
    parser.add_argument('--resume', type=str, default=None, help='model to resume. If None, use the latest checkpoint in exp_dir')
    parser.add_argument('--exp_dir', type=str, default=None, help='experiment directory')

    # inference related: 
    parser.add_argument('--seed', type=int, default=888, help='random seed')

    # device related
    parser.add_argument('--rank', type=int, default=-1, help='GPU rank. -1 means CPU')
    # data related
    parser.add_argument('--output_dir', type=str, help="tag for decoding")
    return parser 

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    train_config = args.exp_dir + '/config.yaml'
    layer_num = 6
    with open(train_config, "r", encoding="utf-8") as f:
        train_args = yaml.safe_load(f)
        train_args = argparse.Namespace(**train_args)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    if args.rank >= 0:
        args.rank = args.rank % torch.cuda.device_count()
        device = torch.device(f'cuda:{args.rank}') # run.pl index from 1
    else:
        device = torch.device('cpu')
    
    tokenizer = ALMTokenizer2(model_path=args.resume, layer_num=layer_num, train_config=train_args, device=device)

    wav_path = './music_test'
    save_path = './exp/alm_v1/sound_epoch5_music_10s_v2'
    import os 
    import glob
    names = glob.glob(f"{wav_path}/*.wav")
    for name in names:
        print('name ', name)
        # assert 1==2
        os.makedirs(save_path, exist_ok=True)
        semantic_codec = tokenizer.tokenize(name, min_duration=12)
        # print('semantic_codec ', semantic_codec)
        # assert 1==2
        bs_name = os.path.basename(name)
        # codec = torch.tensor(codec)
        #assert 1==2
        wav = tokenizer.detokenize([semantic_codec], min_duration=12)
        print(wav.shape)
        torchaudio.save(f'{save_path}/{bs_name}', wav.detach().cpu(), sample_rate=24000)
