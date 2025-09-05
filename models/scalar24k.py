import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchaudio
from torch.nn.utils import weight_norm, remove_weight_norm
from torch.autograd.function import InplaceFunction
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm, remove_weight_norm
import pytorch_lightning as pl
from omegaconf import OmegaConf
from collections import OrderedDict

def get_padding(kernel_size, dilation=1):
    return int((kernel_size*dilation - dilation)/2)

# Scripting this brings model speed up 1.4x
@torch.jit.script
def snake(x, alpha):
    shape = x.shape
    x = x.reshape(shape[0], shape[1], -1)
    x = x + (alpha + 1e-9).reciprocal() * torch.sin(alpha * x).pow(2)
    x = x.reshape(shape)
    return x

class Snake1d(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, channels, 1))
    def forward(self, x):
        return snake(x, self.alpha)

class Conv1d(nn.Conv1d):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int = 1,
                 dilation: int = 1,
                 groups: int = 1,
                 padding_mode: str = 'zeros',
                 bias: bool = True,
                 padding = None,
                 causal: bool = False,
                 w_init_gain = None):
        self.causal = causal
        if padding is None:
            if causal:
                padding = 0
                self.left_padding = dilation * (kernel_size - 1)
            else:
                padding = get_padding(kernel_size, dilation)
        super(Conv1d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            padding_mode=padding_mode,
            bias=bias)
        if w_init_gain is not None:
            torch.nn.init.xavier_uniform_(
                self.weight, gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        if self.causal:
            x = F.pad(x.unsqueeze(2), (self.left_padding, 0, 0, 0)).squeeze(2)

        return super(Conv1d, self).forward(x)

class ConvTranspose1d(nn.ConvTranspose1d):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int = 1,
                 output_padding: int = 0,
                 groups: int = 1,
                 bias: bool = True,
                 dilation: int = 1,
                 padding=None,
                 padding_mode: str = 'zeros',
                 causal: bool = False):
        if padding is None:
            padding = 0 if causal else (kernel_size - stride) // 2
        if causal:
            assert padding == 0, "padding is not allowed in causal ConvTranspose1d."
            assert kernel_size == 2 * stride, "kernel_size must be equal to 2*stride is not allowed in causal ConvTranspose1d."
        super(ConvTranspose1d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            groups=groups,
            bias=bias,
            dilation=dilation,
            padding_mode=padding_mode)
        self.causal = causal
        self.stride = stride

    def forward(self, x):
        x = super(ConvTranspose1d, self).forward(x)
        if self.causal:
            x = x[:, :, :-self.stride]
        return x


class PreProcessor(nn.Module):
    def __init__(self, n_in, n_out, num_samples, kernel_size=7, causal=False):
        super(PreProcessor, self).__init__()
        self.pooling = torch.nn.AvgPool1d(kernel_size=num_samples)
        self.conv = Conv1d(n_in, n_out, kernel_size=kernel_size, causal=causal)
        self.activation = nn.PReLU()

    def forward(self, x):
        output = self.activation(self.conv(x))
        output = self.pooling(output)
        return output


class PostProcessor(nn.Module):
    def __init__(self, n_in, n_out, num_samples, kernel_size=7, causal=False):
        super(PostProcessor, self).__init__()
        self.num_samples = num_samples
        self.conv = Conv1d(n_in, n_out, kernel_size=kernel_size, causal=causal)
        self.activation = nn.PReLU()

    def forward(self, x):
        x = torch.transpose(x, 1, 2)
        B, T, C = x.size()
        x = x.repeat(1, 1, self.num_samples).view(B, -1, C)
        x = torch.transpose(x, 1, 2)
        output = self.activation(self.conv(x))
        return output

class ResidualUnit(nn.Module):
    def __init__(self, n_in, n_out, dilation, res_kernel_size=7, causal=False):
        super(ResidualUnit, self).__init__()
        self.conv1 = weight_norm(Conv1d(n_in, n_out, kernel_size=res_kernel_size, dilation=dilation, causal=causal))
        self.conv2 = weight_norm(Conv1d(n_in, n_out, kernel_size=1, causal=causal))
        self.activation1 = nn.PReLU()
        self.activation2 = nn.PReLU()

    def forward(self, x):
        output = self.activation1(self.conv1(x))
        output = self.activation2(self.conv2(output))
        return output + x


class ResEncoderBlock(nn.Module):
    def __init__(self, n_in, n_out, stride, down_kernel_size, res_kernel_size=7, causal=False):
        super(ResEncoderBlock, self).__init__()
        self.convs = nn.ModuleList([
            ResidualUnit(n_in, n_out // 2, dilation=1, res_kernel_size=res_kernel_size, causal=causal),
            ResidualUnit(n_out // 2, n_out // 2, dilation=3, res_kernel_size=res_kernel_size, causal=causal),
            ResidualUnit(n_out // 2, n_out // 2, dilation=5, res_kernel_size=res_kernel_size, causal=causal),
            ResidualUnit(n_out // 2, n_out // 2, dilation=7, res_kernel_size=res_kernel_size, causal=causal),
            ResidualUnit(n_out // 2, n_out // 2, dilation=9, res_kernel_size=res_kernel_size, causal=causal),
        ])

        self.down_conv = DownsampleLayer(
            n_in, n_out, down_kernel_size, stride=stride, causal=causal)


    def forward(self, x):
        for conv in self.convs:
            x = conv(x)
        x = self.down_conv(x)
        return x


class ResDecoderBlock(nn.Module):
    def __init__(self, n_in, n_out, stride, up_kernel_size, res_kernel_size=7, causal=False):
        super(ResDecoderBlock, self).__init__()
        self.up_conv = UpsampleLayer(
            n_in, n_out, kernel_size=up_kernel_size, stride=stride, causal=causal, activation=None)

        self.convs = nn.ModuleList([
            ResidualUnit(n_out, n_out, dilation=1, res_kernel_size=res_kernel_size, causal=causal),
            ResidualUnit(n_out, n_out, dilation=3, res_kernel_size=res_kernel_size, causal=causal),
            ResidualUnit(n_out, n_out, dilation=5, res_kernel_size=res_kernel_size, causal=causal),
            ResidualUnit(n_out, n_out, dilation=7, res_kernel_size=res_kernel_size, causal=causal),
            ResidualUnit(n_out, n_out, dilation=9, res_kernel_size=res_kernel_size, causal=causal),
        ])

    def forward(self, x):
        x = self.up_conv(x)
        for conv in self.convs:
            x = conv(x)
        return x

class DownsampleLayer(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int = 1,
                 causal: bool = False,
                 activation=nn.PReLU(),
                 use_weight_norm: bool = True,
                 pooling: bool = False):
        super(DownsampleLayer, self).__init__()
        self.pooling = pooling
        self.stride = stride
        self.activation = activation
        self.use_weight_norm = use_weight_norm
        if pooling:
            self.layer = Conv1d(
                in_channels, out_channels, kernel_size, causal=causal)
            self.pooling = nn.AvgPool1d(kernel_size=stride)
        else:
            self.layer = Conv1d(
                in_channels, out_channels, kernel_size, stride=stride, causal=causal)
        if use_weight_norm:
            self.layer = weight_norm(self.layer)

    def forward(self, x):
        x = self.layer(x)
        x = self.activation(x) if self.activation is not None else x
        if self.pooling:
            x = self.pooling(x)
        return x

    def remove_weight_norm(self):
        if self.use_weight_norm:
            remove_weight_norm(self.layer)


class UpsampleLayer(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int = 1,
                 causal: bool = False,
                 activation=nn.PReLU(),
                 use_weight_norm: bool = True,
                 repeat: bool = False):
        super(UpsampleLayer, self).__init__()
        self.repeat = repeat
        self.stride = stride
        self.activation = activation
        self.use_weight_norm = use_weight_norm
        if repeat:
            self.layer = Conv1d(
                in_channels, out_channels, kernel_size, causal=causal)
        else:
            self.layer = ConvTranspose1d(
                in_channels, out_channels, kernel_size, stride=stride, causal=causal)
        if use_weight_norm:
            self.layer = weight_norm(self.layer)

    def forward(self, x):
        x = self.layer(x)
        x = self.activation(x) if self.activation is not None else x
        if self.repeat:
            x = torch.transpose(x, 1, 2)
            B, T, C = x.size()
            x = x.repeat(1, 1, self.stride).view(B, -1, C)
            x = torch.transpose(x, 1, 2)
        return x

    def remove_weight_norm(self):
        if self.use_weight_norm:
            remove_weight_norm(self.layer)

class round_func5(InplaceFunction):
    @staticmethod
    def forward(ctx, input):
        ctx.input = input
        return torch.round(5*input)/5
    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input


class round_func9(InplaceFunction):
    @staticmethod
    def forward(ctx, input):
        ctx.input = input
        return torch.round(9*input)/9
    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input

class round_func_binary(InplaceFunction):
    @staticmethod
    def forward(ctx, input):
        ctx.input = input
        return torch.round(input)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input

class ScalarModel(nn.Module):
    def __init__(self, num_bands, sample_rate, causal, num_samples, downsample_factors, downsample_kernel_sizes,
                       upsample_factors, upsample_kernel_sizes, latent_hidden_dim, default_kernel_size,
                       delay_kernel_size, init_channel, res_kernel_size):
        super(ScalarModel, self).__init__()
        # self.args = args
        self.encoder = []
        self.decoder = []
        self.vq = round_func9() # using 9
        # Encoder parts
        self.encoder.append(
            weight_norm(
                Conv1d(
                    num_bands,
                    init_channel,
                    kernel_size=default_kernel_size,
                    causal=causal
                )
            )
        )
        if num_samples > 1:
            # Downsampling
            self.encoder.append(
                PreProcessor(init_channel,
                             init_channel,
                             num_samples,
                             kernel_size=default_kernel_size,
                             causal=causal))
        for i, down_factor in enumerate(downsample_factors):
            self.encoder.append(
                ResEncoderBlock(init_channel * np.power(2, i),
                                init_channel * np.power(2, i+1),
                                down_factor,
                                downsample_kernel_sizes[i],
                                res_kernel_size,
                                causal=causal))
        self.encoder.append(
            weight_norm(Conv1d(
                init_channel * np.power(2, len(downsample_factors)),
                latent_hidden_dim,
                kernel_size=default_kernel_size,
                causal=causal)))
        # Decoder
        # look ahead 
        self.decoder.append(
            weight_norm(Conv1d(
                latent_hidden_dim,
                init_channel * np.power(2, len(upsample_factors)),
                kernel_size=delay_kernel_size)))
        for i, upsample_factor in enumerate(upsample_factors):
            self.decoder.append(
                ResDecoderBlock(init_channel * np.power(2, len(upsample_factors) - i),
                             init_channel * np.power(2, len(upsample_factors) - i - 1),
                             upsample_factor,
                             upsample_kernel_sizes[i],
                             res_kernel_size,
                             causal=causal))
        if num_samples > 1:
            self.decoder.append(
                PostProcessor(
                    init_channel,
                    init_channel,
                    num_samples,
                    kernel_size=default_kernel_size,
                    causal=causal))
        self.decoder.append(
            weight_norm(Conv1d(
                init_channel,
                num_bands,
                kernel_size=default_kernel_size,
                causal=causal)))
        self.encoder = nn.ModuleList(self.encoder)
        self.decoder = nn.ModuleList(self.decoder)

    def forward(self, x):
        for i, layer in enumerate(self.encoder):
            if i != len(self.encoder) - 1:
                x = layer(x)
            else:
                x = F.tanh(layer(x))
        # import pdb; pdb.set_trace()
        x = self.vq.apply(x) # vq
        for i, layer in enumerate(self.decoder):
            x = layer(x)
        return x
    
    def encode(self, x):
        for i, layer in enumerate(self.encoder):
            if i != len(self.encoder) - 1:
                x = layer(x)
            else:
                x = F.tanh(layer(x)) # reverse to tanh
        emb = x 
        # import pdb; pdb.set_trace()
        emb_quant = self.vq.apply(emb) # vq
        return emb
    
    def decode(self, x):
        x = self.vq.apply(x) # make sure the prediction follow the similar disctribution
        for i, layer in enumerate(self.decoder):
            x = layer(x)
        return x

    def inference(self, x):
        for i, layer in enumerate(self.encoder):
            if i != len(self.encoder) - 1:
                x = layer(x)
            else:
                x = F.tanh(layer(x)) # reverse to tanh
        emb = x 
        # import pdb; pdb.set_trace()
        emb_quant = self.vq.apply(emb) # vq
        x = emb_quant
        for i, layer in enumerate(self.decoder):
            x = layer(x)
        return emb, emb_quant, x


class ScalarAE(pl.LightningModule):
    def __init__(self, scalar_config, resume_path=None):
        super().__init__()
        self.scalar_config = scalar_config
        self.resume_path = resume_path
        exp_model_config = OmegaConf.load(self.scalar_config)
        self.model = ScalarModel(**exp_model_config.generator.config)
        if resume_path is not None:
            self.resume_model()

    def resume_model(self):
        parameter_dict = torch.load(self.resume_path)
        new_state_dict = OrderedDict()
        self.model.load_state_dict(parameter_dict['codec_model']) # load model
    
    def forward(self, x):
        codes = self.encode(x)
        wav = self.decode(codes)
        return wav

    def encode(self, x):
        return self.model.encode(x)
    
    def decode(self, x):
        return self.model.decode(x)

# sq_codec_config="/turing_music_fs/music_data/ydc/checkpoints/sq_dim136/config.yaml"
# sq_codec_ckpt="/turing_music_fs/music_data/ydc/checkpoints/sq_dim136/ckpt_00615000.pth"
# # from torchprofile import profile_macs
# SQCodec = ScalarAE(scalar_config=sq_codec_config, resume_path=sq_codec_ckpt)
# import torchaudio

# audio, sr = torchaudio.load('/root/code2/0_refer.wav')
# audio = torchaudio.functional.resample(audio, sr, 24000)
# if audio.shape[0] == 2:
#     audio = audio.mean(dim=0, keepdim=True)
# audio = audio.unsqueeze(0)
# print('audio ', audio.shape)
# c = SQCodec.encode(audio)
# rec = SQCodec.decode(c)

# rec = rec.detach().cpu().squeeze(0)

# torchaudio.save('sq_128.wav', rec, sample_rate=24000)
