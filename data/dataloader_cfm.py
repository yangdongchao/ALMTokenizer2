
import gzip
import os
import sys
import time
import torch
import copy
import random
import logging
import torchaudio
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader, Dataset
from multiprocessing import Lock, Process
import torch.distributed as dist
from torch.nn.utils.rnn import pad_sequence
from transformers import WhisperFeatureExtractor
from torchaudio import transforms as T
import torch.nn as nn 

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


class CFMDataset(Dataset):
    def __init__(self, args, data_path, whisper_path):
        self.sample_rate = 24000
        self.sample_rate_16k = 16000
        self.music_sr_thresh = 42000  # 认为 >42k 的为音乐
        self.music_duration_threshold = 40
        self.music_read_sec = args.segment_duration
        self.file_list = self.get_filelist(data_path)
        self.wav_processor = WhisperFeatureExtractor.from_pretrained(whisper_path)
        self.volume_norm = VolumeNorm(params=[-16, 3], sample_rate=24000, energy_threshold=1e-6)
        self.args = args

    def get_filelist(self, fpath):
        file_list = []
        with open(fpath, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 4:
                    name, path, dur, sr = parts
                    file_list.append((name, path, float(dur), int(sr)))
        return file_list

    def __getitem__(self, index):
        name, fname, duration, orig_sr = self.file_list[index]
        try:
            if duration >= self.music_read_sec:
                num_frames = int(orig_sr * self.music_read_sec)
                waveform, fs = torchaudio.load(fname, frame_offset=0, num_frames=num_frames)
                duration = self.music_read_sec # 确保后续 duration 统一
            else:
                waveform, fs = torchaudio.load(fname)

            if waveform.shape[0] == 2:
                waveform = waveform.mean(dim=0, keepdim=True)
            
            duration_after_resample = waveform.shape[-1] / fs
            if duration_after_resample < self.music_read_sec: # make sure the segment is 20 seconds
                waveform = torch.cat([waveform, torch.zeros(waveform.shape[0], int(self.music_read_sec*fs) - waveform.shape[1])], dim=-1)
            
            if fs != self.sample_rate:
                waveform = torchaudio.functional.resample(waveform, fs, self.sample_rate)
                fs = self.sample_rate
                waveform_norm = self.volume_norm(waveform)
                waveform_16k = torchaudio.functional.resample(waveform_norm, fs, self.sample_rate_16k)
            else:
                waveform_norm = self.volume_norm(waveform)
                waveform_16k = torchaudio.functional.resample(waveform_norm, fs, self.sample_rate_16k)
            
            spectrogram = self.wav_processor(
                waveform_16k.squeeze(0),
                sampling_rate=self.sample_rate_16k,
                return_tensors="pt"
            )["input_features"].squeeze()

            return waveform_norm, spectrogram, duration_after_resample

        except Exception as e:
            print(f"[WARNING] Failed to load waveform, skipping! filename: {fname} Error: {e}")
            return self[random.randrange(len(self))]

    def __len__(self):
        return len(self.file_list)


def collate_fn(batch_data):
    audios = pad_sequence([data[0].t() for data in batch_data], batch_first=True, padding_value=0).transpose(1,2)
    spectrograms = pad_sequence([data[1].t() for data in batch_data], batch_first=True, padding_value=0).transpose(1,2)
    durations = [data[2] for data in batch_data]
    return audios, spectrograms, durations


def get_data_iterator_tokenizer_vocabulary(
        args,
        rank,
        world_size
    ):

    logging.basicConfig(
        stream=sys.stdout,
        level=logging.DEBUG,
        format=f"%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s",
    )

    train_set = CFMDataset(args, args.train_data_path, args.whisper_path)
    test_set = CFMDataset(args, args.val_data_path, args.whisper_path)

    train_sampler = DistributedSampler(train_set, num_replicas=world_size, rank=rank, shuffle=True)
    test_sampler = DistributedSampler(test_set, num_replicas=world_size, rank=rank, shuffle=True)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, sampler=train_sampler, num_workers=8, collate_fn=collate_fn)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, sampler=test_sampler, num_workers=8, collate_fn=collate_fn)
    return train_loader, test_loader


if __name__ == "__main__":
    # get_data_iterator_tokenizer_vocabulary(sys.argv[1:2], sys.argv[2:3], n_worker=1)
    train_set = CFMDataset('/home/ydc/code3/AnyToken2Audio/data/val.scp')
    train_loader = DataLoader(train_set, batch_size=16, num_workers=8,
                              collate_fn=collate_fn)
    for idx, batch in enumerate(train_loader):
        audios, durations = batch
        print('audios ', audios.shape, durations)
        # print(f'step: {idx}, chunk_id: {train_set.chunk_id}')
        time.sleep(1)
    