import numpy as np
from utils import *
from transform_data import *
from nnAudio.Spectrogram import MFCC, MelSpectrogram
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import librosa
from librosa import power_to_db
from librosa.util import fix_length, pad_center


class CovidDataset(Dataset):
    def __init__(self, df, config, meta_col, sub='train', has_transform=True):
        super(CovidDataset, self).__init__()

        assert sub in ['train', 'validation', 'test']

        self.df = df
        self.config = config
        self.meta_col = meta_col
        self.sub = sub
        self.has_transform = has_transform
        if self.has_transform:
            self.transform = aug(self.sub)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        path = self.df.loc[idx, 'file_path']
        y, sr = librosa.load(path, sr=self.config.sr)
        start, end = self.df.loc[idx, ['start', 'end']]

        if end > start:
            y_clean = y[start:end]
        else:
            y_clean = y
        if np.max(y_clean) > 0:
            y_clean /= np.max(y_clean)

        if self.has_transform:
            new_y = self.transform(data=y_clean)['data']
        else:
            new_y = y_clean
        new_y = torch.from_numpy(new_y).float()

        mf = MFCC(**self.config.mfcc_config)(new_y)
        mf = fix_length(mf.detach().cpu().numpy(), self.config.max_length)
        mf = torch.tensor(mf).float()

        S = MelSpectrogram(**self.config.melspectrogram_config)(new_y)
        mel = power_to_db(S, ref=np.max)
        mel = mono_to_color(mel) / 255
        mel = fix_length(mel, self.config.max_length)
        mel = torch.tensor(mel).float()

        if self.sub != 'test':
            label = self.df.loc[idx, 'assessment_result']
            meta_data = torch.tensor(self.df.loc[idx, self.meta_col].astype(np.float32))
            label = torch.tensor([1 - label, label]).to(torch.float32)
            return mf, mel, meta_data, label
        return mf, mel, None

    def collate_fn(self, batch):
        mf = torch.stack([i[0] for i in batch])
        mel = torch.stack([i[1] for i in batch])
        if self.sub != 'test':
            meta_data = torch.stack([i[2] for i in batch])
            label = torch.stack([i[3] for i in batch])
            return mf, mel, meta_data, label
        return mf, mel, None
