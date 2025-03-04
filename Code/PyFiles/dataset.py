import os
import sys

import torch
from tqdm import tqdm
import cv2
import numpy as np

from lib.utils import generate_string_hash
from lib.datasets import code15, ptb_xl, sph

class CLIP_ECG_Dataset(torch.utils.data.Dataset):
    def __init__(self, ecg_files, captions, config):
        self.config = config
        self.ecg_files = ecg_files
        self.captions = captions
        self.ecgs = load_and_preprocess_list(self.ecg_files, self.config, config.cache_path)
        
    def __len__(self, ):
        return len(self.ecg_files)

    def __getitem__(self, idx):
        caption = self.captions[idx]
        image = self.ecgs[idx]
        return {'image': image, 'caption': caption}
        
    def process_ecg(self, ecg, sr):
        new_shape = int(self.config.ecg_sr * ecg.shape[1] / sr)
        ecg = resample(ecg, new_shape)
        return ecg

class CLIP_ECG_Pretrain_Dataset(torch.utils.data.Dataset):
    def __init__(self, ecg_files, captions, config):
        self.config = config
        self.ecg_files = ecg_files
        self.captions = captions

        targets = list()
        for i, name in enumerate(self.config.train_classes):
            class_mask = np.array([name in caption for caption in self.captions]).astype('uint8')
            targets.append(class_mask)
        self.targets = np.stack(targets)
        self.ecgs = load_and_preprocess_list(self.ecg_files, self.config, config.cache_path)
        
    def __len__(self, ):
        return len(self.ecg_files)

    def __getitem__(self, idx):
        caption = self.captions[idx]
        image = self.ecgs[idx]
        targets = self.targets[:, idx].astype('float32')
        return {'image': image, 'caption': caption, 'targets': targets}
        
    def process_ecg(self, ecg, sr):
        new_shape = int(self.config.ecg_sr * ecg.shape[1] / sr)
        ecg = resample(ecg, new_shape)
        return ecg
    
def load_and_preprocess(ecg_file, new_sr, cache_path,  config):
    ecg_file_adj = f'{ecg_file}_{new_sr}'
    ecg_fie_hash = generate_string_hash(ecg_file_adj)
    cache_name = f'{cache_path}/{ecg_fie_hash}.npy'
    if not os.path.isfile(cache_name):
        
        if ecg_file.endswith('.hea'):
            ecg, leads, old_sr = ptb_xl.load_ecg(ecg_file) 
        elif ecg_file.endswith('.h5'):
            ecg, leads, old_sr = sph.load_ecg(ecg_file) 
        elif ecg_file.endswith('.hdf5'):
            ecg, leads, old_sr = code15.load_ecg(ecg_file) 
        else:
            raise ValueError('Unsupported ECG format')

        ecg = ecg - ecg.mean(axis=1)[:, None]
        ecg = ecg / (ecg.std(axis=1)[:, None] + 1e-9)
        
        new_shape = int(new_sr * ecg.shape[1] / old_sr)
        ecg = resample(ecg, new_shape)

        if ecg.shape[1] > config.window:
            ecg = ecg[:, :config.window]

        if ecg.shape[1] < config.window:
            padded_ecg = np.zeros((ecg.shape[0], config.window), dtype=ecg.dtype)
            padded_ecg[:, :ecg.shape[1]] = ecg
            ecg = padded_ecg
        
        np.save(cache_name, ecg)


    ecg = np.load(cache_name)
    return ecg


def resample(ecg, shape):
    resized = cv2.resize(ecg, (shape, ecg.shape[0]))
    resized = resized.astype(ecg.dtype)
    return resized


def load_and_preprocess_list(ecg_files, config, cache_path):
    return np.stack([load_and_preprocess(ecg_file, config.ecg_sr, cache_path, config) for ecg_file in tqdm(ecg_files, desc='Loading ecg files', file=sys.stdout)])
    