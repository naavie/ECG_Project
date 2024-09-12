import os

import torch
from tqdm import tqdm
import cv2
import numpy as np

from lib.utils import load_wsdb, generate_string_hash

class CLIP_ECG_Dataset(torch.utils.data.Dataset):
    def __init__(self, df, config):
        self.df = df
        self.config = config
        
        self.ecg_files = self.df['ecg_file'].values
        self.captions = self.df['label'].values

        self.ecgs = load_and_preprocess_list(self.ecg_files, self.config, config.cache_path)
        
    def __len__(self, ):
        return len(self.df)

    def __getitem__(self, idx):
        caption = self.captions[idx]
        image = self.ecgs[idx]
        return {'image': image, 'caption': caption}
        
    def process_ecg(self, ecg, sr):
        new_shape = int(self.config.ecg_sr * ecg.shape[1] / sr)
        ecg = resample(ecg, new_shape)
        return ecg
    
def load_and_preprocess(ecg_file, new_sr, cache_path):
    ecg_file_adj = f'{ecg_file}_{new_sr}'
    ecg_fie_hash = generate_string_hash(ecg_file_adj)
    cache_name = f'{cache_path}/{ecg_fie_hash}.npy'
    if not os.path.isfile(cache_name):
        ecg, leads, old_sr = load_wsdb(ecg_file)    
        new_shape = int(new_sr * ecg.shape[1] / old_sr)
        ecg = resample(ecg, new_shape)
        np.save(cache_name, ecg)
    return np.load(cache_name)


def resample(ecg, shape):
    resized = cv2.resize(ecg, (shape, ecg.shape[0]))
    resized = resized.astype(ecg.dtype)
    return resized


def load_and_preprocess_list(ecg_files, config, cache_path):
    return np.stack([load_and_preprocess(ecg_file, config.ecg_sr, cache_path) for ecg_file in tqdm(ecg_files)])
    