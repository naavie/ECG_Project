import os
import random
from glob import glob
import hashlib
import json
import h5py

import wfdb
import numpy as np
import pandas as pd
import torch
import cv2

from lib import codes

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def calsses_from_captions(captions, threshold=100):
    all_classes = [name.strip() for caption in captions for name in caption.strip().split(',')]
    counts = pd.Series(all_classes).value_counts()
    classes = counts[counts >= threshold].index.to_list()
    classes = sorted(classes)
    return classes

def generate_string_hash(some_string, cut=12):
    result = hashlib.md5(some_string.encode()).hexdigest()
    result = str(result)[:cut]
    return result

def generate_list_hash(some_list, cut=12):
    some_list = [str(el) for el in some_list]
    some_list = sorted(some_list)
    some_string = ''.join(some_list)
    return generate_string_hash(some_string, cut=cut)

def generate_dict_hash(some_dict, cut=12):
    some_list = list()
    keys = sorted(list(some_dict.keys()))
    for key in keys:
        some_list.append(key)
        some_list.append(some_dict[key])
    return generate_list_hash(some_list, cut=cut)

class CFG:
    to_hash = ['seed', 'ecg_sr', 'window', 'text_embedding_size', 'projection_dim', 'dropout', 'text_encoder_model', 'text_tokenizer',
         'temperature', 'head_lr', 'image_encoder_lr', 'epochs', 'max_length', 'ecg_encoder_model', 'ecg_embedding_size', 'ecg_channels',
         'normal_class', 'test_datasets', 'train_classes', 'noteval_classes', 'zeroshot_classes', 'pretrained', 'train_datasets'] 
    def __init__(self, cfg=None):
        if cfg is not None:
            for key, val in cfg.items():
                self.__dict__[key] = val

    def __repr__(self, ):
        result = ''
        for key, val in self.__dict__.items():
            if not key.startswith('__'):
                result += f'{key}: {val}\n'
        return result

    def hash(self, ):
        config = {k:v for k, v in self.__dict__.items() if not k.startswith('__')}
        config = {k: v for k, v in config.items() if k in self.to_hash}
        return generate_dict_hash(config)

def open_cfg(file):
    with open(file, 'rt') as f:
        data = json.loads(f.read())
    cfg = CFG(data)
    return cfg