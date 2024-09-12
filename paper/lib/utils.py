import os
import random
from glob import glob
import hashlib

import wfdb
import numpy as np
import pandas as pd
import torch
import cv2



def load_ann(file):
    ann = dict()
    with open(file, 'rt') as f:
        lines = f.readlines()

    num_leads = int(lines.pop(0).split()[1])

    leads = list()
    for i in range(num_leads):
        leads.append(lines.pop(0).split()[-1])

    ann['num_leads'] = num_leads
    
    for index, lead in enumerate(leads):
        ann[f'lead_{lead}'] = index
#     ann['leads'] = np.array(leads)

    for line in lines:
        if line.startswith('# Age'):
            ann['age'] = float(line.split(' ')[-1].strip())
        if line.startswith('# Sex'):
            ann['sex'] = line.split(' ')[-1].strip()
        if line.startswith('# Dx'):
            diagnoses = line.split()[-1].strip()
            if diagnoses != '':
                for diagnosis in diagnoses.split(','):
                    if diagnosis != '':
                        ann[f'Dx_{diagnosis.strip()}'] = 1
    return ann

def load_dxs(file, decode_dict):
    ann = load_ann(file)
    code_list = [decode_dict[key][1] for key, val in ann.items() if key in decode_dict]
    return ', '.join(code_list)

def load_wsdb(file):
    file = os.path.splitext(file)[0]
    record = wfdb.io.rdrecord(file)
    ecg = record.p_signal.T.astype('float32')
    leads = tuple(record.sig_name)
    sr = record.fs
    ecg[np.isnan(ecg)] = 0.0
    return ecg, leads, sr

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_data(data_path, decode_dict):
    ecgs = glob(f'{data_path}/*/*.hea')
    df = pd.DataFrame(ecgs)
    df.columns = ['ecg_file', ]
    df['label'] = df['ecg_file'].apply(lambda x: load_dxs(x, decode_dict)) 
    return df

def get_data_cached(data_path, decode_dict, cache_path):
    if not os.path.isfile(cache_path):
        df = get_data(data_path, decode_dict)
        df.to_csv(cache_path, index=False)
    return pd.read_csv(cache_path)



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
