import pandas as pd
from glob import glob
import h5py

def load_df():
    data_path = '/ayb/vol1/datasets/ecg_datasets/SPH'
    description_dict = pd.read_csv(f'{data_path}/code.csv').set_index('Code')['Description'].to_dict()
    ecg_files = sorted(glob(f'{data_path}/records/*.h5'))
    df = pd.read_csv(f'{data_path}/metadata.csv')
    df['primary_codes'] = df['AHA_Code'].str.split(';').apply(remove_nonprimary_code)
    df['label'] = df['primary_codes'].apply(codes_to_caption(description_dict))
    df['ecg_file'] = df['ECG_ID'].apply(lambda x: f'{data_path}/records/{x}.h5')
    df['patient_id'] = df['Patient_ID']
    df = df[['ecg_file', 'patient_id', 'label']]
    return df

def remove_nonprimary_code(x):
    r = []
    for cx in x:
        for c in cx.split('+'):
            if int(c) < 200 or int(c) >= 500:
                if c not in r:
                    r.append(c)
    return r

def codes_to_caption(description_dict):
    def _codes_to_caption(codes):
        classes = [description_dict[int(code)].lower() for code in codes]
        caption = ', '.join(classes)
        return caption
    return _codes_to_caption


def load_ecg(file):
    with h5py.File(file, 'r') as f:
        signal = f['ecg'][()]
    fs = 500
    leads = ('I', 'II', 'III', 'aVF', 'aVR', 'aVL', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6')
    return signal.astype('float32'), leads, fs