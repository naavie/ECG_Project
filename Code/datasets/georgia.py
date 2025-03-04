from lib.datasets.comp2021_shared import _load_df, _load_ecg




def load_df():
    return _load_df('/ayb/vol1/datasets/ecg_datasets/physionet.org/files/challenge-2021/1.0.3/training/georgia')

def load_ecg(ecg_file):
    return _load_ecg(ecg_file)


