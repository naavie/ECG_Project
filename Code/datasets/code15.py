import pandas as pd
import numpy as np
import h5py


def load_df():
    decode_dict_code15 = dict()
    decode_dict_code15['1dAVb'] = '1st degree AV block'
    decode_dict_code15['RBBB'] = 'right bundle branch block'
    decode_dict_code15['LBBB'] = 'left bundle branch block'
    decode_dict_code15['SB'] = 'sinus bradycardia'
    decode_dict_code15['ST'] = 'sinus tachycardia'
    decode_dict_code15['AF'] = 'atrial fibrillation'

    def row_to_label(row):
            result = list()
            for key, val in decode_dict_code15.items():
                if row[key]:
                    result.append(val)
            #if len(result) == 0:
            #    result.append('normal ecg')
            return ', '.join(result)

    path = '/ayb/vol1/datasets/ecg_datasets/code15'
    df = pd.read_csv(path + '/exams.csv')
    df['ecg_file'] = df['exam_id'].apply(lambda x: str(x)) + '<EXAM_ID>' + path + '/' + df['trace_file']
    df['label'] = df.apply(row_to_label, axis=1)
    df = df[['ecg_file', 'patient_id', 'label']]
    return df    



def load_ecg(ecg_file):
    path_to_hdf5 = ecg_file.split('<EXAM_ID>')[1]
    ecg_id = int(ecg_file.split('<EXAM_ID>')[0])

    f = h5py.File(path_to_hdf5, "r")

    index = np.where(np.array(f['exam_id']) == ecg_id)[0][0]
    ecg = f['tracings'][index]
    ecg = ecg[:4000, :].T.astype('float32')

    fs = 400
    leads = ('I', 'II', 'III', 'aVF', 'aVR', 'aVL', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6')
    
    return ecg, leads, fs