import scipy.io as sio
import os
import matplotlib.pyplot as plt
import pandas as pd
import torch
import numpy as np
import h5py

class PhysioNetDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, train=False):
        self.dataset_path = dataset_path
        self.dataset_path = [path for path in self.dataset_path if "index.html" not in path]
        self.train = train
        self.file_list = os.listdir(dataset_path)
        self._hea_files = []
        self._mat_files = []
        self._indices_files = []
        self._hea_files_path = []
        self._mat_files_path = []

        self.file_PATHS = []  # Directory to main database folders
        self.data_files = []  # Directory to data files

        # Validation Case: PTB Databases only
        if self.train == False:
            validation_datasets = ['ptb', 'ptb-xl']
            for file in os.listdir(dataset_path):
                if file in validation_datasets:
                    file_path = os.path.join(dataset_path, file)
                    file_path = file_path.replace('\\', '/')
                    self.file_PATHS.append(file_path)

        # Training Case: All Databases excluding PTB
        else:
            validation_datasets = ['ptb', 'ptb-xl']
            for file in os.listdir(dataset_path):
                if file not in validation_datasets:
                    file_path = os.path.join(dataset_path, file)
                    file_path = file_path.replace('\\', '/')
                    self.file_PATHS.append(file_path)

        for path in self.file_PATHS:
            if os.path.isdir(path):
                for sub_folder in os.listdir(path):
                    sub_folder_path = os.path.join(path, sub_folder)
                    sub_folder_path = sub_folder_path.replace('\\', '/')
                    
                    # Ignore index.html files
                    if sub_folder_path.endswith('index.html'):
                        self._indices_files.append(sub_folder_path)
                    else:
                        if os.path.isdir(sub_folder_path):
                            for file in os.listdir(sub_folder_path):
                                # Get all .hea files
                                if file.endswith('.hea'):
                                    file_path = os.path.join(sub_folder_path, file)
                                    file_path = file_path.replace('\\', '/')
                                    self._hea_files.append(file_path)
                                    self._hea_files_path.append(file_path)
                                # Get all .mat files
                                elif file.endswith('.mat'):
                                    file_path = os.path.join(sub_folder_path, file)
                                    file_path = file_path.replace('\\', '/')
                                    self._mat_files.append(file_path)
                                    self._mat_files_path.append(file_path)

    def __getitem__(self, index):
        # 1. Get .hea file
        hea_file_path = self._hea_files[index]
        with open(hea_file_path, 'r') as f:
            header = print(f.read())

        # 2. Get .mat file
        mat_file_path = self._mat_files[index]
        twelve_lead_ecg = sio.loadmat(mat_file_path)
        
        return header, twelve_lead_ecg

    def plot_record(self, index):
        data = sio.loadmat(mat_file_path)
        fig, axs = plt.subplots(nrows=3, ncols=4, figsize=(12, 8))

        for i, ax in enumerate(axs.flat):
            ax.plot(data['val'][i])
            ax.set_xlabel('Sample')
            ax.set_ylabel('Amplitude')
            ax.set_title(f'Lead {i+1}')

        plt.tight_layout()
        plt.show()

    def __len__(self):
        return len(self.file_PATHS)


class CODE15Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, train=False):
        self.dataset_path = sorted(list(dataset_path))
        self.train = train
        self.file_list = os.listdir(dataset_path)
        self._exam_ids = []
        self._tracings_group = []
        self._hdf5_file_PATHS = []
        
        """
        exams.csv column names: 
            - exam_id: id used for identifying the exam
            - age: patient age in years at the moment of the exam
            - is_male: True if the patient is male
            - nn_predicted_age: age predicted by a neural network to the patient
            - 6 modalities: 
                1. First degree atrioventricular block: 1dAVb
                2. Right bundle branch block: RBBB
                3. Left bundle branch: LBBB
                4. Sinus bradycardia: SB
                5. Atrial fibrillation: AF
                6. Sinus tachycardia: ST
            - patient_id: id used for identifying the patient
            - normal_ecg: True if the patient has a normal ECG
            - death: True if the patient dies in the follow-up time. This data is available only in the first exam of the patient.
            - timey: If the patient dies it is the time to the death of the patient
            - trace_file: identify in which hdf5 file the file corresponding to this patient is located
            - exams_part{i}.hdf5: The HDF5 file contains two datasets named `tracings` and other named `exam_id`
                - The `exam_id` is a tensor of dimension `(N,)` containing the exam id (the same as in the csv file) 
                - `tracings` is a `(N, 4096, 12)` tensor containing the ECG tracings in the same order. 
                    - The first dimension corresponds to the different exams; 
                    - The second dimension corresponds to the 4096 signal samples
                    - The third dimension to the 12 different leads of the ECG exams in the following order: `{DI, DII, DIII, AVR, AVL, AVF, V1, V2, V3, V4, V5, V6}`. 
                    - The signals are sampled at 400 Hz. Some signals originally have a duration of 10 seconds (10 * 400 = 4000 samples) and others of 7 seconds (7 * 400 = 2800 samples). 
                    - In order to make them all have the same size (4096 samples), we fill them with zeros on both sizes. For instance, for a 7 seconds ECG signal with 2800 samples we include 648 samples at the beginning and 648 samples at the end, yielding 4096 samples that are then saved in the hdf5 dataset
        """
        exams = os.path.join(dataset_path, 'exams.csv')
        exams = exams.replace('\\', '/')
        self.exams = pd.read_csv(exams)

        # Test data
        if self.train == False:
            test_set = ['exams_part13.hdf5', 'exams_part14.hdf5', 'exams_part15.hdf5', 'exams_part16.hdf5', 'exams_part17.hdf5', 'exams.csv']
            for file in os.listdir(dataset_path):
                if file in test_set:
                    file_path = os.path.join(dataset_path, file)
                    file_path = file_path.replace('\\', '/')
                    self._hdf5_file_PATHS.append(file_path)        

        # Train data
        else:
            test_set = ['exams_part13.hdf5', 'exams_part14.hdf5', 'exams_part15.hdf5', 'exams_part16.hdf5', 'exams_part17.hdf5', 'exams.csv']
            for file in os.listdir(dataset_path):
                if file not in test_set:
                    file_path = os.path.join(dataset_path, file)
                    file_path = file_path.replace('\\', '/')
                    self._hdf5_file_PATHS.append(file_path)
        
        for file_path in self._hdf5_file_PATHS:
            with h5py.File(file_path, 'r') as f:
                for i in range(len(f['exam_id'])):
                    exam_id = np.array(f['exam_id'][i])
                    tracing = f['tracings'][i]
                    self._exam_ids.append(exam_id)
                    self._tracings_group.append(tracing)
    
    def __getitem__(self, index):
        # TODO
        # Get the exam_id and tracing for the given index
        exam_id = self._exam_ids[index]
        tracing = self._tracings_group[index]
        return exam_id, tracing

    def __len__(self):
        return len(self._hdf5_file_PATHS)
    