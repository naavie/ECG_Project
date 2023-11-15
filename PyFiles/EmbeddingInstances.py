import torch
from dataset import PhysioNetDataset
from TextEncoders import TextEncoder
from ECGEncoder import ECGEncoder
from model import OneDimCNN

class InstanceSelector:
    def __init__(self, train_set, processed_train_df, text_encoder, ecg_encoder):
        self.train_set = train_set
        self.processed_train_df = processed_train_df
        self.text_encoder = text_encoder
        self.ecg_encoder = ecg_encoder

    def get_positive_instances(self):
        positive_instances = []
        for i in range(len(self.train_set)):
            # Generate ECG embedding for the current instance in the training set
            ecg_embedding = self.ecg_encoder.encode(self.train_set[i][1]['val'])
            # Generate dx_modality embedding for the current instance in the processed DataFrame
            dx_modality_embedding = self.text_encoder.encode(self.processed_train_df['dx_modality'][i])
            # If the ECG embedding and dx_modality embedding are equal, append them as a positive instance
            if torch.all(torch.eq(ecg_embedding, dx_modality_embedding)):
                positive_instances.append((ecg_embedding, dx_modality_embedding))
        return positive_instances

    def get_negative_instances(self):
        negative_instances = []
        # Get the positive instances
        positive_instances = self.get_positive_instances()
        for i in range(len(self.train_set)):
            # Generate ECG embedding for the current instance in the training set
            ecg_embedding = self.ecg_encoder.encode(self.train_set[i][1]['val'])
            for j in range(len(self.processed_train_df)):
                # Only consider dx_modality embeddings that are not at the same index as the current ECG embedding
                if i != j:
                    # Generate dx_modality embedding for the current instance in the processed DataFrame
                    dx_modality_embedding = self.text_encoder.encode(self.processed_train_df['dx_modality'][j])
                    # If the ECG embedding does not match any of the positive instance embeddings, append it as a negative instance
                    if not any(torch.all(torch.eq(ecg_embedding, pos[1])) for pos in positive_instances):
                        negative_instances.append((ecg_embedding, dx_modality_embedding))
        return negative_instances