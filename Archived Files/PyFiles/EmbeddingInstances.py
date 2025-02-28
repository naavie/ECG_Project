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

    def get_negative_instances(self, current_set, current_set_df, num_neg_instances=5):
        # Initialize an empty list to store the negative instances
        negative_instances = []

        # Loop over each record in the current_set
        for i in tqdm(range(len(current_set)), desc="Generating negative instances"):
            # Convert the 'val' field of the current record to a PyTorch tensor and store it in the 'signal' variable
            signal = torch.from_numpy(current_set[i][1]['val']).float()

            # Add an extra dimension to 'signal' to represent the channels
            signal = signal.unsqueeze(0)

            # Encode the 'signal' using the ECG encoder and store the result in the 'ecg_embedding' variable
            ecg_embedding = self.ecg_encoder.encode(signal)

            # Randomly select 'num_neg_instances' indices from current_set_df
            for j in random.sample(range(len(current_set_df)), num_neg_instances):
                # Check if the current index is not the same as the outer loop index
                if i != j:
                    # Encode the 'dx_modality' field of the selected record using the text encoder and store the result in the 'dx_modality_embedding' variable
                    dx_modality_embedding = self.text_encoder.encode(current_set_df['dx_modality'][j])

                    # Check if the ECG embedding is not equal to the DX modality embedding
                    if not torch.all(torch.eq(ecg_embedding.float(), dx_modality_embedding)):
                        # If they are not equal, append the pair (ECG embedding, DX modality embedding) to the 'negative_instances' list
                        negative_instances.append((ecg_embedding, dx_modality_embedding))

        # Return the list of negative instances
        return negative_instances