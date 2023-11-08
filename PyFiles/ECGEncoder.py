class ECGEncoder(nn.Module):
    def __init__(self):
        super(ECGEncoder, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        self.bert = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

    def forward(self, data):
        metadata, ecg_signal = data
        embeddings = []
        for lead_info, lead_signal in zip(metadata['leads_info'], ecg_signal):
            # Create a textual description of the ECG signal
            description = f"Lead {lead_info['lead_name']} with initial value {lead_info['initial_value']} and checksum {lead_info['checksum']} has signal values {lead_signal}"
            inputs = self.tokenizer(description, return_tensors="pt")
            outputs = self.bert(**inputs)
            embeddings.append(outputs.last_hidden_state[:, 0, :])
        return torch.cat(embeddings, dim=0)