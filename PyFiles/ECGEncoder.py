class ECGEncoder(OneDimCNN):
    def __init__(self, num_classes):
        super(ECGEncoder, self).__init__(num_classes)
        self.fc3 = nn.Linear(126, 768)  # New linear layer

    def encode(self, signal):
        signal = torch.tensor(signal, dtype=torch.float).unsqueeze(0)
        embedding = self.forward(signal)
        return self.fc3(embedding)  # Apply the new linear layer