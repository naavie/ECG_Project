class ECGEncoder(OneDimCNN):
    def __init__(self, num_classes, embedding_size=768):
        super(ECGEncoder, self).__init__(num_classes)
        self.fc = nn.Linear(133, embedding_size)  # Add a fully connected layer

    def encode(self, signal):
        encoded_signal = self.forward(signal)
        encoded_signal = self.fc(encoded_signal)  # Transform the embedding to the common size
        return encoded_signal