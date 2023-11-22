class ECGEncoder(OneDimCNN):
    def __init__(self, num_classes):
        super(ECGEncoder, self).__init__(num_classes)

    def encode(self, signal):
        # Encode the signal using the OneDimCNN model.
        encoded_signal = self.forward(signal)

        # Return the encoded signal.
        return encoded_signal