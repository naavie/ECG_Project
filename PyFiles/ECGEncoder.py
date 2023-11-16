class ECGEncoder(OneDimCNN):
    def __init__(self, num_classes):
        super(ECGEncoder, self).__init__(num_classes)
        self.fc3 = nn.Linear(128, 768)  # Add an additional fully connected layer

    def forward(self, x):
        x = super().forward(x)
        x = self.fc3(x)  # Pass the output through the additional fully connected layer
        return x