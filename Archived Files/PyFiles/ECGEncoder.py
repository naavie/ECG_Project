import torch.nn as nn

class ECGEncoder(nn.Module):
    def __init__(self):
        super(ECGEncoder, self).__init__()

        # Layer 1
        self.conv1 = nn.Conv1d(in_channels=12, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(16)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.AvgPool1d(kernel_size=2, stride=2)

        # Layer 2
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(32)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.AvgPool1d(kernel_size=2, stride=2)

        # Layer 3
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm1d(64)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.AvgPool1d(kernel_size=2, stride=2)

        # Layer 4
        self.conv4 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm1d(128)
        self.relu4 = nn.ReLU()
        self.pool4 = nn.AvgPool1d(kernel_size=2, stride=2)

        # Layer 5
        self.conv5 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm1d(256)
        self.relu5 = nn.ReLU()
        self.pool5 = nn.AvgPool1d(kernel_size=2, stride=2)

        # Layer 6
        self.conv6 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.bn6 = nn.BatchNorm1d(512)
        self.relu6 = nn.ReLU()
        self.pool6 = nn.AvgPool1d(kernel_size=2, stride=2)

        # Layer 7
        self.conv7 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.bn7 = nn.BatchNorm1d(512)
        self.relu7 = nn.ReLU()
        self.pool7 = nn.AvgPool1d(kernel_size=2, stride=2)

        # Layer 8
        self.conv8 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.bn8 = nn.BatchNorm1d(512)
        self.relu8 = nn.ReLU()
        self.pool8 = nn.AvgPool1d(kernel_size=2, stride=2)

        # Layer 9
        self.conv9 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.bn9 = nn.BatchNorm1d(512)
        self.relu9 = nn.ReLU()
        self.pool9 = nn.AvgPool1d(kernel_size=2, stride=2)

        # Layer 10
        self.conv10 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.bn10 = nn.BatchNorm1d(512)
        self.relu10 = nn.ReLU()
        self.pool10 = nn.AvgPool1d(kernel_size=2, stride=2)

        # Fully Connected Layer 1
        # self.fc1 = nn.Linear(512*4, 768)
        self.relu11 = nn.ReLU()

        # Add embed_dim attribute
        self.embed_dim = 768

    def forward(self, x):
        # Layer 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        # Layer 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        # Layer 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.pool3(x)

        # Layer 4
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)
        x = self.pool4(x)

        # Layer 5
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu5(x)
        x = self.pool5(x)

        # Layer 6
        x = self.conv6(x)
        x = self.bn6(x)
        x = self.relu6(x)
        x = self.pool6(x)

        # Layer 7
        x = self.conv7(x)
        x = self.bn7(x)
        x = self.relu7(x)
        x = self.pool7(x)

        # Layer 8
        x = self.conv8(x)
        x = self.bn8(x)
        x = self.relu8(x)
        x = self.pool8(x)

        # Layer 9
        x = self.conv9(x)
        x = self.bn9(x)
        x = self.relu9(x)
        x = self.pool9(x)

        # Layer 10
        x = self.conv10(x)
        x = self.bn10(x)
        x = self.relu10(x)
        x = self.pool10(x)

        # Flatten the output of the convolutional layers
        print(x.size())
        x = x.view(x.size(0), -1)

        # Initialize self.fc1 here, using the size of x
        if not hasattr(self, 'fc1'):
            self.fc1 = nn.Linear(x.size(1), self.embed_dim).to(x.device)
        x = self.fc1(x)
        x = self.relu11(x)

        return x