import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class OneDimCNN(nn.Module):
    def __init__(self, num_classes):
        super(OneDimCNN, self).__init__()

        # Layer 1
        self.conv1 = nn.Conv1d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.AvgPool1d(kernel_size=2, stride=2)

        # Layer 2
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.AvgPool1d(kernel_size=2, stride=2)

        # Layer 3
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.AvgPool1d(kernel_size=2, stride=2)

        # Layer 4
        self.conv4 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm1d(256)
        self.relu4 = nn.ReLU()
        self.pool4 = nn.AvgPool1d(kernel_size=2, stride=2)

        # Fully Connected Layer 1
        self.fc1 = nn.Linear(79872, 128)
        self.relu5 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)

        # Fully Connected Layer 2
        self.fc2 = nn.Linear(128, num_classes)

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

        # Flatten the tensor
        x = x.view(x.size(0), -1)
        print(x.shape)  # Add this line

        # Fully Connected Layer 1
        x = self.fc1(x)
        x = self.relu5(x)
        x = self.dropout1(x)

        # Fully Connected Layer 2
        x = self.fc2(x)

        return x