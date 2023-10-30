import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class OneDimCNN(nn.Module):
    def __init__(self, num_classes):
        super(OneDimCNN, self).__init__()
        
        # Layer 1
        self.conv1 = nn.Conv1d(in_channels=12, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)

        # Layer 2
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        # Layer 3
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm1d(64)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)

        # Layer 4
        self.conv4 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm1d(64)
        self.relu4 = nn.ReLU()
        self.pool4 = nn.MaxPool1d(kernel_size=2, stride=2)

        # Fully Connected Layer 1
        self.fc1 = nn.Linear(256*(1280/16), num_classes)  # If resampled to 128 Hz
        self.relu5 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)

        # Fully Connected Layer 2
        self.fc2 = nn.Linear(num_classes, num_classes)
        self.triplet_loss = TripletLoss(margin=1.0)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

    def forward(self, x1, x2=None, x3=None):
        if x2 is None and x3 is None:
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

            # Flatten
            x = x.reshape(x.size(0), -1)

            # Fully Connected Layer 1
            x = self.fc1(x)
            x = self.relu5(x)
            x = self.dropout1(x)

            # Fully Connected Layer 2
            x = self.fc2(x)

            return x

        else:
            output1 = self.forward(x1)
            output2 = self.forward(x2)
            output3 = self.forward(x3)
            
            loss = self.triplet_loss(output1, output2, output3)

            return loss
            