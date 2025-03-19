# ensemblemodels.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# code is almost the exact same as task2 except I hardcoded some values for
# less confusion on my end, specifically in the FCNN
# Fully Connected Neural Network (FCNN)
class FullyConnectedNN(nn.Module):
    def __init__(self):
        super(FullyConnectedNN, self).__init__()
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1) 
        x = F.relu(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return F.log_softmax(x, dim=1)

# Locally Connected Neural Network (LCNN - No Weight Sharing)
class LocallyConnectedNN(nn.Module):
    def __init__(self):
        super(LocallyConnectedNN, self).__init__()
        self.local1 = nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=0)
        self.local2 = nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=0)
        self.local3 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=0)
        self.fc1 = nn.Linear(4 * 4 * 64, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(self.local1(x))
        x = torch.tanh(self.local2(x))
        x = F.relu(self.local3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return F.log_softmax(x, dim=1)

# Convolutional Neural Network (CNN - Weight Sharing)
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(128 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return F.log_softmax(x, dim=1)
