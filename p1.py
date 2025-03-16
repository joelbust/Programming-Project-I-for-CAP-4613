import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np


# Fully Connected Neural Network
class FullyConnectedNN(nn.Module):
    def __init__(self, input_size=256, hidden_size1=128, hidden_size2=128, hidden_size3=64, output_size=10):
        super(FullyConnectedNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, hidden_size3)
        self.fc4 = nn.Linear(hidden_size3, output_size)
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

# Locally Connected Neural Network
class LocallyConnectedNN(nn.Module):
    def __init__(self, output_size=10):
        super(LocallyConnectedNN, self).__init__()
        self.local1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.local2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.local3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 16 * 16, output_size)
        
    def forward(self, x):
        x = F.relu(self.local1(x))
        x = F.relu(self.local2(x))
        x = torch.sigmoid(self.local3(x))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x

# Convolutional Neural Network
class CNN(nn.Module):
    def __init__(self, output_size=10):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 16 * 16, output_size)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = torch.tanh(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x



