import torch
import torch.nn as nn
import torch.nn.functional as F

# Fully Connected Neural Network (FCNN)
class FullyConnectedNN(nn.Module):
    def __init__(self, input_size=256, hidden_size1=128, hidden_size2=128, hidden_size3=64, output_size=10):
        super(FullyConnectedNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, hidden_size3)
        self.fc4 = nn.Linear(hidden_size3, output_size)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten input
        x = F.relu(self.fc1(x))  # Using ReLU
        x = torch.tanh(self.fc2(x))  # Using Tanh
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return F.log_softmax(x, dim=1)  # Log Softmax for classification

# Locally Connected Neural Network (LCNN - No Weight Sharing)
class LocallyConnectedNN(nn.Module):
    def __init__(self):
        super(LocallyConnectedNN, self).__init__()
        self.local1 = nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=0)  # No weight sharing
        self.local2 = nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=0)
        self.local3 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=0)
        self.fc1 = nn.Linear(4 * 4 * 64, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(self.local1(x))  # Using ReLU
        x = torch.tanh(self.local2(x))  # Using Tanh
        x = F.relu(self.local3(x))
        x = x.view(x.size(0), -1)  # Flatten before FC layers
        x = F.relu(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return F.log_softmax(x, dim=1)

# Convolutional Neural Network (CNN - Weight Sharing)
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)  
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(128 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # Flatten before FC layers
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))  # Using Sigmoid
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return F.log_softmax(x, dim=1)
