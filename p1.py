import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np


# Define all three network architectures
class FullyConnectedNN(nn.Module):
    def __init__(self):
        super(FullyConnectedNN, self).__init__()
        self.fc1 = nn.Linear(256, 128)   
        self.fc2 = nn.Linear(128, 64)   
        self.fc3 = nn.Linear(64, 32)   
        self.fc4 = nn.Linear(32, 10)   
        
    def forward(self, x):
        x = x.view(-1, 256)  
        x = F.relu(self.fc1(x))  
        x = torch.tanh(self.fc2(x))  
        x = torch.sigmoid(self.fc3(x))  
        x = self.fc4(x)  
        return x

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # Convolutional layers with padding and kernel_size=3
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)  
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1) 
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)  
        self.pool = nn.MaxPool2d(2, 2) 

        self.fc1 = nn.Linear(128 * 8 * 8, 256)  # Flattened input to 256 nodes
        self.fc2 = nn.Linear(256, 128)          
        self.fc3 = nn.Linear(128, 64)           
        self.fc4 = nn.Linear(64, 10)            

    def forward(self, x):
        # Apply convolutional layers with ReLU activations
        x = torch.relu(self.conv1(x))  
        x = torch.relu(self.conv2(x))  
        x = torch.relu(self.conv3(x))  
        
        # Apply pooling once after all convolutional layers
        x = self.pool(x)  
        
        # Flatten the output for the fully connected layer
        x = x.view(x.size(0), -1) 

        # Fully connected layers
        x = torch.relu(self.fc1(x)) 
        x = torch.sigmoid(self.fc2(x))  
        x = torch.tanh(self.fc3(x))     
        x = self.fc4(x)                
        
        return x


class LocallyConnectedLayer(nn.Module):
    def __init__(self, in_height, in_width, in_channels, out_channels, kernel_size, stride=1):
        super(LocallyConnectedLayer, self).__init__()
        self.in_height = in_height
        self.in_width = in_width
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        
        # Calculate output dimensions
        self.out_height = (in_height - kernel_size) // stride + 1
        self.out_width = (in_width - kernel_size) // stride + 1
        
        # Create weight and bias for each output position
        self.weight = nn.Parameter(
            torch.randn(
                self.out_height, self.out_width, 
                out_channels, in_channels, kernel_size, kernel_size
            ) * 0.01
        )
        self.bias = nn.Parameter(
            torch.zeros(self.out_height, self.out_width, out_channels)
        )
        
    def forward(self, x):
        batch_size = x.size(0)
        output = torch.zeros(
            batch_size, self.out_channels, self.out_height, self.out_width, 
            device=x.device
        )
        
        # For each output position
        for i in range(self.out_height):
            for j in range(self.out_width):
                h_start = i * self.stride
                h_end = h_start + self.kernel_size
                w_start = j * self.stride
                w_end = w_start + self.kernel_size
                
                # Extract local region
                local_region = x[:, :, h_start:h_end, w_start:w_end]
                local_region = local_region.reshape(batch_size, -1)  # Flatten local region
                
                # Reshape weight for matrix multiplication
                weight = self.weight[i, j].reshape(self.out_channels, -1)
                
                # Perform matrix multiplication and add bias
                output[:, :, i, j] = torch.matmul(weight, local_region.t()).t() + self.bias[i, j]
                
        return output

#Locally Connected Neural Network without weight sharing
class LocallyConnectedNN(nn.Module):
    def __init__(self):
        super(LocallyConnectedNN, self).__init__()
        self.local1 = LocallyConnectedLayer(
            in_height=16, in_width=16, in_channels=1, 
            out_channels=16, kernel_size=5, stride=1
        )
        self.local2 = LocallyConnectedLayer(
            in_height=12, in_width=12, in_channels=16,
            out_channels=32, kernel_size=5, stride=1
        )
        self.local3 = LocallyConnectedLayer(
            in_height=8, in_width=8, in_channels=32,
            out_channels=64, kernel_size=5, stride=1
        )
        self.fc = nn.Linear(4 * 4 * 64, 10)
        
    def forward(self, x):
        x = x.view(-1, 1, 16, 16)
        x = torch.sigmoid(self.local1(x))  
        x = torch.tanh(self.local2(x))  
        x = F.relu(self.local3(x))  
        x = x.view(-1, 4 * 4 * 64)
        x = self.fc(x)  
        return x



