import torch
import torch.nn as nn
import torch.nn.functional as F


# all essentially the same as the one in task 2 but with dropout now included
class FullyConnectedNNWithDropout(nn.Module):
    def __init__(self, input_size=256, hidden_size1=128, hidden_size2=128, hidden_size3=64, output_size=10, dropout_prob=0.5):
        super(FullyConnectedNNWithDropout, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, hidden_size3)
        self.fc4 = nn.Linear(hidden_size3, output_size)
        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.tanh(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.fc4(x)
        return F.log_softmax(x, dim=1)
