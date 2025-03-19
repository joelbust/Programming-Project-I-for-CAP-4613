# ensembledataset.py
# also just about the same as task2, I do not think I changed
# anything
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class DigitDataset(Dataset):
    def __init__(self, file_path):
        data = np.loadtxt(file_path)
        self.labels = torch.tensor(data[:, 0], dtype=torch.long)
        self.features = torch.tensor(data[:, 1:], dtype=torch.float32)
        self.features = self.features.view(-1, 1, 16, 16)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

def get_dataloaders(batch_size=64):
    train_dataset = DigitDataset("zip_train.txt")
    test_dataset = DigitDataset("zip_test.txt")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader
