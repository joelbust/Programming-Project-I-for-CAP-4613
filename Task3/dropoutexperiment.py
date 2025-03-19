# dropout_experiment.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from dropout_model import FullyConnectedNNWithDropout

class DigitDataset(Dataset):
    def __init__(self, file_path):
        data = np.loadtxt(file_path)
        self.labels = torch.tensor(data[:, 0], dtype=torch.long)
        self.features = torch.tensor(data[:, 1:], dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

def train_model(model, dataloader, lr=0.01, num_epochs=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr)
    criterion = nn.NLLLoss()
    loss_history = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for features, labels in dataloader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_loss = running_loss / len(dataloader)
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")
        loss_history.append(avg_loss)
    return loss_history

# Plotting
def plot_losses(losses, labels, title):
    for loss, label in zip(losses, labels):
        plt.plot(loss, label=label)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    plt.savefig(f"{title.replace(' ', '_')}.png")
    plt.show()

if __name__ == "__main__":
    # Load dataset (adjust paths as needed)
    train_dataset = DigitDataset("zip_train.txt")
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # Effective Dropout
    print("\nTraining with Dropout = 0.5 (Effective)")
    model_effective = FullyConnectedNNWithDropout(dropout_prob=0.5)
    loss_effective = train_model(model_effective, train_loader)

    # Ineffective Dropout
    print("\nTraining with Dropout = 0.8 (Ineffective)")
    model_ineffective = FullyConnectedNNWithDropout(dropout_prob=0.8)
    loss_ineffective = train_model(model_ineffective, train_loader)

    # Plot
    plot_losses([loss_effective, loss_ineffective],
                ["Dropout 0.5 (Effective)", "Dropout 0.8 (Ineffective)"],
                "Dropout Effect on Fully Connected NN")
