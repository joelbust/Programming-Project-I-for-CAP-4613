# ensembleexperiment.py

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

from ensembledataset import get_dataloaders
from ensemblemodels import FullyConnectedNN, LocallyConnectedNN, CNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_model(model, dataloader, epochs=5):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.NLLLoss()

    model.train()
    for epoch in range(epochs):
        running_loss = 0
        for features, labels in dataloader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"{model.__class__.__name__} Epoch {epoch+1}, Loss: {running_loss / len(train_loader):.4f}")
    return model

def evaluate_model(model, test_loader):
    # Evaluation mode, prevents randomness
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for features, labels in test_loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            # Gets the highest score from outputs
            preds = outputs.argmax(dim=1)
            # Add up the correct predictions
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    # Return the accuracy
    return correct / total

def ensemble_evaluate(models, test_loader):
    # Evaluation mode, prevents randomness
    for model in models:
        model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for features, labels in test_loader:
            features, labels = features.to(device), labels.to(device)
            outputs = []
            for model in models:
                out = model(features)
                # Convert logsoftmax back to a probability
                outputs.append(torch.exp(out))
            # Get all predictions together and get the mean
            avgout = torch.mean(torch.stack(outputs), dim=0)
            # Grab the highest probability
            preds = avgout.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    # Return the accuracy
    return correct / total

if __name__ == "__main__":
    train_loader, test_loader = get_dataloaders()

    # Initialize models
    fcnn = FullyConnectedNN()
    lcnn = LocallyConnectedNN()
    cnn = CNN()

    # Train models
    fcnn = train_model(fcnn, train_loader)
    lcnn = train_model(lcnn, train_loader)
    cnn = train_model(cnn, train_loader)

    # Evaluate models
    fcnn_acc = evaluate_model(fcnn, test_loader)
    lcnn_acc = evaluate_model(lcnn, test_loader)
    cnn_acc = evaluate_model(cnn, test_loader)

    print(f"\nFullyConnectedNN Test Accuracy: {fcnn_acc * 100:.2f}%")
    print(f"LocallyConnectedNN Test Accuracy: {lcnn_acc * 100:.2f}%")
    print(f"CNN Test Accuracy: {cnn_acc * 100:.2f}%")

    # The ensemble
    ensemble_acc = ensemble_evaluate([fcnn, lcnn, cnn], test_loader)
    print(f"\nEnsemble Test Accuracy: {ensemble_acc * 100:.2f}%")
