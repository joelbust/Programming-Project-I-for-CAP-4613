import torch
import torch.nn as nn
import torch.optim as optim
from utils import initialize_weights

def train_model(model, train_loader, lr=0.01, momentum=0.9, weight_init="balanced_variance", num_epochs=5):
    """
    Trains a neural network model.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Apply weight initialization
    initialize_weights(model, weight_init)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    loss_history = []

    print(f"\n Training {model.__class__.__name__} | LR: {lr}, Momentum: {momentum}, Weight Init: {weight_init}")

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        loss_history.append(avg_loss)
        print(f"Epoch {epoch + 1}, LR {lr}, Momentum {momentum}, Loss: {avg_loss:.4f}")

    return loss_history
