import torch
import torch.optim as optim
from models import FullyConnectedNN
from dataset import get_dataloaders
from utils import initialize_weights, plot_losses

def train_model(model, train_loader, lr, momentum=0.9, epochs=5, weight_init="balanced_variance"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    initialize_weights(model, weight_init)
    
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    criterion = torch.nn.CrossEntropyLoss()
    
    loss_history = []
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs.view(inputs.size(0), -1))  
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        loss_history.append(avg_loss)
        print(f"Epoch {epoch+1}, LR {lr}, Momentum {momentum}, Loss: {avg_loss:.4f}")

    return loss_history

if __name__ == "__main__":
    train_loader, _ = get_dataloaders()

    # Test Different Initialization Strategies
    fc_balanced_variance = FullyConnectedNN()
    loss_balanced_variance = train_model(fc_balanced_variance, train_loader, lr=0.01, weight_init="balanced_variance")

    fc_activation_specific = FullyConnectedNN()
    loss_activation_specific = train_model(fc_activation_specific, train_loader, lr=0.01, weight_init="activation_specific")

    fc_random_baseline = FullyConnectedNN()
    loss_random_baseline = train_model(fc_random_baseline, train_loader, lr=0.01, weight_init="random_baseline")

    plot_losses([loss_balanced_variance, loss_activation_specific, loss_random_baseline], ["balanced_variance", "activation_specific", "random_baseline"], "Impact of Weight Initialization")
