from experiments import train_model
from models import FullyConnectedNN
from dataset import get_dataloaders
from utils import plot_losses
from train import train_model 

train_loader, _ = get_dataloaders()

# Learning Rate Experiment
loss_low = train_model(FullyConnectedNN(), train_loader, lr=0.0001)
loss_optimal = train_model(FullyConnectedNN(), train_loader, lr=0.01)
loss_high = train_model(FullyConnectedNN(), train_loader, lr=1)

plot_losses([loss_low, loss_optimal, loss_high], ["Low LR", "Optimal LR", "High LR"], "Effect of Learning Rate")
