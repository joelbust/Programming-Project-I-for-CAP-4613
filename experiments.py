import torch
from models import FullyConnectedNN, CNN, LocallyConnectedNN
from dataset import get_dataloaders
from training import train_model
from utils import plot_losses

# Load dataset
train_loader, _ = get_dataloaders()

# Define models
models = {
    "FullyConnectedNN": FullyConnectedNN(),
    "CNN": CNN(),
    "LocallyConnectedNN": LocallyConnectedNN()
}

###  (1) Parameter Initialization Experiment
losses_weight_init = {}

for model_name, model in models.items():
    print(f"\n Training {model_name} with different weight initializations...")
    losses_weight_init[model_name] = {
        "Random": train_model(model, train_loader, lr=0.01, weight_init="random_baseline"),
        "Balanced Variance": train_model(model, train_loader, lr=0.01, weight_init="balanced_variance"),
        "Activation Specific": train_model(model, train_loader, lr=0.01, weight_init="activation_specific")
    }

    # Save plots with unique filenames
    plot_losses(
        list(losses_weight_init[model_name].values()),
        list(losses_weight_init[model_name].keys()),
        f"Weight_Initialization_{model_name}"
    )

### (2) Learning Rate Experiment
losses_learning_rate = {}

for model_name, model in models.items():
    print(f"\n Training {model_name} with different learning rates...")

    # Set weight initialization method
    weight_init_method = "random_baseline" if model_name == "LocallyConnectedNN" else "balanced_variance"

    losses_learning_rate[model_name] = {
        "Low LR (0.0001)": train_model(model, train_loader, lr=0.0001, weight_init=weight_init_method),
        "Optimal LR (0.01)": train_model(model, train_loader, lr=0.01, weight_init=weight_init_method),
        "High LR (1)": train_model(model, train_loader, lr=1, weight_init=weight_init_method)
    }

    # Save plots with unique filenames
    plot_losses(
        list(losses_learning_rate[model_name].values()),
        list(losses_learning_rate[model_name].keys()),
        f"Learning_Rate_{model_name}"
    )

###  (3) Batch Size Experiment (CNN only)
print("\n Training CNN with different batch sizes...")
small_batch_loader, _ = get_dataloaders(batch_size=16)
large_batch_loader, _ = get_dataloaders(batch_size=256)

cnn_model = CNN()
loss_small_batch = train_model(cnn_model, small_batch_loader, lr=0.01)
loss_large_batch = train_model(cnn_model, large_batch_loader, lr=0.01)

#  Save plots with unique filenames
plot_losses(
    [loss_small_batch, loss_large_batch],
    ["Small Batch (16)", "Large Batch (256)"],
    "Batch_Size_CNN"
)

###  (4) Momentum Experiment (CNN only)
print("\n Training CNN with different momentum values...")
loss_momentum_05 = train_model(CNN(), train_loader, lr=0.01, momentum=0.5)
loss_momentum_09 = train_model(CNN(), train_loader, lr=0.01, momentum=0.9)
loss_momentum_099 = train_model(CNN(), train_loader, lr=0.01, momentum=0.99)

#  Save plots with unique filenames
plot_losses(
    [loss_momentum_05, loss_momentum_09, loss_momentum_099],
    ["Momentum 0.5", "Momentum 0.9", "Momentum 0.99"],
    "Momentum_CNN"
)

print(" All experiments for Task 2 completed. Results saved.")
