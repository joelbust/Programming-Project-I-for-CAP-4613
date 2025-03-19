import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os

# Ensure the 'results' directory exists
os.makedirs("results", exist_ok=True)

def initialize_weights(model, init_type="balanced_variance"):
    """
    Initializes model weights using the specified initialization strategy.

    :param model: Neural network model whose weights need to be initialized.
    :param init_type: The type of initialization strategy ('random_baseline', 'balanced_variance', or 'activation_specific').
    """
    for layer in model.modules():
        if isinstance(layer, (nn.Linear, nn.Conv2d)):
            if init_type == "random_baseline":
                nn.init.uniform_(layer.weight, -0.5, 0.5)
            elif init_type == "balanced_variance":
                nn.init.xavier_uniform_(layer.weight)
            elif init_type == "activation_specific":
                if isinstance(layer, nn.Conv2d):
                    nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
                else:
                    nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

def plot_losses(loss_list, labels, title):
    """
    Plots and saves loss curves with unique filenames.

    :param loss_list: List of loss values from different training experiments.
    :param labels: Corresponding labels for each loss curve.
    :param title: Title of the plot.
    """
    plt.figure(figsize=(8, 6))

    for loss, label in zip(loss_list, labels):
        plt.plot(loss, label=label)

    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()

    # Generate a unique filename based on the experiment type
    filename = title.replace(" ", "_").lower() + ".png"
    filepath = f"results/{filename}"

    plt.savefig(filepath)
    print(f"Plot saved as {filepath}")
    plt.close()
