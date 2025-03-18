import torch 
import torch.nn.init as init
import matplotlib.pyplot as plt

def initialize_weights(model, strategy="balanced_variance"):
    for layer in model.modules():
        if isinstance(layer, (torch.nn.Linear, torch.nn.Conv2d)):
            if strategy == "balanced_variance":
                init.xavier_uniform_(layer.weight)
            elif strategy == "activation_specific":
                init.kaiming_uniform_(layer.weight, nonlinearity='relu')
            elif strategy == "random_baseline":
                init.uniform_(layer.weight, -0.5, 0.5)  # Bad init
            if layer.bias is not None:
                layer.bias.data.fill_(0.01)

def plot_losses(loss_histories, labels, title, save_path="loss_plot.png"):
    plt.figure(figsize=(10, 5))
    for i, loss_history in enumerate(loss_histories):
        plt.plot(loss_history, label=labels[i])
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.savefig("loss_plot.png")
    print(f"Plot saved as {save_path}")
