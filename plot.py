import matplotlib.pyplot as plt
import pandas as pd
import config


def plot_loss(train_losses, val_losses, model_name):
    # Plot and save the training and validation loss curves
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Train Loss", linewidth=2)
    plt.plot(val_losses, label="Validation Loss", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title(f"{model_name} Loss Curves")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"plots/{config.experiment}/{model_name}_loss_curves.png")
    plt.show()

    # Save the training and validation loss as CSV files
    loss_data = {"Train Loss": train_losses, "Validation Loss": val_losses}
    df = pd.DataFrame(loss_data)
    df.to_csv(f"plots/{config.experiment}/{model_name}_losses.csv", index=False)
