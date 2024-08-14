import os

import config
import torch
import torch.nn as nn
import torch.optim as optim
from data_processor import (
    ok001_test_dataloader,
    ok002_test_dataloader,
    ok003_test_dataloader,
    test_dataloader,
    train_dataloader,
    val_dataloader,
)
from eval import eval_model
from plot import plot_loss
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from tqdm import tqdm
from utils import get_model


def train_model(model_name: str, save_as: str):
    # Set the device to GPU if available
    device = config.device

    # Create directories for saving weights and plots
    os.makedirs(f"saved_weights/{config.experiment}", exist_ok=True)
    os.makedirs(f"plots/{config.experiment}", exist_ok=True)

    # Initialize the model, loss function, and optimizer
    model = get_model(model_name, device)

    if (
        config.experiment
        in [
            "concat_with_diff_dim",
            "add_with_same_dim",
            "multiply_with_same_dim",
        ]
    ) or ("learnable_parameter" in config.experiment):
        criterion = nn.MSELoss()
    elif "hybrid_loss" in config.experiment:
        meteo_criterion = nn.MSELoss()
        image_criterion = nn.MSELoss()
        criterion = nn.MSELoss()

    optimizer = optim.Adam(
        model.parameters(), lr=0.001
    )  # this lr value is used for the first epoch only

    # Define scheduler and warmup parameters
    base_lr = config.BASE_LR
    # total_warmup_epochs=1 means no warmup
    # total_warmup_epochs=2 means from 2nd epoch base_lr will be used and first epoch will be warmup
    # total_warmup_epochs=3 means from 3rd epoch base_lr will be used and first 2 epochs will be warmup
    total_warmup_epochs = config.TOTAL_WARMUP_EPOCHS
    # schedulers will come into effect for epochs after total_warmup_epochs
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2, verbose=True
    )  # factor=0.5 means lr will be reduced by half after 2 epochs (patience=2) if validation loss doesn't improve
    # scheduler = StepLR(optimizer, step_size=1, gamma=0.1)

    # Function to adjust learning rate for warmup
    def lr_warmup(optimizer, epoch, base_lr, total_warmup_epochs):
        warmup_lr = min(((epoch + 1) / total_warmup_epochs) * base_lr, base_lr)
        for param_group in optimizer.param_groups:
            param_group["lr"] = warmup_lr

    # Pretty-print the model architecture
    # print(model)

    # Number of epochs for training
    num_epochs = config.EPOCHS

    # Initialize variables for tracking the best model
    best_loss = float("inf")
    best_model_state_dict = None

    # Lists to store loss values for plotting
    train_losses = []
    val_losses = []

    # Initialize CSV file for learnable parameters if applicable
    if "learnable_parameter" in config.experiment:
        if config.experiment == "one_learnable_parameter":
            csv_headers = "epoch,alpha"
        if config.experiment == "two_learnable_parameters":
            csv_headers = "epoch,alpha,beta"
        with open(
            f"plots/{config.experiment}/{save_as}_{config.experiment}.csv", "w"
        ) as f:
            f.write(csv_headers + "\n")

    print("Training is beginning...")

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_train_loss = 0.0

        for batch in tqdm(train_dataloader, desc=f"Epoch [{epoch + 1}/{num_epochs}]"):
            image_data = batch["image"].to(device)
            meteorological_data = batch["meteorological_data"].to(device)
            labels = batch["vwc"].to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            if (
                config.experiment
                in [
                    "concat_with_diff_dim",
                    "add_with_same_dim",
                    "multiply_with_same_dim",
                ]
            ) or ("learnable_parameter" in config.experiment):
                _, _, combined_output = model(image_data, meteorological_data)
                total_train_loss = criterion(combined_output, labels.unsqueeze(1))

            elif "hybrid_loss" in config.experiment:
                meteo_output, image_output, combined_output = model(
                    image_data, meteorological_data
                )

                meteo_loss = meteo_criterion(meteo_output, labels.unsqueeze(1))
                image_loss = image_criterion(image_output, labels.unsqueeze(1))
                concatenated_criterion = criterion(combined_output, labels.unsqueeze(1))
                total_train_loss = (
                    config.DELTA * concatenated_criterion
                    + config.GAMMA * meteo_loss
                    + config.LAMBDA * image_loss
                )

            else:
                print(
                    f"Experiment: {config.experiment} is not implemented yet. Exiting..."
                )
                exit()

            # Backward pass and optimize
            total_train_loss.backward()
            optimizer.step()

            # Accumulate loss
            running_train_loss += total_train_loss.item()

        # Calculate average training loss for the epoch
        train_loss_avg = running_train_loss / len(train_dataloader)
        train_losses.append(train_loss_avg)
        print(
            f"Epoch [{epoch + 1}/{num_epochs}] Train Loss: {train_loss_avg:.4f} LR: {optimizer.param_groups[0]['lr']}"
        )

        ## Print the alpha and beta learnable parameterand save in csv file for each epoch
        if "learnable_parameter" in config.experiment:
            if config.experiment == "one_learnable_parameter":
                with open(
                    f"plots/{config.experiment}/{save_as}_one_learnable_parameter.csv",
                    "a",
                ) as f:
                    print(f"Alpha: {model.alpha.item()}")
                    f.write(f"{epoch + 1},{model.alpha.item()}\n")
            elif config.experiment == "two_learnable_parameters":
                with open(
                    f"plots/{config.experiment}/{save_as}_two_learnable_parameters.csv",
                    "a",
                ) as f:
                    print(f"Alpha: {model.alpha.item()} Beta: {model.beta.item()}")
                    f.write(f"{epoch + 1},{model.alpha.item()},{model.beta.item()}\n")

        # Validation loop
        model.eval()
        running_val_loss = 0.0

        with torch.no_grad():
            for batch in tqdm(
                val_dataloader, desc=f"Validation [{epoch + 1}/{num_epochs}]"
            ):
                image_data = batch["image"].to(device)
                meteorological_data = batch["meteorological_data"].to(device)
                labels = batch["vwc"].to(device)

                _, _, combined_output = model(image_data, meteorological_data)
                total_val_loss = criterion(combined_output, labels.unsqueeze(1))

                running_val_loss += total_val_loss.item()

        # Calculate average validation loss for the epoch
        val_loss_avg = running_val_loss / len(val_dataloader)
        val_losses.append(val_loss_avg)

        print(f"Epoch [{epoch + 1}/{num_epochs}] Validation Loss: {val_loss_avg:.4f}")

        # Update best model if validation loss improved
        if val_loss_avg <= best_loss:
            best_loss = val_loss_avg
            best_model_state_dict = model.state_dict()

        # Apply warmup for the initial epochs, then switch to scheduler
        if epoch < total_warmup_epochs:
            lr_warmup(optimizer, epoch, base_lr, total_warmup_epochs)
        else:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(
                    val_loss_avg
                )  ## Use this for ReduceLROnPlateau scheduler only
            else:
                scheduler.step()  ## Use this for all LR Scehdulers EXCEPT ReduceLROnPlateau

    print("Training is complete.")

    # Plot and save loss curves
    plot_loss(train_losses, val_losses, save_as)

    print(f"Loading the best model with best validation loss {best_loss:.4f}")
    model.load_state_dict(best_model_state_dict)
    model.eval()

    # Evaluate the model on train, validation, and test sets
    print("Evaluation is beginning...")
    eval_model(model, train_dataloader, "train", save_as)
    eval_model(model, val_dataloader, "val", save_as)
    eval_model(model, test_dataloader, "test", save_as)
    eval_model(model, ok001_test_dataloader, "ok001_test", save_as)
    eval_model(model, ok002_test_dataloader, "ok002_test", save_as)
    eval_model(model, ok003_test_dataloader, "ok003_test", save_as)
    print("Evaluation is complete.")

    # Save the best model
    print(
        f"Saving the best model to ./saved_weights/{config.experiment}/{save_as}_best.pth with loss {best_loss:.4f}"
    )
    torch.save(
        best_model_state_dict, f"./saved_weights/{config.experiment}/{save_as}_best.pth"
    )

    return
