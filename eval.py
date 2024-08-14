import os

import config
import numpy as np
import pandas as pd
import torch
from data_processor import (
    train_dataloader,
    test_dataloader,
    val_dataloader,
    ok001_test_dataloader,
    ok002_test_dataloader,
    ok003_test_dataloader,
)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tqdm import tqdm
from utils import get_model


def eval_model(model, dataloader, split, model_name, save=True):
    """
    Evaluate the model on a given data split.

    Args:
        model: The neural network model to evaluate.
        dataloader: DataLoader for the dataset to evaluate.
        split (str): The name of the split (e.g., 'train', 'val', 'test').
        model_name (str): Name of the model used for saving metrics.
    """

    device = config.device
    model.to(device)

    # Create directory for storing metrics
    os.makedirs(f"metrics/{config.experiment}", exist_ok=True)

    model.eval()  # Set the model to evaluation mode

    predictions, actuals, image_names = [], [], []

    with tqdm(
        total=len(dataloader), desc=f"Evaluating {split.capitalize()} Split"
    ) as pbar:
        with torch.no_grad():
            for batch in dataloader:
                image_data = batch["image"].to(device)
                meteorological_data = batch["meteorological_data"].to(device)
                labels = batch["vwc"].to(device)

                _, _, combined_output = model(image_data, meteorological_data)
                outputs = combined_output

                predictions.extend(outputs.cpu().numpy().ravel())
                actuals.extend(labels.cpu().numpy().ravel())
                image_names.extend(batch["image_name"])

                pbar.update(1)

    predictions = np.array(predictions)
    actuals = np.array(actuals)

    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    mse = mean_squared_error(actuals, predictions)
    mae = mean_absolute_error(actuals, predictions)
    r2 = r2_score(actuals, predictions)
    mpe = np.mean((actuals - predictions) / actuals) * 100
    mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100

    print_metrics(split, rmse, mse, mae, r2, mpe, mape)

    if save:
        save_metrics(split, model_name, rmse, mse, mae, r2, mpe, mape)
        save_predictions(split, model_name, image_names, actuals, predictions)

    return


def print_metrics(split, rmse, mse, mae, r2, mpe, mape):
    """
    Print evaluation metrics.

    Args:
        split (str): Data split name.
        rmse, mse, mae, r2, mpe, mape: Computed metrics.
    """
    print(f"{split.capitalize()} Split Metrics:")
    print(f"RMSE: {rmse:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R-squared: {r2:.4f}")
    print(f"MPE: {mpe:.4f}%")
    print(f"MAPE: {mape:.4f}%")


def save_metrics(split, model_name, rmse, mse, mae, r2, mpe, mape):
    """
    Save the evaluation metrics to a CSV file.

    Args:
        split, model_name: Identifiers for saving the file.
        rmse, mse, mae, r2, mpe, mape: Computed metrics.
    """
    # Prepare data for saving
    metrics_data = {
        "Split": [split],
        "RMSE": [rmse],
        "MSE": [mse],
        "MAE": [mae],
        "R-squared": [r2],
        "MPE": [mpe],
        "MAPE": [mape],
    }
    metrics_df = pd.DataFrame(metrics_data)
    metrics_filename = f"metrics/{config.experiment}/{model_name}_{split}_metrics.csv"

    # Append to existing file or create a new one
    if os.path.isfile(metrics_filename):
        metrics_df.to_csv(metrics_filename, index=False)
    else:
        metrics_df.to_csv(metrics_filename, index=False)


def save_predictions(split, model_name, image_names, actuals, predictions):
    """
    Save the actual and predicted values to a CSV file.

    Args:
        split, model_name: Identifiers for saving the file.
        image_names: List of image names.
        actuals, predictions: Actual and predicted values.
    """
    actuals_predictions_data = {
        "Image_Name": image_names,
        "Actual": actuals,
        "Predicted": predictions,
    }
    actuals_predictions_df = pd.DataFrame(actuals_predictions_data)
    actuals_predictions_filename = (
        f"metrics/{config.experiment}/{model_name}_{split}_actuals_predicted.csv"
    )
    actuals_predictions_df.to_csv(actuals_predictions_filename, index=False)


# Main script to evaluate the model
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n Evaluating on {device} \n")

    model_name = config.image_feature_extractor
    model = get_model(model_name, device=device)

    if config.experiment == "hybrid_loss":
        save_as = f"{config.experiment}_delta-{config.DELTA}_gamma-{config.GAMMA}_lambda-{config.LAMBDA}_{config.image_feature_extractor}_s{config.random_seed}"
    else:
        save_as = f"{config.experiment}_{config.image_feature_extractor}_s{config.random_seed}"

    checkpoint = torch.load(f"./saved_weights/{config.experiment}/{save_as}_best.pth")
    model.load_state_dict(checkpoint)
    model.eval()

    eval_model(model, train_dataloader, "train", save_as, save=True)
    eval_model(model, val_dataloader, "val", save_as, save=True)
    eval_model(model, test_dataloader, "test", save_as, save=True)
    eval_model(model, ok001_test_dataloader, "ok001_test", save_as, save=True)
    eval_model(model, ok002_test_dataloader, "ok002_test", save_as, save=True)
    eval_model(model, ok003_test_dataloader, "ok003_test", save_as, save=True)
