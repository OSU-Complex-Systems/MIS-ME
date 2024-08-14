import os

import config
import model
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from typing_extensions import Literal
from sklearn.preprocessing import StandardScaler

MODEL_NAMES = Literal[
    "resnet18", "densenet121", "vgg16", "inceptionv3", "mobilenetv2", "efficientnetv2"
]


def get_model(
    image_feature_extractor: MODEL_NAMES,
    device: torch.device,
):
    if image_feature_extractor == "resnet18":
        return model.CombinedModel(config.meteorological_feature_size, "resnet18").to(
            device
        )
    elif image_feature_extractor == "densenet121":
        return model.CombinedModel(
            config.meteorological_feature_size, "densenet121"
        ).to(device)
    elif image_feature_extractor == "vgg16":
        return model.CombinedModel(config.meteorological_feature_size, "vgg16").to(
            device
        )
    elif image_feature_extractor == "inceptionv3":
        return model.CombinedModel(
            config.meteorological_feature_size, "inceptionv3"
        ).to(device)
    elif image_feature_extractor == "mobilenetv2":
        return model.CombinedModel(
            config.meteorological_feature_size, "mobilenetv2"
        ).to(device)
    elif image_feature_extractor == "efficientnetv2":
        return model.CombinedModel(
            config.meteorological_feature_size, "efficientnetv2"
        ).to(device)
    else:
        raise ValueError("Invalid model name")


class VWC_Dataset(Dataset):
    def __init__(
        self,
        labels_df,
        image_folder,
        keep_columns,
        transform=None,
        scaler=None,
        mean_values=None,
    ):
        """
        Args:
            labels_df (DataFrame): Pandas DataFrame containing labels and metadata.
            image_folder (string): Path to the image directory.
            keep_columns (list): List of column names to keep for meteorological data.
            transform (callable, optional): Optional transform to be applied on the image.
            scaler (StandardScaler, optional): Scaler for normalizing meteorological data.
            mean_values (dict, optional): Dictionary of mean values for each column.
        """
        self.labels_df = labels_df.copy()
        self.image_folder = image_folder
        self.keep_columns = keep_columns
        self.transform = transform or transforms.Compose(
            [
                transforms.Resize((config.IMG_HEIGHT, config.IMG_WIDTH)),
                transforms.ToTensor(),
                transforms.Normalize(mean=config.IMG_MEAN, std=config.IMG_STD),
            ]
        )

        # Normalize meteorological data
        if scaler is not None and mean_values is not None:
            # Use provided mean values to fill NaNs
            for col in keep_columns:
                self.labels_df[col].fillna(mean_values[col], inplace=True)

            # Use the provided scaler to transform the data
            self.labels_df[keep_columns] = scaler.transform(
                self.labels_df[keep_columns]
            )
        else:
            # Compute mean values, fill NaNs, and fit scaler
            mean_values = {col: self.labels_df[col].mean() for col in keep_columns}
            self.labels_df[keep_columns] = self.labels_df[keep_columns].fillna(
                mean_values
            )
            scaler = StandardScaler()
            self.labels_df[keep_columns] = scaler.fit_transform(
                self.labels_df[keep_columns]
            )

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        img_name = os.path.join(
            self.image_folder, self.labels_df.iloc[idx][config.image_column]
        )
        image = Image.open(img_name).convert("RGB")
        image = self.transform(image)

        # convert meteorological data dtype from object to float
        meteorological_data = self.labels_df.iloc[idx][self.keep_columns]
        meteorological_data = meteorological_data.astype(float)
        meteorological_data = meteorological_data.to_numpy()

        meteorological_data = torch.tensor(meteorological_data, dtype=torch.float32)

        vwc = self.labels_df.iloc[idx][config.label_column]
        vwc = torch.tensor(vwc, dtype=torch.float32)

        image_name = self.labels_df.iloc[idx][config.image_column]

        return {
            "image": image,
            "meteorological_data": meteorological_data,
            "vwc": vwc,
            "image_name": image_name,
        }
