from math import e
from matplotlib.pylab import f
import config
import torch
import torch.nn as nn
from torchvision import models


class CombinedModel(nn.Module):
    def __init__(self, meteorological_feature_size, image_feature_extractor):
        """
        Initialize the CombinedModel.

        Args:
            meteorological_feature_size (int): Number of features in the meteorological data.
            image_feature_extractor (str): Name of the image feature extractor model.
        """
        super(CombinedModel, self).__init__()

        self.image_feature_extractor_name = image_feature_extractor

        # Dictionary mapping extractor names to model initialization functions
        feature_extractors = {
            "resnet18": lambda: models.resnet18(
                weights=models.ResNet18_Weights.DEFAULT
            ),
            "vgg16": lambda: models.vgg16_bn(weights=models.VGG16_BN_Weights.DEFAULT),
            "mobilenetv2": lambda: models.mobilenet_v2(
                weights=models.MobileNet_V2_Weights.DEFAULT
            ),
            "inceptionv3": lambda: models.inception_v3(
                weights=models.Inception_V3_Weights.DEFAULT
            ),
            "efficientnetv2": lambda: models.efficientnet_v2_m(
                weights=models.EfficientNet_V2_M_Weights.DEFAULT
            ),
            "densenet121": lambda: models.densenet121(
                weights=models.DenseNet121_Weights.DEFAULT
            ),
        }

        # Initialize the specified image feature extractor
        if self.image_feature_extractor_name in feature_extractors:
            self.image_extractor = feature_extractors[
                self.image_feature_extractor_name
            ]()
        else:
            raise ValueError(
                f"Unknown image feature extractor: {self.image_feature_extractor_name}"
            )

        # Add Global Average Pooling for each extractor
        if self.image_feature_extractor_name in [
            "resnet18",
            "mobilenetv2",
            "efficientnetv2",
            "densenet121",
        ]:
            # These models end with features that can be pooled directly.
            self.image_extractor.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.image_extractor.classifier = (
                nn.Identity()
            )  # Remove the last classifier layer

        elif self.image_feature_extractor_name == "vgg16":
            # VGG models have 'features' and 'classifier' attributes; adapt accordingly.
            features = self.image_extractor.features
            # Introduce Global Average Pooling after the last convolutional layer
            gap = nn.AdaptiveAvgPool2d((1, 1))
            # Create a sequential model that includes the features, GAP, and flattening. This sequence will transform the 4D tensor from the features to a 2D tensor
            self.image_extractor = nn.Sequential(features, gap, nn.Flatten())

        elif self.image_feature_extractor_name == "inceptionv3":
            # InceptionV3 requires special handling due to its auxiliary output.
            self.image_extractor.aux_logits = False
            self.image_extractor.AvgPool = nn.AdaptiveAvgPool2d((1, 1))
            self.image_extractor.fc = nn.Identity()

        # Determine the output feature size dynamically
        with torch.no_grad():
            if self.image_feature_extractor_name == "inceptionv3":
                # For InceptionV3, use a larger dummy batch and handle auxiliary output
                # Forward through InceptionV3 and handle auxiliary output
                dummy_input = torch.zeros(2, 3, config.IMG_HEIGHT, config.IMG_WIDTH)
                _, aux = self.image_extractor(dummy_input)
                image_feature_size = aux.view(-1).shape[0]
            else:
                dummy_input = torch.zeros(1, 3, config.IMG_HEIGHT, config.IMG_WIDTH)
                image_feature_size = self.image_extractor(dummy_input).view(-1).shape[0]

            print("Image Feature Size:", image_feature_size)

        # Define the meteorological data feature extractor
        self.meteorological_feature_extractor = nn.Sequential(
            nn.Linear(meteorological_feature_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
        )

        if "learnable_parameter" in config.experiment:
            ## initialize with 1
            self.alpha = nn.Parameter(torch.tensor(1.0), requires_grad=True)
            if config.experiment == "two_learnable_parameters":
                ## initialize with 1
                self.beta = nn.Parameter(torch.tensor(1.0), requires_grad=True)

        if config.experiment in ["concat_with_diff_dim", "hybrid_loss"]:
            # Batch Normalization after concatenating the features
            self.concat_batch_norm = nn.BatchNorm1d(image_feature_size + 16)

            # Define the combined predictor
            self.combined_predictor = nn.Sequential(
                nn.Linear(
                    image_feature_size + 16, 256
                ),  ## +16 is the number of output features from the meteorological_feature_extractor
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(128, 1),
            )

        elif config.experiment in [
            "add_with_same_dim",
            "multiply_with_same_dim",
        ]:
            # input_neurons = 8+8 if config.experiment == "concat_with_same_dim" else 8   ## 8 is the number of output features from the meteorological_feature_extractor
            input_neurons = 16

            # Batch Normalization after concatenating the features
            self.concat_batch_norm = nn.BatchNorm1d(input_neurons)

            # Define the combined predictor
            self.combined_predictor = nn.Sequential(
                nn.Linear(input_neurons, 8),
                nn.BatchNorm1d(8),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(8, 4),
                nn.BatchNorm1d(4),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(4, 1),
            )

        if (config.experiment == "hybrid_loss") or (
            "learnable_parameter" in config.experiment
        ):
            # Final prediction layer for only meteorological data
            self.meteo_predictor = nn.Linear(16, 1)

            # Final prediction layer for only image data
            self.image_predictor = nn.Sequential(
                nn.Linear(image_feature_size, image_feature_size // 2),
                nn.BatchNorm1d(image_feature_size // 2),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(image_feature_size // 2, image_feature_size // 4),
                nn.BatchNorm1d(image_feature_size // 4),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(image_feature_size // 4, 1),
            )

    def adjust_dimension(self, input_dim, target_dim):
        intermediate_dim = (
            input_dim + target_dim
        ) // 2  # Calculate an intermediate dimension
        return nn.Sequential(
            nn.Linear(input_dim, intermediate_dim),
            nn.BatchNorm1d(intermediate_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(intermediate_dim, target_dim),
            nn.BatchNorm1d(target_dim),
            nn.ReLU(),
        )

    def forward(self, image, meteorological_data):
        """
        Forward pass of the model.

        Args:
            image (Tensor): Input image tensor.
            meteorological_data (Tensor): Input meteorological data tensor.

        Returns:
            Tensor: Output prediction.
        """

        if config.experiment == "one_learnable_parameter":
            # Extract image features
            image_features = self.image_extractor(image)
            image_features = torch.flatten(image_features, 1) * (1 - self.alpha)

            # Extract features from meteorological data
            meteorological_features = (
                self.meteorological_feature_extractor(meteorological_data) * self.alpha
            )

        elif config.experiment == "two_learnable_parameters":
            # Extract image features
            image_features = self.image_extractor(image)
            image_features = torch.flatten(image_features, 1) * self.beta

            # Extract features from meteorological data
            meteorological_features = self.meteorological_feature_extractor(
                meteorological_data
            ) * (self.alpha)

        else:
            # Extract image features
            image_features = self.image_extractor(image)
            image_features = torch.flatten(image_features, 1)

            # Extract features from meteorological data
            meteorological_features = self.meteorological_feature_extractor(
                meteorological_data
            )

        if config.experiment in ["concat_with_diff_dim", "hybrid_loss"]:
            # Concatenate the features from both extractors
            combined_features = torch.cat(
                (image_features, meteorological_features), dim=1
            )

            # Apply Batch Normalization to the concatenated features
            combined_features = self.concat_batch_norm(combined_features)

            # Pass the combined features through the predictor
            combined_vwc_prediction = self.combined_predictor(combined_features)

            if config.experiment == "hybrid_loss":
                meteo_vwc_prediction = self.meteo_predictor(meteorological_features)
                image_vwc_prediction = self.image_predictor(image_features)
                return (
                    meteo_vwc_prediction,
                    image_vwc_prediction,
                    combined_vwc_prediction,
                )
            else:
                return -1, -1, combined_vwc_prediction

        elif config.experiment in ["add_with_same_dim", "multiply_with_same_dim"]:
            # Adjust the dimension of image features to match meteorological features
            image_features_adjusted = self.adjust_dimension(
                image_features.shape[1], 16
            ).to(image_features.device)(image_features)

            if config.experiment == "add_with_same_dim":
                # Add the features from both extractors
                combined_features = image_features_adjusted + meteorological_features
            elif config.experiment == "multiply_with_same_dim":
                # Multiply the features from both extractors
                combined_features = image_features_adjusted * meteorological_features

            # Apply Batch Normalization to the concatenated features
            combined_features = self.concat_batch_norm(combined_features)

            # Pass the combined features through the predictor
            combined_vwc_prediction = self.combined_predictor(combined_features)

            return -1, -1, combined_vwc_prediction

        elif "learnable_parameter" in config.experiment:
            meteo_vwc_prediction = self.meteo_predictor(meteorological_features)
            image_vwc_prediction = self.image_predictor(image_features)

            if config.experiment == "one_learnable_parameter":
                combined_vwc_prediction = (
                    self.alpha * meteo_vwc_prediction
                    + (1 - self.alpha) * image_vwc_prediction
                )

                return -1, -1, combined_vwc_prediction

            elif config.experiment == "two_learnable_parameters":
                combined_vwc_prediction = (
                    self.alpha * meteo_vwc_prediction + self.beta * image_vwc_prediction
                )

                return -1, -1, combined_vwc_prediction

            else:
                return meteo_vwc_prediction, image_vwc_prediction, -1

        else:
            print(f"Experiment: {config.experiment} is not recognized. Exiting...")
            exit()


if __name__ == "__main__":
    # Instantiate CombinedModel for each image feature extractor
    extractors = [
        "resnet18",
        "vgg16",
        "mobilenetv2",
        "inceptionv3",
        "efficientnetv2",
        "densenet121",
    ]  ## image feature extractor flatten layer size: 1000, 512, 1280, 2048, 1280, 1024

    # Instantiate CombinedModel for one of the extractors (example: "resnet18")
    model_example = CombinedModel(config.meteorological_feature_size, extractors[1])

    # Create a dummy input to pass through the model
    dummy_image = torch.zeros(2, 3, 224, 224)
    # dummy_image = torch.zeros(2, 3, 299, 299)
    dummy_meteorological = torch.zeros(2, config.meteorological_feature_size)

    # Get the model output
    output = model_example(dummy_image, dummy_meteorological)
