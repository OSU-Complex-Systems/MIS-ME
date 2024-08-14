import random

import numpy as np
import pandas as pd
import torch


DEBUG = True  # Set to True to train and test on a small dataset
BATCH_SIZE = 64  # Modify as per your requirement
EPOCHS = 40
BASE_LR = 0.05 # Base learning rate
TOTAL_WARMUP_EPOCHS = 5  # Total number of warmup epochs
SHUFFLE = True  # Shuffle for training set
if DEBUG:
    EPOCHS = 2

## Experiment name
## 3 MIS-ME Approaches:
# experiment = "concat_with_diff_dim" #MIS-ME Approach 1
# experiment = "hybrid_loss"  # MIS-ME  Approach 2
# experiment = "two_learnable_parameters" # MIS-ME  Approach 3

## Ablation Study
# experiment = "add_with_same_dim"
# experiment = "multiply_with_same_dim"
experiment = "one_learnable_parameter"  #Ablation with one learnable parameter

if experiment == "hybrid_loss":
    DELTA = 1.0
    GAMMA = 1.0
    LAMBDA = 1.0


## Set Image Feature Extractor
# image_feature_extractor = "resnet18"
# image_feature_extractor = "inceptionv3"
image_feature_extractor = "mobilenetv2"
# image_feature_extractor = "efficientnetv2"
# ---------------------------------------------------------------------------------------------# (Not testing the two below)
# image_feature_extractor = "densenet121"
# image_feature_extractor = "vgg16"


# Set Image Width and Height
if image_feature_extractor == "inceptionv3":
    IMG_HEIGHT = 299
    IMG_WIDTH = 299
else:
    IMG_HEIGHT = 224
    IMG_WIDTH = 224


# Set random seeds for reproducibility. Tried 3 different seeds and reported best results in the paper
random_seed = 0
# random_seed = 24
# random_seed = 42
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# Specify the image folder
all_data_dir = (
    "/home/rakib/USDA-Soil/usda_dataset/Dataset_Conf_0.5/all_three_stations/images"
)

# Load the labels DataFrame from the generated Excel file
# TRAIN SET
train_labels = "/home/rakib/USDA-Soil/usda_dataset/labels/classification/combined_only_train_set.xlsx"
train_labels_df = pd.read_excel(train_labels)
# VAL SET
val_labels = "/home/rakib/USDA-Soil/usda_dataset/labels/classification/combined_only_val_set.xlsx"
val_labels_df = pd.read_excel(val_labels)
# TEST SET
test_labels = "/home/rakib/USDA-Soil/usda_dataset/labels/classification/combined_only_test_set.xlsx"
test_labels_df = pd.read_excel(test_labels)
# Station-wise test set
ok001_test_labels = (
    "/home/rakib/USDA-Soil/usda_dataset/labels/classification/OK001_only_test_set.xlsx"
)
ok002_test_labels = (
    "/home/rakib/USDA-Soil/usda_dataset/labels/classification/OK002_only_test_set.xlsx"
)
ok003_test_labels = (
    "/home/rakib/USDA-Soil/usda_dataset/labels/classification/OK003_only_test_set.xlsx"
)
ok001_test_labels_df = pd.read_excel(ok001_test_labels)
ok002_test_labels_df = pd.read_excel(ok002_test_labels)
ok003_test_labels_df = pd.read_excel(ok003_test_labels)


# Specify the column names for images and labels
image_column = "image"
label_column = "vwc"


IMG_MEAN = [0.507, 0.468, 0.307]  # Mean of our dataset
IMG_STD = [
    0.194,
    0.201,
    0.170,
]


# Specify the columns to keep for meteorological data
KEEP_COLUMNS = [
    "air_temperature",
    "relative_humidity",
    "precipitation",
    "solar_flux_density",
    "wind_speed",
    "barometric_pressure",
    "tilt_north_south",
    "tilt_west_east",
]
meteorological_feature_size = len(KEEP_COLUMNS)


# Define device based on CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    # print current GPU name and number
    print(
        f"Training on {torch.cuda.get_device_name(torch.cuda.current_device())} with ID {torch.cuda.current_device()}"
    )
else:
    print("Using CPU")


# Print Training Parameters
print(f"Experiment: {experiment}")
print(f"Image Feature Extractor: {image_feature_extractor}")
print(f"Batch Size: {BATCH_SIZE}")
print(f"Epochs: {EPOCHS}")
print(f"Shuffle: {SHUFFLE}")
print(f"Random Seed: {random_seed}")
print(f"Device: {device}")
print(f"Image Height: {IMG_HEIGHT}")
print(f"Image Width: {IMG_WIDTH}")
