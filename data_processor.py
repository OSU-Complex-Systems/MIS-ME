import config
from torch.utils.data import DataLoader, Subset
from utils import VWC_Dataset
from sklearn.preprocessing import StandardScaler

# Print loading message
print("\n Loading datasets... \n")


# Create and fit a scaler with the training data and compute mean values
train_scaler = StandardScaler()
train_mean_values = {
    col: config.train_labels_df[col].mean() for col in config.KEEP_COLUMNS
}
config.train_labels_df[config.KEEP_COLUMNS] = config.train_labels_df[
    config.KEEP_COLUMNS
].fillna(train_mean_values)
config.train_labels_df[config.KEEP_COLUMNS] = train_scaler.fit_transform(
    config.train_labels_df[config.KEEP_COLUMNS]
)

# Create dataset instances
train_dataset = VWC_Dataset(
    labels_df=config.train_labels_df,
    image_folder=config.all_data_dir,
    keep_columns=config.KEEP_COLUMNS,
    scaler=None,  # Scaler already applied in the DataFrame
    mean_values=None,  # Mean values already applied in the DataFrame
)

val_dataset = VWC_Dataset(
    labels_df=config.val_labels_df,
    image_folder=config.all_data_dir,
    keep_columns=config.KEEP_COLUMNS,
    scaler=train_scaler,  # Use the scaler fitted with training data
    mean_values=train_mean_values,  # Use mean values from training data
)

test_dataset = VWC_Dataset(
    labels_df=config.test_labels_df,
    image_folder=config.all_data_dir,
    keep_columns=config.KEEP_COLUMNS,
    scaler=train_scaler,  # Use the scaler fitted with training data
    mean_values=train_mean_values,  # Use mean values from training data
)

ok001_test_dataset = VWC_Dataset(
    labels_df=config.ok001_test_labels_df,
    image_folder=config.all_data_dir,
    keep_columns=config.KEEP_COLUMNS,
    scaler=train_scaler,  # Use the scaler fitted with training data
    mean_values=train_mean_values,  # Use mean values from training data
)

ok002_test_dataset = VWC_Dataset(
    labels_df=config.ok002_test_labels_df,
    image_folder=config.all_data_dir,
    keep_columns=config.KEEP_COLUMNS,
    scaler=train_scaler,  # Use the scaler fitted with training data
    mean_values=train_mean_values,  # Use mean values from training data
)

ok003_test_dataset = VWC_Dataset(
    labels_df=config.ok003_test_labels_df,
    image_folder=config.all_data_dir,
    keep_columns=config.KEEP_COLUMNS,
    scaler=train_scaler,  # Use the scaler fitted with training data
    mean_values=train_mean_values,  # Use mean values from training data
)


## Debugging: Use a small dataset for testing
if config.DEBUG:
    print("\n WARNING: Testing on a small dataset \n")
    train_dataset = Subset(train_dataset, list(range(100)))
    val_dataset = Subset(val_dataset, list(range(100)))
    test_dataset = Subset(test_dataset, list(range(20)))
    ok001_test_dataset = Subset(ok001_test_dataset, list(range(20)))
    ok002_test_dataset = Subset(ok002_test_dataset, list(range(20)))
    ok003_test_dataset = Subset(ok003_test_dataset, list(range(20)))


# Create DataLoaders for training and validation datasets
train_dataloader = DataLoader(
    train_dataset,
    batch_size=config.BATCH_SIZE,
    shuffle=config.SHUFFLE,
    num_workers=8,
    pin_memory=True,
)
val_dataloader = DataLoader(
    val_dataset,
    batch_size=config.BATCH_SIZE,
    shuffle=False,
    num_workers=8,
    pin_memory=True,
)

# Create DataLoader for the test dataset
test_dataloader = DataLoader(
    test_dataset,
    batch_size=config.BATCH_SIZE,
    shuffle=False,
    num_workers=8,
    pin_memory=True,
)
ok001_test_dataloader = DataLoader(
    ok001_test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, pin_memory=True
)
# ok002_test_dataloader = DataLoader(ok002_test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
ok002_test_dataloader = DataLoader(
    ok002_test_dataset, batch_size=65, shuffle=False, pin_memory=True
)  ## manually set batch size to prevent batch norm error
ok003_test_dataloader = DataLoader(
    ok003_test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, pin_memory=True
)


# Print Input Shape and Labels Shape for training, validation, and test sets
print(
    f"Training Input Shape: {next(iter(train_dataloader))['image'].shape} \t Training Labels Shape: {next(iter(train_dataloader))['vwc'][0]}"
)
print(
    f"Validation Input Shape: {next(iter(val_dataloader))['image'].shape} \t Validation Labels Shape: {next(iter(val_dataloader))['vwc'][0]}"
)
print(
    f"Test Input Shape: {next(iter(test_dataloader))['image'].shape} \t Test Labels Shape: {next(iter(test_dataloader))['vwc'][0]}"
)


# Print information about the dataset
print(f"Number of training samples: {len(train_dataset)}")
print(f"Number of validation samples: {len(val_dataset)}")
print(f"Number of test samples: {len(test_dataset)}")


# Write the dataset sizes to a text file
with open("dataset_size.txt", "w") as f:
    f.write(f"Number of training samples: {len(train_dataset)} \n")
    f.write(f"Number of validation samples: {len(val_dataset)} \n")
    f.write(f"Number of test samples: {len(test_dataset)} \n")
    f.write(f"Number of OK001 test samples: {len(ok001_test_dataset)} \n")
    f.write(f"Number of OK002 test samples: {len(ok002_test_dataset)} \n")
    f.write(f"Number of OK003 test samples: {len(ok003_test_dataset)} \n")
