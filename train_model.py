import config
from train import train_model

if __name__ == "__main__":
    if config.experiment == "hybrid_loss":
        train_model(
            model_name=config.image_feature_extractor,
            save_as=f"{config.experiment}_delta-{config.DELTA}_gamma-{config.GAMMA}_lambda-{config.LAMBDA}_{config.image_feature_extractor}_s{config.random_seed}",
        )
    else:
        train_model(
            model_name=config.image_feature_extractor,
            save_as=f"{config.experiment}_{config.image_feature_extractor}_s{config.random_seed}",
        )
