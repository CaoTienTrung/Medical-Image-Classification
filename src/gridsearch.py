import numpy as np
import sys
import os
import argparse
from itertools import product
import joblib
from tqdm import tqdm
import warnings
import yaml

warnings.filterwarnings("ignore")

# Set environment variables for better performance
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.custom_dataset import CustomImageDataset


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generic Grid Search for Feature Extractors and Models"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file",
    )
    return parser.parse_args()


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def load_dataset(data_dir, image_size, color_mode):
    print("Loading datasets...")
    train_data = CustomImageDataset(
        directory=os.path.join(data_dir, "train"),
        label_mode="int",
        color_mode=color_mode,
        image_size=image_size,
        interpolation="bilinear",
    )
    test_data = CustomImageDataset(
        directory=os.path.join(data_dir, "test"),
        label_mode="int",
        color_mode=color_mode,
        image_size=image_size,
        interpolation="bilinear",
    )
    return train_data, test_data


def prepare_data(dataset):
    X, y = [], []
    for img, label in tqdm(dataset, desc="Processing images"):
        img_np = img.numpy().transpose(1, 2, 0)
        X.append(img_np)
        y.append(label)
    return np.array(X), np.array(y)


def main():
    args = parse_args()
    config = load_config(args.config)

    model_module = __import__("src.models.models", fromlist=[config["model"]["name"]])
    fe_module = __import__(
        "src.FeatureExtractors.feature_extractor",
        fromlist=[config["feature_extractor"]["name"]],
    )

    ModelClass = getattr(model_module, config["model"]["name"])
    FeatureExtractorClass = getattr(fe_module, config["feature_extractor"]["name"])

    # Get parameter grids from config
    model_params = config["model"]["params"]
    fe_params = {}  # Empty dict for feature extractor to use defaults

    # Load datasets
    train_data, test_data = load_dataset(
        config["dataset"]["data_dir"],
        tuple(config["dataset"]["image_size"]),
        config["dataset"]["color_mode"],
    )

    # Generate all parameter combinations (only for model parameters now)
    param_combinations = []
    for model_vals in product(*model_params.values()):
        model_dict = dict(zip(model_params.keys(), model_vals))
        param_combinations.append((fe_params, model_dict))

    print(f"\nTotal parameter combinations to try: {len(param_combinations)}")

    best_score = 0
    best_params = None
    best_model = None

    # Create output directory if it doesn't exist
    os.makedirs(config["output"]["dir"], exist_ok=True)

    # Manual grid search
    for i, (fe_dict, model_dict) in enumerate(
        tqdm(param_combinations, desc="Grid Search Progress")
    ):
        # Create feature extractor with default parameters
        feature_extractor = FeatureExtractorClass()
        # No need to update feature extractor params since we're using defaults

        # Create model with current parameters
        model = ModelClass(feature_extractor=feature_extractor, **model_dict)

        # Train and evaluate
        model.train(train_data)
        y_pred, metrics = model.predict(test_data)

        # Update best parameters if current model is better
        if metrics["accuracy"] > best_score:
            best_score = metrics["accuracy"]
            best_params = {**model_dict}
            best_model = model

        print(f"\nCombination {i+1}/{len(param_combinations)}")
        print(f"Parameters: {model_dict}")
        print(f"Accuracy: {metrics['accuracy']:.4f}")

    print("\nBest parameters found:")
    print(best_params)
    print("\nBest accuracy:", best_score)

    # Save results
    print("\nSaving results...")
    results_file = os.path.join(
        config["output"]["dir"],
        f"best_{config['feature_extractor']['name']}_{config['model']['name']}_params.txt",
    )
    with open(results_file, "w") as f:
        for param, value in best_params.items():
            f.write(f"{param}: {value}\n")
    print(f"Best parameters saved to '{results_file}'")

    # Save best model
    model_file = os.path.join(
        config["output"]["dir"],
        f"best_{config['feature_extractor']['name']}_{config['model']['name']}_model.joblib",
    )
    joblib.dump(best_model, model_file)
    print(f"Best model saved as '{model_file}'")


if __name__ == "__main__":
    main()
