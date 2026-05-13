"""Run the end-to-end gas monitoring activity-level pipeline."""

from src.data_preparation import load_config, load_data, preprocess_data
from src.model_training import main as train_models


def main() -> None:
    config = load_config()
    raw_df = load_data(config)
    preprocess_data(raw_df, config)
    train_models()


if __name__ == "__main__":
    main()
