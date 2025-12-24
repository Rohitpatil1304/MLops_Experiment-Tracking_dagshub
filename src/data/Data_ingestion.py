import os
from typing import Tuple

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
import logging


logger = logging.getLogger("data_ingestion")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
console_handler.setFormatter(formatter)

if not logger.handlers:
    logger.addHandler(console_handler)


def load_params(path: str) -> float:
    """Load test_size from params.yaml."""
    try:
        logger.info(f"Loading parameters from {path}")
        with open(path, 'r') as f:
            params = yaml.safe_load(f)

        test_size = float(params['data_ingestion']['test_size'])
        logger.debug(f"Loaded test_size = {test_size}")
        return test_size

    except FileNotFoundError:
        logger.error(f"Params file not found at: {path}")
        raise
    except KeyError:
        logger.error("Key 'data_ingestion -> test_size' not found in params.yaml")
        raise
    except Exception as e:
        logger.exception("Unexpected error while loading params")
        raise e


def load_data(url: str) -> pd.DataFrame:
    """Load dataset from given URL."""
    try:
        logger.info(f"Loading dataset from URL: {url}")
        df = pd.read_csv(url)
        logger.debug(f"Dataset loaded with shape: {df.shape}")
        return df
    except Exception as e:
        logger.exception("Failed to load data")
        raise e


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Filter sentiments and encode target variable."""
    try:
        logger.info("Starting preprocessing step")

        df = df.drop(columns=['tweet_id'])
        final_df = df[df['sentiment'].isin(['happiness', 'sadness'])].copy()

        final_df.loc[:, 'sentiment'] = final_df['sentiment'].replace({
            'happiness': 1,
            'sadness': 0
        })

        logger.debug(f"Preprocessed data shape: {final_df.shape}")
        return final_df

    except KeyError as e:
        logger.error(f"Missing required column: {e}")
        raise
    except Exception as e:
        logger.exception("Error during preprocessing")
        raise e


def split_data(
    df: pd.DataFrame, test_size: float
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split data into train and test sets."""
    try:
        logger.info("Splitting data into train and test sets")

        train_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=42,
            stratify=df['sentiment']
        )

        logger.debug(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")
        return train_df, test_df

    except Exception as e:
        logger.exception("Error while splitting data")
        raise e


def save_data(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: str
) -> None:
    """Save train and test data to disk."""
    try:
        logger.info(f"Saving train and test data to {output_dir}")
        os.makedirs(output_dir, exist_ok=True)

        train_path = os.path.join(output_dir, "train.csv")
        test_path = os.path.join(output_dir, "test.csv")

        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)

        logger.debug(f"Train data saved at: {train_path}")
        logger.debug(f"Test data saved at: {test_path}")

    except Exception as e:
        logger.exception("Error while saving data")
        raise e


def main() -> None:
    DATA_URL = "https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/main/tweet_emotions.csv"
    PARAMS_PATH = "params.yaml"
    OUTPUT_DIR = os.path.join("data", "raw")

    try:
        logger.info("🚀 Starting Data Ingestion Pipeline")

        test_size = load_params(PARAMS_PATH)
        df = load_data(DATA_URL)
        processed_df = preprocess_data(df)
        train_df, test_df = split_data(processed_df, test_size)
        save_data(train_df, test_df, OUTPUT_DIR)

        logger.info("Data ingestion completed successfully.")

    except Exception as e:
        logger.error(f"Data ingestion pipeline failed: {e}")
        raise e


if __name__ == "__main__":
    main()
