import os
import re
from typing import Tuple

import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import logging
import yaml

# ---------------- Logging Configuration ---------------- #
logger = logging.getLogger("data_preprocessing")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
console_handler.setFormatter(formatter)

if not logger.handlers:
    logger.addHandler(console_handler)
# ------------------------------------------------------ #


def load_params(path: str) -> str:
    """Load text_column from params.yaml."""
    try:
        logger.info(f"Loading parameters from {path}")
        with open(path, 'r') as f:
            params = yaml.safe_load(f)

        text_column = params['data_preprocessing']['text_column']
        logger.debug(f"Loaded text_column = {text_column}")
        return text_column

    except FileNotFoundError:
        logger.error(f"Params file not found at: {path}")
        raise
    except KeyError:
        logger.error("Key 'data_preprocessing -> text_column' not found in params.yaml")
        raise
    except Exception as e:
        logger.exception("Unexpected error while loading params")
        raise e


def download_nltk_resources() -> None:
    """Download required NLTK resources."""
    try:
        logger.info("Downloading required NLTK resources")
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        logger.info("NLTK resources downloaded successfully")
    except Exception as e:
        logger.exception("Failed to download NLTK resources")
        raise e


def load_data(path: str) -> pd.DataFrame:
    """Load CSV data from given path."""
    try:
        logger.info(f"Loading data from: {path}")
        df = pd.read_csv(path)
        logger.debug(f"Loaded data shape: {df.shape}")
        return df
    except FileNotFoundError:
        logger.error(f"File not found: {path}")
        raise
    except Exception as e:
        logger.exception("Error loading data")
        raise e


def lower_case(text: str) -> str:
    return " ".join(word.lower() for word in str(text).split())


def remove_stop_words(text: str) -> str:
    stop_words = set(stopwords.words('english'))
    return " ".join(word for word in str(text).split() if word not in stop_words)


def removing_numbers(text: str) -> str:
    return " ".join(word for word in str(text).split() if not word.isdigit())


def removing_punctuations(text: str) -> str:
    text = re.sub(r'[^\w\s]', ' ', str(text))
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def removing_urls(text: str) -> str:
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub('', str(text))


def lemmatization(text: str) -> str:
    lemmatizer = WordNetLemmatizer()
    return " ".join(lemmatizer.lemmatize(word) for word in str(text).split())


def normalize_text_column(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Apply full text normalization pipeline on a column."""
    try:
        logger.info(f"Normalizing text column: {column}")

        if column not in df.columns:
            raise KeyError(f"Column '{column}' not found in dataframe")

        df = df.copy()
        df[column] = df[column].astype(str)
        df[column] = df[column].apply(lower_case)
        df[column] = df[column].apply(remove_stop_words)
        df[column] = df[column].apply(removing_numbers)
        df[column] = df[column].apply(removing_punctuations)
        df[column] = df[column].apply(removing_urls)
        df[column] = df[column].apply(lemmatization)

        logger.debug(f"Text normalization completed for column: {column}")
        return df

    except Exception as e:
        logger.exception("Error during text normalization")
        raise e


def remove_small_sentences(df: pd.DataFrame, column: str, min_len: int = 3) -> pd.DataFrame:
    """Replace short sentences with NaN."""
    try:
        logger.info(f"Removing sentences with less than {min_len} words in column: {column}")

        df = df.copy()
        df[column] = df[column].apply(
            lambda x: x if len(str(x).split()) >= min_len else np.nan
        )

        removed = df[column].isna().sum()
        logger.debug(f"Total short/empty sentences replaced with NaN: {removed}")
        return df

    except Exception as e:
        logger.exception("Error removing small sentences")
        raise e


def save_data(df: pd.DataFrame, path: str) -> None:
    """Save DataFrame to CSV."""
    try:
        logger.info(f"Saving processed data to: {path}")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_csv(path, index=False)
        logger.debug(f"Data saved successfully at: {path}")
    except Exception as e:
        logger.exception("Error saving processed data")
        raise e


def main() -> None:
    RAW_DIR = os.path.join("data", "raw")
    PROCESSED_DIR = os.path.join("data", "processed")
    PARAMS_PATH = "params.yaml"

    train_path = os.path.join(RAW_DIR, "train.csv")
    test_path = os.path.join(RAW_DIR, "test.csv")

    try:
        logger.info("🚀 Starting Data Preprocessing Pipeline")

        text_column = load_params(PARAMS_PATH)
        download_nltk_resources()

        train_df = load_data(train_path)
        test_df = load_data(test_path)

        train_df = normalize_text_column(train_df, text_column)
        test_df = normalize_text_column(test_df, text_column)

        train_df = remove_small_sentences(train_df, text_column)
        test_df = remove_small_sentences(test_df, text_column)

        save_data(train_df, os.path.join(PROCESSED_DIR, "train_processed.csv"))
        save_data(test_df, os.path.join(PROCESSED_DIR, "test_processed.csv"))

        logger.info("Data preprocessing completed successfully.")

    except Exception as e:
        logger.error(f"Data preprocessing failed: {e}")
        raise e


if __name__ == "__main__":
    main()
