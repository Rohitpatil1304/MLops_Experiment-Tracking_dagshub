import os
from typing import Tuple

import pandas as pd
import yaml
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import csr_matrix
import logging
import mlflow

# ---------------- Logging Configuration ---------------- #
logger = logging.getLogger("feature_engineering")
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


def load_params(path: str) -> int:
    """Load max_features from params.yaml."""
    try:
        logger.info(f"Loading parameters from {path}")
        with open(path, 'r') as f:
            params = yaml.safe_load(f)

        max_features = int(params['feature_engineering']['max_features'])
        logger.debug(f"Loaded max_features = {max_features}")
        return max_features

    except FileNotFoundError:
        logger.error(f"Params file not found: {path}")
        raise
    except KeyError:
        logger.error("Key 'feature_engineering -> max_features' not found in params.yaml")
        raise
    except Exception as e:
        logger.exception("Unexpected error while loading params")
        raise e


def load_data(path: str) -> pd.DataFrame:
    """Load processed CSV data."""
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


def prepare_features_and_labels(
    df: pd.DataFrame,
    text_col: str,
    label_col: str
) -> Tuple[pd.Series, pd.Series]:
    """Split dataframe into features and labels."""
    try:
        logger.info("Preparing features and labels")

        if text_col not in df.columns or label_col not in df.columns:
            raise KeyError("Required columns not found in dataframe")

        df = df.copy()
        df[text_col] = df[text_col].fillna('')

        X = df[text_col]
        y = df[label_col]

        logger.debug(f"Features length: {len(X)}, Labels length: {len(y)}")
        return X, y

    except Exception as e:
        logger.exception("Error preparing features and labels")
        raise e


def vectorize_text(
    vectorizer: CountVectorizer,
    X_train: pd.Series,
    X_test: pd.Series
) -> Tuple[csr_matrix, csr_matrix]:
    """Fit vectorizer on train and transform both train & test."""
    try:
        logger.info("Vectorizing text data using CountVectorizer")

        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)

        logger.debug(
            f"Vectorization complete. "
            f"Train shape: {X_train_vec.shape}, Test shape: {X_test_vec.shape}"
        )
        return X_train_vec, X_test_vec

    except Exception as e:
        logger.exception("Vectorization failed")
        raise e


def save_features(
    X: csr_matrix,
    y: pd.Series,
    path: str
) -> None:
    """Save features and labels as CSV."""
    try:
        logger.info(f"Saving features to: {path}")

        df = pd.DataFrame(X.toarray())
        df['label'] = y.values

        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_csv(path, index=False)

        logger.debug(f"Features saved successfully at: {path}")

    except Exception as e:
        logger.exception("Error saving features")
        raise e


def main() -> None:
    PARAMS_PATH = "params.yaml"
    PROCESSED_DIR = os.path.join("data", "processed")
    FEATURES_DIR = os.path.join("data", "features")

    TRAIN_PATH = os.path.join(PROCESSED_DIR, "train_processed.csv")
    TEST_PATH = os.path.join(PROCESSED_DIR, "test_processed.csv")

    TEXT_COL = "content"
    LABEL_COL = "sentiment"

    # Set experiment name
    mlflow.set_experiment("Feature_Engineering_Pipeline")
    
    # Start MLflow run
    with mlflow.start_run() as run:
        try:
            logger.info("🚀 Starting Feature Engineering Pipeline")
            logger.info(f"MLflow Run ID: {run.info.run_id}")

            max_features = load_params(PARAMS_PATH)
            mlflow.log_param("max_features", max_features)
            mlflow.log_param("vectorizer", "CountVectorizer")
            mlflow.log_param("text_column", TEXT_COL)
            mlflow.log_param("label_column", LABEL_COL)
            logger.info("Logged feature engineering parameters to MLflow")

            train_df = load_data(TRAIN_PATH)
            test_df = load_data(TEST_PATH)
            
            # Log input data stats
            mlflow.log_param("train_input_size", len(train_df))
            mlflow.log_param("test_input_size", len(test_df))
            logger.info("Logged input data statistics to MLflow")

            X_train, y_train = prepare_features_and_labels(train_df, TEXT_COL, LABEL_COL)
            X_test, y_test = prepare_features_and_labels(test_df, TEXT_COL, LABEL_COL)

            vectorizer = CountVectorizer(max_features=max_features)
            X_train_bow, X_test_bow = vectorize_text(vectorizer, X_train, X_test)
            
            # Log vectorization stats
            mlflow.log_param("actual_features_generated", X_train_bow.shape[1])
            mlflow.log_param("train_features_shape", X_train_bow.shape[0])
            mlflow.log_param("test_features_shape", X_test_bow.shape[0])
            logger.info("Logged vectorization statistics to MLflow")

            train_bow_path = os.path.join(FEATURES_DIR, "train_bow.csv")
            test_bow_path = os.path.join(FEATURES_DIR, "test_bow.csv")
            
            save_features(X_train_bow, y_train, train_bow_path)
            save_features(X_test_bow, y_test, test_bow_path)
            
            # Log artifacts
            mlflow.log_artifact(train_bow_path, "engineered_features")
            mlflow.log_artifact(test_bow_path, "engineered_features")
            logger.info("Logged engineered features to MLflow")

            logger.info("Feature engineering completed and tracked successfully.")

        except Exception as e:
            logger.error(f"Feature engineering failed: {e}")
            mlflow.log_param("status", "failed")
            mlflow.log_param("error_message", str(e))
            raise e


if __name__ == "__main__":
    main()
