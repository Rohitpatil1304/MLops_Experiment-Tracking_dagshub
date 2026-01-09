import json
import os
import pickle
from typing import Dict, Tuple

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import logging
import mlflow
import mlflow.sklearn


mlflow.set_experiment("Model_Evaluation")

# ---------------- Logging Configuration ---------------- #

logger = logging.getLogger("model_evaluation")
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


def load_features(path: str) -> pd.DataFrame:
    """Load test feature CSV file."""
    try:
        logger.info(f"Loading test features from: {path}")
        df = pd.read_csv(path)
        logger.debug(f"Loaded test data shape: {df.shape}")
        return df
    except FileNotFoundError:
        logger.error(f"Test feature file not found: {path}")
        raise
    except Exception as e:
        logger.exception("Error loading test features")
        raise e


def load_model(path: str) -> BaseEstimator:
    """Load trained model from disk."""
    try:
        logger.info(f"Loading trained model from: {path}")
        with open(path, 'rb') as f:
            model = pickle.load(f)
        logger.debug("Model loaded successfully")
        return model
    except FileNotFoundError:
        logger.error(f"Model file not found: {path}")
        raise
    except Exception as e:
        logger.exception("Error loading model")
        raise e


def split_features_and_labels(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Split dataframe into X and y."""
    try:
        logger.info("Splitting test features and labels")

        if df.shape[1] < 2:
            raise ValueError("Feature dataframe must contain features and label column")

        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]

        logger.debug(f"X_test shape: {X.shape}, y_test length: {len(y)}")
        return X, y

    except Exception as e:
        logger.exception("Error splitting features and labels")
        raise e


def evaluate_model(
    model: BaseEstimator,
    X: pd.DataFrame,
    y: pd.Series
) -> Dict[str, float]:
    """Compute evaluation metrics."""
    try:
        logger.info("Evaluating model on test data")

        y_pred = model.predict(X)

        if hasattr(model, "predict_proba"):
            y_pred_proba = model.predict_proba(X)[:, 1]
            auc = roc_auc_score(y, y_pred_proba)
        else:
            auc = float("nan")

        metrics = {
            "accuracy": accuracy_score(y, y_pred),
            "precision": precision_score(y, y_pred),
            "recall": recall_score(y, y_pred),
            "auc": auc
        }

        logger.debug(f"Evaluation metrics: {metrics}")
        return metrics

    except Exception as e:
        logger.exception("Model evaluation failed")
        raise e


def save_metrics(metrics: Dict[str, float], path: str) -> None:
    """Save metrics dictionary to JSON file."""
    try:
        logger.info(f"Saving metrics to: {path}")

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(metrics, f, indent=4)

        logger.debug("Metrics saved successfully")

    except Exception as e:
        logger.exception("Error saving metrics")
        raise e


def main() -> None:
    TEST_FEATURES_PATH = os.path.join("data", "features", "test_bow.csv")
    MODEL_PATH = os.path.join("models", "model.pkl")
    METRICS_PATH = os.path.join("reports", "metrics.json")

    try:
        logger.info("🚀 Starting Model Evaluation Pipeline")

        test_df = load_features(TEST_FEATURES_PATH)
        model = load_model(MODEL_PATH)

        X_test, y_test = split_features_and_labels(test_df)
        metrics = evaluate_model(model, X_test, y_test)

        save_metrics(metrics, METRICS_PATH)

        logger.info("Model evaluation completed successfully.")

    except Exception as e:
        logger.error(f"Model evaluation pipeline failed: {e}")
        raise e


if __name__ == "__main__":
    main()
