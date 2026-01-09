import os
import pickle
from typing import Tuple

import pandas as pd
import yaml
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.base import BaseEstimator
import logging
import mlflow
import mlflow.sklearn

# Set experiment name
mlflow.set_experiment("GradientBoosting_Model_Building")


# ---------------- Logging Configuration ---------------- #
logger = logging.getLogger("model_building")
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


def load_params(path: str) -> Tuple[int, float]:
    """Load model hyperparameters from params.yaml."""
    try:
        logger.info(f"Loading model parameters from {path}")
        with open(path, 'r') as f:
            params = yaml.safe_load(f)

        n_estimators = int(params['model_building']['n_estimators'])
        learning_rate = float(params['model_building']['learning_rate'])

        logger.debug(
            f"Loaded hyperparameters - n_estimators: {n_estimators}, "
            f"learning_rate: {learning_rate}"
        )
        return n_estimators, learning_rate

    except FileNotFoundError:
        logger.error(f"Params file not found: {path}")
        raise
    except KeyError:
        logger.error("Keys 'model_building -> n_estimators/learning_rate' not found in params.yaml")
        raise
    except Exception as e:
        logger.exception("Unexpected error while loading model parameters")
        raise e


def load_features(path: str) -> pd.DataFrame:
    """Load feature CSV file."""
    try:
        logger.info(f"Loading training features from: {path}")
        df = pd.read_csv(path)
        logger.debug(f"Loaded training data shape: {df.shape}")
        return df
    except FileNotFoundError:
        logger.error(f"Feature file not found: {path}")
        raise
    except Exception as e:
        logger.exception("Error loading training features")
        raise e


def split_features_and_labels(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Split dataframe into X and y."""
    try:
        logger.info("Splitting features and labels")

        if df.shape[1] < 2:
            raise ValueError("Feature dataframe must have at least one feature and one label column")

        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]

        logger.debug(f"X shape: {X.shape}, y length: {len(y)}")
        return X, y

    except Exception as e:
        logger.exception("Error splitting features and labels")
        raise e


def train_model(
    X: pd.DataFrame,
    y: pd.Series,
    n_estimators: int,
    learning_rate: float
) -> BaseEstimator:
    """Train Gradient Boosting model."""
    try:
        logger.info("Training GradientBoostingClassifier")

        model = GradientBoostingClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            random_state=42
        )
        model.fit(X, y)

        logger.info("Model training completed successfully")
        return model

    except Exception as e:
        logger.exception("Model training failed")
        raise e


def save_model(model: BaseEstimator, path: str) -> None:
    """Serialize trained model to disk."""
    try:
        logger.info(f"Saving trained model to: {path}")

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(model, f)

        logger.debug(f"Model saved successfully at: {path}")

    except Exception as e:
        logger.exception("Error saving model")
        raise e

def main() -> None:
    PARAMS_PATH = "params.yaml"
    FEATURES_PATH = os.path.join("data", "features", "train_bow.csv")
    MODEL_PATH = os.path.join("models", "model.pkl")

    # Start MLflow run
    with mlflow.start_run() as run:
        try:
            logger.info("🚀 Starting Model Building Pipeline")
            logger.info(f"MLflow Run ID: {run.info.run_id}")

            n_estimators, learning_rate = load_params(PARAMS_PATH)
            train_df = load_features(FEATURES_PATH)

            X_train, y_train = split_features_and_labels(train_df)
            
            # Log parameters
            mlflow.log_param("n_estimators", n_estimators)
            mlflow.log_param("learning_rate", learning_rate)
            mlflow.log_param("random_state", 42)
            mlflow.log_param("train_samples", len(X_train))
            mlflow.log_param("train_features", X_train.shape[1])
            logger.info("Logged model parameters to MLflow")
            
            model = train_model(X_train, y_train, n_estimators, learning_rate)

            save_model(model, MODEL_PATH)
            logger.info(f"Model saved to {MODEL_PATH}")
            
            # Log model and artifacts
            mlflow.sklearn.log_model(model, "model", registered_model_name="GradientBoosting_Model")
            mlflow.log_artifact(MODEL_PATH, "artifacts")
            mlflow.log_artifact(PARAMS_PATH, "config")
            logger.info("Logged model and artifacts to MLflow")
            
            # Log dataset info
            mlflow.log_param("dataset_samples", len(train_df))
            mlflow.log_param("dataset_features", train_df.shape[1])
            logger.info("Logged dataset information to MLflow")

            logger.info("Model building completed and tracked successfully.")

        except Exception as e:
            logger.error(f"Model building pipeline failed: {e}")
            mlflow.log_param("status", "failed")
            mlflow.log_param("error_message", str(e))
            raise e



if __name__ == "__main__":
    main()
