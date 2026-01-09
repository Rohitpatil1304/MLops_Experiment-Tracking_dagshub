# MLflow Experiment Tracking Guide

This document describes the comprehensive MLflow experiment tracking setup integrated into your ML pipeline.

## Overview

MLflow tracking is now fully integrated across all pipeline stages:
- **Data Ingestion** - Dataset size, test_size ratio, data URL
- **Data Preprocessing** - Text normalization, rows removed, processed dataset size
- **Feature Engineering** - Vectorizer configuration, feature dimensions
- **Model Building** - Hyperparameters, model architecture, dataset stats
- **Model Evaluation** - Performance metrics (accuracy, precision, recall, AUC)

## What Gets Tracked

### 1. **Parameters**
Configuration values and hyperparameters at each stage:
- Data ingestion parameters (test_size, dataset URL)
- Preprocessing configuration (text column)
- Feature engineering parameters (max_features, vectorizer type)
- Model hyperparameters (n_estimators, learning_rate, random_state)
- Dataset statistics (size, feature count, sample count)

### 2. **Metrics**
Performance measurements:
- Model evaluation metrics (accuracy, precision, recall, AUC)
- Data statistics (rows removed, processed data size)
- Feature engineering stats (actual features generated, shape dimensions)

### 3. **Artifacts**
Files logged for reproducibility:
- Raw datasets (train.csv, test.csv)
- Processed datasets (train_processed.csv, test_processed.csv)
- Engineered features (train_bow.csv, test_bow.csv)
- Trained model (model.pkl)
- Configuration files (params.yaml)
- Evaluation metrics (metrics.json)

### 4. **Source Code**
Python source files are logged for complete reproducibility

## Starting MLflow Server

### Local Server
```bash
mlflow server --host 127.0.0.1 --port 5000
```

Then access the UI at: `http://127.0.0.1:5000`

### Dagshub Integration (for remote tracking)
If you're using DagsHub, configure MLflow to track remotely:

```python
import mlflow

# Set Dagshub credentials (before running pipeline)
dagshub_url = "https://dagshub.com/<your-username>/<your-repo-name>"
mlflow.set_tracking_uri(f"{dagshub_url}.mlflow")
```

## Running the Pipeline with Tracking

### Option 1: Run Individual Scripts
```bash
# Data Ingestion
python src/data/Data_ingestion.py

# Data Preprocessing
python src/data/data_preprocessing.py

# Feature Engineering
python src/features/feature_engineering.py

# Model Building
python src/models/model_building.py

# Model Evaluation
python src/models/model_evaluation.py
```

### Option 2: Run with DVC
```bash
# Uncomment the stages in dvc.yaml first

# Run all stages
dvc repro

# Run specific stage
dvc repro -s model_building
```

## Using the Tracking Utility

A comprehensive tracking utility class is provided in `src/mlflow_tracking.py`:

```python
from src.mlflow_tracking import ExperimentTracker

# Create tracker for your pipeline
tracker = ExperimentTracker(experiment_name="My_Pipeline")

# Start a run
with tracker.start_run(run_name="experiment_1"):
    # Log parameters
    tracker.log_parameters({
        "n_estimators": 100,
        "learning_rate": 0.1
    }, stage="model_building")
    
    # Log metrics
    tracker.log_metrics({
        "accuracy": 0.95,
        "precision": 0.93
    }, stage="evaluation")
    
    # Log dataset stats
    tracker.log_dataset_stats(
        name="train_dataset",
        size=10000,
        features=100,
        samples=9000
    )
    
    # Log model
    tracker.log_model(model, model_name="GradientBoosting_Model")
    
    # Log artifacts
    tracker.log_artifacts({
        "models/model.pkl": "model",
        "data/features/train_bow.csv": "features"
    })
```

## Current Implementation in Scripts

### Data Ingestion (`src/data/Data_ingestion.py`)
- Logs: data URL, initial/processed dataset sizes, test ratio
- Artifacts: raw train.csv, test.csv

### Data Preprocessing (`src/data/data_preprocessing.py`)
- Logs: text column, raw/processed dataset sizes, rows removed
- Artifacts: processed datasets

### Feature Engineering (`src/features/feature_engineering.py`)
- Logs: max_features, vectorizer type, feature dimensions
- Artifacts: engineered feature files

### Model Building (`src/models/model_building.py`)
- Logs: n_estimators, learning_rate, random_state, dataset stats
- Artifacts: trained model.pkl, params.yaml
- Registered model: GradientBoosting_Model

### Model Evaluation (`src/models/model_evaluation.py`)
- Logs: accuracy, precision, recall, AUC, test dataset stats
- Artifacts: metrics.json, model.pkl

## MLflow UI Features

Once tracking server is running, you can:

1. **Compare Runs**
   - View parameters and metrics across different runs
   - Identify best performing configurations

2. **View Artifacts**
   - Download trained models
   - Access dataset and feature files
   - Review evaluation metrics

3. **Track Experiments**
   - Organize runs by experiment (Data_Ingestion, Model_Building, etc.)
   - Filter by parameter values
   - Sort by metrics

4. **Model Registry**
   - Track model versions
   - Stage transitions (staging, production)
   - Model comparisons

## Troubleshooting

### Issue: "MLflow tracking not showing anything"

**Solution:** Ensure you're using `mlflow.start_run()` context manager:
```python
with mlflow.start_run():
    # Your tracking code here
    mlflow.log_param("key", "value")
```

### Issue: "Artifacts not being logged"

**Solution:** Verify artifact paths exist:
```python
import os
if os.path.exists(artifact_path):
    mlflow.log_artifact(artifact_path)
```

### Issue: "Can't connect to MLflow server"

**Solution:** Make sure MLflow server is running:
```bash
mlflow server --host 127.0.0.1 --port 5000
```

## Best Practices

1. **Use Meaningful Experiment Names**
   ```python
   mlflow.set_experiment("GradientBoosting_v2_with_validation")
   ```

2. **Log Early and Often**
   - Log parameters before training
   - Log metrics after evaluation
   - Log artifacts for reproducibility

3. **Use Run Names for Organization**
   ```python
   mlflow.start_run(run_name="baseline_model_attempt_1")
   ```

4. **Tag Important Runs**
   ```python
   mlflow.set_tag("production", True)
   mlflow.set_tag("model_version", "1.0")
   ```

5. **Log Source Code**
   ```python
   mlflow.log_artifact("src/models/model_building.py", "source_code")
   ```

## Environment Variables

You can configure MLflow with environment variables:

```bash
# Set tracking URI
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000

# Set experiment name
export MLFLOW_EXPERIMENT_NAME=My_Pipeline

# Enable autologging
export MLFLOW_AUTOLOG_ENABLED=TRUE
```

## Integration with DagsHub

To push your experiments to DagsHub:

1. Set up remote tracking:
```python
mlflow.set_tracking_uri("https://dagshub.com/<user>/<repo>.mlflow")
```

2. Configure credentials in `.dagshub/config.yaml`

3. Runs will be automatically tracked on DagsHub platform

## Next Steps

1. Start MLflow server
2. Run the pipeline scripts
3. Open MLflow UI to view tracked experiments
4. Compare runs and identify best configurations
5. Register promising models for deployment

---

For more information, visit: https://mlflow.org/docs/latest/tracking/
