# MLflow Experiment Tracking Implementation Summary

## Overview
Comprehensive MLflow experiment tracking has been integrated across your entire ML pipeline. This tracks parameters, metrics, artifacts, source code, datasets, and models at each stage of execution.

## What Was Added

### 1. **Core Pipeline Modifications**

All five pipeline stages now include MLflow tracking:

#### Data Ingestion (`src/data/Data_ingestion.py`)
- **Parameters Tracked:**
  - `data_url` - Source of the dataset
  - `initial_dataset_size` - Raw data size
  - `initial_features` - Number of columns
  - `test_size` - Train-test split ratio
  - `processed_dataset_size` - After filtering
  - `removed_rows` - Rows removed during preprocessing
  - `train_set_size`, `test_set_size`
  
- **Artifacts Logged:**
  - `train.csv` - Training dataset
  - `test.csv` - Testing dataset

#### Data Preprocessing (`src/data/data_preprocessing.py`)
- **Parameters Tracked:**
  - `text_column` - Column being processed
  - `train_raw_size`, `test_raw_size` - Input sizes
  - `train_processed_size`, `test_processed_size` - Output sizes
  - `train_removed_rows`, `test_removed_rows` - Rows removed
  
- **Artifacts Logged:**
  - `train_processed.csv` - Processed training data
  - `test_processed.csv` - Processed testing data

#### Feature Engineering (`src/features/feature_engineering.py`)
- **Parameters Tracked:**
  - `max_features` - CountVectorizer max features
  - `vectorizer` - Type of vectorizer used
  - `text_column`, `label_column` - Data columns
  - `actual_features_generated` - Generated feature count
  - `train_features_shape`, `test_features_shape` - Output dimensions
  
- **Artifacts Logged:**
  - `train_bow.csv` - Training features
  - `test_bow.csv` - Testing features

#### Model Building (`src/models/model_building.py`)
- **Parameters Tracked:**
  - `n_estimators` - Number of boosting stages
  - `learning_rate` - Boosting learning rate
  - `random_state` - Random seed
  - `train_samples`, `train_features` - Training data dimensions
  - `dataset_samples`, `dataset_features` - Dataset statistics
  
- **Artifacts Logged:**
  - `model.pkl` - Trained model
  - `params.yaml` - Configuration file
  - **Registered Model:** `GradientBoosting_Model`

#### Model Evaluation (`src/models/model_evaluation.py`)
- **Metrics Tracked:**
  - `accuracy` - Classification accuracy
  - `precision` - Precision score
  - `recall` - Recall score
  - `auc` - ROC-AUC score
  
- **Parameters Logged:**
  - `test_samples`, `test_features` - Test data dimensions
  
- **Artifacts Logged:**
  - `metrics.json` - Evaluation results
  - `model.pkl` - Model reference

### 2. **New Utility Module**

**File:** `src/mlflow_tracking.py`

Provides an `ExperimentTracker` class with convenient methods:
- `log_parameters()` - Log multiple parameters at once
- `log_metrics()` - Log performance metrics
- `log_artifact()` / `log_artifacts()` - Log files
- `log_dataset_stats()` - Log dataset information
- `log_model()` - Register and log models
- `log_source_code()` - Track source files
- `log_pipeline_stage()` - Log stage execution info
- `log_config()` - Save configuration as JSON
- `log_error()` - Track error information

**Example Usage:**
```python
from src.mlflow_tracking import ExperimentTracker

tracker = ExperimentTracker(experiment_name="My_Pipeline")
with tracker.start_run(run_name="experiment_1"):
    tracker.log_parameters({"param1": value1}, stage="stage_name")
    tracker.log_metrics({"metric1": 0.95})
    tracker.log_model(model, model_name="MyModel")
```

### 3. **Helper Scripts**

#### `run_pipeline_with_tracking.py`
Automated script to run the entire pipeline with logging:
```bash
python run_pipeline_with_tracking.py
```
- Executes all stages sequentially
- Captures and reports success/failure
- Measures execution duration
- Provides helpful next steps

#### `start_mlflow_server.py`
Easy-to-use script to start MLflow UI:
```bash
python start_mlflow_server.py
python start_mlflow_server.py --host 127.0.0.1 --port 5000
```

### 4. **Documentation**

**File:** `MLFLOW_TRACKING.md`
- Comprehensive guide on what's tracked
- How to start the tracking server
- How to run the pipeline
- Usage of the tracking utility
- Best practices
- Troubleshooting guide
- DagsHub integration instructions

## How to Use

### Step 1: Ensure MLflow is Installed
```bash
pip install mlflow
```

### Step 2: Start MLflow Server
```bash
python start_mlflow_server.py
# OR manually
mlflow server --host 127.0.0.1 --port 5000
```

### Step 3: Run the Pipeline
```bash
# Option A: Run all stages automatically
python run_pipeline_with_tracking.py

# Option B: Run individual scripts
python src/data/Data_ingestion.py
python src/data/data_preprocessing.py
python src/features/feature_engineering.py
python src/models/model_building.py
python src/models/model_evaluation.py

# Option C: Use DVC (if configured)
dvc repro
```

### Step 4: View Tracked Experiments
Open browser: `http://127.0.0.1:5000`

## Key Features

✓ **Parameters Tracking** - All hyperparameters and configurations
✓ **Metrics Tracking** - Model performance metrics
✓ **Artifacts Logging** - Datasets, models, configs
✓ **Source Code Tracking** - Reproducibility
✓ **Dataset Statistics** - Data shape, size, counts
✓ **Model Registry** - Model versioning and staging
✓ **Pipeline Stages** - Track each stage separately
✓ **Error Logging** - Capture failures with context

## MLflow Experiments Created

The following experiments are automatically created:

1. **Data_Ingestion_Pipeline** - Data loading and splitting
2. **Data_Preprocessing_Pipeline** - Text normalization
3. **Feature_Engineering_Pipeline** - Feature extraction
4. **GradientBoosting_Model_Building** - Model training
5. **Model_Evaluation** - Model evaluation

## Directory Structure

```
project/
├── MLFLOW_TRACKING.md                    # Tracking guide
├── run_pipeline_with_tracking.py          # Automated pipeline runner
├── start_mlflow_server.py                 # MLflow server starter
├── mlruns/                                # MLflow tracking data
├── src/
│   ├── mlflow_tracking.py                # Tracking utility class
│   ├── data/
│   │   ├── Data_ingestion.py             # With MLflow tracking
│   │   └── data_preprocessing.py          # With MLflow tracking
│   ├── features/
│   │   └── feature_engineering.py         # With MLflow tracking
│   └── models/
│       ├── model_building.py              # With MLflow tracking
│       └── model_evaluation.py            # With MLflow tracking
└── ...
```

## Troubleshooting

### Tracking Not Showing Up?
1. Ensure MLflow server is running: `mlflow server --host 127.0.0.1 --port 5000`
2. Verify `mlflow.start_run()` context is used in scripts
3. Check that artifact paths exist before logging
4. Check logs for any error messages

### Can't Connect to Server?
```bash
# Check if MLflow is installed
pip list | grep mlflow

# Restart server
python start_mlflow_server.py
```

### Artifacts Not Logging?
- Verify paths exist: `if os.path.exists(path):`
- Ensure scripts have write permissions
- Check current working directory is correct

## Next Steps

1. ✓ MLflow tracking is now integrated
2. Run the pipeline: `python run_pipeline_with_tracking.py`
3. View experiments in MLflow UI: `http://127.0.0.1:5000`
4. Compare runs and identify best configurations
5. Register best models for deployment

## Integration with DagsHub

To push experiments to DagsHub:

1. Update tracking URI in scripts:
   ```python
   mlflow.set_tracking_uri("https://dagshub.com/<user>/<repo>.mlflow")
   ```

2. Set credentials in `.dagshub/config.yaml`

3. Runs will automatically sync to DagsHub

## References

- [MLflow Documentation](https://mlflow.org/docs/)
- [MLflow Tracking](https://mlflow.org/docs/latest/tracking/)
- [MLflow Model Registry](https://mlflow.org/docs/latest/model-registry/)
- [DagsHub Docs](https://dagshub.com/)

---

**Status:** ✓ Complete - All pipeline stages now have comprehensive MLflow tracking
