# Quick Reference: MLflow Experiment Tracking

## TL;DR - Get Started in 30 Seconds

### 1. Start MLflow UI
```bash
python start_mlflow_server.py
```

### 2. Run Pipeline
```bash
python run_pipeline_with_tracking.py
```

### 3. View Results
Open: `http://127.0.0.1:5000`

---

## What Gets Tracked Automatically

### Every Pipeline Stage Logs:
- ✓ **Parameters** - Hyperparameters & configurations
- ✓ **Metrics** - Performance scores (accuracy, precision, recall, AUC)
- ✓ **Artifacts** - Datasets, models, configs
- ✓ **Dataset Info** - Size, shape, feature count
- ✓ **Run ID** - Unique identifier for each experiment run

### Specific Tracking by Stage:

| Stage | Parameters | Metrics | Artifacts |
|-------|------------|---------|-----------|
| **Data Ingestion** | test_size, data_url, sizes | - | train.csv, test.csv |
| **Data Preprocessing** | text_column, removed_rows | - | processed CSVs |
| **Feature Engineering** | max_features, actual_features | - | bow CSVs |
| **Model Building** | n_estimators, learning_rate | - | model.pkl |
| **Model Evaluation** | test samples | accuracy, precision, recall, AUC | metrics.json |

---

## Common Commands

### Start MLflow Server
```bash
# Easy way (recommended)
python start_mlflow_server.py

# Manual way
mlflow server --host 127.0.0.1 --port 5000

# Custom configuration
python start_mlflow_server.py --port 8080 --host 0.0.0.0
```

### Run Pipeline
```bash
# All at once
python run_pipeline_with_tracking.py

# Individual stages
python src/data/Data_ingestion.py
python src/data/data_preprocessing.py
python src/features/feature_engineering.py
python src/models/model_building.py
python src/models/model_evaluation.py
```

### Access MLflow UI
```
http://127.0.0.1:5000
```

---

## MLflow UI Features

### Experiments View
- See all experiments (Data_Ingestion, Model_Building, etc.)
- Filter by experiment type
- Compare multiple runs

### Run Details
- Parameters used
- Metrics achieved
- Artifacts logged
- Execution timeline

### Model Registry
- View registered models
- Track model versions
- Stage transitions (dev → staging → production)

---

## Using the Tracking Utility

```python
from src.mlflow_tracking import ExperimentTracker

# Create tracker
tracker = ExperimentTracker("My_Pipeline")

# Start run
with tracker.start_run(run_name="experiment_1"):
    # Log parameters
    tracker.log_parameters({
        "param1": 100,
        "param2": 0.1
    }, stage="training")
    
    # Log metrics
    tracker.log_metrics({
        "accuracy": 0.95,
        "f1_score": 0.93
    })
    
    # Log dataset info
    tracker.log_dataset_stats(
        name="train_data",
        size=10000,
        features=50,
        samples=8000
    )
    
    # Log model
    tracker.log_model(model, "MyModel")
    
    # Log artifacts
    tracker.log_artifacts({
        "path/to/file.pkl": "model"
    })
```

---

## Viewing Tracked Data in MLflow UI

### 1. **Parameters Tab**
Shows all logged parameters with values

### 2. **Metrics Tab**
Shows performance metrics with graphs over time

### 3. **Artifacts Tab**
Download logged files:
- Trained models
- Datasets
- Configuration files
- Results

### 4. **Runs Comparison**
Compare parameters and metrics across multiple runs

---

## Integrating with DagsHub (Remote Tracking)

### Setup
1. Create account at https://dagshub.com
2. Create a repository
3. Update tracking URI in your scripts:

```python
import mlflow

mlflow.set_tracking_uri("https://dagshub.com/<username>/<repo>.mlflow")
```

### Push Experiments
```bash
# Run pipeline as usual
python run_pipeline_with_tracking.py

# Experiments automatically sync to DagsHub
```

### View on DagsHub
- Go to your repository
- Click "Experiments" tab
- See all tracked runs

---

## Troubleshooting

### Issue: "MLflow is not tracking anything"
**Solution:** Make sure `mlflow.start_run()` is being used:
```python
with mlflow.start_run():
    mlflow.log_param("param", value)
```

### Issue: "Can't access MLflow UI"
**Solution:** Check if server is running:
```bash
# Restart it
python start_mlflow_server.py
```

### Issue: "Artifacts are empty"
**Solution:** Verify artifact paths exist:
```python
import os
if os.path.exists(path):
    mlflow.log_artifact(path)
```

### Issue: "Can't connect to remote MLflow"
**Solution:** Check tracking URI:
```python
import mlflow
print(mlflow.get_tracking_uri())
```

---

## Useful MLflow Links

- 📚 [MLflow Docs](https://mlflow.org/)
- 🤖 [MLflow Tracking](https://mlflow.org/docs/latest/tracking/)
- 🏛️ [Model Registry](https://mlflow.org/docs/latest/model-registry/)
- 🔬 [DagsHub Integration](https://dagshub.com/)

---

## Environment Variables

```bash
# Set tracking URI
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000

# Set default experiment
export MLFLOW_EXPERIMENT_NAME=My_Pipeline

# Enable autologging
export MLFLOW_AUTOLOG_ENABLED=TRUE
```

---

## Next Steps

1. ✓ Integrated MLflow tracking in all pipeline stages
2. ⏭️ Start MLflow server: `python start_mlflow_server.py`
3. ⏭️ Run pipeline: `python run_pipeline_with_tracking.py`
4. ⏭️ View experiments: Open http://127.0.0.1:5000
5. ⏭️ Compare runs and optimize hyperparameters
6. ⏭️ Register best model for production

---

**Need help?** Check `MLFLOW_TRACKING.md` for detailed documentation.
