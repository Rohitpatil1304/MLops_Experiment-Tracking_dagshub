"""
MLflow Tracking Utility Module
This module provides utility functions for comprehensive experiment tracking
across all pipeline stages.
"""

import os
import mlflow
import mlflow.sklearn
import logging
from typing import Dict, Any, Optional
import json
from datetime import datetime

logger = logging.getLogger("mlflow_tracking")


class ExperimentTracker:
    """
    Centralized class for managing MLflow experiment tracking across the pipeline.
    Tracks parameters, metrics, artifacts, and dataset information at each stage.
    """
    
    def __init__(self, experiment_name: str, tracking_uri: str = "mlruns"):
        """
        Initialize the experiment tracker.
        
        Args:
            experiment_name: Name of the MLflow experiment
            tracking_uri: MLflow tracking URI (default: local mlruns directory)
        """
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        self.experiment_name = experiment_name
        self.run_id = None
        logger.info(f"Initialized ExperimentTracker for experiment: {experiment_name}")
    
    def start_run(self, run_name: Optional[str] = None):
        """
        Start a new MLflow run.
        
        Args:
            run_name: Optional name for the run
            
        Returns:
            MLflow run context
        """
        self.run = mlflow.start_run(run_name=run_name)
        self.run_id = self.run.info.run_id
        logger.info(f"Started MLflow run with ID: {self.run_id}")
        return self.run
    
    def end_run(self):
        """End the current MLflow run."""
        if self.run_id:
            mlflow.end_run()
            logger.info(f"Ended MLflow run: {self.run_id}")
    
    def log_parameters(self, params: Dict[str, Any], stage: str = ""):
        """
        Log multiple parameters at once.
        
        Args:
            params: Dictionary of parameters to log
            stage: Pipeline stage name for context
        """
        mlflow.log_params(params)
        logger.info(f"Logged {len(params)} parameters{' at stage: ' + stage if stage else ''}")
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Parameters: {params}")
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None, stage: str = ""):
        """
        Log multiple metrics at once.
        
        Args:
            metrics: Dictionary of metrics to log
            step: Optional step number for metric tracking
            stage: Pipeline stage name for context
        """
        mlflow.log_metrics(metrics, step=step)
        logger.info(f"Logged {len(metrics)} metrics{' at stage: ' + stage if stage else ''}")
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Metrics: {metrics}")
    
    def log_artifact(self, artifact_path: str, artifact_type: str = ""):
        """
        Log a single artifact.
        
        Args:
            artifact_path: Path to the artifact file
            artifact_type: Type/category of artifact (e.g., 'model', 'dataset')
        """
        if os.path.exists(artifact_path):
            mlflow.log_artifact(artifact_path, artifact_type)
            logger.info(f"Logged artifact: {artifact_path}")
        else:
            logger.warning(f"Artifact path does not exist: {artifact_path}")
    
    def log_artifacts(self, artifact_paths: Dict[str, str]):
        """
        Log multiple artifacts.
        
        Args:
            artifact_paths: Dictionary mapping artifact path to artifact type
        """
        for path, artifact_type in artifact_paths.items():
            self.log_artifact(path, artifact_type)
    
    def log_dataset_stats(self, name: str, size: int, features: int, 
                         samples: int, additional_info: Dict = None):
        """
        Log comprehensive dataset statistics.
        
        Args:
            name: Dataset name
            size: Total size in bytes (or rows)
            features: Number of features/columns
            samples: Number of samples/rows
            additional_info: Additional dataset information
        """
        params = {
            f"{name}_size": size,
            f"{name}_features": features,
            f"{name}_samples": samples
        }
        
        if additional_info:
            for key, value in additional_info.items():
                params[f"{name}_{key}"] = value
        
        mlflow.log_params(params)
        logger.info(f"Logged dataset stats for: {name}")
    
    def log_model(self, model, model_name: str = "model", 
                  artifact_path: str = "model"):
        """
        Log a trained model.
        
        Args:
            model: Trained model object
            model_name: Name of the model for registration
            artifact_path: Path within MLflow artifacts
        """
        mlflow.sklearn.log_model(
            model, 
            artifact_path, 
            registered_model_name=model_name
        )
        logger.info(f"Logged model: {model_name}")
    
    def log_source_code(self, source_file_path: str, artifact_type: str = "source_code"):
        """
        Log source code file for reproducibility.
        
        Args:
            source_file_path: Path to source code file
            artifact_type: Type of artifact
        """
        if os.path.exists(source_file_path):
            mlflow.log_artifact(source_file_path, artifact_type)
            logger.info(f"Logged source code: {source_file_path}")
        else:
            logger.warning(f"Source code file not found: {source_file_path}")
    
    def log_pipeline_stage(self, stage_name: str, status: str, 
                          duration: float = None, additional_info: Dict = None):
        """
        Log information about a pipeline stage execution.
        
        Args:
            stage_name: Name of the pipeline stage
            status: Execution status (success/failed)
            duration: Execution duration in seconds
            additional_info: Additional stage information
        """
        params = {
            f"{stage_name}_status": status
        }
        
        if duration:
            params[f"{stage_name}_duration_seconds"] = duration
        
        if additional_info:
            for key, value in additional_info.items():
                params[f"{stage_name}_{key}"] = value
        
        mlflow.log_params(params)
        logger.info(f"Logged pipeline stage: {stage_name} - {status}")
    
    def log_config(self, config_dict: Dict[str, Any], config_file_path: str = None):
        """
        Log configuration as JSON artifact.
        
        Args:
            config_dict: Configuration dictionary
            config_file_path: Path to save config file
        """
        if config_file_path is None:
            config_file_path = f"config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(config_file_path, 'w') as f:
            json.dump(config_dict, f, indent=4)
        
        mlflow.log_artifact(config_file_path, "config")
        os.remove(config_file_path)  # Clean up temp file
        logger.info(f"Logged configuration artifact")
    
    def get_run_id(self) -> str:
        """Get the current run ID."""
        return self.run_id
    
    def log_error(self, error_message: str, stage: str = ""):
        """
        Log error information.
        
        Args:
            error_message: Error message
            stage: Pipeline stage where error occurred
        """
        mlflow.log_param(f"error_stage", stage or "unknown")
        mlflow.log_param(f"error_message", error_message)
        logger.error(f"Logged error at stage {stage}: {error_message}")


def create_pipeline_tracker(pipeline_name: str) -> ExperimentTracker:
    """
    Factory function to create a pipeline tracker.
    
    Args:
        pipeline_name: Name of the pipeline
        
    Returns:
        ExperimentTracker instance
    """
    return ExperimentTracker(experiment_name=pipeline_name)


def log_pipeline_summary(tracker: ExperimentTracker, stages_info: Dict[str, Dict]):
    """
    Log a comprehensive summary of the entire pipeline execution.
    
    Args:
        tracker: ExperimentTracker instance
        stages_info: Dictionary containing information about each stage
    """
    summary = {
        "timestamp": datetime.now().isoformat(),
        "stages": stages_info
    }
    
    tracker.log_config(summary)
    logger.info("Logged complete pipeline summary")
