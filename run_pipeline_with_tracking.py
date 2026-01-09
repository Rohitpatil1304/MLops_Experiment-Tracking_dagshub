#!/usr/bin/env python
"""
MLflow Tracking - Quick Start Script
This script demonstrates how to run the entire pipeline with comprehensive MLflow tracking.

Usage:
    python run_pipeline_with_tracking.py
"""

import os
import sys
import subprocess
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_stage(stage_name: str, script_path: str) -> bool:
    """
    Run a pipeline stage and handle errors.
    
    Args:
        stage_name: Human-readable stage name
        script_path: Path to the Python script
        
    Returns:
        True if successful, False otherwise
    """
    logger.info(f"{'='*60}")
    logger.info(f"Running: {stage_name}")
    logger.info(f"Script: {script_path}")
    logger.info(f"{'='*60}")
    
    if not os.path.exists(script_path):
        logger.error(f"Script not found: {script_path}")
        return False
    
    try:
        result = subprocess.run(
            [sys.executable, script_path],
            check=True,
            capture_output=False
        )
        logger.info(f"✓ {stage_name} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"✗ {stage_name} failed with exit code {e.returncode}")
        return False
    except Exception as e:
        logger.error(f"✗ {stage_name} failed with error: {e}")
        return False


def main():
    """Run the complete pipeline with MLflow tracking."""
    
    logger.info("\n" + "="*60)
    logger.info("Starting MLflow Tracked ML Pipeline")
    logger.info("="*60 + "\n")
    
    start_time = datetime.now()
    
    # Define pipeline stages
    stages = [
        ("Data Ingestion", "src/data/Data_ingestion.py"),
        ("Data Preprocessing", "src/data/data_preprocessing.py"),
        ("Feature Engineering", "src/features/feature_engineering.py"),
        ("Model Building", "src/models/model_building.py"),
        ("Model Evaluation", "src/models/model_evaluation.py"),
    ]
    
    results = {}
    for stage_name, script_path in stages:
        success = run_stage(stage_name, script_path)
        results[stage_name] = "✓ Success" if success else "✗ Failed"
        
        if not success:
            logger.error(f"\nPipeline stopped at {stage_name}")
            break
        
        logger.info("")  # Add spacing
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("Pipeline Execution Summary")
    logger.info("="*60)
    
    for stage_name, result in results.items():
        logger.info(f"{stage_name:.<40} {result}")
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    logger.info(f"\nTotal Duration: {duration:.2f} seconds")
    logger.info("="*60 + "\n")
    
    # Check if all stages completed
    all_success = all("Success" in result for result in results.values())
    
    if all_success:
        logger.info("\n✓ Pipeline completed successfully!")
        logger.info("\nNext steps:")
        logger.info("1. Start MLflow server: mlflow server --host 127.0.0.1 --port 5000")
        logger.info("2. Open browser: http://127.0.0.1:5000")
        logger.info("3. View your tracked experiments and metrics\n")
        return 0
    else:
        logger.error("\n✗ Pipeline failed. Check logs above for details.\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
