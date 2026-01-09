#!/usr/bin/env python
"""
Example: Verify MLflow Tracking Setup
This script helps you verify that MLflow is properly configured and tracking.

Usage:
    python verify_mlflow_setup.py
"""

import os
import sys
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def check_mlflow_installed():
    """Check if MLflow is installed."""
    try:
        import mlflow
        logger.info(f"✓ MLflow {mlflow.__version__} is installed")
        return True
    except ImportError:
        logger.error("✗ MLflow not installed. Run: pip install mlflow")
        return False


def check_pipeline_files():
    """Check if all pipeline files have MLflow tracking."""
    files_to_check = [
        ("src/data/Data_ingestion.py", "Data Ingestion"),
        ("src/data/data_preprocessing.py", "Data Preprocessing"),
        ("src/features/feature_engineering.py", "Feature Engineering"),
        ("src/models/model_building.py", "Model Building"),
        ("src/models/model_evaluation.py", "Model Evaluation"),
    ]
    
    logger.info("\n📋 Checking Pipeline Files:")
    logger.info("=" * 60)
    
    all_good = True
    for filepath, stage_name in files_to_check:
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                content = f.read()
                has_mlflow = "import mlflow" in content
                has_start_run = "mlflow.start_run()" in content
                
                status = "✓" if (has_mlflow and has_start_run) else "✗"
                logger.info(f"{status} {stage_name:.<40} {filepath}")
                
                if not has_mlflow:
                    logger.warning(f"  - Missing: 'import mlflow'")
                    all_good = False
                if not has_start_run:
                    logger.warning(f"  - Missing: 'mlflow.start_run()'")
                    all_good = False
        else:
            logger.warning(f"✗ {stage_name:.<40} FILE NOT FOUND")
            all_good = False
    
    return all_good


def check_utility_module():
    """Check if MLflow tracking utility exists."""
    logger.info("\n📦 Checking Utility Module:")
    logger.info("=" * 60)
    
    if os.path.exists("src/mlflow_tracking.py"):
        logger.info("✓ MLflow Tracking Utility:.......... src/mlflow_tracking.py")
        return True
    else:
        logger.error("✗ MLflow Tracking Utility not found")
        return False


def check_helper_scripts():
    """Check if helper scripts exist."""
    logger.info("\n🔧 Checking Helper Scripts:")
    logger.info("=" * 60)
    
    scripts = [
        ("run_pipeline_with_tracking.py", "Pipeline Runner"),
        ("start_mlflow_server.py", "MLflow Server"),
    ]
    
    all_good = True
    for script, description in scripts:
        if os.path.exists(script):
            logger.info(f"✓ {description:.<40} {script}")
        else:
            logger.error(f"✗ {description:.<40} {script} NOT FOUND")
            all_good = False
    
    return all_good


def check_documentation():
    """Check if documentation files exist."""
    logger.info("\n📚 Checking Documentation:")
    logger.info("=" * 60)
    
    docs = [
        ("QUICK_REFERENCE.md", "Quick Reference"),
        ("MLFLOW_TRACKING.md", "Detailed Guide"),
        ("TRACKING_IMPLEMENTATION_SUMMARY.md", "Implementation Summary"),
        ("IMPLEMENTATION_COMPLETE.md", "Completion Status"),
    ]
    
    all_good = True
    for doc, description in docs:
        if os.path.exists(doc):
            logger.info(f"✓ {description:.<40} {doc}")
        else:
            logger.warning(f"⚠ {description:.<40} {doc} NOT FOUND")
            all_good = False
    
    return all_good


def test_mlflow_locally():
    """Test MLflow with a simple experiment."""
    try:
        import mlflow
        import tempfile
        import shutil
        
        logger.info("\n🧪 Testing MLflow Locally:")
        logger.info("=" * 60)
        
        # Create temp directory for test
        temp_dir = tempfile.mkdtemp()
        
        try:
            mlflow.set_tracking_uri(f"file:{temp_dir}")
            mlflow.set_experiment("test_experiment")
            
            with mlflow.start_run() as run:
                mlflow.log_param("test_param", "value")
                mlflow.log_metric("test_metric", 0.95)
                logger.info(f"✓ MLflow tracking works!")
                logger.info(f"  - Run ID: {run.info.run_id}")
                logger.info(f"  - Experiment: test_experiment")
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
        
        return True
        
    except Exception as e:
        logger.error(f"✗ MLflow test failed: {e}")
        return False


def print_summary(results):
    """Print verification summary."""
    logger.info("\n" + "=" * 60)
    logger.info("VERIFICATION SUMMARY")
    logger.info("=" * 60)
    
    all_passed = all(results.values())
    
    for check_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        logger.info(f"{status} - {check_name}")
    
    logger.info("=" * 60)
    
    if all_passed:
        logger.info("\n✓ All checks passed! MLflow tracking is properly set up.\n")
        logger.info("Next steps:")
        logger.info("1. Start MLflow server: python start_mlflow_server.py")
        logger.info("2. Run pipeline: python run_pipeline_with_tracking.py")
        logger.info("3. View experiments: http://127.0.0.1:5000")
        logger.info("")
    else:
        logger.error("\n✗ Some checks failed. Please review the issues above.\n")
    
    return all_passed


def main():
    """Run all verification checks."""
    logger.info("\n" + "=" * 60)
    logger.info("MLflow Experiment Tracking - Verification Tool")
    logger.info("=" * 60)
    
    results = {
        "MLflow Installation": check_mlflow_installed(),
        "Pipeline Files": check_pipeline_files(),
        "Utility Module": check_utility_module(),
        "Helper Scripts": check_helper_scripts(),
        "Documentation": check_documentation(),
        "MLflow Functionality": test_mlflow_locally(),
    }
    
    all_passed = print_summary(results)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
