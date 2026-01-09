#!/usr/bin/env python
"""
MLflow Server Setup & Launch Script
This script helps you start and configure MLflow for tracking experiments.

Usage:
    python start_mlflow_server.py [--host 127.0.0.1] [--port 5000]
"""

import subprocess
import sys
import logging
import argparse
import time
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_mlflow_backend(backend_store_uri: str = "mlruns") -> bool:
    """
    Set up MLflow backend storage.
    
    Args:
        backend_store_uri: Path for storing MLflow data (default: ./mlruns)
        
    Returns:
        True if setup successful
    """
    try:
        Path(backend_store_uri).mkdir(exist_ok=True)
        logger.info(f"MLflow backend directory ready: {backend_store_uri}")
        return True
    except Exception as e:
        logger.error(f"Failed to set up MLflow backend: {e}")
        return False


def start_mlflow_server(
    host: str = "127.0.0.1",
    port: int = 5000,
    backend_store_uri: str = "mlruns",
    default_artifact_root: str = None
) -> None:
    """
    Start MLflow tracking server.
    
    Args:
        host: Server host (default: 127.0.0.1)
        port: Server port (default: 5000)
        backend_store_uri: Backend storage URI
        default_artifact_root: Default artifact storage location
    """
    
    logger.info("\n" + "="*60)
    logger.info("Starting MLflow Tracking Server")
    logger.info("="*60 + "\n")
    
    # Set up backend
    if not setup_mlflow_backend(backend_store_uri):
        logger.error("Failed to set up MLflow backend")
        return
    
    # Build command
    cmd = ["mlflow", "server", "--host", host, "--port", str(port)]
    
    if default_artifact_root:
        cmd.extend(["--default-artifact-root", default_artifact_root])
    
    logger.info(f"Command: {' '.join(cmd)}\n")
    logger.info(f"MLflow UI will be available at: http://{host}:{port}")
    logger.info("\nPress Ctrl+C to stop the server\n")
    logger.info("="*60 + "\n")
    
    try:
        subprocess.run(cmd, check=False)
    except KeyboardInterrupt:
        logger.info("\n\nMLflow server stopped by user")
    except FileNotFoundError:
        logger.error("\nMLflow is not installed. Install it with: pip install mlflow")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\nError starting MLflow server: {e}")
        sys.exit(1)


def check_mlflow_installed() -> bool:
    """Check if MLflow is installed."""
    try:
        import mlflow
        logger.info(f"✓ MLflow {mlflow.__version__} is installed")
        return True
    except ImportError:
        logger.error("✗ MLflow is not installed")
        logger.error("Install it with: pip install mlflow")
        return False


def main():
    """Main entry point."""
    
    parser = argparse.ArgumentParser(
        description="Start MLflow tracking server for experiment tracking"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Server host (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5000,
        help="Server port (default: 5000)"
    )
    parser.add_argument(
        "--backend-store",
        type=str,
        default="mlruns",
        help="Backend storage URI (default: mlruns)"
    )
    parser.add_argument(
        "--artifact-root",
        type=str,
        default=None,
        help="Default artifact root directory"
    )
    
    args = parser.parse_args()
    
    # Check if MLflow is installed
    if not check_mlflow_installed():
        sys.exit(1)
    
    logger.info("\n" + "="*60)
    logger.info("MLflow Server Configuration")
    logger.info("="*60)
    logger.info(f"Host: {args.host}")
    logger.info(f"Port: {args.port}")
    logger.info(f"Backend Storage: {args.backend_store}")
    if args.artifact_root:
        logger.info(f"Artifact Root: {args.artifact_root}")
    logger.info("="*60 + "\n")
    
    # Start server
    start_mlflow_server(
        host=args.host,
        port=args.port,
        backend_store_uri=args.backend_store,
        default_artifact_root=args.artifact_root
    )


if __name__ == "__main__":
    main()
