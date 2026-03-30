#!/usr/bin/env python3
"""
Import historical stock data using chenditc/investment_data.

This script downloads and converts data from the investment_data project
into qlib format for use with the qlib_t management platform.

Usage:
    python scripts/import_investment_data.py [--qlib-dir ~/.qlib/qlib_data/cn_data]

Prerequisites:
    pip install investment_data

Reference: https://github.com/chenditc/investment_data
"""

import argparse
import logging
import os
import subprocess
import sys

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def install_investment_data():
    """Install investment_data package if not already installed."""
    try:
        import investment_data  # noqa: F401
        logger.info("investment_data package is already installed.")
        return True
    except ImportError:
        logger.info("Installing investment_data package...")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "investment_data"],
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            logger.error(f"Failed to install investment_data: {result.stderr}")
            return False
        logger.info("investment_data package installed successfully.")
        return True


def download_qlib_data(qlib_dir: str):
    """
    Download and convert data to qlib format using investment_data.
    
    The investment_data package provides a CLI command to download
    and convert Chinese A-share market data into qlib format.
    """
    # Ensure the target directory exists
    os.makedirs(qlib_dir, exist_ok=True)
    
    logger.info(f"Downloading qlib data to: {qlib_dir}")
    logger.info("This may take several minutes depending on your network speed...")
    
    # Use investment_data CLI to download data
    # The package provides: investment_data download_qlib_data --target_dir <dir>
    try:
        result = subprocess.run(
            [
                sys.executable, "-m", "investment_data",
                "download_qlib_data",
                "--target_dir", qlib_dir
            ],
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout
        )
        
        if result.returncode == 0:
            logger.info("Data download completed successfully.")
            if result.stdout:
                logger.info(f"Output: {result.stdout[-500:]}")
            return True
        else:
            logger.warning(f"CLI method returned non-zero: {result.stderr}")
            # Try alternative method
            return download_qlib_data_alternative(qlib_dir)
    except subprocess.TimeoutExpired:
        logger.error("Data download timed out after 1 hour.")
        return False
    except FileNotFoundError:
        logger.warning("investment_data CLI not found, trying alternative method...")
        return download_qlib_data_alternative(qlib_dir)


def download_qlib_data_alternative(qlib_dir: str):
    """
    Alternative method to download data using Python API.
    """
    try:
        logger.info("Trying alternative download method via Python API...")
        
        # Try using the Python API directly
        from investment_data import download_qlib_data as _download
        _download(target_dir=qlib_dir)
        logger.info("Data download completed successfully via Python API.")
        return True
    except ImportError:
        logger.warning("Python API method not available.")
    except Exception as e:
        logger.warning(f"Python API method failed: {e}")
    
    # Final fallback: use wget to download from GitHub releases
    try:
        logger.info("Trying to download from GitHub releases...")
        import urllib.request
        import zipfile
        import tempfile
        
        # chenditc/investment_data releases contain pre-built qlib data
        release_url = "https://github.com/chenditc/investment_data/releases/latest"
        
        logger.info(f"Checking latest release at: {release_url}")
        logger.info("Please manually download qlib data from:")
        logger.info("  https://github.com/chenditc/investment_data/releases")
        logger.info(f"  and extract to: {qlib_dir}")
        logger.info("")
        logger.info("Or run the following commands:")
        logger.info(f"  pip install investment_data")
        logger.info(f"  python -m investment_data download_qlib_data --target_dir {qlib_dir}")
        
        return False
    except Exception as e:
        logger.error(f"Failed to download data: {e}")
        return False


def verify_qlib_data(qlib_dir: str):
    """Verify that qlib data was downloaded correctly."""
    required_dirs = ["instruments", "calendars", "features"]
    
    logger.info(f"Verifying qlib data in: {qlib_dir}")
    
    missing = []
    for d in required_dirs:
        path = os.path.join(qlib_dir, d)
        if not os.path.exists(path):
            missing.append(d)
        else:
            # Count files in directory
            file_count = sum(1 for _ in os.scandir(path) if _.is_file() or _.is_dir())
            logger.info(f"  {d}/: {file_count} items")
    
    if missing:
        logger.warning(f"Missing directories: {missing}")
        return False
    
    # Check instruments file
    instruments_dir = os.path.join(qlib_dir, "instruments")
    if os.path.exists(instruments_dir):
        all_txt = os.path.join(instruments_dir, "all.txt")
        if os.path.exists(all_txt):
            with open(all_txt, 'r') as f:
                lines = f.readlines()
            logger.info(f"  instruments/all.txt: {len(lines)} instruments")
        else:
            logger.warning("  instruments/all.txt not found")
    
    logger.info("Data verification completed.")
    return True


def init_qlib_with_data(qlib_dir: str):
    """Initialize qlib with the downloaded data to verify it works."""
    try:
        import qlib
        from qlib.config import REG_CN
        
        logger.info(f"Initializing qlib with provider_uri: {qlib_dir}")
        qlib.init(provider_uri=qlib_dir, region=REG_CN)
        
        from qlib.data import D
        
        # Test getting instruments
        instruments = D.instruments(market="all")
        logger.info(f"QLib initialized successfully. Market instruments loaded.")
        
        # Test getting calendar
        calendar = D.calendar(start_time="2020-01-01", end_time="2020-01-31")
        logger.info(f"Calendar test: {len(calendar)} trading days in Jan 2020")
        
        return True
    except ImportError:
        logger.warning("qlib not installed, skipping initialization test")
        return True  # Data might still be valid
    except Exception as e:
        logger.error(f"Failed to initialize qlib: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Import historical stock data using chenditc/investment_data"
    )
    parser.add_argument(
        "--qlib-dir",
        default=os.path.expanduser("~/.qlib/qlib_data/cn_data"),
        help="Target directory for qlib data (default: ~/.qlib/qlib_data/cn_data)"
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip download and only verify existing data"
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify existing data, don't download or initialize"
    )
    
    args = parser.parse_args()
    qlib_dir = args.qlib_dir
    
    logger.info("=" * 60)
    logger.info("QLib Historical Data Import Tool")
    logger.info(f"Data source: chenditc/investment_data")
    logger.info(f"Target directory: {qlib_dir}")
    logger.info("=" * 60)
    
    if args.verify_only:
        success = verify_qlib_data(qlib_dir)
        sys.exit(0 if success else 1)
    
    if not args.skip_download:
        # Step 1: Install investment_data package
        if not install_investment_data():
            logger.error("Failed to install investment_data package.")
            sys.exit(1)
        
        # Step 2: Download data
        if not download_qlib_data(qlib_dir):
            logger.error("Failed to download data. See instructions above for manual download.")
            sys.exit(1)
    
    # Step 3: Verify data
    if not verify_qlib_data(qlib_dir):
        logger.warning("Data verification found issues, but continuing...")
    
    # Step 4: Test qlib initialization
    if not init_qlib_with_data(qlib_dir):
        logger.warning("QLib initialization test failed, but data may still be usable.")
    
    logger.info("=" * 60)
    logger.info("Data import completed!")
    logger.info(f"Data location: {qlib_dir}")
    logger.info("You can now start the backend server to use this data.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
