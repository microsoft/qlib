#!/usr/bin/env python3
"""
Qlib Configuration Module
=========================

This module provides centralized configuration for Qlib data paths and initialization.
Update the DATA_PATH variable to point to your Qlib data location.

Author: AI Assistant
Date: 2025-09-06
"""

import os
from pathlib import Path

# =============================================================================
# CONFIGURATION SETTINGS - MODIFY THESE AS NEEDED
# =============================================================================

# Primary data path (workspace location)
DATA_PATH = "/workspace/qlib/data/qlib_data/cn_data"

# Backup data path (original location) 
BACKUP_DATA_PATH = "~/.qlib/qlib_data/cn_data"

# Output directory for analysis results
OUTPUT_DIR = "/workspace/qlib/qlib_analysis_output"

# Region setting
REGION = "cn"  # China market

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_data_path():
    """
    Get the appropriate data path, checking if primary path exists.
    Returns the workspace path if available, otherwise falls back to home directory.
    """
    primary_path = Path(DATA_PATH)
    backup_path = Path(os.path.expanduser(BACKUP_DATA_PATH))
    
    if primary_path.exists():
        return str(primary_path)
    elif backup_path.exists():
        print(f"⚠️  Primary path not found, using backup: {backup_path}")
        return str(backup_path)
    else:
        raise FileNotFoundError(f"Qlib data not found in either location:\n- {primary_path}\n- {backup_path}")

def initialize_qlib():
    """
    Initialize Qlib with the configured data path.
    
    Returns:
        bool: True if initialization successful, False otherwise
    """
    try:
        import qlib
        from qlib.constant import REG_CN
        
        # Get appropriate data path
        data_path = get_data_path()
        
        # Initialize Qlib
        qlib.init(provider_uri=data_path, region=REG_CN)
        
        print("✅ Qlib initialized successfully!")
        print(f"📁 Data path: {data_path}")
        return True
        
    except Exception as e:
        print(f"❌ Failed to initialize Qlib: {e}")
        return False

def ensure_output_dir():
    """Ensure output directory exists."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    return OUTPUT_DIR

def get_sample_stocks(market_segment='csi300', limit=5):
    """
    Get sample stocks from specified market segment.
    
    Args:
        market_segment (str): Market segment (csi300, csi500, etc.)
        limit (int): Number of stocks to return
        
    Returns:
        list: List of stock symbols
    """
    try:
        from qlib.data import D
        
        if not initialize_qlib():
            return []
            
        instruments = D.instruments(market_segment)
        stocks = D.list_instruments(
            instruments=instruments, 
            start_time='2024-01-01', 
            end_time='2025-09-06', 
            as_list=True
        )
        
        return stocks[:limit] if stocks else []
        
    except Exception as e:
        print(f"Failed to get sample stocks: {e}")
        return []

def check_data_health():
    """
    Check if Qlib data is accessible and healthy.
    
    Returns:
        dict: Health check results
    """
    try:
        from qlib.data import D
        
        if not initialize_qlib():
            return {'status': 'failed', 'message': 'Initialization failed'}
        
        # Test basic functionality
        calendar = D.calendar(start_time='2024-01-01', end_time='2024-01-02')
        instruments = D.instruments('csi300')
        
        stocks = D.list_instruments(
            instruments=instruments, 
            start_time='2024-01-01', 
            end_time='2024-01-02', 
            as_list=True
        )
        
        if len(stocks) == 0:
            return {'status': 'warning', 'message': 'No instruments found'}
        
        # Try to get sample data
        sample_data = D.features(
            instruments=[stocks[0]], 
            fields=['$close'],
            start_time='2024-09-01', 
            end_time='2024-09-06'
        )
        
        return {
            'status': 'healthy',
            'message': 'All checks passed',
            'data_path': get_data_path(),
            'calendar_days': len(calendar),
            'sample_stocks': len(stocks),
            'sample_data_shape': sample_data.shape if not sample_data.empty else (0, 0)
        }
        
    except Exception as e:
        return {
            'status': 'error', 
            'message': f'Health check failed: {str(e)}'
        }

# =============================================================================
# CONFIGURATION DISPLAY
# =============================================================================

def print_config():
    """Print current configuration settings."""
    print("\n" + "="*60)
    print("🔧 QLIB CONFIGURATION")
    print("="*60)
    print(f"📁 Primary Data Path: {DATA_PATH}")
    print(f"📁 Backup Data Path:  {BACKUP_DATA_PATH}")
    print(f"📊 Output Directory:  {OUTPUT_DIR}")
    print(f"🌏 Region:           {REGION}")
    print(f"✅ Current Data Path: {get_data_path() if Path(DATA_PATH).exists() or Path(os.path.expanduser(BACKUP_DATA_PATH)).exists() else 'NOT FOUND'}")
    print("="*60)

if __name__ == "__main__":
    # Run configuration check when script is executed directly
    print_config()
    
    # Run health check
    print("\n🏥 Running health check...")
    health = check_data_health()
    print(f"Status: {health['status']}")
    print(f"Message: {health['message']}")
    
    if health['status'] == 'healthy':
        print(f"📊 Calendar days: {health['calendar_days']}")
        print(f"🏢 Sample stocks: {health['sample_stocks']}")
        print(f"📈 Sample data shape: {health['sample_data_shape']}")
        
    print("\n🎯 Sample stocks from CSI 300:")
    samples = get_sample_stocks('csi300', 5)
    for stock in samples:
        print(f"   {stock}")