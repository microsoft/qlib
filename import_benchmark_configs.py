#!/usr/bin/env python3
"""
Import benchmark configuration files into the system as experiment templates.
"""

import os
import sys
import yaml
import glob
from datetime import datetime

# Add the backend directory to Python path
sys.path.append('/home/qlib_t/backend')

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.models.config import Config, ConfigType

def collect_benchmark_configs(base_dir):
    """
    Collect all workflow_config_*.yaml files from benchmarks directory.
    """
    configs = []
    
    for model_dir in os.listdir(base_dir):
        model_path = os.path.join(base_dir, model_dir)
        if not os.path.isdir(model_path):
            continue
        
        config_files = glob.glob(os.path.join(model_path, "workflow_config_*.yaml"))
        
        for config_file in config_files:
            try:
                with open(config_file, 'r') as f:
                    config_content = f.read()
                
                filename = os.path.basename(config_file)
                name_parts = filename[len("workflow_config_"):-5].split('_')
                
                model_name = model_dir.lower()
                dataset_name = "alpha158"
                benchmark_name = "csi300"
                
                # Try to extract dataset and benchmark from filename
                for part in name_parts[1:]:
                    lower_part = part.lower()
                    if lower_part in ["alpha158", "alpha360"]:
                        dataset_name = lower_part
                    elif lower_part in ["csi300", "csi500"]:
                        benchmark_name = lower_part
                
                # Handle special cases
                if "full" in filename.lower():
                    benchmark_name = f"{benchmark_name}_full"
                elif "multi_freq" in filename.lower():
                    benchmark_name = f"{benchmark_name}_multi_freq"
                elif "early_stop" in filename.lower():
                    benchmark_name = f"{benchmark_name}_early_stop"
                elif "configurable_dataset" in filename.lower():
                    benchmark_name = f"{benchmark_name}_configurable"
                
                configs.append((config_file, config_content, model_name, dataset_name, benchmark_name))
                
            except Exception as e:
                print(f"Error processing {config_file}: {e}")
    
    return configs

def import_configs():
    """
    Import collected benchmark configs into the database.
    """
    # Database connection
    DATABASE_URL = "sqlite:////home/qlib_t/backend/test.db"  # Absolute path to the backend database
    engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    
    # Collect configs
    BASE_DIR = "/home/qlib_t/examples/benchmarks"
    configs = collect_benchmark_configs(BASE_DIR)
    
    print(f"Found {len(configs)} configs to import")
    
    # Import each config
    db = SessionLocal()
    try:
        for file_path, config_content, model_name, dataset_name, benchmark_name in configs:
            # Generate unique name
            config_name = f"{model_name}_{dataset_name}_{benchmark_name}"
            
            # Check if config already exists
            existing_config = db.query(Config).filter(Config.name == config_name).first()
            if existing_config:
                print(f"Config {config_name} already exists, skipping")
                continue
            
            # Create config description
            description = f"Benchmark config for {model_name} model on {dataset_name} dataset with {benchmark_name} benchmark"
            
            # Create config object
            new_config = Config(
                name=config_name,
                description=description,
                content=config_content,
                type=ConfigType.EXPERIMENT_TEMPLATE
            )
            
            # Add to database
            db.add(new_config)
            print(f"Imported config: {config_name}")
        
        # Commit all changes
        db.commit()
        print(f"Successfully imported {len(configs)} configs")
        
    except Exception as e:
        db.rollback()
        print(f"Error importing configs: {e}")
        raise
    finally:
        db.close()

if __name__ == "__main__":
    import_configs()
