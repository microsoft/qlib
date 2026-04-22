#!/usr/bin/env python3
"""
Collect benchmark configuration files from benchmarks directory.
"""

import os
import yaml
import glob

def collect_benchmark_configs(base_dir):
    """
    Collect all workflow_config_*.yaml files from benchmarks directory.
    
    Args:
        base_dir: Base directory to search for benchmark configs
        
    Returns:
        List of tuples: (file_path, config_content, model_name, dataset_name)
    """
    configs = []
    
    # Iterate through all subdirectories in benchmarks
    for model_dir in os.listdir(base_dir):
        model_path = os.path.join(base_dir, model_dir)
        if not os.path.isdir(model_path):
            continue
        
        # Find all workflow_config_*.yaml files
        config_files = glob.glob(os.path.join(model_path, "workflow_config_*.yaml"))
        
        for config_file in config_files:
            try:
                # Read and parse the config file
                with open(config_file, 'r') as f:
                    config_content = f.read()
                
                # Parse the YAML to get dataset information
                yaml_data = yaml.safe_load(config_content)
                
                # Extract model name and dataset name from filename
                filename = os.path.basename(config_file)
                # Remove "workflow_config_" prefix and ".yaml" suffix
                name_parts = filename[len("workflow_config_"):-5].split('_')
                
                model_name = model_dir.lower()
                dataset_name = "alpha158"  # Default value
                
                # Try to extract dataset name from filename or config
                if len(name_parts) > 1:
                    # Check if the second part is a dataset name
                    if name_parts[1].lower() in ["alpha158", "alpha360"]:
                        dataset_name = name_parts[1].lower()
                
                configs.append((config_file, config_content, model_name, dataset_name))
                print(f"Found config: {config_file}")
                
            except Exception as e:
                print(f"Error processing {config_file}: {e}")
    
    return configs

if __name__ == "__main__":
    BASE_DIR = "/home/qlib_t/examples/benchmarks"
    configs = collect_benchmark_configs(BASE_DIR)
    print(f"\nTotal configs found: {len(configs)}")
    
    # Print summary
    for i, (file_path, _, model_name, dataset_name) in enumerate(configs):
        print(f"{i+1}. {model_name} - {dataset_name}: {file_path}")
