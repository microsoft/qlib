#!/usr/bin/env python3
"""
Test the BenchmarkService directly.
"""

import sys
import os

# Add the backend directory to Python path
sys.path.append('/home/qlib_t/backend')

from app.services.benchmark import BenchmarkService, BENCHMARK_DIR

print(f"Current directory: {os.getcwd()}")
print(f"Benchmark directory: {BENCHMARK_DIR}")
print(f"Benchmark directory exists: {os.path.exists(BENCHMARK_DIR)}")

try:
    # Call the get_benchmarks method directly
    benchmarks = BenchmarkService.get_benchmarks()
    print(f"Number of benchmarks found: {len(benchmarks)}")
    
    if benchmarks:
        print("Sample benchmarks:")
        for benchmark in benchmarks[:3]:  # Show first 3 benchmarks
            print(f"  ID: {benchmark['id']}")
            print(f"  Name: {benchmark['name']}")
            print(f"  Model: {benchmark['model']}")
            print(f"  File: {benchmark['file_name']}")
            print("  --")
    else:
        print("No benchmarks found. Let's debug further...")
        
        # Try to manually iterate through the directory
        if os.path.exists(BenchmarkService.BENCHMARK_DIR):
            print("Manual directory iteration:")
            for model_name in os.listdir(BenchmarkService.BENCHMARK_DIR):
                model_path = os.path.join(BenchmarkService.BENCHMARK_DIR, model_name)
                if os.path.isdir(model_path):
                    print(f"  Directory: {model_name}")
                    yaml_files = [f for f in os.listdir(model_path) if f.endswith('.yaml')]
                    print(f"    YAML files: {yaml_files}")
                    
                    # Try to read one yaml file
                    if yaml_files:
                        test_file = os.path.join(model_path, yaml_files[0])
                        print(f"    Testing file: {test_file}")
                        try:
                            with open(test_file, 'r', encoding='utf-8') as f:
                                content = f.read()
                            print(f"      Successfully read file: {len(content)} bytes")
                        except Exception as e:
                            print(f"      Error reading file: {e}")
except Exception as e:
    print(f"Error in get_benchmarks: {e}")
    import traceback
    traceback.print_exc()
