#!/usr/bin/env python3
"""
Test the fixed benchmark directory path and access.
"""

import os

# 修复后的路径计算
current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BENCHMARK_DIR = os.path.join(current_dir, "..", "examples", "benchmarks")

print(f"Current file path: {os.path.abspath(__file__)}")
print(f"Current directory: {current_dir}")
print(f"Benchmark directory: {BENCHMARK_DIR}")
print(f"Benchmark directory exists: {os.path.exists(BENCHMARK_DIR)}")
print(f"Absolute path: {os.path.abspath(BENCHMARK_DIR)}")

if os.path.exists(BENCHMARK_DIR):
    print(f"Benchmark directory contents:")
    for item in os.listdir(BENCHMARK_DIR):
        item_path = os.path.join(BENCHMARK_DIR, item)
        if os.path.isdir(item_path):
            print(f"  Directory: {item}")
            # 检查是否有YAML文件
            yaml_files = [f for f in os.listdir(item_path) if f.endswith('.yaml')]
            print(f"    YAML files: {yaml_files[:2]}... ({len(yaml_files)} total)")
        else:
            print(f"  File: {item}")
