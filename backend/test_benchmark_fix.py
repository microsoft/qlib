#!/usr/bin/env python3

# Test script to verify the benchmark service fix
import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.abspath('.'))

from app.services.benchmark import BenchmarkService

def test_benchmark_service():
    """Test the benchmark service to verify it handles missing directories gracefully"""
    print("Testing BenchmarkService.get_benchmarks()...")
    
    try:
        benchmarks = BenchmarkService.get_benchmarks()
        print(f"✓ Success! Returned {len(benchmarks)} benchmarks")
        print(f"✓ No exception raised when accessing missing directory")
        return True
    except Exception as e:
        print(f"✗ Failed! Exception raised: {e}")
        return False

if __name__ == "__main__":
    success = test_benchmark_service()
    sys.exit(0 if success else 1)
