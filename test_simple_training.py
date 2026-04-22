#!/usr/bin/env python3
"""
Simple test script for TrainingClient API calls to ddns.hoo.ink server
"""

import asyncio
import sys
import os

# Add the project root and backend directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'backend'))

from backend.app.services.training_client import training_client

async def test_simple_training():
    """Test simple training API endpoints"""
    print(f"Testing training API at: {training_client.base_url}")
    
    try:
        # Test 1: Health check
        print("\n1. Testing health check...")
        response = await training_client.client.get("/health")
        print(f"   Status: {response.status_code}")
        print(f"   Response: {response.json()}")
        
        # Test 2: Try to get experiments list
        print("\n2. Testing experiments list...")
        try:
            experiments_response = await training_client.client.get("/api/experiments")
            print(f"   Status: {experiments_response.status_code}")
            experiments = experiments_response.json()
            print(f"   Number of experiments: {len(experiments)}")
            if experiments:
                print(f"   First experiment: {experiments[0]['name']}")
        except Exception as e:
            print(f"   Error getting experiments: {e}")
        
        return True
    except Exception as e:
        print(f"\nAPI test failed: {e}")
        return False
    finally:
        await training_client.close()

if __name__ == "__main__":
    asyncio.run(test_simple_training())
