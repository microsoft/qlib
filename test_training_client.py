#!/usr/bin/env python3
"""
Test script for TrainingClient connection to ddns.hoo.ink server
"""

import asyncio
import sys
import os

# Add the project root and backend directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'backend'))

from backend.app.services.training_client import training_client

async def test_connection():
    """Test connection to training server"""
    print(f"Testing connection to training server: {training_client.base_url}")
    
    try:
        # Test a simple GET request to check connectivity
        # Note: This assumes there's a health check endpoint or similar
        response = await training_client.client.get("/")
        print(f"Response status: {response.status_code}")
        print(f"Response content: {response.text[:200]}...")
        return True
    except Exception as e:
        print(f"Connection test failed: {e}")
        return False
    finally:
        await training_client.close()

if __name__ == "__main__":
    asyncio.run(test_connection())
