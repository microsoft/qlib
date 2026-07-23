#!/usr/bin/env python3
"""
Test script for TrainingClient with authentication to ddns.hoo.ink server
"""

import asyncio
import sys
import os

# Add the project root and backend directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'backend'))

import httpx
from backend.app.services.training_client import training_client

async def test_auth_training():
    """Test training API with authentication"""
    print(f"Testing training API at: {training_client.base_url}")
    
    # Create a new client with follow_redirects=True
    client = httpx.AsyncClient(
        base_url=training_client.base_url,
        timeout=training_client.timeout,
        headers={"Content-Type": "application/json"},
        follow_redirects=True
    )
    
    try:
        # Test 1: Health check (should not require auth)
        print("\n1. Testing health check...")
        response = await client.get("/health")
        print(f"   Status: {response.status_code}")
        print(f"   Response: {response.json()}")
        
        # Test 2: Try to get experiments with auth
        print("\n2. Testing experiments list with auth...")
        try:
            # First, try to login to get a token
            login_response = await client.post("/api/auth/login", json={
                "username": "admin",
                "password": "admin"
            })
            print(f"   Login status: {login_response.status_code}")
            
            if login_response.status_code == 200:
                login_data = login_response.json()
                token = login_data.get("access_token")
                print(f"   Login successful, got token: {token[:20]}...")
                
                # Use the token to get experiments
                experiments_response = await client.get(
                    "/api/experiments",
                    headers={"Authorization": f"Bearer {token}"}
                )
                print(f"   Experiments status: {experiments_response.status_code}")
                experiments = experiments_response.json()
                print(f"   Number of experiments: {len(experiments)}")
                if experiments:
                    print(f"   First experiment: {experiments[0]['name']}")
            else:
                print(f"   Login failed, using default credentials")
        except Exception as e:
            print(f"   Error with auth: {e}")
        
        return True
    except Exception as e:
        print(f"\nAPI test failed: {e}")
        return False
    finally:
        await client.aclose()

if __name__ == "__main__":
    asyncio.run(test_auth_training())
