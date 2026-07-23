#!/usr/bin/env python3
"""测试远程客户端功能"""

from dotenv import load_dotenv
load_dotenv()

from app.utils.remote_client import RemoteClient
import asyncio

async def test_remote_client():
    """测试远程客户端的各项功能"""
    client = RemoteClient()
    
    print("Testing Remote Client...")
    
    # 测试健康检查
    print("\n1. Testing health check...")
    try:
        health = await client.health_check()
        print(f"   Health check result: {health}")
    except Exception as e:
        print(f"   Health check failed: {e}")
    
    # 测试服务器状态
    print("\n2. Testing server status...")
    try:
        status = await client.server_status()
        print(f"   Server status result: {status}")
    except Exception as e:
        print(f"   Server status failed: {e}")
    
    # 测试任务提交
    print("\n3. Testing task submission...")
    try:
        task_data = {
            "experiment_id": 1,
            "name": "Test Experiment",
            "config": {"test": "config"},
            "task_id": 1
        }
        result = await client.submit_task(task_data)
        print(f"   Task submission result: {result}")
    except Exception as e:
        print(f"   Task submission failed: {e}")
    
    print("\nRemote Client Testing Complete!")

if __name__ == "__main__":
    asyncio.run(test_remote_client())
