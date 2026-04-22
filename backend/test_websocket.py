#!/usr/bin/env python3
"""测试WebSocket通信功能"""

from dotenv import load_dotenv
load_dotenv()

from app.utils.remote_client import RemoteClient
import asyncio

async def test_websocket_communication():
    """测试WebSocket通信"""
    client = RemoteClient()
    
    print("Testing WebSocket communication...")
    
    # 测试任务提交
    task_data = {
        "experiment_id": 1,
        "name": "Test Experiment",
        "config": {"test": "config"},
        "task_id": "test-task-123"
    }
    
    print("\n1. Submitting task...")
    result = await client.submit_task(task_data)
    print(f"   Task submission result: {result}")
    
    if result and "task_id" in result:
        task_id = result["task_id"]
        print(f"\n2. Connecting to WebSocket for task {task_id}...")
        
        # 定义WebSocket消息处理函数
        def handle_message(data):
            print(f"   Received WebSocket message: {data}")
        
        # 连接WebSocket（这个测试可能会超时，因为服务器可能没有实际的WebSocket端点）
        try:
            # 设置超时
            await asyncio.wait_for(client.connect_websocket(task_id, handle_message), timeout=10)
        except asyncio.TimeoutError:
            print("   WebSocket connection timed out (expected for this test)")
        except Exception as e:
            print(f"   WebSocket error: {e}")
    
    print("\nWebSocket communication test complete!")

if __name__ == "__main__":
    asyncio.run(test_websocket_communication())
