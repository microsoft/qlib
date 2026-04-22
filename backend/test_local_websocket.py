#!/usr/bin/env python3
"""测试本地WebSocket通信功能"""

import asyncio
import websockets

async def test_local_websocket():
    """测试本地WebSocket连接"""
    task_id = "test-task-123"
    websocket_url = f"ws://localhost:8000/ws/train/{task_id}"
    
    print(f"Testing WebSocket connection to {websocket_url}...")
    
    try:
        # 连接到WebSocket
        async with websockets.connect(websocket_url) as websocket:
            print("✅ WebSocket connection established successfully!")
            
            # 发送一条测试消息
            test_message = "ping"
            print(f"📤 Sending message: {test_message}")
            await websocket.send(test_message)
            
            # 等待接收消息（1秒超时）
            try:
                # 这里设置超时是因为我们不期望立即收到消息
                response = await asyncio.wait_for(websocket.recv(), timeout=1)
                print(f"📥 Received response: {response}")
            except asyncio.TimeoutError:
                print("⏱️  No response received within timeout (expected for this test)")
            
            print("✅ WebSocket connection closed successfully!")
            return True
            
    except Exception as e:
        print(f"❌ WebSocket connection failed: {e}")
        return False

async def main():
    """主函数"""
    result = await test_local_websocket()
    if result:
        print("\n🎉 Local WebSocket test PASSED!")
    else:
        print("\n❌ Local WebSocket test FAILED!")

if __name__ == "__main__":
    asyncio.run(main())
