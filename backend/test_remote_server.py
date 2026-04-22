import asyncio
from app.utils.remote_client import RemoteClient

async def test_remote_server():
    client = RemoteClient()
    print("Testing remote server health...")
    
    # Test health check
    health_status = await client.health_check()
    print(f"Health check result: {health_status}")
    
    # Test server status
    server_status = await client.server_status()
    print(f"Server status: {server_status}")

if __name__ == "__main__":
    asyncio.run(test_remote_server())