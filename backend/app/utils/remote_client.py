import aiohttp
import json
from typing import Dict, Any, Optional, Callable
from app.config import settings
import logging
import asyncio

logger = logging.getLogger(__name__)

class RemoteClient:
    """DDNS服务器客户端工具"""
    
    def __init__(self):
        self.server_url = settings.training_server_url
        self.timeout = settings.training_server_timeout
        self.max_retries = 3  # 最大重试次数
        self.base_retry_delay = 1.0  # 基础重试延迟（秒）
        # 远程训练服务器认证信息
        self.username = "idea"  # 远程训练服务器账号
        self.password = "moshou99"  # 远程训练服务器密码
    
    async def _make_request(self, method: str, url: str, **kwargs) -> Optional[Dict[str, Any]]:
        """通用请求方法，带有重试机制和详细的异常处理"""
        retries = 0
        
        while retries < self.max_retries:
            try:
                logger.info(f"Making {method} request to {url} (attempt {retries + 1}/{self.max_retries})")
                
                # 添加基本认证
                auth = aiohttp.BasicAuth(self.username, self.password)
                
                async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
                    async with getattr(session, method)(url, auth=auth, **kwargs) as response:
                        logger.info(f"Received response from {url}: {response.status}")
                        
                        if response.status == 200:
                            return await response.json()
                        elif response.status in [400, 401, 403, 404]:
                            # 客户端错误，不需要重试
                            logger.error(f"Client error from {url}: {response.status} - {await response.text()}")
                            return None
                        else:
                            # 服务器错误，需要重试
                            logger.warning(f"Server error from {url}: {response.status} - {await response.text()}")
            except asyncio.TimeoutError:
                logger.warning(f"Request to {url} timed out (attempt {retries + 1}/{self.max_retries})")
            except aiohttp.ClientConnectionError as e:
                logger.warning(f"Connection error to {url}: {e} (attempt {retries + 1}/{self.max_retries})")
            except aiohttp.ClientError as e:
                logger.warning(f"Client error to {url}: {e} (attempt {retries + 1}/{self.max_retries})")
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON response from {url}: {e}")
                return None
            except Exception as e:
                logger.error(f"Unexpected error making request to {url}: {e}")
                return None
            
            # 计算重试延迟（指数退避 + 抖动）
            retries += 1
            if retries < self.max_retries:
                retry_delay = self.base_retry_delay * (2 ** retries) + (retries * 0.5)  # 指数退避 + 抖动
                logger.info(f"Retrying request to {url} in {retry_delay:.2f} seconds...")
                await asyncio.sleep(retry_delay)
        
        logger.error(f"Failed to make {method} request to {url} after {self.max_retries} attempts")
        return None
    
    async def submit_task(self, task_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """提交任务到DDNS服务器"""
        url = f"{self.server_url}/api/train"
        headers = {"Content-Type": "application/json"}
        
        try:
            logger.info(f"Submitting task to {url} with data: {task_data}")
            return await self._make_request("post", url, json=task_data, headers=headers)
        except Exception as e:
            logger.error(f"Error submitting task: {e}")
            return None
    
    async def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """获取远程任务状态"""
        url = f"{self.server_url}/api/train/tasks/{task_id}"
        logger.info(f"Getting status for task {task_id} from {url}")
        return await self._make_request("get", url)
    
    async def cancel_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """取消远程任务"""
        url = f"{self.server_url}/api/train/tasks/{task_id}/cancel"
        logger.info(f"Cancelling task {task_id} at {url}")
        headers = {"Content-Type": "application/json"}
        return await self._make_request("post", url, headers=headers)
    
    async def get_task_results(self, task_id: str) -> Optional[Dict[str, Any]]:
        """获取远程任务结果"""
        # 先尝试直接获取结果
        results_url = f"{self.server_url}/api/train/tasks/{task_id}/results"
        logger.info(f"Getting results for task {task_id} from {results_url}")
        results = await self._make_request("get", results_url)
        
        if results:
            return {
                "task_id": task_id,
                "results": results
            }
        
        # 如果直接获取失败，尝试从任务状态中获取结果
        logger.info(f"Falling back to getting results from task status for task {task_id}")
        status = await self.get_task_status(task_id)
        if status:
            state = status.get("state", "")
            if state == "SUCCESS":
                return {
                    "task_id": task_id,
                    "results": status.get("result", {})
                }
            elif state in ["FAILED", "CANCELLED"]:
                return {
                    "task_id": task_id,
                    "results": {},
                    "status": state,
                    "error": status.get("error", "")
                }
        
        logger.warning(f"No results available for task {task_id}")
        return None
    
    async def health_check(self) -> bool:
        """检查远程服务器健康状态"""
        url = f"{self.server_url}/health"
        logger.info(f"Performing health check on {url}")
        result = await self._make_request("get", url)
        if result:
            logger.info(f"Health check passed for {self.server_url}")
            return True
        else:
            logger.error(f"Health check failed for {self.server_url}")
            return False
    
    def sync_health_check(self) -> bool:
        """同步版本的健康检查，用于系统启动时验证"""
        import asyncio
        try:
            return asyncio.run(self.health_check())
        except Exception as e:
            logger.error(f"Error during sync health check: {e}")
            return False
    
    async def server_status(self) -> Optional[Dict[str, Any]]:
        """获取远程服务器状态信息"""
        url = f"{self.server_url}/status"
        logger.info(f"Getting server status from {url}")
        return await self._make_request("get", url)
    
    async def connect_websocket(self, task_id: str, callback: Callable[[Dict[str, Any]], None]) -> None:
        """连接到WebSocket端点，接收实时训练更新"""
        # 将HTTP URL转换为WebSocket URL
        ws_url = self.server_url.replace("http://", "ws://").replace("https://", "wss://")
        ws_url = f"{ws_url}/ws/train/{task_id}"
        
        logger.info(f"Connecting to WebSocket endpoint: {ws_url}")
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.ws_connect(ws_url) as ws:
                    logger.info(f"WebSocket connected to {ws_url}")
                    async for msg in ws:
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            try:
                                data = json.loads(msg.data)
                                callback(data)
                            except json.JSONDecodeError:
                                logger.error(f"Failed to parse WebSocket message: {msg.data}")
                        elif msg.type == aiohttp.WSMsgType.CLOSED:
                            logger.info(f"WebSocket connection closed: {ws.close_code}")
                            break
                        elif msg.type == aiohttp.WSMsgType.ERROR:
                            logger.error(f"WebSocket error: {ws.exception()}")
                            break
        except Exception as e:
            logger.error(f"Error connecting to WebSocket: {e}")
            # 尝试重新连接
            await asyncio.sleep(5)
            await self.connect_websocket(task_id, callback)