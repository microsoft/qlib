#!/usr/bin/env python3
"""
详细测试远程训练服务器的训练API
"""

import os
import sys
import asyncio
import logging
import aiohttp

# 将backend目录添加到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from app.utils.remote_client import RemoteClient

# 配置详细日志
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_request_with_session(session, method, url, auth, **kwargs):
    """使用会话测试单个请求"""
    logger.debug(f"测试请求: {method} {url}")
    logger.debug(f"请求参数: {kwargs}")
    
    try:
        async with getattr(session, method)(url, auth=auth, **kwargs) as response:
            logger.debug(f"响应状态码: {response.status}")
            logger.debug(f"响应头: {response.headers}")
            response_text = await response.text()
            logger.debug(f"响应内容: {response_text}")
            return {
                "status": response.status,
                "headers": dict(response.headers),
                "content": response_text
            }
    except Exception as e:
        logger.error(f"请求错误: {e}")
        import traceback
        traceback.print_exc()
        return None

async def main():
    """主函数"""
    logger.info("开始详细测试远程训练服务器的训练API...")
    
    try:
        # 创建RemoteClient实例
        client = RemoteClient()
        logger.info(f"远程服务器URL: {client.server_url}")
        logger.info(f"账号: {client.username}")
        logger.info(f"密码: {'*' * len(client.password)}")
        
        # 1. 测试健康检查（已知工作）
        logger.info("1. 正在执行健康检查...")
        health_result = await client.health_check()
        logger.info(f"健康检查结果: {health_result}")
        
        if not health_result:
            logger.error("健康检查失败，无法继续测试")
            return
        
        # 2. 使用aiohttp直接测试训练API
        logger.info("2. 正在使用aiohttp直接测试训练API...")
        
        # 测试基本认证
        auth = aiohttp.BasicAuth(client.username, client.password)
        
        # 测试不同的训练路径
        test_task = {
            "experiment_id": "test_exp_123",
            "name": "Test Experiment",
            "config": {
                "model": "LightGBM",
                "params": {
                    "learning_rate": 0.1,
                    "n_estimators": 100
                }
            },
            "task_id": "test_task_123"
        }
        
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
            # 测试根路径
            logger.info("  测试根路径...")
            await test_request_with_session(session, "get", f"{client.server_url}/", auth)
            
            # 测试/api路径
            logger.info("  测试/api路径...")
            await test_request_with_session(session, "get", f"{client.server_url}/api", auth)
            
            # 测试/api/train路径（POST）
            logger.info("  测试/api/train路径（POST）...")
            train_result = await test_request_with_session(
                session, 
                "post", 
                f"{client.server_url}/api/train", 
                auth,
                json=test_task,
                headers={"Content-Type": "application/json"}
            )
            
            # 测试/api/train路径（GET）
            logger.info("  测试/api/train路径（GET）...")
            await test_request_with_session(session, "get", f"{client.server_url}/api/train", auth)
            
            # 测试/api/train/tasks路径（POST）
            logger.info("  测试/api/train/tasks路径（POST）...")
            tasks_result = await test_request_with_session(
                session, 
                "post", 
                f"{client.server_url}/api/train/tasks", 
                auth,
                json=test_task,
                headers={"Content-Type": "application/json"}
            )
            
            # 测试/api/train/tasks路径（GET）
            logger.info("  测试/api/train/tasks路径（GET）...")
            await test_request_with_session(session, "get", f"{client.server_url}/api/train/tasks", auth)
        
        logger.info("所有测试完成!")
        
    except Exception as e:
        logger.error(f"测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
