#!/usr/bin/env python3
"""
测试远程训练服务器的不同API路径
"""

import os
import sys
import asyncio
import logging

# 将backend目录添加到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from app.utils.remote_client import RemoteClient

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_api_path(client, path, method="get", json_data=None):
    """测试单个API路径"""
    url = f"{client.server_url}{path}"
    logger.info(f"测试路径: {url} (方法: {method})")
    
    try:
        if method == "get":
            result = await client._make_request("get", url)
        elif method == "post":
            headers = {"Content-Type": "application/json"}
            result = await client._make_request("post", url, json=json_data, headers=headers)
        
        logger.info(f"  结果: {result}")
        return result
    except Exception as e:
        logger.error(f"  错误: {e}")
        return None

async def main():
    """主函数"""
    logger.info("开始测试远程训练服务器的API路径...")
    
    try:
        # 创建RemoteClient实例
        client = RemoteClient()
        logger.info(f"远程服务器URL: {client.server_url}")
        
        # 1. 测试健康检查（已知工作）
        logger.info("1. 正在执行健康检查...")
        health_result = await client.health_check()
        logger.info(f"健康检查结果: {health_result}")
        
        if not health_result:
            logger.error("健康检查失败，无法继续测试")
            return
        
        # 2. 尝试不同的API路径
        logger.info("2. 正在测试不同的API路径...")
        
        # 测试基本根路径
        await test_api_path(client, "/")
        await test_api_path(client, "/api")
        
        # 测试可能的训练相关路径
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
        
        # 尝试不同的训练路径
        train_paths = [
            ("post", "/api/train", test_task),
            ("post", "/train", test_task),
            ("post", "/api/v1/train", test_task),
            ("post", "/api/tasks", test_task),
            ("post", "/tasks", test_task),
        ]
        
        for method, path, data in train_paths:
            await test_api_path(client, path, method, data)
        
        # 3. 测试可能的API文档或状态路径
        logger.info("3. 正在测试可能的API文档或状态路径...")
        info_paths = [
            ("get", "/docs"),
            ("get", "/redoc"),
            ("get", "/openapi.json"),
            ("get", "/swagger.json"),
            ("get", "/status"),
            ("get", "/api/status"),
            ("get", "/info"),
            ("get", "/api/info"),
        ]
        
        for method, path in info_paths:
            await test_api_path(client, path, method)
        
        logger.info("所有测试完成!")
        
    except Exception as e:
        logger.error(f"测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
