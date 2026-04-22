#!/usr/bin/env python3
"""
测试远程训练服务器的训练API
"""

import os
import sys
import asyncio
import logging
import json

# 将backend目录添加到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from app.utils.remote_client import RemoteClient

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def main():
    """主函数"""
    logger.info("开始测试远程训练服务器的训练API...")
    
    try:
        # 创建RemoteClient实例
        client = RemoteClient()
        logger.info(f"远程服务器URL: {client.server_url}")
        logger.info(f"账号: {client.username}")
        
        # 1. 测试健康检查
        logger.info("1. 正在执行健康检查...")
        health_result = await client.health_check()
        logger.info(f"健康检查结果: {health_result}")
        
        if not health_result:
            logger.error("健康检查失败，无法继续测试")
            return
        
        # 2. 测试训练任务提交API
        logger.info("2. 正在测试训练任务提交API...")
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
        
        submit_result = await client.submit_task(test_task)
        logger.info(f"任务提交结果: {submit_result}")
        
        if submit_result:
            # 3. 测试任务状态查询API
            remote_task_id = submit_result.get("task_id")
            if remote_task_id:
                logger.info(f"3. 正在测试任务状态查询API，任务ID: {remote_task_id}...")
                status_result = await client.get_task_status(remote_task_id)
                logger.info(f"任务状态结果: {status_result}")
            else:
                logger.warning("没有返回远程任务ID，无法测试任务状态查询")
        else:
            logger.warning("任务提交失败，无法测试后续API")
        
        # 4. 测试取消任务API（如果有任务ID）
        if submit_result and remote_task_id:
            logger.info(f"4. 正在测试取消任务API，任务ID: {remote_task_id}...")
            cancel_result = await client.cancel_task(remote_task_id)
            logger.info(f"取消任务结果: {cancel_result}")
        
        logger.info("所有测试完成!")
        
    except Exception as e:
        logger.error(f"测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
