#!/usr/bin/env python3
"""
测试与远程训练服务器的连接
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

async def main():
    """主函数"""
    logger.info("开始测试与远程训练服务器的连接...")
    
    try:
        # 创建RemoteClient实例
        client = RemoteClient()
        logger.info(f"远程服务器URL: {client.server_url}")
        logger.info(f"账号: {client.username}")
        
        # 测试健康检查
        logger.info("正在执行健康检查...")
        health_result = await client.health_check()
        logger.info(f"健康检查结果: {health_result}")
        
        # 测试服务器状态
        logger.info("正在获取服务器状态...")
        status_result = await client.server_status()
        logger.info(f"服务器状态: {status_result}")
        
        # 测试API访问
        logger.info("正在测试API访问...")
        # 这里可以添加更多的API测试
        
        logger.info("所有测试完成!")
        
    except Exception as e:
        logger.error(f"测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
