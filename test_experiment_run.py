#!/usr/bin/env python3
"""
测试实验运行功能
"""

import os
import sys
import requests
import json
import time

# 将backend目录添加到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

# 配置日志
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 后端API基础URL
BASE_URL = "http://localhost:8000/api"

# 管理员账号密码（用于获取访问令牌）
ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "admin123"

def get_access_token():
    """获取访问令牌"""
    logger.info("正在获取访问令牌...")
    url = f"{BASE_URL}/auth/token"
    data = {
        "username": ADMIN_USERNAME,
        "password": ADMIN_PASSWORD
    }
    
    try:
        response = requests.post(url, data=data)
        if response.status_code == 200:
            token_data = response.json()
            logger.info("成功获取访问令牌")
            return token_data.get("access_token")
        else:
            logger.error(f"获取访问令牌失败: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        logger.error(f"获取访问令牌过程中发生错误: {e}")
        return None

def create_experiment(access_token):
    """创建测试实验"""
    logger.info("正在创建测试实验...")
    url = f"{BASE_URL}/experiments"
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json"
    }
    
    experiment_data = {
        "name": "Test Experiment",
        "description": "A test experiment for running functionality",
        "config": {
            "model": "LightGBM",
            "params": {
                "learning_rate": 0.1,
                "n_estimators": 100,
                "max_depth": 5
            },
            "data": {
                "start_date": "2020-01-01",
                "end_date": "2020-12-31",
                "feature_cols": ["f1", "f2", "f3"],
                "target_col": "y"
            }
        }
    }
    
    try:
        response = requests.post(url, headers=headers, json=experiment_data)
        if response.status_code == 200:
            experiment = response.json()
            logger.info(f"成功创建实验: {experiment['id']} - {experiment['name']}")
            return experiment
        else:
            logger.error(f"创建实验失败: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        logger.error(f"创建实验过程中发生错误: {e}")
        return None

def run_experiment(access_token, experiment_id):
    """运行实验"""
    logger.info(f"正在运行实验: {experiment_id}...")
    url = f"{BASE_URL}/experiments/{experiment_id}/run"
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.post(url, headers=headers)
        if response.status_code == 200:
            result = response.json()
            logger.info(f"成功启动实验运行: {result}")
            return result
        else:
            logger.error(f"启动实验运行失败: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        logger.error(f"启动实验运行过程中发生错误: {e}")
        return None

def get_experiment_status(access_token, experiment_id):
    """获取实验状态"""
    logger.info(f"正在获取实验状态: {experiment_id}...")
    url = f"{BASE_URL}/experiments/{experiment_id}"
    headers = {
        "Authorization": f"Bearer {access_token}"
    }
    
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            experiment = response.json()
            logger.info(f"实验状态: {experiment['status']}")
            return experiment
        else:
            logger.error(f"获取实验状态失败: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        logger.error(f"获取实验状态过程中发生错误: {e}")
        return None

def get_experiment_tasks(access_token, experiment_id):
    """获取实验关联的任务"""
    logger.info(f"正在获取实验关联的任务: {experiment_id}...")
    url = f"{BASE_URL}/experiments/{experiment_id}/tasks"
    headers = {
        "Authorization": f"Bearer {access_token}"
    }
    
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            tasks = response.json()
            logger.info(f"找到 {len(tasks)} 个关联任务")
            for task in tasks:
                logger.info(f"  任务: {task['id']} - 状态: {task['status']} - 进度: {task['progress']}%")
            return tasks
        else:
            logger.error(f"获取实验关联任务失败: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        logger.error(f"获取实验关联任务过程中发生错误: {e}")
        return None

def main():
    """主函数"""
    logger.info("开始测试实验运行功能...")
    
    # 1. 获取访问令牌
    access_token = get_access_token()
    if not access_token:
        logger.error("无法获取访问令牌，测试终止")
        return
    
    # 2. 创建测试实验
    experiment = create_experiment(access_token)
    if not experiment:
        logger.error("无法创建测试实验，测试终止")
        return
    
    experiment_id = experiment['id']
    
    # 3. 运行实验
    run_result = run_experiment(access_token, experiment_id)
    if not run_result:
        logger.error("无法启动实验运行，测试终止")
        return
    
    # 4. 等待一段时间，然后检查实验状态和任务
    logger.info("正在等待5秒，然后检查实验状态...")
    time.sleep(5)
    
    # 5. 检查实验状态
    experiment = get_experiment_status(access_token, experiment_id)
    if experiment:
        logger.info(f"实验状态: {experiment['status']}")
    
    # 6. 检查实验关联的任务
    tasks = get_experiment_tasks(access_token, experiment_id)
    if tasks:
        logger.info(f"找到 {len(tasks)} 个关联任务")
        for task in tasks:
            logger.info(f"  任务: {task['id']} - 状态: {task['status']} - 进度: {task['progress']}%")
    
    # 7. 持续监控实验状态，直到完成或失败
    logger.info("正在持续监控实验状态，每5秒检查一次...")
    max_wait_time = 60  # 最大等待时间（秒）
    start_time = time.time()
    
    while True:
        experiment = get_experiment_status(access_token, experiment_id)
        if experiment:
            status = experiment['status']
            logger.info(f"当前实验状态: {status}")
            
            # 检查任务状态
            tasks = get_experiment_tasks(access_token, experiment_id)
            
            if status in ['completed', 'failed', 'cancelled']:
                logger.info(f"实验已完成，最终状态: {status}")
                break
            
        # 检查是否超时
        if time.time() - start_time > max_wait_time:
            logger.info(f"实验监控超时（{max_wait_time}秒），停止监控")
            break
        
        # 等待5秒后再次检查
        time.sleep(5)
    
    logger.info("实验运行功能测试完成!")

if __name__ == "__main__":
    main()
