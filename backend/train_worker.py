#!/usr/bin/env python3
"""
训练节点应用，用于执行模型训练任务
从数据库中获取待执行的任务，并调用现有的训练函数来执行任务
"""

import asyncio
import logging
import os
import sys
from sqlalchemy.orm import Session
from app.db.database import engine, SessionLocal, Base
from app.services.task import TaskService
from app.utils.remote_client import RemoteClient
from app.models.experiment import Experiment

# 配置详细日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('/home/qlib_t/backend/logs/train_worker_init.log', mode='a')
    ]
)
logger = logging.getLogger(__name__)

# 启动时的环境检查
logger.info(f"Starting train_worker.py with Python {sys.version}")
logger.info(f"Current working directory: {os.getcwd()}")
logger.info(f"PYTHONPATH: {os.environ.get('PYTHONPATH', 'Not set')}")

# 验证必要的模块是否可用
try:
    import sqlalchemy
    logger.info(f"SQLAlchemy version: {sqlalchemy.__version__}")
except ImportError as e:
    logger.error(f"Failed to import SQLAlchemy: {e}")
    sys.exit(1)

try:
    import aiohttp
    logger.info(f"aiohttp version: {aiohttp.__version__}")
except ImportError as e:
    logger.error(f"Failed to import aiohttp: {e}")
    sys.exit(1)

class TrainWorker:
    def __init__(self, worker_id: str, max_workers: int = 2):
        self.worker_id = worker_id
        self.max_workers = max_workers
        self.running = False
        try:
            logger.info("Initializing RemoteClient...")
            self.remote_client = RemoteClient()
            logger.info("RemoteClient initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize RemoteClient: {e}")
            self.remote_client = None
    
    def get_db(self):
        """获取数据库会话"""
        return SessionLocal()
    
    async def run(self):
        """启动任务执行器"""
        self.running = True
        logger.info(f"Train worker {self.worker_id} started")
        
        # 验证与远程服务器的连接
        if self.remote_client:
            if not await self.remote_client.health_check():
                logger.error("Failed to connect to remote training server. Worker will continue but tasks may fail.")
        else:
            logger.error("RemoteClient not initialized. Worker will continue but tasks may fail.")
        
        while self.running:
            try:
                db = self.get_db()
                # 获取待处理任务
                tasks = TaskService.get_pending_tasks(db, limit=self.max_workers)
                
                if not tasks:
                    # 没有待处理任务，等待一段时间
                    logger.info(f"No pending tasks, waiting for 5 seconds...")
                    await asyncio.sleep(5)
                    continue
                
                # 并行执行任务
                await asyncio.gather(*[self.process_task(task) for task in tasks])
            except Exception as e:
                logger.error(f"Error in train worker: {e}")
                await asyncio.sleep(5)
    
    async def process_task(self, task):
        """处理单个任务"""
        logger.info(f"Processing task {task.id} for experiment {task.experiment_id}")
        
        db = self.get_db()
        try:
            # 获取实验信息
            experiment = db.query(Experiment).filter(Experiment.id == task.experiment_id).first()
            if not experiment:
                logger.error(f"Experiment {task.experiment_id} not found for task {task.id}")
                TaskService.update_task_status(db, task.id, "failed", error="Experiment not found")
                return
            
            # 更新任务状态为运行中
            TaskService.update_task_status(db, task.id, "running", progress=0)
            
            # 检查remote_client是否可用
            if not self.remote_client:
                logger.error(f"Cannot process task {task.id}: RemoteClient not initialized")
                TaskService.update_task_status(db, task.id, "failed", error="RemoteClient not initialized")
                return
            
            # 执行任务
            if task.task_type == "train":
                # 构建任务数据
                task_data = {
                    "experiment_id": task.experiment_id,
                    "name": experiment.name,
                    "config": experiment.config,
                    "task_id": task.id
                }
                
                # 提交任务到远程服务器
                remote_result = await self.remote_client.submit_task(task_data)
                
                if not remote_result:
                    logger.error(f"Failed to submit task {task.id} to remote server")
                    TaskService.update_task_status(db, task.id, "failed", error="Failed to submit task to remote server")
                    return
                
                # 获取远程任务ID
                remote_task_id = remote_result.get("task_id")
                if not remote_task_id:
                    logger.error(f"No task_id returned from remote server for task {task.id}")
                    TaskService.update_task_status(db, task.id, "failed", error="No task_id returned from remote server")
                    return
                
                logger.info(f"Task {task.id} submitted to remote server with remote_task_id: {remote_task_id}")
                
                # 轮询任务状态
                await self.poll_task_status(task, remote_task_id, db)
            else:
                # 其他任务类型
                logger.warning(f"Unknown task type: {task.task_type}")
                TaskService.update_task_status(db, task.id, "failed", error=f"Unknown task type: {task.task_type}")
        except Exception as e:
            # 更新任务状态为失败
            logger.error(f"Task {task.id} failed: {e}")
            TaskService.update_task_status(db, task.id, "failed", error=str(e))
        finally:
            db.close()
    
    async def process_websocket_updates(self, task, remote_task_id, db):
        """处理WebSocket更新"""
        def update_callback(data):
            """处理WebSocket更新的回调函数"""
            try:
                logger.info(f"Received WebSocket update for task {task.id}: {data}")
                
                # 更新本地任务状态和进度
                status = data.get("status", "unknown")
                progress = data.get("progress", 0)
                error = data.get("error", None)
                result = data.get("result", None)
                
                # 更新任务状态
                db = self.get_db()
                TaskService.update_task_status(db, task.id, status, progress=progress, error=error, result=result)
                db.close()
                
                # 如果任务完成，记录日志
                if status in ["completed", "failed", "cancelled"]:
                    logger.info(f"Task {task.id} completed with status: {status}")
            except Exception as e:
                logger.error(f"Error handling WebSocket update: {e}")
        
        # 连接WebSocket
        await self.remote_client.connect_websocket(remote_task_id, update_callback)
    
    async def poll_task_status(self, task, remote_task_id, db):
        """轮询远程任务状态"""
        logger.info(f"Starting to poll status for remote task {remote_task_id}")
        
        # 启动WebSocket更新处理
        ws_task = asyncio.create_task(self.process_websocket_updates(task, remote_task_id, db))
        
        try:
            while True:
                # 获取远程任务状态（作为备份，主要依赖WebSocket更新）
                status_result = await self.remote_client.get_task_status(remote_task_id)
                
                if status_result:
                    # 解析任务状态
                    remote_status = status_result.get("status", "unknown")
                    progress = status_result.get("progress", 0)
                    error = status_result.get("error", None)
                    
                    logger.info(f"Remote task {remote_task_id} status: {remote_status}, progress: {progress}%")
                    
                    # 更新本地任务状态和进度
                    TaskService.update_task_status(db, task.id, remote_status, progress=progress, error=error)
                    
                    # 检查任务是否完成
                    if remote_status in ["completed", "failed", "cancelled"]:
                        logger.info(f"Task {task.id} with remote_task_id {remote_task_id} completed with status: {remote_status}")
                        
                        if remote_status == "completed":
                            # 获取任务结果
                            results = await self.remote_client.get_task_results(remote_task_id)
                            if results:
                                logger.info(f"Got results for task {task.id}: {results}")
                                # 更新任务结果
                                TaskService.update_task_status(db, task.id, "completed", progress=100, result=results)
                        elif remote_status == "failed":
                            # 检查是否需要重试
                            if task.retries < task.max_retries:
                                logger.info(f"Task {task.id} failed, retrying... (attempt {task.retries + 1}/{task.max_retries})")
                                # 计算重试延迟
                                import random
                                exponential_delay = task.retry_delay * (2 ** task.retries)
                                jitter = random.randint(0, 10)
                                retry_delay = min(exponential_delay + jitter, 300)  # 最大延迟5分钟
                                
                                logger.info(f"Waiting {retry_delay} seconds before retrying task {task.id}")
                                await asyncio.sleep(retry_delay)
                                
                                # 重试任务
                                TaskService.retry_task(db, task.id)
                                return
                            else:
                                logger.info(f"Task {task.id} failed after {task.max_retries} attempts, marking as final failed")
                        
                        # 取消WebSocket任务
                        ws_task.cancel()
                        break
                
                # 等待一段时间后继续轮询（作为备份）
                await asyncio.sleep(30)
        except Exception as e:
            logger.error(f"Error polling task status for {remote_task_id}: {e}")
            ws_task.cancel()
            
            # 检查是否需要重试
            if task.retries < task.max_retries:
                logger.info(f"Error polling task status, retrying... (attempt {task.retries + 1}/{task.max_retries})")
                # 计算重试延迟
                import random
                exponential_delay = task.retry_delay * (2 ** task.retries)
                jitter = random.randint(0, 10)
                retry_delay = min(exponential_delay + jitter, 300)  # 最大延迟5分钟
                
                logger.info(f"Waiting {retry_delay} seconds before retrying task {task.id}")
                await asyncio.sleep(retry_delay)
                
                # 重试任务
                TaskService.retry_task(db, task.id)
            else:
                logger.info(f"Error polling task status after {task.max_retries} attempts, marking as final failed")
                TaskService.update_task_status(db, task.id, "failed", error=f"Error polling remote task status: {str(e)}")
        
        # 等待WebSocket任务完成
        try:
            await ws_task
        except asyncio.CancelledError:
            logger.info("WebSocket task cancelled")
        except Exception as e:
            logger.error(f"Error in WebSocket task: {e}")
    
    def stop(self):
        """停止任务执行器"""
        self.running = False
        logger.info(f"Train worker {self.worker_id} stopped")

async def main():
    """主函数"""
    # 创建训练节点实例
    worker = TrainWorker("local_train_worker", max_workers=2)
    
    try:
        # 启动训练节点
        await worker.run()
    except KeyboardInterrupt:
        # 处理键盘中断
        worker.stop()
        logger.info("Train worker stopped by user")

if __name__ == "__main__":
    # 运行主函数
    asyncio.run(main())
