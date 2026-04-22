import asyncio
import logging
from sqlalchemy.orm import Session
from app.db.database import engine, SessionLocal
from app.services.task import TaskService
from app.tasks.train import train_model_task as original_train_task

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TaskWorker:
    def __init__(self, worker_id: str, max_workers: int = 1):
        self.worker_id = worker_id
        self.max_workers = max_workers
        self.running = False
        self.session = None
    
    def get_db(self):
        """获取数据库会话"""
        if self.session is None:
            self.session = SessionLocal()
        return self.session
    
    async def run(self):
        """启动任务执行器"""
        self.running = True
        logger.info(f"Task worker {self.worker_id} started")
        
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
                logger.error(f"Error in task worker: {e}")
                await asyncio.sleep(5)
    
    async def process_task(self, task):
        """处理单个任务"""
        logger.info(f"Processing task {task.id} for experiment {task.experiment_id}")
        
        db = self.get_db()
        
        try:
            # 更新任务状态为运行中
            TaskService.update_task_status(db, task.id, "running", progress=0)
            
            # 执行任务
            if task.task_type == "train":
                # 调用训练函数
                await original_train_task(task.experiment_id, {}, db)
                
                # 更新任务状态为完成
                TaskService.update_task_status(db, task.id, "completed", progress=100)
                logger.info(f"Task {task.id} completed successfully")
            else:
                # 其他任务类型
                logger.warning(f"Unknown task type: {task.task_type}")
                TaskService.update_task_status(db, task.id, "failed", error=f"Unknown task type: {task.task_type}")
        except Exception as e:
            # 更新任务状态为失败
            logger.error(f"Task {task.id} failed: {e}")
            TaskService.update_task_status(db, task.id, "failed", error=str(e))
    
    def stop(self):
        """停止任务执行器"""
        self.running = False
        if self.session:
            self.session.close()
        logger.info(f"Task worker {self.worker_id} stopped")

# 主函数，用于独立运行任务执行器
if __name__ == "__main__":
    worker = TaskWorker("worker_1", max_workers=2)
    try:
        asyncio.run(worker.run())
    except KeyboardInterrupt:
        worker.stop()
        logger.info("Task worker stopped by user")
