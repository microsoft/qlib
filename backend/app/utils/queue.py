import logging
from typing import Optional, Dict, Any
from app.db.database import get_db
from app.services.task import TaskService
from app.models.task import Task

logger = logging.getLogger(__name__)

class TaskQueue:
    def __init__(self):
        self.db = next(get_db())
    
    def add_task(self, task_id: int, task_data: Dict[str, Any]) -> bool:
        """添加任务到队列（更新任务状态为pending）"""
        try:
            # 任务已经在数据库中，不需要再次创建，只需要确保状态为pending
            db = next(get_db())
            task = db.query(Task).filter(Task.id == task_id).first()
            if task:
                task.status = "pending"
                db.commit()
                logger.info(f"Task {task_id} added to queue")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to add task {task_id} to queue: {e}")
            return False
    
    def get_task(self) -> Optional[Dict[str, Any]]:
        """从队列获取任务（获取状态为pending的任务）"""
        try:
            db = next(get_db())
            # 获取状态为pending的任务，按优先级和创建时间排序
            task = db.query(Task).filter(Task.status == "pending").order_by(Task.priority.desc(), Task.created_at).first()
            if task:
                # 更新任务状态为running
                task.status = "running"
                db.commit()
                task_info = {
                    "task_id": task.id,
                    "data": {
                        "task_id": task.id,
                        "experiment_id": task.experiment_id,
                        "task_type": task.task_type
                    }
                }
                logger.info(f"Task {task.id} retrieved from queue")
                return task_info
            return None
        except Exception as e:
            logger.error(f"Failed to get task from queue: {e}")
            return None
    
    def task_count(self) -> int:
        """获取队列中的任务数量（状态为pending的任务数量）"""
        try:
            db = next(get_db())
            return db.query(Task).filter(Task.status == "pending").count()
        except Exception as e:
            logger.error(f"Failed to get task count: {e}")
            return 0
    
    def clear_queue(self) -> bool:
        """清空队列（将所有pending状态的任务标记为failed）"""
        try:
            db = next(get_db())
            pending_tasks = db.query(Task).filter(Task.status == "pending").all()
            for task in pending_tasks:
                task.status = "failed"
                task.error = "Queue cleared"
            db.commit()
            logger.info(f"Queue cleared, {len(pending_tasks)} tasks marked as failed")
            return True
        except Exception as e:
            logger.error(f"Failed to clear queue: {e}")
            return False

# 创建全局任务队列实例
queue = TaskQueue()

# 任务处理函数
def process_task_from_queue():
    """从队列中获取并处理任务"""
    task_info = queue.get_task()
    if not task_info:
        return
    
    task_id = task_info["task_id"]
    task_data = task_info["data"]
    
    db = next(get_db())
    try:
        # 更新任务状态为运行中
        TaskService.update_task_status(db, task_id, "running", progress=0)
        
        # 执行任务
        if task_data["task_type"] == "train":
            # 导入训练函数
            from app.tasks.train import train_model_task
            import asyncio
            
            # 执行训练任务
            asyncio.run(train_model_task(task_data["experiment_id"], {}, db))
            
            # 更新任务状态为完成
            TaskService.update_task_status(db, task_id, "completed", progress=100)
            logger.info(f"Task {task_id} completed successfully")
        else:
            # 其他任务类型
            logger.warning(f"Unknown task type: {task_data['task_type']}")
            TaskService.update_task_status(db, task_id, "failed", error=f"Unknown task type: {task_data['task_type']}")
    except Exception as e:
        # 更新任务状态为失败
        logger.error(f"Task {task_id} failed: {e}")
        TaskService.update_task_status(db, task_id, "failed", error=str(e))
    finally:
        db.close()