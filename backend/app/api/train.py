from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import Dict, Any, List
import uuid
from datetime import datetime
import asyncio

from app.db.database import get_db
from app.api.deps import get_current_active_user, get_current_developer_user
from app.models.user import User

router = APIRouter()

# 训练任务管理器
class TrainingTaskManager:
    def __init__(self):
        self.tasks: Dict[str, Dict] = {}
        self.lock = asyncio.Lock()
    
    async def create_task(self, config: Dict[str, Any]) -> str:
        """创建新的训练任务"""
        task_id = f"task-{uuid.uuid4()}"
        task = {
            "id": task_id,
            "status": "pending",
            "progress": 0.0,
            "config": config,
            "result": {},
            "error": "",
            "created_at": datetime.now().isoformat(),
            "started_at": None,
            "completed_at": None
        }
        
        async with self.lock:
            self.tasks[task_id] = task
        
        return task_id
    
    async def get_task(self, task_id: str) -> Dict[str, Any]:
        """获取任务信息"""
        async with self.lock:
            task = self.tasks.get(task_id)
            if not task:
                raise HTTPException(status_code=404, detail="Task not found")
            return task.copy()
    
    async def get_all_tasks(self) -> List[Dict[str, Any]]:
        """获取所有任务信息"""
        async with self.lock:
            return list(self.tasks.values())
    
    async def update_task(self, task_id: str, **kwargs) -> Dict[str, Any]:
        """更新任务信息"""
        async with self.lock:
            task = self.tasks.get(task_id)
            if not task:
                raise HTTPException(status_code=404, detail="Task not found")
            
            task.update(kwargs)
            return task.copy()
    
    async def delete_task(self, task_id: str) -> bool:
        """删除任务"""
        async with self.lock:
            if task_id in self.tasks:
                del self.tasks[task_id]
                return True
            return False

# 创建训练任务管理器实例
task_manager = TrainingTaskManager()

# 模拟模型训练函数
async def train_model(task_id: str, config: Dict[str, Any]):
    """模拟模型训练过程"""
    try:
        # Try to import from train_server first (for standalone training server)
        from train_server import manager as websocket_manager
    except ImportError:
        # Fallback to main import (for integrated server)
        from main import manager as websocket_manager
    
    # 更新任务状态为运行中
    await task_manager.update_task(
        task_id,
        status="running",
        started_at=datetime.now().isoformat()
    )
    
    try:
        # 发送训练开始通知
        await websocket_manager.send_update({
            "task_id": task_id,
            "status": "running",
            "progress": 0.0,
            "message": "Training started",
            "timestamp": datetime.now().isoformat()
        }, task_id)
        
        # 模拟训练过程
        for i in range(10):
            # 执行训练步骤
            await asyncio.sleep(2)  # 模拟训练耗时
            
            # 更新训练进度
            progress = (i + 1) * 10
            await task_manager.update_task(
                task_id,
                progress=progress,
                status="running"
            )
            
            # 发送进度更新
            await websocket_manager.send_update({
                "task_id": task_id,
                "status": "running",
                "progress": progress,
                "message": f"Training step {i + 1}/10 completed",
                "timestamp": datetime.now().isoformat()
            }, task_id)
        
        # 训练完成
        await task_manager.update_task(
            task_id,
            status="completed",
            progress=100.0,
            result={"accuracy": 0.95, "loss": 0.05},
            completed_at=datetime.now().isoformat()
        )
        
        # 发送训练完成通知
        await websocket_manager.send_update({
            "task_id": task_id,
            "status": "completed",
            "progress": 100.0,
            "message": "Training completed successfully",
            "result": {"accuracy": 0.95, "loss": 0.05},
            "timestamp": datetime.now().isoformat()
        }, task_id)
    except Exception as e:
        # 训练失败
        error_msg = str(e)
        await task_manager.update_task(
            task_id,
            status="failed",
            error=error_msg,
            completed_at=datetime.now().isoformat()
        )
        
        # 发送训练失败通知
        await websocket_manager.send_update({
            "task_id": task_id,
            "status": "failed",
            "progress": 0.0,
            "message": f"Training failed: {error_msg}",
            "error": error_msg,
            "timestamp": datetime.now().isoformat()
        }, task_id)

# 训练API端点
@router.post("/train", response_model=Dict[str, Any])
async def train_endpoint(config: Dict[str, Any], db: Session = Depends(get_db), current_user: User = Depends(get_current_developer_user)):
    """创建训练任务"""
    task_id = await task_manager.create_task(config)
    
    # 异步执行训练任务
    asyncio.create_task(train_model(task_id, config))
    
    return {
        "task_id": task_id,
        "status": "success",
        "message": "Training task created successfully"
    }

@router.get("/tasks", response_model=List[Dict[str, Any]])
async def get_tasks(db: Session = Depends(get_db), current_user: User = Depends(get_current_active_user)):
    """获取所有训练任务"""
    return await task_manager.get_all_tasks()

@router.get("/tasks/{task_id}", response_model=Dict[str, Any])
async def get_task(task_id: str, db: Session = Depends(get_db), current_user: User = Depends(get_current_active_user)):
    """获取特定训练任务"""
    return await task_manager.get_task(task_id)

@router.delete("/tasks/{task_id}", response_model=Dict[str, Any])
async def delete_task(task_id: str, db: Session = Depends(get_db), current_user: User = Depends(get_current_developer_user)):
    """删除训练任务"""
    success = await task_manager.delete_task(task_id)
    if not success:
        raise HTTPException(status_code=404, detail="Task not found")
    return {
        "status": "success",
        "message": "Task deleted successfully"
    }
