from app.db.database import SessionLocal
from app.models.task import Task
from app.models.experiment import Experiment

# 创建数据库会话
db = SessionLocal()

# 获取任务
task = db.query(Task).filter(Task.id == 1).first()

if task:
    # 重置任务状态
    task.status = 'pending'
    
    # 获取相关实验
    experiment = db.query(Experiment).filter(Experiment.id == task.experiment_id).first()
    
    if experiment:
        # 重置实验状态
        experiment.status = 'pending'
        experiment.progress = 0
        experiment.error = None
    
    # 提交更改
    db.commit()
    print('Task and experiment status reset to pending')
else:
    print('Task not found')

# 关闭数据库会话
db.close()
