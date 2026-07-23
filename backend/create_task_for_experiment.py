from app.db.database import SessionLocal
from app.services.task import TaskService

# Create a task for experiment 7
db = SessionLocal()
task = TaskService.create_task(db, experiment_id=7, task_type="train", priority=1)
print(f"Created task with ID: {task.id}")
print(f"Task status: {task.status}")
print(f"Task experiment ID: {task.experiment_id}")
db.close()
