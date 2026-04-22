from app.db.database import engine, Base
from app.models.task import Task

# 自动创建或更新表结构
# Base.metadata.create_all(bind=engine) 会创建所有表，但不会更新现有表

# 使用SQLAlchemy的inspect来检查表结构，然后手动添加缺少的列
from sqlalchemy import inspect, text

with engine.connect() as conn:
    inspector = inspect(engine)
    # 获取现有表的列
    existing_columns = [col['name'] for col in inspector.get_columns('tasks')]
    print(f"Existing columns: {existing_columns}")
    
    # 定义需要的列
    required_columns = ['retries', 'max_retries', 'retry_delay']
    
    # 添加缺少的列
    for column in required_columns:
        if column not in existing_columns:
            if column == 'retries':
                conn.execute(text('ALTER TABLE tasks ADD COLUMN retries INT DEFAULT 0 NOT NULL'))
                print(f"Added column: {column}")
            elif column == 'max_retries':
                conn.execute(text('ALTER TABLE tasks ADD COLUMN max_retries INT DEFAULT 3 NOT NULL'))
                print(f"Added column: {column}")
            elif column == 'retry_delay':
                conn.execute(text('ALTER TABLE tasks ADD COLUMN retry_delay INT DEFAULT 5 NOT NULL'))
                print(f"Added column: {column}")
    
    # 提交事务
    conn.commit()
    print("All changes committed")
