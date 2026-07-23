from app.db.database import engine
from sqlalchemy import text

# 连接数据库
with engine.connect() as conn:
    # 显示所有表
    print("Tables in database:")
    result = conn.execute(text('SHOW TABLES'))
    tables = [row[0] for row in result]
    print(tables)
    
    # 检查tasks表是否存在
    if 'tasks' in tables:
        print("\nTasks table structure:")
        result = conn.execute(text('DESCRIBE tasks'))
        for row in result:
            print(row)
    else:
        print("\nTasks table does not exist!")
    
    # 检查是否有init_db.py或类似的脚本用于创建表
    print("\nChecking if tasks table is defined in models...")
    from app.models.task import Task
    print("Task model is defined.")
    print(f"Task model columns: {[column.name for column in Task.__table__.columns]}")
