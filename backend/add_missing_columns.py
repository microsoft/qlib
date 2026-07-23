from app.db.database import engine
from sqlalchemy import text

# 连接数据库
with engine.connect() as conn:
    # 添加缺少的列
    try:
        # 添加retries列
        conn.execute(text('ALTER TABLE tasks ADD COLUMN retries INT DEFAULT 0 NOT NULL'))
        print("Added retries column")
    except Exception as e:
        print(f"Error adding retries column: {e}")
    
    try:
        # 添加max_retries列
        conn.execute(text('ALTER TABLE tasks ADD COLUMN max_retries INT DEFAULT 3 NOT NULL'))
        print("Added max_retries column")
    except Exception as e:
        print(f"Error adding max_retries column: {e}")
    
    try:
        # 添加retry_delay列
        conn.execute(text('ALTER TABLE tasks ADD COLUMN retry_delay INT DEFAULT 5 NOT NULL'))
        print("Added retry_delay column")
    except Exception as e:
        print(f"Error adding retry_delay column: {e}")
    
    # 提交事务
    conn.commit()
    print("All changes committed")
    
    # 验证表结构
    result = conn.execute(text('DESCRIBE tasks'))
    print("\nUpdated tasks table structure:")
    for row in result:
        print(row)
