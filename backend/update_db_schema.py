from app.db.database import engine
from sqlalchemy import text

with engine.connect() as conn:
    try:
        # 尝试删除factors表上的唯一索引
        conn.execute(text('DROP INDEX IF EXISTS ix_factors_name'))
        conn.execute(text('DROP INDEX IF EXISTS factors_name_key'))
        conn.execute(text('DROP INDEX IF EXISTS sqlite_autoindex_factors_1'))
        conn.commit()
        print("已尝试删除factors表上的唯一索引")
    except Exception as e:
        print(f"删除索引时出错: {e}")
    
    # 重新创建一个非唯一索引
    try:
        conn.execute(text('CREATE INDEX IF NOT EXISTS ix_factors_name ON factors(name)'))
        conn.commit()
        print("已创建非唯一索引ix_factors_name")
    except Exception as e:
        print(f"创建索引时出错: {e}")
    
    print("数据库表结构更新完成")
