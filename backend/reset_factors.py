from app.db.database import engine
from sqlalchemy import text

with engine.connect() as conn:
    # 删除所有旧因子
    conn.execute(text('DELETE FROM factors'))
    # 重置因子组计数
    conn.execute(text('UPDATE factor_groups SET factor_count = 0'))
    conn.commit()
    print('旧因子已删除，因子组计数已重置')
