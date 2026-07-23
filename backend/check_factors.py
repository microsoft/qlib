from app.db.database import engine
from sqlalchemy import text

with engine.connect() as conn:
    print("检查158因子组:")
    result = conn.execute(text('SELECT COUNT(*) FROM factors WHERE group_id = 1'))
    print('因子数量:', result.scalar())
    
    result = conn.execute(text('SELECT name, formula FROM factors WHERE group_id = 1 LIMIT 5'))
    print('前5个因子:')
    for row in result:
        print(f"  {row.name}: {row.formula}")
    
    print("\n检查360因子组:")
    result = conn.execute(text('SELECT COUNT(*) FROM factors WHERE group_id = 2'))
    print('因子数量:', result.scalar())
    
    result = conn.execute(text('SELECT name, formula FROM factors WHERE group_id = 2 LIMIT 5'))
    print('前5个因子:')
    for row in result:
        print(f"  {row.name}: {row.formula}")
