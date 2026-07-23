import pymysql
import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 获取数据库URL
DATABASE_URL = os.getenv('DATABASE_URL')

# 解析数据库URL
# 格式: mysql+pymysql://user:password@host:port/dbname 或 mysql+pymysql://user:password@host/dbname
parts = DATABASE_URL.replace('mysql+pymysql://', '').split('/')
user_pass, host_port = parts[0].split('@')
user, password = user_pass.split(':')
# 处理没有端口号的情况，使用默认端口3306
if ':' in host_port:
    host, port = host_port.split(':')
    port = int(port)
else:
    host = host_port
    port = 3306  # MySQL默认端口
db_name = parts[1]

# 连接数据库
conn = pymysql.connect(
    host=host,
    port=int(port),
    user=user,
    password=password,
    database=db_name
)
cursor = conn.cursor()

# 添加缺少的列
try:
    # 添加retries列
    cursor.execute('ALTER TABLE tasks ADD COLUMN retries INT DEFAULT 0 NOT NULL')
    print("Added retries column")
except Exception as e:
    print(f"Error adding retries column: {e}")

try:
    # 添加max_retries列
    cursor.execute('ALTER TABLE tasks ADD COLUMN max_retries INT DEFAULT 3 NOT NULL')
    print("Added max_retries column")
except Exception as e:
    print(f"Error adding max_retries column: {e}")

try:
    # 添加retry_delay列
    cursor.execute('ALTER TABLE tasks ADD COLUMN retry_delay INT DEFAULT 5 NOT NULL')
    print("Added retry_delay column")
except Exception as e:
    print(f"Error adding retry_delay column: {e}")

# 提交并关闭连接
conn.commit()
cursor.close()
conn.close()

print("All changes committed")
