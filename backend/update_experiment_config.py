#!/usr/bin/env python3

import sqlite3
import json
from datetime import datetime

# 连接到数据库
conn = sqlite3.connect('test.db')
cursor = conn.cursor()

# 获取所有实验
cursor.execute("SELECT id, config FROM experiments")
experiments = cursor.fetchall()

for experiment in experiments:
    experiment_id, config_json = experiment
    config = json.loads(config_json)
    
    # 转换配置中的时间戳，移除时区信息
    def convert_timestamps(obj):
        """递归转换配置中的时间戳，移除时区信息"""
        if isinstance(obj, dict):
            return {k: convert_timestamps(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_timestamps(item) for item in obj]
        elif isinstance(obj, str) and 'T' in obj and ('Z' in obj or '+' in obj or '-' in obj):
            # 解析带时区的时间戳
            dt = datetime.fromisoformat(obj.replace('Z', '+00:00'))
            # 转换为不带时区的时间戳
            return dt.strftime('%Y-%m-%d')
        else:
            return obj
    
    # 转换配置中的时间戳
    converted_config = convert_timestamps(config)
    
    # 更新数据库中的配置
    cursor.execute("UPDATE experiments SET config = ? WHERE id = ?", (json.dumps(converted_config), experiment_id))
    conn.commit()
    
    print(f"Updated experiment {experiment_id} configuration, converted timestamps to naive format")

# 关闭连接
conn.close()