#!/bin/bash

# 启动训练服务
echo "Starting QLib Training Server..."
uvicorn train_server:app --host 0.0.0.0 --port 8000 --reload --log-config none
