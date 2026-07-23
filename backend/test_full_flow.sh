#!/bin/bash

# 设置访问令牌
TOKEN="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJhZG1pbiIsImV4cCI6MTc2NDU3NDAyOX0.pYyrE6IvWx958VQbX3OrnjnDIbdzAhgicQNHmLqe58E"

# 1. 创建新实验
echo "Step 1: Creating new experiment..."
EXPERIMENT_RESPONSE=$(curl -s -X POST -H "Content-Type: application/json" -H "Authorization: Bearer $TOKEN" -d '{"name": "full-test-experiment", "description": "Complete test experiment for model training and evaluation", "config": {"model": "LightGBM", "dataset": "CSI300", "start_time": "2020-01-01", "end_time": "2023-12-31", "factors": ["alpha001", "alpha002", "alpha003"], "hyperparams": {"n_estimators": 100, "learning_rate": 0.1}}}' http://localhost:8000/api/experiments/)

# 提取实验ID
EXPERIMENT_ID=$(echo $EXPERIMENT_RESPONSE | jq -r '.id')
echo "Created experiment with ID: $EXPERIMENT_ID"
echo "Experiment details: $EXPERIMENT_RESPONSE"
echo

# 2. 运行实验
echo "Step 2: Running experiment..."
RUN_RESPONSE=$(curl -s -X POST -H "Authorization: Bearer $TOKEN" http://localhost:8000/api/experiments/$EXPERIMENT_ID/run)
echo "Run response: $RUN_RESPONSE"
echo

# 3. 等待一段时间，让实验开始运行
echo "Step 3: Waiting for experiment to start..."
sleep 5
echo

# 4. 查看实验状态
echo "Step 4: Checking experiment status..."
STATUS_RESPONSE=$(curl -s -X GET -H "Authorization: Bearer $TOKEN" http://localhost:8000/api/experiments/$EXPERIMENT_ID)
echo "Experiment status: $(echo $STATUS_RESPONSE | jq -r '.status')"
echo "Experiment details: $STATUS_RESPONSE"
echo

# 5. 等待实验完成
echo "Step 5: Waiting for experiment to complete..."
for i in {1..10}; do
    STATUS_RESPONSE=$(curl -s -X GET -H "Authorization: Bearer $TOKEN" http://localhost:8000/api/experiments/$EXPERIMENT_ID)
    STATUS=$(echo $STATUS_RESPONSE | jq -r '.status')
    echo "Experiment status: $STATUS (Attempt $i/10)"
    if [ "$STATUS" = "completed" ] || [ "$STATUS" = "failed" ]; then
        break
    fi
    sleep 10
done
echo

# 6. 查看最终结果
echo "Step 6: Viewing final results..."
FINAL_RESPONSE=$(curl -s -X GET -H "Authorization: Bearer $TOKEN" http://localhost:8000/api/experiments/$EXPERIMENT_ID)
echo "Final experiment status: $(echo $FINAL_RESPONSE | jq -r '.status')"
echo "Performance metrics: $(echo $FINAL_RESPONSE | jq -r '.performance')"
echo "Complete experiment details: $FINAL_RESPONSE"
echo

# 7. 总结
echo "Step 7: Summary"
echo "- Experiment ID: $EXPERIMENT_ID"
echo "- Status: $(echo $FINAL_RESPONSE | jq -r '.status')"
echo "- Total Return: $(echo $FINAL_RESPONSE | jq -r '.performance.total_return')"
echo "- Annual Return: $(echo $FINAL_RESPONSE | jq -r '.performance.annual_return')"
echo "- Sharpe Ratio: $(echo $FINAL_RESPONSE | jq -r '.performance.sharpe_ratio')"
echo "- Max Drawdown: $(echo $FINAL_RESPONSE | jq -r '.performance.max_drawdown')"
echo "- Win Rate: $(echo $FINAL_RESPONSE | jq -r '.performance.win_rate')"