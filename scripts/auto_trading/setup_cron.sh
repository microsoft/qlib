#!/bin/bash

# A股/港股/ETF 自动化交易系统 - Cron 定时任务配置脚本
#
# 功能：配置每日自动运行交易流程
# 运行时间：每个交易日下午 4:00 (股市收盘后)
#
# 使用方法：
#   bash setup_cron.sh

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}================================${NC}"
echo -e "${GREEN}配置自动化交易定时任务${NC}"
echo -e "${GREEN}================================${NC}"
echo ""

# 获取脚本所在目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"

echo -e "${YELLOW}项目目录: ${PROJECT_DIR}${NC}"
echo -e "${YELLOW}脚本目录: ${SCRIPT_DIR}${NC}"
echo ""

# 日志目录
LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "$LOG_DIR"

echo -e "${YELLOW}日志目录: ${LOG_DIR}${NC}"
echo ""

# 配置文件路径
CONFIG_FILE="${SCRIPT_DIR}/config.yaml"

if [ ! -f "$CONFIG_FILE" ]; then
    echo -e "${RED}错误: 配置文件不存在: ${CONFIG_FILE}${NC}"
    echo -e "${YELLOW}请先创建配置文件${NC}"
    exit 1
fi

# Python 路径
PYTHON_CMD=$(which python3 || which python)
echo -e "${YELLOW}Python 命令: ${PYTHON_CMD}${NC}"
echo ""

# Cron 任务定义
CRON_SCHEDULE="0 16 * * 1-5"  # 周一到周五下午4点
CRON_COMMAND="cd ${PROJECT_DIR} && ${PYTHON_CMD} ${SCRIPT_DIR}/main_controller.py --config ${CONFIG_FILE} >> ${LOG_DIR}/auto_trading_\$(date +\%Y\%m\%d).log 2>&1"

echo -e "${GREEN}Cron 任务配置:${NC}"
echo -e "${YELLOW}时间: ${CRON_SCHEDULE} (周一到周五下午4点)${NC}"
echo -e "${YELLOW}命令: ${CRON_COMMAND}${NC}"
echo ""

# 检查是否已存在相同的任务
if crontab -l 2>/dev/null | grep -q "main_controller.py"; then
    echo -e "${YELLOW}检测到已存在的定时任务${NC}"
    read -p "是否要替换现有任务? (y/N): " -n 1 -r
    echo ""

    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${YELLOW}取消配置${NC}"
        exit 0
    fi

    # 删除旧任务
    crontab -l 2>/dev/null | grep -v "main_controller.py" | crontab - || true
    echo -e "${GREEN}✓ 已删除旧任务${NC}"
fi

# 添加新任务
(crontab -l 2>/dev/null; echo "# A股/港股/ETF 自动化交易系统 - 每个交易日下午4点运行"; echo "${CRON_SCHEDULE} ${CRON_COMMAND}") | crontab -

echo -e "${GREEN}✓ 定时任务配置完成!${NC}"
echo ""

# 显示当前的 crontab
echo -e "${GREEN}当前 Crontab 任务列表:${NC}"
echo -e "${YELLOW}================================${NC}"
crontab -l
echo -e "${YELLOW}================================${NC}"
echo ""

# 提示
echo -e "${GREEN}配置说明:${NC}"
echo -e "1. 任务将在每个交易日（周一到周五）下午 4:00 自动运行"
echo -e "2. 日志文件保存在: ${LOG_DIR}"
echo -e "3. 日志文件格式: auto_trading_YYYYMMDD.log"
echo ""

echo -e "${GREEN}管理命令:${NC}"
echo -e "  查看定时任务: ${YELLOW}crontab -l${NC}"
echo -e "  编辑定时任务: ${YELLOW}crontab -e${NC}"
echo -e "  删除所有任务: ${YELLOW}crontab -r${NC}"
echo ""

echo -e "${GREEN}测试运行:${NC}"
echo -e "  手动测试:     ${YELLOW}${PYTHON_CMD} ${SCRIPT_DIR}/main_controller.py --config ${CONFIG_FILE}${NC}"
echo -e "  跳过数据更新: ${YELLOW}${PYTHON_CMD} ${SCRIPT_DIR}/main_controller.py --config ${CONFIG_FILE} --skip-data-update${NC}"
echo ""

echo -e "${GREEN}================================${NC}"
echo -e "${GREEN}配置完成！${NC}"
echo -e "${GREEN}================================${NC}"
