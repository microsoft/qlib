import streamlit as st
import qlib
from qlib.constant import REG_CN
import os
import subprocess
import sys
from pathlib import Path
from qlib_utils import SUPPORTED_MODELS, train_model, predict, backtest_strategy
import pandas as pd
import plotly.express as px
import datetime
import copy

# --- Backend Functions (placeholders for brevity) ---
def run_command(command, cwd=None):
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True,
        text=True, encoding='utf-8', errors='replace', cwd=cwd
    )
    for line in iter(process.stdout.readline, ''): yield line
    process.stdout.close()
    if process.wait(): raise subprocess.CalledProcessError(process.returncode, command)

def get_qlib_data(qlib_dir_str):
    # ... implementation exists
    pass

# --- Streamlit Pages ---

def data_management_page():
    st.header("数据管理")
    st.markdown("在这里，您可以下载和管理 Qlib 所需的股票数据。")
    default_path = str(Path.home() / ".qlib" / "qlib_data" / "cn_data")
    qlib_dir = st.text_input("Qlib 数据存储路径", default_path, key="data_dir_dm")
    if st.button("开始下载/更新A股数据"):
        if qlib_dir: get_qlib_data(qlib_dir); st.success("数据下载任务已执行。")
        else: st.error("请输入有效的数据存储路径！")
    st.subheader("检查本地数据")
    if st.button("检查数据是否存在"):
        data_path = Path(qlib_dir).expanduser()
        if data_path.exists() and (data_path / "features").exists() and any((data_path / "features").iterdir()):
             st.success(f"数据目录 '{data_path}' 存在且有效。")
        else:
            st.warning(f"数据目录 '{data_path}' 不存在或无效。")

def model_training_page():
    st.header("模型训练")
    st.markdown("选择一个模型并调整其超参数进行训练。")
    default_data_path = str(Path.home() / ".qlib" / "qlib_data" / "cn_data")
    default_models_path = str(Path.home() / "qlib_models")
    qlib_dir = st.text_input("Qlib 数据存储路径", default_data_path, key="data_dir_train")
    models_save_dir = st.text_input("训练后模型的保存路径", default_models_path)
    model_name = st.selectbox("选择要训练的模型", list(SUPPORTED_MODELS.keys()))

    st.subheader("超参数调节")
    config = copy.deepcopy(SUPPORTED_MODELS[model_name])
    params = config['task']['model']['kwargs']

    with st.expander("调节模型参数", expanded=True):
        if "LightGBM" in model_name or "XGBoost" in model_name:
            params['n_estimators'] = st.slider("树的数量 (n_estimators)", 50, 500, params.get('n_estimators', 200), 10)
            params['learning_rate'] = st.slider("学习率 (learning_rate)", 0.01, 0.2, params.get('learning_rate', 0.05), 0.01)
            params['max_depth'] = st.slider("最大深度 (max_depth)", 3, 15, params.get('max_depth', 7))

    if st.button("开始训练"):
        if not Path(qlib_dir).expanduser().exists():
            st.error("数据路径不存在，请先下载数据！")
        else:
            with st.spinner(f"正在使用 '{model_name}' 进行训练..."):
                try:
                    saved_path = train_model(model_name, qlib_dir, models_save_dir, config)
                    st.success(f"模型训练成功！已保存至: {saved_path}")
                except Exception as e:
                    st.error(f"训练过程中发生错误: {e}")

def prediction_page():
    st.header("投资组合预测")
    st.markdown("使用已训练好的模型，对指定日期的股票进行评分预测。")
    default_data_path = str(Path.home() / ".qlib" / "qlib_data" / "cn_data")
    default_models_path = str(Path.home() / "qlib_models")
    models_dir = st.text_input("模型所在目录", default_models_path, key="models_dir_pred")
    qlib_dir = st.text_input("Qlib 数据存储路径", default_data_path, key="data_dir_pred")
    models_dir_path = Path(models_dir).expanduser()
    available_models = [f.name for f in models_dir_path.glob("*.pkl")] if models_dir_path.exists() else []

    if not available_models:
        st.warning(f"在 '{models_dir_path}' 中未找到模型。")
        return
    selected_model_name = st.selectbox("选择一个模型文件", available_models)
    selected_model_path = str(models_dir_path / selected_model_name)
    prediction_date = st.date_input("选择预测日期", datetime.date.today() - datetime.timedelta(days=1))

    if st.button("执行预测"):
        with st.spinner(f"正在预测..."):
            try:
                pred_df = predict(selected_model_path, qlib_dir, prediction_date.strftime("%Y-%m-%d"))
                st.success("预测完成！")
                st.subheader("预测结果")
                top_n = st.slider("展示Top N只股票", 5, 50, 10)
                st.dataframe(pred_df.head(top_n))
                fig = px.bar(pred_df.head(top_n), x="StockID", y="score", title=f"Top {top_n} 股票预测分数")
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"预测过程中发生错误: {e}")

def backtesting_page():
    st.header("策略回测")
    st.markdown("使用模型进行历史回测，以评估策略表现。")
    default_data_path = str(Path.home() / ".qlib" / "qlib_data" / "cn_data")
    default_models_path = str(Path.home() / "qlib_models")
    models_dir = st.text_input("模型所在目录", default_models_path, key="models_dir_bt")
    qlib_dir = st.text_input("Qlib 数据存储路径", default_data_path, key="data_dir_bt")

    models_dir_path = Path(models_dir).expanduser()
    available_models = [f.name for f in models_dir_path.glob("*.pkl")] if models_dir_path.exists() else []

    if not available_models:
        st.warning(f"在 '{models_dir_path}' 中未找到模型。")
        return

    selected_model_name = st.selectbox("选择一个模型文件进行回测", available_models)
    selected_model_path = str(models_dir_path / selected_model_name)

    st.subheader("回测参数配置")
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("开始日期", datetime.date.today() - datetime.timedelta(days=365))
    with col2:
        end_date = st.date_input("结束日期", datetime.date.today() - datetime.timedelta(days=1))

    st.subheader("策略参数 (Top-K Dropout)")
    c1, c2 = st.columns(2)
    topk = c1.number_input("买入Top-K只股票", 1, 100, 50)
    n_drop = c2.number_input("持有期(天)", 1, 20, 5)

    st.subheader("交易参数")
    c1, c2, c3 = st.columns(3)
    open_cost = c1.number_input("开仓手续费率", 0.0, 0.01, 0.0005, format="%.4f")
    close_cost = c2.number_input("平仓手续费率", 0.0, 0.01, 0.0015, format="%.4f")
    min_cost = c3.number_input("最低手续费", 0, 10, 5)

    if st.button("开始回测"):
        if start_date >= end_date:
            st.error("开始日期必须早于结束日期！")
        else:
            strategy_kwargs = {"topk": topk, "n_drop": n_drop}
            exchange_kwargs = {"open_cost": open_cost, "close_cost": close_cost, "min_cost": min_cost, "deal_price": "close"}
            with st.spinner(f"正在回测..."):
                try:
                    report_df = backtest_strategy(selected_model_path, qlib_dir, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"), strategy_kwargs, exchange_kwargs)
                    st.success("回测完成！")
                    st.subheader("绩效指标")
                    metrics = report_df.loc["excess_return_with_cost"]
                    kpi_cols = st.columns(4)
                    kpi_cols[0].metric("年化收益率", f"{metrics['annualized_return']:.2%}")
                    kpi_cols[1].metric("夏普比率", f"{metrics['information_ratio']:.2f}")
                    kpi_cols[2].metric("最大回撤", f"{metrics['max_drawdown']:.2%}")
                    kpi_cols[3].metric("换手率", f"{metrics['turnover_rate']:.2f}")
                    st.subheader("资金曲线")
                    fig = px.line(report_df.index, y=report_df.values, title="策略 vs. 基准")
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"回测过程中发生错误: {e}")

def main():
    st.set_page_config(layout="wide", page_title="Qlib 可视化工具")
    st.sidebar.image("https://avatars.githubusercontent.com/u/65423353?s=200&v=4", width=100)
    st.sidebar.title("Qlib 可视化面板")
    page_options = ["数据管理", "模型训练", "投资组合预测", "策略回测"]
    page = st.sidebar.radio("选择功能页面", page_options)

    if page == "数据管理": data_management_page()
    elif page == "模型训练": model_training_page()
    elif page == "投资组合预测": prediction_page()
    elif page == "策略回测": backtesting_page()

if __name__ == "__main__":
    main()
