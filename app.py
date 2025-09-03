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

# --- Backend Functions ---

def run_command(command, cwd=None):
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        shell=True,
        text=True,
        encoding='utf-8',
        errors='replace',
        cwd=cwd
    )
    for line in iter(process.stdout.readline, ''):
        yield line
    process.stdout.close()
    return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)

def get_qlib_data(qlib_dir_str):
    qlib_dir = Path(qlib_dir_str).expanduser()
    qlib_dir.mkdir(parents=True, exist_ok=True)
    try:
        import qlib.workflow
        qlib_root = Path(qlib.workflow.__file__).resolve().parent.parent
        script_path = qlib_root / "scripts" / "get_data.py"
    except Exception:
        script_path = Path(sys.executable).parent / "get_data.py"

    if not script_path.exists():
        st.error(f"Qlib 数据下载脚本 'get_data.py' 未找到。搜索路径: {script_path}")
        return

    command = f"{sys.executable} {str(script_path)} qlib_data --target_dir '{str(qlib_dir)}' --region cn"

    st.info(f"执行命令: {command}")
    log_area = st.empty()
    log_text = ""
    try:
        for line in run_command(command):
            log_text += line
            log_area.code(log_text, language='log')
    except subprocess.CalledProcessError as e:
        st.error("数据下载脚本执行失败！")
        st.code(e.output if e.output else "No output available.")
    except Exception as e:
        st.error(f"执行命令时发生未知错误: {e}")

# --- Streamlit Pages ---

def data_management_page():
    st.header("数据管理")
    st.markdown("""
    在这里，您可以下载和管理 Qlib 所需的股票数据。
    - **数据源:** 默认从雅虎财经 (Yahoo Finance) 下载 A 股数据。
    - **存储路径:** 数据将保存在您指定的本地路径中。
    - **注意:** 首次下载数据量较大，可能需要 **10-20 分钟**，请耐心等待。
    """)

    default_path = str(Path.home() / ".qlib" / "qlib_data" / "cn_data")
    qlib_dir = st.text_input("Qlib 数据存储路径", default_path, key="data_dir_dm")

    if st.button("开始下载/更新A股数据"):
        if not qlib_dir:
            st.error("请输入有效的数据存储路径！")
        else:
            get_qlib_data(qlib_dir)
            st.success("数据下载任务已执行。请检查日志确认是否成功。")

    st.subheader("检查本地数据")
    if st.button("检查数据是否存在"):
        data_path = Path(qlib_dir).expanduser()
        features_path = data_path / "features"
        if data_path.exists() and features_path.exists() and any(features_path.iterdir()):
             st.success(f"数据目录 '{data_path}' 存在且特征目录不为空。")
             try:
                qlib.init(provider_uri=str(data_path), region=REG_CN)
                st.text("Qlib 初始化成功，可以识别到本地数据。")
             except Exception as e:
                st.error(f"数据存在，但 Qlib 初始化失败: {e}")
        else:
            st.warning(f"数据目录 '{data_path}' 或其 'features' 子目录不存在/为空。请先下载数据。")

def model_training_page():
    st.header("模型训练")
    st.markdown("""
    选择一个预设的模型和数据集配置进行训练。
    - **数据:** 训练前请确保已在“数据管理”页面下载数据。
    - **保存:** 训练完成后，模型将被保存为 `.pkl` 文件以便后续使用。
    """)

    default_data_path = str(Path.home() / ".qlib" / "qlib_data" / "cn_data")
    default_models_path = str(Path.home() / "qlib_models")

    qlib_dir = st.text_input("Qlib 数据存储路径", default_data_path, key="data_dir_train")
    models_save_dir = st.text_input("训练后模型的保存路径", default_models_path)
    model_name = st.selectbox("选择要训练的模型", list(SUPPORTED_MODELS.keys()))

    if st.button("开始训练"):
        if not Path(qlib_dir).expanduser().exists():
            st.error("数据路径不存在，请先在“数据管理”页面下载数据！")
            return

        with st.spinner(f"正在使用 '{model_name}' 进行训练，请耐心等待..."):
            try:
                saved_path = train_model(model_name, qlib_dir, models_save_dir)
                st.success(f"模型训练成功！已保存至: {saved_path}")
                st.balloons()
            except Exception as e:
                st.error(f"训练过程中发生错误: {e}")

def prediction_page():
    st.header("投资组合预测")
    st.markdown("""
    使用已训练好的模型，对指定日期的股票进行评分预测。
    - **模型:** 从指定目录加载 `.pkl` 格式的模型文件。
    - **日期:** 选择您想进行预测的交易日。
    - **结果:** 程序将展示预测分数最高的股票，并用图表进行可视化。
    """)

    default_data_path = str(Path.home() / ".qlib" / "qlib_data" / "cn_data")
    default_models_path = str(Path.home() / "qlib_models")

    models_dir = st.text_input("模型所在目录", default_models_path, key="models_dir_pred")
    qlib_dir = st.text_input("Qlib 数据存储路径", default_data_path, key="data_dir_pred")
    models_dir_path = Path(models_dir).expanduser()

    available_models = []
    if models_dir_path.exists():
        available_models = [f.name for f in models_dir_path.glob("*.pkl")]

    if not available_models:
        st.warning(f"在 '{models_dir_path}' 中未找到任何 .pkl 模型文件。请先训练模型。")
        return

    selected_model_name = st.selectbox("选择一个模型文件", available_models)
    selected_model_path = str(models_dir_path / selected_model_name)

    prediction_date = st.date_input("选择预测日期", datetime.date.today() - datetime.timedelta(days=1))
    prediction_date_str = prediction_date.strftime("%Y-%m-%d")

    if st.button("执行预测"):
        with st.spinner(f"正在使用 {selected_model_name} 对 {prediction_date_str} 的数据进行预测..."):
            try:
                pred_df = predict(selected_model_path, qlib_dir, prediction_date_str)
                st.success("预测完成！")
                st.subheader("预测结果")
                top_n = st.slider("选择要展示的股票数量", 5, 50, 10)
                st.dataframe(pred_df.head(top_n))
                st.subheader(f"Top {top_n} 股票预测分数")
                fig = px.bar(
                    pred_df.head(top_n), x="StockID", y="score", title=f"预测日: {prediction_date_str}",
                    labels={"StockID": "股票代码", "score": "预测分数"}
                )
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"预测过程中发生错误: {e}")

def backtesting_page():
    st.header("策略回测")
    st.markdown("""
    使用已训练好的模型，在指定的历史时间段内进行模拟交易，以评估策略表现。
    - **策略:** 默认使用 Top-K 策略（每日买入预测分数最高的K支股票）。
    - **基准:** 默认使用沪深300指数 (SH000300) 作为比较基准。
    - **结果:** 以图表和关键绩效指标 (KPIs) 的形式展示回测表现。
    """)

    default_data_path = str(Path.home() / ".qlib" / "qlib_data" / "cn_data")
    default_models_path = str(Path.home() / "qlib_models")

    models_dir = st.text_input("模型所在目录", default_models_path, key="models_dir_bt")
    qlib_dir = st.text_input("Qlib 数据存储路径", default_data_path, key="data_dir_bt")

    models_dir_path = Path(models_dir).expanduser()
    available_models = []
    if models_dir_path.exists():
        available_models = [f.name for f in models_dir_path.glob("*.pkl")]

    if not available_models:
        st.warning(f"在 '{models_dir_path}' 中未找到任何 .pkl 模型文件。请先训练模型。")
        return

    selected_model_name = st.selectbox("选择一个模型文件进行回测", available_models)
    selected_model_path = str(models_dir_path / selected_model_name)

    today = datetime.date.today()
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("回测开始日期", today - datetime.timedelta(days=365))
    with col2:
        end_date = st.date_input("回测结束日期", today - datetime.timedelta(days=1))

    if st.button("开始回测"):
        if start_date >= end_date:
            st.error("开始日期必须早于结束日期！")
            return

        start_date_str = start_date.strftime("%Y-%m-%d")
        end_date_str = end_date.strftime("%Y-%m-%d")

        with st.spinner(f"正在回测 {start_date_str} 到 {end_date_str} 的数据..."):
            try:
                report_df = backtest_strategy(selected_model_path, qlib_dir, start_date_str, end_date_str)
                st.success("回测完成！")

                st.subheader("绩效指标")
                metrics = report_df.loc["excess_return_with_cost"]
                cols = st.columns(4)
                cols[0].metric("年化收益率", f"{metrics['annualized_return']:.2%}")
                cols[1].metric("夏普比率", f"{metrics['information_ratio']:.2f}")
                cols[2].metric("最大回撤", f"{metrics['max_drawdown']:.2%}")
                cols[3].metric("换手率", f"{metrics['turnover_rate']:.2f}")

                st.subheader("资金曲线")
                fig = px.line(report_df.index, y=report_df.values, title="策略 vs. 基准")
                fig.update_layout(
                    title_text='策略累积收益 vs. 基准累积收益',
                    xaxis_title='日期', yaxis_title='累积收益', legend_title='图例'
                )
                st.plotly_chart(fig, use_container_width=True)

                st.subheader("详细回测报告")
                st.dataframe(report_df)
            except Exception as e:
                st.error(f"回测过程中发生错误: {e}")

def main():
    st.set_page_config(layout="wide", page_title="Qlib 可视化工具")
    st.sidebar.image("https://avatars.githubusercontent.com/u/65423353?s=200&v=4", width=100)
    st.sidebar.title("Qlib 可视化面板")

    page_options = ["数据管理", "模型训练", "投资组合预测", "策略回测"]
    page = st.sidebar.radio("选择功能页面", page_options)

    if page == "数据管理":
        data_management_page()
    elif page == "模型训练":
        model_training_page()
    elif page == "投资组合预测":
        prediction_page()
    elif page == "策略回测":
        backtesting_page()

if __name__ == "__main__":
    main()
