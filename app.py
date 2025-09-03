import streamlit as st
import qlib
from qlib.constant import REG_CN
import os
import subprocess
import sys
from pathlib import Path
from qlib_utils import (
    SUPPORTED_MODELS, train_model, predict, backtest_strategy,
    download_all_data, update_daily_data, check_data_health
)
import pandas as pd
import plotly.express as px
import datetime
import copy

# --- Streamlit Pages ---

def data_management_page():
    st.header("数据管理")
    st.markdown("在这里，您可以下载、更新和检查 Qlib 所需的股票数据。这是使用本工具的第一步。")
    default_path = str(Path.home() / ".qlib" / "qlib_data")
    qlib_dir = st.text_input("Qlib 数据存储根路径", default_path, key="data_dir_dm")
    qlib_1d_dir = str(Path(qlib_dir) / "cn_data")
    log_placeholder = st.empty()

    st.subheader("1. 全量下载 (首次使用)")
    st.info("如果您是第一次使用，或数据不完整，请点此按钮。这将从头下载所有A股日线数据，过程非常耗时（可能超过30分钟）。")
    if st.button("开始全量下载"):
        with st.spinner("正在执行全量下载..."):
            try:
                download_all_data(qlib_1d_dir, log_placeholder)
                st.success("全量下载命令已成功执行！")
            except Exception as e:
                st.error(f"全量下载过程中发生错误: {e}")

    st.subheader("2. 增量更新 (日常使用)")
    st.info("如果已有全量数据，可在此处更新到指定日期。")
    col1, col2 = st.columns(2)
    start_date = col1.date_input("更新开始日期", datetime.date.today() - datetime.timedelta(days=7))
    end_date = col2.date_input("更新结束日期", datetime.date.today())
    if st.button("开始增量更新"):
        with st.spinner(f"正在更新从 {start_date.strftime('%Y-%m-%d')} 到 {end_date.strftime('%Y-%m-%d')} 的数据..."):
            try:
                update_daily_data(qlib_1d_dir, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"), log_placeholder)
                st.success("增量更新命令已成功执行！")
            except Exception as e:
                st.error(f"增量更新过程中发生错误: {e}")

    st.subheader("3. 数据健康度检查")
    st.info("检查本地数据的完整性和质量。")
    if st.button("开始检查数据健康度"):
        with st.spinner("正在检查数据..."):
            try:
                check_data_health(qlib_1d_dir, log_placeholder)
                st.success("数据健康度检查已完成！详情请查看上方日志。")
            except Exception as e:
                st.error(f"检查过程中发生错误: {e}")

def model_training_page():
    st.header("模型训练")
    default_data_path = str(Path.home() / ".qlib" / "qlib_data" / "cn_data")
    default_models_path = str(Path.home() / "qlib_models")
    qlib_dir = st.text_input("Qlib 数据存储路径", default_data_path, key="data_dir_train")
    models_save_dir = st.text_input("训练后模型的保存路径", default_models_path)
    st.subheader("1. 训练模式与模型配置")
    train_mode = st.radio("选择训练模式", ["从零开始新训练", "在旧模型上继续训练 (Finetune)"], key="train_mode", horizontal=True)
    finetune_model_path = None
    if train_mode == "在旧模型上继续训练 (Finetune)":
        finetune_model_dir = st.text_input("要加载的旧模型所在目录", default_models_path, key="finetune_dir")
        finetune_dir_path = Path(finetune_model_dir).expanduser()
        available_finetune_models = [f.name for f in finetune_dir_path.glob("*.pkl")] if finetune_dir_path.exists() else []
        if available_finetune_models:
            selected_finetune_model = st.selectbox("选择一个要继续训练的模型", available_finetune_models)
            finetune_model_path = str(finetune_dir_path / selected_finetune_model)
        else:
            st.warning(f"在 '{finetune_dir_path}' 中未找到任何 .pkl 模型文件。")
            return
    col1, col2 = st.columns(2)
    model_name_key = col1.selectbox("选择模型和因子", list(SUPPORTED_MODELS.keys()))
    stock_pool = col2.selectbox("选择股票池", ["csi300", "csi500"], index=0)
    custom_model_name = st.text_input("为新模型命名 (可选, 留空则使用默认名)")
    if "ALSTM" in model_name_key:
        st.warning("️️️**注意：** ALSTM是深度学习模型，训练时间非常长，对电脑性能要求很高。")
    st.subheader("2. 超参数调节")
    config = copy.deepcopy(SUPPORTED_MODELS[model_name_key])
    params = config['task']['model']['kwargs']
    with st.expander("调节模型参数", expanded=True):
        if any(m in model_name_key for m in ["LightGBM", "XGBoost", "CatBoost"]):
            if "CatBoost" in model_name_key:
                params['iterations'] = st.slider("迭代次数", 50, 500, params.get('iterations', 200), 10, key=f"it_{model_name_key}")
                params['depth'] = st.slider("最大深度", 3, 15, params.get('depth', 7), key=f"depth_{model_name_key}")
            else:
                params['n_estimators'] = st.slider("树的数量", 50, 500, params.get('n_estimators', 200), 10, key=f"n_est_{model_name_key}")
                params['max_depth'] = st.slider("最大深度", 3, 15, params.get('max_depth', 7), key=f"depth_{model_name_key}")
            params['learning_rate'] = st.slider("学习率", 0.01, 0.2, params.get('learning_rate', 0.05), 0.01, key=f"lr_{model_name_key}")
        elif "ALSTM" in model_name_key:
            st.info("ALSTM模型的超参数调节暂未在此界面支持。")
    if st.button("开始训练"):
        with st.spinner("正在准备训练环境..."):
            try:
                saved_path = train_model(model_name_key, qlib_dir, models_save_dir, config, custom_model_name if custom_model_name else None, stock_pool, finetune_model_path)
                st.success(f"模型训练成功！已保存至: {saved_path}")
                st.balloons()
            except Exception as e:
                st.error(f"训练过程中发生错误: {e}")

def prediction_page():
    st.header("投资组合预测")
    st.markdown("使用一个或多个已训练好的模型，对指定日期的股票进行评分预测，并进行对比。")
    default_data_path = str(Path.home() / ".qlib" / "qlib_data" / "cn_data")
    default_models_path = str(Path.home() / "qlib_models")
    models_dir = st.text_input("模型所在目录", default_models_path, key="models_dir_pred")
    qlib_dir = st.text_input("Qlib 数据存储路径", default_data_path, key="data_dir_pred")
    models_dir_path = Path(models_dir).expanduser()
    available_models = [f.name for f in models_dir_path.glob("*.pkl")] if models_dir_path.exists() else []
    if not available_models:
        st.warning(f"在 '{models_dir_path}' 中未找到模型。")
        return
    selected_models = st.multiselect("选择一个或多个模型进行对比预测", available_models)
    prediction_date = st.date_input("选择预测日期", datetime.date.today() - datetime.timedelta(days=1))
    if st.button("执行对比预测") and selected_models:
        with st.spinner("正在执行预测..."):
            try:
                all_preds = []
                for model_name in selected_models:
                    model_path = str(models_dir_path / model_name)
                    pred_df = predict(model_path, qlib_dir, prediction_date.strftime("%Y-%m-%d"))
                    pred_df = pred_df.rename(columns={"score": f"score_{model_name.replace('.pkl', '')}"})
                    all_preds.append(pred_df.set_index('StockID')[f"score_{model_name.replace('.pkl', '')}"])
                combined_df = pd.concat(all_preds, axis=1).reset_index()
                st.success("预测完成！")
                st.subheader("多模型预测结果对比")
                st.dataframe(combined_df)
                st.subheader("Top-10 股票分数对比图")
                score_cols = [col for col in combined_df.columns if 'score' in col]
                combined_df['average_score'] = combined_df[score_cols].mean(axis=1)
                top_10_stocks = combined_df.nlargest(10, 'average_score')
                plot_df = top_10_stocks.melt(id_vars=['StockID'], value_vars=score_cols, var_name='Model', value_name='Score')
                plot_df['Model'] = plot_df['Model'].str.replace('score_', '')
                fig = px.bar(plot_df, x="StockID", y="Score", color="Model", barmode='group', title="Top-10 股票多模型分数对比")
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
    start_date = col1.date_input("开始日期", datetime.date.today() - datetime.timedelta(days=365))
    end_date = col2.date_input("结束日期", datetime.date.today() - datetime.timedelta(days=1))
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
            with st.spinner("正在回测..."):
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
    page = st.sidebar.radio("选择功能页面", page_options, horizontal=True)
    if page == "数据管理": data_management_page()
    elif page == "模型训练": model_training_page()
    elif page == "投资组合预测": prediction_page()
    elif page == "策略回测": backtesting_page()

if __name__ == "__main__":
    main()
