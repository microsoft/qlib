"""
完整训练 + 回测 + 可视化 workflow

基于 workflow_config_lightgbm_Alpha360.yaml 的配置，
增加回测结果的可视化输出。

运行方式：
    python examples/my_workflow_with_visual.py

如果不在 notebook 中运行，图表将保存为 HTML 文件到当前目录。
"""

import pandas as pd
import qlib
from qlib.constant import REG_CN
from qlib.utils import init_instance_by_config
from qlib.utils.time import Freq
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord, SigAnaRecord
from qlib.data import D
from qlib.backtest import backtest
from qlib.backtest.executor import SimulatorExecutor
from qlib.contrib.evaluate import risk_analysis
from qlib.contrib.strategy import TopkDropoutStrategy

# ── 可视化模块 ──────────────────────────────────────────
from qlib.contrib.report.analysis_position import (
    report_graph,
    score_ic_graph,
    risk_analysis_graph,
    cumulative_return_graph,
)

# ── 判断是否在 notebook 环境 ──────────────────────────────
try:
    get_ipython()
    IN_NOTEBOOK = True
except NameError:
    IN_NOTEBOOK = False


def show_or_save(figs, filename):
    """在 notebook 中内嵌显示，否则保存为 HTML"""
    if IN_NOTEBOOK:
        from qlib.contrib.report.graph import BaseGraph

        BaseGraph.show_graph_in_notebook(figs)
        print(f"[notebook] {filename} 已显示")
    else:
        html = "\n".join(fig.to_html(include_plotlyjs="cdn") for fig in figs)
        path = f"{filename}.html"
        with open(path, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"[file] 图表已保存 -> {path}")


def main():
    # ── 1. 初始化 Qlib ────────────────────────────────────
    provider_uri = "~/.qlib/qlib_data/cn_data"
    qlib.init(provider_uri=provider_uri, region=REG_CN)

    market = "csi300"
    benchmark = "SH000300"

    # ── 2. 数据集配置（与 Alpha360 yaml 一致）─────────────
    dataset_config = {
        "class": "DatasetH",
        "module_path": "qlib.data.dataset",
        "kwargs": {
            "handler": {
                "class": "Alpha360",
                "module_path": "qlib.contrib.data.handler",
                "kwargs": {
                    "start_time": "2008-01-01",
                    "end_time": "2020-08-01",
                    "fit_start_time": "2008-01-01",
                    "fit_end_time": "2014-12-31",
                    "instruments": market,
                    "infer_processors": [],
                    "learn_processors": [
                        {"class": "DropnaLabel"},
                        {"class": "CSRankNorm", "kwargs": {"fields_group": "label"}},
                    ],
                    "label": ["Ref($close, -2) / Ref($close, -1) - 1"],
                },
            },
            "segments": {
                "train": ("2008-01-01", "2014-12-31"),
                "valid": ("2015-01-01", "2016-12-31"),
                "test": ("2017-01-01", "2020-08-01"),
            },
        },
    }

    # ── 3. 模型配置（与 Alpha360 yaml 一致）───────────────
    model_config = {
        "class": "LGBModel",
        "module_path": "qlib.contrib.model.gbdt",
        "kwargs": {
            "loss": "mse",
            "colsample_bytree": 0.8879,
            "learning_rate": 0.0421,
            "subsample": 0.8789,
            "lambda_l1": 205.6999,
            "lambda_l2": 580.9768,
            "max_depth": 8,
            "num_leaves": 210,
            "num_threads": 20,
        },
    }

    # ── 4. 实例化 dataset 和 model ────────────────────────
    dataset = init_instance_by_config(dataset_config)
    model = init_instance_by_config(model_config)

    # 看一眼训练集数据
    df_train = dataset.prepare("train", col_set=["feature", "label"])
    print(f"训练集: features={df_train['feature'].shape}, labels={df_train['label'].shape}")

    # ── 5. 训练 + 预测 ────────────────────────────────────
    with R.start(experiment_name="my_workflow_visual", recorder_name="run1"):
        model.fit(dataset)
        recorder = R.get_recorder()

        sr = SignalRecord(model, dataset, recorder)
        sr.generate()

        sar = SigAnaRecord(recorder, ana_long_short=False, ann_scaler=252)
        sar.generate()

        # 获取预测信号
        pred_df = recorder.load_object("pred.pkl")
        print(f"预测信号: {pred_df.shape}, index={pred_df.index.names}")

    # ── 6. 回测 ───────────────────────────────────────────
    STRATEGY_CONFIG = {
        "topk": 50,
        "n_drop": 5,
        "signal": pred_df,
    }
    EXECUTOR_CONFIG = {
        "time_per_step": "day",
        "generate_portfolio_metrics": True,
    }
    backtest_config = {
        "start_time": "2017-01-01",
        "end_time": "2020-08-01",
        "account": 100000000,
        "benchmark": benchmark,
        "exchange_kwargs": {
            "freq": "day",
            "limit_threshold": 0.095,
            "deal_price": "close",
            "open_cost": 0.0005,
            "close_cost": 0.0015,
            "min_cost": 5,
        },
    }

    strategy_obj = TopkDropoutStrategy(**STRATEGY_CONFIG)
    executor_obj = SimulatorExecutor(**EXECUTOR_CONFIG)
    portfolio_metric_dict, indicator_dict = backtest(
        executor=executor_obj, strategy=strategy_obj, **backtest_config
    )

    analysis_freq = f"{Freq.parse('day')[0]}{Freq.parse('day')[1]}"
    report_normal_df, positions = portfolio_metric_dict.get(analysis_freq)
    print(f"\n回测报告: {report_normal_df.shape}, columns={list(report_normal_df.columns)}")

    # ── 7. 打印风险指标 ────────────────────────────────────
    print("\n" + "=" * 60)
    print("基准收益 (1day)")
    print(risk_analysis(report_normal_df["bench"]))
    print("\n超额收益（无成本）")
    print(risk_analysis(report_normal_df["return"] - report_normal_df["bench"]))
    print("\n超额收益（含成本）")
    print(
        risk_analysis(
            report_normal_df["return"] - report_normal_df["bench"] - report_normal_df["cost"]
        )
    )
    print("=" * 60)

    # ── 8. 可视化 ─────────────────────────────────────────

    # 8a. 回测全景图
    figs = report_graph(report_normal_df, show_notebook=False)
    show_or_save(figs, "report_overview")

    # 8b. IC 分析图
    pred_df_dates = pred_df.index.get_level_values("datetime")
    label_data = D.features(
        D.instruments(market),
        ["Ref($close, -2) / Ref($close, -1) - 1"],
        pred_df_dates.min(),
        pred_df_dates.max(),
    )
    label_data.columns = ["label"]
    pred_label = pd.concat([label_data, pred_df], axis=1, sort=True).reindex(label_data.index)
    print(f"\nIC 数据: {pred_label.shape}")

    figs = score_ic_graph(pred_label, show_notebook=False)
    show_or_save(figs, "score_ic")

    # 8c. 风险分析图
    analysis = {}
    analysis["excess_return_without_cost"] = risk_analysis(
        report_normal_df["return"] - report_normal_df["bench"]
    )
    analysis["excess_return_with_cost"] = risk_analysis(
        report_normal_df["return"] - report_normal_df["bench"] - report_normal_df["cost"]
    )
    analysis_df = pd.concat(analysis)
    print(f"\n风险分析: {analysis_df}")

    figs = risk_analysis_graph(analysis_df, report_normal_df, show_notebook=False)
    show_or_save(figs, "risk_analysis")

    # 8d. 买卖持仓分析
    figs = cumulative_return_graph(
        positions, report_normal_df, label_data, show_notebook=False,
        start_date="2017-01-01", end_date="2020-08-01",
    )
    show_or_save(figs, "cumulative_return")

    # ── 9. 打印关键指标摘要 ────────────────────────────────
    print("\n" + "=" * 60)
    print("关键指标摘要")
    print("=" * 60)
    total_cost = report_normal_df["cost"].sum()
    total_turnover = report_normal_df["turnover"].sum()
    avg_turnover = report_normal_df["turnover"].mean()
    # 估算单边换手率 = turnover / 2（因为一买一卖各算一次）
    print(f"累计成本:        {total_cost:.4f}")
    print(f"总换手率:         {total_turnover:.2f}")
    print(f"日均换手率:       {avg_turnover:.4f}")
    print(f"成本/换手率:      {total_cost / total_turnover * 10000:.1f} bps (每次交易平均成本)")

    print("\n完成！所有图表已输出。")


if __name__ == "__main__":
    main()
