# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
绘图工具模块 - 生成回测分析图表

功能:
1. 收益曲线图 (累计收益、超额收益)
2. 回撤图 (Drawdown)
3. IC分析图 (时间序列、月度热力图、直方图)
4. 月度收益热力图
5. 风险指标统计
6. 按时间和模型分文件夹保存图片
"""

import os
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import rcParams

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 120
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['savefig.bbox'] = 'tight'


def get_plot_output_dir(base_dir: Path, experiment_name: str = None) -> Path:
    """
    获取图片输出目录，按日期和实验名组织

    Parameters
    ----------
    base_dir : Path
        基础目录
    experiment_name : str
        实验名称

    Returns
    -------
    Path
        输出目录路径
    """
    today = datetime.now().strftime("%Y-%m-%d")
    if experiment_name:
        output_dir = base_dir / "plots" / today / experiment_name
    else:
        output_dir = base_dir / "plots" / today

    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def save_fig(fig: plt.Figure, filename: str, output_dir: Path) -> str:
    """保存图片并返回路径"""
    filepath = output_dir / filename
    fig.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return str(filepath)


def calculate_cumulative_returns(returns: pd.Series) -> pd.Series:
    """计算累计收益"""
    return (1 + returns).cumprod() - 1


def calculate_drawdown(returns: pd.Series) -> pd.Series:
    """计算回撤序列"""
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    return drawdown


def plot_portfolio_performance(
    report_df: pd.DataFrame,
    output_dir: Path,
    title_prefix: str = ""
) -> Dict[str, str]:
    """
    绘制投资组合表现综合图

    Parameters
    ----------
    report_df : pd.DataFrame
        包含 'return', 'bench', 'cost', 'turnover' 等列的DataFrame
    output_dir : Path
        输出目录
    title_prefix : str
        标题前缀

    Returns
    -------
    Dict[str, str]
        保存的文件路径字典
    """
    saved_files = {}

    # 准备数据
    if isinstance(report_df.index, pd.DatetimeIndex):
        report_df.index = pd.to_datetime(report_df.index)

    portfolio_return = report_df['return'].dropna()
    bench_return = report_df['bench'].dropna() if 'bench' in report_df.columns else None

    # 对齐数据
    common_idx = portfolio_return.index
    if bench_return is not None:
        common_idx = common_idx.intersection(bench_return.index)
        portfolio_return = portfolio_return.loc[common_idx]
        bench_return = bench_return.loc[common_idx]

    # 计算累计收益
    cum_portfolio = calculate_cumulative_returns(portfolio_return)
    cum_bench = calculate_cumulative_returns(bench_return) if bench_return is not None else None
    cum_excess = cum_portfolio - cum_bench if cum_bench is not None else cum_portfolio

    # 计算回撤
    drawdown = calculate_drawdown(portfolio_return)

    # ========== 图1: 累计收益对比 ==========
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # 上图: 累计收益曲线
    ax1 = axes[0]
    ax1.plot(cum_portfolio.index, cum_portfolio.values * 100,
             label='Portfolio', linewidth=1.5, color='#2E86AB')
    if cum_bench is not None:
        ax1.plot(cum_bench.index, cum_bench.values * 100,
                 label='Benchmark (CSI300)', linewidth=1.5, color='#E94F37')
    ax1.plot(cum_excess.index, cum_excess.values * 100,
             label='Excess Return', linewidth=1.5, color='#F39C12', linestyle='--')

    ax1.set_title(f'{title_prefix}Cumulative Returns Comparison', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Cumulative Return (%)', fontsize=11)
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)

    # 填充超额收益区域
    if cum_bench is not None:
        ax1.fill_between(cum_portfolio.index, cum_portfolio.values * 100,
                         cum_bench.values * 100, alpha=0.2, color='green',
                         label='Excess')

    # 下图: 回撤图
    ax2 = axes[1]
    ax2.fill_between(drawdown.index, drawdown.values * 100, 0,
                     alpha=0.4, color='#E74C3C')
    ax2.plot(drawdown.index, drawdown.values * 100,
             linewidth=1, color='#C0392B')
    ax2.set_title(f'{title_prefix}Portfolio Drawdown', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Drawdown (%)', fontsize=11)
    ax2.set_xlabel('Date', fontsize=11)
    ax2.grid(True, alpha=0.3)

    # 标记最大回撤
    max_dd_idx = drawdown.idxmin()
    max_dd = drawdown.min() * 100
    ax2.annotate(f'Max DD: {max_dd:.2f}%',
                 xy=(max_dd_idx, max_dd),
                 xytext=(max_dd_idx, max_dd - 5),
                 fontsize=10,
                 arrowprops=dict(arrowstyle='->', color='black'),
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    saved_files['cumulative_returns'] = save_fig(fig, '01_cumulative_returns.png', output_dir)

    # ========== 图2: 收益和换手率分析 ==========
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))

    # 月度收益
    monthly_returns = portfolio_return.resample('ME').apply(lambda x: (1 + x).prod() - 1)
    colors = ['#27AE60' if x >= 0 else '#E74C3C' for x in monthly_returns.values]

    ax1 = axes[0]
    ax1.bar(range(len(monthly_returns)), monthly_returns.values * 100, color=colors, alpha=0.8)
    ax1.set_xticks(range(len(monthly_returns)))
    ax1.set_xticklabels([d.strftime('%Y-%m') for d in monthly_returns.index], rotation=45, ha='right')
    ax1.set_title(f'{title_prefix}Monthly Returns', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Monthly Return (%)', fontsize=11)
    ax1.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
    ax1.grid(True, alpha=0.3, axis='y')

    # 日收益分布
    ax2 = axes[1]
    ax2.hist(portfolio_return.values * 100, bins=50, color='#3498DB',
             alpha=0.7, edgecolor='white', density=True)
    ax2.axvline(x=portfolio_return.mean() * 100, color='red', linestyle='--',
                label=f'Mean: {portfolio_return.mean()*100:.3f}%')
    ax2.axvline(x=0, color='gray', linestyle='-', linewidth=0.5)
    ax2.set_title(f'{title_prefix}Daily Return Distribution', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Daily Return (%)', fontsize=11)
    ax2.set_ylabel('Density', fontsize=11)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    # 换手率
    if 'turnover' in report_df.columns:
        turnover = report_df['turnover'].dropna()
        ax3 = axes[2]
        ax3.plot(turnover.index, turnover.values * 100, linewidth=0.8, color='#9B59B6')
        ax3.fill_between(turnover.index, turnover.values * 100, alpha=0.3, color='#9B59B6')
        ax3.set_title(f'{title_prefix}Daily Turnover Rate', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Turnover (%)', fontsize=11)
        ax3.set_xlabel('Date', fontsize=11)
        ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    saved_files['monthly_returns'] = save_fig(fig, '02_monthly_returns.png', output_dir)

    return saved_files


def plot_ic_analysis(
    ic_df: pd.DataFrame,
    ric_df: pd.DataFrame = None,
    output_dir: Path = None,
    title_prefix: str = ""
) -> Dict[str, str]:
    """
    绘制IC分析图

    Parameters
    ----------
    ic_df : pd.DataFrame
        IC时间序列
    ric_df : pd.DataFrame
        Rank IC时间序列
    output_dir : Path
        输出目录
    title_prefix : str
        标题前缀

    Returns
    -------
    Dict[str, str]
        保存的文件路径字典
    """
    saved_files = {}

    if ic_df.empty:
        return saved_files

    # 准备数据
    if isinstance(ic_df.index, pd.DatetimeIndex):
        ic_df.index = pd.to_datetime(ic_df.index)

    # 处理DataFrame或Series
    if hasattr(ic_df, 'columns'):
        ic_series = ic_df.iloc[:, 0] if len(ic_df.columns) > 0 else ic_df.iloc[:, 0]
    else:
        ic_series = ic_df  # 已经是Series
    if hasattr(ic_series, 'dropna'):
        ic_series = ic_series.dropna()

    ric_series = None
    if ric_df is not None and not ric_df.empty:
        if isinstance(ric_df.index, pd.DatetimeIndex):
            ric_df.index = pd.to_datetime(ric_df.index)
        if hasattr(ric_df, 'columns'):
            ric_series = ric_df.iloc[:, 0] if len(ric_df.columns) > 0 else ric_df.iloc[:, 0]
        else:
            ric_series = ric_df  # 已经是Series
        if hasattr(ric_series, 'dropna'):
            ric_series = ric_series.dropna()

    # ========== 图1: IC时间序列和月度IC热力图 ==========
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1.2])

    # IC时间序列
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(ic_series.index, ic_series.values, linewidth=1, color='#3498DB', label='IC')
    ax1.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
    ax1.axhline(y=ic_series.mean(), color='red', linestyle='--', linewidth=1,
                label=f'Mean: {ic_series.mean():.4f}')
    ax1.axhline(y=0.05, color='green', linestyle=':', linewidth=1, alpha=0.7)
    ax1.axhline(y=-0.05, color='green', linestyle=':', linewidth=1, alpha=0.7)
    ax1.fill_between(ic_series.index, ic_series.values, 0,
                     where=(ic_series.values > 0), alpha=0.3, color='green')
    ax1.fill_between(ic_series.index, ic_series.values, 0,
                     where=(ic_series.values < 0), alpha=0.3, color='red')
    ax1.set_title(f'{title_prefix}IC Time Series', fontsize=14, fontweight='bold')
    ax1.set_ylabel('IC', fontsize=11)
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3)

    # IC统计信息
    ic_stats = {
        'IC Mean': ic_series.mean(),
        'IC Std': ic_series.std(),
        'IC IR': ic_series.mean() / ic_series.std() if ic_series.std() > 0 else 0,
        'IC > 0 Ratio': (ic_series > 0).mean(),
        '|IC| > 0.02 Ratio': (ic_series.abs() > 0.02).mean(),
        '|IC| > 0.05 Ratio': (ic_series.abs() > 0.05).mean(),
    }
    stats_text = '\n'.join([f'{k}: {v:.4f}' for k, v in ic_stats.items()])
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, fontsize=9,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # 月度IC热力图
    ax2 = fig.add_subplot(gs[1, 0])
    ic_monthly = ic_series.resample('ME').mean()
    ic_pivot = ic_monthly.to_frame()
    ic_pivot['year'] = ic_pivot.index.year
    ic_pivot['month'] = ic_pivot.index.month
    ic_matrix = ic_pivot.pivot(index='year', columns='month', values=ic_pivot.columns[0])

    # 绘制热力图
    im = ax2.imshow(ic_matrix.values, cmap='RdYlGn', aspect='auto', vmin=-0.1, vmax=0.1)
    ax2.set_xticks(range(12))
    ax2.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                         'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    ax2.set_yticks(range(len(ic_matrix.index)))
    ax2.set_yticklabels(ic_matrix.index)
    ax2.set_title(f'{title_prefix}Monthly IC Heatmap', fontsize=14, fontweight='bold')

    # 添加数值标注
    for i in range(len(ic_matrix.index)):
        for j in range(len(ic_matrix.columns)):
            val = ic_matrix.iloc[i, j]
            if not np.isnan(val):
                color = 'white' if abs(val) > 0.05 else 'black'
                ax2.text(j, i, f'{val:.3f}', ha='center', va='center', fontsize=8, color=color)

    plt.colorbar(im, ax=ax2, label='IC')

    # IC分布直方图
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.hist(ic_series.values, bins=50, color='#3498DB', alpha=0.7,
             edgecolor='white', density=True, label='IC Distribution')

    # 拟合正态分布
    from scipy import stats
    x = np.linspace(ic_series.min(), ic_series.max(), 100)
    pdf = stats.norm.pdf(x, ic_series.mean(), ic_series.std())
    ax3.plot(x, pdf, 'r-', linewidth=2, label='Normal Fit')

    ax3.axvline(x=0, color='gray', linestyle='-', linewidth=0.5)
    ax3.axvline(x=ic_series.mean(), color='red', linestyle='--', linewidth=1,
                label=f'Mean: {ic_series.mean():.4f}')
    ax3.set_title(f'{title_prefix}IC Distribution', fontsize=14, fontweight='bold')
    ax3.set_xlabel('IC', fontsize=11)
    ax3.set_ylabel('Density', fontsize=11)
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    saved_files['ic_analysis'] = save_fig(fig, '03_ic_analysis.png', output_dir)

    # ========== 图2: Rank IC分析 (如果可用) ==========
    if ric_series is not None and not ric_series.empty:
        fig, axes = plt.subplots(2, 1, figsize=(14, 8))

        ax1 = axes[0]
        ax1.plot(ric_series.index, ric_series.values, linewidth=1, color='#9B59B6', label='Rank IC')
        ax1.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
        ax1.axhline(y=ric_series.mean(), color='red', linestyle='--', linewidth=1,
                    label=f'Mean: {ric_series.mean():.4f}')
        ax1.fill_between(ric_series.index, ric_series.values, 0,
                         where=(ric_series.values > 0), alpha=0.3, color='green')
        ax1.fill_between(ric_series.index, ric_series.values, 0,
                         where=(ric_series.values < 0), alpha=0.3, color='red')
        ax1.set_title(f'{title_prefix}Rank IC Time Series', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Rank IC', fontsize=11)
        ax1.legend(loc='upper right', fontsize=10)
        ax1.grid(True, alpha=0.3)

        ric_stats = {
            'Rank IC Mean': ric_series.mean(),
            'Rank IC Std': ric_series.std(),
            'Rank IC IR': ric_series.mean() / ric_series.std() if ric_series.std() > 0 else 0,
            'Rank IC > 0 Ratio': (ric_series > 0).mean(),
        }
        stats_text = '\n'.join([f'{k}: {v:.4f}' for k, v in ric_stats.items()])
        ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, fontsize=9,
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        ax2 = axes[1]
        ax2.hist(ric_series.values, bins=50, color='#9B59B6', alpha=0.7,
                 edgecolor='white', density=True, label='Rank IC Distribution')
        x = np.linspace(ric_series.min(), ric_series.max(), 100)
        pdf = stats.norm.pdf(x, ric_series.mean(), ric_series.std())
        ax2.plot(x, pdf, 'r-', linewidth=2, label='Normal Fit')
        ax2.axvline(x=0, color='gray', linestyle='-', linewidth=0.5)
        ax2.set_title(f'{title_prefix}Rank IC Distribution', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Rank IC', fontsize=11)
        ax2.set_ylabel('Density', fontsize=11)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        saved_files['rank_ic_analysis'] = save_fig(fig, '04_rank_ic_analysis.png', output_dir)

    return saved_files


def plot_risk_metrics(
    port_analysis_df: pd.DataFrame,
    output_dir: Path,
    title_prefix: str = ""
) -> Dict[str, str]:
    """
    绘制风险指标柱状图

    Parameters
    ----------
    port_analysis_df : pd.DataFrame
        风险分析DataFrame
    output_dir : Path
        输出目录
    title_prefix : str
        标题前缀

    Returns
    -------
    Dict[str, str]
        保存的文件路径字典
    """
    saved_files = {}

    if port_analysis_df.empty:
        return saved_files

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # 无成本超额收益指标
    ax1 = axes[0]
    if 'excess_return_without_cost' in port_analysis_df.index:
        metrics = port_analysis_df.loc['excess_return_without_cost']
        if isinstance(metrics, pd.DataFrame):
            metrics = metrics.iloc[:, 0]
        metrics_dict = {
            'Annualized\nReturn': metrics.get('annualized_return', 0) * 100,
            'Information\nRatio': metrics.get('information_ratio', 0),
            'Std': metrics.get('std', 0) * 100,
            'Max\nDrawdown': abs(metrics.get('max_drawdown', 0)) * 100,
        }
    else:
        # 尝试直接获取
        metrics = port_analysis_df.iloc[:, 0] if len(port_analysis_df.columns) > 0 else port_analysis_df
        metrics_dict = {
            'Annualized\nReturn': metrics.get('annualized_return', 0) * 100 if hasattr(metrics, 'get') else 0,
            'Information\nRatio': metrics.get('information_ratio', 0) if hasattr(metrics, 'get') else 0,
        }

    colors = ['#27AE60', '#3498DB', '#E74C3C', '#F39C12']
    bars = ax1.bar(metrics_dict.keys(), metrics_dict.values(), color=colors, alpha=0.8)
    ax1.set_title(f'{title_prefix}Risk Metrics (Without Cost)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Value', fontsize=11)
    ax1.grid(True, alpha=0.3, axis='y')

    for bar, val in zip(bars, metrics_dict.values()):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f'{val:.2f}', ha='center', va='bottom', fontsize=10)

    # 有成本超额收益指标
    ax2 = axes[1]
    if 'excess_return_with_cost' in port_analysis_df.index:
        metrics = port_analysis_df.loc['excess_return_with_cost']
        if isinstance(metrics, pd.DataFrame):
            metrics = metrics.iloc[:, 0]
        metrics_dict = {
            'Annualized\nReturn': metrics.get('annualized_return', 0) * 100,
            'Information\nRatio': metrics.get('information_ratio', 0),
            'Std': metrics.get('std', 0) * 100,
            'Max\nDrawdown': abs(metrics.get('max_drawdown', 0)) * 100,
        }
    else:
        metrics_dict = {}

    colors = ['#27AE60', '#3498DB', '#E74C3C', '#F39C12']
    bars = ax2.bar(metrics_dict.keys(), metrics_dict.values(), color=colors, alpha=0.8)
    ax2.set_title(f'{title_prefix}Risk Metrics (With Cost)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Value', fontsize=11)
    ax2.grid(True, alpha=0.3, axis='y')

    for bar, val in zip(bars, metrics_dict.values()):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f'{val:.2f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    saved_files['risk_metrics'] = save_fig(fig, '05_risk_metrics.png', output_dir)

    return saved_files


def plot_monthly_returns_heatmap(
    report_df: pd.DataFrame,
    output_dir: Path,
    title_prefix: str = ""
) -> Dict[str, str]:
    """
    绘制月度收益热力图

    Parameters
    ----------
    report_df : pd.DataFrame
        包含 'return' 列的DataFrame
    output_dir : Path
        输出目录
    title_prefix : str
        标题前缀

    Returns
    -------
    Dict[str, str]
        保存的文件路径字典
    """
    saved_files = {}

    if 'return' not in report_df.columns or report_df.empty:
        return saved_files

    if isinstance(report_df.index, pd.DatetimeIndex):
        report_df.index = pd.to_datetime(report_df.index)

    returns = report_df['return'].dropna()

    # 计算月度收益
    monthly_returns = returns.resample('ME').apply(lambda x: (1 + x).prod() - 1)

    # 创建透视表
    monthly_df = monthly_returns.to_frame()
    monthly_df['year'] = monthly_df.index.year
    monthly_df['month'] = monthly_df.index.month
    pivot = monthly_df.pivot(index='year', columns='month', values='return')

    # 绘制热力图
    fig, ax = plt.subplots(figsize=(14, 8))

    im = ax.imshow(pivot.values * 100, cmap='RdYlGn', aspect='auto', vmin=-15, vmax=15)

    # 设置标签
    ax.set_xticks(range(12))
    ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)

    ax.set_title(f'{title_prefix}Monthly Returns Heatmap (%)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Month', fontsize=11)
    ax.set_ylabel('Year', fontsize=11)

    # 添加数值标注
    for i in range(len(pivot.index)):
        for j in range(12):
            if j < len(pivot.columns):
                val = pivot.iloc[i, j]
                if not np.isnan(val):
                    color = 'white' if abs(val) > 7 else 'black'
                    ax.text(j, i, f'{val:.1f}', ha='center', va='center',
                            fontsize=9, color=color, fontweight='bold')

    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax, label='Monthly Return (%)')
    cbar.ax.tick_params(labelsize=10)

    plt.tight_layout()
    saved_files['monthly_heatmap'] = save_fig(fig, '06_monthly_returns_heatmap.png', output_dir)

    return saved_files


def generate_summary_statistics(
    report_df: pd.DataFrame,
    ic_df: pd.DataFrame,
    port_analysis_df: pd.DataFrame = None,
    ric_df: pd.DataFrame = None
) -> Dict[str, Any]:
    """
    生成汇总统计信息

    Parameters
    ----------
    report_df : pd.DataFrame
        投资组合报告数据
    ic_df : pd.DataFrame
        IC数据
    port_analysis_df : pd.DataFrame
        风险分析数据
    ric_df : pd.DataFrame
        Rank IC数据

    Returns
    -------
    Dict[str, Any]
        统计信息字典
    """
    stats = {}

    if report_df is not None and not report_df.empty:
        if isinstance(report_df.index, pd.DatetimeIndex):
            report_df.index = pd.to_datetime(report_df.index)

        returns = report_df['return'].dropna()
        bench = report_df['bench'].dropna() if 'bench' in report_df.columns else None

        # 对齐
        common_idx = returns.index
        if bench is not None:
            common_idx = common_idx.intersection(bench.index)
            returns = returns.loc[common_idx]
            bench = bench.loc[common_idx]

        # 计算累计收益
        cum_return = (1 + returns).prod() - 1
        cum_bench = (1 + bench).prod() - 1 if bench is not None else 0
        excess_return = cum_return - cum_bench

        # 日收益统计
        stats['daily_return'] = {
            'mean': returns.mean(),
            'std': returns.std(),
            'min': returns.min(),
            'max': returns.max(),
            'skewness': returns.skew(),
            'kurtosis': returns.kurtosis(),
            'positive_days': (returns > 0).sum(),
            'negative_days': (returns < 0).sum(),
            'total_days': len(returns),
        }

        # 整体表现
        stats['performance'] = {
            'cumulative_return': cum_return,
            'cumulative_benchmark': cum_bench,
            'excess_return': excess_return,
            'annualized_return': returns.mean() * 252,
            'annualized_volatility': returns.std() * np.sqrt(252),
            'sharpe_ratio': (returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0,
        }

        # 回撤
        cum = (1 + returns).cumprod()
        running_max = cum.cummax()
        drawdown = (cum - running_max) / running_max
        stats['drawdown'] = {
            'max_drawdown': drawdown.min(),
            'max_drawdown_date': drawdown.idxmin(),
            'avg_drawdown': drawdown.mean(),
        }

        # 月度统计
        monthly = returns.resample('ME').apply(lambda x: (1 + x).prod() - 1)
        stats['monthly'] = {
            'mean': monthly.mean(),
            'std': monthly.std(),
            'positive_months': (monthly > 0).sum(),
            'total_months': len(monthly),
            'best_month': monthly.max(),
            'worst_month': monthly.min(),
        }

    if ic_df is not None and not ic_df.empty:
        if hasattr(ic_df, 'columns'):
            ic_series = ic_df.iloc[:, 0] if len(ic_df.columns) > 0 else ic_df.iloc[:, 0]
        else:
            ic_series = ic_df
        if hasattr(ic_series, 'dropna'):
            ic_series = ic_series.dropna()

        stats['ic'] = {
            'mean': ic_series.mean(),
            'std': ic_series.std(),
            'ir': ic_series.mean() / ic_series.std() if ic_series.std() > 0 else 0,
            'positive_ratio': (ic_series > 0).mean(),
            'ic_0_02': (ic_series.abs() > 0.02).mean(),
            'ic_0_05': (ic_series.abs() > 0.05).mean(),
            'ic_0_10': (ic_series.abs() > 0.10).mean(),
        }

    if ric_df is not None and not ric_df.empty:
        if hasattr(ric_df, 'columns'):
            ric_series = ric_df.iloc[:, 0] if len(ric_df.columns) > 0 else ric_df.iloc[:, 0]
        else:
            ric_series = ric_df
        if hasattr(ric_series, 'dropna'):
            ric_series = ric_series.dropna()

        stats['rank_ic'] = {
            'mean': ric_series.mean(),
            'std': ric_series.std(),
            'ir': ric_series.mean() / ric_series.std() if ric_series.std() > 0 else 0,
            'positive_ratio': (ric_series > 0).mean(),
        }

    if port_analysis_df is not None and not port_analysis_df.empty:
        try:
            if 'excess_return_without_cost' in port_analysis_df.index:
                metrics = port_analysis_df.loc['excess_return_without_cost']
                if isinstance(metrics, pd.DataFrame):
                    metrics = metrics.iloc[:, 0]
                stats['risk_no_cost'] = {
                    'annualized_return': metrics.get('annualized_return', 0),
                    'information_ratio': metrics.get('information_ratio', 0),
                    'max_drawdown': metrics.get('max_drawdown', 0),
                }

            if 'excess_return_with_cost' in port_analysis_df.index:
                metrics = port_analysis_df.loc['excess_return_with_cost']
                if isinstance(metrics, pd.DataFrame):
                    metrics = metrics.iloc[:, 0]
                stats['risk_with_cost'] = {
                    'annualized_return': metrics.get('annualized_return', 0),
                    'information_ratio': metrics.get('information_ratio', 0),
                    'max_drawdown': metrics.get('max_drawdown', 0),
                }
        except:
            pass

    return stats


def print_statistics(stats: Dict[str, Any]) -> None:
    """打印统计信息"""
    print("\n" + "="*60)
    print("                    策略表现汇总统计")
    print("="*60)

    if 'performance' in stats:
        p = stats['performance']
        print("\n【整体表现】")
        print(f"  累计收益:     {p.get('cumulative_return', 0)*100:.2f}%")
        print(f"  基准收益:     {p.get('cumulative_benchmark', 0)*100:.2f}%")
        print(f"  超额收益:     {p.get('excess_return', 0)*100:.2f}%")
        print(f"  年化收益:     {p.get('annualized_return', 0)*100:.2f}%")
        print(f"  年化波动率:   {p.get('annualized_volatility', 0)*100:.2f}%")
        print(f"  夏普比率:     {p.get('sharpe_ratio', 0):.4f}")

    if 'daily_return' in stats:
        d = stats['daily_return']
        print("\n【日收益统计】")
        print(f"  日均收益:     {d.get('mean', 0)*100:.4f}%")
        print(f"  日收益标准差: {d.get('std', 0)*100:.4f}%")
        print(f"  最大日收益:   {d.get('max', 0)*100:.2f}%")
        print(f"  最小日收益:   {d.get('min', 0)*100:.2f}%")
        print(f"  正收益天数:   {d.get('positive_days', 0)}/{d.get('total_days', 0)}")

    if 'drawdown' in stats:
        dd = stats['drawdown']
        print("\n【回撤分析】")
        print(f"  最大回撤:     {abs(dd.get('max_drawdown', 0))*100:.2f}%")
        print(f"  最大回撤日期: {dd.get('max_drawdown_date', 'N/A')}")

    if 'monthly' in stats:
        m = stats['monthly']
        print("\n【月度统计】")
        print(f"  月均收益:     {m.get('mean', 0)*100:.2f}%")
        print(f"  月收益标准差: {m.get('std', 0)*100:.2f}%")
        print(f"  正收益月份:   {m.get('positive_months', 0)}/{m.get('total_months', 0)}")
        print(f"  最佳月份:     {m.get('best_month', 0)*100:.2f}%")
        print(f"  最差月份:     {m.get('worst_month', 0)*100:.2f}%")

    if 'ic' in stats:
        ic = stats['ic']
        print("\n【IC分析】")
        print(f"  IC均值:      {ic.get('mean', 0):.4f}")
        print(f"  IC标准差:    {ic.get('std', 0):.4f}")
        print(f"  IC_IR:       {ic.get('ir', 0):.4f}")
        print(f"  IC>0比例:    {ic.get('positive_ratio', 0)*100:.1f}%")
        print(f"  |IC|>0.02:   {ic.get('ic_0_02', 0)*100:.1f}%")
        print(f"  |IC|>0.05:   {ic.get('ic_0_05', 0)*100:.1f}%")

    if 'rank_ic' in stats:
        ric = stats['rank_ic']
        print("\n【Rank IC分析】")
        print(f"  Rank IC均值: {ric.get('mean', 0):.4f}")
        print(f"  Rank IC标准差: {ric.get('std', 0):.4f}")
        print(f"  Rank IC_IR:  {ric.get('ir', 0):.4f}")

    if 'risk_no_cost' in stats:
        r = stats['risk_no_cost']
        print("\n【风险指标 (无成本)】")
        print(f"  年化收益:     {r.get('annualized_return', 0)*100:.2f}%")
        print(f"  信息比率:     {r.get('information_ratio', 0):.4f}")
        print(f"  最大回撤:     {abs(r.get('max_drawdown', 0))*100:.2f}%")

    if 'risk_with_cost' in stats:
        r = stats['risk_with_cost']
        print("\n【风险指标 (有成本)】")
        print(f"  年化收益:     {r.get('annualized_return', 0)*100:.2f}%")
        print(f"  信息比率:     {r.get('information_ratio', 0):.4f}")
        print(f"  最大回撤:     {abs(r.get('max_drawdown', 0))*100:.2f}%")

    print("\n" + "="*60)


def save_statistics_to_file(stats: Dict[str, Any], filepath: str) -> None:
    """保存统计信息到文件"""
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("策略表现汇总统计\n")
        f.write("="*60 + "\n\n")

        if 'performance' in stats:
            p = stats['performance']
            f.write("【整体表现】\n")
            f.write(f"  累计收益:     {p.get('cumulative_return', 0)*100:.2f}%\n")
            f.write(f"  基准收益:     {p.get('cumulative_benchmark', 0)*100:.2f}%\n")
            f.write(f"  超额收益:     {p.get('excess_return', 0)*100:.2f}%\n")
            f.write(f"  年化收益:     {p.get('annualized_return', 0)*100:.2f}%\n")
            f.write(f"  年化波动率:   {p.get('annualized_volatility', 0)*100:.2f}%\n")
            f.write(f"  夏普比率:     {p.get('sharpe_ratio', 0):.4f}\n\n")

        if 'daily_return' in stats:
            d = stats['daily_return']
            f.write("【日收益统计】\n")
            f.write(f"  日均收益:     {d.get('mean', 0)*100:.4f}%\n")
            f.write(f"  日收益标准差: {d.get('std', 0)*100:.4f}%\n")
            f.write(f"  最大日收益:   {d.get('max', 0)*100:.2f}%\n")
            f.write(f"  最小日收益:   {d.get('min', 0)*100:.2f}%\n")
            f.write(f"  正收益天数:   {d.get('positive_days', 0)}/{d.get('total_days', 0)}\n\n")

        if 'ic' in stats:
            ic = stats['ic']
            f.write("【IC分析】\n")
            f.write(f"  IC均值:      {ic.get('mean', 0):.4f}\n")
            f.write(f"  IC标准差:    {ic.get('std', 0):.4f}\n")
            f.write(f"  IC_IR:       {ic.get('ir', 0):.4f}\n")
            f.write(f"  IC>0比例:    {ic.get('positive_ratio', 0)*100:.1f}%\n")
            f.write(f"  |IC|>0.02:   {ic.get('ic_0_02', 0)*100:.1f}%\n")
            f.write(f"  |IC|>0.05:   {ic.get('ic_0_05', 0)*100:.1f}%\n\n")

        f.write("\n生成时间: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n")


def generate_all_plots(
    recorder,
    output_dir: Path,
    experiment_name: str = ""
) -> Dict[str, Any]:
    """
    生成所有图表

    Parameters
    ----------
    recorder : Recorder
        qlib workflow recorder
    output_dir : Path
        输出目录
    experiment_name : str
        实验名称

    Returns
    -------
    Dict[str, Any]
        统计信息
    """
    results = {
        'saved_files': {},
        'statistics': {},
    }

    title_prefix = f"[{experiment_name}] " if experiment_name else ""

    # 加载数据
    try:
        report_df = recorder.load_object("portfolio_analysis/report_normal_1day.pkl")
    except Exception:
        report_df = None

    try:
        ic_df = recorder.load_object("sig_analysis/ic.pkl")
    except Exception:
        ic_df = None

    try:
        ric_df = recorder.load_object("sig_analysis/ric.pkl")
    except Exception:
        ric_df = None

    try:
        port_analysis_df = recorder.load_object("portfolio_analysis/port_analysis_1day.pkl")
    except Exception:
        port_analysis_df = None

    # 生成图表
    if report_df is not None and not report_df.empty:
        results['saved_files'].update(
            plot_portfolio_performance(report_df, output_dir, title_prefix)
        )
        results['saved_files'].update(
            plot_monthly_returns_heatmap(report_df, output_dir, title_prefix)
        )

    if ic_df is not None and not ic_df.empty:
        results['saved_files'].update(
            plot_ic_analysis(ic_df, ric_df, output_dir, title_prefix)
        )

    if port_analysis_df is not None and not port_analysis_df.empty:
        results['saved_files'].update(
            plot_risk_metrics(port_analysis_df, output_dir, title_prefix)
        )

    # 生成统计信息
    results['statistics'] = generate_summary_statistics(report_df, ic_df, port_analysis_df, ric_df)

    return results
