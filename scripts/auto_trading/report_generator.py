# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
自动化报告生成系统

生成可视化的交易报告和分析图表，包括：
- HTML格式日报
- Excel格式订单表
- 绩效分析图表
- 风险分析报告
"""

from pathlib import Path
from typing import Dict, Optional
from datetime import datetime
import pandas as pd
import numpy as np
from loguru import logger


class ReportGenerator:
    """自动化报告生成器

    生成 HTML/Excel 格式的交易报告

    Attributes:
        output_dir (Path): 报告输出目录

    Examples:
        >>> reporter = ReportGenerator(output_dir='./reports')
        >>> files = reporter.generate_daily_report(
        ...     date='2024-11-14',
        ...     signals=signals,
        ...     rebalance_plan=plan,
        ...     backtest_results=results,
        ...     risk_analysis=risk
        ... )
    """

    def __init__(self, output_dir: str = './reports'):
        """初始化报告生成器

        Args:
            output_dir: 报告输出目录
        """
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"报告生成器初始化 [输出目录: {self.output_dir}]")

    def generate_daily_report(
        self,
        date: str,
        signals: pd.Series,
        rebalance_plan: Dict,
        backtest_results: Dict,
        risk_analysis: Dict,
        signal_quality: Optional[Dict] = None
    ) -> Dict[str, Path]:
        """生成每日报告

        Args:
            date: 报告日期
            signals: 交易信号
            rebalance_plan: 再平衡计划
            backtest_results: 回测结果
            risk_analysis: 风险分析
            signal_quality: 信号质量指标

        Returns:
            Dict[str, Path]: 生成的文件路径

        Examples:
            >>> files = reporter.generate_daily_report(...)
            >>> print(files['html_report'])
        """
        logger.info(f"生成每日报告 (日期: {date})")

        report_date = pd.to_datetime(date).strftime('%Y%m%d')

        # 1. 生成 HTML 报告
        html_path = self.output_dir / f'report_{report_date}.html'
        self._create_html_report(
            output_path=html_path,
            date=date,
            signals=signals,
            rebalance_plan=rebalance_plan,
            backtest_results=backtest_results,
            risk_analysis=risk_analysis,
            signal_quality=signal_quality
        )

        # 2. 生成 Excel 订单表
        excel_path = self.output_dir / f'orders_{report_date}.xlsx'
        self._create_excel_report(
            output_path=excel_path,
            rebalance_plan=rebalance_plan,
            risk_analysis=risk_analysis,
            backtest_results=backtest_results
        )

        logger.success(f"✓ 报告生成完成 [HTML: {html_path.name}, Excel: {excel_path.name}]")

        return {
            'html_report': html_path,
            'excel_report': excel_path,
        }

    def _create_html_report(
        self,
        output_path: Path,
        date: str,
        signals: pd.Series,
        rebalance_plan: Dict,
        backtest_results: Dict,
        risk_analysis: Dict,
        signal_quality: Optional[Dict]
    ):
        """创建 HTML 报告

        Args:
            output_path: 输出文件路径
            date: 日期
            signals: 信号
            rebalance_plan: 再平衡计划
            backtest_results: 回测结果
            risk_analysis: 风险分析
            signal_quality: 信号质量
        """
        html = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>量化交易日报 - {date}</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Microsoft YaHei', sans-serif;
            background-color: #f5f5f5;
            padding: 20px;
            line-height: 1.6;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
            margin-bottom: 20px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 30px;
            margin-bottom: 15px;
            border-left: 4px solid #3498db;
            padding-left: 10px;
        }}
        .summary {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 30px;
        }}
        .summary h3 {{
            margin-bottom: 15px;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }}
        .metric-card {{
            background: rgba(255,255,255,0.1);
            padding: 15px;
            border-radius: 6px;
        }}
        .metric-label {{
            font-size: 0.9em;
            opacity: 0.9;
            margin-bottom: 5px;
        }}
        .metric-value {{
            font-size: 1.5em;
            font-weight: bold;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            background: white;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #3498db;
            color: white;
            font-weight: 600;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
        .buy {{
            color: #e74c3c;
            font-weight: bold;
        }}
        .sell {{
            color: #27ae60;
            font-weight: bold;
        }}
        .positive {{
            color: #27ae60;
        }}
        .negative {{
            color: #e74c3c;
        }}
        .risk-section {{
            background: #ecf0f1;
            padding: 20px;
            border-radius: 6px;
            margin: 20px 0;
        }}
        .footer {{
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
            text-align: center;
            color: #7f8c8d;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 量化交易日报</h1>
        <p style="color: #7f8c8d; margin-bottom: 30px;">报告日期: {date} | 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

        <div class="summary">
            <h3>📈 核心指标概览</h3>
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-label">回测年化收益</div>
                    <div class="metric-value">{backtest_results.get('annual_return', 0):.2%}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">夏普比率</div>
                    <div class="metric-value">{backtest_results.get('sharpe_ratio', 0):.2f}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">最大回撤</div>
                    <div class="metric-value">{backtest_results.get('max_drawdown', 0):.2%}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">信息比率</div>
                    <div class="metric-value">{backtest_results.get('information_ratio', 0):.2f}</div>
                </div>
            </div>
        </div>

        <h2>🎯 交易建议</h2>
        <div style="background: #fff3cd; padding: 15px; border-radius: 6px; margin: 10px 0;">
            <strong>总换手率:</strong> {rebalance_plan['turnover']:.2%} |
            <strong>买入数量:</strong> {rebalance_plan['n_buy']} |
            <strong>卖出数量:</strong> {rebalance_plan['n_sell']} |
            <strong>持有数量:</strong> {rebalance_plan['n_hold']}
        </div>

        <table>
            <thead>
                <tr>
                    <th>股票代码</th>
                    <th>操作</th>
                    <th>当前权重</th>
                    <th>目标权重</th>
                    <th>变动</th>
                    <th>交易金额</th>
                </tr>
            </thead>
            <tbody>
"""

        # 添加订单表
        orders = rebalance_plan['orders']
        if len(orders) > 0:
            # 只显示前20个订单
            for _, order in orders.head(20).iterrows():
                direction_class = 'buy' if order['direction'] == 'BUY' else 'sell'
                html += f"""
                <tr>
                    <td>{order['stock']}</td>
                    <td class="{direction_class}">{order['direction']}</td>
                    <td>{order['current_weight']:.2%}</td>
                    <td>{order['target_weight']:.2%}</td>
                    <td class="{'positive' if order['delta_weight'] > 0 else 'negative'}">{order['delta_weight']:+.2%}</td>
                    <td>¥{order['amount_value']:,.0f}</td>
                </tr>
"""
            if len(orders) > 20:
                html += f"""
                <tr style="background: #f8f9fa;">
                    <td colspan="6" style="text-align: center; font-style: italic;">
                        ... 还有 {len(orders) - 20} 个订单 (请查看 Excel 文件)
                    </td>
                </tr>
"""
        else:
            html += """
                <tr>
                    <td colspan="6" style="text-align: center; color: #7f8c8d;">无需调仓</td>
                </tr>
"""

        html += """
            </tbody>
        </table>

        <h2>⚠️ 风险分析</h2>
        <div class="risk-section">
            <div class="metrics-grid">
                <div>
                    <strong>跟踪误差:</strong> {tracking_error:.2%}
                </div>
                <div>
                    <strong>最大持仓:</strong> {max_position:.2%}
                </div>
                <div>
                    <strong>持仓集中度:</strong> {concentration:.4f}
                </div>
                <div>
                    <strong>有效股票数:</strong> {effective_n_stocks:.1f}
                </div>
            </div>
            <div style="margin-top: 15px;">
                <strong>风险分解:</strong><br>
                因子风险: {factor_risk_pct:.1%} | 特异性风险: {specific_risk_pct:.1%}
            </div>
        </div>
""".format(
            tracking_error=risk_analysis.get('tracking_error', 0),
            max_position=risk_analysis.get('max_position', 0),
            concentration=risk_analysis.get('concentration', 0),
            effective_n_stocks=risk_analysis.get('effective_n_stocks', 0),
            factor_risk_pct=risk_analysis.get('risk_decomp', {}).get('factor', 0),
            specific_risk_pct=risk_analysis.get('risk_decomp', {}).get('specific', 0),
        )

        # 信号质量部分
        if signal_quality:
            html += f"""
        <h2>📡 信号质量</h2>
        <div style="background: #e8f5e9; padding: 15px; border-radius: 6px;">
            <div class="metrics-grid">
                <div>
                    <strong>IC均值:</strong> {signal_quality.get('ic_mean', 0):.4f}
                </div>
                <div>
                    <strong>IC标准差:</strong> {signal_quality.get('ic_std', 0):.4f}
                </div>
                <div>
                    <strong>IC IR:</strong> {signal_quality.get('ic_ir', 0):.4f}
                </div>
                <div>
                    <strong>信号覆盖:</strong> {signal_quality.get('signal_coverage', 0):.2%}
                </div>
            </div>
        </div>
"""

        html += """
        <div class="footer">
            <p>本报告由自动化交易系统生成 | Powered by Qlib</p>
            <p style="margin-top: 5px;">
                <em>免责声明: 本报告仅供参考，不构成投资建议。投资有风险，决策需谨慎。</em>
            </p>
        </div>
    </div>
</body>
</html>
"""

        # 写入文件
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html)

    def _create_excel_report(
        self,
        output_path: Path,
        rebalance_plan: Dict,
        risk_analysis: Dict,
        backtest_results: Dict
    ):
        """创建 Excel 报告

        Args:
            output_path: 输出路径
            rebalance_plan: 再平衡计划
            risk_analysis: 风险分析
            backtest_results: 回测结果
        """
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # Sheet 1: 交易订单
            if len(rebalance_plan['orders']) > 0:
                rebalance_plan['orders'].to_excel(writer, sheet_name='交易订单', index=False)

            # Sheet 2: 目标权重
            target_weights_df = rebalance_plan['target_weights'].to_frame('target_weight')
            target_weights_df = target_weights_df[target_weights_df['target_weight'] > 0]
            target_weights_df = target_weights_df.sort_values('target_weight', ascending=False)
            target_weights_df.to_excel(writer, sheet_name='目标权重')

            # Sheet 3: 风险分析
            risk_df = pd.DataFrame([{
                '跟踪误差': risk_analysis.get('tracking_error', 0),
                '最大持仓': risk_analysis.get('max_position', 0),
                '持仓集中度': risk_analysis.get('concentration', 0),
                '有效股票数': risk_analysis.get('effective_n_stocks', 0),
                '持仓数量': risk_analysis.get('n_positions', 0),
                '因子风险占比': risk_analysis.get('risk_decomp', {}).get('factor', 0),
                '特异性风险占比': risk_analysis.get('risk_decomp', {}).get('specific', 0),
            }])
            risk_df.to_excel(writer, sheet_name='风险分析', index=False)

            # Sheet 4: 回测绩效
            performance_df = pd.DataFrame([{
                '总收益率': backtest_results.get('total_return', 0),
                '年化收益': backtest_results.get('annual_return', 0),
                '基准收益': backtest_results.get('bench_total_return', 0),
                '超额收益': backtest_results.get('excess_return', 0),
                '夏普比率': backtest_results.get('sharpe_ratio', 0),
                '信息比率': backtest_results.get('information_ratio', 0),
                '最大回撤': backtest_results.get('max_drawdown', 0),
                '年化波动': backtest_results.get('volatility', 0),
                '胜率': backtest_results.get('win_rate', 0),
                '平均换手': backtest_results.get('turnover', 0),
            }])
            performance_df.to_excel(writer, sheet_name='回测绩效', index=False)


def main():
    """命令行入口（用于测试）"""
    # 生成示例报告
    reporter = ReportGenerator(output_dir='./test_reports')

    # 模拟数据
    signals = pd.Series({'SH600000': 0.8, 'SH600036': 0.6}, name='signal')
    rebalance_plan = {
        'target_weights': pd.Series({'SH600000': 0.05, 'SH600036': 0.03}),
        'orders': pd.DataFrame([
            {'stock': 'SH600000', 'direction': 'BUY', 'current_weight': 0.0,
             'target_weight': 0.05, 'delta_weight': 0.05, 'amount_value': 5000000}
        ]),
        'turnover': 0.08,
        'n_buy': 1,
        'n_sell': 0,
        'n_hold': 29,
    }
    backtest_results = {
        'annual_return': 0.15,
        'sharpe_ratio': 1.5,
        'max_drawdown': -0.08,
        'information_ratio': 1.2,
        'total_return': 0.12,
        'bench_total_return': 0.08,
        'excess_return': 0.04,
        'volatility': 0.18,
        'win_rate': 0.55,
        'turnover': 0.08,
    }
    risk_analysis = {
        'tracking_error': 0.05,
        'max_position': 0.05,
        'concentration': 0.05,
        'effective_n_stocks': 25,
        'n_positions': 30,
        'risk_decomp': {'factor': 0.6, 'specific': 0.4},
    }

    files = reporter.generate_daily_report(
        date='2024-11-14',
        signals=signals,
        rebalance_plan=rebalance_plan,
        backtest_results=backtest_results,
        risk_analysis=risk_analysis,
    )

    logger.info(f"测试报告已生成: {files}")


if __name__ == '__main__':
    main()
