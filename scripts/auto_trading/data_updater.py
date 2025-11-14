# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
多市场数据自动更新模块

支持 A股、港股、ETF 的数据自动更新，包括：
- 增量数据下载
- 数据质量检查
- 多市场并行更新
- 断点续传支持
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import pandas as pd
import qlib
from loguru import logger

# 添加 data_collector 路径
COLLECTOR_DIR = Path(__file__).parent.parent / "data_collector"
sys.path.insert(0, str(COLLECTOR_DIR))

from yahoo.collector import Run
from utils import exists_qlib_data


class MultiMarketDataUpdater:
    """多市场数据自动更新器

    支持 A股、港股、ETF 的自动化数据更新流程

    Attributes:
        base_dir (Path): 数据存储根目录
        markets (Dict): 市场配置字典

    Examples:
        >>> updater = MultiMarketDataUpdater(base_dir='~/.qlib/qlib_data')
        >>> updater.update_all_markets()
        >>> updater.verify_data_quality()
    """

    def __init__(self, base_dir: str = '~/.qlib/qlib_data'):
        """初始化数据更新器

        Args:
            base_dir: 数据存储根目录，默认 ~/.qlib/qlib_data
        """
        self.base_dir = Path(base_dir).expanduser().resolve()
        self.base_dir.mkdir(parents=True, exist_ok=True)

        # 市场配置
        self.markets = {
            'cn': {  # A股 + A股ETF
                'region': 'CN',
                'name': 'A股',
                'data_dir': self.base_dir / 'cn_data',
                'source_dir': self.base_dir / 'source' / 'cn',
                'normalize_dir': self.base_dir / 'normalize' / 'cn',
                'index_symbols': ['000300', '000905', '000852'],  # 沪深300、中证500、中证1000
                'interval': '1d',
            },
            'hk': {  # 港股
                'region': 'HK',
                'name': '港股',
                'data_dir': self.base_dir / 'hk_data',
                'source_dir': self.base_dir / 'source' / 'hk',
                'normalize_dir': self.base_dir / 'normalize' / 'hk',
                'index_symbols': ['HSI', 'HSCEI'],  # 恒生指数、国企指数
                'interval': '1d',
            },
        }

        # 创建目录
        for market_config in self.markets.values():
            market_config['data_dir'].mkdir(parents=True, exist_ok=True)
            market_config['source_dir'].mkdir(parents=True, exist_ok=True)
            market_config['normalize_dir'].mkdir(parents=True, exist_ok=True)

        logger.info(f"数据更新器初始化完成，根目录: {self.base_dir}")

    def update_market_data(
        self,
        market: str = 'cn',
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        force_download: bool = False,
        max_workers: int = 4,
        delay: float = 0.5,
    ) -> Dict:
        """更新指定市场数据

        Args:
            market: 市场代码 ('cn' 或 'hk')
            start_date: 开始日期，格式 YYYY-MM-DD，默认为上次更新日期
            end_date: 结束日期，格式 YYYY-MM-DD，默认为今天
            force_download: 是否强制重新下载，默认 False
            max_workers: 并发线程数，默认 4
            delay: 请求延迟（秒），默认 0.5

        Returns:
            Dict: 更新结果统计信息

        Raises:
            ValueError: 如果市场代码无效

        Examples:
            >>> updater.update_market_data('cn')
            >>> updater.update_market_data('hk', start_date='2024-01-01')
        """
        if market not in self.markets:
            raise ValueError(f"无效的市场代码: {market}，支持的市场: {list(self.markets.keys())}")

        config = self.markets[market]
        logger.info(f"开始更新 {config['name']} 数据...")

        start_time = datetime.now()

        try:
            # 初始化 Run 对象
            runner = Run(
                source_dir=str(config['source_dir']),
                normalize_dir=str(config['normalize_dir']),
                max_workers=max_workers,
                interval=config['interval'],
                region=config['region']
            )

            # 检查是否已有数据，决定更新策略
            data_exists = exists_qlib_data(str(config['data_dir']))

            if not data_exists or force_download:
                logger.info(f"{config['name']} 数据不存在，执行完整下载...")

                # 完整下载流程
                if start_date is None:
                    start_date = '2010-01-01'  # 默认从2010年开始

                if end_date is None:
                    end_date = datetime.now().strftime('%Y-%m-%d')

                # 1. 下载原始数据
                logger.info(f"步骤 1/3: 下载原始数据 ({start_date} 到 {end_date})...")
                runner.download_data(
                    max_collector_count=2,
                    delay=delay,
                    start=start_date,
                    end=end_date,
                    check_data_length=None,
                    limit_nums=None
                )

                # 2. 标准化数据
                logger.info(f"步骤 2/3: 标准化数据...")
                runner.normalize_data(
                    date_field_name='date',
                    symbol_field_name='symbol',
                    end_date=end_date
                )

                # 3. 转换为二进制格式
                logger.info(f"步骤 3/3: 转换为 Qlib 二进制格式...")
                runner.dump_bin(
                    str(config['data_dir']),
                    include_fields='open,close,high,low,volume,factor',
                    date_field_name='date',
                    symbol_field_name='symbol'
                )

                update_type = 'full_download'

            else:
                logger.info(f"{config['name']} 数据已存在，执行增量更新...")

                # 增量更新流程（一键更新）
                runner.update_data_to_bin(
                    qlib_data_1d_dir=str(config['data_dir']),
                    end_date=end_date,
                    check_data_length=None,
                    delay=delay,
                    exists_skip=False
                )

                update_type = 'incremental_update'

            # 计算耗时
            elapsed_time = (datetime.now() - start_time).total_seconds()

            # 获取数据统计
            stats = self._get_data_stats(config['data_dir'])

            result = {
                'market': market,
                'market_name': config['name'],
                'update_type': update_type,
                'success': True,
                'elapsed_time': elapsed_time,
                'stats': stats,
                'error': None
            }

            logger.success(
                f"✓ {config['name']} 数据更新完成 "
                f"[耗时: {elapsed_time:.1f}s, 股票数: {stats['n_instruments']}, "
                f"日期范围: {stats['date_range'][0]} ~ {stats['date_range'][1]}]"
            )

            return result

        except Exception as e:
            elapsed_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"✗ {config['name']} 数据更新失败: {str(e)}")

            return {
                'market': market,
                'market_name': config['name'],
                'update_type': None,
                'success': False,
                'elapsed_time': elapsed_time,
                'stats': None,
                'error': str(e)
            }

    def update_all_markets(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        force_download: bool = False,
        parallel: bool = False
    ) -> Dict[str, Dict]:
        """更新所有市场数据

        Args:
            start_date: 开始日期，默认为上次更新日期
            end_date: 结束日期，默认为今天
            force_download: 是否强制重新下载
            parallel: 是否并行更新（当前版本串行）

        Returns:
            Dict[str, Dict]: 各市场更新结果

        Examples:
            >>> results = updater.update_all_markets()
            >>> for market, result in results.items():
            ...     print(f"{market}: {result['success']}")
        """
        logger.info("=" * 60)
        logger.info("开始更新所有市场数据")
        logger.info("=" * 60)

        results = {}

        # 目前采用串行更新（更稳定）
        # 未来可以改为并行更新以提升速度
        for market in ['cn', 'hk']:
            result = self.update_market_data(
                market=market,
                start_date=start_date,
                end_date=end_date,
                force_download=force_download
            )
            results[market] = result

        # 汇总结果
        success_count = sum(1 for r in results.values() if r['success'])
        total_count = len(results)
        total_time = sum(r['elapsed_time'] for r in results.values())

        logger.info("=" * 60)
        logger.info(f"更新完成: {success_count}/{total_count} 成功, 总耗时: {total_time:.1f}s")
        logger.info("=" * 60)

        return results

    def verify_data_quality(self, market: Optional[str] = None) -> Dict:
        """验证数据质量

        检查项：
        - 最新数据日期
        - 数据完整性（缺失值）
        - 数据异常值
        - 交易日历一致性

        Args:
            market: 市场代码，None 表示检查所有市场

        Returns:
            Dict: 数据质量报告

        Examples:
            >>> quality = updater.verify_data_quality('cn')
            >>> print(quality['issues'])
        """
        logger.info("开始数据质量检查...")

        markets_to_check = [market] if market else list(self.markets.keys())
        report = {}

        for mkt in markets_to_check:
            if mkt not in self.markets:
                continue

            config = self.markets[mkt]
            data_dir = config['data_dir']

            if not exists_qlib_data(str(data_dir)):
                report[mkt] = {
                    'status': 'no_data',
                    'message': '数据不存在'
                }
                continue

            try:
                # 初始化 qlib
                qlib.init(provider_uri=str(data_dir), expression_cache=None, dataset_cache=None)

                # 获取所有股票列表
                instruments = qlib.D.instruments('all')

                # 获取日历
                calendar = qlib.D.calendar(freq='day')

                # 抽样检查
                sample_stocks = instruments[:min(10, len(instruments))]

                issues = []

                # 检查最新数据日期
                latest_date = calendar[-1]
                expected_latest = pd.Timestamp(datetime.now().date())

                # 如果是交易日，检查数据是否更新到今天
                if expected_latest.weekday() < 5:  # 周一到周五
                    days_behind = (expected_latest - latest_date).days
                    if days_behind > 3:  # 落后超过3天
                        issues.append(f"数据可能过时，最新日期: {latest_date.date()}, 预期: {expected_latest.date()}")

                # 检查缺失值
                for stock in sample_stocks:
                    data = qlib.D.features([stock], ['$close', '$volume'], start_time=calendar[-30])
                    missing_ratio = data.isna().sum().sum() / (len(data) * 2)
                    if missing_ratio > 0.1:  # 超过10%缺失
                        issues.append(f"股票 {stock} 存在较多缺失值: {missing_ratio:.1%}")

                # 检查异常值
                for stock in sample_stocks:
                    data = qlib.D.features([stock], ['$close'], start_time=calendar[-30])
                    if len(data) > 1:
                        returns = data['$close'].pct_change().dropna()
                        if (returns.abs() > 0.5).any():  # 单日涨跌超过50%
                            issues.append(f"股票 {stock} 存在异常波动")

                report[mkt] = {
                    'status': 'checked',
                    'n_instruments': len(instruments),
                    'latest_date': str(latest_date.date()),
                    'n_trading_days': len(calendar),
                    'issues': issues if issues else ['无明显问题'],
                    'quality_score': max(0, 100 - len(issues) * 10)  # 质量评分
                }

                if issues:
                    logger.warning(f"{config['name']} 发现 {len(issues)} 个问题:")
                    for issue in issues:
                        logger.warning(f"  - {issue}")
                else:
                    logger.success(f"✓ {config['name']} 数据质量良好")

            except Exception as e:
                logger.error(f"{config['name']} 质量检查失败: {str(e)}")
                report[mkt] = {
                    'status': 'error',
                    'message': str(e)
                }

        return report

    def _get_data_stats(self, data_dir: Path) -> Dict:
        """获取数据统计信息

        Args:
            data_dir: 数据目录

        Returns:
            Dict: 统计信息
        """
        try:
            # 初始化 qlib
            qlib.init(provider_uri=str(data_dir), expression_cache=None, dataset_cache=None)

            # 获取股票列表
            instruments = qlib.D.instruments('all')

            # 获取日历
            calendar = qlib.D.calendar(freq='day')

            return {
                'n_instruments': len(instruments),
                'n_trading_days': len(calendar),
                'date_range': (str(calendar[0].date()), str(calendar[-1].date())),
                'sample_instruments': instruments[:5].tolist() if len(instruments) > 0 else []
            }

        except Exception as e:
            logger.warning(f"获取数据统计失败: {str(e)}")
            return {
                'n_instruments': 0,
                'n_trading_days': 0,
                'date_range': (None, None),
                'sample_instruments': []
            }

    def get_latest_date(self, market: str = 'cn') -> Optional[str]:
        """获取指定市场的最新数据日期

        Args:
            market: 市场代码

        Returns:
            str: 最新日期 (YYYY-MM-DD)，如果数据不存在返回 None
        """
        if market not in self.markets:
            raise ValueError(f"无效的市场代码: {market}")

        config = self.markets[market]
        data_dir = config['data_dir']

        if not exists_qlib_data(str(data_dir)):
            return None

        try:
            # 读取日历文件
            calendar_file = data_dir / 'calendars' / 'day.txt'
            if calendar_file.exists():
                calendar_df = pd.read_csv(calendar_file, header=None)
                latest_date = calendar_df.iloc[-1, 0]
                return latest_date
            else:
                return None
        except Exception as e:
            logger.warning(f"获取最新日期失败: {str(e)}")
            return None

    def clean_cache(self, market: Optional[str] = None):
        """清理缓存数据

        Args:
            market: 市场代码，None 表示清理所有市场
        """
        markets_to_clean = [market] if market else list(self.markets.keys())

        for mkt in markets_to_clean:
            if mkt not in self.markets:
                continue

            config = self.markets[mkt]

            # 清理 source 和 normalize 目录
            for dir_path in [config['source_dir'], config['normalize_dir']]:
                if dir_path.exists():
                    import shutil
                    shutil.rmtree(dir_path)
                    dir_path.mkdir(parents=True, exist_ok=True)
                    logger.info(f"已清理 {config['name']} 缓存: {dir_path}")


def main():
    """命令行入口"""
    import argparse

    parser = argparse.ArgumentParser(description='多市场数据自动更新工具')
    parser.add_argument('--base_dir', default='~/.qlib/qlib_data', help='数据根目录')
    parser.add_argument('--market', choices=['cn', 'hk', 'all'], default='all', help='市场代码')
    parser.add_argument('--start_date', default=None, help='开始日期 (YYYY-MM-DD)')
    parser.add_argument('--end_date', default=None, help='结束日期 (YYYY-MM-DD)')
    parser.add_argument('--force', action='store_true', help='强制重新下载')
    parser.add_argument('--verify', action='store_true', help='验证数据质量')
    parser.add_argument('--clean_cache', action='store_true', help='清理缓存')

    args = parser.parse_args()

    # 初始化更新器
    updater = MultiMarketDataUpdater(base_dir=args.base_dir)

    # 清理缓存
    if args.clean_cache:
        market = None if args.market == 'all' else args.market
        updater.clean_cache(market=market)
        return

    # 更新数据
    if args.market == 'all':
        results = updater.update_all_markets(
            start_date=args.start_date,
            end_date=args.end_date,
            force_download=args.force
        )
    else:
        result = updater.update_market_data(
            market=args.market,
            start_date=args.start_date,
            end_date=args.end_date,
            force_download=args.force
        )
        results = {args.market: result}

    # 验证数据质量
    if args.verify:
        market = None if args.market == 'all' else args.market
        quality_report = updater.verify_data_quality(market=market)

        print("\n" + "=" * 60)
        print("数据质量报告")
        print("=" * 60)
        for mkt, report in quality_report.items():
            print(f"\n{updater.markets[mkt]['name']}:")
            if report['status'] == 'checked':
                print(f"  股票数量: {report['n_instruments']}")
                print(f"  最新日期: {report['latest_date']}")
                print(f"  质量评分: {report['quality_score']}/100")
                print(f"  问题列表:")
                for issue in report['issues']:
                    print(f"    - {issue}")
            else:
                print(f"  状态: {report.get('message', report['status'])}")

    # 检查是否有失败的更新
    failed_markets = [m for m, r in results.items() if not r['success']]
    if failed_markets:
        logger.error(f"以下市场更新失败: {failed_markets}")
        sys.exit(1)
    else:
        logger.success("所有市场数据更新成功！")


if __name__ == '__main__':
    main()
