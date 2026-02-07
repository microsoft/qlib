"""
数据管理模块
============
负责市场数据的下载、检查和管理。
"""

import os
from pathlib import Path
from typing import Optional

from qlib_trader.config import REGIONS
from qlib_trader.utils import (
    Style,
    check_data_exists,
    confirm,
    get_choice,
    get_input,
    print_menu,
    print_section,
    print_table,
)


class DataManager:
    """数据管理器"""

    def run_menu(self):
        """数据管理交互菜单"""
        while True:
            print_menu("数据管理", [
                "查看数据状态",
                "下载市场数据",
                "下载精简数据 (快速测试)",
                "检查数据完整性",
            ])
            choice = get_choice("请选择", 4)
            if choice == 0:
                return
            elif choice == 1:
                self.show_data_status()
            elif choice == 2:
                self.download_data()
            elif choice == 3:
                self.download_simple_data()
            elif choice == 4:
                self.check_data_health()

    def show_data_status(self):
        """显示各市场数据状态"""
        print_section("数据状态")
        rows = []
        for key, conf in REGIONS.items():
            uri = conf["provider_uri"]
            path = Path(uri).expanduser()
            if check_data_exists(uri):
                status = Style.success("已就绪")
                # Count features
                features_dir = path / "features"
                if features_dir.exists():
                    stock_count = len(list(features_dir.iterdir()))
                else:
                    stock_count = 0
                size = self._get_dir_size(path)
            else:
                status = Style.warning("未下载")
                stock_count = 0
                size = "-"
            rows.append([
                conf["name"],
                key,
                status,
                str(stock_count),
                size,
                str(path),
            ])
        print_table(
            ["市场", "代码", "状态", "股票数", "大小", "路径"],
            rows,
            [10, 6, 10, 8, 10, 30],
        )

    def download_data(self, region: Optional[str] = None):
        """下载完整市场数据"""
        if region is None:
            print_section("下载市场数据")
            options = [f"{v['name']} ({k})" for k, v in REGIONS.items()]
            print_menu("选择市场", options, show_back=True)
            choice = get_choice("请选择", len(options))
            if choice == 0:
                return
            region = list(REGIONS.keys())[choice - 1]

        region_conf = REGIONS[region]
        target_dir = region_conf["provider_uri"]

        if check_data_exists(target_dir):
            print(Style.warning(f"\n  数据已存在: {target_dir}"))
            if not confirm("是否重新下载？", default=False):
                return

        print(Style.info(f"\n  正在下载 {region_conf['name']} 数据..."))
        print(Style.info(f"  目标目录: {target_dir}"))
        print(Style.info("  数据来源: Yahoo Finance (通过 Qlib 提供)"))
        print()

        try:
            from qlib.tests.data import GetData
            GetData().qlib_data(
                target_dir=target_dir,
                region=region,
                exists_skip=False,
                delete_old=False,
            )
            print(Style.success("\n  数据下载完成!"))
        except Exception as e:
            print(Style.error(f"\n  下载失败: {e}"))
            print(Style.info("  提示: 请检查网络连接，或手动下载数据"))
            print(Style.info(f"  手动下载命令: python scripts/get_data.py qlib_data --target_dir {target_dir} --region {region}"))

    def download_simple_data(self, region: Optional[str] = None):
        """下载精简数据（用于快速测试）"""
        if region is None:
            print_section("下载精简数据")
            options = [f"{v['name']} ({k})" for k, v in REGIONS.items()]
            print_menu("选择市场", options, show_back=True)
            choice = get_choice("请选择", len(options))
            if choice == 0:
                return
            region = list(REGIONS.keys())[choice - 1]

        region_conf = REGIONS[region]
        target_dir = region_conf["provider_uri"]

        print(Style.info(f"\n  正在下载 {region_conf['name']} 精简数据..."))
        print(Style.info("  精简数据体积小，适合快速验证流程"))
        print()

        try:
            from qlib.tests.data import GetData
            GetData().qlib_data(
                name="qlib_data_simple",
                target_dir=target_dir,
                region=region,
                exists_skip=False,
                delete_old=False,
            )
            print(Style.success("\n  精简数据下载完成!"))
        except Exception as e:
            print(Style.error(f"\n  下载失败: {e}"))
            print(Style.info("  提示: 请检查网络连接"))

    def check_data_health(self, region: Optional[str] = None):
        """检查数据完整性"""
        if region is None:
            print_section("检查数据完整性")
            # Find available data
            available = []
            for key, conf in REGIONS.items():
                if check_data_exists(conf["provider_uri"]):
                    available.append((key, conf))

            if not available:
                print(Style.warning("  没有找到已下载的数据，请先下载数据"))
                return

            options = [f"{v['name']} ({k})" for k, v in available]
            print_menu("选择市场", options, show_back=True)
            choice = get_choice("请选择", len(options))
            if choice == 0:
                return
            region = available[choice - 1][0]

        region_conf = REGIONS[region]
        target_dir = region_conf["provider_uri"]
        data_path = Path(target_dir).expanduser()

        print(Style.info(f"\n  正在检查 {region_conf['name']} 数据完整性..."))

        issues = []
        checks_passed = 0

        # Check directory structure
        for subdir in ["calendars", "instruments", "features"]:
            subpath = data_path / subdir
            if subpath.exists():
                checks_passed += 1
                print(Style.success(f"    [OK] {subdir}/ 目录存在"))
            else:
                issues.append(f"{subdir}/ 目录缺失")
                print(Style.error(f"    [FAIL] {subdir}/ 目录缺失"))

        # Check calendar files
        cal_path = data_path / "calendars"
        if cal_path.exists():
            cal_files = list(cal_path.iterdir())
            if cal_files:
                checks_passed += 1
                print(Style.success(f"    [OK] 日历文件: {len(cal_files)} 个"))
            else:
                issues.append("日历文件为空")
                print(Style.error("    [FAIL] 日历文件为空"))

        # Check instruments
        inst_path = data_path / "instruments"
        if inst_path.exists():
            inst_files = list(inst_path.iterdir())
            if inst_files:
                checks_passed += 1
                print(Style.success(f"    [OK] 指数/市场文件: {len(inst_files)} 个"))
            else:
                issues.append("指数文件为空")

        # Check features
        feat_path = data_path / "features"
        if feat_path.exists():
            stock_dirs = [d for d in feat_path.iterdir() if d.is_dir()]
            if stock_dirs:
                checks_passed += 1
                print(Style.success(f"    [OK] 股票特征: {len(stock_dirs)} 只股票"))
                # Check a sample stock
                sample = stock_dirs[0]
                feat_files = list(sample.iterdir())
                print(Style.info(f"    [INFO] 样本 {sample.name}: {len(feat_files)} 个特征文件"))
            else:
                issues.append("特征数据为空")

        print()
        if issues:
            print(Style.warning(f"  检查完成: {checks_passed} 项通过, {len(issues)} 项异常"))
            for issue in issues:
                print(Style.error(f"    - {issue}"))
        else:
            print(Style.success(f"  检查完成: {checks_passed} 项全部通过，数据完整!"))

    @staticmethod
    def _get_dir_size(path: Path) -> str:
        """获取目录大小"""
        try:
            total = 0
            for f in path.rglob("*"):
                if f.is_file():
                    total += f.stat().st_size
            if total > 1e9:
                return f"{total / 1e9:.1f} GB"
            elif total > 1e6:
                return f"{total / 1e6:.1f} MB"
            else:
                return f"{total / 1e3:.1f} KB"
        except Exception:
            return "N/A"

    def ensure_data(self, region: str) -> bool:
        """确保数据存在，不存在则提示下载"""
        region_conf = REGIONS[region]
        if check_data_exists(region_conf["provider_uri"]):
            return True

        print(Style.warning(f"\n  未找到 {region_conf['name']} 市场数据"))
        if confirm("是否立即下载？"):
            self.download_data(region)
            return check_data_exists(region_conf["provider_uri"])
        return False
