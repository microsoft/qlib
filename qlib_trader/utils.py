"""
工具函数模块
"""

import sys
import time
from pathlib import Path
from typing import List, Optional


# ============================================================
# 终端颜色 / 样式
# ============================================================

class Style:
    """终端样式工具"""
    BOLD = "\033[1m"
    DIM = "\033[2m"
    UNDERLINE = "\033[4m"
    RESET = "\033[0m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"

    @staticmethod
    def colorize(text: str, color: str) -> str:
        return f"{color}{text}{Style.RESET}"

    @staticmethod
    def bold(text: str) -> str:
        return f"{Style.BOLD}{text}{Style.RESET}"

    @staticmethod
    def success(text: str) -> str:
        return Style.colorize(text, Style.GREEN)

    @staticmethod
    def error(text: str) -> str:
        return Style.colorize(text, Style.RED)

    @staticmethod
    def warning(text: str) -> str:
        return Style.colorize(text, Style.YELLOW)

    @staticmethod
    def info(text: str) -> str:
        return Style.colorize(text, Style.CYAN)

    @staticmethod
    def header(text: str) -> str:
        return Style.colorize(Style.bold(text), Style.BLUE)


# ============================================================
# 打印工具
# ============================================================

def print_banner():
    """打印启动横幅"""
    banner = r"""
    ╔══════════════════════════════════════════════════════════╗
    ║                                                          ║
    ║           QLib Trader - 量化交易集成平台                 ║
    ║           Quantitative Trading Platform                  ║
    ║                                                          ║
    ║           基于微软 Qlib 框架                             ║
    ║           Powered by Microsoft Qlib                      ║
    ║                                                          ║
    ╚══════════════════════════════════════════════════════════╝
    """
    print(Style.colorize(banner, Style.CYAN))


def print_section(title: str):
    """打印分节标题"""
    width = 60
    print()
    print(Style.header("=" * width))
    print(Style.header(f"  {title}"))
    print(Style.header("=" * width))


def print_menu(title: str, options: List[str], show_back: bool = True):
    """打印菜单选项"""
    print_section(title)
    for i, option in enumerate(options, 1):
        print(f"  {Style.bold(str(i))}. {option}")
    if show_back:
        print(f"  {Style.bold('0')}. 返回上一级")
    print()


def print_table(headers: List[str], rows: List[List[str]], col_widths: Optional[List[int]] = None):
    """打印简单表格"""
    if col_widths is None:
        col_widths = []
        for i, h in enumerate(headers):
            max_w = len(h)
            for row in rows:
                if i < len(row):
                    max_w = max(max_w, len(str(row[i])))
            col_widths.append(min(max_w + 2, 40))

    # Header
    header_line = ""
    for h, w in zip(headers, col_widths):
        header_line += f"  {h:<{w}}"
    print(Style.bold(header_line))
    print("  " + "-" * (sum(col_widths) + len(col_widths) * 2))

    # Rows
    for row in rows:
        line = ""
        for val, w in zip(row, col_widths):
            line += f"  {str(val):<{w}}"
        print(line)


def get_input(prompt: str, default: Optional[str] = None) -> str:
    """获取用户输入"""
    if default:
        prompt_text = f"  {prompt} [{default}]: "
    else:
        prompt_text = f"  {prompt}: "
    try:
        value = input(prompt_text).strip()
        return value if value else (default or "")
    except (EOFError, KeyboardInterrupt):
        print()
        return default or ""


def get_choice(prompt: str, max_val: int, allow_zero: bool = True) -> int:
    """获取用户菜单选择"""
    while True:
        try:
            raw = input(f"  {prompt}: ").strip()
            if not raw:
                continue
            val = int(raw)
            if allow_zero and val == 0:
                return 0
            if 1 <= val <= max_val:
                return val
            print(Style.warning(f"  请输入 {'0-' if allow_zero else '1-'}{max_val} 之间的数字"))
        except ValueError:
            print(Style.warning("  请输入有效数字"))
        except (EOFError, KeyboardInterrupt):
            print()
            return 0


def confirm(prompt: str, default: bool = True) -> bool:
    """确认操作"""
    suffix = "[Y/n]" if default else "[y/N]"
    try:
        raw = input(f"  {prompt} {suffix}: ").strip().lower()
        if not raw:
            return default
        return raw in ("y", "yes", "是")
    except (EOFError, KeyboardInterrupt):
        print()
        return False


def print_progress(current: int, total: int, prefix: str = "", width: int = 40):
    """打印进度条"""
    percent = current / total if total > 0 else 0
    filled = int(width * percent)
    bar = "█" * filled + "░" * (width - filled)
    sys.stdout.write(f"\r  {prefix} |{bar}| {percent:.0%}")
    if current >= total:
        sys.stdout.write("\n")
    sys.stdout.flush()


def format_number(n: float) -> str:
    """格式化数字"""
    if abs(n) >= 1e8:
        return f"{n / 1e8:.2f}亿"
    elif abs(n) >= 1e4:
        return f"{n / 1e4:.2f}万"
    else:
        return f"{n:.2f}"


def check_data_exists(provider_uri: str) -> bool:
    """检查数据目录是否存在"""
    data_path = Path(provider_uri).expanduser()
    if not data_path.exists():
        return False
    # Check for essential subdirectories
    for subdir in ["calendars", "instruments", "features"]:
        if not (data_path / subdir).exists():
            return False
    return True
