# Qlib 可视化工具使用教程 (最终版)

欢迎使用 Qlib 可视化工具！本工具旨在帮助不熟悉编程的用户也能利用 Qlib 强大的量化金融能力。

## 1. 安装与环境配置

在开始之前，请确保您的电脑已经安装了 Python (建议版本 3.9, 3.10, 3.11)。

**步骤一：下载项目文件**

请将我们创建的所有应用文件下载到您电脑的同一个文件夹中：
- `app.py` (应用主程序)
- `qlib_utils.py` (核心功能库)
- `requirements.txt` (依赖列表)
- `TUTORIAL.md` (本教程)
- `QLIB_INTRO.md` (概念入门)

**步骤二：下载 Qlib 官方脚本 (非常重要！)**

本工具的“增量更新”和“数据健康度检查”功能需要依赖 `qlib` 官方提供的数据处理脚本。
1.  请访问 Qlib 的 GitHub 仓库地址：[https://github.com/microsoft/qlib](https://github.com/microsoft/qlib)
2.  找到并点击绿色的 `<> Code` 按钮，然后选择 `Download ZIP` 将整个项目下载下来。
3.  解压后，找到里面的 `scripts` 文件夹。
4.  将这个**完整的 `scripts` 文件夹**复制到与 `app.py` 相同的目录下。

最终您的文件夹结构应该如下所示：
```
your_project_folder/
├── app.py
├── qlib_utils.py
├── requirements.txt
├── TUTORIAL.md
├── QLIB_INTRO.md
└── scripts/      <-- 完整的 scripts 文件夹
    ├── check_data_health.py
    └── data_collector/
        └── ...
```

**步骤三：安装依赖库**

打开您电脑的终端（在 Windows 上是 `Command Prompt` 或 `PowerShell`，在 macOS 或 Linux 上是 `Terminal`），进入您的项目文件夹，然后运行以下命令：

```bash
pip install -r requirements.txt
```

## 2. 运行本工具

所有依赖都安装好后，在同一个终端窗口和文件夹下，运行以下命令：

```bash
streamlit run app.py
```

运行后，您的浏览器会自动打开一个新的网页，地址通常是 `http://localhost:8501`。这个网页就是我们的可视化工具界面！

## 3. 功能页面详解

### 页面一：数据管理

这是您需要使用的第一个页面，所有后续操作都依赖于这里下载的数据。

- **1. 全量数据部署 (首次使用)**：由于直接从雅虎财经大量下载数据可能不稳定，我们**强烈推荐**您通过此区域提供的命令，手动下载由社区维护的预处理数据包。这是最快、最稳定的初始化方法。您只需**复制框中的命令，并依次在您的终端中执行**。
- **2. 增量更新 (日常使用)**：当您完成首次部署后，可以使用此功能来获取最新的日线数据。选择日期范围后，点击按钮即可。
- **3. 数据健康度检查**：下载或更新数据后，点击此按钮，程序会检查本地数据的完整性和质量，并将报告显示在日志区。

### (其他页面说明保持不变)
...
