# Qlib 可视化工具使用教程 (最终版)

欢迎使用 Qlib 可视化工具！本工具旨在帮助不熟悉编程的用户也能利用 Qlib 强大的量化金融能力。

## 1. 安装与运行

在开始之前，请确保您的电脑已经安装了 Python (建议版本 3.9, 3.10, 3.11)。

**步骤一：下载所有项目文件**

请将我们创建的所有应用文件下载到您电脑的同一个文件夹中。该文件夹应包含：
- `app.py`
- `qlib_utils.py`
- `requirements.txt`
- `run.bat`  **(Windows 用户快速启动脚本)**
- `TUTORIAL.md` (本教程)
- `DEEP_DIVE_TUTORIAL.md` (深度量化教程)

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
├── run.bat
├── TUTORIAL.md
├── DEEP_DIVE_TUTORIAL.md
└── scripts/      <-- 完整的 scripts 文件夹
    └── ...
```

**步骤三：启动应用**

我们提供了两种启动方式：

- **(推荐) Windows 用户一键启动**:
  - 直接**双击 `run.bat` 文件**。它会自动打开一个终端窗口，安装所有必需的依赖库，然后启动应用程序。

- **手动启动 (macOS, Linux, 或高级用户)**:
  - 打开您的终端，进入项目文件夹，然后运行以下两个命令：
  ```bash
  # 第一步：安装依赖 (仅首次运行时需要)
  pip install -r requirements.txt

  # 第二步：启动应用
  streamlit run app.py
  ```

运行后，您的浏览器会自动打开一个新的网页，地址通常是 `http://localhost:8501`。

---
## 2. 功能页面详解

(此部分内容无变化，保持之前版本的详细说明)

---
## 3. 想要了解更多？

如果您对“因子”、“模型”、“回测”这些概念背后的原理感兴趣，或者想更深入地理解量化投资，请阅读我们为您准备的：
- **[《深度量化教程》(`DEEP_DIVE_TUTORIAL.md`)](./DEEP_DIVE_TUTORIAL.md)**

希望这份最终版的教程能帮助您顺利地开启量化投资之旅！
