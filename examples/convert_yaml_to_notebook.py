import os
import yaml
import json
from pathlib import Path

# 读取workflow_by_code.ipynb作为模板
def read_template():
    template_path = Path("/home/lianzw/Project/PythonProject/qlib/examples/workflow_by_code.ipynb")
    with open(template_path, 'r', encoding='utf-8') as f:
        return json.load(f)

# 读取yaml配置文件
def read_yaml(yaml_path):
    with open(yaml_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

# 生成新的notebook文件
def generate_notebook(yaml_path, output_dir):
    # 读取模板
    template = read_template()
    
    # 读取yaml配置
    config = read_yaml(yaml_path)
    
    # 获取文件名
    yaml_name = os.path.basename(yaml_path)
    notebook_name = yaml_name.replace('.yaml', '.ipynb')
    output_path = os.path.join(output_dir, notebook_name)
    
    # 创建新的notebook结构
    new_notebook = {
        "cells": [],
        "metadata": template["metadata"],
        "nbformat": template["nbformat"],
        "nbformat_minor": template["nbformat_minor"]
    }
    
    # 添加标题单元格
    new_notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            f"# {yaml_name} Notebook"
        ]
    })
    
    # 添加版权信息单元格
    new_notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "#  Copyright (c) Microsoft Corporation.\n",
            "#  Licensed under the MIT License."
        ]
    })
    
    # 添加安装和环境设置单元格
    new_notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import sys, site\n",
            "from pathlib import Path\n",
            "\n",
            "################################# NOTE #################################\n",
            "#  Please be aware that if colab installs the latest numpy and pyqlib  #\n",
            "#  in this cell, users should RESTART the runtime in order to run the  #\n",
            "#  following cells successfully.                                       #\n",
            "########################################################################\n",
            "\n",
            "try:\n",
            "    import qlib\n",
            "except ImportError:\n",
            "    # install qlib\n",
            "    ! pip install --upgrade numpy\n",
            "    ! pip install pyqlib\n",
            "    if \"google.colab\" in sys.modules:\n",
            "        # The Google colab environment is a little outdated. We have to downgrade the pyyaml to make it compatible with other packages\n",
            "        ! pip install pyyaml==5.4.1\n",
            "    # reload\n",
            "    site.main()\n",
            "\n",
            "scripts_dir = Path.cwd().parent.joinpath(\"scripts\")\n",
            "if not scripts_dir.joinpath(\"get_data.py\").exists():\n",
            "    # download get_data.py script\n",
            "    scripts_dir = Path(\"~/tmp/qlib_code/scripts\").expanduser().resolve()\n",
            "    scripts_dir.mkdir(parents=True, exist_ok=True)\n",
            "    import requests\n",
            "\n",
            "    with requests.get(\"https://raw.githubusercontent.com/microsoft/qlib/main/scripts/get_data.py\", timeout=10) as resp:\n",
            "        with open(scripts_dir.joinpath(\"get_data.py\"), \"wb\") as fp:\n",
            "            fp.write(resp.content)"
        ]
    })
    
    # 添加导入单元格
    new_notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import qlib\n",
            "import pandas as pd\n",
            "import yaml\n",
            "from qlib.constant import REG_CN\n",
            "from qlib.utils import exists_qlib_data, init_instance_by_config\n",
            "from qlib.workflow import R\n",
            "from qlib.workflow.record_temp import SignalRecord, PortAnaRecord\n",
            "from qlib.utils import flatten_dict"
        ]
    })
    
    # 添加数据初始化单元格
    new_notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {"tags": []},
        "outputs": [],
        "source": [
            "# use default data\n",
            "# NOTE: need to download data from remote: python scripts/get_data.py qlib_data_cn --target_dir ~/.qlib/qlib_data/cn_data\n",
            "provider_uri = \"~/.qlib/qlib_data/cn_data\"  # target_dir\n",
            "if not exists_qlib_data(provider_uri):\n",
            "    print(f\"Qlib data is not found in {provider_uri}\")\n",
            "    sys.path.append(str(scripts_dir))\n",
            "    from get_data import GetData\n",
            "\n",
            "    GetData().qlib_data(target_dir=provider_uri, region=REG_CN)\n",
            "qlib.init(provider_uri=provider_uri, region=REG_CN)"
        ]
    })
    
    # 添加market和benchmark定义单元格
    market_source = []
    if 'market' in config:
        market_value = config['market']
        if isinstance(market_value, str) and market_value.startswith('*'):
            # 处理锚点引用
            market_key = market_value[1:]
            if market_key in config:
                market_value = config[market_key]
        market_source.append(f"market = \"{market_value}\"\n")
    else:
        market_source.append("market = \"csi300\"\n")
    
    if 'benchmark' in config:
        benchmark_value = config['benchmark']
        if isinstance(benchmark_value, str) and benchmark_value.startswith('*'):
            # 处理锚点引用
            benchmark_key = benchmark_value[1:]
            if benchmark_key in config:
                benchmark_value = config[benchmark_key]
        market_source.append(f"benchmark = \"{benchmark_value}\"\n")
    else:
        market_source.append("benchmark = \"SH000300\"\n")
    
    new_notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": market_source
    })
    
    # 添加模型训练标题单元格
    new_notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": ["# train model"]
    })
    
    # 添加模型训练代码单元格
    train_source = []
    train_source.append("###################################\n")
    train_source.append("# train model\n")
    train_source.append("###################################\n")
    train_source.append(f"# Load configuration from {yaml_name}\n")
    train_source.append(f"config = yaml.safe_load(open('{yaml_path}', 'r'))\n")
    train_source.append("\n")
    
    # 提取data_handler_config
    if 'data_handler_config' in config:
        train_source.append("data_handler_config = config.get('data_handler_config')\n")
    else:
        train_source.append("data_handler_config = {\n")
        train_source.append("    \"start_time\": \"2008-01-01\",\n")
        train_source.append("    \"end_time\": \"2020-08-01\",\n")
        train_source.append("    \"fit_start_time\": \"2008-01-01\",\n")
        train_source.append("    \"fit_end_time\": \"2014-12-31\",\n")
        train_source.append("    \"instruments\": market,\n")
        train_source.append("}\n")
    train_source.append("\n")
    
    # 提取task配置
    train_source.append("task = config.get('task')\n")
    train_source.append("\n")
    
    train_source.append("# model initialization\n")
    train_source.append("model = init_instance_by_config(task['model'])\n")
    train_source.append("dataset = init_instance_by_config(task['dataset'])\n")
    train_source.append("\n")
    train_source.append("# start exp to train model\n")
    train_source.append("with R.start(experiment_name=\"train_model\"):\n")
    train_source.append("    R.log_params(**flatten_dict(task))\n")
    train_source.append("    model.fit(dataset)\n")
    train_source.append("    R.save_objects(trained_model=model)\n")
    train_source.append("    rid = R.get_recorder().id\n")
    
    new_notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {"tags": []},
        "outputs": [],
        "source": train_source
    })
    
    # 添加回测和分析标题单元格
    new_notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": ["# prediction, backtest & analysis"]
    })
    
    # 添加回测和分析代码单元格
    backtest_source = []
    backtest_source.append("###################################\n")
    backtest_source.append("# prediction, backtest & analysis\n")
    backtest_source.append("###################################\n")
    
    # 提取port_analysis_config
    backtest_source.append("port_analysis_config = config.get('port_analysis_config')\n")
    backtest_source.append("\n")
    
    backtest_source.append("# backtest and analysis\n")
    backtest_source.append("with R.start(experiment_name=\"backtest_analysis\"):\n")
    backtest_source.append("    recorder = R.get_recorder(recorder_id=rid, experiment_name=\"train_model\")\n")
    backtest_source.append("    model = recorder.load_object(\"trained_model\")\n")
    backtest_source.append("\n")
    backtest_source.append("    # prediction\n")
    backtest_source.append("    recorder = R.get_recorder()\n")
    backtest_source.append("    ba_rid = recorder.id\n")
    backtest_source.append("    sr = SignalRecord(model, dataset, recorder)\n")
    backtest_source.append("    sr.generate()\n")
    backtest_source.append("\n")
    backtest_source.append("    # backtest & analysis\n")
    backtest_source.append("    par = PortAnaRecord(recorder, port_analysis_config, \"day\")\n")
    backtest_source.append("    par.generate()\n")
    
    new_notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {"tags": []},
        "outputs": [],
        "source": backtest_source
    })
    
    # 添加分析图表标题单元格
    new_notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": ["# analyze graphs"]
    })
    
    # 添加分析图表代码单元格
    new_notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "from qlib.contrib.report import analysis_model, analysis_position\n",
            "from qlib.data import D\n",
            "\n",
            "recorder = R.get_recorder(recorder_id=ba_rid, experiment_name=\"backtest_analysis\")\n",
            "print(recorder)\n",
            "pred_df = recorder.load_object(\"pred.pkl\")\n",
            "report_normal_df = recorder.load_object(\"portfolio_analysis/report_normal_1day.pkl\")\n",
            "positions = recorder.load_object(\"portfolio_analysis/positions_normal_1day.pkl\")\n",
            "analysis_df = recorder.load_object(\"portfolio_analysis/port_analysis_1day.pkl\")"
        ]
    })
    
    # 添加分析位置标题单元格
    new_notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## analysis position"]
    })
    
    # 添加报告标题单元格
    new_notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": ["### report"]
    })
    
    # 添加报告代码单元格
    new_notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": ["analysis_position.report_graph(report_normal_df)"]
    })
    
    # 添加风险分析标题单元格
    new_notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": ["### risk analysis"]
    })
    
    # 添加风险分析代码单元格
    new_notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": ["analysis_position.risk_analysis_graph(analysis_df, report_normal_df)"]
    })
    
    # 添加模型分析标题单元格
    new_notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## analysis model"]
    })
    
    # 添加标签准备代码单元格
    new_notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "label_df = dataset.prepare(\"test\", col_set=\"label\")\n",
            "label_df.columns = [\"label\"]"
        ]
    })
    
    # 添加score IC标题单元格
    new_notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": ["### score IC"]
    })
    
    # 添加score IC代码单元格
    new_notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "pred_label = pd.concat([label_df, pred_df], axis=1, sort=True).reindex(label_df.index)\n",
            "analysis_position.score_ic_graph(pred_label)"
        ]
    })
    
    # 添加模型性能标题单元格
    new_notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": ["### model performance"]
    })
    
    # 添加模型性能代码单元格
    new_notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": ["analysis_model.model_performance_graph(pred_label)"]
    })
    
    # 保存新的notebook文件
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(new_notebook, f, indent=2, ensure_ascii=False)
    
    print(f"Generated notebook: {output_path}")

# 主函数
def main():
    # 定义输入和输出目录
    benchmarks_dir = "/home/lianzw/Project/PythonProject/qlib/examples/benchmarks"
    output_dir = "/examples/notebook_pynb"
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 遍历所有yaml文件
    for root, dirs, files in os.walk(benchmarks_dir):
        for file in files:
            if file.endswith('.yaml'):
                yaml_path = os.path.join(root, file)
                generate_notebook(yaml_path, output_dir)

if __name__ == "__main__":
    main()
