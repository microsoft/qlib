import os
import yaml
from typing import List, Dict, Any

# Benchmark目录路径 - 使用绝对路径
BENCHMARK_DIR = "/home/qlib_t/examples/benchmarks"

class BenchmarkService:
    @staticmethod
    def get_benchmarks() -> List[Dict[str, Any]]:
        """获取所有可用的benchmark样例"""
        benchmarks = []
        
        try:
            # 检查benchmark目录是否存在
            if not os.path.exists(BENCHMARK_DIR):
                print(f"Benchmark directory not found: {BENCHMARK_DIR}")
                return benchmarks
                
            # 遍历benchmark目录
            for model_name in os.listdir(BENCHMARK_DIR):
                model_path = os.path.join(BENCHMARK_DIR, model_name)
                if os.path.isdir(model_path):
                    # 查找yaml配置文件
                    yaml_files = [f for f in os.listdir(model_path) if f.endswith('.yaml')]
                    for yaml_file in yaml_files:
                        file_path = os.path.join(model_path, yaml_file)
                        with open(file_path, 'r', encoding='utf-8') as f:
                            try:
                                # 读取yaml内容
                                content = f.read()
                                # 解析yaml获取基本信息
                                config = yaml.safe_load(content)
                                
                                # 构建benchmark信息
                                benchmark = {
                                    'id': f"{model_name}_{yaml_file}",
                                    'name': f"{model_name} - {yaml_file.replace('workflow_config_', '').replace('.yaml', '')}",
                                    'model': model_name,
                                    'file_name': yaml_file,
                                    'path': file_path,
                                    'content': content
                                }
                                benchmarks.append(benchmark)
                            except Exception as e:
                                print(f"Error reading benchmark {yaml_file}: {e}")
        except Exception as e:
            print(f"Error accessing benchmark directory {BENCHMARK_DIR}: {e}")
        
        return benchmarks
    
    @staticmethod
    def get_benchmark(benchmark_id: str) -> Dict[str, Any] or None:
        """获取特定的benchmark样例"""
        benchmarks = BenchmarkService.get_benchmarks()
        for benchmark in benchmarks:
            if benchmark['id'] == benchmark_id:
                return benchmark
        return None
    
    @staticmethod
    def get_benchmark_by_path(path: str) -> Dict[str, Any] or None:
        """通过路径获取benchmark样例"""
        if os.path.exists(path) and path.endswith('.yaml'):
            with open(path, 'r', encoding='utf-8') as f:
                try:
                    content = f.read()
                    model_name = os.path.basename(os.path.dirname(path))
                    file_name = os.path.basename(path)
                    
                    benchmark = {
                        'id': f"{model_name}_{file_name}",
                        'name': f"{model_name} - {file_name.replace('workflow_config_', '').replace('.yaml', '')}",
                        'model': model_name,
                        'file_name': file_name,
                        'path': path,
                        'content': content
                    }
                    return benchmark
                except Exception as e:
                    print(f"Error reading benchmark {path}: {e}")
        return None
