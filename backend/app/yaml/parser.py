import yaml
from typing import Dict, Any, List

class QLibYAMLParser:
    @staticmethod
    def parse_yaml(yaml_str: str) -> Dict[str, Any]:
        """解析QLib YAML配置"""
        return yaml.safe_load(yaml_str)
    
    @staticmethod
    def generate_yaml(config: Dict[str, Any]) -> str:
        """生成QLib YAML配置"""
        return yaml.dump(config, default_flow_style=False)
    
    @staticmethod
    def validate_yaml(config: Dict[str, Any]) -> bool:
        """验证QLib YAML配置的合法性"""
        # 检查必需的顶级键
        required_keys = ['task']
        for key in required_keys:
            if key not in config:
                return False
        
        # 检查task配置
        task_config = config['task']
        if not isinstance(task_config, dict):
            return False
        
        # 检查task中必需的键
        task_required_keys = ['model', 'dataset']
        for key in task_required_keys:
            if key not in task_config:
                return False
        
        # 检查model配置
        model_config = task_config['model']
        if not isinstance(model_config, dict):
            return False
        
        # 检查model中必需的键
        model_required_keys = ['class', 'module_path']
        for key in model_required_keys:
            if key not in model_config:
                return False
        
        # 检查dataset配置
        dataset_config = task_config['dataset']
        if not isinstance(dataset_config, dict):
            return False
        
        # 检查dataset中必需的键
        dataset_required_keys = ['class', 'module_path']
        for key in dataset_required_keys:
            if key not in dataset_config:
                return False
        
        return True
    
    @staticmethod
    def get_supported_models() -> List[str]:
        """获取支持的模型列表"""
        return [
            'LinearModel',
            'LGBModel',
            'XGBModel',
            'CatBoostModel',
            'MLP',
            'LSTM',
            'GRU',
            'Transformer',
            'TabNet'
        ]
    
    @staticmethod
    def get_supported_datasets() -> List[str]:
        """获取支持的数据集列表"""
        return [
            'DatasetH',
            'TSDatasetH'
        ]
