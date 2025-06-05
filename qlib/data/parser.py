# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import division
from __future__ import print_function

import abc
import re
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Union, Optional, Any
from ..log import get_module_logger

#################### Cache Management ####################

class ExpressionCacheManager:
    """管理表达式计算的中间结果缓存"""
    
    def __init__(self, max_size: int = 1000):
        self.cache: Dict[Tuple, Any] = {}  # 缓存中间结果
        self.cache_stats: Dict[Tuple, int] = {}  # 缓存使用统计
        self.max_size = max_size
        
    def get_cache_key(self, expr: str, instrument: str, start_index: int, end_index: int, *args) -> Tuple:
        """生成缓存键"""
        return (str(expr), instrument, start_index, end_index, *args)
        
    def get(self, key: Tuple) -> Optional[pd.Series]:
        """获取缓存结果"""
        if key in self.cache:
            self.cache_stats[key] = self.cache_stats.get(key, 0) + 1
            return self.cache[key]
        return None
        
    def set(self, key: Tuple, value: pd.Series):
        """设置缓存结果"""
        if len(self.cache) >= self.max_size:
            self._remove_least_used()
        self.cache[key] = value
        self.cache_stats[key] = 0
        
    def _remove_least_used(self):
        """移除最少使用的缓存"""
        if not self.cache_stats:
            return
        min_key = min(self.cache_stats.items(), key=lambda x: x[1])[0]
        del self.cache[min_key]
        del self.cache_stats[min_key]
        
    def clear(self):
        """清除缓存"""
        self.cache.clear()
        self.cache_stats.clear()

#################### Expression Nodes ####################

class ExpressionNode:
    """Expression syntax tree node base class"""
    
    def __init__(self):
        self.children: List[ExpressionNode] = []
        self.parent: Optional[ExpressionNode] = None
        
    def add_child(self, child: 'ExpressionNode'):
        """添加子节点"""
        self.children.append(child)
        child.parent = self
        
    def evaluate(self) -> Any:
        """评估表达式节点"""
        raise NotImplementedError
        
    def __str__(self) -> str:
        return self.__class__.__name__

class FeatureNode(ExpressionNode):
    """Feature node (e.g. $close)"""
    
    def __init__(self, name: str):
        super().__init__()
        self.name = name
        
    def evaluate(self):
        from .base import Feature
        return Feature(self.name)
        
    def __str__(self) -> str:
        return f"${self.name}"

class OperatorNode(ExpressionNode):
    """Operator node (e.g. +, -, *, /)"""
    
    def __init__(self, op_type: str):
        super().__init__()
        self.op_type = op_type
        
    def evaluate(self):
        if len(self.children) != 2:
            raise ValueError(f"Operator {self.op_type} requires exactly 2 operands")
            
        left = self.children[0].evaluate()
        right = self.children[1].evaluate()
        
        from .ops import Add, Sub, Mul, Div, Gt, Lt, Eq, Ne, And, Or
        op_map = {
            '+': Add,
            '-': Sub,
            '*': Mul,
            '/': Div,
            '>': Gt,
            '<': Lt,
            '==': Eq,
            '!=': Ne,
            '&': And,
            '|': Or
        }
        
        return op_map[self.op_type](left, right)
        
    def __str__(self) -> str:
        return f"{self.children[0]} {self.op_type} {self.children[1]}"

class FunctionNode(ExpressionNode):
    """Function node (e.g. Mean, Std)"""
    
    def __init__(self, func_name: str, *args):
        super().__init__()
        self.func_name = func_name
        self.args = args
        
    def evaluate(self):
        if not self.children:
            raise ValueError(f"Function {self.func_name} requires at least one argument")
            
        args = [child.evaluate() for child in self.children]
        from .ops import Operators
        return getattr(Operators, self.func_name)(*args, *self.args)
        
    def __str__(self) -> str:
        args_str = ", ".join(str(child) for child in self.children)
        return f"{self.func_name}({args_str})"

#################### Expression Parser ####################

class ExpressionParser:
    """Expression parser that builds syntax tree"""
    
    def __init__(self):
        self.tokens: List[Tuple[str, str]] = []
        self.current: int = 0
        
    def parse(self, expression: str) -> ExpressionNode:
        """Parse expression string into syntax tree"""
        self.tokens = self.tokenize(expression)
        self.current = 0
        return self.parse_expression()
        
    def tokenize(self, expression: str) -> List[Tuple[str, str]]:
        """Tokenize expression string"""
        tokens = []
        i = 0
        while i < len(expression):
            if expression[i] == '$':
                # 处理特征名
                j = i + 1
                while j < len(expression) and (expression[j].isalnum() or expression[j] == '_'):
                    j += 1
                tokens.append(('FEATURE', expression[i+1:j]))
                i = j
            elif expression[i] in '+-*/()<>=!&|':
                # 处理运算符
                if expression[i:i+2] in ('==', '!=', '&&', '||'):
                    tokens.append(('OPERATOR', expression[i:i+2]))
                    i += 2
                else:
                    tokens.append(('OPERATOR', expression[i]))
                    i += 1
            elif expression[i].isalpha():
                # 处理函数名
                j = i
                while j < len(expression) and (expression[j].isalnum() or expression[j] == '_'):
                    j += 1
                tokens.append(('FUNCTION', expression[i:j]))
                i = j
            elif expression[i].isdigit():
                # 处理数字
                j = i
                while j < len(expression) and (expression[j].isdigit() or expression[j] == '.'):
                    j += 1
                tokens.append(('NUMBER', float(expression[i:j])))
                i = j
            else:
                i += 1
        return tokens
        
    def parse_expression(self) -> ExpressionNode:
        """Parse expression into syntax tree"""
        return self.parse_term()
        
    def parse_term(self) -> ExpressionNode:
        """Parse term (expression with + and -)"""
        expr = self.parse_factor()
        
        while self.current < len(self.tokens):
            token = self.tokens[self.current]
            if token[0] == 'OPERATOR' and token[1] in '+-':
                self.current += 1
                right = self.parse_factor()
                node = OperatorNode(token[1])
                node.add_child(expr)
                node.add_child(right)
                expr = node
            else:
                break
                
        return expr
        
    def parse_factor(self) -> ExpressionNode:
        """Parse factor (expression with * and /)"""
        expr = self.parse_primary()
        
        while self.current < len(self.tokens):
            token = self.tokens[self.current]
            if token[0] == 'OPERATOR' and token[1] in '*/':
                self.current += 1
                right = self.parse_primary()
                node = OperatorNode(token[1])
                node.add_child(expr)
                node.add_child(right)
                expr = node
            else:
                break
                
        return expr
        
    def parse_primary(self) -> ExpressionNode:
        """Parse primary expression (features, numbers, functions)"""
        token = self.tokens[self.current]
        self.current += 1
        
        if token[0] == 'FEATURE':
            return FeatureNode(token[1])
        elif token[0] == 'NUMBER':
            return ConstantNode(token[1])
        elif token[0] == 'FUNCTION':
            node = FunctionNode(token[1])
            if self.tokens[self.current][1] == '(':
                self.current += 1  # skip '('
                while True:
                    node.add_child(self.parse_expression())
                    if self.tokens[self.current][1] == ')':
                        self.current += 1
                        break
                    if self.tokens[self.current][1] != ',':
                        raise ValueError("Expected ',' or ')'")
                    self.current += 1
            return node
        elif token[1] == '(':
            expr = self.parse_expression()
            if self.tokens[self.current][1] != ')':
                raise ValueError("Expected ')'")
            self.current += 1
            return expr
        else:
            raise ValueError(f"Unexpected token: {token}")

#################### Expression Provider ####################

class ExpressionProvider(abc.ABC):
    """Expression provider class with syntax tree support"""
    
    def __init__(self):
        self.expression_instance_cache = {}
        self.parser = ExpressionParser()
        self.cache_manager = ExpressionCacheManager()
        
    def get_expression_instance(self, field: str) -> Any:
        """Get expression instance from field string"""
        try:
            if field in self.expression_instance_cache:
                return self.expression_instance_cache[field]
                
            # 使用语法树解析表达式
            syntax_tree = self.parser.parse(field)
            expression = syntax_tree.evaluate()
            self.expression_instance_cache[field] = expression
            return expression
            
        except Exception as e:
            get_module_logger("data").exception(f"Error parsing expression {field}: {str(e)}")
            raise
            
    def load(self, instrument: str, start_index: int, end_index: int, *args) -> pd.Series:
        """Load data with cache support"""
        cache_key = self.cache_manager.get_cache_key(
            self, instrument, start_index, end_index, *args
        )
        
        # 检查缓存
        cached_result = self.cache_manager.get(cache_key)
        if cached_result is not None:
            return cached_result
            
        # 计算新结果
        try:
            result = self._load_internal(instrument, start_index, end_index, *args)
            # 缓存结果
            self.cache_manager.set(cache_key, result)
            return result
        except Exception as e:
            get_module_logger("data").exception(
                f"Error loading expression {self}: {str(e)}"
            )
            raise
            
    @abc.abstractmethod
    def _load_internal(self, instrument: str, start_index: int, end_index: int, *args) -> pd.Series:
        """Internal load implementation"""
        raise NotImplementedError 