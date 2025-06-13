"""
一组对 DataContainer 进行操作的纯函数库。
每个函数都接收 DataContainer 和其他参数，并返回一个新的 DataContainer。
"""
import numpy as np
from abc import ABC, abstractmethod
import pandas as pd
from factor_engine.data_layer.containers import PanelContainer
from typing import Dict, Any, List, Tuple

class Operator(ABC):
    """
    Base class for all operators in the factor engine.
    Provides common interface and functionality for different types of operators.
    """
    
    def __init__(self, name: str, description: str = ""):
        """
        Initialize operator with a name and description.
        
        Args:
            name: The name of the operator (e.g., 'add', 'ts_mean', 'cs_rank')
            description: A description of what the operator does
        """
        self._name = name
        self._description = description
    
    @property
    def name(self) -> str:
        """Get the operator name."""
        return self._name
    
    @property
    def description(self) -> str:
        """Get the operator description."""
        return self._description
    
    @abstractmethod
    def get_required_inputs(self) -> List[Tuple[str, type]]:
        """
        Get the required input parameters and their types.
        
        Returns:
            List of tuples containing (parameter_name, parameter_type)
        """
        pass
    
    def __call__(self, *args, **kwargs):
        """
        Make operator callable. Delegates to the execute method.
        """
        return self.execute(*args, **kwargs)
    
    @abstractmethod
    def execute(self, *args, **kwargs) -> PanelContainer:
        """
        Execute the operator with given arguments.
        Must be implemented by subclasses.
        
        Returns:
            PanelContainer: Result of the operation
        """
        pass
    
    def validate_inputs(self, *args):
        """
        Validate input arguments before execution.
        Can be overridden by subclasses for specific validation logic.
        
        Args:
            *args: Input arguments to validate
            
        Raises:
            TypeError: If input types are invalid
            ValueError: If input values are invalid
        """
        pass
    
    def validate_output(self, result: PanelContainer) -> bool:
        """
        Validate the output of the operation.
        
        Args:
            result: The output PanelContainer to validate
            
        Returns:
            bool: True if output is valid
            
        Raises:
            ValueError: If output is invalid
        """
        if not isinstance(result, PanelContainer):
            raise ValueError(f"Output must be PanelContainer, got {type(result)}")
        return True
    
    def get_required_window(self) -> int:
        """
        Get the required lookback window for this operator.
        Used by the planner to determine data requirements.
        
        Returns:
            int: Number of periods required for lookback (0 for no requirement)
        """
        return 0
    
    def get_metadata(self) -> Dict[str, Any]:
        """
        Get metadata about the operator.
        
        Returns:
            Dict containing operator metadata
        """
        return {
            "name": self.name,
            "description": self.description,
            "required_window": self.get_required_window(),
            "required_inputs": self.get_required_inputs()
        }
    
    def __repr__(self):
        return f"{self.__class__.__name__}(name='{self.name}')"


class BinaryOperator(Operator):
    """
    Base class for binary operators that take two PanelContainer inputs.
    """
    
    def get_required_inputs(self) -> List[Tuple[str, type]]:
        return [("a", PanelContainer), ("b", PanelContainer)]
    
    def validate_inputs(self, a, b):
        """Validate that both inputs are PanelContainer instances."""
        if not isinstance(a, PanelContainer) or not isinstance(b, PanelContainer):
            raise TypeError(f"操作数都必须是 PanelContainer，但得到了 {type(a)} 和 {type(b)}")


class UnaryOperator(Operator):
    """
    Base class for unary operators that take one PanelContainer input.
    """
    
    def get_required_inputs(self) -> List[Tuple[str, type]]:
        return [("data", PanelContainer)]
    
    def validate_inputs(self, data):
        """Validate that input is a PanelContainer instance."""
        if not isinstance(data, PanelContainer):
            raise TypeError(f"操作数必须是 PanelContainer，但得到了 {type(data)}")


class TimeSeriesOperator(Operator):
    """
    Base class for time series operators that operate along the time dimension.
    These operators typically require a lookback window.
    """
    
    def __init__(self, name: str, window: int = 1, description: str = ""):
        """
        Initialize time series operator.
        
        Args:
            name: The name of the operator
            window: The lookback window size
            description: A description of what the operator does
        """
        super().__init__(name, description)
        self.window = window
    
    def get_required_inputs(self) -> List[Tuple[str, type]]:
        return [("data", PanelContainer), ("window", int)]
    
    def get_required_window(self) -> int:
        """Return the required lookback window."""
        return max(0, self.window - 1)
    
    def validate_inputs(self, data, *args):
        """Validate that input is a PanelContainer and window is valid."""
        if not isinstance(data, PanelContainer):
            raise TypeError(f"操作数必须是 PanelContainer，但得到了 {type(data)}")
        if self.window <= 0:
            raise ValueError(f"窗口大小必须为正数，但得到了 {self.window}")


class CrossSectionOperator(Operator):
    """
    Base class for cross-sectional operators that operate across stocks at each time point.
    """
    
    def get_required_inputs(self) -> List[Tuple[str, type]]:
        return [("data", PanelContainer)]
    
    def validate_inputs(self, data):
        """Validate that input is a PanelContainer instance."""
        if not isinstance(data, PanelContainer):
            raise TypeError(f"操作数必须是 PanelContainer，但得到了 {type(data)}")

# --- Binary Operators ---

class AddOperator(BinaryOperator):
    """对两个 PanelContainer 的数据执行元素级加法。"""
    
    def __init__(self):
        super().__init__("add", "Element-wise addition of two PanelContainers")
    
    def execute(self, a: PanelContainer, b: PanelContainer) -> PanelContainer:
        self.validate_inputs(a, b)
        result_data = a.get_data().add(b.get_data())
        result = PanelContainer(result_data)
        self.validate_output(result)
        return result

class SubtractOperator(BinaryOperator):
    """对两个 PanelContainer 的数据执行元素级减法。"""
    
    def __init__(self):
        super().__init__("subtract", "Element-wise subtraction of two PanelContainers")
    
    def execute(self, a: PanelContainer, b: PanelContainer) -> PanelContainer:
        self.validate_inputs(a, b)
        result_data = a.get_data().sub(b.get_data())
        result = PanelContainer(result_data)
        self.validate_output(result)
        return result

class MultiplyOperator(BinaryOperator):
    """对两个 PanelContainer 的数据执行元素级乘法。"""
    
    def __init__(self):
        super().__init__("multiply", "Element-wise multiplication of two PanelContainers")
    
    def execute(self, a: PanelContainer, b: PanelContainer) -> PanelContainer:
        self.validate_inputs(a, b)
        result_data = a.get_data().mul(b.get_data())
        result = PanelContainer(result_data)
        self.validate_output(result)
        return result

class DivideOperator(BinaryOperator):
    """对两个 PanelContainer 的数据执行元素级除法。"""
    
    def __init__(self):
        super().__init__("divide", "Element-wise division of two PanelContainers")
    
    def execute(self, a: PanelContainer, b: PanelContainer) -> PanelContainer:
        self.validate_inputs(a, b)
        a_data = a.get_data()
        b_data = b.get_data()
        # a/b, 用 NaN 替换无穷大值，然后用 0 填充所有 NaN
        result_data = a_data.div(b_data).replace([np.inf, -np.inf], np.nan).fillna(0)
        result = PanelContainer(result_data)
        self.validate_output(result)
        return result

class MaxOperator(BinaryOperator):
    """对两个 PanelContainer 的数据执行元素级最大值。"""
    
    def __init__(self):
        super().__init__("max", "Element-wise maximum of two PanelContainers")
    
    def execute(self, a: PanelContainer, b: PanelContainer) -> PanelContainer:
        self.validate_inputs(a, b)
        result_data = np.maximum(a.get_data(), b.get_data())
        result = PanelContainer(result_data)
        self.validate_output(result)
        return result

class MinOperator(BinaryOperator):
    """对两个 PanelContainer 的数据执行元素级最小值。"""
    
    def __init__(self):
        super().__init__("min", "Element-wise minimum of two PanelContainers")
    
    def execute(self, a: PanelContainer, b: PanelContainer) -> PanelContainer:
        self.validate_inputs(a, b)
        result_data = np.minimum(a.get_data(), b.get_data())
        result = PanelContainer(result_data)
        self.validate_output(result)
        return result

# --- Unary Operators ---

class NegateOperator(UnaryOperator):
    """对 PanelContainer 的数据执行元素级取负。"""
    
    def __init__(self):
        super().__init__("negate", "Element-wise negation of PanelContainer")
    
    def execute(self, data: PanelContainer) -> PanelContainer:
        self.validate_inputs(data)
        result = PanelContainer(-data.get_data())
        self.validate_output(result)
        return result

class AbsOperator(UnaryOperator):
    """对 PanelContainer 的数据执行元素级绝对值。"""
    
    def __init__(self):
        super().__init__("abs", "Element-wise absolute value of PanelContainer")
    
    def execute(self, data: PanelContainer) -> PanelContainer:
        self.validate_inputs(data)
        result = PanelContainer(data.get_data().abs())
        self.validate_output(result)
        return result

class LogOperator(UnaryOperator):
    """对 PanelContainer 的数据执行元素级自然对数。"""
    
    def __init__(self):
        super().__init__("log", "Element-wise natural logarithm of PanelContainer")
    
    def execute(self, data: PanelContainer) -> PanelContainer:
        self.validate_inputs(data)
        # log of non-positive numbers is undefined, results in NaN
        result_data = np.log(data.get_data())
        result = PanelContainer(result_data.replace([np.inf, -np.inf], np.nan))
        self.validate_output(result)
        return result

# --- Time-Series Operators ---

class TimeSeriesMeanOperator(TimeSeriesOperator):
    """计算面板数据在时间序列上的滚动平均值。"""
    
    def __init__(self, window: int = 1):
        super().__init__("ts_mean", window, "Rolling mean over time series")
    
    def execute(self, data: PanelContainer, window: int = None) -> PanelContainer:
        self.validate_inputs(data)
        if window is not None:
            self.window = window
        result_data = data.get_data().rolling(window=self.window, min_periods=1).mean()
        result = PanelContainer(result_data)
        self.validate_output(result)
        return result

class TimeSeriesStdOperator(TimeSeriesOperator):
    """计算面板数据在时间序列上的滚动标准差。"""
    
    def __init__(self, window: int = 1):
        super().__init__("ts_std", window, "Rolling standard deviation over time series")
    
    def execute(self, data: PanelContainer, window: int = None) -> PanelContainer:
        self.validate_inputs(data)
        if window is not None:
            self.window = window
        result_data = data.get_data().rolling(window=self.window, min_periods=1).std()
        result = PanelContainer(result_data)
        self.validate_output(result)
        return result

class ShiftOperator(TimeSeriesOperator):
    """将面板数据在时间序列上向前平移（滞后）。"""
    
    def __init__(self, window: int = 1):
        super().__init__("shift", window, "Shift data forward in time series")
    
    def execute(self, data: PanelContainer, window: int = None) -> PanelContainer:
        self.validate_inputs(data)
        if window is not None:
            self.window = window
        result_data = data.get_data().shift(periods=self.window)
        result = PanelContainer(result_data)
        self.validate_output(result)
        return result

class TimeSeriesMaxOperator(TimeSeriesOperator):
    """计算面板数据在时间序列上的滚动最大值。"""
    
    def __init__(self, window: int = 1):
        super().__init__("ts_max", window, "Rolling maximum over time series")
    
    def execute(self, data: PanelContainer, window: int = None) -> PanelContainer:
        self.validate_inputs(data)
        if window is not None:
            self.window = window
        result_data = data.get_data().rolling(window=self.window, min_periods=1).max()
        result = PanelContainer(result_data)
        self.validate_output(result)
        return result

class TimeSeriesMinOperator(TimeSeriesOperator):
    """计算面板数据在时间序列上的滚动最小值。"""
    
    def __init__(self, window: int = 1):
        super().__init__("ts_min", window, "Rolling minimum over time series")
    
    def execute(self, data: PanelContainer, window: int = None) -> PanelContainer:
        self.validate_inputs(data)
        if window is not None:
            self.window = window
        result_data = data.get_data().rolling(window=self.window, min_periods=1).min()
        result = PanelContainer(result_data)
        self.validate_output(result)
        return result

class TimeSeriesRankOperator(TimeSeriesOperator):
    """计算面板数据在时间序列上的滚动排名。"""
    
    def __init__(self, window: int = 1):
        super().__init__("ts_rank", window, "Rolling rank over time series")
    
    def execute(self, data: PanelContainer, window: int = None) -> PanelContainer:
        self.validate_inputs(data)
        if window is not None:
            self.window = window
        # apply rank on a rolling window. The last element of the window is ranked against the elements in the window.
        result_data = data.get_data().rolling(window=self.window).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
        )
        result = PanelContainer(result_data)
        self.validate_output(result)
        return result

# --- Cross-Sectional Operators ---

class CrossSectionRankOperator(CrossSectionOperator):
    """在每个时间点上（行）对数据进行横截面排名。"""
    
    def __init__(self):
        super().__init__("cs_rank", "Cross-sectional rank at each time point")
    
    def execute(self, data: PanelContainer) -> PanelContainer:
        self.validate_inputs(data)
        result_data = data.get_data().rank(axis=1, pct=True)
        result = PanelContainer(result_data)
        self.validate_output(result)
        return result

class CrossSectionNormalizeOperator(CrossSectionOperator):
    """在每个时间点上（行）对数据进行横截面 Z-score 标准化。"""
    
    def __init__(self):
        super().__init__("cs_normalize", "Cross-sectional Z-score normalization at each time point")
    
    def execute(self, data: PanelContainer) -> PanelContainer:
        self.validate_inputs(data)
        df = data.get_data()
        mean = df.mean(axis=1)
        std = df.std(axis=1)
        # df.sub(mean, axis=0) -> subtracts the mean of each row from each element in that row
        # df.div(std, axis=0) -> divides each element in each row by the std of that row
        result_data = df.sub(mean, axis=0).div(std, axis=0)
        result_data.replace([np.inf, -np.inf], np.nan, inplace=True)
        result_data.fillna(0, inplace=True)
        result = PanelContainer(result_data)
        self.validate_output(result)
        return result 
