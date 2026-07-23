import logging
from typing import List, Dict, Any
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QLibService:
    _initialized = False
    _qlib = None
    _D = None
    _REG_CN = None
    
    @classmethod
    def _try_import_qlib(cls) -> bool:
        """
        Try to import QLib modules
        
        Returns:
            bool: True if import successful, False otherwise
        """
        try:
            import qlib
            from qlib.data import D
            from qlib.config import REG_CN
            
            cls._qlib = qlib
            cls._D = D
            cls._REG_CN = REG_CN
            return True
        except ImportError as e:
            logger.error(f"Failed to import QLib modules: {e}")
            return False
    
    @classmethod
    def init_qlib(cls, provider_uri: str = "~/.qlib/qlib_data/cn_data", force: bool = False) -> bool:
        """
        Initialize QLib
        
        Args:
            provider_uri: Path to QLib data directory
            force: Whether to force reinitialization even if already initialized
            
        Returns:
            bool: True if initialization successful, False otherwise
        """
        if cls._initialized and not force:
            logger.info("QLib is already initialized")
            return True
        
        try:
            # Try to import QLib modules if not already imported
            if cls._qlib is None:
                if not cls._try_import_qlib():
                    return False
            
            logger.info(f"Initializing QLib with provider_uri: {provider_uri}")
            
            # Check if provider_uri exists
            if not os.path.exists(provider_uri):
                logger.warning(f"QLib data directory not found: {provider_uri}")
                logger.info(f"Creating QLib data directory: {provider_uri}")
                os.makedirs(provider_uri, exist_ok=True)
            
            # Try different initialization methods
            try:
                # Method 1: Using qlib.init
                cls._qlib.init(provider_uri=provider_uri, region=cls._REG_CN)
                cls._initialized = True
                logger.info("QLib initialized successfully using qlib.init")
                return True
            except AttributeError:
                # Method 2: QLib might not have init method, try direct initialization
                logger.info("QLib.init() not found, trying direct initialization")
                # Check if data is accessible
                try:
                    cls._D.instruments(market="all")
                    cls._initialized = True
                    logger.info("QLib initialized successfully using direct initialization")
                    return True
                except Exception as e:
                    logger.error(f"Failed to initialize QLib directly: {e}")
                    return False
        except Exception as e:
            logger.error(f"Failed to initialize QLib: {e}")
            logger.exception(e)
            return False
    
    @classmethod
    def get_instruments(cls, market: str = "all") -> List[str]:
        """
        Get instruments from QLib
        
        Args:
            market: Market name or list of instruments
            
        Returns:
            List[str]: List of instruments
        """
        if not cls._initialized:
            if not cls.init_qlib():
                logger.warning("Failed to initialize QLib, returning default instruments")
                return ["SH600000", "SH600001", "SZ000001", "SZ000002"]
        
        try:
            instruments = cls._D.instruments(market)
            logger.info(f"Got instruments from market: {market}, type: {type(instruments)}")
            
            # 处理不同类型的返回值
            if isinstance(instruments, dict):
                # 如果返回的是字典，尝试从instrument文件中读取实际的股票代码
                logger.warning(f"D.instruments()返回的是字典，尝试从instrument文件中读取股票代码")
                
                # 构建instrument文件路径
                provider_uri = os.path.expanduser("~/.qlib/qlib_data/cn_data")
                instrument_file = os.path.join(provider_uri, "instruments", f"{market}.txt")
                
                # 如果指定的market文件不存在，使用all.txt
                if not os.path.exists(instrument_file):
                    instrument_file = os.path.join(provider_uri, "instruments", "all.txt")
                    logger.info(f"使用默认的instrument文件: {instrument_file}")
                
                try:
                    # 读取instrument文件，提取股票代码
                    with open(instrument_file, 'r') as f:
                        # 每行格式: 股票代码	上市日期	退市日期
                        # 只提取股票代码部分
                        stock_codes = [line.split('\t')[0] for line in f if line.strip()]
                    
                    logger.info(f"从文件中读取到 {len(stock_codes)} 个股票代码")
                    
                    # 过滤出SH和SZ开头的股票代码
                    sh_sz_codes = [code for code in stock_codes if code.startswith(('SH', 'SZ'))]
                    logger.info(f"过滤后得到 {len(sh_sz_codes)} 个SH/SZ开头的股票代码")
                    
                    return sh_sz_codes
                except Exception as file_e:
                    logger.error(f"读取instrument文件失败: {file_e}")
                    logger.exception(file_e)
                    # 如果读取文件失败，返回默认股票代码列表
                    return ["SH600000", "SH600001", "SZ000001", "SZ000002"]
            elif hasattr(instruments, '__iter__') and not isinstance(instruments, (str, bytes)):
                # 如果返回的是可迭代对象，转换为列表
                return list(instruments)
            else:
                # 否则，返回默认股票代码列表
                logger.warning(f"D.instruments()返回的是未知类型，使用默认股票代码列表")
                return ["SH600000", "SH600001", "SZ000001", "SZ000002"]
        except Exception as e:
            logger.error(f"Failed to get instruments: {e}")
            logger.exception(e)
            return ["SH600000", "SH600001", "SZ000001", "SZ000002"]
    
    @classmethod
    def get_stock_data(cls, instrument: str, start_date: str, end_date: str, fields: List[str] = None) -> Dict[str, Any]:
        """
        Get stock data from QLib
        
        Args:
            instrument: Stock code
            start_date: Start date in format YYYY-MM-DD
            end_date: End date in format YYYY-MM-DD
            fields: List of fields to get
            
        Returns:
            Dict[str, Any]: Stock data in format {(date_str, field): value}
        """
        if not cls._initialized:
            if not cls.init_qlib():
                logger.warning("Failed to initialize QLib, returning empty stock data")
                return {}
        
        try:
            if fields is None:
                # 在QLIB中，字段名需要使用$前缀，比如$close而不是close
                fields = ["$open", "$high", "$low", "$close", "$volume"]
            
            logger.info(f"正在从QLIB获取股票 {instrument} 从 {start_date} 到 {end_date} 的数据，字段: {fields}")
            
            # Get data from QLib
            data = cls._D.features([instrument], fields, start_date, end_date)
            
            logger.info(f"从QLIB获取到股票 {instrument} 的数据，形状: {data.shape}")
            logger.debug(f"数据列: {data.columns.tolist()}")
            logger.debug(f"数据索引: {data.index.tolist()[:5]}")
            
            # 检查数据是否为空
            if data.empty:
                logger.warning(f"从QLIB获取到的股票 {instrument} 的数据为空")
                return {}
            
            # 转换数据格式为 {(date_str, field): value}
            result = {}
            for date in data.index:
                logger.debug(f"处理日期: {date}, 类型: {type(date)}")
                
                # 解析日期，处理不同类型的索引
                if isinstance(date, tuple):
                    # 如果索引是元组，第二个元素是时间戳
                    timestamp = date[1]
                else:
                    # 否则，直接使用日期
                    timestamp = date
                
                # 格式化日期为YYYY-MM-DD
                date_str = timestamp.strftime("%Y-%m-%d")
                logger.debug(f"格式化后的日期: {date_str}")
                
                for field in data.columns:
                    # 移除字段名中的$前缀，以便存储到数据库中
                    field_name = field.replace("$", "")
                    result[(date_str, field_name)] = data.loc[date, field]
                    logger.debug(f"处理字段: {field}, 存储为: {field_name}, 值: {data.loc[date, field]}")
            
            logger.info(f"成功转换股票 {instrument} 的数据，共 {len(result)} 条记录")
            logger.debug(f"转换后的数据: {list(result.items())[:5]}")
            return result
        except Exception as e:
            logger.error(f"从QLIB获取股票 {instrument} 数据失败: {e}")
            logger.exception(e)
            return {}
    
    @classmethod
    def get_factors(cls, instrument: str, start_date: str, end_date: str, factors: List[str] = None) -> Dict[str, Any]:
        """
        Get factors from QLib
        
        Args:
            instrument: Stock code
            start_date: Start date in format YYYY-MM-DD
            end_date: End date in format YYYY-MM-DD
            factors: List of factors to get
            
        Returns:
            Dict[str, Any]: Factor data
        """
        if not cls._initialized:
            if not cls.init_qlib():
                logger.warning("Failed to initialize QLib, returning empty factor data")
                return {}
        
        try:
            if factors is None:
                factors = ["MACD", "RSI", "KDJ"]
            
            # Get factor data from QLib
            data = cls._D.features([instrument], factors, start_date, end_date)
            logger.info(f"Got factors for {instrument} from {start_date} to {end_date}")
            return data.to_dict()
        except Exception as e:
            logger.error(f"Failed to get factors: {e}")
            logger.exception(e)
            return {}
    
    @classmethod
    def is_initialized(cls) -> bool:
        """
        Check if QLib is initialized
        
        Returns:
            bool: True if initialized, False otherwise
        """
        return cls._initialized
    
    @classmethod
    def is_available(cls) -> bool:
        """
        Check if QLib is available
        
        Returns:
            bool: True if QLib is available, False otherwise
        """
        if cls._qlib is None:
            return cls._try_import_qlib()
        return True

# Create a singleton instance
qlib_service = QLibService()