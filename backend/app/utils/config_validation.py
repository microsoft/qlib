import os
import logging
import re
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class ConfigValidator:
    """配置文件验证器"""
    
    # 必需的环境变量
    REQUIRED_ENV_VARS = [
        "DATABASE_URL",
        "SECRET_KEY",
        "ALGORITHM",
        "ACCESS_TOKEN_EXPIRE_MINUTES",
        "TRAINING_SERVER_URL",
        "TRAINING_SERVER_TIMEOUT"
    ]
    
    # 数据库URL模式
    DB_URL_PATTERNS = [
        r"mysql://",
        r"mysql\+pymysql://",
        r"postgresql://",
        r"postgresql\+psycopg2://",
        r"sqlite://"
    ]
    
    # 有效的算法列表
    VALID_ALGORITHMS = ["HS256", "HS384", "HS512", "RS256", "RS384", "RS512"]
    
    @staticmethod
    def validate_environment_variables() -> Dict[str, Any]:
        """验证环境变量"""
        results = {
            "valid": True,
            "errors": [],
            "warnings": []
        }
        
        logger.info("Starting environment variables validation...")
        
        # 检查必需的环境变量
        for var in ConfigValidator.REQUIRED_ENV_VARS:
            if var not in os.environ or not os.environ[var]:
                results["valid"] = False
                results["errors"].append(f"Required environment variable '{var}' is missing or empty")
                logger.error(f"Required environment variable '{var}' is missing or empty")
            else:
                logger.info(f"Environment variable '{var}' is present")
        
        # 验证DATABASE_URL格式
        if "DATABASE_URL" in os.environ and os.environ["DATABASE_URL"]:
            db_url = os.environ["DATABASE_URL"]
            valid_db_url = any(pattern in db_url for pattern in ConfigValidator.DB_URL_PATTERNS)
            if not valid_db_url:
                results["errors"].append(f"Invalid DATABASE_URL format: {db_url}. Supported formats: mysql://, mysql+pymysql://, postgresql://, postgresql+psycopg2://, sqlite://")
                results["valid"] = False
                logger.error(f"Invalid DATABASE_URL format: {db_url}")
            else:
                logger.info(f"DATABASE_URL format is valid: {db_url}")
        
        # 验证ACCESS_TOKEN_EXPIRE_MINUTES
        if "ACCESS_TOKEN_EXPIRE_MINUTES" in os.environ and os.environ["ACCESS_TOKEN_EXPIRE_MINUTES"]:
            try:
                expire_minutes = int(os.environ["ACCESS_TOKEN_EXPIRE_MINUTES"])
                logger.info(f"ACCESS_TOKEN_EXPIRE_MINUTES is a valid integer: {expire_minutes}")
                
                # 检查合理范围
                if expire_minutes < 5:
                    results["warnings"].append(f"ACCESS_TOKEN_EXPIRE_MINUTES is too short: {expire_minutes} minutes. Recommended: 30-120 minutes")
                    logger.warning(f"ACCESS_TOKEN_EXPIRE_MINUTES is too short: {expire_minutes} minutes")
                elif expire_minutes > 1440:
                    results["warnings"].append(f"ACCESS_TOKEN_EXPIRE_MINUTES is too long: {expire_minutes} minutes. Recommended: 30-120 minutes")
                    logger.warning(f"ACCESS_TOKEN_EXPIRE_MINUTES is too long: {expire_minutes} minutes")
            except ValueError:
                results["errors"].append(f"ACCESS_TOKEN_EXPIRE_MINUTES must be an integer, got: {os.environ['ACCESS_TOKEN_EXPIRE_MINUTES']}")
                results["valid"] = False
                logger.error(f"ACCESS_TOKEN_EXPIRE_MINUTES is not a valid integer: {os.environ['ACCESS_TOKEN_EXPIRE_MINUTES']}")
        
        # 验证TRAINING_SERVER_TIMEOUT
        if "TRAINING_SERVER_TIMEOUT" in os.environ and os.environ["TRAINING_SERVER_TIMEOUT"]:
            try:
                timeout = int(os.environ["TRAINING_SERVER_TIMEOUT"])
                logger.info(f"TRAINING_SERVER_TIMEOUT is a valid integer: {timeout}")
                
                # 检查合理范围
                if timeout < 60:
                    results["warnings"].append(f"TRAINING_SERVER_TIMEOUT is too short: {timeout} seconds. Recommended: 300-3600 seconds")
                    logger.warning(f"TRAINING_SERVER_TIMEOUT is too short: {timeout} seconds")
                elif timeout > 36000:
                    results["warnings"].append(f"TRAINING_SERVER_TIMEOUT is too long: {timeout} seconds. Recommended: 300-3600 seconds")
                    logger.warning(f"TRAINING_SERVER_TIMEOUT is too long: {timeout} seconds")
            except ValueError:
                results["errors"].append(f"TRAINING_SERVER_TIMEOUT must be an integer, got: {os.environ['TRAINING_SERVER_TIMEOUT']}")
                results["valid"] = False
                logger.error(f"TRAINING_SERVER_TIMEOUT is not a valid integer: {os.environ['TRAINING_SERVER_TIMEOUT']}")
        
        # 验证ALGORITHM
        if "ALGORITHM" in os.environ and os.environ["ALGORITHM"]:
            algorithm = os.environ["ALGORITHM"]
            if algorithm not in ConfigValidator.VALID_ALGORITHMS:
                results["errors"].append(f"Invalid algorithm: {algorithm}. Valid algorithms: {', '.join(ConfigValidator.VALID_ALGORITHMS)}")
                results["valid"] = False
                logger.error(f"Invalid algorithm: {algorithm}")
            else:
                logger.info(f"ALGORITHM is valid: {algorithm}")
        
        # 验证TRAINING_SERVER_URL格式
        if "TRAINING_SERVER_URL" in os.environ and os.environ["TRAINING_SERVER_URL"]:
            server_url = os.environ["TRAINING_SERVER_URL"]
            url_pattern = re.compile(r'^https?://[^\s/$.?#].[^\s]*$')
            if not url_pattern.match(server_url):
                results["errors"].append(f"Invalid TRAINING_SERVER_URL format: {server_url}. Must be a valid HTTP/HTTPS URL")
                results["valid"] = False
                logger.error(f"Invalid TRAINING_SERVER_URL format: {server_url}")
            else:
                logger.info(f"TRAINING_SERVER_URL format is valid: {server_url}")
        
        # 检查可选环境变量的推荐值
        if "SECRET_KEY" in os.environ and len(os.environ["SECRET_KEY"]) < 32:
            results["warnings"].append("SECRET_KEY is recommended to be at least 32 characters long for better security")
            logger.warning("SECRET_KEY is less than 32 characters long")
        
        if results["valid"]:
            logger.info("All environment variables are valid!")
        else:
            logger.error(f"Environment variables validation failed with {len(results['errors'])} errors and {len(results['warnings'])} warnings")
        
        return results
    
    @staticmethod
    def test_database_connection() -> Dict[str, Any]:
        """测试数据库连接"""
        results = {
            "valid": True,
            "errors": [],
            "warnings": []
        }
        
        if "DATABASE_URL" not in os.environ or not os.environ["DATABASE_URL"]:
            results["valid"] = False
            results["errors"].append("DATABASE_URL is not set")
            return results
        
        logger.info("Testing database connection...")
        
        try:
            from sqlalchemy import create_engine
            from sqlalchemy.exc import SQLAlchemyError
            
            db_url = os.environ["DATABASE_URL"]
            engine = create_engine(db_url)
            
            # 测试连接
            with engine.connect() as conn:
                logger.info(f"Successfully connected to database: {db_url}")
        except SQLAlchemyError as e:
            results["valid"] = False
            results["errors"].append(f"Failed to connect to database: {e}")
            logger.error(f"Failed to connect to database: {e}")
        except Exception as e:
            results["valid"] = False
            results["errors"].append(f"Unexpected error testing database connection: {e}")
            logger.error(f"Unexpected error testing database connection: {e}")
        
        return results
    
    @staticmethod
    def test_training_server_connection() -> Dict[str, Any]:
        """测试训练服务器连接"""
        results = {
            "valid": True,
            "errors": [],
            "warnings": []
        }
        
        if "TRAINING_SERVER_URL" not in os.environ or not os.environ["TRAINING_SERVER_URL"]:
            results["valid"] = False
            results["errors"].append("TRAINING_SERVER_URL is not set")
            return results
        
        logger.info("Testing training server connection...")
        
        try:
            from app.utils.remote_client import RemoteClient
            import asyncio
            
            remote_client = RemoteClient()
            is_healthy = asyncio.run(remote_client.health_check())
            
            if is_healthy:
                logger.info(f"Successfully connected to training server: {os.environ['TRAINING_SERVER_URL']}")
            else:
                results["valid"] = False
                results["errors"].append(f"Training server is not healthy: {os.environ['TRAINING_SERVER_URL']}")
                logger.error(f"Training server is not healthy: {os.environ['TRAINING_SERVER_URL']}")
        except Exception as e:
            results["valid"] = False
            results["errors"].append(f"Failed to connect to training server: {e}")
            logger.error(f"Failed to connect to training server: {e}")
        
        return results
    
    @staticmethod
    def print_validation_results(results: Dict[str, Any]) -> None:
        """打印验证结果"""
        print("\n" + "=" * 50)
        print("CONFIGURATION VALIDATION RESULTS")
        print("=" * 50)
        
        if results["valid"]:
            print("✓ Validation PASSED!")
        else:
            print("✗ Validation FAILED!")
        
        if results["errors"]:
            print(f"\nErrors ({len(results['errors'])}):")
            for error in results["errors"]:
                print(f"  • {error}")
        
        if results["warnings"]:
            print(f"\nWarnings ({len(results['warnings'])}):")
            for warning in results["warnings"]:
                print(f"  • {warning}")
        
        print("\n" + "=" * 50)
    
    @staticmethod
    def validate_config() -> bool:
        """验证配置并打印结果"""
        all_results = {
            "valid": True,
            "errors": [],
            "warnings": []
        }
        
        # 验证环境变量
        env_results = ConfigValidator.validate_environment_variables()
        all_results["valid"] &= env_results["valid"]
        all_results["errors"].extend(env_results["errors"])
        all_results["warnings"].extend(env_results["warnings"])
        
        # 只有当环境变量验证通过后，才进行后续测试
        if env_results["valid"]:
            # 测试数据库连接
            db_results = ConfigValidator.test_database_connection()
            all_results["valid"] &= db_results["valid"]
            all_results["errors"].extend(db_results["errors"])
            all_results["warnings"].extend(db_results["warnings"])
            
            # 测试训练服务器连接
            server_results = ConfigValidator.test_training_server_connection()
            all_results["valid"] &= server_results["valid"]
            all_results["errors"].extend(server_results["errors"])
            all_results["warnings"].extend(server_results["warnings"])
        
        ConfigValidator.print_validation_results(all_results)
        return all_results["valid"]
