from sqlalchemy.orm import Session
from sqlalchemy import and_
from app.models.stock_data import StockData
from app.schemas.data import StockDataCreate


def create_stock_data(db: Session, stock_data: StockDataCreate):
    db_stock_data = StockData(**stock_data.model_dump())
    db.add(db_stock_data)
    db.commit()
    db.refresh(db_stock_data)
    return db_stock_data


def get_stock_data(db: Session, data_id: int):
    return db.query(StockData).filter(StockData.id == data_id).first()


def get_stock_data_by_code_and_date(db: Session, stock_code: str, date):
    return db.query(StockData).filter(
        StockData.stock_code == stock_code,
        StockData.date == date
    ).first()


def get_stock_data_list(
    db: Session,
    stock_code: str = None,
    start_date = None,
    end_date = None,
    skip: int = 0,
    limit: int = 100
):
    query = db.query(StockData)
    
    # Apply filters
    filters = []
    if stock_code:
        filters.append(StockData.stock_code == stock_code)
    if start_date:
        filters.append(StockData.date >= start_date)
    if end_date:
        filters.append(StockData.date <= end_date)
    
    if filters:
        query = query.filter(and_(*filters))
    
    # Get total count
    total = query.count()
    
    # Apply pagination and sorting
    data = query.order_by(StockData.date.desc()).offset(skip).limit(limit).all()
    
    return data, total


def get_stock_codes(db: Session):
    return db.query(StockData.stock_code).distinct().all()


def refresh_stock_data_from_qlib(db: Session):
    """
    从QLIB刷新股票数据到数据库
    
    Args:
        db: 数据库会话对象
    
    Returns:
        dict: 包含刷新结果的字典
    """
    import logging
    from datetime import datetime, timedelta, date as DateType
    import os
    from app.services.qlib_service import qlib_service
    
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("开始从QLIB刷新股票数据到数据库")
        
        # 使用正确的QLIB数据目录
        provider_uri = os.path.expanduser("~/.qlib/qlib_data/cn_data")
        
        # 强制重新初始化QLIB，以便读取新的数据文件
        qlib_service.init_qlib(provider_uri=provider_uri, force=True)
        
        # 获取实际的股票代码
        real_instruments = qlib_service.get_instruments()
        logger.info(f"从QLIB获取到 {len(real_instruments)} 个股票代码")
        
        # 如果没有实际的股票代码，使用默认代码
        if not real_instruments:
            stock_codes = ["SH600000", "SH600001", "SZ000001", "SZ000002"]
        else:
            # 使用前100个股票代码进行数据刷新，避免资源消耗过大
            stock_codes = real_instruments[:100]
        
        logger.info(f"将刷新以下股票的最新数据: {stock_codes[:10]}... (共 {len(stock_codes)} 个)")
        
        # 获取当前日期和过去30天
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        # 格式化日期
        start_date_str = start_date.strftime("%Y-%m-%d")
        end_date_str = end_date.strftime("%Y-%m-%d")
        
        logger.info(f"刷新时间范围: {start_date_str} 到 {end_date_str}")
        
        # 刷新每个股票代码的数据
        refreshed_count = 0
        for code in stock_codes:
            try:
                logger.info(f"开始刷新股票 {code} 的数据")
                
                # 从QLIB获取最新数据
                stock_data_dict = qlib_service.get_stock_data(code, start_date_str, end_date_str)
                
                logger.info(f"从QLIB获取到股票 {code} 的数据，共 {len(stock_data_dict)} 个键值对")
                logger.debug(f"从QLIB获取到股票 {code} 的数据: {list(stock_data_dict.items())[:5]}")
                
                if stock_data_dict:
                    # 按日期分组数据
                    data_by_date = {}
                    for (date_str, field), value in stock_data_dict.items():
                        if date_str not in data_by_date:
                            data_by_date[date_str] = {}
                        data_by_date[date_str][field] = value
                    
                    logger.info(f"按日期分组后的数据，共 {len(data_by_date)} 个日期")
                    logger.debug(f"按日期分组后的数据: {list(data_by_date.items())[:5]}")
                    
                    # 插入或更新数据
                    for date_str, fields in data_by_date.items():
                        logger.info(f"处理日期 {date_str} 的数据: {fields}")
                        
                        # 确保日期格式正确
                        try:
                            # 解析日期字符串为date对象
                            parsed_date = datetime.strptime(date_str, "%Y-%m-%d").date()
                            logger.debug(f"解析后的日期: {parsed_date}, 类型: {type(parsed_date)}")
                        except ValueError as e:
                            logger.error(f"日期格式错误: {date_str}, 错误: {e}")
                            continue
                        
                        # 检查数据是否已存在
                        existing_data = get_stock_data_by_code_and_date(db, code, parsed_date)
                        
                        if existing_data:
                            # 更新现有数据
                            existing_data.open = fields.get("open", existing_data.open)
                            existing_data.high = fields.get("high", existing_data.high)
                            existing_data.low = fields.get("low", existing_data.low)
                            existing_data.close = fields.get("close", existing_data.close)
                            existing_data.volume = fields.get("volume", existing_data.volume)
                            logger.info(f"更新了股票 {code} 在 {parsed_date} 的数据")
                        else:
                            # 插入新数据
                            new_data = StockData(
                                stock_code=code,
                                date=parsed_date,
                                open=fields.get("open", 0),
                                high=fields.get("high", 0),
                                low=fields.get("low", 0),
                                close=fields.get("close", 0),
                                volume=fields.get("volume", 0)
                            )
                            db.add(new_data)
                            logger.info(f"插入了股票 {code} 在 {parsed_date} 的新数据")
                        
                        refreshed_count += 1
                    
                    logger.info(f"成功刷新股票 {code} 的数据，共处理 {len(data_by_date)} 条记录")
                else:
                    logger.warning(f"未从QLIB获取到股票 {code} 的数据")
            except Exception as e:
                logger.error(f"刷新股票 {code} 数据失败: {str(e)}")
                logger.exception(e)
                continue
        
        # 提交数据库更改
        db.commit()
        
        logger.info(f"从QLIB刷新股票数据完成，共刷新 {refreshed_count} 条记录")
        
        return {
            "status": "success",
            "message": f"从QLIB刷新股票数据完成，共刷新 {refreshed_count} 条记录",
            "refreshed_count": refreshed_count
        }
    except Exception as e:
        logger.error(f"从QLIB刷新股票数据失败: {str(e)}")
        logger.exception(e)
        return {
            "status": "error",
            "message": f"从QLIB刷新股票数据失败: {str(e)}"
        }


def align_data(mode: str, date: str, db: Session = None):
    """
    执行数据对齐操作
    
    Args:
        mode: 对齐模式，'auto' 或 'manual'
        date: 对齐日期，格式为 YYYY-MM-DD
        db: 数据库会话对象，可选
    
    Returns:
        dict: 包含执行结果的字典
    """
    import subprocess
    import logging
    import os
    
    logger = logging.getLogger(__name__)
    
    try:
        # 构建下载URL
        url = f"https://github.com/chenditc/investment_data/releases/download/{date}/qlib_bin.tar.gz"
        
        # 下载文件
        logger.info(f"开始下载数据文件: {url}")
        wget_cmd = ["wget", url, "-O", "/tmp/qlib_bin.tar.gz"]
        wget_result = subprocess.run(wget_cmd, capture_output=True, text=True, check=True)
        logger.info(f"文件下载完成: {wget_result.stdout}")
        
        # 解压文件到目标目录
        target_dir = os.path.expanduser("~/.qlib/qlib_data/cn_data")
        # 确保目标目录存在
        os.makedirs(target_dir, exist_ok=True)
        
        logger.info(f"开始解压文件到目录: {target_dir}")
        tar_cmd = ["tar", "-zxvf", "/tmp/qlib_bin.tar.gz", "-C", target_dir, "--strip-components=1"]
        tar_result = subprocess.run(tar_cmd, capture_output=True, text=True, check=True)
        logger.info(f"文件解压完成: {tar_result.stdout}")
        
        # 清理临时文件
        os.remove("/tmp/qlib_bin.tar.gz")
        logger.info("临时文件已清理")
        
        # 如果提供了数据库会话，从QLIB刷新数据到数据库
        refresh_result = None
        if db:
            refresh_result = refresh_stock_data_from_qlib(db)
        
        result = {
            "status": "success",
            "message": "数据对齐完成",
            "mode": mode,
            "date": date,
            "wget_output": wget_result.stdout,
            "tar_output": tar_result.stdout
        }
        
        # 如果有刷新结果，将其合并到返回结果中
        if refresh_result:
            result["refresh_result"] = refresh_result
        
        return result
    except subprocess.CalledProcessError as e:
        logger.error(f"命令执行失败: {e.cmd}，返回码: {e.returncode}，错误输出: {e.stderr}")
        return {
            "status": "error",
            "message": f"命令执行失败: {e.stderr}",
            "mode": mode,
            "date": date,
            "command": e.cmd,
            "returncode": e.returncode
        }
    except Exception as e:
        logger.error(f"数据对齐失败: {str(e)}")
        return {
            "status": "error",
            "message": f"数据对齐失败: {str(e)}",
            "mode": mode,
            "date": date
        }
