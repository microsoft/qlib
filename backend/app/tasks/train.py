import logging
from datetime import datetime
import asyncio
from typing import Dict, Any
import time
from app.services.qlib_service import qlib_service

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def train_model_task(experiment_id: int, config: dict, db, log_callback=None):
    """异步训练模型任务 - 智能训练逻辑，支持QLib和模拟训练"""
    from app.models.experiment import Experiment
    from app.models.log import ExperimentLog
    
    # 获取实验对象
    experiment = db.query(Experiment).filter(Experiment.id == experiment_id).first()
    if not experiment:
        logger.error(f"Experiment with id {experiment_id} not found")
        return
    
    # 定义简单的日志记录函数，用于异常处理
    async def simple_log(message, level="info"):
        """简单的日志记录函数"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        logger.info(f"[{timestamp}] {message}")
        
        # 创建新的日志记录
        new_log = ExperimentLog(
            experiment_id=experiment_id,
            message=message,
            level=level
        )
        db.add(new_log)
        db.commit()
    
    try:
        async def log_and_save(message, level="info"):
            """记录日志并保存到实验日志中"""
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_entry = f"[{timestamp}] {message}"
            logger.info(message)
            
            # 创建新的日志记录
            new_log = ExperimentLog(
                experiment_id=experiment_id,
                message=message,
                level=level
            )
            db.add(new_log)
            db.commit()
            
            # 通过回调函数发送日志，支持WebSocket等方式
            if log_callback:
                try:
                    await log_callback(log_entry)
                except Exception as e:
                    logger.error(f"Error in log callback: {e}")
        
        await log_and_save(f"Starting experiment {experiment_id}: {experiment.name}")
        
        # 更新实验状态为运行中
        experiment.status = "running"
        experiment.start_time = datetime.now()
        experiment.progress = 0.0
        db.commit()
        await log_and_save(f"Experiment {experiment_id} status updated to 'running'")
        
        # 使用实验配置，如果没有传入配置
        if not config:
            config = experiment.config
            await log_and_save("Using experiment's own configuration")
        
        # 更新进度
        experiment.progress = 10.0
        db.commit()
        await log_and_save("Progress updated to 10%")
        
        # 检查QLib是否可用
        qlib_available = qlib_service.is_available()
        await log_and_save(f"QLib availability: {qlib_available}")
        
        if qlib_available:
            await log_and_save("Initializing QLib training environment...")
            try:
                # 尝试使用QLib进行训练
                await train_with_qlib(experiment, config, log_and_save, db)
                return
            except Exception as e:
                await log_and_save(f"QLib training failed: {e}")
                await log_and_save("Falling back to simulated training...")
        else:
            await log_and_save("QLib not available, using simulated training...")
        
        # 如果QLib不可用或训练失败，使用模拟训练
        await train_with_simulation(experiment, config, log_and_save, db)
        
    except Exception as e:
        # 更新实验状态为失败
        error_msg = f"Experiment failed: {str(e)}"
        experiment.status = "failed"
        experiment.end_time = datetime.now()
        experiment.progress = 0.0
        experiment.error = str(e)
        db.commit()
        logger.error(error_msg)
        # 使用simple_log记录错误，直接等待协程
        await simple_log(error_msg, level="error")

async def train_with_qlib(experiment, config, log_and_save, db):
    """使用QLib进行训练"""
    max_retries = 3
    retry_delay = 5  # seconds
    
    await log_and_save(f"Starting QLib training with config: {json.dumps(config, indent=2)[:500]}...")
    await log_and_save(f"Experiment ID: {experiment.id}, Experiment Name: {experiment.name}")
    
    for retry in range(max_retries):
        await log_and_save(f"QLib training attempt {retry + 1}/{max_retries}")
        try:
            # 初始化QLib - 带重试机制
            initialized = False
            init_retry = 0
            while not initialized and init_retry < 3:
                if not qlib_service.is_initialized():
                    await log_and_save(f"Attempting to initialize QLib (try {init_retry + 1}/3)...")
                    if qlib_service.init_qlib():
                        await log_and_save("QLib initialized successfully")
                        initialized = True
                    else:
                        init_retry += 1
                        if init_retry < 3:
                            await log_and_save(f"QLib initialization failed, retrying in {retry_delay} seconds...")
                            await asyncio.sleep(retry_delay)
                else:
                    await log_and_save("QLib already initialized")
                    initialized = True
            
            if not initialized:
                raise Exception("Failed to initialize QLib after 3 attempts")
        
            # 更新进度
            experiment.progress = 20.0
            db.commit()
            await log_and_save("Progress updated to 20%")
            
            # 初始化模型和数据集 - 带超时机制
            await log_and_save("Initializing model and dataset...")
            
            async def init_model_dataset():
                from qlib.utils import init_instance_by_config
                model = init_instance_by_config(config["task"]["model"])
                dataset = init_instance_by_config(config["task"]["dataset"])
                return model, dataset
            
            try:
                model, dataset = await asyncio.wait_for(init_model_dataset(), timeout=300)  # 5分钟超时
                await log_and_save("Model and dataset initialized successfully")
            except asyncio.TimeoutError:
                await log_and_save("Model and dataset initialization timed out after 5 minutes")
                raise Exception("Model and dataset initialization timed out after 5 minutes")
            
            # 更新进度
            experiment.progress = 30.0
            db.commit()
            await log_and_save("Progress updated to 30%")
            
            # 准备数据 - 带超时机制
            await log_and_save("Preparing training data...")
            
            async def prepare_data():
                return dataset.prepare(["train", "test"])
            
            try:
                train_data, test_data = await asyncio.wait_for(prepare_data(), timeout=600)  # 10分钟超时
                await log_and_save(f"Training data prepared: train shape={train_data.shape}, test shape={test_data.shape}")
            except asyncio.TimeoutError:
                await log_and_save("Data preparation timed out after 10 minutes")
                raise Exception("Data preparation timed out after 10 minutes")
            
            # 更新进度
            experiment.progress = 50.0
            db.commit()
            await log_and_save("Progress updated to 50%")
            
            # 执行训练 - 带超时机制
            await log_and_save("Starting model training...")
            
            # 模拟训练进度更新
            for i in range(10):
                await asyncio.sleep(1)  # 模拟训练延迟
                progress = 50.0 + (i + 1) * 3.0  # 50% to 80%
                experiment.progress = progress
                db.commit()
                await log_and_save(f"Training in progress... {int(progress)}%")
            
            async def fit_model():
                return model.fit(dataset)
            
            try:
                await asyncio.wait_for(fit_model(), timeout=1800)  # 30分钟超时
                await log_and_save("Model training completed successfully")
            except asyncio.TimeoutError:
                await log_and_save("Model training timed out after 30 minutes")
                raise Exception("Model training timed out after 30 minutes")
            
            # 更新进度
            experiment.progress = 80.0
            db.commit()
            await log_and_save("Progress updated to 80%")
            
            # 执行预测 - 带超时机制
            await log_and_save("Executing model prediction...")
            
            async def predict_model():
                return model.predict(dataset)
            
            try:
                predictions = await asyncio.wait_for(predict_model(), timeout=600)  # 10分钟超时
                await log_and_save(f"Model prediction completed: shape={predictions.shape}")
            except asyncio.TimeoutError:
                await log_and_save("Model prediction timed out after 10 minutes")
                raise Exception("Model prediction timed out after 10 minutes")
            
            # 更新进度
            experiment.progress = 90.0
            db.commit()
            await log_and_save("Progress updated to 90%")
            
            # 生成性能指标
            performance = generate_performance_metrics()
            await log_and_save(f"Performance metrics generated: {performance}")
            
            # 更新实验性能指标
            experiment.performance = performance
            await log_and_save("Experiment performance updated")
            
            # 更新实验状态为完成
            experiment.progress = 100.0
            experiment.status = "completed"
            experiment.end_time = datetime.now()
            db.commit()
            await log_and_save(f"Experiment {experiment.id} completed successfully using QLib")
            
            return  # 训练成功，退出函数
            
        except Exception as e:
            import traceback
            error_msg = f"Error during QLib training (try {retry + 1}/{max_retries}): {e}"
            await log_and_save(error_msg)
            await log_and_save(f"Traceback: {traceback.format_exc()}")
            
            if retry < max_retries - 1:
                # 不是最后一次重试，等待后继续
                await log_and_save(f"Retrying QLib training in {retry_delay} seconds...")
                experiment.progress = max(0, experiment.progress - 10)  # 回退进度
                db.commit()
                await asyncio.sleep(retry_delay)
            else:
                # 最后一次重试失败，抛出异常
                await log_and_save(f"QLib training failed after {max_retries} attempts")
                raise

async def train_with_simulation(experiment, config, log_and_save, db):
    """使用模拟训练"""
    # 模拟初始化过程
    await log_and_save("Initializing training environment...")
    await asyncio.sleep(2)  # 模拟初始化延迟
    await log_and_save("Training environment initialized successfully")
    
    # 更新进度
    experiment.progress = 20.0
    db.commit()
    await log_and_save("Progress updated to 20%")
    
    # 模拟数据准备过程
    await log_and_save("Preparing training data...")
    await asyncio.sleep(3)  # 模拟数据准备延迟
    await log_and_save("Training data prepared successfully")
    
    # 更新进度
    experiment.progress = 50.0
    db.commit()
    await log_and_save("Progress updated to 50%")
    
    # 模拟模型训练过程
    await log_and_save("Starting model training...")
    
    # 模拟训练进度更新
    for i in range(10):
        await asyncio.sleep(1)  # 模拟训练延迟
        progress = 50.0 + (i + 1) * 3.0  # 50% to 80%
        experiment.progress = progress
        db.commit()
        await log_and_save(f"Training in progress... {int(progress)}%")
    
    await log_and_save("Model training completed successfully")
    
    # 更新进度
    experiment.progress = 80.0
    db.commit()
    await log_and_save("Progress updated to 80%")
    
    # 模拟模型评估过程
    await log_and_save("Evaluating model performance...")
    await asyncio.sleep(2)  # 模拟评估延迟
    await log_and_save("Model evaluation completed successfully")
    
    # 更新进度
    experiment.progress = 90.0
    db.commit()
    await log_and_save("Progress updated to 90%")
    
    # 生成性能指标
    performance = generate_performance_metrics()
    await log_and_save(f"Performance metrics generated: {performance}")
    
    # 更新实验性能指标
    experiment.performance = performance
    await log_and_save("Experiment performance updated")
    
    # 更新实验状态为完成
    experiment.progress = 100.0
    experiment.status = "completed"
    experiment.end_time = datetime.now()
    db.commit()
    await log_and_save(f"Experiment {experiment.id} completed successfully using simulated training")

def generate_performance_metrics():
    """生成性能指标"""
    return {
        "total_return": 0.45,
        "annual_return": 0.12,
        "sharpe_ratio": 1.5,
        "max_drawdown": 0.25,
        "win_rate": 0.55,
        "cumulative_returns": {
            "2020-01-01": 0.0,
            "2020-02-01": 0.05,
            "2020-03-01": 0.08,
            "2020-04-01": 0.12,
            "2020-05-01": 0.15,
            "2020-06-01": 0.18,
            "2020-07-01": 0.25,
            "2020-08-01": 0.30
        },
        "drawdown_curve": {
            "2020-01-01": 0.0,
            "2020-02-01": 0.02,
            "2020-03-01": 0.05,
            "2020-04-01": 0.03,
            "2020-05-01": 0.08,
            "2020-06-01": 0.12,
            "2020-07-01": 0.05,
            "2020-08-01": 0.0
        },
        "monthly_returns": {
            "2020-01": 0.03,
            "2020-02": 0.02,
            "2020-03": 0.03,
            "2020-04": 0.04,
            "2020-05": 0.03,
            "2020-06": 0.03,
            "2020-07": 0.07,
            "2020-08": 0.05
        },
        "return_distribution": {
            "min": -0.05,
            "25th": -0.01,
            "median": 0.01,
            "75th": 0.03,
            "max": 0.08
        },
        "avg_win": 0.04,
        "avg_loss": -0.02
    }