from datetime import datetime
from sqlalchemy.orm import Session
from app.models.experiment import Experiment
from app.schemas.experiment import ExperimentCreate, ExperimentUpdate


def convert_timestamps(config):
    """递归转换配置中的时间戳，移除时区信息"""
    if isinstance(config, dict):
        return {k: convert_timestamps(v) for k, v in config.items()}
    elif isinstance(config, list):
        return [convert_timestamps(item) for item in config]
    elif isinstance(config, str) and 'T' in config and ('Z' in config or '+' in config or '-' in config):
        # 解析带时区的时间戳
        dt = datetime.fromisoformat(config.replace('Z', '+00:00'))
        # 转换为不带时区的时间戳
        return dt.strftime('%Y-%m-%d')
    else:
        return config


def create_experiment(db: Session, experiment: ExperimentCreate):
    now = datetime.now()
    
    # 转换配置中的时间戳，移除时区信息
    converted_config = convert_timestamps(experiment.config)
    
    db_experiment = Experiment(
        name=experiment.name,
        description=experiment.description,
        config=converted_config,
        created_at=now,
        updated_at=now
    )
    db.add(db_experiment)
    db.commit()
    db.refresh(db_experiment)
    return db_experiment



def get_experiment(db: Session, experiment_id: int):
    return db.query(Experiment).filter(Experiment.id == experiment_id).first()



def get_experiments(db: Session, skip: int = 0, limit: int = 100):
    return db.query(Experiment).offset(skip).limit(limit).all()



def update_experiment(db: Session, experiment_id: int, experiment: ExperimentUpdate):
    db_experiment = get_experiment(db, experiment_id)
    if not db_experiment:
        return None
    
    update_data = experiment.model_dump(exclude_unset=True)
    
    # 如果更新了配置，转换配置中的时间戳
    if 'config' in update_data:
        update_data['config'] = convert_timestamps(update_data['config'])
    
    for field, value in update_data.items():
        setattr(db_experiment, field, value)
    
    db.commit()
    db.refresh(db_experiment)
    return db_experiment



def delete_experiment(db: Session, experiment_id: int):
    db_experiment = get_experiment(db, experiment_id)
    if not db_experiment:
        return False
    
    db.delete(db_experiment)
    db.commit()
    return True
