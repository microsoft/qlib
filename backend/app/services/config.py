from sqlalchemy.orm import Session
from app.models.config import Config
from app.schemas.config import ConfigCreate, ConfigUpdate


def create_config(db: Session, config: ConfigCreate):
    db_config = Config(
        name=config.name,
        description=config.description,
        content=config.content,
        type=config.type
    )
    db.add(db_config)
    db.commit()
    db.refresh(db_config)
    return db_config


def get_config(db: Session, config_id: int):
    return db.query(Config).filter(Config.id == config_id).first()


def get_configs(db: Session, skip: int = 0, limit: int = 100):
    return db.query(Config).offset(skip).limit(limit).all()


def update_config(db: Session, config_id: int, config: ConfigUpdate):
    db_config = get_config(db, config_id)
    if not db_config:
        return None
    
    update_data = config.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(db_config, field, value)
    
    db.commit()
    db.refresh(db_config)
    return db_config


def delete_config(db: Session, config_id: int):
    db_config = get_config(db, config_id)
    if not db_config:
        return False
    
    db.delete(db_config)
    db.commit()
    return True
