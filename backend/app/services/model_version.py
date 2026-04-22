from sqlalchemy.orm import Session
from app.models.model_version import ModelVersion
from app.schemas.model_version import ModelVersionCreate


def create_model_version(db: Session, model: ModelVersionCreate):
    db_model = ModelVersion(
        experiment_id=model.experiment_id,
        version=model.version,
        name=model.name,
        metrics=model.metrics,
        path=model.path,
        performance=model.performance
    )
    db.add(db_model)
    db.commit()
    db.refresh(db_model)
    return db_model


def get_model_version(db: Session, model_id: int):
    return db.query(ModelVersion).filter(ModelVersion.id == model_id).first()


def get_model_versions(db: Session, skip: int = 0, limit: int = 100, experiment_id: int = None):
    query = db.query(ModelVersion)
    if experiment_id:
        query = query.filter(ModelVersion.experiment_id == experiment_id)
    total = query.count()
    items = query.offset(skip).limit(limit).all()
    return items, total


def delete_model_version(db: Session, model_id: int):
    db_model = get_model_version(db, model_id)
    if not db_model:
        return False
    
    db.delete(db_model)
    db.commit()
    return True
