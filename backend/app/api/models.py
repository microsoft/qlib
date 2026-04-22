from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.models.model_version import ModelVersion
from app.models.user import User
from app.schemas.model_version import ModelVersionResponse, ModelResponse
from app.services.model_version import get_model_version, get_model_versions, delete_model_version
from app.api.deps import get_current_active_user, get_current_developer_user, get_db

router = APIRouter()

@router.get("/", response_model=ModelResponse)
def read_model_versions(page: int = 1, per_page: int = 10, db: Session = Depends(get_db), current_user: User = Depends(get_current_active_user)):
    skip = (page - 1) * per_page
    items, total = get_model_versions(db=db, skip=skip, limit=per_page)
    return ModelResponse(data=items, total=total, page=page, per_page=per_page)

@router.get("/{model_id}", response_model=ModelVersionResponse)
def read_model_version(model_id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_active_user)):
    return get_model_version(db=db, model_id=model_id)

@router.delete("/{model_id}")
def delete_existing_model(model_id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_developer_user)):
    return delete_model_version(db=db, model_id=model_id)

@router.get("/experiment/{experiment_id}", response_model=ModelResponse)
def read_model_versions_by_experiment(experiment_id: int, page: int = 1, per_page: int = 10, db: Session = Depends(get_db), current_user: User = Depends(get_current_active_user)):
    skip = (page - 1) * per_page
    items, total = get_model_versions(db=db, experiment_id=experiment_id, skip=skip, limit=per_page)
    return ModelResponse(data=items, total=total, page=page, per_page=per_page)
