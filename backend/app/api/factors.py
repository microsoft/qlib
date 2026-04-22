from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.db.database import get_db
from app.models.user import User
from app.schemas.factor import (
    FactorCreate, FactorResponse, FactorUpdate,
    FactorGroupCreate, FactorGroupResponse, FactorGroupUpdate, FactorGroupWithFactors
)
from app.services.factor import (
    create_factor, get_factor, get_factors, update_factor, delete_factor, get_factor_by_name,
    create_factor_group, get_factor_group, get_factor_groups, update_factor_group, delete_factor_group,
    get_factor_group_by_name, get_factor_group_with_factors
)
from app.api.deps import get_current_active_user, get_current_developer_user

router = APIRouter()

# Factor Group endpoints
@router.post("/groups", response_model=FactorGroupResponse)
def create_new_factor_group(
    factor_group: FactorGroupCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_developer_user)
):
    db_factor_group = get_factor_group_by_name(db, name=factor_group.name)
    if db_factor_group:
        raise HTTPException(status_code=400, detail="Factor group with this name already exists")
    return create_factor_group(db=db, factor_group=factor_group)

@router.get("/groups", response_model=list[FactorGroupResponse])
def read_factor_groups(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    return get_factor_groups(db=db, skip=skip, limit=limit)

@router.get("/groups/{factor_group_id}", response_model=FactorGroupResponse)
def read_factor_group(
    factor_group_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    db_factor_group = get_factor_group(db=db, factor_group_id=factor_group_id)
    if db_factor_group is None:
        raise HTTPException(status_code=404, detail="Factor group not found")
    return db_factor_group

@router.get("/groups/{factor_group_id}/factors", response_model=FactorGroupWithFactors)
def read_factor_group_with_factors(
    factor_group_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    db_factor_group = get_factor_group_with_factors(db=db, factor_group_id=factor_group_id)
    if db_factor_group is None:
        raise HTTPException(status_code=404, detail="Factor group not found")
    return db_factor_group

@router.put("/groups/{factor_group_id}", response_model=FactorGroupResponse)
def update_existing_factor_group(
    factor_group_id: int,
    factor_group: FactorGroupUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_developer_user)
):
    db_factor_group = update_factor_group(db=db, factor_group_id=factor_group_id, factor_group=factor_group)
    if db_factor_group is None:
        raise HTTPException(status_code=404, detail="Factor group not found")
    return db_factor_group

@router.delete("/groups/{factor_group_id}")
def delete_existing_factor_group(
    factor_group_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_developer_user)
):
    db_factor_group = delete_factor_group(db=db, factor_group_id=factor_group_id)
    if db_factor_group is None:
        raise HTTPException(status_code=404, detail="Factor group not found")
    return {"message": "Factor group deleted successfully"}

# Factor endpoints
@router.post("/", response_model=FactorResponse)
def create_new_factor(
    factor: FactorCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_developer_user)
):
    db_factor = get_factor_by_name(db, name=factor.name)
    if db_factor:
        raise HTTPException(status_code=400, detail="Factor with this name already exists")
    return create_factor(db=db, factor=factor)

@router.get("/", response_model=list[FactorResponse])
def read_factors(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    return get_factors(db=db, skip=skip, limit=limit)

@router.get("/{factor_id}", response_model=FactorResponse)
def read_factor(
    factor_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    db_factor = get_factor(db=db, factor_id=factor_id)
    if db_factor is None:
        raise HTTPException(status_code=404, detail="Factor not found")
    return db_factor

@router.put("/{factor_id}", response_model=FactorResponse)
def update_existing_factor(
    factor_id: int,
    factor: FactorUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_developer_user)
):
    db_factor = update_factor(db=db, factor_id=factor_id, factor=factor)
    if db_factor is None:
        raise HTTPException(status_code=404, detail="Factor not found")
    return db_factor

@router.delete("/{factor_id}")
def delete_existing_factor(
    factor_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_developer_user)
):
    db_factor = delete_factor(db=db, factor_id=factor_id)
    if db_factor is None:
        raise HTTPException(status_code=404, detail="Factor not found")
    return {"message": "Factor deleted successfully"}
