from sqlalchemy.orm import Session, joinedload
from app.models.factor import Factor, FactorGroup
from app.schemas.factor import FactorCreate, FactorUpdate, FactorGroupCreate, FactorGroupUpdate
from app.services.qlib_factor import QlibFactorService

# Factor Group functions
def create_factor_group(db: Session, factor_group: FactorGroupCreate):
    db_factor_group = FactorGroup(**factor_group.model_dump())
    db.add(db_factor_group)
    db.commit()
    db.refresh(db_factor_group)
    return db_factor_group


def get_factor_group(db: Session, factor_group_id: int):
    return db.query(FactorGroup).filter(FactorGroup.id == factor_group_id).first()


def get_factor_group_by_name(db: Session, name: str):
    return db.query(FactorGroup).filter(FactorGroup.name == name).first()


def get_factor_groups(db: Session, skip: int = 0, limit: int = 100):
    return db.query(FactorGroup).offset(skip).limit(limit).all()


def get_factor_group_with_factors(db: Session, factor_group_id: int):
    return db.query(FactorGroup).options(joinedload(FactorGroup.factors)).filter(FactorGroup.id == factor_group_id).first()


def update_factor_group(db: Session, factor_group_id: int, factor_group: FactorGroupUpdate):
    db_factor_group = get_factor_group(db, factor_group_id)
    if not db_factor_group:
        return None
    update_data = factor_group.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(db_factor_group, field, value)
    db.commit()
    db.refresh(db_factor_group)
    return db_factor_group


def delete_factor_group(db: Session, factor_group_id: int):
    db_factor_group = get_factor_group(db, factor_group_id)
    if not db_factor_group:
        return None
    db.delete(db_factor_group)
    db.commit()
    return db_factor_group

# Factor functions
def create_factor(db: Session, factor: FactorCreate):
    db_factor = Factor(**factor.model_dump())
    db.add(db_factor)
    db.commit()
    db.refresh(db_factor)
    return db_factor


def get_factor(db: Session, factor_id: int):
    return db.query(Factor).options(joinedload(Factor.group)).filter(Factor.id == factor_id).first()


def get_factor_by_name(db: Session, name: str):
    return db.query(Factor).filter(Factor.name == name).first()


def get_factors(db: Session, skip: int = 0, limit: int = 100):
    try:
        # Try with joined load of group
        return db.query(Factor).options(joinedload(Factor.group)).offset(skip).limit(limit).all()
    except Exception as e:
        # Fallback: try without joined load if group_id column doesn't exist
        print(f"Error loading factors with group: {e}")
        return db.query(Factor).offset(skip).limit(limit).all()


def update_factor(db: Session, factor_id: int, factor: FactorUpdate):
    db_factor = get_factor(db, factor_id)
    if not db_factor:
        return None
    update_data = factor.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(db_factor, field, value)
    db.commit()
    db.refresh(db_factor)
    return db_factor


def delete_factor(db: Session, factor_id: int):
    db_factor = get_factor(db, factor_id)
    if not db_factor:
        return None
    db.delete(db_factor)
    db.commit()
    return db_factor
