from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.db.database import get_db
from app.models.config import Config
from app.models.user import User
from app.schemas.config import ConfigCreate, ConfigResponse, ConfigUpdate
from app.services.config import create_config, get_config, get_configs, update_config, delete_config
from app.yaml.parser import QLibYAMLParser
from app.api.deps import get_current_active_user, get_current_developer_user

router = APIRouter()

@router.post("/", response_model=ConfigResponse)
def create_new_config(config: ConfigCreate, db: Session = Depends(get_db), current_user: User = Depends(get_current_developer_user)):
    # Validate YAML configuration
    try:
        parsed_config = QLibYAMLParser.parse_yaml(config.content)
        if not QLibYAMLParser.validate_yaml(parsed_config):
            raise HTTPException(status_code=400, detail="Invalid QLib YAML configuration")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid YAML format: {str(e)}")
    
    return create_config(db=db, config=config)

@router.get("/", response_model=list[ConfigResponse])
def read_configs(skip: int = 0, limit: int = 100, db: Session = Depends(get_db), current_user: User = Depends(get_current_active_user)):
    return get_configs(db=db, skip=skip, limit=limit)

@router.get("/{config_id}", response_model=ConfigResponse)
def read_config(config_id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_active_user)):
    return get_config(db=db, config_id=config_id)

@router.put("/{config_id}", response_model=ConfigResponse)
def update_existing_config(config_id: int, config: ConfigUpdate, db: Session = Depends(get_db), current_user: User = Depends(get_current_developer_user)):
    # Validate YAML configuration if content is provided
    if config.content:
        try:
            parsed_config = QLibYAMLParser.parse_yaml(config.content)
            if not QLibYAMLParser.validate_yaml(parsed_config):
                raise HTTPException(status_code=400, detail="Invalid QLib YAML configuration")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid YAML format: {str(e)}")
    
    return update_config(db=db, config_id=config_id, config=config)

@router.delete("/{config_id}")
def delete_existing_config(config_id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_developer_user)):
    return delete_config(db=db, config_id=config_id)
