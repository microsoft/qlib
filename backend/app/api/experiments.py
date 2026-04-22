from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.db.database import get_db
from app.models.experiment import Experiment
from app.models.user import User
from app.schemas.experiment import ExperimentCreate, ExperimentResponse, ExperimentUpdate
from app.services.experiment import create_experiment, get_experiment, get_experiments, update_experiment, delete_experiment
from app.services.analysis import AnalysisService
from app.services.task import TaskService
from app.api.deps import get_current_active_user, get_current_developer_user

router = APIRouter()

@router.post("/", response_model=ExperimentResponse)
def create_new_experiment(experiment: ExperimentCreate, db: Session = Depends(get_db), current_user: User = Depends(get_current_developer_user)):
    return create_experiment(db=db, experiment=experiment)

@router.get("/", response_model=list[ExperimentResponse])
def read_experiments(skip: int = 0, limit: int = 100, db: Session = Depends(get_db), current_user: User = Depends(get_current_active_user)):
    return get_experiments(db=db, skip=skip, limit=limit)



@router.get("/{experiment_id}", response_model=ExperimentResponse)
def read_experiment(experiment_id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_active_user)):
    return get_experiment(db=db, experiment_id=experiment_id)

@router.put("/{experiment_id}", response_model=ExperimentResponse)
def update_existing_experiment(experiment_id: int, experiment: ExperimentUpdate, db: Session = Depends(get_db), current_user: User = Depends(get_current_developer_user)):
    return update_experiment(db=db, experiment_id=experiment_id, experiment=experiment)

@router.delete("/{experiment_id}")
def delete_existing_experiment(experiment_id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_developer_user)):
    return delete_experiment(db=db, experiment_id=experiment_id)

@router.get("/{experiment_id}/logs")
def get_experiment_logs(experiment_id: int, skip: int = 0, limit: int = 100, db: Session = Depends(get_db), current_user: User = Depends(get_current_active_user)):
    from app.models.log import ExperimentLog
    
    experiment = get_experiment(db=db, experiment_id=experiment_id)
    if not experiment:
        raise HTTPException(status_code=404, detail="Experiment not found")
    
    # 获取日志记录，按时间倒序排列
    logs = db.query(ExperimentLog).filter(ExperimentLog.experiment_id == experiment_id).order_by(ExperimentLog.created_at.desc()).offset(skip).limit(limit).all()
    
    # 格式化日志输出
    formatted_logs = []
    for log in logs:
        timestamp = log.created_at.strftime("%Y-%m-%d %H:%M:%S")
        formatted_logs.append(f"[{timestamp}] {log.message}")
    
    # 获取日志总数
    total = db.query(ExperimentLog).filter(ExperimentLog.experiment_id == experiment_id).count()
    
    return {
        "logs": "\n".join(formatted_logs),
        "total": total,
        "skip": skip,
        "limit": limit
    }

@router.post("/{experiment_id}/run")
def run_experiment(experiment_id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_developer_user)):
    experiment = get_experiment(db=db, experiment_id=experiment_id)
    if not experiment:
        raise HTTPException(status_code=404, detail="Experiment not found")
    
    # Update experiment status to pending immediately
    experiment.status = "pending"
    experiment.progress = 0.0
    experiment.error = None
    # Don't set experiment.logs to None - it's a relationship, not a column
    db.commit()
    
    try:
        # Create task to run the experiment
        task = TaskService.create_task(db=db, experiment_id=experiment_id, task_type="train")
        
        return {"message": "Experiment started successfully", "experiment_id": experiment_id, "task_id": task.id}
    except Exception as e:
        import traceback
        traceback_str = traceback.format_exc()
        print(f"Error in run_experiment: {e}")
        print(f"Traceback: {traceback_str}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.get("/{experiment_id}/analysis")
def get_experiment_analysis(experiment_id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_active_user)):
    """Get all analysis data for an experiment"""
    experiment = get_experiment(db=db, experiment_id=experiment_id)
    if not experiment:
        raise HTTPException(status_code=404, detail="Experiment not found")
    
    return AnalysisService.get_full_analysis(experiment)

@router.get("/{experiment_id}/analysis/signal")
def get_signal_analysis(experiment_id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_active_user)):
    """Get signal analysis data"""
    experiment = get_experiment(db=db, experiment_id=experiment_id)
    if not experiment:
        raise HTTPException(status_code=404, detail="Experiment not found")
    
    return AnalysisService.generate_signal_analysis(experiment)

@router.get("/{experiment_id}/analysis/portfolio")
def get_portfolio_analysis(experiment_id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_active_user)):
    """Get portfolio analysis data"""
    experiment = get_experiment(db=db, experiment_id=experiment_id)
    if not experiment:
        raise HTTPException(status_code=404, detail="Experiment not found")
    
    return AnalysisService.generate_portfolio_analysis(experiment)

@router.get("/{experiment_id}/analysis/backtest")
def get_backtest_analysis(experiment_id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_active_user)):
    """Get backtest analysis data"""
    experiment = get_experiment(db=db, experiment_id=experiment_id)
    if not experiment:
        raise HTTPException(status_code=404, detail="Experiment not found")
    
    return AnalysisService.generate_backtest_analysis(experiment)

@router.get("/profit-loss")
def get_profit_loss(db: Session = Depends(get_db), current_user: User = Depends(get_current_active_user)):
    """Get profit loss data for all experiments"""
    return AnalysisService.generate_profit_loss_data()
