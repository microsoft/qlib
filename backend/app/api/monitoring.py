from fastapi import APIRouter, Depends
from app.api.deps import get_current_active_user, get_current_developer_user
from app.models.user import User
from app.services.monitoring import monitor

router = APIRouter()

@router.get("/system-metrics")
def get_system_metrics(limit: int = 100, current_user: User = Depends(get_current_developer_user)):
    """获取系统监控指标"""
    return monitor.get_metrics(limit=limit)

@router.get("/system-metrics/current")
def get_current_system_metrics(current_user: User = Depends(get_current_developer_user)):
    """获取当前系统监控指标"""
    return monitor.get_current_metrics()

@router.get("/health")
def health_check():
    """健康检查端点"""
    from app.utils.remote_client import RemoteClient
    remote_client = RemoteClient()
    
    # 检查本地服务状态
    local_status = "healthy"
    
    # 检查远程服务器状态
    try:
        remote_healthy = remote_client.sync_health_check()
        remote_status = "healthy" if remote_healthy else "unhealthy"
    except Exception:
        remote_status = "unreachable"
    
    return {
        "status": "healthy",
        "components": {
            "local": local_status,
            "remote_server": remote_status
        },
        "timestamp": monitor.get_current_metrics()["timestamp"]
    }

@router.get("/service-status")
def get_service_status(current_user: User = Depends(get_current_developer_user)):
    """获取服务状态信息，包括DDNS模型训练服务"""
    return monitor.get_service_status()

@router.get("/service-status/details")
def get_detailed_service_status(current_user: User = Depends(get_current_developer_user)):
    """获取详细的服务状态信息，包括DDNS模型训练服务的详细状态"""
    return monitor.get_service_status()
