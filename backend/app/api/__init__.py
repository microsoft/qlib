from fastapi import APIRouter
from app.api import auth, experiments, models, configs, benchmarks, factors, data, train, monitoring, tasks

# Create main API router
api_router = APIRouter()

# Include sub-routers
api_router.include_router(auth.router, prefix="/auth", tags=["auth"])
api_router.include_router(experiments.router, prefix="/experiments", tags=["experiments"])
api_router.include_router(models.router, prefix="/models", tags=["models"])
api_router.include_router(configs.router, prefix="/configs", tags=["configs"])
api_router.include_router(benchmarks.router, prefix="/benchmarks", tags=["benchmarks"])
api_router.include_router(factors.router, prefix="/factors", tags=["factors"])
api_router.include_router(data.router, prefix="/data", tags=["data"])
api_router.include_router(train.router, prefix="/train", tags=["train"])
api_router.include_router(monitoring.router, prefix="/monitoring", tags=["monitoring"])
api_router.include_router(tasks.router, prefix="/tasks", tags=["tasks"])


