from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from sqlalchemy.exc import SQLAlchemyError
import logging
import psutil
import os
from datetime import datetime, timedelta

from app.db.database import engine, Base

# Create all database tables
Base.metadata.create_all(bind=engine)

# WebSocket manager for real-time log streaming
class ConnectionManager:
    def __init__(self):
        # key: task_id, value: list of WebSocket connections
        self.active_connections: dict[str, list[WebSocket]] = {}
    
    async def connect(self, websocket: WebSocket, task_id: str):
        await websocket.accept()
        if task_id not in self.active_connections:
            self.active_connections[task_id] = []
        self.active_connections[task_id].append(websocket)
    
    def disconnect(self, websocket: WebSocket, task_id: str):
        if task_id in self.active_connections:
            self.active_connections[task_id].remove(websocket)
            if not self.active_connections[task_id]:
                del self.active_connections[task_id]
    
    async def send_update(self, message: dict, task_id: str):
        if task_id in self.active_connections:
            for connection in self.active_connections[task_id]:
                await connection.send_json(message)
    
    async def send_log(self, log: str, task_id: str):
        if task_id in self.active_connections:
            for connection in self.active_connections[task_id]:
                await connection.send_text(log)

# Create connection manager instance
manager = ConnectionManager()

# Configure logging with detailed format
log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
os.makedirs(log_dir, exist_ok=True)  # Ensure log directory exists
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(log_dir, 'uvicorn.log'), mode='a')  # Use absolute path for log file
    ]
)
logger = logging.getLogger(__name__)

# Create FastAPI application
app = FastAPI(
    title="QLib AI API",
    description="API for managing QLib experiments and models",
    version="1.0.0",
)

# Configure CORS from environment variable if available, otherwise use production origins
from app.db.database import settings

# Get allowed origins from settings or use default production origins
allow_origins = getattr(settings, "cors_origins", [
    "http://localhost:3001",     # Frontend dev server
    "http://localhost:3000",     # Frontend alt dev server
    "http://localhost:8000",     # Backend dev server
    "http://127.0.0.1:3001",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:8000",
    "http://116.62.59.244",      # Production IP
    "http://qlib.hoo.ink",       # Production domain
    "http://ddns.hoo.ink:8000"   # DDNS server for training
])

# Ensure allow_origins is a list
if isinstance(allow_origins, str):
    allow_origins = [o.strip() for o in allow_origins.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,  # Only allow production origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add custom exception handlers
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Custom validation exception handler with detailed debugging information"""
    # Log detailed validation error with request context
    logger.error(
        f"Validation error for {request.method} {request.url}: {exc}",
        extra={
            "request_method": request.method,
            "request_url": str(request.url),
            "client_ip": request.client.host if request.client else "unknown",
            "errors": exc.errors(),
            "body": await request.body() if hasattr(request, 'body') else "N/A"
        }
    )
    return JSONResponse(
        status_code=422,
        content={
            "detail": "Validation error",
            "errors": exc.errors(),
            "message": "Invalid input data",
            "request_info": {
                "method": request.method,
                "url": str(request.url),
                "timestamp": datetime.utcnow().isoformat()
            }
        }
    )

@app.exception_handler(SQLAlchemyError)
async def sqlalchemy_exception_handler(request: Request, exc: SQLAlchemyError):
    """Custom SQLAlchemy exception handler with detailed debugging information"""
    # Log detailed database error with request context
    logger.error(
        f"Database error for {request.method} {request.url}: {exc}",
        extra={
            "request_method": request.method,
            "request_url": str(request.url),
            "client_ip": request.client.host if request.client else "unknown",
            "error_type": type(exc).__name__
        }
    )
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Database error",
            "message": f"An error occurred while accessing the database: {type(exc).__name__}",
            "request_info": {
                "method": request.method,
                "url": str(request.url),
                "timestamp": datetime.utcnow().isoformat()
            }
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Custom general exception handler with detailed debugging information"""
    # Log detailed general error with request context
    logger.error(
        f"General error for {request.method} {request.url}: {exc}",
        exc_info=True,  # Include traceback in logs
        extra={
            "request_method": request.method,
            "request_url": str(request.url),
            "client_ip": request.client.host if request.client else "unknown",
            "error_type": type(exc).__name__
        }
    )
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error",
            "message": f"An unexpected error occurred: {type(exc).__name__}",
            "request_info": {
                "method": request.method,
                "url": str(request.url),
                "timestamp": datetime.utcnow().isoformat()
            }
        }
    )

# Add request logging middleware
@app.middleware("http")
async def log_requests(request, call_next):
    """Middleware to log all incoming requests"""
    import time
    
    start_time = time.time()
    client_ip = request.client.host if request.client else "unknown"
    
    logger.info(f"Request: {request.method} {request.url} from {client_ip}")
    
    response = await call_next(request)
    
    process_time = time.time() - start_time
    logger.info(f"Response: {request.method} {request.url} - Status: {response.status_code} - Time: {process_time:.4f}s")
    
    return response

# Add performance monitoring middleware - optimized version
@app.middleware("http")
async def monitor_performance(request, call_next):
    """Middleware to monitor request performance"""
    import time
    
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    
    # Add X-Process-Time header to response
    response.headers["X-Process-Time"] = f"{process_time:.4f}"
    
    # Log slow requests
    if process_time > 1.0:  # Log requests taking more than 1 second
        client_ip = request.client.host if request.client else "unknown"
        logger.warning(f"Slow Request: {request.method} {request.url} from {client_ip} - Time: {process_time:.4f}s")
    
    return response

# Include API router without training endpoints
from app.api import auth, experiments, models, configs, benchmarks, factors, data, monitoring, train, tasks

# Create main API router without training
main_api_router = APIRouter()

# Include sub-routers except training
main_api_router.include_router(auth.router, prefix="/auth", tags=["auth"])
main_api_router.include_router(experiments.router, prefix="/experiments", tags=["experiments"])
main_api_router.include_router(models.router, prefix="/models", tags=["models"])
main_api_router.include_router(configs.router, prefix="/configs", tags=["configs"])
main_api_router.include_router(benchmarks.router, prefix="/benchmarks", tags=["benchmarks"])
main_api_router.include_router(factors.router, prefix="/factors", tags=["factors"])
main_api_router.include_router(data.router, prefix="/data", tags=["data"])
main_api_router.include_router(monitoring.router, prefix="/monitoring", tags=["monitoring"])
main_api_router.include_router(train.router, prefix="/train", tags=["train"])
main_api_router.include_router(tasks.router, prefix="/tasks", tags=["tasks"])

# Include main API router
app.include_router(main_api_router, prefix="/api")

# Root endpoint
@app.get("/")
def root():
    return {"message": "Welcome to QLib AI API"}

# Health check endpoint
@app.get("/health")
def health_check():
    """Health check endpoint for monitoring"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "QLib AI API",
        "version": "1.0.0"
    }



# System status endpoint with detailed monitoring information
@app.get("/status")
def system_status():
    """System status endpoint with detailed monitoring information"""
    # Get system resource usage using psutil
    cpu_percent = psutil.cpu_percent(interval=0.1)
    mem = psutil.virtual_memory()
    disk = psutil.disk_usage('/')
    net_io = psutil.net_io_counters()
    process = psutil.Process(os.getpid())
    
    # Get active connections count from WebSocket manager
    active_connections = sum(len(conn_list) for conn_list in manager.active_connections.values())
    active_tasks = len(manager.active_connections)
    
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "QLib AI API",
        "version": "1.0.0",
        "system": {
            "process_id": os.getpid(),
            "cpu_percent": cpu_percent,
            "memory": {
                "total": mem.total,
                "available": mem.available,
                "used": mem.used,
                "percent": mem.percent
            },
            "disk": {
                "total": disk.total,
                "available": disk.free,
                "used": disk.used,
                "percent": disk.percent
            },
            "network": {
                "bytes_sent": net_io.bytes_sent,
                "bytes_recv": net_io.bytes_recv,
                "packets_sent": net_io.packets_sent,
                "packets_recv": net_io.packets_recv
            },
            "process": {
                "cpu_percent": process.cpu_percent(interval=0.1),
                "memory_percent": process.memory_percent(),
                "threads": process.num_threads(),
                "open_files": len(process.open_files())
            }
        },
        "websocket": {
            "active_connections": active_connections,
            "active_tasks": active_tasks
        },
        "api_docs": {
            "swagger_ui": "/docs",
            "redoc": "/redoc",
            "openapi_json": "/openapi.json"
        }
    }



# Export WebSocket manager for use in other modules
global_websocket_manager = manager
