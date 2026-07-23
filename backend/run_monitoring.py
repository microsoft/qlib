#!/usr/bin/env python3.12
"""
Run monitoring checks periodically
"""

import time
import logging
import sys
from app.services.monitoring import MonitoringService
from app.db.database import SessionLocal

# Configure detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('/home/qlib_t/backend/logs/monitoring_init.log', mode='a')
    ]
)
logger = logging.getLogger(__name__)

# Log startup information
logger.info(f"Starting run_monitoring.py with Python {sys.version}")
logger.info(f"Current working directory: {sys.path[0]}")
logger.info(f"PYTHONPATH: {':'.join(sys.path)}")

# Verify imports
try:
    from app.services.monitoring import MonitoringService
    logger.info("Successfully imported MonitoringService")
except ImportError as e:
    logger.error(f"Failed to import MonitoringService: {e}")
    sys.exit(1)

def run_monitoring():
    """Run monitoring checks"""
    logger.info("Starting monitoring check...")
    db = None
    try:
        db = SessionLocal()
        logger.info("Successfully created database session")
        
        monitoring_service = MonitoringService(db)
        logger.info("Successfully initialized MonitoringService")
        
        result = monitoring_service.run_monitoring()
        logger.info(f"Monitoring check completed successfully: {result}")
        return result
    except Exception as e:
        logger.error(f"Error during monitoring check: {e}")
        import traceback
        traceback.print_exc()
        return None
    finally:
        if db:
            try:
                db.close()
                logger.info("Successfully closed database session")
            except Exception as e:
                logger.error(f"Error closing database session: {e}")

if __name__ == "__main__":
    logger.info("Starting monitoring service main loop...")
    # Run monitoring every 5 minutes
    while True:
        run_monitoring()
        logger.info("Monitoring check completed, sleeping for 300 seconds...")
        time.sleep(300)
