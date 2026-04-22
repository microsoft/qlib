import httpx
from app.config import settings
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class TrainingClient:
    """API client for communicating with the training server"""
    
    def __init__(self):
        self.base_url = settings.training_server_url
        self.timeout = settings.training_server_timeout
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=self.timeout,
            headers={"Content-Type": "application/json"}
        )
    
    async def start_training(self, experiment_id: int, config: Dict[str, Any]) -> Dict[str, Any]:
        """Start a training task on the server by creating a task in the database"""
        try:
            logger.info(f"Attempting to start training for experiment {experiment_id} on {self.base_url}")
            logger.debug(f"Training config: {config}")
            
            # In this architecture, we create a task via the experiments API
            # The train worker will pick it up from the database
            response = await self.client.post(
                f"/api/experiments/{experiment_id}/run"
            )
            
            logger.info(f"Training start response: {response.status_code}")
            logger.debug(f"Response content: {response.text}")
            
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            logger.error(f"HTTP error when starting training: {e}")
            logger.error(f"Error details: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error when starting training: {e}")
            logger.error(f"Error type: {type(e).__name__}")
            raise
    
    async def get_training_status(self, task_id: str) -> Dict[str, Any]:
        """Get the status of a training task"""
        try:
            logger.info(f"Attempting to get status for task {task_id} on {self.base_url}")
            
            # Get task status from the API
            response = await self.client.get(f"/api/tasks/{task_id}")
            
            logger.info(f"Training status response: {response.status_code}")
            logger.debug(f"Response content: {response.text}")
            
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            logger.error(f"HTTP error when getting training status: {e}")
            logger.error(f"Error details: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error when getting training status: {e}")
            logger.error(f"Error type: {type(e).__name__}")
            raise
    
    async def get_training_results(self, task_id: str) -> Dict[str, Any]:
        """Get the results of a completed training task"""
        try:
            logger.info(f"Attempting to get results for task {task_id} on {self.base_url}")
            
            # Get task results from the API
            response = await self.client.get(f"/api/tasks/{task_id}/results")
            
            logger.info(f"Training results response: {response.status_code}")
            logger.debug(f"Response content: {response.text}")
            
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            logger.error(f"HTTP error when getting training results: {e}")
            logger.error(f"Error details: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error when getting training results: {e}")
            logger.error(f"Error type: {type(e).__name__}")
            raise
    
    async def stop_training(self, task_id: str) -> Dict[str, Any]:
        """Stop a running training task"""
        try:
            logger.info(f"Attempting to stop training for task {task_id} on {self.base_url}")
            
            # Stop task via the API
            response = await self.client.post(f"/api/tasks/{task_id}/stop")
            
            logger.info(f"Training stop response: {response.status_code}")
            logger.debug(f"Response content: {response.text}")
            
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            logger.error(f"HTTP error when stopping training: {e}")
            logger.error(f"Error details: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error when stopping training: {e}")
            logger.error(f"Error type: {type(e).__name__}")
            raise
    
    async def close(self):
        """Close the HTTP client"""
        await self.client.aclose()

# Create a singleton instance
training_client = TrainingClient()
