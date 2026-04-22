import asyncio
import json
from app.utils.remote_client import RemoteClient

async def test_training_api():
    client = RemoteClient()
    print("Testing training API...")
    
    # Test health check again to be sure
    health_status = await client.health_check()
    print(f"Health check result: {health_status}")
    
    if not health_status:
        print("Training server is not healthy. Aborting test.")
        return
    
    # Test task submission
    test_task_data = {
        "experiment_id": 1,
        "config": {
            "model": {
                "class": "LGBModel",
                "module_path": "qlib.contrib.model.gbdt",
                "kwargs": {
                    "loss": "mse",
                    "colsample_bytree": 0.8879,
                    "learning_rate": 0.0421,
                    "subsample": 0.8789,
                    "lambda_l1": 205.6999,
                    "lambda_l2": 580.9768,
                    "max_depth": 8,
                    "num_leaves": 210,
                    "num_threads": 20
                }
            },
            "dataset": {
                "class": "DatasetH",
                "module_path": "qlib.data.dataset",
                "kwargs": {
                    "handler": {
                        "class": "Alpha158",
                        "module_path": "qlib.contrib.data.handler",
                        "kwargs": {
                            "start_time": "2017-01-01",
                            "end_time": "2020-08-01",
                            "fit_start_time": "2017-01-01",
                            "fit_end_time": "2019-12-31",
                            "instruments": "csi300",
                            "freq": "day"
                        }
                    },
                    "segments": {
                        "train": ["2017-01-01", "2019-06-30"],
                        "valid": ["2019-07-01", "2019-12-31"],
                        "test": ["2020-01-01", "2020-08-01"]
                    }
                }
            }
        }
    }
    
    print("Submitting test task...")
    result = await client.submit_task(test_task_data)
    print(f"Task submission result: {result}")
    
    if result and "task_id" in result:
        task_id = result["task_id"]
        print(f"\nTask submitted successfully with ID: {task_id}")
        
        # Test task status check
        print("\nChecking task status...")
        status = await client.get_task_status(task_id)
        print(f"Task status: {status}")
        
        # Test task cancellation
        print("\nCancelling task...")
        cancel_result = await client.cancel_task(task_id)
        print(f"Task cancellation result: {cancel_result}")
    else:
        print("Task submission failed.")

if __name__ == "__main__":
    asyncio.run(test_training_api())