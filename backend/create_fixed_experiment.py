from app.db.database import SessionLocal
from app.models.experiment import Experiment
from app.services.task import TaskService
from datetime import datetime

# Create a simple experiment with correct LinearModel parameters
simple_config = {
    "task": {
        "model": {
            "class": "LinearModel",
            "module_path": "qlib.contrib.model.linear",
            "kwargs": {
                "fit_intercept": True
                # LinearModel doesn't accept normalize parameter
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
                        "start_time": "2020-01-01",
                        "end_time": "2020-02-01",
                        "fit_start_time": "2020-01-01",
                        "fit_end_time": "2020-01-15",
                        "instruments": "csi300",
                        "infer_processors": [
                            {
                                "class": "RobustZScoreNorm",
                                "kwargs": {
                                    "fields_group": "feature",
                                    "clip_outlier": True
                                }
                            },
                            {
                                "class": "Fillna"
                            }
                        ],
                        "learn_processors": [
                            {
                                "class": "DropnaLabel"
                            },
                            {
                                "class": "CSRankNorm",
                                "kwargs": {
                                    "fields_group": "label"
                                }
                            }
                        ]
                    }
                },
                "segments": {
                    "train": ["2020-01-01", "2020-01-15"],
                    "test": ["2020-01-16", "2020-02-01"]
                }
            }
        }
    }
}

db = SessionLocal()

# Create a new simple experiment
simple_experiment = Experiment(
    name="Fixed Test Experiment",
    description="A fixed test experiment to verify the performance calculation fix",
    config=simple_config,
    status="pending",
    created_at=datetime.now()
)

db.add(simple_experiment)
db.commit()
experiment_id = simple_experiment.id
print(f"Created fixed experiment with ID: {experiment_id}")

# Create a task for this experiment
task = TaskService.create_task(db, experiment_id=experiment_id, task_type="train", priority=1)
print(f"Created task with ID: {task.id}")
print(f"Task status: {task.status}")

db.close()
