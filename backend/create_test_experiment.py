from app.db.database import SessionLocal
from app.models.experiment import Experiment
from datetime import datetime

# Create a simple test experiment configuration
test_config = {
    "task": {
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
                        ],
                        "label": [
                            {
                                "class": "RollingMeanNorm",
                                "kwargs": {
                                    "fields": ["Ref($close, -2)/Ref($close, -1) - 1"],
                                    "windows": 20
                                }
                            }
                        ]
                    }
                },
                "segments": {
                    "train": ["2017-01-01", "2018-12-31"],
                    "valid": ["2019-01-01", "2019-12-31"],
                    "test": ["2020-01-01", "2020-08-01"]
                }
            }
        }
    }
}

db = SessionLocal()

# Create a new test experiment
test_experiment = Experiment(
    name="Test Experiment for Performance Fix",
    description="A test experiment to verify the performance calculation fix",
    config=test_config,
    status="pending",
    created_at=datetime.now()
)

db.add(test_experiment)
db.commit()
print(f"Created test experiment with ID: {test_experiment.id}")
print(f"Experiment status: {test_experiment.status}")

db.close()
