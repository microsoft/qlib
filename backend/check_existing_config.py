from app.db.database import SessionLocal
from app.models.experiment import Experiment

db = SessionLocal()
# Get experiment 2 which has performance data
experiment = db.query(Experiment).filter(Experiment.id == 2).first()
if experiment:
    print(f'Experiment {experiment.id} Configuration:')
    print(f'Config keys: {list(experiment.config.keys())}')
    if 'task' in experiment.config:
        task_config = experiment.config['task']
        print(f'Task keys: {list(task_config.keys())}')
        if 'dataset' in task_config:
            dataset_config = task_config['dataset']
            print(f'Dataset config: {dataset_config}')
db.close()
