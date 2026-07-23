from app.db.database import SessionLocal
from app.models.experiment import Experiment

db = SessionLocal()
# Get experiment 2 which has performance data
experiment = db.query(Experiment).filter(Experiment.id == 2).first()
if experiment:
    print(f'Experiment {experiment.id} Logs:')
    print(f'First 1000 characters:')
    print(experiment.logs[:1000])
    print(f'\nLast 1000 characters:')
    print(experiment.logs[-1000:])
db.close()
