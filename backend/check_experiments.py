from app.db.database import SessionLocal
from app.models.experiment import Experiment

db = SessionLocal()
experiments = db.query(Experiment).all()
print(f'Total experiments: {len(experiments)}')
for exp in experiments:
    print(f'ID: {exp.id}, Status: {exp.status}, Has performance: {bool(exp.performance)}')
    if exp.performance:
        print(f'  Performance keys: {list(exp.performance.keys())}')
db.close()
