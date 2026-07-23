from app.db.database import SessionLocal
from app.models.experiment import Experiment

db = SessionLocal()
# Check experiment 2 which has performance data
experiment = db.query(Experiment).filter(Experiment.id == 2).first()
if experiment and experiment.performance:
    print(f'Experiment {experiment.id} Performance Data:')
    print(f'Keys: {list(experiment.performance.keys())}')
    print(f'Has cumulative_returns: {"cumulative_returns" in experiment.performance}')
    if "cumulative_returns" in experiment.performance:
        cum_returns = experiment.performance["cumulative_returns"]
        print(f'Cumulative returns type: {type(cum_returns)}')
        print(f'Cumulative returns sample: {dict(list(cum_returns.items())[:5])}')
        print(f'Number of data points: {len(cum_returns)}')
    print(f'Has total_return: {"total_return" in experiment.performance}')
    if "total_return" in experiment.performance:
        print(f'Total return: {experiment.performance["total_return"]}')
db.close()
