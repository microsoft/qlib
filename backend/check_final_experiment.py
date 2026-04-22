from app.db.database import SessionLocal
from app.models.experiment import Experiment

db = SessionLocal()
# Check experiment 10 which just completed
experiment = db.query(Experiment).filter(Experiment.id == 10).first()
if experiment:
    print(f'Experiment {experiment.id} Status: {experiment.status}')
    print(f'Has performance: {bool(experiment.performance)}')
    if experiment.performance:
        print(f'Performance keys: {list(experiment.performance.keys())}')
        print(f'Has cumulative_returns: {"cumulative_returns" in experiment.performance}')
        if "cumulative_returns" in experiment.performance:
            cum_returns = experiment.performance["cumulative_returns"]
            print(f'Cumulative returns type: {type(cum_returns)}')
            print(f'Number of data points: {len(cum_returns)}')
            if len(cum_returns) > 0:
                print(f'Cumulative returns sample: {dict(list(cum_returns.items())[:5])}')
        print(f'Total return: {experiment.performance.get("total_return", "N/A")}')
        print(f'Annual return: {experiment.performance.get("annual_return", "N/A")}')
        print(f'Sharpe ratio: {experiment.performance.get("sharpe_ratio", "N/A")}')
    print(f'Error: {experiment.error}')
db.close()
