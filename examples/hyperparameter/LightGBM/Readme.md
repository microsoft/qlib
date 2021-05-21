# LightGBM hyperparameter

## Alpha158
First terminal
```
optuna create-study --study LGBM_158 --storage sqlite:///db.sqlite3
optuna-dashboard --port 5000 --host 0.0.0.0 sqlite:///db.sqlite3
```
Second terminal
```
python hyperparameter_158.py
```

## Alpha360
First terminal
```
optuna create-study --study LGBM_360 --storage sqlite:///db.sqlite3
optuna-dashboard --port 5000 --host 0.0.0.0 sqlite:///db.sqlite3
```
Second terminal
```
python hyperparameter_360.py
```
