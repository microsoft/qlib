# Tensorboard

Tensorboard allows you to visualize data, neural network model and to analyze how error behaves during training and validation. 

## Execution

To execute tensorboard go into `tensorboard` folder and execute the following script

```
tensorboard --logdir=runs
```

Inside the `runs` folder it should have after each model execution each folder with the model name in it, such as `mlp`, `catboost` and so on.

## Parameters

The models that current accept tensorboard visualization are XGBoost, CatBoost and Multilayer Perceptron. For any of those models it needs to be provided the following parameters and the `yml` file

```
tensorboard: true
tensorboard_name: <<tensor board name>>
```
### Examples of `yml` file with tensorboard configuration

The following files contains examples of how to configure tensorboard:

- `workflow_config_mlp_Alpha360.yml`
- `workflow_config_catboost_Alpha360.yml`
- `workflow_config_xgboost_Alpha360.yml`
