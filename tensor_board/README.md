# Tensorboard

Tensorboard allows you to visualize data, neural network model and to analyze how error behaves during training and validation. 

## Execution

To execute tensorboard go into `tensorboard` folder and execute the following script

```
tensorboard --logdir=runs
```

## Parameters

The models that current accept tensorboard visualization are XGBoost, CatBoost and Multilayer Perceptron. For any of those models it needs to be provided the following parameters and the `yml` file

```
tensorboard: true
tensorboard_name: <<tensor board name>>
```
### Examples of `yml` file with tensorboard configuration

The following files contains examples of how to configure tensorboard:

- `workflow_config_mlp_Alpha360.yml`
- `workflow_config_mlp_Alpha360.yml`
- `workflow_config_mlp_Alpha360.yml`
