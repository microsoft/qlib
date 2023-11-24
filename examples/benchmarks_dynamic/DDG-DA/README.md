# Introduction
This is the implementation of `DDG-DA` based on `Meta Controller` component provided by `Qlib`.

Please refer to the paper for more details: *DDG-DA: Data Distribution Generation for Predictable Concept Drift Adaptation* [[arXiv](https://arxiv.org/abs/2201.04038)]


# Background
In many real-world scenarios, we often deal with streaming data that is sequentially collected over time. Due to the non-stationary nature of the environment, the streaming data distribution may change in unpredictable ways, which is known as concept drift. To handle concept drift, previous methods first detect when/where the concept drift happens and then adapt models to fit the distribution of the latest data. However, there are still many cases that some underlying factors of environment evolution are predictable, making it possible to model the future concept drift trend of the streaming data, while such cases are not fully explored in previous work.

Therefore, we propose a novel method `DDG-DA`, that can effectively forecast the evolution of data distribution and improve the performance of models. Specifically, we first train a predictor to estimate the future data distribution, then leverage it to generate training samples, and finally train models on the generated data.

# Dataset
The data in the paper are private. So we conduct experiments on Qlib's public dataset.
Though the dataset is different, the conclusion remains the same. By applying `DDG-DA`, users can see rising trends at the test phase both in the proxy models' ICs and the performances of the forecasting models.

# Run the Code
Users can try `DDG-DA` by running the following command:
```bash
    python workflow.py run
```

The default forecasting models are `Linear`. Users can choose other forecasting models by changing the `forecast_model` parameter when `DDG-DA` initializes. For example, users can try `LightGBM` forecasting models by running the following command:
```bash
    python workflow.py --conf_path=../workflow_config_lightgbm_Alpha158.yaml run
```

# Results
The results of related methods in Qlib's public dataset can be found [here](../)

# Requirements
Here are the minimal hardware requirements to run the ``workflow.py`` of DDG-DA.
* Memory: 45G
* Disk: 4G

Pytorch with CPU & RAM will be enough for this example.
