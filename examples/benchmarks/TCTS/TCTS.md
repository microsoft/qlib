# Temporally Correlated Task Scheduling for Sequence Learning
We provide the code for reproducing the stock trend forecasting experiments in [Temporally Correlated Task Scheduling for Sequence Learning](https://www.overleaf.com/project/5eb8efb42dcf710001d781d6).

### Background
Sequence learning has attracted much research attention from the machine learning community in recent years. In many applications, a sequence learning task is usually associated with multiple temporally correlated auxiliary tasks, which are different in terms of how much input information to use or which future step to predict. In stock trend forecasting, as demonstrated in Figure1, one can predict the price of a stock in different future days (e.g., tomorrow, the day after tomorrow). In this paper, we propose a framework to make use of those temporally correlated tasks to help each other. 

![Temporally Correlated Tasks.](task_description.png)


### Method
Given that there are usually multiple temporally correlated tasks, the key challenge lies in which tasks to use and when to use them in the training process. In this work, we introduce a learnable task scheduler for sequence learning, which adaptively selects temporally correlated tasks during the training process. The scheduler accesses the model status and the current training data (e.g., in current minibatch), and selects the best auxiliary task to help the training of the main task. The scheduler and the model for the main task are jointly trained through bi-level optimization: the scheduler is trained to maximize the validation performance of the model, and the model is trained to minimize the training loss guided by the scheduler. The process is demonstrated in Figure2.


![The optimization workflow of one episode.](workflow.png)


At step $s$, with training data $(x_s,y_s)$, the scheduler $\varphi(\cdots;\omega)$ chooses a suitable task $T_{i_s}$ (green solid lines) to update the model $f(\cdots;\theta)$ (blue solid lines). After $S$ steps, we evaluate the model $f$ on the validation set $\Ddev$ and update the scheduler $\varphi$ (green dashed lines).

### DataSet
* We use the historical transaction data for 300 stocks on [CSI300](http://www.csindex.com.cn/en/indices/index-detail/000300) from 01/01/2008 to 08/01/2020. 
* We split the data into training (01/01/2008-12/31/2013), validation (01/01/2014-12/31/2015), and test sets (01/01/2016-08/01/2020) based on the transaction time. 

### Experiments
#### Task Description
* The main tasks $T_k$ ($task_k$ in Figure1) refers to forecasting return of stock $i$ as following,

```math
$
r_{i}^k = \frac{\price_i^{t+k}}{\price_i^{t+k-1}} - 1
$
```
* Temporally correlated task sets $\domT_k = {T_1, T_2, ... , T_k}$, in this paper, $\domT_3$, $\domT_5$ and $\domT_10$ are used.
#### Baselines
* GRU/MLP/LightGBM (LGB)/Graph Attention Networks (GAT)
* Multi-task learning (MTL): In multi-task learning, multiple tasks are jointly trained and mutually boosted. Each task is treated equally, while in our setting, we focus on the main task.
* Curriculum transfer learning (CL): Transfer learning also leverages auxiliary tasks to boost the main task. [Curriculum transfer learning](https://arxiv.org/pdf/1804.00810.pdf) is one kind of transfer learning which schedules auxiliary tasks according to certain rules. Our problem can also be regarded as a special kind of transfer learning, where the auxiliary tasks are temporally correlated with the main task. Our learning process is dynamically controlled by a scheduler rather than some pre-defined rules. In the CL baseline, we start from the task T_1, then T_2, and gradually move to the last one.
#### Result
| Methods | $T_1$ | $T_2$ | $T_3$ |
| :----: | :----: | :----: | :----: |
| GRU | 0.049 / 1.903 | 0.018 / 1.972 | 0.014 / 1.989 |
| MLP | 0.023 / 1.961 | 0.022 / 1.962 | 0.015 / 1.978 |
| LGB | 0.038 / 1.883 | 0.023 / 1.952 | 0.007 / 1.987 |
| GAT | 0.052 / 1.898 | 0.024 / 1.954 | 0.015 / 1.973 |
| MTL($\domT_3$)  | 0.061 / 1.862  | 0.023 / 1.942  | 0.012 / 1.956 |
| CL($\domT_3$)  | 0.051 / 1.880  | 0.028 / 1.941  | 0.016 / 1.962 |
| Ours($\domT_3$)  | 0.071 / 1.851  | 0.030 / 1.939  | 0.017 / 1.963 |
| MTL($\domT_5$)  | 0.057 / 1.875  | 0.021 / 1.939  | 0.017 / 1.959 |
| CL($\domT_5$)  | 0.056 / 1.877  | 0.028 / 1.942  | 0.015 / 1.962 |
| Ours($\domT_5$)  | 0.075 / 1.849  | 0.032 /1.939  | 0.021 / 1.955  | 
| MTL($\domT_{10}$)  | 0.052 / 1.882  | 0.020 / 1.947  | 0.019 / 1.952 |
| CL($\domT_{10}$)  | 0.051 / 1.882  | 0.028 / 1.950  | 0.016 / 1.961 |
| Ours($\domT_{10}$)  | 0.067 /  1.867  | 0.030 / 1.960  | 0.022 / 1.942|