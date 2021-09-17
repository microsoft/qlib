# Temporally Correlated Task Scheduling for Sequence Learning
We provide the [code](https://github.com/microsoft/qlib/blob/main/qlib/contrib/model/pytorch_tcts.py) for reproducing the stock trend forecasting experiments.

### Background
Sequence learning has attracted much research attention from the machine learning community in recent years. In many applications, a sequence learning task is usually associated with multiple temporally correlated auxiliary tasks, which are different in terms of how much input information to use or which future step to predict. In stock trend forecasting, as demonstrated in Figure1, one can predict the price of a stock in different future days (e.g., tomorrow, the day after tomorrow). In this paper, we propose a framework to make use of those temporally correlated tasks to help each other. 

### Method
Given that there are usually multiple temporally correlated tasks, the key challenge lies in which tasks to use and when to use them in the training process. In this work, we introduce a learnable task scheduler for sequence learning, which adaptively selects temporally correlated tasks during the training process. The scheduler accesses the model status and the current training data (e.g., in current minibatch), and selects the best auxiliary task to help the training of the main task. The scheduler and the model for the main task are jointly trained through bi-level optimization: the scheduler is trained to maximize the validation performance of the model, and the model is trained to minimize the training loss guided by the scheduler. The process is demonstrated in Figure2.

<p align="center"> 
<img src="workflow.png"/>
</p>

At step <img src="https://render.githubusercontent.com/render/math?math=s">, with training data <img src="https://render.githubusercontent.com/render/math?math=x_s,y_s">, the scheduler <img src="https://render.githubusercontent.com/render/math?math=\varphi"> chooses a suitable task <img src="https://render.githubusercontent.com/render/math?math=T_{i_s}"> (green solid lines) to update the model <img src="https://render.githubusercontent.com/render/math?math=f"> (blue solid lines). After <img src="https://render.githubusercontent.com/render/math?math=S"> steps, we evaluate the model <img src="https://render.githubusercontent.com/render/math?math=f"> on the validation set and update the scheduler <img src="https://render.githubusercontent.com/render/math?math=\varphi"> (green dashed lines).

### DataSet
* We use the historical transaction data for 300 stocks on [CSI300](http://www.csindex.com.cn/en/indices/index-detail/000300) from 01/01/2008 to 08/01/2020. 
* We split the data into training (01/01/2008-12/31/2013), validation (01/01/2014-12/31/2015), and test sets (01/01/2016-08/01/2020) based on the transaction time. 

### Experiments
#### Task Description
* The main tasks <img src="https://render.githubusercontent.com/render/math?math=T_k"> refers to forecasting return of stock <img src="https://render.githubusercontent.com/render/math?math=i"> as following,
<div align=center>
<img src="https://render.githubusercontent.com/render/math?math=r_{i}^k = \frac{\price_i^{t+k}}{\price_i^{t}} - 1">
</div>

* Temporally correlated task sets <img src="https://render.githubusercontent.com/render/math?math=\mathcal{T}_k = \{T_1, T_2, ... , T_k\}">, in this paper, <img src="https://render.githubusercontent.com/render/math?math=\mathcal{T}_3">, <img src="https://render.githubusercontent.com/render/math?math=\mathcal{T}_5"> and <img src="https://render.githubusercontent.com/render/math?math=\mathcal{T}_10"> are used.
#### Baselines
* GRU/MLP/LightGBM (LGB)
* Multi-task learning (MTL): In multi-task learning, multiple tasks are jointly trained and mutually boosted. Each task is treated equally, while in our setting, we focus on the main task.
* Curriculum transfer learning (CL): Transfer learning also leverages auxiliary tasks to boost the main task. [Curriculum transfer learning](https://arxiv.org/pdf/1804.00810.pdf) is one kind of transfer learning which schedules auxiliary tasks according to certain rules. Our problem can also be regarded as a special kind of transfer learning, where the auxiliary tasks are temporally correlated with the main task. Our learning process is dynamically controlled by a scheduler rather than some pre-defined rules. In the CL baseline, we start from the task <img src="https://render.githubusercontent.com/render/math?math=T_1" >, then <img src="https://render.githubusercontent.com/render/math?math=T_2" >, and gradually move to the last one.
#### Result
|<img src="https://render.githubusercontent.com/render/math?math=T_1" >     |IC             |ICIR           |Rank IC        |Rank ICIR      |mse           |
|----------|---------------|---------------|---------------|---------------|--------------|
|GRU       |0.0431±0.00988 |0.3247±0.08182 |0.0293±0.01606 |0.2242±0.12596 |0.9925±0.00000|
|MLP       |0.0163±0.00229 |0.1275±0.01836 |0.0040±0.00453 |0.0311±0.03507 |1.0275±0.00000|
|LGB       |-0.0069±0.00000|-0.0497±0.00000|-0.0397±0.00000|-0.2667±0.00000|0.9858±0.00000|
|MTL(<img src="https://render.githubusercontent.com/render/math?math=\mathcal{T}_3">)  |0.0514±0.00000 |0.3877±0.00000 |0.0343±0.00000 |0.2564±0.00000 |0.9761±0.00000|
|CL(<img src="https://render.githubusercontent.com/render/math?math=\mathcal{T}_3">)   |0.0364±0.00000 |0.2659±0.00000 |0.0220±0.00000 |0.1568±0.00000 |1.0067±0.00000|
|Ours(<img src="https://render.githubusercontent.com/render/math?math=\mathcal{T}_3">) |0.0496±0.00000 |0.3959±0.00000 |0.0488±0.00000 |0.3930±0.00000 |0.9587±0.00000|
|MTL(<img src="https://render.githubusercontent.com/render/math?math=\mathcal{T}_5">)  |0.0363±0.00000 |0.2620±0.00000 |0.0218±0.00000 |0.1488±0.00000 |1.0070±0.00000|
|CL(<img src="https://render.githubusercontent.com/render/math?math=\mathcal{T}_5">)   |0.0361±0.00000 |0.2454±0.00000 |0.0134±0.00000 |0.0873±0.00000 |0.9971±0.00000|
|Ours(<img src="https://render.githubusercontent.com/render/math?math=\mathcal{T}_5">) |0.0459±0.00000 |0.3321±0.00000 |0.0350±0.00000 |0.2434±0.00000 |0.9896±0.00000|
|MTL(<img src="https://render.githubusercontent.com/render/math?math=\mathcal{T}_{10}">) |0.0289±0.00000 |0.1883±0.00000 |0.0049±0.00000 |0.0308±0.00000 |1.0129±0.00000|
|CL(<img src="https://render.githubusercontent.com/render/math?math=\mathcal{T}_{10}">)  |0.0298±0.00000 |0.2019±0.00000 |0.0082±0.00000 |0.0524±0.00000 |1.0185±0.00000|
|Ours(<img src="https://render.githubusercontent.com/render/math?math=\mathcal{T}_{10}">)|0.0495±0.00730 |0.3841±0.06384 |0.0395±0.00995 |0.3129±0.08926 |0.9724±0.01227|

|<img src="https://render.githubusercontent.com/render/math?math=T_2">     |IC             |ICIR           |Rank IC        |Rank ICIR      |mse           |
|----------|---------------|---------------|---------------|---------------|--------------|
|GRU       |0.0520±0.00653 |0.3892±0.04805 |0.0426±0.00645 |0.3222±0.04896 |0.9694±0.00000|
|MLP       |0.0299±0.00459 |0.2349±0.04098 |0.0204±0.00817 |0.1593±0.06554 |1.0109±0.00000|
|LGB       |0.0066±0.00000 |0.0509±0.00000 |-0.0065±0.00000|-0.0451±0.00000|1.0663±0.00000|
|MTL(<img src="https://render.githubusercontent.com/render/math?math=\mathcal{T}_3">)  |0.0514±0.00000 |0.3877±0.00000 |0.0343±0.00000 |0.2564±0.00000 |0.9770±0.00000|
|CL(<img src="https://render.githubusercontent.com/render/math?math=\mathcal{T}_3">)   |0.0517±0.00000 |0.3863±0.00000 |0.0449±0.00539 |0.3636±0.04696 |0.9685±0.01224|
|Ours(<img src="https://render.githubusercontent.com/render/math?math=\mathcal{T}_3">) |0.0603±0.00648 |0.4667±0.03682 |0.0601±0.00000 |0.4589±0.00000 |0.9527±0.00000|
|MTL(<img src="https://render.githubusercontent.com/render/math?math=\mathcal{T}_5">)  |0.0452±0.00000 |0.3234±0.00000 |0.0266±0.00000 |0.1857±0.00000 |0.9888±0.00000|
|CL(<img src="https://render.githubusercontent.com/render/math?math=\mathcal{T}_5">)   |0.0361±0.00000 |0.2454±0.00000 |0.0134±0.00000 |0.0873±0.00000 |0.9986±0.00000|
|Ours(<img src="https://render.githubusercontent.com/render/math?math=\mathcal{T}_5">) |0.0533±0.00561 |0.4206±0.04413 |0.0489±0.00678 |0.3908±0.05173 |0.9708±0.00955|
|MTL(<img src="https://render.githubusercontent.com/render/math?math=\mathcal{T}_{10}">) |0.0522±0.00000 |0.3944±0.00000 |0.0430±0.00000 |0.3266±0.00000 |0.9842±0.00000|
|CL(<img src="https://render.githubusercontent.com/render/math?math=\mathcal{T}_{10}">)  |0.0430±0.00000 |0.3097±0.00000 |0.0299±0.00000 |0.2079±0.00000 |0.9951±0.00000|
|Ours(<img src="https://render.githubusercontent.com/render/math?math=\mathcal{T}_{10}">)|0.0534±0.00167 |0.4188±0.02562 |0.0373±0.00668 |0.3053±0.07595 |0.9779±0.00362|

|<img src="https://render.githubusercontent.com/render/math?math=T_3">     |IC             |ICIR           |Rank IC        |Rank ICIR      |mse           |
|----------|---------------|---------------|---------------|---------------|--------------|
|GRU       |0.0516±0.00367 |0.3803±0.02405 |0.0365±0.00394 |0.2686±0.02117 |0.9772±0.00000|
|MLP       |0.0312±0.00190 |0.2508±0.01817 |0.0237±0.00319 |0.1872±0.02822 |1.0256±0.00000|
|LGB       |0.0220±0.00000 |0.1698±0.00000 |0.0090±0.00000 |0.0616±0.00000 |1.0323±0.00000|
|MTL(<img src="https://render.githubusercontent.com/render/math?math=\mathcal{T}_3">)  |0.0617±0.00000 |0.4599±0.00000 |0.0411±0.00000 |0.3121±0.00000 |0.9573±0.00000|
|CL(<img src="https://render.githubusercontent.com/render/math?math=\mathcal{T}_3">)   |0.0504±0.00000 |0.3624±0.00000 |0.0249±0.00000 |0.1761±0.00000 |0.9782±0.00000|
|Ours(<img src="https://render.githubusercontent.com/render/math?math=\mathcal{T}_3">) |0.0623±0.00438 |0.4854±0.01557 |0.0474±0.00385 |0.3885±0.02639 |0.9696±0.01283|
|MTL(<img src="https://render.githubusercontent.com/render/math?math=\mathcal{T}_5">)  |0.0539±0.00000 |0.3876±0.00000 |0.0399±0.00000 |0.2913±0.00000 |0.9657±0.00000|
|CL(<img src="https://render.githubusercontent.com/render/math?math=\mathcal{T}_5">)   |0.0361±0.00000 |0.2454±0.00000 |0.0134±0.00000 |0.0873±0.00000 |1.0007±0.00000|
|Ours(<img src="https://render.githubusercontent.com/render/math?math=\mathcal{T}_5">) |0.0521±0.00278 |0.3973±0.03210 |0.0466±0.00483 |0.3584±0.05129 |0.9796±0.00375|
|MTL(<img src="https://render.githubusercontent.com/render/math?math=\mathcal{T}_{10}">) |0.0522±0.00000 |0.3944±0.00000 |0.0430±0.00000 |0.3266±0.00000 |0.9869±0.00000|
|CL(<img src="https://render.githubusercontent.com/render/math?math=\mathcal{T}_{10}">)  |0.0430±0.00000 |0.3097±0.00000 |0.0299±0.00000 |0.2079±0.00000 |0.9967±0.00000|
|Ours(<img src="https://render.githubusercontent.com/render/math?math=\mathcal{T}_{10}">)|0.0562±0.00390 |0.4439±0.03595 |0.0438±0.00465 |0.3666±0.05158 |0.9726±0.00646|