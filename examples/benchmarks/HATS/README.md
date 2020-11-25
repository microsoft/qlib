##Requirement

* pandas==1.1.2
* numpy==1.17.4
* scikit_learn==0.23.2
* torch==1.7.0

##HATS

* HATS is a a hierarchical attention network for stock prediction which uses relational data for stock market prediction. HATS selectively aggregates information
on different relation types and adds the information to the representations of each company. HATS is used as a relational modeling module with initialized node representations.Furthermore, HATS
can  predict not only individual stock prices but also market index movements, which is similar to the graph classification task.

* HATS uses pretrained model of GRU and LSTM. The code of GRU and LSTM used in Qlib is a pyTorch implemention of GRU and LSTM.
* Paper address:HATS: A Hierarchical Graph Attention Network for Stock Movement Prediction https://arxiv.org/pdf/1908.07999.pdf