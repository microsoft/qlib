

# Introduction

What is GeneralPtNN
- Fix previous design that fail to support both Time-series and tabular data
- Now you can just replace the Pytorch model structure to run a NN model.

We provide an example to demonstrate the effectiveness of the current design.
- `workflow_config_gru.yaml` align with previous results [GRU(Kyunghyun Cho, et al.)](../README.md#Alpha158-dataset)
  - `workflow_config_gru2mlp.yaml` to demonstrate we can convert config from time-series to tabular data with minimal changes
    - You only have to change the net & dataset class to make the conversion.
- `workflow_config_mlp.yaml` achieved similar functionality with [MLP](../README.md#Alpha158-dataset)

# TODO

- We will align existing models to current design.

- The result of `workflow_config_mlp.yaml` is different with the result of [MLP](../README.md#Alpha158-dataset) since GeneralPtNN has a different stopping method compared to previous implementations. Specificly, GeneralPtNN controls training according to epoches, whereas previous methods controlled by max_steps. 
