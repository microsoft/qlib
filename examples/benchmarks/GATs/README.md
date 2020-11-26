# GATs
* Graph Attention Networks(GATs) leverage masked self-attentional layers on graph-structured data. The nodes in stacked layers have different weights and they are able to attend over their
neighborhoods’ features, without requiring any kind of costly matrix operation (such as inversion) or depending on knowing the graph structure upfront.
* This code used in Qlib is implemented with PyTorch by ourselves.
* Paper: Graph Attention Networks https://arxiv.org/pdf/1710.10903.pdf