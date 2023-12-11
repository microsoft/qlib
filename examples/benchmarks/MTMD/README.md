# MTMD: Multi-Scale Temporal Memory Learning and Efficient Debiasing Framework for Stock Trend Forecasting
The official implementation of the paper "[MTMD: Multi-Scale Temporal Memory Learning and Efficient Debiasing Framework for Stock Trend Forecasting](https://arxiv.org/abs/2212.08656)".
![image](https://i.ibb.co/5MFPqTJ/12.png)

## Environment
1. Install python3.7, 3.8 or 3.9. 
2. Install the requirements in [requirements.txt](https://github.com/Wentao-Xu/HIST/blob/main/requirements.txt).
3. Install the quantitative investment platform [Qlib](https://github.com/microsoft/qlib) and download the data from Qlib:
	```
	# install Qlib from source
	pip install --upgrade  cython
	git clone https://github.com/microsoft/qlib.git && cd qlib
	python setup.py install

	# Download the stock features of Alpha360 from Qlib
	python scripts/get_data.py qlib_data --target_dir ~/.qlib/qlib_data/cn_data --region cn --version v2
	mkdir data
	```
4. Please download the [concept matrix](https://github.com/Wentao-Xu/HIST/tree/main/data), which is provided by [tushare](https://tushare.pro/document/2?doc_id=81).
5. Please put the concept data and stock data in the new' data' folder.

## Reproduce the stock trend forecasting results
![image](https://i.ibb.co/X7CVp2v/res.png)
```
git clone https://github.com/MingjieWang0606/MTMD-Public.git
cd MTMD-Public
mkdir output
```
### Reproduce our MTMD framework
```
# CSI 100
python learn_memory.py --model_name HIST --data_set csi100 --hidden_size 128 --num_layers 2 --outdir ./output/csi100_MTMD

# CSI 300
python learn_memory.py --model_name HIST --data_set csi300 --hidden_size 128 --num_layers 2 --outdir ./output/csi300_MTMD
```

### Reproduce our HIST framework
```
# CSI 100
python learn.py --model_name HIST --data_set csi100 --hidden_size 128 --num_layers 2 --outdir ./output/csi100_HIST

# CSI 300
python learn.py --model_name HIST --data_set csi300 --hidden_size 128 --num_layers 2 --outdir ./output/csi300_HIST
```
### Reproduce the baselines
* MLP 
```
# MLP on CSI 100
python learn.py --model_name MLP --data_set csi100 --hidden_size 512 --num_layers 3 --outdir ./output/csi100_MLP

# MLP on CSI 300
python learn.py --model_name MLP --data_set csi300 --hidden_size 512 --num_layers 3 --outdir ./output/csi300_MLP
```

* LSTM
```
# LSTM on CSI 100
python learn.py --model_name LSTM --data_set csi100 --hidden_size 128 --num_layers 2 --outdir ./output/csi100_LSTM

# LSTM on CSI 300
python learn.py --model_name LSTM --data_set csi300 --hidden_size 128 --num_layers 2 --outdir ./output/csi300_LSTM
```

* GRU
```
# GRU on CSI 100
python learn.py --model_name GRU --data_set csi100 --hidden_size 128 --num_layers 2 --outdir ./output/csi100_GRU

# GRU on CSI 300
python learn.py --model_name GRU --data_set csi300 --hidden_size 64 --num_layers 2 --outdir ./output/csi300_GRU
```

* SFM
```
# SFM on CSI 100
python learn.py --model_name SFM --data_set csi100 --hidden_size 64 --num_layers 2 --outdir ./output/csi100_SFM

# SFM on CSI 300
python learn.py --model_name SFM --data_set csi300 --hidden_size 128 --num_layers 2 --outdir ./output/csi300_SFM
```

* GATs
```
# GATs on CSI 100
python learn.py --model_name GATs --data_set csi100 --hidden_size 128 --num_layers 2 --outdir ./output/csi100_GATs

# GATs on CSI 300
python learn.py --model_name GATs --data_set csi300 --hidden_size 64 --num_layers 2 --outdir ./output/csi300_GATs
```

* ALSTM
```
# ALSTM on CSI 100
python learn.py --model_name ALSTM --data_set csi100 --hidden_size 64 --num_layers 2 --outdir ./output/csi100_ALSTM

# ALSTM on CSI 300
python learn.py --model_name ALSTM --data_set csi300 --hidden_size 128 --num_layers 2 --outdir ./output/csi300_ALSTM
```

* Transformer
```
# Transformer on CSI 100
python learn.py --model_name Transformer --data_set csi100 --hidden_size 32 --num_layers 3 --outdir ./output/csi100_Transformer

# Transformer on CSI 300
python learn.py --model_name Transformer --data_set csi300 --hidden_size 32 --num_layers 3 --outdir ./output/csi300_Transformer
```

* ALSTM+TRA 

We reproduce the ALSTM+TRA with its [source code](https://github.com/microsoft/qlib/tree/main/examples/benchmarks/TRA).

### Acknowledgements
Special thanks to ChenFeng, Zhang Mingze,Tian Junxi and LiTingXin for the their help and discussion!  
Thanks for the clean and efficient [HIST](https://github.com/Wentao-Xu/HIST) code.  


## Citation
Please cite the following paper if you use this code in your work.
```
@article{wang2022mtmd,
  title={MTMD: Multi-Scale Temporal Memory Learning and Efficient Debiasing Framework for Stock Trend Forecasting},
  author={Mingjie Wang and Mingze Zhang and Jianxiong Guo and Weijia Jia},
  journal={arXiv preprint arXiv:2212.08656},
  year={2022}
}
```

