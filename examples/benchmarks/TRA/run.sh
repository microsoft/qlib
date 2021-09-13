#!/bin/bash

# we used random seed(1 1000 2000 3000 4000 5000) in our experiments 

# Directly run from Qlib command `qrun`
qrun configs/config_alstm.yaml

qrun configs/config_transformer.yaml

qrun configs/config_transformer_tra_init.yaml
qrun configs/config_transformer_tra.yaml

qrun configs/config_alstm_tra_init.yaml
qrun configs/config_alstm_tra.yaml


# Or setting different parameters with example.py
python example.py --config_file configs/config_alstm.yaml

python example.py --config_file configs/config_transformer.yaml

python example.py --config_file configs/config_transformer_tra_init.yaml
python example.py --config_file configs/config_transformer_tra.yaml

python example.py --config_file configs/config_alstm_tra_init.yaml
python example.py --config_file configs/config_alstm_tra.yaml



