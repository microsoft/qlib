if [ ! -d "./logs" ]; then
     mkdir ./logs
fi
CUDA_VISIBLE_DEVICES=0 python -u main.py run_all --forecast_model GRU         -num_head 8 --tau 10 --lr 0.01  --market csi300 --data_dir crowd_data --rank_label False --begin_valid_epoch 30 > logs/gru300.log 2>&1
CUDA_VISIBLE_DEVICES=0 python -u main.py run_all --forecast_model LSTM        -num_head 8 --tau 10 --lr 0.01  --market csi300 --data_dir crowd_data --rank_label False --begin_valid_epoch 30 > logs/lstm300.log 2>&1
CUDA_VISIBLE_DEVICES=0 python -u main.py run_all --forecast_model ALSTM       -num_head 8 --tau 10 --lr 0.01  --market csi300 --data_dir crowd_data --rank_label False --begin_valid_epoch 30 > logs/alstm300.log 2>&1
CUDA_VISIBLE_DEVICES=0 python -u main.py run_all --forecast_model Transformer -num_head 8 --tau 10 --lr 0.01  --market csi300 --data_dir crowd_data --rank_label False --begin_valid_epoch 30 > logs/tfm300.log 2>&1
CUDA_VISIBLE_DEVICES=0 python -u main.py run_all --forecast_model GRU         -num_head 8 --tau 10 --lr 0.01  --market csi500 --data_dir crowd_data --rank_label False --begin_valid_epoch 30 > logs/gru500.log 2>&1
CUDA_VISIBLE_DEVICES=0 python -u main.py run_all --forecast_model LSTM        -num_head 8 --tau 10 --lr 0.01  --market csi500 --data_dir crowd_data --rank_label False --begin_valid_epoch 20 > logs/lstm500.log 2>&1
CUDA_VISIBLE_DEVICES=0 python -u main.py run_all --forecast_model ALSTM       -num_head 8 --tau 10 --lr 0.01  --market csi500 --data_dir crowd_data --rank_label False --begin_valid_epoch 30 > logs/alstm500.log 2>&1
CUDA_VISIBLE_DEVICES=0 python -u main.py run_all --forecast_model Transformer -num_head 8 --tau 10 --lr 0.001 --market csi500 --data_dir crowd_data --rank_label False --begin_valid_epoch 20 > logs/tfm500.log 2>&1
