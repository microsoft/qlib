nohup python main.py --market CN --loss e2e --predictor mlp --solver bpqp>> ./CNmlp_e2e_bpqp.log 2>&1 &
nohup python main.py --market CN --loss mse --predictor mlp --solver bpqp>> ./CNmlp_mse_bpqp.log 2>&1 &
nohup python main.py --market CN --loss e2e --predictor mlp --solver dc3>> ./CNmlp_e2e_dc3.log 2>&1 &