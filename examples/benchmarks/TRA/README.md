# Learning Multiple Stock Trading Patterns with Temporal Routing Adaptor and Optimal Transport

This code provides a PyTorch implementation for TRA (Temporal Routing Adaptor), as described in the paper [Learning Multiple Stock Trading Patterns with Temporal Routing Adaptor and Optimal Transport](http://arxiv.org/abs/2106.12950).

* TRA (Temporal Routing Adaptor) is a lightweight module that consists of a set of independent predictors for learning multiple patterns as well as a router to dispatch samples to different predictors.
* We also design a learning algorithm based on Optimal Transport (OT) to obtain the optimal sample to predictor assignment and effectively optimize the router with such assignment through an auxiliary loss term.


# Running TRA 

## Requirements
- Install `Qlib` main branch

## Running 

We attach our running scripts for the paper in `run.sh`.

And here are two ways to run the model:

* Running from scripts with default parameters
    You can directly run from Qlib command `qrun`:
    ```
    qrun configs/config_alstm.yaml
    ```

* Running from code with self-defined parameters
    Setting different parameters is also allowed. See codes in `example.py`:
    ```
    python example.py --config_file configs/config_alstm.yaml
    ```

Here we trained TRA on a pretrained backbone model. Therefore we run `*_init.yaml` before TRA's scipts.

# Results

## Outputs

After running the scripts, you can find result files in path `./output`:

`info.json` - config settings and result metrics.

`log.csv` - running logs.

`model.bin` - the model parameter dictionary.

`pred.pkl` - the prediction scores and output for inference.

## Our Results
| Methods | MSE| MAE| IC | ICIR | AR | AV | SR | MDD |
|-------------------|-------------------|---------------------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|
|Linear|0.163|0.327|0.020|0.132|-3.2%|16.8%|-0.191|32.1%|
|LightGBM|0.160(0.000)|0.323(0.000)|0.041|0.292|7.8%|15.5%|0.503|25.7%|
|MLP|0.160(0.002)|0.323(0.003)|0.037|0.273|3.7%|15.3%|0.264|26.2%|
|SFM|0.159(0.001)	|0.321(0.001)	|0.047	|0.381	|7.1%	|14.3%	|0.497	|22.9%|
|ALSTM|0.158(0.001)	|0.320(0.001)	|0.053	|0.419	|12.3%	|13.7%	|0.897	|20.2%|
|Trans.|0.158(0.001)	|0.322(0.001)	|0.051	|0.400	|14.5%	|14.2%	|1.028	|22.5%|
|ALSTM+TS|0.160(0.002)	|0.321(0.002)	|0.039	|0.291	|6.7%	|14.6%	|0.480|22.3%|
|Trans.+TS|0.160(0.004)	|0.324(0.005)	|0.037	|0.278	|10.4%	|14.7%	|0.722	|23.7%|
|ALSTM+TRA(Ours)|0.157(0.000)	|0.318(0.000)	|0.059	|0.460	|12.4%	|14.0%	|0.885	|20.4%|
|Trans.+TRA(Ours)|0.157(0.000)	|0.320(0.000)	|0.056	|0.442	|16.1%	|14.2%	|1.133	|23.1%|

A more detailed demo for our experiment results in the paper can be found in `Report.ipynb`.

# Common Issues

For help or issues using TRA, please submit a GitHub issue.

Sometimes we might encounter situation where the loss is `NaN`, please check the `epsilon` parameter in the sinkhorn algorithm, adjusting the `epsilon` according to input's scale is important. 

# Citation
If you find this repository useful in your research, please cite:
```
@inproceedings{HengxuKDD2021,
 author = {Hengxu Lin and Dong Zhou and Weiqing Liu and Jiang Bian},
 title = {Learning Multiple Stock Trading Patterns with Temporal Routing Adaptor and Optimal Transport},
 booktitle = {Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery \& Data Mining},
 series = {KDD '21},
 year = {2021},
 publisher = {ACM},
}
```
