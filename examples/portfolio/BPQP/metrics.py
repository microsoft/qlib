import numpy as np
import torch
from empyrical import max_drawdown


def metric_fn(preds):
    preds.index.name = "datetime"
    preds = preds[~np.isnan(preds["label"])]

    # prediction metrics
    ic = preds.groupby("datetime").apply(lambda x: x.label.corr(x.score)).mean()
    icir = ic / preds.groupby("datetime").apply(lambda x: x.label.corr(x.score)).std()
    rank_ic = preds.groupby("datetime").apply(lambda x: x.label.corr(x.score, method="spearman")).mean()
    rank_icir = rank_ic / preds.groupby("datetime").apply(lambda x: x.label.corr(x.score, method="spearman")).std()

    # portfolio metrics
    avg_ret = (
        preds.groupby("datetime").apply(lambda x: x.label.dot(x.weight_pred) - x.label.mean()).mean()
    )
    avg_std = preds.groupby("datetime").apply(lambda x: x.label.dot(x.weight_pred) - x.label.mean()).std()
    net_value = (preds.groupby("datetime").apply(lambda x: x.label.dot(x.weight_pred) - x.label.mean()) + 1).cumprod()[
        -1
    ]
    ret = preds.groupby("datetime").apply(lambda x: x.label.dot(x.weight_pred) - x.label.mean())
    mdd = max_drawdown(ret)

    return ic, rank_ic, avg_ret, avg_std, net_value, mdd, icir, rank_icir


def obj_fn(weight, rets, variance, args):
    return weight.T @ rets - 0.5 * args["sigma"] * weight.T @ variance @ weight


def regret_loss(weight_pred, exact_weight, pred, y, variance, args):
    return (obj_fn(weight_pred, y, variance, args) - obj_fn(exact_weight, y, variance, args)) ** 2


def mse_loss(pred, y, args):
    return torch.mean((pred - y) ** 2)


def huber_loss(weight_pred, pred, y, variance, args):
    reg_l = mse_loss(pred, y, args) - (args['gamma']*obj_fn(weight_pred, y, variance, args))
    if reg_l > args["zeta"] ** 2:
        return args["zeta"] * (reg_l - args["zeta"])
    else:
        return reg_l


def e2e_loss(weight_pred, exact_weight, pred, y, variance, args):
    gamma = args["gamma"]
    assert gamma > 0
    return regret_loss(weight_pred, exact_weight, pred, y, variance, args)*gamma + mse_loss(pred, y, args)


def soft_loss(weight_pred, pred, y, variance, args):
    gamma = args["gamma"]
    assert gamma > 0
    return mse_loss(pred, y, args)  - (weight_pred.T @ y)*gamma
