import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from solver import BPQP
from metrics import regret_loss, mse_loss, e2e_loss, soft_loss, huber_loss

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def train_epoch(model, solver, optimizer, train_loader, args):
    model.train()
    running_loss = 0
    for i, slc in tqdm(train_loader.iter_daily(), total=train_loader.daily_length):
        feature, label, variance, _, _ = train_loader.get_daily(i, slc)

        # predict return
        pred = model(feature)

        # differentiable solver & loss
        if args["loss"] == "e2e":
            weight_pred = solver(variance, pred)
            exact_weight = solver(variance, label)
            loss = e2e_loss(weight_pred, exact_weight, pred, label, variance, args)  # supervised by ground truth weight
        elif args["loss"] == "regret":
            weight_pred = solver(variance, pred)
            exact_weight = solver(variance, label)
            loss = regret_loss(weight_pred, exact_weight, pred, label, variance, args)
        elif args["loss"] == "mse":
            loss = mse_loss(pred, label, args)
        elif args["loss"] == "huber_loss":
            weight_pred = solver(variance, pred)
            loss = huber_loss(weight_pred, pred, label, variance, args)
        elif args["loss"] == "softloss":
            weight_pred = solver(variance, pred)
            loss = soft_loss(weight_pred, pred, label, variance, args)
        else:
            raise NotImplementedError
        running_loss += loss

        if i % args["fre_d"] == 0 and i > 0:
            running_loss = running_loss / args["fre_d"]
            running_loss.backward()
            torch.nn.utils.clip_grad_value_(model.parameters(), 3.0)
            optimizer.step()
            optimizer.zero_grad()
            running_loss = 0


def test_epoch(model, solver, metric_fn, test_loader, args, prefix="Test"):
    model.eval()

    losses = []
    regrets = []
    mse = []
    preds = []
    if args['solver'] in ['dc3', 'naive_nn']:
        outer_solver = BPQP(args)

    for i, slc in tqdm(test_loader.iter_daily(), desc=prefix, total=test_loader.daily_length):
        feature, label, variance, _, _ = test_loader.get_daily(i, slc)

        with torch.no_grad():
            pred = model(feature)

            try:
                if args['solver'] in ['dc3','naive_nn']:
                    weight_pred = solver(variance, pred)
                    exact_weight = outer_solver(variance, label)
                else:
                    weight_pred = solver(variance, pred)
                    exact_weight = solver(variance, label)
            except TypeError:
                # Extreme situations: zero or one tradeable stock
                print("Extreme situations: zero or one tradeable stock @", test_loader.get_daily_date(i),
                      " Tradeable stocks:", pred.item())
                continue

            if args["loss"] == "e2e":
                loss = e2e_loss(weight_pred, exact_weight, pred, label, variance, args)
                regret = regret_loss(weight_pred, exact_weight, pred, label, variance, args)
                _mse = mse_loss(pred, label, args)
            elif args["loss"] == "regret":
                loss = regret_loss(weight_pred, exact_weight, pred, label, variance, args)
                regret = loss
                _mse = mse_loss(pred, label, args)
            elif args["loss"] == "mse":
                # regret = regret_loss(weight_pred, exact_weight, pred, label, variance, args)
                regret = torch.zeros(1).to(device)
                loss = mse_loss(pred, label, args)
                _mse = loss
            elif args["loss"] == "huber_loss":
                regret = regret_loss(weight_pred, exact_weight, pred, label, variance, args)
                loss = huber_loss(weight_pred, pred, label, variance, args)
                _mse = mse_loss(pred, label, args)
            elif args["loss"] == "softloss":
                regret = regret_loss(weight_pred, exact_weight, pred, label, variance, args)
                loss = soft_loss(weight_pred, pred, label, variance, args)
                _mse = mse_loss(pred, label, args)
            else:
                raise NotImplementedError

            preds.append(
                pd.DataFrame(
                    {
                        "score": pred.cpu().numpy(),
                        "label": label.cpu().numpy(),
                        "weight_pred": weight_pred.cpu().numpy(),
                        "exact_weight": exact_weight.cpu().numpy(),
                    },
                    index=[test_loader.get_daily_date(i)] * len(pred),
                )
            )
        regrets.append(regret.item())
        mse.append(_mse.item())
        losses.append(loss.item())
    # evaluate
    preds = pd.concat(preds, axis=0)
    ic, rank_ic, avg_ret, avg_std, net_value, mdd, icir, rankicir = metric_fn(preds)

    scores = ic


    return (
        np.nanmean(losses),
        np.nanmean(regrets),
        np.nanmean(mse),
        scores,
        ic,
        rank_ic,
        avg_ret,
        avg_std,
        net_value,
        mdd,
        icir,
        rankicir,
    )


def inference(model, solver, data_loader, args):
    model.eval()

    preds = []
    for i, slc in tqdm(data_loader.iter_daily(), total=data_loader.daily_length):
        feature, label, variance, stock_index, _ = data_loader.get_daily(i, slc)
        with torch.no_grad():
            pred = model(feature)
            weight_pred = solver(variance, pred)
            preds.append(
                pd.DataFrame(
                    {
                        "stock_index": stock_index,
                        "score": pred.cpu().numpy(),
                        "label": label.cpu().numpy(),
                        "weight_pred": weight_pred.cpu().numpy(),
                    },
                    index=[data_loader.get_daily_date(i)] * len(pred),
                )
            )

    preds = pd.concat(preds, axis=0)
    return preds
