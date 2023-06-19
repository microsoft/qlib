import copy
from collections import defaultdict

import numpy as np
from qlib.model.meta import MetaTaskDataset

from qlib.model.meta.model import MetaTaskModel

from tqdm import tqdm
import pandas as pd
import torch
from torch import optim
import torch.nn.functional as F
import higher
from . import higher_optim  # IMPORTANT, DO NOT DELETE

from .utils import override_state
from .dataset import MetaDatasetInc
from .net import DoubleAdapt, ForecastModel


class MetaModelInc(MetaTaskModel):
    def __init__(
        self,
        task_config,
        lr_model=0.001,
        first_order=True,
        is_rnn=False,
        x_dim=None,
        alpha=360,
        pretrained_model=None,
        begin_valid_epoch=0,
        **kwargs,
    ):
        self.task_config = task_config
        self.lr_model = lr_model
        self.first_order = first_order
        self.is_rnn = is_rnn
        self.begin_valid_epoch = begin_valid_epoch
        self.framework = self._init_framework(task_config, x_dim, lr_model, need_permute=int(alpha) == 360,
                                              model=pretrained_model, **kwargs)
        self.opt = self._init_meta_optimizer(**kwargs)

    def _init_framework(self, task_config, x_dim=None, lr_model=0.001, need_permute=False, model=None, **kwargs):
        return ForecastModel(task_config, x_dim=x_dim, lr=lr_model, need_permute=need_permute, model=model)

    def _init_meta_optimizer(self, **kwargs):
        return self.framework.opt

    def fit(self, meta_dataset: MetaDatasetInc):

        phases = ["train", "test"]
        meta_tasks_l = meta_dataset.prepare_tasks(phases)

        self.cnt = 0
        self.framework.train()
        torch.set_grad_enabled(True)
        # run training
        best_ic, over_patience = -1e3, 8
        patience = over_patience
        best_checkpoint = copy.deepcopy(self.framework.state_dict())
        for epoch in tqdm(range(300), desc="epoch"):
            for phase, task_list in zip(phases, meta_tasks_l):
                if phase == "test":
                    if epoch < self.begin_valid_epoch:
                        continue
                pred_y, ic = self.run_epoch(phase, task_list)
                if phase == "test":
                    if ic < best_ic:
                        patience -= 1
                    else:
                        best_ic = ic
                        print("best ic:", best_ic)
                        patience = over_patience
                        best_checkpoint = copy.deepcopy(self.framework.state_dict())
            if patience <= 0:
                # R.save_objects(**{"model.pkl": self.tn})
                break
        self.fitted = True
        self.framework.load_state_dict(best_checkpoint)

    def run_epoch(self, phase, task_list, tqdm_show=False):
        pred_y_all, mse_all = [], 0
        indices = np.arange(len(task_list))
        if phase == "test":
            checkpoint = copy.deepcopy(self.framework.state_dict())
            checkpoint_opt = copy.deepcopy(self.framework.opt.state_dict())
            checkpoint_opt_meta = copy.deepcopy(self.opt.state_dict())
        elif phase == "train":
            np.random.shuffle(indices)
        self.phase = phase
        for i in tqdm(indices, desc=phase) if tqdm_show else indices:
            # torch.cuda.empty_cache()
            meta_input = task_list[i].get_meta_input()
            if not isinstance(meta_input['X_train'], torch.Tensor):
                meta_input = {
                    k: torch.tensor(v, device=self.framework.device, dtype=torch.float32) if 'idx' not in k else v
                    for k, v in meta_input.items()
                }
            pred = self.run_task(meta_input, phase)
            if phase != "train":
                test_idx = meta_input["test_idx"]
                pred_y_all.append(
                    pd.DataFrame(
                        {
                            "pred": pd.Series(pred, index=test_idx),
                            "label": pd.Series(meta_input["y_test"], index=test_idx),
                        }
                    )
                )
        if phase == "test":
            self.framework.load_state_dict(checkpoint)
            self.framework.opt.load_state_dict(checkpoint_opt)
            self.opt.load_state_dict(checkpoint_opt_meta)
        if phase != "train":
            pred_y_all = pd.concat(pred_y_all)
        if phase == "test":
            ic = pred_y_all.groupby("datetime").apply(lambda df: df["pred"].corr(df["label"], method="pearson")).mean()
            print(ic)
            return pred_y_all, ic
        return pred_y_all, None

    def run_task(self, meta_input, phase):
        """ Naive incremental learning """
        self.framework.opt.zero_grad()
        y_hat = self.framework(meta_input["X_train"].to(self.framework.device), None, transform=False)
        loss = self.framework.criterion(y_hat, meta_input["y_train"].to(self.framework.device))
        loss.backward()
        self.framework.opt.step()
        self.framework.opt.zero_grad()
        with torch.no_grad():
            pred = self.framework(meta_input["X_test"].to(self.framework.device), None, transform=False)
        return pred.detach().cpu().numpy()

    def inference(self, meta_dataset: MetaTaskDataset):
        meta_tasks_test = meta_dataset.prepare_tasks("test")
        self.framework.train()
        pred_y_all, ic = self.run_epoch("online", meta_tasks_test, tqdm_show=True)
        return pred_y_all, ic


class DoubleAdaptManager(MetaModelInc):
    def __init__(
        self,
        task_config,
        lr_model=0.001,
        lr_da=0.01,
        lr_ma=0.001,
        reg=0.5,
        adapt_x=True,
        adapt_y=True,
        first_order=True,
        is_rnn=False,
        factor_num=6,
        x_dim=360,
        alpha=360,
        num_head=8,
        temperature=10,
        pretrained_model=None,
        begin_valid_epoch=0,
    ):
        super(DoubleAdaptManager, self).__init__(task_config, x_dim=x_dim, lr_model=lr_model,
                                                 first_order=first_order, is_rnn=is_rnn, alpha=alpha,
                                                 pretrained_model=pretrained_model,
                                                 begin_valid_epoch=begin_valid_epoch)
        self.lr_da = lr_da
        self.lr_ma = lr_ma
        self.adapt_x = adapt_x
        self.adapt_y = adapt_y
        self.reg = reg
        self.sigma = 1 ** 2 * 2
        self.factor_num = factor_num
        self.lamda = 0.5
        self.num_head = num_head
        self.temperature = temperature

    def _init_framework(self, task_config, x_dim=None, lr_model=0.001, need_permute=False, model=None,
                        num_head=8, temperature=10, factor_num=6, **kwargs):
        return DoubleAdapt(
            task_config, x_dim=x_dim, lr=lr_model, need_permute=need_permute, model=model,
            factor_num=factor_num, num_head=num_head, temperature=temperature,
        )

    def _init_meta_optimizer(self, lr_da=0.01, **kwargs):
        return optim.Adam(self.framework.meta_params, lr=lr_da)    # To optimize the data adapter
        # return optim.Adam([{'params': self.tn.teacher_y.parameters(), 'lr': self.lr_y},
        #                    {'params': self.tn.teacher_x.parameters()}], lr=self.lr)

    def run_task(self, meta_input, phase):

        self.framework.opt.zero_grad()
        self.opt.zero_grad()

        """ Incremental data adaptation & Model adaptation """
        X = meta_input["X_train"].to(self.framework.device)
        with higher.innerloop_ctx(
            self.framework.model,
            self.framework.opt,
            copy_initial_weights=False,
            track_higher_grads=not self.first_order,
            override={'lr': [self.lr_model]}
        ) as (fmodel, diffopt):
            with torch.backends.cudnn.flags(enabled=self.first_order or not self.is_rnn):
                y_hat, _ = self.framework(X, model=fmodel, transform=self.adapt_x)
        y = meta_input["y_train"].to(self.framework.device)
        if self.adapt_y:
            raw_y = y
            y = self.framework.teacher_y(X, raw_y, inverse=False)
        loss2 = self.framework.criterion(y_hat, y)
        diffopt.step(loss2)

        """ Online inference """
        if phase != "train" and "X_extra" in meta_input and meta_input["X_extra"].shape[0] > 0:
            X_test = torch.cat([meta_input["X_extra"].to(self.framework.device), meta_input["X_test"].to(self.framework.device), ], 0, )
            y_test = torch.cat([meta_input["y_extra"].to(self.framework.device), meta_input["y_test"].to(self.framework.device), ], 0, )
        else:
            X_test = meta_input["X_test"].to(self.framework.device)
            y_test = meta_input["y_test"].to(self.framework.device)
        pred, X_test_adapted = self.framework(X_test, model=fmodel, transform=self.adapt_x)
        mask_y = meta_input.get("mask_y")
        if self.adapt_y:
            pred = self.framework.teacher_y(X_test, pred, inverse=True)
        if phase != "train":
            test_begin = len(meta_input["y_extra"]) if "y_extra" in meta_input else 0
            meta_end = test_begin + meta_input["meta_end"]
            output = pred[test_begin:].detach().cpu().numpy()
            X_test = X_test[:meta_end]
            X_test_adapted = X_test_adapted[:meta_end]
            if mask_y is not None:
                pred = pred[mask_y]
                meta_end = sum(mask_y[:meta_end])
            pred = pred[:meta_end]
            y_test = y_test[:meta_end]
        else:
            output = pred.detach().cpu().numpy()

        """ Optimization of meta-learners """
        loss = self.framework.criterion(pred, y_test)
        if self.adapt_y:
            if not self.first_order:
                y = self.framework.teacher_y(X, raw_y, inverse=False)
            loss_y = F.mse_loss(y, raw_y)
            if self.first_order:
                """ Please refer to Appendix C in https://arxiv.org/pdf/2306.09862.pdf """
                with torch.no_grad():
                    pred2, _ = self.framework(X_test_adapted, model=None, transform=False, )
                    pred2 = self.framework.teacher_y(X_test, pred2, inverse=True).detach()
                    loss_old = self.framework.criterion(pred2.view_as(y_test), y_test)
                loss_y = (loss_old.item() - loss.item()) / self.sigma * loss_y + loss_y * self.reg
            else:
                loss_y = loss_y * self.reg
            loss_y.backward()
        loss.backward()
        if self.adapt_x or self.adapt_y:
            self.opt.step()
        self.framework.opt.step()
        return output


