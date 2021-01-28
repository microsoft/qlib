from tianshou.policy import BasePolicy
from tianshou.data import Batch
import numpy as np
import torch
from env import nan_weighted_avg


class TWAP(BasePolicy):
    """ The TWAP strategy. """

    def __init__(self, config):
        super().__init__()
        self.max_step_num = config["max_step_num"]
        self.num_cpus = config["num_cpus"]

    # @njit(parallel=True)
    def forward(self, batch: Batch, state=None, **kwargs) -> Batch:
        act = [1] * len(batch.obs.private)
        return Batch(act=act, state=state)

    def learn(self, batch, batch_size, repeat):
        pass

    def process_fn(self, batch, buffer, indice):
        pass


class VWAP(BasePolicy):
    """ The VWAP strategy."""

    def __init__(self, config):
        super().__init__()

    def forward(self, batch, state, **kwargs):
        obs = batch.obs
        r = np.stack(obs.prediction).reshape(-1)
        return Batch(act=r, state=state)

    def learn(self, batch, batch_size, repeat):
        pass

    def process_fn(self, batch, buffer, indice):
        pass


class AC(VWAP):
    """Almgren-Chriss strategy."""

    def __init__(self, config):
        super().__init__(config)
        self.T = config["max_step_num"]
        self.gamma = 0
        self.tau = 1
        self.lamb = config["lambda"]
        self.eps = 0.0625
        self.alpha = 0.02
        self.eta = 2.5e-6

    def forward(self, batch, state, **kwargs):
        obs = batch.obs
        sig = np.stack(obs.prediction).reshape(-1)
        sell = ~np.stack(obs.is_buy).astype(np.bool)
        data = np.stack(obs.private)
        t = data[:, 2]
        t = t + 1
        k_tild = self.lamb / self.eta * sig * sig
        k = np.arccosh(k_tild / 2 + 1)
        act = (np.sinh(k * (self.T - t)) - np.sinh(k * (self.T - t - 1))) / np.sinh(k * self.T)
        return Batch(act=act, state=state)
