# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Generator, Iterable, List, Optional, OrderedDict, Tuple, cast

import gym
import numpy as np
import torch
import torch.nn as nn
from gym.spaces import Discrete
from tianshou.data import Batch, ReplayBuffer, to_torch
from tianshou.policy import BasePolicy, DQNPolicy, PPOPolicy

from qlib.rl.trainer.trainer import Trainer

__all__ = ["AllOne", "PPO", "DQN"]


# baselines #


class NonLearnablePolicy(BasePolicy):
    """Tianshou's BasePolicy with empty ``learn`` and ``process_fn``.

    This could be moved outside in future.
    """

    def __init__(self, obs_space: gym.Space, action_space: gym.Space) -> None:
        super().__init__()

    def learn(self, batch: Batch, **kwargs: Any) -> Dict[str, Any]:
        return {}

    def process_fn(
        self,
        batch: Batch,
        buffer: ReplayBuffer,
        indices: np.ndarray,
    ) -> Batch:
        return Batch({})


class AllOne(NonLearnablePolicy):
    """Forward returns a batch full of 1.

    Useful when implementing some baselines (e.g., TWAP).
    """

    def __init__(self, obs_space: gym.Space, action_space: gym.Space, fill_value: float | int = 1.0) -> None:
        super().__init__(obs_space, action_space)

        self.fill_value = fill_value

    def forward(
        self,
        batch: Batch,
        state: dict | Batch | np.ndarray = None,
        **kwargs: Any,
    ) -> Batch:
        return Batch(act=np.full(len(batch), self.fill_value), state=state)


# ppo #


class PPOActor(nn.Module):
    def __init__(self, extractor: nn.Module, action_dim: int) -> None:
        super().__init__()
        self.extractor = extractor
        self.layer_out = nn.Sequential(nn.Linear(cast(int, extractor.output_dim), action_dim), nn.Softmax(dim=-1))

    def forward(
        self,
        obs: torch.Tensor,
        state: torch.Tensor = None,
        info: dict = {},
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        feature = self.extractor(to_torch(obs, device=auto_device(self)))
        out = self.layer_out(feature)
        return out, state


class PPOCritic(nn.Module):
    def __init__(self, extractor: nn.Module) -> None:
        super().__init__()
        self.extractor = extractor
        self.value_out = nn.Linear(cast(int, extractor.output_dim), 1)

    def forward(
        self,
        obs: torch.Tensor,
        state: torch.Tensor = None,
        info: dict = {},
    ) -> torch.Tensor:
        feature = self.extractor(to_torch(obs, device=auto_device(self)))
        return self.value_out(feature).squeeze(dim=-1)


class PPO(PPOPolicy):
    """A wrapper of tianshou PPOPolicy.

    Differences:

    - Auto-create actor and critic network. Supports discrete action space only.
    - Dedup common parameters between actor network and critic network
      (not sure whether this is included in latest tianshou or not).
    - Support a ``weight_file`` that supports loading checkpoint.
    - Some parameters' default values are different from original.
    """

    def __init__(
        self,
        network: nn.Module,
        obs_space: gym.Space,
        action_space: gym.Space,
        lr: float,
        weight_decay: float = 0.0,
        discount_factor: float = 1.0,
        max_grad_norm: float = 100.0,
        reward_normalization: bool = True,
        eps_clip: float = 0.3,
        value_clip: bool = True,
        vf_coef: float = 1.0,
        gae_lambda: float = 1.0,
        max_batch_size: int = 256,
        deterministic_eval: bool = True,
        weight_file: Optional[Path] = None,
    ) -> None:
        assert isinstance(action_space, Discrete)
        actor = PPOActor(network, action_space.n)
        critic = PPOCritic(network)
        optimizer = torch.optim.Adam(
            chain_dedup(actor.parameters(), critic.parameters()),
            lr=lr,
            weight_decay=weight_decay,
        )
        super().__init__(
            actor,
            critic,
            optimizer,
            torch.distributions.Categorical,
            discount_factor=discount_factor,
            max_grad_norm=max_grad_norm,
            reward_normalization=reward_normalization,
            eps_clip=eps_clip,
            value_clip=value_clip,
            vf_coef=vf_coef,
            gae_lambda=gae_lambda,
            max_batchsize=max_batch_size,
            deterministic_eval=deterministic_eval,
            observation_space=obs_space,
            action_space=action_space,
        )
        if weight_file is not None:
            set_weight(self, Trainer.get_policy_state_dict(weight_file))


class OPD(PPO):
    """Oracle Policy Distillation.

    Reference:
        Universal Trading for Order Execution with Oracle Policy Distillation. https://arxiv.org/abs/2103.10860
    """

    def __init__(
        self,
        network: nn.Module,
        obs_space: gym.Space,
        action_space: gym.Space,
        lr: float,
        weight_decay: float = 0.0,
        discount_factor: float = 1.0,
        max_grad_norm: float = 100.0,
        reward_normalization: bool = True,
        eps_clip: float = 0.3,
        value_clip: bool = True,
        vf_coef: float = 1.0,
        gae_lambda: float = 1.0,
        max_batch_size: int = 256,
        deterministic_eval: bool = True,
        dis_coef: float = 0.01,
        weight_file: Optional[Path] = None,
    ) -> None:
        self._weight_dis = dis_coef
        super().__init__(
            network,
            obs_space,
            action_space,
            lr,
            weight_decay,
            discount_factor,
            max_grad_norm,
            reward_normalization,
            eps_clip,
            value_clip,
            vf_coef,
            gae_lambda,
            max_batch_size,
            deterministic_eval,
            weight_file,
        )

    def learn(  # type: ignore
        self, batch: Batch, batch_size: int, repeat: int, **kwargs: Any
    ) -> Dict[str, List[float]]:
        losses, clip_losses, vf_losses, dis_losses, ent_losses = [], [], [], [], []
        for step in range(repeat):
            if self._recompute_adv and step > 0:
                batch = self._compute_returns(batch, self._buffer, self._indices)
            for minibatch in batch.split(batch_size, merge_last=True):
                # calculate loss for actor
                out = self(minibatch)
                dist = out.dist
                if self._norm_adv:
                    mean, std = minibatch.adv.mean(), minibatch.adv.std()
                    minibatch.adv = (minibatch.adv - mean) / std  # per-batch norm
                ratio = (dist.log_prob(minibatch.act) - minibatch.logp_old).exp().float()
                ratio = ratio.reshape(ratio.size(0), -1).transpose(0, 1)
                surr1 = ratio * minibatch.adv
                surr2 = ratio.clamp(1.0 - self._eps_clip, 1.0 + self._eps_clip) * minibatch.adv
                if self._dual_clip:
                    clip1 = torch.min(surr1, surr2)
                    clip2 = torch.max(clip1, self._dual_clip * minibatch.adv)
                    clip_loss = -torch.where(minibatch.adv < 0, clip2, clip1).mean()
                else:
                    clip_loss = -torch.min(surr1, surr2).mean()
                # calculate loss for critic
                value = self.critic(minibatch.obs).flatten()
                if self._value_clip:
                    v_clip = minibatch.v_s + (value - minibatch.v_s).clamp(-self._eps_clip, self._eps_clip)
                    vf1 = (minibatch.returns - value).pow(2)
                    vf2 = (minibatch.returns - v_clip).pow(2)
                    vf_loss = torch.max(vf1, vf2).mean()
                else:
                    vf_loss = (minibatch.returns - value).pow(2).mean()
                # calculate distillation loss
                teacher_action = torch.tensor(minibatch.obs["teacher_action"]).long()
                logits = out.logits
                dis_loss = nn.functional.nll_loss(logits.log(), teacher_action)
                # calculate regularization and overall loss
                ent_loss = dist.entropy().mean()
                loss = clip_loss + self._weight_vf * vf_loss - self._weight_ent * ent_loss + self._weight_dis * dis_loss
                self.optim.zero_grad()
                loss.backward()
                if self._grad_norm:  # clip large gradient
                    nn.utils.clip_grad_norm_(self._actor_critic.parameters(), max_norm=self._grad_norm)
                self.optim.step()
                clip_losses.append(clip_loss.item())
                vf_losses.append(vf_loss.item())
                dis_losses.append(dis_loss.item())
                ent_losses.append(ent_loss.item())
                losses.append(loss.item())

        return {
            "loss": losses,
            "loss/clip": clip_losses,
            "loss/vf": vf_losses,
            "loss/ent": ent_losses,
        }


DQNModel = PPOActor  # Reuse PPOActor.


class DQN(DQNPolicy):
    """A wrapper of tianshou DQNPolicy.

    Differences:

    - Auto-create model network. Supports discrete action space only.
    - Support a ``weight_file`` that supports loading checkpoint.
    """

    def __init__(
        self,
        network: nn.Module,
        obs_space: gym.Space,
        action_space: gym.Space,
        lr: float,
        weight_decay: float = 0.0,
        discount_factor: float = 0.99,
        estimation_step: int = 1,
        target_update_freq: int = 0,
        reward_normalization: bool = False,
        is_double: bool = True,
        clip_loss_grad: bool = False,
        weight_file: Optional[Path] = None,
    ) -> None:
        assert isinstance(action_space, Discrete)

        model = DQNModel(network, action_space.n)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

        super().__init__(
            model,
            optimizer,
            discount_factor=discount_factor,
            estimation_step=estimation_step,
            target_update_freq=target_update_freq,
            reward_normalization=reward_normalization,
            is_double=is_double,
            clip_loss_grad=clip_loss_grad,
        )
        if weight_file is not None:
            set_weight(self, Trainer.get_policy_state_dict(weight_file))


# utilities: these should be put in a separate (common) file. #


def auto_device(module: nn.Module) -> torch.device:
    for param in module.parameters():
        return param.device
    return torch.device("cpu")  # fallback to cpu


def set_weight(policy: nn.Module, loaded_weight: OrderedDict) -> None:
    try:
        policy.load_state_dict(loaded_weight)
    except RuntimeError:
        # try again by loading the converted weight
        # https://github.com/thu-ml/tianshou/issues/468
        for k in list(loaded_weight):
            loaded_weight["_actor_critic." + k] = loaded_weight[k]
        policy.load_state_dict(loaded_weight)


def chain_dedup(*iterables: Iterable) -> Generator[Any, None, None]:
    seen = set()
    for iterable in iterables:
        for i in iterable:
            if i not in seen:
                seen.add(i)
                yield i
