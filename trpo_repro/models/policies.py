import math
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical, Independent, Normal, kl_divergence

from trpo_repro.models.cnn import AtariBody
from trpo_repro.models.mlp import build_mlp


def _build_body_mlp(input_dim: int, hidden_sizes: list[int], activation: str) -> nn.Sequential:
    layers: list[nn.Module] = []
    prev = input_dim
    act_cls = {"tanh": nn.Tanh, "relu": nn.ReLU, "elu": nn.ELU}[activation.lower()]
    for size in hidden_sizes:
        layers.append(nn.Linear(prev, size))
        layers.append(act_cls())
        prev = size
    return nn.Sequential(*layers)


@dataclass
class DistBatch:
    dist: torch.distributions.Distribution
    entropy: torch.Tensor


class BasePolicy(nn.Module):
    is_discrete: bool = False

    def distribution(self, obs: torch.Tensor):
        raise NotImplementedError

    def log_prob_from_dist(self, dist, action: torch.Tensor) -> torch.Tensor:
        return dist.log_prob(action)

    def act(self, obs: torch.Tensor, deterministic: bool = False):
        dist = self.distribution(obs)
        if self.is_discrete:
            action = torch.argmax(dist.probs, dim=-1) if deterministic else dist.sample()
        else:
            action = dist.mean if deterministic else dist.sample()
        logp = self.log_prob_from_dist(dist, action)
        return action, logp, dist

    def entropy(self, obs: torch.Tensor) -> torch.Tensor:
        return self.distribution(obs).entropy().mean()


class GaussianPolicy(BasePolicy):
    is_discrete = False

    def __init__(self, obs_shape: tuple[int, ...], action_dim: int, cfg) -> None:
        super().__init__()
        activation = cfg.model.get("activation", "tanh")
        policy_hidden_sizes = list(cfg.model.get("policy_hidden_sizes", [64, 64]))
        if len(obs_shape) == 3:
            self.body = AtariBody(obs_shape[0], hidden_dim=int(cfg.model.get("cnn_fc_dim", 20)))
            self.mean_layer = nn.Linear(self.body.output_dim, action_dim)
        else:
            input_dim = int(np.prod(obs_shape))
            if policy_hidden_sizes:
                self.body = _build_body_mlp(input_dim, policy_hidden_sizes, activation=activation)
                body_out_dim = policy_hidden_sizes[-1]
            else:
                self.body = nn.Identity()
                body_out_dim = input_dim
            self.mean_layer = nn.Linear(body_out_dim, action_dim)

        init_log_std = float(cfg.model.get("init_log_std", -0.5))
        self.log_std = nn.Parameter(torch.ones(action_dim) * init_log_std)

    def _features(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.ndim > 2 and isinstance(self.body, AtariBody):
            return self.body(obs)
        if obs.ndim > 2:
            obs = obs.view(obs.shape[0], -1)
        return self.body(obs)

    def distribution(self, obs: torch.Tensor):
        feat = self._features(obs)
        mean = self.mean_layer(feat)
        std = torch.exp(self.log_std).expand_as(mean)
        return Independent(Normal(mean, std), 1)


class CategoricalPolicy(BasePolicy):
    is_discrete = True

    def __init__(self, obs_shape: tuple[int, ...], num_actions: int, cfg) -> None:
        super().__init__()
        activation = cfg.model.get("activation", "tanh")
        policy_hidden_sizes = list(cfg.model.get("policy_hidden_sizes", [64, 64]))
        if len(obs_shape) == 3:
            self.body = AtariBody(obs_shape[0], hidden_dim=int(cfg.model.get("cnn_fc_dim", 20)))
            self.logits_layer = nn.Linear(self.body.output_dim, num_actions)
        else:
            input_dim = int(np.prod(obs_shape))
            if policy_hidden_sizes:
                self.body = _build_body_mlp(input_dim, policy_hidden_sizes, activation=activation)
                body_out_dim = policy_hidden_sizes[-1]
            else:
                self.body = nn.Identity()
                body_out_dim = input_dim
            self.logits_layer = nn.Linear(body_out_dim, num_actions)

    def _features(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.ndim > 2 and isinstance(self.body, AtariBody):
            return self.body(obs)
        if obs.ndim > 2:
            obs = obs.view(obs.shape[0], -1)
        return self.body(obs)

    def distribution(self, obs: torch.Tensor):
        feat = self._features(obs)
        logits = self.logits_layer(feat)
        return Categorical(logits=logits)


def make_policy(obs_space, act_space, cfg):
    obs_shape = tuple(obs_space.shape)
    if hasattr(act_space, "n"):
        return CategoricalPolicy(obs_shape=obs_shape, num_actions=act_space.n, cfg=cfg)
    if hasattr(act_space, "shape"):
        return GaussianPolicy(obs_shape=obs_shape, action_dim=int(np.prod(act_space.shape)), cfg=cfg)
    raise TypeError("Unsupported action space")


def per_state_kl(old_dist, new_dist) -> torch.Tensor:
    """Return the per-sample KL divergence between two policy batches."""
    return kl_divergence(old_dist, new_dist)


def mean_kl(old_dist, new_dist) -> torch.Tensor:
    return per_state_kl(old_dist, new_dist).mean()


def max_kl(old_dist, new_dist) -> torch.Tensor:
    return per_state_kl(old_dist, new_dist).max()
