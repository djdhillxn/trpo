import numpy as np
import torch

from trpo_repro.methods.base import BaseMethod, MethodUpdateStats


class RandomPolicyMethod(BaseMethod):
    trainable = False
    supports_checkpoints = False

    def __init__(self, obs_space, act_space, cfg, device: torch.device) -> None:
        super().__init__(cfg=cfg, device=device)
        self.obs_space = obs_space
        self.act_space = act_space
        self.is_discrete = hasattr(act_space, "n")

    @property
    def name(self) -> str:
        return "random"

    @property
    def variant(self) -> str:
        return str(self.cfg.method.get("variant", "uniform"))

    def act(self, obs: torch.Tensor, deterministic: bool = False):
        del deterministic
        batch_size = obs.shape[0]
        if self.is_discrete:
            actions = np.asarray([self.act_space.sample() for _ in range(batch_size)], dtype=np.int64)
            action_t = torch.as_tensor(actions, dtype=torch.long, device=obs.device)
        else:
            samples = np.asarray([self.act_space.sample() for _ in range(batch_size)], dtype=np.float32)
            action_t = torch.as_tensor(samples, dtype=torch.float32, device=obs.device)
        value_t = torch.zeros(batch_size, dtype=torch.float32, device=obs.device)
        logp_t = torch.zeros(batch_size, dtype=torch.float32, device=obs.device)
        return action_t, value_t, logp_t

    def update(self, batch: dict[str, torch.Tensor] | None) -> MethodUpdateStats:
        del batch
        return MethodUpdateStats(did_update=0.0)
