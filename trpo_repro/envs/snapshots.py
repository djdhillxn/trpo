from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class EnvSnapshot:
    payload: Any
    np_random_state: Any | None = None


class SnapshotError(RuntimeError):
    pass


def clone_env_state(env) -> EnvSnapshot:
    unwrapped = env.unwrapped
    np_state = None
    if hasattr(unwrapped, "np_random") and hasattr(unwrapped.np_random, "bit_generator"):
        np_state = copy.deepcopy(unwrapped.np_random.bit_generator.state)

    if hasattr(unwrapped, "ale"):
        ale = unwrapped.ale
        if hasattr(ale, "cloneSystemState"):
            return EnvSnapshot(payload=ale.cloneSystemState(), np_random_state=np_state)
        if hasattr(ale, "cloneState"):
            return EnvSnapshot(payload=ale.cloneState(), np_random_state=np_state)

    if hasattr(unwrapped, "data") and hasattr(unwrapped, "set_state"):
        qpos = np.array(unwrapped.data.qpos).copy()
        qvel = np.array(unwrapped.data.qvel).copy()
        payload = {"qpos": qpos, "qvel": qvel}
        return EnvSnapshot(payload=payload, np_random_state=np_state)

    raise SnapshotError("Environment does not expose a known state snapshot API.")


def restore_env_state(env, snapshot: EnvSnapshot) -> None:
    unwrapped = env.unwrapped
    if hasattr(unwrapped, "np_random") and hasattr(unwrapped.np_random, "bit_generator") and snapshot.np_random_state is not None:
        unwrapped.np_random.bit_generator.state = snapshot.np_random_state

    if hasattr(unwrapped, "ale"):
        ale = unwrapped.ale
        if hasattr(ale, "restoreSystemState"):
            ale.restoreSystemState(snapshot.payload)
            return
        if hasattr(ale, "restoreState"):
            ale.restoreState(snapshot.payload)
            return

    if hasattr(unwrapped, "set_state") and isinstance(snapshot.payload, dict):
        unwrapped.set_state(snapshot.payload["qpos"], snapshot.payload["qvel"])
        return

    raise SnapshotError("Environment does not expose a known state restore API.")
