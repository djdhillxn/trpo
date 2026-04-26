import numpy as np

EstimatorName = str


_ALIAS_MAP = {
    # Paper-faithful single-path TRPO / Monte Carlo returns
    "trpo_paper": "trpo_paper",
    "trpo-paper": "trpo_paper",
    "paper_returns": "trpo_paper",
    "paper-returns": "trpo_paper",
    "paper_mc": "trpo_paper",
    "paper": "trpo_paper",
    # Monte Carlo returns with learned value baseline
    "value_baseline": "value_baseline",
    "returns_value_baseline": "value_baseline",
    "returns-value-baseline": "value_baseline",
    "mc_value_baseline": "value_baseline",
    "mc-value-baseline": "value_baseline",
    "mc_baseline": "value_baseline",
    "mc": "value_baseline",
    # Generalized advantage estimation
    "gae": "gae",
    "generalized_advantage_estimation": "gae",
}


_ESTIMATOR_DESCRIPTIONS = {
    "trpo_paper": "Paper-faithful Monte Carlo returns / trajectory Q-estimates with no learned value baseline.",
    "value_baseline": "Monte Carlo returns with a learned value baseline subtracted from the policy-training weights.",
    "gae": "Generalized Advantage Estimation using lambda-weighted TD residuals and a learned value function.",
}


def canonicalize_estimator(estimator: str | None) -> EstimatorName:
    if estimator is None:
        raise ValueError("Estimator name must not be None.")
    normalized = str(estimator).strip().lower()
    if normalized not in _ALIAS_MAP:
        raise ValueError(f"Unknown estimator mode: {estimator}")
    return _ALIAS_MAP[normalized]



def estimator_description(estimator: str | None) -> str:
    return _ESTIMATOR_DESCRIPTIONS[canonicalize_estimator(estimator)]



def discounted_cumsum(x: np.ndarray, discount: float) -> np.ndarray:
    values = np.asarray(x, dtype=np.float32)
    out = np.zeros_like(values, dtype=np.float32)
    running = 0.0
    for i in reversed(range(len(values))):
        running = float(values[i]) + discount * running
        out[i] = running
    return out



def compute_path_targets(
    estimator: str,
    rewards: np.ndarray | list[float],
    values: np.ndarray | list[float],
    gamma: float,
    lam: float,
    last_val: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute return targets and policy-training weights/advantages for one path.

    Returns
    -------
    returns:
        Discounted return targets for value-function fitting.
    weights:
        Policy-training weights / advantages.
        * trpo_paper: identical to returns
        * value_baseline: returns minus baseline values
        * gae: lambda-weighted generalized advantage estimate
    """
    mode = canonicalize_estimator(estimator)
    rews = np.asarray(rewards, dtype=np.float32)
    vals = np.asarray(values, dtype=np.float32)

    if rews.ndim != 1 or vals.ndim != 1:
        raise ValueError("rewards and values must be one-dimensional.")
    if len(rews) != len(vals):
        raise ValueError("rewards and values must have the same length.")

    rews_ext = np.append(rews, np.float32(last_val))

    if mode == "trpo_paper":
        returns = discounted_cumsum(rews_ext, gamma)[:-1]
        weights = returns.copy()
    elif mode == "value_baseline":
        returns = discounted_cumsum(rews_ext, gamma)[:-1]
        weights = returns - vals
    elif mode == "gae":
        vals_ext = np.append(vals, np.float32(last_val))
        deltas = rews_ext[:-1] + gamma * vals_ext[1:] - vals_ext[:-1]
        weights = discounted_cumsum(deltas.astype(np.float32, copy=False), gamma * lam)
        returns = discounted_cumsum(rews_ext, gamma)[:-1]
    else:  # pragma: no cover
        raise ValueError(f"Unsupported estimator mode: {mode}")

    returns = np.asarray(returns, dtype=np.float32)
    weights = np.asarray(weights, dtype=np.float32)
    if len(returns) != len(rews) or len(weights) != len(rews):
        raise RuntimeError("Computed path targets do not match path length.")
    return returns, weights
