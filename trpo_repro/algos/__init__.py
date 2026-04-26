"""Optimization algorithms and shared advantage helpers."""

from trpo_repro.algos.advantages import canonicalize_estimator, compute_path_targets, discounted_cumsum

__all__ = ["canonicalize_estimator", "compute_path_targets", "discounted_cumsum"]
