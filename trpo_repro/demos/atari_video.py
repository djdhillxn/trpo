from __future__ import annotations

import csv
import json
import re
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch

from trpo_repro.config import DotDict, load_config
from trpo_repro.envs.factory import make_env
from trpo_repro.methods import make_method
from trpo_repro.utils.torch_utils import RunningMeanStd

PolicyMode = Literal["deterministic", "stochastic"]
RecordSelection = Literal["best", "first"]


@dataclass
class EpisodeSummary:
    episode: int
    seed: int
    return_: float
    length: int
    video_path: str | None = None


@dataclass
class VideoRunSummary:
    config_path: str
    checkpoint_path: str
    output_dir: str
    run_name: str
    environment: str
    method: str
    variant: str
    checkpoint_name: str
    policy_mode: str
    record_selection: str
    seed: int
    episodes: int
    selected_episode: int
    selected_return: float
    selected_length: int
    return_mean: float
    return_std: float
    length_mean: float
    artifact_stem: str
    video_path: str
    json_path: str
    csv_path: str


def _set_nested(cfg: DotDict, dotted_key: str, value: Any) -> None:
    cursor: dict[str, Any] = cfg
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        if part not in cursor or not isinstance(cursor[part], dict):
            cursor[part] = {}
        cursor = cursor[part]
    cursor[parts[-1]] = value


def _load_checkpoint(path: str | Path, device: torch.device) -> dict[str, Any]:
    path = Path(path).expanduser()
    try:
        ckpt = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location=device)
    if not isinstance(ckpt, dict):
        raise ValueError(f"Checkpoint must be a dict-like object, got {type(ckpt)!r}: {path}")
    return ckpt


def _restore_obs_rms(ckpt: dict[str, Any], cfg: DotDict, env) -> RunningMeanStd | None:
    normalize_obs = bool(cfg.train.get("normalize_obs", False)) and len(env.observation_space.shape) == 1
    mean = ckpt.get("obs_rms_mean")
    var = ckpt.get("obs_rms_var")
    count = ckpt.get("obs_rms_count")
    if not normalize_obs or mean is None or var is None or count is None:
        return None
    rms = RunningMeanStd(shape=tuple(env.observation_space.shape))
    rms.mean = np.asarray(mean, dtype=np.float64)
    rms.var = np.asarray(var, dtype=np.float64)
    rms.count = float(count)
    return rms


def _preprocess_obs(obs: np.ndarray, obs_rms: RunningMeanStd | None) -> np.ndarray:
    obs = np.asarray(obs)
    if obs_rms is not None:
        obs = obs_rms.normalize(obs)
    return obs.astype(np.float32)


def _checkpoint_state(ckpt: dict[str, Any]) -> dict[str, Any]:
    state = ckpt.get("state")
    if state is None and "policy" in ckpt:
        state = {"policy": ckpt.get("policy"), "value_fn": ckpt.get("value_fn")}
    if not isinstance(state, dict):
        raise ValueError("Checkpoint does not contain a usable method state. Expected key 'state' or 'policy'.")
    return state


def _filename_token(value: Any, fallback: str = "artifact") -> str:
    token = re.sub(r"[^a-z0-9_-]+", "_", str(value).strip().lower())
    token = re.sub(r"_+", "_", token).strip("_-")
    return token or fallback


def _artifact_stem(
    *,
    cfg: DotDict,
    method,
    checkpoint_path: str | Path,
    policy_mode: PolicyMode,
    seed: int,
    record_selection: RecordSelection,
    episodes: int,
) -> str:
    run_name = _filename_token(cfg.train.get("run_name", "atari_policy"), fallback="atari_policy")
    method_name = _filename_token(method.name, fallback="policy")
    variant = _filename_token(method.variant, fallback="default")
    checkpoint_name = _filename_token(Path(checkpoint_path).stem, fallback="checkpoint")
    return "__".join(
        [
            run_name,
            f"{method_name}-{variant}",
            f"ckpt-{checkpoint_name}",
            _filename_token(policy_mode),
            f"seed-{int(seed)}",
            f"{_filename_token(record_selection)}-of-{int(episodes)}",
        ]
    )


def _next_available_artifact_stem(output_dir: Path, base_stem: str) -> str:
    if not any(output_dir.glob(f"{base_stem}__*")):
        return base_stem
    run_number = 2
    while any(output_dir.glob(f"{base_stem}__run-{run_number:02d}__*")):
        run_number += 1
    return f"{base_stem}__run-{run_number:02d}"


def _seed_rollout(seed: int, device: torch.device) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _make_render_env(cfg: DotDict, seed: int):
    # The training config should stay untouched on disk. For video only, ask Gymnasium/ALE
    # to expose RGB frames via env.render().
    _set_nested(cfg, "env.render_mode", "rgb_array")
    return make_env(cfg, seed=seed)


def load_policy_for_video(config_path: str | Path, checkpoint_path: str | Path, device: str | torch.device, seed: int):
    """Build the configured TRPO/PPO method and load its checkpoint for inference."""
    device = torch.device(device)
    cfg = load_config(config_path)
    env = _make_render_env(cfg, seed=seed)

    method = make_method(env.observation_space, env.action_space, cfg, device)
    ckpt = _load_checkpoint(checkpoint_path, device=device)
    method.load_state_dict(_checkpoint_state(ckpt))

    if hasattr(method, "policy"):
        method.policy.to(device)
        method.policy.eval()
    if getattr(method, "value_fn", None) is not None:
        method.value_fn.to(device)
        method.value_fn.eval()

    obs_rms = _restore_obs_rms(ckpt, cfg, env)
    return cfg, env, method, obs_rms


def _choose_action(method, obs: np.ndarray, device: torch.device, policy_mode: PolicyMode):
    obs_t = torch.as_tensor(obs[None, ...], dtype=torch.float32, device=device)
    deterministic = policy_mode == "deterministic"
    with torch.no_grad():
        action_t, _logp_t, dist = method.act(obs_t, deterministic=deterministic)
    action_np = action_t.squeeze(0).detach().cpu().numpy()

    chosen_prob = None
    entropy = None
    if hasattr(dist, "probs"):
        action_int = int(action_np)
        probs = dist.probs.squeeze(0).detach().cpu().numpy()
        chosen_prob = float(probs[action_int])
        entropy = float(dist.entropy().squeeze(0).detach().cpu().numpy())
    return action_np, chosen_prob, entropy


def _render_rgb(env) -> np.ndarray | None:
    frame = env.render()
    if frame is None and hasattr(env, "unwrapped"):
        frame = env.unwrapped.render()
    if frame is None:
        return None
    frame = np.asarray(frame)
    if frame.ndim == 2:
        frame = np.repeat(frame[..., None], 3, axis=-1)
    if frame.shape[-1] == 4:
        frame = frame[..., :3]
    return frame.astype(np.uint8)


def _decorate_frame(
    frame: np.ndarray,
    *,
    title: str,
    episode: int,
    step: int,
    score: float,
    action: int | str,
    action_prob: float | None,
    policy_mode: str,
    scale: int,
) -> np.ndarray:
    try:
        from PIL import Image, ImageDraw
    except ImportError as exc:
        raise ImportError("Pillow is required for video overlays. Install with: pip install Pillow") from exc

    image = Image.fromarray(frame)
    if scale > 1:
        image = image.resize((image.width * scale, image.height * scale), resample=Image.Resampling.NEAREST)

    bar_h = max(34, 17 * scale)
    canvas = Image.new("RGB", (image.width, image.height + bar_h), (10, 10, 10))
    canvas.paste(image, (0, bar_h))
    draw = ImageDraw.Draw(canvas)

    prob_txt = "" if action_prob is None else f" | p(a)={action_prob:.2f}"
    line1 = f"{title} | {policy_mode} rollout"
    line2 = f"episode={episode} | step={step} | score={score:.0f} | action={action}{prob_txt}"
    draw.text((8, 4), line1, fill=(245, 245, 245))
    draw.text((8, 18), line2, fill=(215, 215, 215))
    return np.asarray(canvas, dtype=np.uint8)


def _write_mp4(frames: list[np.ndarray], path: Path, fps: int) -> None:
    try:
        import imageio.v2 as imageio
    except ImportError as exc:
        raise ImportError("imageio[ffmpeg] is required for MP4 writing. Install with: pip install 'imageio[ffmpeg]'") from exc

    path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(path, fps=fps, codec="libx264", quality=8, macro_block_size=16) as writer:
        for frame in frames:
            writer.append_data(frame)


def _write_summaries(output_dir: Path, episodes: list[EpisodeSummary], run_summary: VideoRunSummary) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = Path(run_summary.csv_path)
    json_path = Path(run_summary.json_path)

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["episode", "seed", "return", "length", "video_path"])
        writer.writeheader()
        for ep in episodes:
            row = asdict(ep)
            row["return"] = row.pop("return_")
            writer.writerow(row)

    payload = asdict(run_summary)
    payload["episodes_detail"] = [asdict(ep) | {"return": ep.return_} for ep in episodes]
    for item in payload["episodes_detail"]:
        item.pop("return_", None)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def render_policy_video(
    *,
    config_path: str | Path,
    checkpoint_path: str | Path,
    output_dir: str | Path = "outputs/videos/qbert_demo",
    device: str = "cpu",
    seed: int = 0,
    episodes: int = 3,
    policy_mode: PolicyMode = "deterministic",
    record_selection: RecordSelection = "best",
    fps: int = 30,
    scale: int = 3,
    max_steps: int | None = None,
    title: str | None = None,
) -> VideoRunSummary:
    """Run a trained Atari policy, save the selected rollout as MP4, and log evaluation stats.

    Deterministic mode takes argmax_a pi(a|s). Stochastic mode samples a ~ pi(.|s).
    Both modes produce one real trajectory through the environment; no backtracking or state reset
    happens inside an episode.
    """
    if policy_mode not in {"deterministic", "stochastic"}:
        raise ValueError("policy_mode must be 'deterministic' or 'stochastic'.")
    if record_selection not in {"best", "first"}:
        raise ValueError("record_selection must be 'best' or 'first'.")

    output_dir = Path(output_dir).expanduser()
    device_t = torch.device(device)
    _seed_rollout(int(seed), device_t)
    cfg, env, method, obs_rms = load_policy_for_video(config_path, checkpoint_path, device_t, seed=seed)

    title = title or f"{method.name.upper()} {method.variant} on {cfg.env.id}"
    max_ep_len = int(max_steps or cfg.train.get("max_ep_len", 10000))

    selected_frames: list[np.ndarray] | None = None
    selected_summary: EpisodeSummary | None = None
    episode_summaries: list[EpisodeSummary] = []

    for ep_idx in range(int(episodes)):
        episode_seed = int(seed) + ep_idx
        obs, _info = env.reset(seed=episode_seed)
        obs = _preprocess_obs(obs, obs_rms)
        ep_ret = 0.0
        ep_len = 0
        done = False
        frames: list[np.ndarray] = []

        frame = _render_rgb(env)
        if frame is not None:
            frames.append(
                _decorate_frame(
                    frame,
                    title=title,
                    episode=ep_idx,
                    step=0,
                    score=0.0,
                    action="reset",
                    action_prob=None,
                    policy_mode=policy_mode,
                    scale=scale,
                )
            )

        while not done:
            action_np, action_prob, _entropy = _choose_action(method, obs, device_t, policy_mode)
            env_action = int(action_np) if hasattr(env.action_space, "n") else action_np
            obs, reward, terminated, truncated, _info = env.step(env_action)
            obs = _preprocess_obs(obs, obs_rms)
            ep_ret += float(reward)
            ep_len += 1
            done = bool(terminated or truncated or ep_len >= max_ep_len)

            frame = _render_rgb(env)
            if frame is not None:
                frames.append(
                    _decorate_frame(
                        frame,
                        title=title,
                        episode=ep_idx,
                        step=ep_len,
                        score=ep_ret,
                        action=env_action,
                        action_prob=action_prob,
                        policy_mode=policy_mode,
                        scale=scale,
                    )
                )

        ep_summary = EpisodeSummary(episode=ep_idx, seed=episode_seed, return_=ep_ret, length=ep_len)
        episode_summaries.append(ep_summary)

        should_select = False
        if selected_summary is None:
            should_select = True
        elif record_selection == "best" and ep_ret > selected_summary.return_:
            should_select = True

        if should_select:
            selected_summary = ep_summary
            selected_frames = frames

        if record_selection == "first":
            break

    if selected_summary is None or selected_frames is None:
        raise RuntimeError("No episode was recorded. Check that the Atari env supports rgb_array rendering.")

    output_dir.mkdir(parents=True, exist_ok=True)
    base_stem = _artifact_stem(
        cfg=cfg,
        method=method,
        checkpoint_path=checkpoint_path,
        policy_mode=policy_mode,
        seed=seed,
        record_selection=record_selection,
        episodes=len(episode_summaries),
    )
    artifact_stem = _next_available_artifact_stem(output_dir, base_stem)
    video_path = output_dir / f"{artifact_stem}__selected-ep-{selected_summary.episode}.mp4"
    _write_mp4(selected_frames, video_path, fps=int(fps))
    selected_summary.video_path = str(video_path)

    returns = [ep.return_ for ep in episode_summaries]
    lengths = [ep.length for ep in episode_summaries]
    csv_path = output_dir / f"{artifact_stem}__episodes.csv"
    json_path = output_dir / f"{artifact_stem}__summary.json"
    run_summary = VideoRunSummary(
        config_path=str(config_path),
        checkpoint_path=str(checkpoint_path),
        output_dir=str(output_dir),
        run_name=str(cfg.train.get("run_name", "atari_policy")),
        environment=str(cfg.env.id),
        method=str(method.name),
        variant=str(method.variant),
        checkpoint_name=Path(checkpoint_path).name,
        policy_mode=policy_mode,
        record_selection=record_selection,
        seed=int(seed),
        episodes=len(episode_summaries),
        selected_episode=selected_summary.episode,
        selected_return=selected_summary.return_,
        selected_length=selected_summary.length,
        return_mean=float(statistics.mean(returns)),
        return_std=float(statistics.pstdev(returns)) if len(returns) > 1 else 0.0,
        length_mean=float(statistics.mean(lengths)),
        artifact_stem=artifact_stem,
        video_path=str(video_path),
        json_path=str(json_path),
        csv_path=str(csv_path),
    )
    _write_summaries(output_dir, episode_summaries, run_summary)
    env.close()
    return run_summary
