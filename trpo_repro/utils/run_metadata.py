import os
import platform
import shlex
import socket
import subprocess
import sys
import time
import traceback
import uuid
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Sequence

from trpo_repro.utils.utils import find_git_root, write_json


PACKAGE_VERSION_NAMES = [
    "trpo-repro",
    "torch",
    "numpy",
    "gymnasium",
    "ale-py",
    "mujoco",
    "PyYAML",
    "matplotlib",
    "pandas",
    "tqdm",
]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def monotonic_time() -> float:
    return time.monotonic()


def format_command(argv: Sequence[str]) -> str:
    return " ".join(shlex.quote(str(arg)) for arg in argv)


def build_launch_metadata(
    *,
    argv: Sequence[str],
    cli_args: dict[str, Any],
    cli_overrides: dict[str, Any],
    config_path: str | Path,
    output_dir: str | Path,
    package_path: str,
) -> dict[str, Any]:
    return {
        "argv": [str(arg) for arg in argv],
        "command": format_command(argv),
        "cwd": str(Path.cwd().resolve()),
        "config_path": str(Path(config_path).resolve()),
        "output_dir": str(Path(output_dir).resolve()),
        "cli_args": cli_args,
        "cli_overrides": cli_overrides,
        "package_path": package_path,
    }


def collect_package_versions(package_names: Sequence[str] = PACKAGE_VERSION_NAMES) -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package_name in package_names:
        try:
            versions[package_name] = metadata.version(package_name)
        except metadata.PackageNotFoundError:
            versions[package_name] = None
    return versions


def collect_environment_metadata(device: str | None = None) -> dict[str, Any]:
    env_vars = {
        key: os.environ.get(key)
        for key in ["CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "MKL_NUM_THREADS"]
        if os.environ.get(key) is not None
    }
    cuda: dict[str, Any] = {"available": False}
    try:
        import torch

        cuda_available = bool(torch.cuda.is_available())
        cuda = {
            "available": cuda_available,
            "device_count": int(torch.cuda.device_count()) if cuda_available else 0,
            "current_device": int(torch.cuda.current_device()) if cuda_available else None,
            "device_name": torch.cuda.get_device_name(0) if cuda_available else None,
            "torch_cuda_version": getattr(torch.version, "cuda", None),
        }
    except Exception as exc:
        cuda = {
            "available": False,
            "error": repr(exc),
        }

    return {
        "captured_at": utc_now_iso(),
        "python": {
            "version": sys.version,
            "executable": sys.executable,
            "implementation": platform.python_implementation(),
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "platform": platform.platform(),
        },
        "host": {
            "hostname": socket.gethostname(),
            "cpu_count": os.cpu_count(),
        },
        "requested_device": device,
        "cuda": cuda,
        "package_versions": collect_package_versions(),
        "environment_variables": env_vars,
    }


def _run_git(git_root: Path, args: Sequence[str]) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(git_root), *args],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    return result.stdout.strip()


def collect_git_metadata(start: str | Path | None = None) -> dict[str, Any]:
    git_root = find_git_root(start)
    if git_root is None:
        return {
            "available": False,
            "root": None,
            "commit": None,
            "branch": None,
            "dirty": None,
            "status_short": None,
        }

    status = _run_git(git_root, ["status", "--short"])
    return {
        "available": True,
        "root": str(git_root),
        "commit": _run_git(git_root, ["rev-parse", "HEAD"]),
        "branch": _run_git(git_root, ["branch", "--show-current"]),
        "dirty": bool(status),
        "status_short": status or "",
    }


def write_git_diff_snapshot(output_dir: str | Path, start: str | Path | None = None) -> str | None:
    git_root = find_git_root(start)
    if git_root is None:
        return None
    try:
        result = subprocess.run(
            ["git", "-C", str(git_root), "diff", "--no-ext-diff", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    if not result.stdout:
        return None
    diff_path = Path(output_dir) / "git_diff.patch"
    diff_path.write_text(result.stdout, encoding="utf-8")
    return str(diff_path.resolve())


def build_base_run_metadata(
    *,
    launch: dict[str, Any],
    environment: dict[str, Any],
    git: dict[str, Any],
    method: str,
    method_variant: str,
    estimator: str | None,
    env_id: str,
    suite: str,
    seed: int,
    run_name: str,
    trainable: bool,
    runtime: dict[str, Any],
) -> dict[str, Any]:
    started_at = utc_now_iso()
    return {
        "schema_version": 1,
        "run_id": uuid.uuid4().hex,
        "status": "running",
        "created_at": started_at,
        "started_at": started_at,
        "ended_at": None,
        "duration_sec": None,
        "method": method,
        "method_variant": method_variant,
        "estimator": estimator,
        "env_id": env_id,
        "suite": suite,
        "seed": seed,
        "run_name": run_name,
        "trainable": trainable,
        "launch": launch,
        "runtime": runtime,
        "environment": environment,
        "git": git,
        "git_commit_hash": git.get("commit"),
    }


def build_failure_record(exc: BaseException, *, epoch: int | None = None) -> dict[str, Any]:
    return {
        "failed_at": utc_now_iso(),
        "epoch": epoch,
        "exception_type": type(exc).__name__,
        "exception_message": str(exc),
        "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
    }


def write_run_summary(output_dir: str | Path, summary: dict[str, Any]) -> Path:
    return write_json(summary, Path(output_dir) / "run_summary.json")


def write_failure(output_dir: str | Path, failure: dict[str, Any]) -> Path:
    return write_json(failure, Path(output_dir) / "failure.json")
