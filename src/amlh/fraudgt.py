"""FraudGT region audit: its published code and configuration, run from its own repository. NOT YET EXECUTED.

FraudGT's process and global configuration stay apart from everything else, so training runs in fraudgt_driver.py as a
child process. This job starts it, passes on stop requests and turns its exit code into the runner's job states.
Which published model name each repository configuration corresponds to is not stated in the repository.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from .fraudgt_driver import OUT_OF_TIME
from .isolate import explain_exit
from .runner import Context, Interrupted, atomic_write_json, read_json

NAME = {"HI-Small": "Small-HI", "LI-Small": "Small-LI", "HI-Medium": "Medium-HI", "LI-Medium": "Medium-LI"}
POLL_SECONDS = 20


class FraudgtPaths:
    code: Path = Path("third_party/FraudGT")
    work: Path = Path("cache/fraudgt")


def driver_command(spec_path: Path) -> list[str]:
    return [sys.executable, "-m", "amlh.fraudgt_driver", str(spec_path)]


def overrides(cfg: dict, data_dir: Path, log_dir: Path) -> list[str]:
    """Changes only where files go, how many CPU workers this machine has and what gets logged. The experiment stays as published."""
    out = ["out_dir", str(log_dir), "dataset.dir", str(data_dir), "seed", str(cfg["seed"]), "wandb.use", "False", "train.auto_resume", "True",
           "num_workers", str(cfg.get("num_workers", 2)), "num_threads", str(cfg.get("num_threads", 4)), "train.tqdm", "False", "val.tqdm", "False"]
    for key, value in cfg.get("opts", {}).items():  # pilot only: a short run to measure time
        out += [key, str(value)]
    return out


def aml_root(data_dir: Path) -> Path:
    """FraudGT's loader appends the dataset format to dataset.dir, so the raw file and the formatted file live in dir/AML."""
    return Path(data_dir) / "AML"


def link_raw(raw_csv: Path, data_dir: Path) -> None:
    """FraudGT reads the raw file from its dataset root and writes beside it, and inputs are read-only. Links it, or copies where links are not allowed."""
    root = aml_root(data_dir)
    root.mkdir(parents=True, exist_ok=True)
    link = root / raw_csv.name
    if link.exists():
        return
    try:
        os.symlink(Path(raw_csv).resolve(), link)
    except OSError:
        shutil.copyfile(raw_csv, link)


def fraudgt_train(ctx: Context, cfg: dict) -> dict:
    from .jobs_gbt import Paths
    ds = cfg["dataset"]
    raw_csv = Path(Paths.raw[ds]) / f"{ds}_Trans.csv"
    work = Path(FraudgtPaths.work).resolve()
    data_dir, log_dir = work / ds, work / "logs" / ctx.spec.job_id
    link_raw(raw_csv, data_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    spec = {"job_id": ctx.spec.job_id, "job_dir": str(ctx.dir.resolve()), "code": str(Path(FraudgtPaths.code).resolve()), "config": f"configs/{cfg['config']}",
            "config_name": cfg["config"], "dataset": ds, "data_dir": str(data_dir), "overrides": overrides(cfg, data_dir, log_dir),
            "max_train_hours": cfg.get("max_train_hours", 20), "score_limit": cfg.get("score_limit"), "deadline": time.time() + ctx.runner.remaining()}
    spec_path = log_dir / "spec.json"
    atomic_write_json(spec_path, spec)
    env = {**os.environ, "WANDB_MODE": "disabled", "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1",
           "PYTHONPATH": str(Path(__file__).resolve().parents[1]) + os.pathsep + os.environ.get("PYTHONPATH", "")}
    stderr_path = log_dir / "stderr.txt"
    with open(stderr_path, "w") as err:
        proc = subprocess.Popen(driver_command(spec_path), env=env, stdout=subprocess.DEVNULL, stderr=err)
        while proc.poll() is None:
            time.sleep(cfg.get("poll_seconds", POLL_SECONDS))
            ctx.runner.heartbeat(ctx.spec.job_id)
            ctx.runner.after_checkpoint()
            if ctx.runner.stop_requested:
                proc.terminate()
    if proc.returncode == OUT_OF_TIME:
        raise Interrupted("FraudGT stopped for lack of session time")
    if proc.returncode != 0:
        raise RuntimeError(f"FraudGT failed with {explain_exit(proc.returncode)}. Its last output:\n{stderr_path.read_text()[-1500:]}")
    result = read_json(ctx.dir / "result.json")
    if result is None:
        raise RuntimeError("FraudGT exited cleanly without writing a result")
    return result


REGISTRY = {"fraudgt_train": fraudgt_train}
