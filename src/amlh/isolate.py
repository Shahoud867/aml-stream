"""Runs a job function in its own process, so its memory and GPU memory go back to the system when it ends.

Python does not return the memory of freed objects to the system, and the official Multi-GNN and FraudGT preprocessing builds millions of them.
In one process that memory piles up across jobs until the system kills something (exit -9). A job that fails or finishes in a child
leaves the notebook's own process small. The child gets the same Context, so checkpoints, logs and stop checks behave as before.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from importlib import import_module
from pathlib import Path

from .runner import Context, Interrupted, JobSpec, atomic_write_json, read_json

OUT_OF_TIME = 75
FORCED_KILL = 137        # the pilot's forced kill: exit with no cleanup, like an abrupt end of the session
RESULT_FILE = "isolated_result.json"
POLL_SECONDS = 5


class ChildRunner:
    """What Context needs from a Runner. The parent process owns the ledger, the mirror and the job's status."""

    def __init__(self, deadline: float, jobs_dir: Path | None = None):
        self.deadline = deadline
        self.jobs_dir = Path(jobs_dir) if jobs_dir else None
        self.kill_after = int(os.environ.get("AMLH_KILL_AFTER_CHECKS", "0"))
        self.n_checks = 0
        self.stop_requested = False

    def remaining(self) -> float:
        return self.deadline - time.time()

    def heartbeat(self, job_id: str):
        pass

    def after_checkpoint(self):
        pass


def ram_gb() -> float | None:
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal"):
                return int(line.split()[1]) / 1024 ** 2
    except OSError:
        pass
    return None


def explain_exit(code: int) -> str:
    """Exit -9 is the system's kill signal, which on a small machine almost always means it ran out of memory."""
    if code == -9:
        ram = ram_gb()
        return f"exit -9: the system killed the process, most likely for lack of memory ({ram:.1f} GB of RAM on this machine)" if ram else "exit -9: the system killed the process, most likely for lack of memory"
    return f"exit {code}"


def run_isolated(ctx: Context, cfg: dict, module: str, name: str, extra_path: str | None = None, poll_seconds: float = POLL_SECONDS) -> dict:
    """Calls module.name(ctx, cfg) in a new process and returns its metrics. Out of time becomes Interrupted, a crash becomes an error with its text."""
    from .fraudgt import FraudgtPaths
    from .jobs_gbt import Paths
    from .mgnn import MgnnPaths
    spec = {"module": module, "name": name, "cfg": cfg, "job_id": ctx.spec.job_id, "job_dir": str(ctx.dir.resolve()), "jobs_dir": str(ctx.runner.jobs_dir.resolve()),
            "deadline": time.time() + ctx.runner.remaining(), "raw": {k: str(v) for k, v in Paths.raw.items()}, "cache": str(Path(Paths.cache).resolve()),
            "mgnn_code": str(Path(MgnnPaths.code).resolve()), "mgnn_work": str(Path(MgnnPaths.work).resolve()),
            "fraudgt_code": str(Path(FraudgtPaths.code).resolve()), "fraudgt_work": str(Path(FraudgtPaths.work).resolve())}
    ctx.log({"event": "isolated_start", "function": f"{module}.{name}", "ram_gb": ram_gb()})
    scratch = Path(tempfile.mkdtemp(prefix="amlh_iso_"))
    spec_path, err_path = scratch / "spec.json", scratch / "stderr.txt"
    atomic_write_json(spec_path, spec)
    result_path = ctx.dir / RESULT_FILE
    if result_path.exists():
        result_path.unlink()
    env = {**os.environ, "AMLH_ISOLATE": str(spec_path),
           "PYTHONPATH": os.pathsep.join(filter(None, [str(Path(__file__).resolve().parents[1]), extra_path or "", os.environ.get("PYTHONPATH", "")]))}
    with open(err_path, "w") as err:
        proc = subprocess.Popen([sys.executable, "-c", "from amlh.isolate import child_main; child_main()"], env=env, stdout=subprocess.DEVNULL, stderr=err)
        while proc.poll() is None:
            time.sleep(poll_seconds)
            ctx.runner.heartbeat(ctx.spec.job_id)
            ctx.runner.after_checkpoint()
            if ctx.runner.stop_requested:
                proc.terminate()
    code = proc.returncode
    if code == FORCED_KILL and os.environ.get("AMLH_KILL_AFTER_CHECKS"):
        os._exit(FORCED_KILL)  # the whole session ends abruptly, as the pilot's forced kill intends
    if code == OUT_OF_TIME:
        raise Interrupted(f"{name} stopped for lack of session time")
    if code != 0:
        raise RuntimeError(f"{name} failed with {explain_exit(code)}. Its last output:\n{err_path.read_text()[-1500:]}")
    result = read_json(result_path)
    if result is None:
        raise RuntimeError(f"{name} exited cleanly without writing a result")
    return result


def child_main() -> None:
    """Entry point of the child process."""
    spec = json.loads(Path(os.environ["AMLH_ISOLATE"]).read_text())
    from .fraudgt import FraudgtPaths
    from .jobs_gbt import Paths
    from .mgnn import MgnnPaths
    Paths.raw = {k: Path(v) for k, v in spec["raw"].items()}
    Paths.cache = Path(spec["cache"])
    MgnnPaths.code, MgnnPaths.work = Path(spec["mgnn_code"]), Path(spec["mgnn_work"])
    FraudgtPaths.code, FraudgtPaths.work = Path(spec["fraudgt_code"]), Path(spec["fraudgt_work"])
    runner = ChildRunner(spec["deadline"], Path(spec["jobs_dir"]))

    def stop(signum, frame):
        runner.stop_requested = True
    signal.signal(signal.SIGTERM, stop)
    ctx = Context(Path(spec["job_dir"]), runner, JobSpec(spec["job_id"], spec["name"], spec["cfg"]))
    fn = getattr(import_module(spec["module"]), spec["name"])
    try:
        result = fn(ctx, spec["cfg"]) or {}
    except Interrupted:
        sys.exit(OUT_OF_TIME)
    try:
        import resource
        result = {**result, "isolated_peak_rss_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2, 2)}
    except ImportError:
        pass
    atomic_write_json(Path(spec["job_dir"]) / RESULT_FILE, result)
