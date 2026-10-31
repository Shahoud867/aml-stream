"""Job functions run inside child processes by test_isolate.py. Not collected as tests: the file name does not start with test_."""
import os
import signal

from amlh.jobs_gbt import Paths
from amlh.runner import Interrupted


def ok(ctx, cfg):
    ctx.log({"event": "hello"})
    return {"echo": cfg["x"], "raw": sorted(Paths.raw), "has_jobs_dir": ctx.runner.jobs_dir.is_dir(), "pid": os.getpid()}


def out_of_time(ctx, cfg):
    raise Interrupted("no time")


def crash(ctx, cfg):
    raise ValueError("a very specific failure")


def silent(ctx, cfg):
    os.kill(os.getpid(), signal.SIGKILL) if hasattr(signal, "SIGKILL") else os._exit(9)


def count_with_checkpoints(ctx, cfg):
    """Counts to 4, saving a checkpoint after each step and checking for a stop before each, so a forced kill lands between two steps."""
    saved = ctx.load_checkpoint(lambda p: int(p.read_text()))
    n = saved or 0
    while n < 4:
        ctx.check()
        n += 1
        ctx.save_checkpoint(lambda p, v=n: p.write_text(str(v)))
    return {"count": n, "resumed_from": saved or 0}
