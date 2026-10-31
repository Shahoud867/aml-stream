"""Running a job in its own process: results, stops, crashes, kills and resume. The functions run in the child live in isolated_jobs.py."""
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

import amlh.isolate as iso
from amlh.jobs_gbt import Paths
from amlh.runner import Interrupted, JobSpec, Runner, verify_job
from test_notebooks import restore_globals  # noqa: F401 (autouse fixture)

TESTS = str(Path(__file__).resolve().parent)
KW = {"extra_path": TESTS, "poll_seconds": 0.05}


class _Spec:
    job_id = "iso-job"


def _ctx(tmp_path):
    from amlh.runner import Context
    runner = Runner(tmp_path / "run", "NB1", 3600.0, code_version="x")
    (runner.jobs_dir / "iso-job").mkdir(parents=True)
    return Context(runner.jobs_dir / "iso-job", runner, JobSpec("iso-job", "f", {}))


def test_the_job_runs_in_another_process_and_returns_its_result(tmp_path, monkeypatch):
    monkeypatch.setattr(Paths, "raw", {"HI-Small": tmp_path})
    out = iso.run_isolated(_ctx(tmp_path), {"x": 7}, "isolated_jobs", "ok", **KW)
    assert out["echo"] == 7 and out["raw"] == ["HI-Small"] and out["has_jobs_dir"] and out["pid"] != os.getpid()
    assert out.get("isolated_peak_rss_gb", 1.0) > 0  # absent on Windows, which has no resource module
    assert (tmp_path / "run/jobs/iso-job/log.jsonl").read_text().count("hello") == 1


def test_running_out_of_time_becomes_an_interruption(tmp_path):
    with pytest.raises(Interrupted):
        iso.run_isolated(_ctx(tmp_path), {}, "isolated_jobs", "out_of_time", **KW)


def test_a_crash_reports_the_childs_error_text(tmp_path):
    with pytest.raises(RuntimeError, match="a very specific failure"):
        iso.run_isolated(_ctx(tmp_path), {}, "isolated_jobs", "crash", **KW)


@pytest.mark.skipif(sys.platform == "win32", reason="a kill signal is a POSIX notion")
def test_a_system_kill_is_explained_as_a_likely_out_of_memory(tmp_path):
    with pytest.raises(RuntimeError, match="killed the process, most likely for lack of memory"):
        iso.run_isolated(_ctx(tmp_path), {}, "isolated_jobs", "silent", **KW)


def test_explain_exit_only_blames_memory_for_the_kill_signal():
    assert "lack of memory" in iso.explain_exit(-9) and "lack of memory" not in iso.explain_exit(1)


def test_a_stop_request_ends_the_child(tmp_path):
    ctx = _ctx(tmp_path)
    ctx.runner.stop_requested = True
    with pytest.raises((Interrupted, RuntimeError)):  # terminated before it finishes: no result is ever accepted
        iso.run_isolated(ctx, {}, "isolated_jobs", "count_with_checkpoints", **KW)


SCRIPT = textwrap.dedent('''
    import sys
    sys.path.insert(0, {tests!r})
    from pathlib import Path
    from amlh.runner import JobSpec, Runner
    from amlh.isolate import run_isolated
    r = Runner(Path({out!r}), "NB1", 3600.0, code_version="x")
    spec = JobSpec("iso-count", lambda ctx, cfg: run_isolated(ctx, cfg, "isolated_jobs", "count_with_checkpoints", extra_path={tests!r}, poll_seconds=0.05), {{}})
    summary = r.run([spec], ("P0",))
    sys.exit(0 if summary["complete"] else 3)
''')


def test_a_forced_kill_ends_the_whole_session_and_the_next_run_resumes(tmp_path):
    """What the pilot does: the child dies with no cleanup, the notebook's process dies with it, and a new process continues from the checkpoint."""
    script = tmp_path / "run.py"
    script.write_text(SCRIPT.format(tests=TESTS, out=str(tmp_path / "out")))
    env = {k: v for k, v in os.environ.items() if k != "AMLH_KILL_AFTER_CHECKS"}
    env["PYTHONPATH"] = str(Path(iso.__file__).resolve().parents[1])
    first = subprocess.run([sys.executable, str(script)], env={**env, "AMLH_KILL_AFTER_CHECKS": "3"}, capture_output=True, text=True)
    assert first.returncode == 137, first.stderr[-800:]
    job = tmp_path / "out/jobs/iso-count"
    assert (job / "ckpt/latest.bin").read_text() == "2", "two steps were saved before the kill at the third check"
    assert not verify_job(job)[0]
    second = subprocess.run([sys.executable, str(script)], env=env, capture_output=True, text=True)
    assert second.returncode == 0, second.stderr[-800:]
    assert verify_job(job)[0]
    import json
    assert json.loads((job / "metrics.json").read_text()) | {"isolated_peak_rss_gb": 0} == {"count": 4, "resumed_from": 2, "isolated_peak_rss_gb": 0}


def test_the_multi_gnn_jobs_are_registered_as_isolated():
    import amlh.mgnn as mg
    assert set(mg.REGISTRY) == {"mgnn_train", "mgnn_replay", "mgnn_train_norm", "mgnn_format_check"}
    assert all(fn.__name__ == name for name, fn in mg.REGISTRY.items())
    assert mg.mgnn_train is not mg.REGISTRY["mgnn_train"], "the raw function stays importable for the child"
