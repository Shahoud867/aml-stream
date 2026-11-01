import json
import os
import pickle
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from amlh.runner import Interrupted, JobSpec, KaggleDatasetMirror, Runner, config_hash, verify_job  # noqa: E402
from amlh.sealing import SealedError, UnsealLog, guard_test_metrics, is_sealed, score_health  # noqa: E402


def counting_job(calls):
    def fn(ctx, cfg):
        calls.append(cfg["x"])
        (ctx.dir / "out.txt").write_text(str(cfg["x"]))
        return {"value": cfg["x"] * 2}
    return fn


def test_done_job_is_not_repeated(tmp_path):
    calls = []
    fn = counting_job(calls)
    for _ in range(2):
        s = Runner(tmp_path, "t", 100, code_version="c").run([JobSpec("a", fn, {"x": 1})])
    assert calls == [1] and s["complete"]
    assert json.loads((tmp_path / "jobs/a/metrics.json").read_text())["value"] == 2


def test_failure_does_not_stop_queue_and_is_retried(tmp_path):
    attempts = []

    def bad(ctx, cfg):
        attempts.append(1)
        raise RuntimeError("boom")
    calls = []
    s = Runner(tmp_path, "t", 100, code_version="c", retries=3).run([JobSpec("bad", bad, {}), JobSpec("ok", counting_job(calls), {"x": 3})])
    assert calls == [3]
    assert len(attempts) == 3
    st = {r["job_id"]: r["status"] for r in s["jobs"]}
    assert st == {"bad": "FAILED", "ok": "DONE"}
    assert "boom" in json.loads((tmp_path / "jobs/bad/status.json").read_text())["traceback"]
    assert not s["complete"]


def test_interrupt_then_resume_from_checkpoint(tmp_path):
    seen = []

    def fn(ctx, cfg):
        start = ctx.load_checkpoint(lambda p: pickle.loads(p.read_bytes())) or 0
        seen.append(start)
        for epoch in range(start, 5):
            if epoch == 2 and len(seen) == 1:
                raise Interrupted("session ended")
            ctx.save_checkpoint(lambda p, v=epoch + 1: p.write_bytes(pickle.dumps(v)))
        return {"epochs": 5}
    s1 = Runner(tmp_path, "t", 100, code_version="c").run([JobSpec("a", fn, {})])
    assert s1["jobs"][0]["status"] == "INTERRUPTED"
    assert json.loads((tmp_path / "jobs/a/status.json").read_text())["attempts"] == 0
    s2 = Runner(tmp_path, "t", 100, code_version="c").run([JobSpec("a", fn, {})])
    assert s2["complete"] and seen == [0, 2]
    assert not (tmp_path / "jobs/a/ckpt/latest.bin").exists()


def test_corrupt_latest_falls_back_to_previous(tmp_path):
    states = []

    def first(ctx, cfg):
        ctx.save_checkpoint(lambda p: p.write_bytes(pickle.dumps(1)))
        ctx.save_checkpoint(lambda p: p.write_bytes(pickle.dumps(2)))
        raise Interrupted("x")
    Runner(tmp_path, "t", 100, code_version="c").run([JobSpec("a", first, {})])
    with open(tmp_path / "jobs/a/ckpt/latest.bin", "wb") as fh:
        fh.write(b"garbage")

    def second(ctx, cfg):
        states.append(ctx.load_checkpoint(lambda p: pickle.loads(p.read_bytes())))
        return {}
    Runner(tmp_path, "t", 100, code_version="c").run([JobSpec("a", second, {})])
    assert states == [1]


def test_config_or_code_change_blocks_resume(tmp_path):
    def stop(ctx, cfg):
        ctx.save_state({"i": 1})
        raise Interrupted("x")
    Runner(tmp_path / "a", "t", 100, code_version="c").run([JobSpec("j", stop, {"x": 1})])
    s = Runner(tmp_path / "a", "t", 100, code_version="c").run([JobSpec("j", stop, {"x": 2})])
    assert s["jobs"][0]["status"] == "BLOCKED"
    Runner(tmp_path / "b", "t", 100, code_version="c").run([JobSpec("j", stop, {"x": 1})])
    s = Runner(tmp_path / "b", "t", 100, code_version="DIFFERENT").run([JobSpec("j", stop, {"x": 1})])
    assert s["jobs"][0]["status"] == "BLOCKED"


def test_tampered_artefact_fails_verification(tmp_path):
    Runner(tmp_path, "t", 100, code_version="c").run([JobSpec("a", counting_job([]), {"x": 1})])
    d = tmp_path / "jobs/a"
    assert verify_job(d)[0]
    (d / "out.txt").write_text("tampered")
    assert verify_job(d) == (False, "hash mismatch out.txt")
    (d / "out.txt").unlink()
    assert verify_job(d) == (False, "missing out.txt")


def test_requires_defers_until_dependency_done(tmp_path):
    calls = []
    jobs = [JobSpec("late", counting_job(calls), {"x": 2}, requires=["early"]), JobSpec("early", counting_job(calls), {"x": 1})]
    s = Runner(tmp_path, "t", 100, code_version="c").run(jobs)
    assert calls == [1, 2] and s["complete"]


def test_budget_exhausted_runs_nothing_more(tmp_path):
    calls = []
    s = Runner(tmp_path, "t", -1, code_version="c").run([JobSpec("a", counting_job(calls), {"x": 1})])
    assert calls == [] and not s["complete"]


def test_job_too_long_for_session_is_left_for_next(tmp_path):
    calls = []
    r = Runner(tmp_path, "t", 100, code_version="c")
    assert r.run_job(JobSpec("a", counting_job(calls), {"x": 1}, est_seconds=1000)) == "TODO"
    assert calls == []


def test_context_check_raises_when_next_unit_will_not_fit(tmp_path):
    r = Runner(tmp_path, "t", 100, code_version="c")

    def fn(ctx, cfg):
        ctx.check(need_seconds=5000)
    assert r.run_job(JobSpec("a", fn, {})) == "INTERRUPTED"


def test_restore_prefers_verified_and_newest(tmp_path):
    old = tmp_path / "old"
    Runner(old, "t", 100, code_version="c").run([JobSpec("a", counting_job([]), {"x": 1})])
    new = tmp_path / "new"
    r = Runner(new, "t", 100, code_version="c")
    assert r.restore([old]) == 1
    calls = []
    assert r.run([JobSpec("a", counting_job(calls), {"x": 1})])["complete"]
    assert calls == []


def test_manifest_records_no_identity(tmp_path):
    r = Runner(tmp_path, "t", 100, code_version="c")
    r.write_manifest()
    text = (tmp_path / "RUN_MANIFEST.json").read_text().lower()
    for word in ("user", "account", "hostname", "node"):
        assert word not in text
    assert os.environ.get("USERNAME", "zzzzzz").lower() not in text


def test_mirror_failure_is_not_fatal(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "a.txt").write_text("x")
    m = KaggleDatasetMirror(run, "me/x", command=("definitely-not-a-command",))
    assert m.maybe_sync(force=True) is False
    assert m.failures == 1
    assert m.maybe_sync() is False  # throttled, so no second attempt inside the interval
    assert m.failures == 1


def test_config_hash_ignores_key_order():
    assert config_hash({"a": 1, "b": 2}) == config_hash({"b": 2, "a": 1})


def test_sealing():
    assert is_sealed("LI-Small") and not is_sealed("HI-Small")
    with pytest.raises(SealedError):
        guard_test_metrics("HI-Medium")
    guard_test_metrics("HI-Small")
    guard_test_metrics("HI-Medium", unsealed=True)
    assert score_health([0.1, 0.2])["n"] == 2


def test_unseal_log_labels_reruns(tmp_path):
    log = UnsealLog(tmp_path / "u.jsonl")
    assert log.record("LI-Small", "c1", 3, True) == "first"
    assert log.record("LI-Small", "c1", 3, True) == "repeat"
    assert log.record("LI-Small", "c2", 3, True) == "exploratory"
