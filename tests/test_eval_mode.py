"""Multi-GNN's official code evaluates in training mode, because it never calls model.eval(). The audit keeps that as the primary
behaviour and adds a deterministic evaluation-mode set of scores. The official code is not run here."""
import json

import numpy as np
import polars as pl
import pytest

import amlh.session as sess
from amlh.jobs_gbt import Paths, get_stream
from amlh.merge import Collected, official_audit
from amlh.mgnn import official_mode
from test_notebooks import restore_globals  # noqa: F401 (autouse fixture)


def test_official_behaviour_is_the_default_and_can_be_switched_off():
    assert official_mode({}) and official_mode({"eval_mode": "official"}) and not official_mode({"eval_mode": "eval"})


def test_the_official_code_really_never_switches_mode():
    """The reason for the whole design. Skipped if the read-only copy of the official code is absent."""
    from pathlib import Path
    root = Path(__file__).resolve().parents[1] / "third_party" / "Multi-GNN"
    if not root.exists():
        pytest.skip("official code not present")
    source = "".join(p.read_text(errors="ignore") for p in root.rglob("*.py"))
    assert ".eval()" not in source and ".train()" not in source


def test_resume_check_compares_the_deterministic_scores_when_there_are_two_sets(tmp_path):
    ref, other = tmp_path / "ref", tmp_path / "other"
    for d in (ref, other):
        d.mkdir()
    ids = list(range(50))
    rng = np.random.default_rng(0)
    eval_scores = rng.uniform(size=50)
    for d, noise in ((ref, 0.0), (other, 0.6)):  # the dropout-noisy files differ a lot, the evaluation-mode files do not
        pl.DataFrame({"event_id": ids, "score": eval_scores}).write_parquet(d / "scores_test_eval.parquet")
        pl.DataFrame({"event_id": ids, "score": np.clip(eval_scores + noise * rng.normal(size=50), 0, 1)}).write_parquet(d / "scores_test.parquet")
    assert min(v["correlation"] for v in sess.compare_scores(ref, other).values()) < 0.95
    only_eval = sess.compare_scores(ref, other, "*_eval.parquet")
    assert list(only_eval) == ["scores_test_eval.parquet"] and only_eval["scores_test_eval.parquet"]["correlation"] == pytest.approx(1.0)


def test_official_audit_reports_both_score_sets_and_only_the_official_one_gets_a_published_reference(small_data, tmp_path):
    Paths.raw = {"HI-Small": small_data / "raw"}
    Paths.cache = small_data / "processed"
    try:
        s, _ = get_stream({"dataset": "HI-Small", "protocol": "released"})
    except Exception:
        pytest.skip("fixture layout differs")
    d = tmp_path / "nb1-mgnn-gin-HI-Small-s1"
    d.mkdir()
    (d / "config.json").write_text(json.dumps({"cfg": {"model": "gin"}}))
    rng = np.random.default_rng(1)
    for suffix in ("", "_eval"):
        for name, rows in ((f"scores_val{suffix}", s.va), (f"scores_test{suffix}", s.te)):
            score = np.clip(s.y[rows] * 0.8 + rng.uniform(0, 0.2, len(rows)), 0, 1).astype(np.float32)
            pl.DataFrame({"event_id": s.event_id[rows], "score": score, "sampled": np.ones(len(rows), dtype=bool)}).write_parquet(d / f"{name}.parquet")
    out = official_audit(Collected(jobs={d.name: d}), s)
    assert set(out) == {d.name, d.name + " [eval mode]"}
    assert out[d.name]["published_reference"] == 0.6479 and out[d.name + " [eval mode]"]["published_reference"] is None
