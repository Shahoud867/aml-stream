"""End-to-end on a tiny dataset: preparation, jobs, resume, sealing and the merge step."""
import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from amlh.datasets import attempts_in_window, load_stream
from amlh.evaluate import equal_alert_threshold, heldout_metrics, heldout_outcomes
from amlh.jobs_gbt import Paths, get_stream, read_scores
from amlh.merge import collect, merge, plan_status
from amlh.plans import PLANS, plan
from amlh.prepare import ensure_prepared, multignn_split_days, prepare_saml
from amlh.runner import JobSpec, Runner, verify_job
from amlh.sealing import SealedError

GBT = {"dataset": "HI-Small", "protocol": "released", "family": "wall", "neg_rate": 0.25}


def runner(path, name="NB4"):
    from amlh.plans import registry
    return Runner(path, name, 3600, code_version="test", registry=registry())


def test_prepared_tables(small_data):
    p = ensure_prepared("HI-Small", Paths.raw["HI-Small"], Paths.cache)
    card = json.loads((p / "data_card.json").read_text())
    ev = pl.read_parquet(p / "events.parquet")
    assert ev["t"].is_sorted()
    assert card["pattern_lines_unmatched"] == 0
    assert card["n_illicit"] == card["illicit_events_in_patterns"]
    assert ev.filter(pl.col("label") == 1).height == pl.read_parquet(p / "pattern_members.parquet").height
    s = load_stream(Paths.cache, "HI-Small")
    assert s.tr.max() < s.va.min() < s.te.min()


def test_split_rule_matches_slow_version():
    rng = np.random.default_rng(0)
    for _ in range(20):
        daily = rng.integers(1, 1000, rng.integers(5, 25))
        best, best_score = None, np.inf
        for i in range(1, len(daily)):
            for j in range(i + 1, len(daily)):
                totals = [daily[:i].sum(), daily[i:j].sum(), daily[j:].sum()]
                total = sum(totals)
                score = max(abs(v / total - p) / p for v, p in zip(totals, (0.6, 0.2, 0.2)))
                if score < best_score:
                    best, best_score = (i, j), score
        assert multignn_split_days(daily) == best


def test_protocols_and_folds(small_data):
    released = load_stream(Paths.cache, "HI-Small")
    documented = load_stream(Paths.cache, "HI-Small", "documented")
    assert documented.n <= released.n and documented.t.max() < 10 * 86400
    assert np.array_equal(released.t[:documented.n], documented.t)
    fold = load_stream(Paths.cache, "HI-Small", "documented", fold=(4, 5, 7))
    assert fold.t.max() < 7 * 86400 and fold.tr.max() < fold.va.min()
    assert fold.core_days == 2


def test_stale_card_cannot_turn_the_tail_into_core(small_data, tmp_path):
    """A missing documented span in an old data card must not silently count tail days as core."""
    import shutil
    ensure_prepared("HI-Small", Paths.raw["HI-Small"], Paths.cache)
    stale = tmp_path / "HI-Small"
    shutil.copytree(Paths.cache / "HI-Small", stale)
    card = json.loads((stale / "data_card.json").read_text())
    del card["documented_days"]
    (stale / "data_card.json").write_text(json.dumps(card))
    s = load_stream(tmp_path, "HI-Small")
    assert s.core_end_t == 10 * 86400


def test_spend_survives_tied_scores():
    from amlh.jobs_gbt import spend
    tied = np.zeros(100)
    assert spend([tied, tied], (0.5, 0.5), 20) == [0.0, float("inf")]


def test_sweep_job_and_resume_is_a_noop(small_data, tmp_path):
    cfg = {**GBT, "H": "3600", "seed": 0}
    r = runner(tmp_path)
    s = r.run([JobSpec("sweep", "gbt_sweep", cfg)])
    assert s["complete"]
    d = tmp_path / "jobs/sweep"
    m = json.loads((d / "metrics.json").read_text())
    assert "test_results" in m and m["test_results"]["test"]["pr_auc"] > 0.3
    before = (d / "DONE").read_text()
    assert runner(tmp_path).run([JobSpec("sweep", "gbt_sweep", cfg)])["complete"]
    assert (d / "DONE").read_text() == before


def test_sealed_dataset_writes_scores_but_no_test_metrics(small_data, tmp_path):
    cfg = {**GBT, "dataset": "LI-Small", "H": "0", "seed": 0}
    assert runner(tmp_path).run([JobSpec("sealed", "gbt_sweep", cfg)])["complete"]
    d = tmp_path / "jobs/sealed"
    text = (d / "metrics.json").read_text()
    m = json.loads(text)
    assert "test_results" not in m and "val_f1" in m and m["test_health"]["n_nan"] == 0
    assert (d / "scores_test.parquet").exists()
    s, _ = get_stream(cfg)
    with pytest.raises(SealedError):
        heldout_metrics(s, np.zeros(len(s.va)), np.zeros(len(s.te)), "0")


def test_strict_past_identical_with_and_without_tail(small_data, tmp_path):
    """Causality: removing future days cannot change a strict-past view."""
    from amlh.design import design
    from amlh.features import build_graph
    a = load_stream(Paths.cache, "HI-Small")
    b = load_stream(Paths.cache, "HI-Small", "documented")
    rows = b.te[:400]
    Xa = design(a, build_graph(a.t, a.src, a.dst, a.usd), rows, 0, a.view_end, a.t)
    Xb = design(b, build_graph(b.t, b.src, b.dst, b.usd), rows, 0, b.view_end, b.t)
    assert np.array_equal(Xa, Xb)


def test_families_have_matching_columns(small_data):
    from amlh.design import design, keep_columns, names_for
    from amlh.features import build_graph
    s = load_stream(Paths.cache, "HI-Small")
    g = build_graph(s.t, s.src, s.dst, s.usd)
    rows = s.te[:50]
    for fam in ("wall", "logical", "combined"):
        for H in (0, 3600):
            assert design(s, g, rows, H, s.view_end, s.t, fam).shape[1] == len(names_for(fam))
    assert "hour" not in names_for("logical")
    assert not any(n.startswith("past") and "pair" not in n for n in names_for("logical"))
    names, keep = keep_columns("wall", "^m_")
    assert len(keep) == len(names_for("wall")) - 5


def test_policy_job_uses_validation_only(small_data, tmp_path):
    jobs = [JobSpec(f"nb4-gbt-wall-HI-Small-released-H{h}-s0", "gbt_sweep", {**GBT, "H": h, "seed": 0}) for h in ("1", "3600")]
    pol = JobSpec("nb4-policy", "alert_policy", {"dataset": "HI-Small", "protocol": "released", "seed": 0, "checkpoints": ["1", "3600"], "grid": 0.5,
                                                 "sweep_jobs": [j.job_id for j in jobs], "budgets": [20]}, requires=[j.job_id for j in jobs])
    assert runner(tmp_path).run(jobs + [pol])["complete"]
    info = json.loads((tmp_path / "jobs/nb4-policy/metrics.json").read_text())["budgets"]["20"]
    assert abs(sum(info["alpha"]) - 1) < 1e-9 and len(info["taus"]) == 2


def test_attack_job_saves_each_condition_and_changes_only_times(small_data, tmp_path):
    cfg = {"dataset": "HI-Small", "protocol": "released", "family": "logical", "H": "0", "seed": 0, "Ws": [3600], "conditions": ["none", "stretch", "blend", "shift_blend"]}
    assert runner(tmp_path).run([JobSpec("atk", "gbt_attack", cfg)])["complete"]
    d = tmp_path / "jobs/atk"
    clean = pl.read_parquet(d / "attack_W3600_none.parquet")
    other = pl.read_parquet(d / "attack_W3600_stretch.parquet")
    assert clean["event_id"].to_list() == other["event_id"].to_list()
    assert clean["usd"].to_list() == other["usd"].to_list()
    assert (clean["t"] != other["t"]).any()
    assert ((clean["t"] - other["t"]).abs() <= 3600).all()
    assert (d / "ckpt/best.bin").exists()


def test_equal_alert_threshold_budget(small_data):
    s = load_stream(Paths.cache, "HI-Small")
    sc = np.random.default_rng(0).random(len(s.te))
    tau = equal_alert_threshold(sc, 30, s.core_days)
    assert (sc >= tau).sum() == 30 * s.core_days


def test_tdgnn_trains_resumes_and_attacks(small_data, tmp_path):
    pytest.importorskip("torch")
    cfg = {"dataset": "HI-Small", "protocol": "released", "H": "3600", "seed": 0, "epochs": 2, "neg_rate": 0.2, "d": 16, "batch": 256, "sp": 4, "sf": 4, "s2p": 2, "s2f": 2}
    r = runner(tmp_path, "NB3")
    train = JobSpec("td", "tdgnn_train", cfg)
    assert r.run_job(train) == "DONE"
    m = json.loads((tmp_path / "jobs/td/metrics.json").read_text())
    assert m["epochs_run"] == 2 and "test_results" in m
    assert (tmp_path / "jobs/td/ckpt/best.bin").exists()
    atk = JobSpec("td-attack", "tdgnn_attack", {**cfg, "train_job": "td", "Ws": [3600], "conditions": ["none", "burst"]}, requires=["td"])
    assert r.run_job(atk) == "DONE"
    assert (tmp_path / "jobs/td-attack/attack_W3600_burst.parquet").exists()
    for flag in ("shared_direction", "one_hop", "no_count"):
        assert r.run_job(JobSpec(f"td-{flag}", "tdgnn_train", {**cfg, "epochs": 1, flag: True})) == "DONE"


def test_tdgnn_resume_continues_from_saved_epoch(small_data, tmp_path):
    pytest.importorskip("torch")
    cfg = {"dataset": "HI-Small", "protocol": "released", "H": "0", "seed": 0, "epochs": 3, "patience": 9, "neg_rate": 0.2, "d": 16, "batch": 256, "sp": 4, "sf": 0, "s2p": 2, "s2f": 0}
    r = runner(tmp_path, "NB3")
    r.budget = 10**9
    spec = JobSpec("td", "tdgnn_train", cfg)
    from amlh import jobs_gnn
    real = jobs_gnn._predict
    calls = {"n": 0}

    def interrupt_after_two(*a, **k):
        calls["n"] += 1
        if calls["n"] == 3:
            from amlh.runner import Interrupted
            raise Interrupted("simulated session end")
        return real(*a, **k)
    jobs_gnn._predict = interrupt_after_two
    try:
        assert r.run_job(spec) == "INTERRUPTED"
    finally:
        jobs_gnn._predict = real
    assert runner(tmp_path, "NB3").run_job(spec) == "DONE"
    log = [json.loads(x) for x in (tmp_path / "jobs/td/log.jsonl").read_text().splitlines()]
    assert [x["epoch"] for x in log if x["event"] == "resumed"] == [2]
    assert [x["epoch"] for x in log if x["event"] == "epoch"] == [0, 1, 2]


def test_saml_loader_drops_the_leaky_column(tmp_path):
    rows = ["Time,Date,Sender_account,Receiver_account,Amount,Payment_currency,Received_currency,Sender_bank_location,Receiver_bank_location,Payment_type,Is_laundering,Laundering_type"]
    rng = np.random.default_rng(0)
    for d in range(1, 21):
        for i in range(60):
            rows.append(f"{10 + i % 12:02d}:{i % 60:02d}:{(i * 7) % 60:02d},2022-10-{d:02d},{rng.integers(0, 50)},{rng.integers(0, 50)},{rng.uniform(1, 900):.2f},UK pounds,UK pounds,UK,UK,ACH,0,Normal_Fan_Out")
        rows.append(f"23:00:00,2022-10-{d:02d},900,{901 + d % 2},500.0,UK pounds,Dirham,UK,UAE,Cross-border,1,Structuring")
    raw = tmp_path / "SAML-D.csv"
    raw.write_text("\n".join(rows) + "\n")
    card = prepare_saml(raw, tmp_path / "out")
    ev = pl.read_parquet(tmp_path / "out/events.parquet")
    assert "_type" not in ev.columns and "Laundering_type" not in ev.columns
    assert (ev["usd"] == ev["amt_paid"]).all()
    assert card["attempts_approximate"] and card["n_illicit"] == 20
    att = pl.read_parquet(tmp_path / "out/attempts.parquet")
    assert att["n_matched"].sum() == 20 and att["typology"].unique().to_list() == ["Structuring"]
    assert ev["t"].is_sorted() and (ev["t"] % 60 != 0).any()  # second-level resolution survives


def test_plans_are_consistent():
    seen = set()
    for name, fn in PLANS.items():
        jobs = fn()
        assert jobs, name
        ids = [j.job_id for j in jobs]
        assert len(ids) == len(set(ids)), name
        assert not (seen & set(ids)), name
        seen |= set(ids)
        known = set(ids)
        for j in jobs:
            assert j.tier in ("P0", "P1", "P2")
            for r in j.requires:
                assert r in known, f"{j.job_id} requires {r}, outside its notebook"
    first = PLANS["NB4"]()[0]
    assert first.cfg.get("shuffle_labels") and first.tier == "P0"


def test_sharding_is_a_partition():
    full = [j.job_id for j in plan("NB1")]
    a, b = [j.job_id for j in plan("NB1", (0, 2))], [j.job_id for j in plan("NB1", (1, 2))]
    assert sorted(a + b) == sorted(full) and not set(a) & set(b)


def test_merge_keeps_incomplete_dataset_sealed_and_gives_verdicts(small_data, tmp_path):
    H = ["0", "1", "60", "600", "3600", "21600", "86400", "259200", "inf"]
    jobs = []
    for ds in ("HI-Small", "LI-Small"):
        horizons = H if ds == "HI-Small" else ["0"]
        jobs += [JobSpec(f"nb4-gbt-wall-{ds}-released-H{h}-s{k}", "gbt_sweep", {**GBT, "dataset": ds, "H": h, "seed": k}) for h in horizons for k in (0,)]
    shuffled = JobSpec("nb4-gbt-wall-HI-Small-released-H3600-shuffled-s0", "gbt_sweep", {**GBT, "H": "3600", "seed": 0, "shuffle_labels": True})
    run = tmp_path / "run"
    r = runner(run)
    r.write_manifest({"data_hashes": {"HI-Small": "abc"}})
    assert r.run(jobs + [shuffled])["complete"]
    rep = merge([run], tmp_path / "out", ["HI-Small", "LI-Small"])
    assert "LI-Small" in rep["not_unsealed"]
    v = rep["datasets"]["HI-Small"]["verdicts"]
    for key in ("C1", "C2", "C3", "C3b", "C4"):
        assert key in v, key
    assert isinstance(v["C3b"]["pass"], bool) and "H_star" in v["C3b"]
    leak = rep["datasets"]["HI-Small"]["leakage"]
    assert all(x["pass"] for x in leak.values()) and leak
    text = (tmp_path / "out/MERGE_REPORT.md").read_text()
    assert "Still sealed" in text and "HI-Small" in text
    status = plan_status(collect([run]))
    assert status["NB4"]["P0"][0] >= 1


def test_tamper_is_reported_not_used(small_data, tmp_path):
    run = tmp_path / "run"
    r = runner(run)
    r.write_manifest()
    jid = "nb4-gbt-wall-HI-Small-released-H0-s0"
    assert r.run([JobSpec(jid, "gbt_sweep", {**GBT, "H": "0", "seed": 0})])["complete"]
    assert verify_job(run / "jobs" / jid)[0]
    with open(run / "jobs" / jid / "scores_test.parquet", "ab") as fh:
        fh.write(b"x")
    c = collect([run])
    assert jid not in c.jobs and any("hash mismatch" in p for p in c.problems)


def test_read_scores_rejects_missing_events(small_data, tmp_path):
    s = load_stream(Paths.cache, "HI-Small")
    pl.DataFrame({"event_id": s.event_id[s.te][:10], "score": np.zeros(10, dtype=np.float32)}).write_parquet(tmp_path / "scores_test.parquet")
    with pytest.raises(ValueError):
        read_scores(tmp_path, "scores_test", s, s.te)


def test_attempts_window_and_outcomes(small_data):
    s = load_stream(Paths.cache, "HI-Small")
    aid, apos = attempts_in_window(s, s.te)
    assert len(aid) > 0 and (s.y[apos] == 1).all()
    sc = np.zeros(len(s.te))
    o = heldout_outcomes(s, sc, "0", 0.5)
    assert not o["detected"].any() and (o["value_share"] == 0).all()
