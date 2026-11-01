"""Notebook bootstrap, raw-file checks, and the verdict paths for C5, C6 and C8."""
import json

import numpy as np
import pytest

from amlh.jobs_gbt import Paths
from amlh.merge import attack_table, collect, policy_verdict, read_job_pr, suppression, tail_invariance, timing_verdicts
from amlh.mgnn import _dedupe
from amlh.mgnn_format import format_transactions
from amlh.plans import registry
from amlh.runner import JobSpec, Runner
from amlh.session import find_raw, prepared_hash, print_summary, run_notebook, verify_raw

BASE = {"dataset": "HI-Small", "protocol": "released", "neg_rate": 0.25}
CONDS = ["none", "burst", "stretch", "blend", "shift_burst", "shift_stretch", "shift_blend"]


def test_find_raw_names_what_is_missing(small_data, tmp_path):
    assert find_raw(["HI-Small"], small_data / "raw")["HI-Small"] == small_data / "raw"
    with pytest.raises(FileNotFoundError, match="SAML-D.csv"):
        find_raw(["SAML-D"], tmp_path)


def test_verify_raw_rejects_a_truncated_file(tmp_path):
    (tmp_path / "HI-Small_Trans.csv").write_text("header\nrow\n")
    with pytest.raises(ValueError, match="truncated"):
        verify_raw("HI-Small", tmp_path)
    assert verify_raw("SAML-D", _saml(tmp_path), check_hash=False)["verified"] is False


def _saml(tmp_path):
    (tmp_path / "SAML-D.csv").write_text("x\n")
    return tmp_path


def test_run_notebook_from_scratch_then_resume(small_data, tmp_path):
    """The whole notebook path, twice: nothing is repeated, and outputs carry versions but no identity."""
    jobs = [JobSpec("nb4-gbt-wall-HI-Small-released-H0-s0", "gbt_sweep", {**BASE, "family": "wall", "H": "0", "seed": 0})]
    kw = dict(tiers=("P0",), input_root=small_data / "raw", working=tmp_path / "w", cache_root=tmp_path / "c", verify_hashes=False, jobs=jobs, restore_roots=[])
    Paths_before = (dict(Paths.raw), Paths.cache)
    with pytest.raises(ValueError):  # the small fixture is not the real file, so the row check must stop it
        run_notebook("NB4", **kw)
    import amlh.session as sess
    saved = dict(sess.EXPECTED_RAW)
    sess.EXPECTED_RAW.clear()
    try:
        first = run_notebook("NB4", **kw)
        assert first["complete"]
        stamp = (tmp_path / "w/aml_run/jobs" / jobs[0].job_id / "DONE").read_text()
        assert run_notebook("NB4", **kw)["complete"]
        assert (tmp_path / "w/aml_run/jobs" / jobs[0].job_id / "DONE").read_text() == stamp
    finally:
        sess.EXPECTED_RAW.update(saved)
        Paths.raw, Paths.cache = Paths_before
    manifest = json.loads((tmp_path / "w/aml_run/RUN_MANIFEST.json").read_text())
    assert manifest["code_hash"] != "unversioned" and manifest["data_hashes"]["HI-Small"] == prepared_hash(tmp_path / "c", "HI-Small")
    text = (tmp_path / "w/aml_run/RUN_MANIFEST.json").read_text().lower()
    assert "shaho" not in text and "username" not in text
    assert (tmp_path / "w/aml_run/summary.md").exists()
    assert print_summary(tmp_path / "w")["complete"]


def test_dedupe_averages_one_row_per_edge():
    ids, mean, sampled = _dedupe(np.array([5, 3, 5]), np.array([0.2, 0.9, 0.4]), np.array([True, False, False]))
    assert ids.tolist() == [3, 5] and sampled.tolist() == [False, True]
    assert mean == pytest.approx([0.9, 0.3])


def test_format_port_ids_follow_raw_order(tmp_path):
    raw = tmp_path / "r.csv"
    raw.write_text("Timestamp,From Bank,Account,To Bank,Account,Amount Received,Receiving Currency,Amount Paid,Payment Currency,Payment Format,Is Laundering\n"
                   "2022/09/01 00:20,010,A,020,B,5.00,Euro,6.00,US Dollar,ACH,0\n"
                   "2022/09/01 00:00,020,B,010,A,7.00,US Dollar,7.00,US Dollar,Wire,1\n")
    out = tmp_path / "f.csv"
    info = format_transactions(raw, out)
    lines = out.read_text().splitlines()
    assert lines[0].startswith("EdgeID,from_id,to_id,Timestamp")
    assert lines[1].split(",")[:4] == ["1", "1", "0", "10"]  # sorted by time; the raw row number is kept as EdgeID
    assert lines[2].split(",")[:4] == ["0", "0", "1", "1210"]
    assert info == {"rows": 2, "accounts": 2, "currencies": 2, "formats": 2}


@pytest.fixture(scope="module")
def verdict_run(small_data, tmp_path_factory):
    run = tmp_path_factory.mktemp("verdicts") / "run"
    jobs = []
    for k in (0, 1, 2):
        for fam in ("wall", "logical"):
            nb = "nb4" if fam == "wall" else "nb5"
            jobs.append(JobSpec(f"{nb}-gbt-{fam}-HI-Small-released-H0-s{k}", "gbt_sweep", {**BASE, "family": fam, "H": "0", "seed": k}))
            jobs.append(JobSpec(f"nb5-attack-{fam}-HI-Small-H0-s{k}", "gbt_attack", {**BASE, "family": fam, "H": "0", "seed": k, "Ws": [21600], "conditions": CONDS}))
        jobs.append(JobSpec(f"nb5-attack-wall-HI-Small-H1-s{k}", "gbt_attack", {**BASE, "family": "wall", "H": "1", "seed": k, "Ws": [21600], "conditions": CONDS}))
        jobs.append(JobSpec(f"nb4-gbt-wall-HI-Small-documented-H0-s{k}", "gbt_sweep", {**BASE, "protocol": "documented", "family": "wall", "H": "0", "seed": k}))
    sweeps = [f"nb4-gbt-wall-HI-Small-released-H{h}-s0" for h in ("1", "600", "3600", "21600", "86400")]
    jobs += [JobSpec(j, "gbt_sweep", {**BASE, "family": "wall", "H": j.rsplit("-H", 1)[1].split("-")[0], "seed": 0}) for j in sweeps]
    jobs.append(JobSpec("nb4-policy-HI-Small-released-s0", "alert_policy", {"dataset": "HI-Small", "protocol": "released", "seed": 0, "checkpoints": ["1", "600", "3600", "21600", "86400"],
                                                                         "sweep_jobs": sweeps, "budgets": [500], "grid": 0.5}, requires=sweeps))
    r = Runner(run, "V", 3600, code_version="test", registry=registry())
    r.write_manifest()
    s = r.run(jobs)
    assert s["complete"], [x for x in s["jobs"] if not x["verified"]]
    return run


def test_attack_table_shape_and_controls(small_data, verdict_run):
    from amlh.jobs_gbt import get_stream
    c = collect([verdict_run])
    s, _ = get_stream({"dataset": "HI-Small", "protocol": "released"})
    table = attack_table(c, s, "wall", "0", 21600)
    assert set(table) == set(CONDS)
    n = len(table["none"]["detected"])
    assert n > 0 and all(len(v["recall"]) == n for v in table.values())
    res = suppression(table, "burst")
    assert res is not None and len(res["ci"]) == 2
    assert suppression({"burst": table["burst"]}, "burst") is None


def test_c6_c8_and_policy_verdicts(small_data, verdict_run):
    from amlh.jobs_gbt import get_stream
    c = collect([verdict_run])
    s, _ = get_stream({"dataset": "HI-Small", "protocol": "released"})
    v = timing_verdicts(c, s)
    assert isinstance(v["C6"]["pass"], bool)
    c8 = v["C8@H=0"]
    assert isinstance(c8["pass"], bool) and {"stretch", "blend", "clean_pr_ratio"} <= set(c8)
    assert read_job_pr(c, s, "logical", "0", 0) is not None
    pol = policy_verdict(c, s, [0])
    assert isinstance(pol["C5"]["pass"], bool)


def test_tail_invariance_holds_on_real_runs(small_data, verdict_run):
    """Trained at strict past, with and without tail days present: scores on the core must agree."""
    result = tail_invariance(collect([verdict_run]), "HI-Small")
    assert result == {0: True, 1: True, 2: True}
