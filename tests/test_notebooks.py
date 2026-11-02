"""The notebooks, the pilot, GPU workers and the FraudGT job's process handling. Official code is never imported here."""
import json
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import polars as pl
import pytest

import amlh.fraudgt as fg
import amlh.session as sess
from amlh.jobs_gbt import Paths
from amlh.plans import KILL_POINT, PLANS, _chains, pilot, plan
from amlh.runner import Interrupted, JobSpec, Runner, code_hash, verify_job

ROOT = Path(__file__).resolve().parents[1]
BASE = {"dataset": "HI-Small", "protocol": "released", "neg_rate": 0.25}
SWEEP = {**BASE, "family": "wall", "H": "3600", "seed": 0}


@pytest.fixture(autouse=True)
def restore_globals():
    """run_notebook points the global paths at its own folders; other tests share the session fixture's."""
    import amlh.jobs_gbt as jg
    import amlh.mgnn as mg
    saved = (dict(Paths.raw), Paths.cache, mg.MgnnPaths.code, mg.MgnnPaths.work, fg.FraudgtPaths.code, fg.FraudgtPaths.work)
    yield
    Paths.raw, Paths.cache, mg.MgnnPaths.code, mg.MgnnPaths.work, fg.FraudgtPaths.code, fg.FraudgtPaths.work = saved
    jg._stream_cache.clear()


def cells(name):
    return json.loads((ROOT / "notebooks" / f"{name}.ipynb").read_text())["cells"]


def source(cell):
    return "".join(cell["source"])


@pytest.mark.parametrize("name", ["nb1", "nb2", "nb3", "nb4", "nb5", "merge"])
def test_notebook_is_valid_and_carries_this_code_version(name):
    code_cells = [source(c) for c in cells(name) if c["cell_type"] == "code"]
    for text in code_cells:
        compile(text, name, "exec")
    assert f"EXPECTED_CODE_HASH = '{code_hash(ROOT)}'" in "\n".join(code_cells), "rebuild with scripts/build_notebooks.py"


@pytest.mark.parametrize("name", ["nb1", "nb2", "nb3", "nb4", "nb5", "merge"])
def test_notebook_holds_no_credentials_or_identities(name):
    text = (ROOT / "notebooks" / f"{name}.ipynb").read_text().lower()
    assert not re.search(r"kaggle_key['\"]?\]?\s*=\s*['\"][a-z0-9]{8,}", text)
    for word in ("shaho", "saif", "family", "gmail", "my account"):
        assert word not in text


def test_every_notebook_has_a_kill_point_in_its_pilot():
    for nb in PLANS:
        ids = {j.job_id for j in pilot(nb)}
        assert KILL_POINT[nb][0] in ids
        assert {r for j in pilot(nb) for r in j.requires} <= ids


def test_pilot_never_models_a_sealed_dataset():
    from amlh.sealing import is_sealed
    for nb in PLANS:
        for j in pilot(nb):
            assert j.job_id.startswith("pilot-")
            if is_sealed(j.cfg.get("dataset", "")):
                assert j.fn == "prepare_check"


def test_shards_keep_chains_together_and_cover_every_job():
    for nb in PLANS:
        jobs = plan(nb)
        parts = [plan(nb, (i, 2)) for i in range(2)]
        assert sorted(j.job_id for p in parts for j in p) == sorted(j.job_id for j in jobs)
        for part in parts:
            ids = {j.job_id for j in part}
            assert all(r in ids for j in part for r in j.requires)
        assert sum(len(c) for c in _chains(jobs)) == len(jobs)


def test_cadence_models_are_scored_at_their_own_cadence():
    jobs = {j.job_id: j for j in PLANS["NB1"]()}
    retrained = [j for j in jobs.values() if j.fn == "mgnn_train" and j.cfg.get("train_cadence")]
    assert retrained
    for t in retrained:
        replay = jobs[f"{t.job_id}-replay{t.cfg['train_cadence']}"]
        assert replay.cfg["cadence"] == t.cfg["train_cadence"] and replay.requires == [t.job_id]


def _small_input(small_data, tmp_path):
    inp = tmp_path / "input"
    shutil.copytree(small_data / "raw", inp / "raw")
    return inp


def _common(inp, tmp_path):
    return dict(input_root=inp, working=tmp_path / "w", cache_root=tmp_path / "c", official_root=tmp_path / "o", verify_hashes=False, check_raw=False)


def test_pilot_kills_resumes_and_reports(small_data, tmp_path):
    specs = [JobSpec("pilot-sweep", "gbt_sweep", SWEEP)]
    report = sess.run_pilot("NB4", specs=specs, kill_point=("pilot-sweep", 1), budget_hours=1.0, **_common(_small_input(small_data, tmp_path), tmp_path))
    assert report["resume"]["killed_exit_code"] == 137 and report["resume"]["resumed_exit_code"] == 0
    assert report["resume"]["scores"]["scores_test.parquet"]["max_abs_diff"] <= 1e-5
    assert report["pass"]
    assert (tmp_path / "w/aml_pilot/PILOT_REPORT.md").read_text().startswith("# NB4 pilot: PASS")
    assert not (tmp_path / "w/aml_run").exists(), "pilot output must stay apart from production"
    assert report["projected_hours"]["P0"]["hours"] > 0


def test_pilot_fails_when_the_killed_job_never_reaches_the_kill_point(small_data, tmp_path):
    specs = [JobSpec("pilot-sweep", "gbt_sweep", SWEEP)]
    report = sess.run_pilot("NB4", specs=specs, kill_point=("pilot-sweep", 10 ** 6), budget_hours=1.0, **_common(_small_input(small_data, tmp_path), tmp_path))
    assert report["resume"]["killed_exit_code"] == 0 and not report["pass"]


def test_restore_and_merge_ignore_pilot_output(small_data, tmp_path):
    inp = _small_input(small_data, tmp_path)
    sess.run_pilot("NB4", specs=[JobSpec("pilot-sweep", "gbt_sweep", SWEEP)], kill_point=("pilot-sweep", 1), budget_hours=1.0, **_common(inp, tmp_path))
    from amlh.merge import collect
    assert not collect([tmp_path / "w"]).jobs
    r = Runner(tmp_path / "next" / "aml_run", "NB4", 60.0, code_version="x")
    assert r.restore([tmp_path / "w"]) == 0


def test_two_workers_split_the_work_without_overlap(small_data, tmp_path):
    specs = [JobSpec(f"nb4-gbt-wall-HI-Small-released-H{h}-s0", "gbt_sweep", {**SWEEP, "H": h}) for h in ("0", "3600")]
    kw = _common(_small_input(small_data, tmp_path), tmp_path)
    out = sess.run_workers("NB4", 2, tiers=("P0",), budget_hours=1.0, restore_roots=[], specs=specs, **kw)
    assert out["worker_exit_codes"] == [0, 0]
    assert [sorted(p.name for p in (tmp_path / f"w/gpu{i}/aml_run/jobs").iterdir()) for i in range(2)] == [[specs[0].job_id], [specs[1].job_id]]
    assert all(s["complete"] for s in out["summaries"])
    a = pl.read_parquet(tmp_path / f"w/gpu0/aml_run/jobs/{specs[0].job_id}/scores_test.parquet")
    assert len(a) and verify_job(tmp_path / f"w/gpu0/aml_run/jobs/{specs[0].job_id}")[0]


def test_final_checklist_says_what_to_do_next(small_data, tmp_path, monkeypatch, capsys):
    spec = JobSpec("nb4-gbt-wall-HI-Small-released-H3600-s0", "gbt_sweep", SWEEP)
    monkeypatch.setattr(sess, "plan", lambda nb, shard=None: [spec])
    assert not sess.final_checklist("NB4", ("P0",), tmp_path / "empty")
    kw = _common(_small_input(small_data, tmp_path), tmp_path)
    sess.run_notebook("NB4", tiers=("P0",), budget_hours=1.0, jobs=[spec], restore_roots=[], **kw)
    capsys.readouterr()
    assert sess.final_checklist("NB4", ("P0",), tmp_path / "w")
    assert "aml-out-nb4-v1" in capsys.readouterr().out
    monkeypatch.setattr(sess, "plan", lambda nb, shard=None: [spec, JobSpec("not-run-yet", "gbt_sweep", SWEEP)])
    assert not sess.final_checklist("NB4", ("P0",), tmp_path / "w")
    assert "jobs remain" in capsys.readouterr().out


def _run_cells(name, tmp_path, inp, **replace):
    """Runs a notebook's real code cells, with only the Kaggle paths and settings replaced."""
    ns = {}
    for c in cells(name):
        if c["cell_type"] != "code":
            continue
        text = source(c).replace("/kaggle/input", inp.as_posix()).replace("/kaggle/working", (tmp_path / "w").as_posix()).replace("/tmp/aml_code", (tmp_path / "code").as_posix())
        for old, new in replace.items():
            text = text.replace(old, new)
        exec(compile(text, name, "exec"), ns)
    return ns


def _package(inp):
    """The real code package, built by the real script."""
    subprocess.run([sys.executable, str(ROOT / "scripts" / "make_code_dataset.py"), str(inp / "code")], check=True, capture_output=True)


def test_notebook_cells_run_end_to_end_in_production_mode(small_data, tmp_path, monkeypatch):
    inp = _small_input(small_data, tmp_path)
    _package(inp)
    spec = JobSpec("nb4-gbt-wall-HI-Small-released-H3600-s0", "gbt_sweep", SWEEP)
    monkeypatch.setattr(sess, "plan", lambda nb, shard=None: [spec])
    monkeypatch.setattr(sess, "EXPECTED_RAW", {})
    monkeypatch.setattr(sess, "OFFICIAL_NEEDED", {})
    monkeypatch.setattr(sys, "path", list(sys.path))
    default_cache = sess.run_notebook.__defaults__
    ns = _run_cells("nb4", tmp_path, inp, **{"MODE = 'pilot'": "MODE = 'production'", "INSTALL = True": "INSTALL = False", "RUN_TESTS = True": "RUN_TESTS = False",
                                            "shard=SHARD,": f"shard=SHARD, cache_root=Path({(tmp_path / 'c').as_posix()!r}), official_root=Path({(tmp_path / 'o').as_posix()!r}),"})
    assert ns["MODE"] == "production" and sess.run_notebook.__defaults__ == default_cache
    assert verify_job(tmp_path / "w/aml_run/jobs" / spec.job_id)[0]
    assert (tmp_path / "w/aml_run/summary.md").exists()
    assert code_hash(tmp_path / "code") == code_hash(ROOT)


def _bootstrap(tmp_path, inp, expected):
    """Runs the paths cell and then the bootstrap cell, as a notebook does, on a Kaggle-shaped input folder."""
    paths = [source(c) for c in cells("nb4") if "Where things live" in source(c)][0]
    boot = [source(c) for c in cells("nb4") if "Unpacks the code" in source(c)][0]
    ns = {"EXPECTED_CODE_HASH": expected, "INSTALL": False, "NOTEBOOK": "NB4", "SHARD": None}
    for text in (paths, boot):
        exec(text.replace("/kaggle/input", inp.as_posix()).replace("/tmp/aml_code", (tmp_path / "code").as_posix()), ns)


def test_bootstrap_accepts_a_package_kaggle_unzipped(tmp_path, monkeypatch):
    inp = tmp_path / "input"
    _package(inp)
    archive = next((inp / "code").glob("aml-code-*.zip"))
    zipfile.ZipFile(archive).extractall(inp / "unzipped")
    archive.unlink()
    monkeypatch.setattr(sys, "path", list(sys.path))
    _bootstrap(tmp_path, inp, code_hash(ROOT))
    assert code_hash(tmp_path / "code") == code_hash(ROOT)


def test_bootstrap_refuses_another_code_version(tmp_path, monkeypatch):
    inp = tmp_path / "input"
    _package(inp)
    monkeypatch.setattr(sys, "path", list(sys.path))
    with pytest.raises(AssertionError, match="this notebook expects"):
        _bootstrap(tmp_path, inp, "different")


def test_bootstrap_refuses_two_code_packages(tmp_path, monkeypatch):
    inp = tmp_path / "input"
    _package(inp)
    shutil.copy(next((inp / "code").glob("aml-code-*.zip")), inp / "code" / "aml-code-other.zip")
    monkeypatch.setattr(sys, "path", list(sys.path))
    with pytest.raises(AssertionError, match="exactly one code package"):
        _bootstrap(tmp_path, inp, code_hash(ROOT))


def test_compare_scores_joins_on_event_id_and_reports_differences(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir(), b.mkdir()
    pl.DataFrame({"event_id": [1, 2, 3], "score": [0.1, 0.2, 0.9]}).write_parquet(a / "scores_test.parquet")
    pl.DataFrame({"event_id": [3, 1, 2], "score": [0.9, 0.1, 0.2]}).write_parquet(b / "scores_test.parquet")
    assert sess.compare_scores(a, b)["scores_test.parquet"]["max_abs_diff"] == 0
    pl.DataFrame({"event_id": [1, 2, 3], "score": [0.1, 0.2, 0.5]}).write_parquet(b / "scores_test.parquet")
    assert sess.compare_scores(a, b)["scores_test.parquet"]["max_abs_diff"] == pytest.approx(0.4)
    (b / "scores_test.parquet").unlink()
    assert sess.compare_scores(a, b)["scores_test.parquet"]["missing"]


def test_pinned_commits_match_the_code_that_was_read():
    for name, (_, commit) in sess.OFFICIAL.items():
        head = ROOT / "third_party" / name / ".git" / "HEAD"
        if head.exists():
            out = subprocess.run(["git", "-C", str(head.parents[1]), "rev-parse", "HEAD"], capture_output=True, text=True)
            assert out.stdout.strip() == commit


# FraudGT job: its child process is replaced by a small script, so only this side of the boundary is tested.
class _Spec:
    job_id = "pilot-nb2-fraudgt"


class _Ctx:
    def __init__(self, d, runner):
        self.dir, self.runner, self.spec = d, runner, _Spec()


def _fake_driver(tmp_path, monkeypatch, body):
    script = tmp_path / "fake_driver.py"
    script.write_text(body)
    monkeypatch.setattr(fg, "driver_command", lambda spec: [sys.executable, str(script), str(spec)])
    monkeypatch.setattr(fg.FraudgtPaths, "work", tmp_path / "work")
    monkeypatch.setattr(fg.FraudgtPaths, "code", tmp_path / "fraudgt_code")
    monkeypatch.setattr(fg, "link_raw", lambda raw_csv, data_dir: data_dir.mkdir(parents=True, exist_ok=True))  # symlinks need extra rights on Windows
    raw = tmp_path / "raw"
    raw.mkdir(exist_ok=True)
    (raw / "HI-Small_Trans.csv").write_text("x\n")
    monkeypatch.setattr(Paths, "raw", {"HI-Small": raw})
    job = tmp_path / "job"
    job.mkdir(exist_ok=True)
    runner = Runner(tmp_path / "run", "NB2", 3600.0, code_version="x")
    (runner.jobs_dir / _Spec.job_id).mkdir(parents=True, exist_ok=True)
    return _Ctx(job, runner)


CFG = {"dataset": "HI-Small", "config": "a.yaml", "seed": 42, "poll_seconds": 0.05}


def test_fraudgt_success_returns_the_drivers_result(tmp_path, monkeypatch):
    body = "import json, sys\nspec = json.load(open(sys.argv[1]))\nopen(spec['job_dir'] + '/result.json', 'w').write(json.dumps({'seed': 42, 'overrides': spec['overrides']}))\n"
    result = fg.fraudgt_train(_fake_driver(tmp_path, monkeypatch, body), CFG)
    assert result["seed"] == 42
    o = result["overrides"]
    assert o[o.index("wandb.use") + 1] == "False" and o[o.index("seed") + 1] == "42" and "optim.max_epoch" not in o


def test_fraudgt_out_of_time_becomes_an_interruption(tmp_path, monkeypatch):
    ctx = _fake_driver(tmp_path, monkeypatch, f"import sys\nsys.exit({fg.OUT_OF_TIME})\n")
    with pytest.raises(Interrupted):
        fg.fraudgt_train(ctx, CFG)


def test_fraudgt_crash_reports_the_error_text(tmp_path, monkeypatch):
    ctx = _fake_driver(tmp_path, monkeypatch, "raise ValueError('a very specific failure')\n")
    with pytest.raises(RuntimeError, match="a very specific failure"):
        fg.fraudgt_train(ctx, CFG)


def test_fraudgt_silent_exit_is_not_a_success(tmp_path, monkeypatch):
    ctx = _fake_driver(tmp_path, monkeypatch, "pass\n")
    with pytest.raises(RuntimeError, match="without writing a result"):
        fg.fraudgt_train(ctx, CFG)


def test_fraudgt_stop_request_reaches_the_child(tmp_path, monkeypatch):
    body = "import sys, time\nprint('started', file=sys.stderr, flush=True)\ntime.sleep(60)\n"
    ctx = _fake_driver(tmp_path, monkeypatch, body)
    ctx.runner.stop_requested = True
    with pytest.raises(RuntimeError, match="failed with"):  # a killed child without its out-of-time exit code is reported, not treated as success
        fg.fraudgt_train(ctx, CFG)


def test_fraudgt_overrides_only_move_files_unless_the_pilot_says_otherwise():
    o = fg.overrides({"seed": 1}, Path("d"), Path("l"))
    assert not {"optim.max_epoch", "train.iter_per_epoch", "train.batch_size", "gt.layers"} & set(o)
    p = fg.overrides({"seed": 1, "opts": {"optim.max_epoch": 2}}, Path("d"), Path("l"))
    assert p[p.index("optim.max_epoch") + 1] == "2"


class _Store(dict):
    """Behaves like a PyG edge store for the fields add_back touches."""
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


def test_add_back_appends_exactly_the_missing_seed_edges():
    torch = pytest.importorskip("torch")
    from amlh.fraudgt_driver import add_back, probability
    full = _Store(edge_index=torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]]), edge_attr=torch.arange(8.0).view(4, 2), y=torch.tensor([0, 1, 0, 1]))
    nodes = torch.tensor([7, 1, 2, 0, 3])  # global ids of this batch's nodes, not sorted
    bs = _Store(edge_index=torch.tensor([[1], [2]]), edge_attr=full.edge_attr[[1]], y=full.y[[1]], e_id=torch.tensor([1]), input_id=torch.tensor([0, 1]))
    added = add_back(bs, nodes, full, torch.tensor([1, 3]))
    assert added.tolist() == [3]
    assert bs.e_id.tolist() == [1, 3] and bs.y.tolist() == [1, 1] and bs.input_id.tolist() == [0, 1]
    assert nodes[bs.edge_index[:, 1]].tolist() == [3, 0], "local indices must point at edge 3's endpoints"
    assert torch.equal(bs.edge_attr[1], full.edge_attr[3])
    assert add_back(bs, nodes, full, torch.tensor([1, 3])).tolist() == [], "nothing to add twice"
    assert probability(torch.tensor([[0.0, 0.0]])).item() == pytest.approx(0.5)
    assert probability(torch.tensor([[0.0]])).item() == pytest.approx(0.5)


def test_add_back_refuses_an_endpoint_outside_the_batch():
    torch = pytest.importorskip("torch")
    from amlh.fraudgt_driver import add_back
    full = _Store(edge_index=torch.tensor([[0, 5], [1, 6]]), y=torch.tensor([0, 1]))
    bs = _Store(edge_index=torch.tensor([[0], [1]]), y=torch.tensor([0]), e_id=torch.tensor([0]))
    with pytest.raises(ValueError, match="not in its batch"):
        add_back(bs, torch.tensor([0, 1, 5]), full, torch.tensor([0, 1]))


def test_map_to_rows_rejects_a_misaligned_graph():
    torch = pytest.importorskip("torch")
    from amlh.fraudgt_driver import map_to_rows
    table = pl.DataFrame({"EdgeID": [40, 41, 42], "from_id": [0, 1, 2], "to_id": [1, 2, 0]})
    store = _Store(edge_index=torch.tensor([[0, 1, 2], [1, 2, 0]]))
    assert map_to_rows(np.array([2, 0]), store, table).tolist() == [42, 40]
    store = _Store(edge_index=torch.tensor([[0, 2, 1], [1, 0, 2]]))
    with pytest.raises(ValueError, match="do not match"):
        map_to_rows(np.array([1]), store, table)


def test_official_audit_reports_regions_and_official_style_f1(small_data, tmp_path):
    from amlh.jobs_gbt import get_stream
    from amlh.merge import Collected, official_audit, render
    Paths.raw = {"HI-Small": small_data / "raw"}
    Paths.cache = small_data / "processed"
    try:
        s, _ = get_stream({"dataset": "HI-Small", "protocol": "released"})
    except Exception:
        pytest.skip("fixture layout differs")
    train = tmp_path / "nb1-mgnn-gin-HI-Small-s1"
    replay = tmp_path / "nb1-mgnn-gin-HI-Small-s1-replay3600"
    rng = np.random.default_rng(0)
    for d, cfg in ((train, {"model": "gin"}), (replay, {"model": "gin", "train_job": train.name, "cadence": 3600})):
        d.mkdir()
        (d / "config.json").write_text(json.dumps({"cfg": cfg}))
        for name, rows in (("scores_val", s.va), ("scores_test", s.te)):
            score = np.clip(s.y[rows] * 0.8 + rng.uniform(0, 0.2, len(rows)), 0, 1).astype(np.float32)
            pl.DataFrame({"event_id": s.event_id[rows], "score": score, "sampled": np.ones(len(rows), dtype=bool)}).write_parquet(d / f"{name}.parquet")
    (replay / "scores_val.parquet").unlink()  # falls back to the training job's validation scores
    c = Collected(jobs={train.name: train, replay.name: replay})
    out = official_audit(c, s)
    assert out[train.name]["H"] == "inf" and out[replay.name]["H"] == "3600"
    assert out[train.name]["official_style_f1"] == pytest.approx(1.0) and out[train.name]["coverage"] == 1.0
    assert out[train.name]["published_reference"] == 0.6479
    text = render({"status": {}, "checklist": [], "not_unsealed": {}, "datasets": {"HI-Small": {"unseal": "not sealed", "verdicts": {}, "leakage": {}, "official": out}}})
    assert train.name in text and "Published models" in text
