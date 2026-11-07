"""What each thin notebook calls: set up the machine, find the data, check it, restore earlier progress, run the plan, verify, summarise.

A notebook needs only: run_notebook("NB4", tiers=("P0",), ...). Outputs record hardware and code version, never who ran it.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

from .plans import KILL_POINT, PLANS, pilot, plan, registry
from .runner import JobSpec, KaggleDatasetMirror, Runner, atomic_write_json, code_hash, read_json, sha256_file, verify_job

# Verified on download; SPEC.md section 2. Row counts exclude the header.
EXPECTED_RAW = {
    "HI-Small": (5078345, "b19d39f515523373f991b689c07e11e7b0b95c17a2c27a87d91584ae16c5b040"),
    "LI-Small": (6924049, "f7a9940339c78b5d1476071505b5867c0f5319640b9c4279d13f531bf965319d"),
    "HI-Medium": (31898238, "3126afb8155e7c8815d62bc5370549a7b5ae6bf7dfe872b3cd7813e66a3d7ff5"),
    "LI-Medium": (31251483, "0fc89584453c97472b1bfde7ca746e21d7167140c22a6c4875272b197a4a3910"),
}
RAW_FILE = {"SAML-D": "SAML-D.csv", "Ethereum": "MulDiGraph.pkl"}
OFFICIAL = {"Multi-GNN": ("https://github.com/IBM/Multi-GNN.git", "252b0252afca109d1d216c411c59ff70753b25fc"),
            "FraudGT": ("https://github.com/junhongmit/FraudGT.git", "f88db9dc003a6f521a90631c4741541686043119")}
OFFICIAL_NEEDED = {"NB1": ["Multi-GNN"], "NB2": ["Multi-GNN", "FraudGT"]}
GPU_NOTEBOOKS = {"NB1", "NB2", "NB3"}
FRAUDGT_EXTRAS = ["yacs", "torchmetrics", "ogb", "gdown", "prettytable", "tensorboardX", "networkx", "matplotlib", "seaborn", "deepsnap"]
OUTPUT_LIMIT_GB = 20
PILOT_BUDGET_HOURS = 2.0
PILOT_MIN_RAM_GB = 24   # preparing a Medium file peaked at 12.1 GB here, and running out of memory kills the whole kernel
LARGE = {"HI-Medium", "LI-Medium"}
# Jobs whose failure is recorded and reported but does not fail the pilot: snapml may have no build for this Python, and the GFP comparison row is allowed to end as a recorded failure.
OPTIONAL_PILOT_JOBS = {"pilot-nb5-gfp"}
# Used only for the pilot's rough projection. SAML-D's figure is approximate.
ROWS = {**{k: v[0] for k, v in EXPECTED_RAW.items()}, "SAML-D": 9_500_000}
PRODUCTION_EPOCHS = {"mgnn_train": 100, "tdgnn_train": 6}
FRAUDGT_ITER_PER_EPOCH = 256
FRAUDGT_MAX_EPOCH = 500


def available_ram_gb() -> float | None:
    """Total RAM, or None where it cannot be read. A Colab free runtime has about 12 GB."""
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal"):
                return int(line.split()[1]) / 1024 ** 2
    except OSError:
        pass
    return None


def pilot_specs(notebook: str, ram_gb: float | None = None) -> tuple[list[JobSpec], list[tuple[str, str]]]:
    """The pilot's jobs, minus any that would load a Medium file into too little memory. Returns (jobs, [(skipped job, reason)])."""
    ram = ram_gb if ram_gb is not None else available_ram_gb()
    keep, skipped = [], []
    for j in pilot(notebook):
        if j.fn == "prepare_check" and j.cfg.get("dataset") in LARGE and ram is not None and ram < PILOT_MIN_RAM_GB:
            skipped.append((j.job_id, f"this machine has {ram:.0f} GB of RAM and the check needs {PILOT_MIN_RAM_GB}"))
        else:
            keep.append(j)
    return keep, skipped


def _pip(*args: str) -> bool:
    return subprocess.run([sys.executable, "-m", "pip", "install", "-q", *args], capture_output=True, text=True).returncode == 0


def accelerators() -> list[str]:
    """Reads the GPU names without importing torch, which must not load before its extensions are installed."""
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return []
    return [line.strip() for line in out.stdout.splitlines() if line.startswith("GPU")] if out.returncode == 0 else []


# Known-good (torch, index) pairs to fall back to, in order, when the session's own torch has no matching PyG wheels.
# 2.5.1+cu121 is what every finished GPU result in this project (RTX 3080s, compute capability <= sm_86) was produced
# under, so it stays first and nothing about those results' environment changes. 2.9.1+cu130 was added after an RTX PRO
# 6000 Blackwell card (compute capability sm_120, needs CUDA >= 12.8) hit "no matching torch-scatter and torch-sparse
# wheels, even for torch 2.5.1": PyG's wheel index (data.pyg.org/whl/) stopped publishing torch-scatter/torch-sparse
# builds after torch-2.9.1+cu130, so that is the newest combination with real wheels, verified end to end on that
# card (torch, torch-scatter, torch-sparse, torch-geometric's GINEConv, all on CUDA) via a live-paired session.
FALLBACK_TORCH = [("2.5.1", "cu121"), ("2.9.1", "cu130")]


def install_gnn_stack() -> str:
    """PyG extensions must match torch exactly, so the wheels follow whichever torch this session has, with known fallbacks."""
    from importlib.metadata import version

    def wheels(v: str) -> bool:
        return _pip("--only-binary", ":all:", "torch-scatter", "torch-sparse", "-f", f"https://data.pyg.org/whl/torch-{v}.html")
    torch_version = version("torch")
    if not wheels(torch_version):
        tried = []
        for v, cu in FALLBACK_TORCH:
            torch_version = f"{v}+{cu}"
            tried.append(torch_version)
            if _pip(f"torch=={v}", "--index-url", f"https://download.pytorch.org/whl/{cu}") and wheels(torch_version):
                break
        else:
            raise RuntimeError(f"no matching torch-scatter and torch-sparse wheels, even for {tried}")
    # 2.6.1 breaks on current torch: its to_hetero tracer reads st.List from torch.fx, which no longer exports it. 2.7.0 imports List from typing.
    if not _pip("torch-geometric==2.7.0", "wandb", "tqdm"):
        raise RuntimeError("could not install torch-geometric")
    return torch_version


def fetch_official(name: str, root: Path) -> Path:
    """Official code at the pinned commit. Nothing newer can slip in, because the checked-out commit is compared."""
    url, commit = OFFICIAL[name]
    target = Path(root) / name
    if not (target / ".git").exists():
        subprocess.run(["git", "clone", "-q", url, str(target)], check=True)
    subprocess.run(["git", "-C", str(target), "checkout", "-q", commit], check=True)
    head = subprocess.run(["git", "-C", str(target), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    if head != commit:
        raise RuntimeError(f"{name}: at {head}, expected {commit}")
    return target


def setup_environment(notebook: str, official_root: Path = Path("/tmp/aml_official")) -> dict:
    """Installs what this notebook needs beyond requirements-kaggle.txt and stops early if the machine is wrong."""
    gpus = accelerators()
    info = {"gpus": gpus}
    if notebook in GPU_NOTEBOOKS and not gpus:
        raise RuntimeError(f"{notebook} needs a GPU. Kaggle: set Accelerator to GPU T4 x2. Colab: Runtime, Change runtime type, T4 GPU")
    if notebook not in GPU_NOTEBOOKS and gpus:
        print(f"Warning: {notebook} uses only the CPU. A GPU here spends quota for nothing: set Accelerator to None.")
    if any("P100" in g for g in gpus):
        print("Warning: recent torch builds may not support the P100. Choose GPU T4 x2 if the GPU check fails.")
    if notebook in OFFICIAL_NEEDED:
        info["torch"] = install_gnn_stack()
        for name in OFFICIAL_NEEDED[notebook]:
            fetch_official(name, official_root)
    if "FraudGT" in OFFICIAL_NEEDED.get(notebook, []):
        info["fraudgt_extras_failed"] = [p for p in FRAUDGT_EXTRAS if not _pip(p)]
    if notebook == "NB5":
        info["snapml_installed"] = _pip("snapml")  # GFP is one job; without this package that job fails and the rest run
        if not info["snapml_installed"]:
            print("Warning: snapml could not be installed on this Python, so the GFP baseline will fail. Nothing else depends on it.")
    return info


def run_preflight_tests(code_root: Path) -> None:
    """A failing test stops the notebook before it spends quota."""
    env = {**os.environ, "PYTHONPATH": str(Path(code_root) / "src")}
    res = subprocess.run([sys.executable, "-m", "pytest", "tests", "-x", "-q", "--no-header", "-p", "no:cacheprovider"], cwd=code_root, env=env, capture_output=True, text=True)
    print(res.stdout[-800:])
    if res.returncode != 0:
        raise RuntimeError(f"preflight tests failed:\n{res.stdout[-3000:]}\n{res.stderr[-800:]}")


def datasets_of(notebook: str, tiers) -> list[str]:
    return sorted({j.cfg["dataset"] for j in plan(notebook) if j.tier in tiers and "dataset" in j.cfg})


def find_raw(names: list[str], input_root: Path) -> dict[str, Path]:
    """Locates each dataset's folder among attached inputs. A missing dataset is an error, not a silent skip."""
    found, missing = {}, []
    for name in names:
        target = RAW_FILE.get(name, f"{name}_Trans.csv")
        hits = sorted(Path(input_root).rglob(target))
        if hits:
            found[name] = hits[0].parent
        else:
            missing.append(target)
    if missing:
        raise FileNotFoundError(f"attach these files as notebook inputs: {missing}")
    return found


def verify_raw(name: str, folder: Path, check_hash: bool = True) -> dict:
    """Row count and SHA-256 against the verified values. One download endpoint has truncated a file before."""
    path = folder / RAW_FILE.get(name, f"{name}_Trans.csv")
    if name not in EXPECTED_RAW:
        return {"dataset": name, "sha256": sha256_file(path) if check_hash else None, "verified": False, "note": "no expected value on record"}
    rows, digest = EXPECTED_RAW[name]
    with open(path, "rb") as fh:
        counted = sum(buf.count(b"\n") for buf in iter(lambda: fh.read(1 << 22), b"")) - 1
    if counted != rows:
        raise ValueError(f"{name}: expected {rows} rows, found {counted}: the file is truncated or different")
    actual = sha256_file(path) if check_hash else None
    if check_hash and actual != digest:
        raise ValueError(f"{name}: SHA-256 {actual} differs from the verified value")
    return {"dataset": name, "rows": counted, "sha256": actual or digest, "verified": True}


def prepared_hash(cache: Path, name: str) -> str:
    """Hash of the prepared columns' values, so it does not depend on the parquet writer's version."""
    import polars as pl
    ev = pl.read_parquet(Path(cache) / name / "events.parquet", columns=["event_id", "t", "src", "dst", "label", "split"])
    h = hashlib.sha256()
    for col in ev.columns:
        h.update(ev[col].to_numpy().tobytes())
    return h.hexdigest()[:16]


def _set_official_paths(official_root, cache_root):
    from .fraudgt import FraudgtPaths
    from .mgnn import MgnnPaths
    MgnnPaths.code, MgnnPaths.work = Path(official_root) / "Multi-GNN", Path(cache_root) / "mgnn"
    FraudgtPaths.code, FraudgtPaths.work = Path(official_root) / "FraudGT", Path(cache_root) / "fraudgt"
    MgnnPaths.work.mkdir(parents=True, exist_ok=True)
    FraudgtPaths.work.mkdir(parents=True, exist_ok=True)


def run_notebook(notebook: str, tiers=("P0",), budget_hours: float = 10.5, code_root: Path | None = None, input_root: Path = Path("/kaggle/input"),
                 working: Path = Path("/kaggle/working"), cache_root: Path = Path("/tmp/aml_cache"), shard: tuple[int, int] | None = None,
                 mirror_slug: str | None = None, restore_roots: list[Path] | None = None, verify_hashes: bool = True, jobs=None,
                 run_name: str = "aml_run", check_raw: bool = True, official_root: Path = Path("/tmp/aml_official")) -> dict:
    """Runs from scratch or resumes: finished jobs are never repeated, and interrupted ones continue from their checkpoints."""
    from .jobs_gbt import Paths
    root = Path(code_root) if code_root else Path(__file__).resolve().parents[2]
    run_dir = Path(working) / run_name
    mirror = KaggleDatasetMirror(run_dir, mirror_slug, staging=Path(cache_root) / "mirror_staging") if mirror_slug else None
    r = Runner(run_dir, notebook, budget_hours * 3600, code_version=code_hash(root), registry=registry(), mirror=mirror)
    r.install_signal_handlers()
    _set_official_paths(official_root, cache_root)
    specs = jobs if jobs is not None else plan(notebook, shard)
    names = sorted({j.cfg["dataset"] for j in specs if j.tier in tiers and "dataset" in j.cfg})
    raw = find_raw(names, input_root)
    checks = [verify_raw(n, raw[n], verify_hashes) for n in names] if check_raw else []
    Paths.raw = raw
    Paths.cache = Path(cache_root)
    Paths.cache.mkdir(parents=True, exist_ok=True)
    restored = r.restore(restore_roots if restore_roots is not None else [Path(input_root)])
    from .prepare import ensure_prepared
    hashes, prepare_seconds = {}, {}
    # A dataset that only a prepare_check job touches is prepared inside that job, so a bad file fails one job and not the whole session.
    modelled = {j.cfg["dataset"] for j in specs if j.tier in tiers and "dataset" in j.cfg and j.fn != "prepare_check"}
    for n in [x for x in names if x in modelled]:
        t0 = time.time()
        ensure_prepared(n, raw[n], Paths.cache)
        hashes[n] = prepared_hash(Paths.cache, n)
        prepare_seconds[n] = round(time.time() - t0, 1)
    r.write_manifest({"tiers": list(tiers), "shard": shard, "raw_checks": checks, "data_hashes": hashes, "prepare_seconds": prepare_seconds, "restored_jobs": restored})
    summary = r.run(specs, tiers)
    if mirror is not None:
        mirror.maybe_sync(force=True)
        summary["mirror_failures"] = mirror.failures
    atomic_write_json(run_dir / "summary.json", summary)
    return summary


def _spawn(payload: dict, env_extra: dict) -> subprocess.Popen:
    env = {k: v for k, v in os.environ.items() if k != "AMLH_KILL_AFTER_CHECKS"}
    env.update(env_extra)
    env["AMLH_CHILD"] = json.dumps(payload, default=str)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1]) + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.Popen([sys.executable, "-c", "from amlh.session import child_main; child_main()"], env=env)


def child_main():
    """Entry point of a second process: a worker on its own GPU, or one half of the pilot's forced-kill check."""
    p = json.loads(os.environ["AMLH_CHILD"])
    jobs = [JobSpec(**j) for j in p.pop("jobs")]
    tiers = tuple(p.pop("tiers"))
    run_notebook(p.pop("notebook"), tiers=tiers, jobs=jobs, **p)


def run_workers(notebook: str, gpus: int, tiers=("P0",), budget_hours: float = 10.5, input_root: Path = Path("/kaggle/input"), working: Path = Path("/kaggle/working"),
                cache_root: Path = Path("/tmp/aml_cache"), mirror_slug: str | None = None, restore_roots: list[Path] | None = None, verify_hashes: bool = True,
                official_root: Path = Path("/tmp/aml_official"), check_raw: bool = True, specs: list[JobSpec] | None = None) -> dict:
    """One process per GPU, each with its own dependency chains and its own output folder. Data is prepared once, first."""
    from .jobs_gbt import Paths
    from .plans import _chains
    from .prepare import ensure_prepared
    t0 = time.time()
    all_jobs = specs if specs is not None else plan(notebook)
    names = sorted({j.cfg["dataset"] for j in all_jobs if j.tier in tiers and "dataset" in j.cfg})
    raw = find_raw(names, input_root)
    if check_raw:
        for n in names:
            verify_raw(n, raw[n], verify_hashes)
    Paths.cache = Path(cache_root)
    Paths.cache.mkdir(parents=True, exist_ok=True)
    for n in names:
        ensure_prepared(n, raw[n], Paths.cache)
    chains = _chains(all_jobs)
    procs = []
    for i in range(gpus):
        keep = {j.job_id for k, chain in enumerate(chains) if k % gpus == i for j in chain}
        payload = {"notebook": notebook, "tiers": list(tiers), "jobs": [asdict(j) for j in all_jobs if j.job_id in keep], "budget_hours": budget_hours - (time.time() - t0) / 3600,
                   "input_root": str(input_root), "working": str(Path(working) / f"gpu{i}"), "cache_root": str(cache_root), "official_root": str(official_root),
                   "mirror_slug": f"{mirror_slug}-gpu{i}" if mirror_slug else None, "restore_roots": [str(x) for x in restore_roots] if restore_roots is not None else None,
                   "verify_hashes": False, "check_raw": False}
        procs.append(_spawn(payload, {"CUDA_VISIBLE_DEVICES": str(i)}))
    codes = [p.wait() for p in procs]
    return {"worker_exit_codes": codes, "summaries": [read_json(Path(working) / f"gpu{i}" / "aml_run" / "summary.json", {}) for i in range(gpus)]}


def compare_scores(a: Path, b: Path, pattern: str = "*.parquet") -> dict:
    """Score files of two runs of one job. Rows join on event_id when there is one, and otherwise follow file order."""
    import numpy as np
    import polars as pl
    out = {}
    for f in sorted(Path(a).glob(pattern)):
        other = Path(b) / f.name
        x = pl.read_parquet(f)
        if "score" not in x.columns:
            continue
        if not other.exists():
            out[f.name] = {"missing": True}
            continue
        y = pl.read_parquet(other)
        if "event_id" in x.columns:
            x, y = x.sort("event_id"), y.sort("event_id")
        if len(x) != len(y):
            out[f.name] = {"rows": [len(x), len(y)], "missing": True}
            continue
        p, q = x["score"].to_numpy().astype(np.float64), y["score"].to_numpy().astype(np.float64)
        d = np.abs(p - q)
        corr = float(np.corrcoef(p, q)[0, 1]) if p.std() > 0 and q.std() > 0 else float(np.array_equal(p, q))
        out[f.name] = {"rows": len(p), "max_abs_diff": float(d.max()), "mean_abs_diff": float(d.mean()), "correlation": corr}
    return out


def resume_check(notebook: str, ref: JobSpec, kill_at: int, common: dict) -> dict:
    """Pilot gate: kills a copy of a job with no cleanup, as an abrupt session end would, resumes it in a new process and compares
    it with the uninterrupted run. CPU boosted-tree jobs must match exactly. GPU jobs cannot, so a correlation floor applies."""
    run_dir = Path(common["working"]) / common["run_name"]
    if not verify_job(run_dir / "jobs" / ref.job_id)[0]:
        return {"pass": False, "note": f"{ref.job_id} did not finish, so there is nothing to compare with"}
    killed = JobSpec(ref.job_id + "-killed", ref.fn, ref.cfg, ref.tier)
    payload = {**common, "notebook": notebook, "jobs": [asdict(killed)], "tiers": [killed.tier]}
    first = _spawn(payload, {"AMLH_KILL_AFTER_CHECKS": str(kill_at)}).wait()
    second = _spawn(payload, {}).wait()
    done, why = verify_job(run_dir / "jobs" / killed.job_id)
    # Official Multi-GNN scoring keeps dropout on, so two runs differ by chance. Compare its evaluation-mode scores, which depend only on the weights.
    pattern = "*_eval.parquet" if list((run_dir / "jobs" / ref.job_id).glob("*_eval.parquet")) else "*.parquet"
    cmp = compare_scores(run_dir / "jobs" / ref.job_id, run_dir / "jobs" / killed.job_id, pattern) if done else {}
    exact = ref.fn.startswith("gbt_")
    close = bool(cmp) and all(not v.get("missing") and (v["max_abs_diff"] <= 1e-5 if exact else v["correlation"] >= 0.95) for v in cmp.values())
    return {"pass": bool(first == 137 and second == 0 and done and close), "killed_exit_code": first, "resumed_exit_code": second, "resumed_done": done,
            "detail": "" if done else why, "rule": "identical within 1e-5" if exact else "correlation at least 0.95", "scores": cmp}


def _job_seconds(run_dir: Path) -> dict[str, float]:
    out: dict[str, float] = {}
    path = run_dir / "ledger.jsonl"
    if path.exists():
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if row.get("event") == "done":
                out[row["job"]] = out.get(row["job"], 0.0) + float(row.get("seconds", 0.0))
    return out


def _epoch_seconds(job_dir: Path) -> float:
    import numpy as np
    path = job_dir / "log.jsonl"
    rows = [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []
    vals = [r["seconds"] for r in rows if r.get("event") == "epoch" and "seconds" in r]
    return float(np.median(vals)) if vals else 0.0


def project_hours(notebook: str, specs: list[JobSpec], run_dir: Path) -> dict:
    """A rough projection from one short run: scaled by row count, with epochs added at the measured rate. It replaces guesses, not measurements."""
    seconds = _job_seconds(run_dir)
    base = {}
    for j in specs:
        if j.job_id in seconds:
            base[j.fn] = {"seconds": seconds[j.job_id], "epoch": _epoch_seconds(run_dir / "jobs" / j.job_id), "cfg": j.cfg}
    tiers: dict[str, dict] = {}
    for j in PLANS[notebook]():
        row = tiers.setdefault(j.tier, {"hours": 0.0, "jobs": 0, "not_projected": 0})
        b, ds = base.get(j.fn), j.cfg.get("dataset")
        if b is None or ds not in ROWS or "HI-Small" not in ROWS:
            row["not_projected"] += 1
            continue
        s = b["seconds"]
        if j.fn in PRODUCTION_EPOCHS:
            pilot_epochs = b["cfg"].get("n_epochs", b["cfg"].get("epochs", 0))
            s += max(PRODUCTION_EPOCHS[j.fn] - pilot_epochs, 0) * b["epoch"]
        elif j.fn == "fraudgt_train":
            per_iter = b["epoch"] / max(b["cfg"]["opts"]["train.iter_per_epoch"], 1)
            s = min(j.cfg.get("max_train_hours", 20) * 3600, per_iter * FRAUDGT_ITER_PER_EPOCH * FRAUDGT_MAX_EPOCH)
        row["hours"] += s * ROWS[ds] / ROWS["HI-Small"] / 3600
        row["jobs"] += 1
    return tiers


def run_pilot(notebook: str, budget_hours: float = PILOT_BUDGET_HOURS, input_root: Path = Path("/kaggle/input"), working: Path = Path("/kaggle/working"),
              cache_root: Path = Path("/tmp/aml_cache"), official_root: Path = Path("/tmp/aml_official"), verify_hashes: bool = True, check_raw: bool = True,
              specs: list[JobSpec] | None = None, kill_point: tuple[str, int] | None = None) -> dict:
    """Runs the short jobs in a separate folder that never mixes with production, then the forced-kill check, then reports each gate."""
    t0 = time.time()
    skipped: list[tuple[str, str]] = []
    if specs is None:
        specs, skipped = pilot_specs(notebook)
    runnable = []
    for j in specs:
        if j.fn == "prepare_check":
            try:
                find_raw([j.cfg["dataset"]], input_root)
            except FileNotFoundError:
                skipped.append((j.job_id, "its file is not attached"))
                continue
        runnable.append(j)
    specs = runnable
    summary = run_notebook(notebook, tiers=("P0",), budget_hours=budget_hours, input_root=input_root, working=working, cache_root=cache_root, verify_hashes=verify_hashes,
                           jobs=specs, run_name="aml_pilot", restore_roots=[], check_raw=check_raw, official_root=official_root)
    run_dir = Path(working) / "aml_pilot"
    job_id, kill_at = kill_point or KILL_POINT[notebook]
    ref = next(j for j in specs if j.job_id == job_id)
    left = budget_hours - (time.time() - t0) / 3600
    common = {"budget_hours": max(left, 0.05), "input_root": str(input_root), "working": str(working), "cache_root": str(cache_root), "official_root": str(official_root),
              "run_name": "aml_pilot", "restore_roots": [], "check_raw": False, "verify_hashes": False}
    resume = resume_check(notebook, ref, kill_at, common)
    failures = []
    for r in summary["jobs"]:
        if not r["verified"]:
            st = read_json(run_dir / "jobs" / r["job_id"] / "status.json", {})
            failures.append({"job": r["job_id"], "status": st.get("status"), "reason": st.get("reason") or r.get("detail") or "", "traceback": (st.get("traceback") or "")[-1500:]})
    unfinished = [r["job_id"] for r in summary["jobs"] if not r["verified"] and r["job_id"] not in OPTIONAL_PILOT_JOBS]
    optional = [(r["job_id"], r["reason"] or "") for r in summary["jobs"] if not r["verified"] and r["job_id"] in OPTIONAL_PILOT_JOBS]
    gates = [("Every pilot job finished and passed verification", not unfinished, ", ".join(unfinished)),
             ("A killed job resumed and agreed with the uninterrupted run", resume["pass"], json.dumps(resume, default=str))]
    gates += [(f"Skipped, check by hand later: {job}", None, why) for job, why in skipped]
    gates += [(f"Optional job did not finish, check by hand later: {job}", None, why[:200]) for job, why in optional]
    if notebook in GPU_NOTEBOOKS:
        names = accelerators()
        gates.append(("A GPU was attached", bool(names), "; ".join(names)))
        gates.append(("Check by hand: compare the quota meter with this session's length. Does T4 x2 count double?", None, f"session length {(time.time() - t0) / 3600:.2f} h"))
    if notebook == "NB1":
        fmt = read_json(run_dir / "jobs" / "pilot-nb1-format-check" / "metrics.json", {})
        gates.append(("The formatter port matches the official formatter", fmt.get("match"), json.dumps(fmt.get("differences", {}))))
        gates.append(("Check after production: nb1-mgnn-gin-HI-Small-s1 reproduces the published F1 of 64.79% within its spread", None, "official_style_test_f1 in that job's metrics.json"))
    report = {"notebook": notebook, "pass": all(g[1] is not False for g in gates), "gates": gates, "resume": resume, "failures": failures,
              "job_seconds": {k: round(v, 1) for k, v in _job_seconds(run_dir).items()}, "projected_hours": project_hours(notebook, specs, run_dir)}
    atomic_write_json(run_dir / "PILOT_REPORT.json", report)
    lines = [f"# {notebook} pilot: {'PASS' if report['pass'] else 'FAIL'}", ""]
    lines += [f"- [{'x' if ok else ' '}] {'CHECK BY HAND: ' if ok is None else ''}{name}{': ' + detail if detail else ''}" for name, ok, detail in gates]
    lines += ["", "## Measured seconds", ""] + [f"- {k}: {v}" for k, v in report["job_seconds"].items()]
    lines += ["", "## Projected production hours (rough: one short run, scaled by row count)", ""]
    lines += [f"- {tier}: {v['hours']:.1f} h over {v['jobs']} jobs, {v['not_projected']} jobs not projected" for tier, v in sorted(report["projected_hours"].items())]
    if failures:
        lines += ["", "## Failed or unfinished jobs", ""]
        for fl in failures:
            lines += [f"### {fl['job']} ({fl['status']})", "", str(fl["reason"])[:1200], ""] + (["```", fl["traceback"], "```", ""] if fl["traceback"] else [])
    text = "\n".join(lines) + "\n"
    (run_dir / "PILOT_REPORT.md").write_text(text)
    print(text)
    return report


def _summaries(working: Path) -> list[dict]:
    return [read_json(p, {}) for p in sorted(Path(working).glob("**/aml_run/summary.json"))]


def print_summary(working: Path = Path("/kaggle/working")):
    out = None
    for s in _summaries(working):
        out = s
        print(f"{s.get('notebook')}: complete={s.get('complete')} counts={s.get('counts')} session={s.get('session_seconds', 0) / 3600:.2f}h")
        for row in s.get("jobs", []):
            if row["status"] != "DONE":
                print(f"  {row['status']:<12} {row['job_id']}  {(row['reason'] or '')[:80]}")
    return out


def _folder_gb(root: Path) -> float:
    return sum(p.stat().st_size for p in Path(root).rglob("*") if p.is_file()) / 1024 ** 3


def final_checklist(notebook: str, tiers=("P0",), working: Path = Path("/kaggle/working"), shard: tuple[int, int] | None = None, platform: str = "kaggle") -> bool:
    """The last cell. Prints pass or fail for each check, and exactly what to do next. True means the listed tiers are complete."""
    working = Path(working)
    runs = [p.parent for p in sorted(working.glob("**/aml_run/RUN_MANIFEST.json"))]
    jobs = {}
    for r in runs:
        jobs.update({d.name: d for d in (r / "jobs").iterdir() if d.is_dir()})
    planned = [j for j in plan(notebook, shard) if j.tier in tiers]
    checks = {j.job_id: verify_job(jobs[j.job_id]) if j.job_id in jobs else (False, "not started") for j in planned}
    unverified = [k for k, (ok, _) in checks.items() if not ok]
    damaged = [k for k, (ok, why) in checks.items() if not ok and k in jobs and (jobs[k] / "DONE").exists()]
    codes = {read_json(r / "RUN_MANIFEST.json", {}).get("code_hash") for r in runs}
    size = _folder_gb(working)
    rows = [("A run manifest exists", bool(runs), f"{len(runs)} found"),
            ("One code version in this output", len(codes) == 1, ", ".join(str(c) for c in codes)),
            ("No finished job fails its hash check", not damaged, ", ".join(damaged[:5])),
            (f"Output is under the {OUTPUT_LIMIT_GB} GB limit" if platform == "kaggle" else "Output size (Drive space is the only limit)", size < OUTPUT_LIMIT_GB or platform != "kaggle", f"{size:.2f} GB"),
            (f"All planned {'+'.join(tiers)} jobs are verified", not unverified, f"{len(planned) - len(unverified)} of {len(planned)}")]
    for name, ok, detail in rows:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    complete = all(ok for _, ok, _ in rows)
    if complete and platform == "colab":
        print(f"\nNEXT: finished. The output is already in Drive at {working}. Share that folder with whoever runs the merge notebook.")
    elif unverified and not damaged and platform == "colab":
        print(f"\nNEXT: {len(unverified)} jobs remain. Run all cells again (Runtime, Run all). The output stays in Drive, so finished jobs are not repeated and interrupted ones continue from their checkpoints.")
    elif complete:
        print("\nNEXT: publish this version's output as a private dataset named aml-out-" + notebook.lower() + "-v1 and attach it to the merge notebook.")
    elif unverified and not damaged:
        print(f"\nNEXT: {len(unverified)} jobs remain. Save this version, attach its output as an input (Add Input, Notebook Output), and use Save & Run All again. Finished jobs are not repeated.")
    else:
        print("\nNEXT: read the failures above before running again.")
    return complete


__all__ = ["run_notebook", "run_workers", "run_pilot", "setup_environment", "run_preflight_tests", "final_checklist", "print_summary"]
