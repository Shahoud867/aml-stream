"""Drive one NB1/NB2/NB3 GPU notebook from a plain Windows or Linux shell, without Kaggle or Colab.

This file lives outside src/amlh and scripts/ on purpose: the code hash covers only those two folders, so adding it
does not invalidate the code package or the committed notebooks.

Examples (run from the repository root, with the GNN environment active):
  python gpu_session/run_gpu.py --notebook NB3 --mode list --shard 0/2
  python gpu_session/run_gpu.py --notebook NB3 --mode setup
  python gpu_session/run_gpu.py --notebook NB3 --mode pilot
  python gpu_session/run_gpu.py --notebook NB3 --mode production --until 11:45 --shard 0/2
  python gpu_session/run_gpu.py --notebook NB1 --mode production --until 14:30 --only "gin-HI-Small-s1"

Re-running the same command resumes: finished jobs are skipped and interrupted ones continue from checkpoints.
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
# Job processes inherit this. See compat/sitecustomize.py for why it is needed.
os.environ["PYTHONPATH"] = os.pathsep.join(filter(None, [str(Path(__file__).resolve().parent / "compat"), os.environ.get("PYTHONPATH", "")]))

EXPECTED_CODE_HASH = "9686ce184d525c0a"


def hours_until(hhmm: str) -> float:
    now = dt.datetime.now()
    h, m = (int(x) for x in hhmm.split(":"))
    end = now.replace(hour=h, minute=m, second=0, microsecond=0)
    if end < now:
        end += dt.timedelta(days=1)
    return (end - now).total_seconds() / 3600


def closure(jobs, keep_ids: set[str]) -> set[str]:
    by_id = {j.job_id: j for j in jobs}
    out, todo = set(), list(keep_ids)
    while todo:
        k = todo.pop()
        if k not in out:
            out.add(k)
            todo += by_id[k].requires
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--notebook", required=True, choices=["NB1", "NB2", "NB3"])
    ap.add_argument("--mode", required=True, choices=["list", "setup", "pilot", "production"])
    ap.add_argument("--tiers", default="P0", help="comma separated, e.g. P0 or P0,P1")
    ap.add_argument("--until", help="wall-clock stop time HH:MM (24 h); the run stops cleanly then")
    ap.add_argument("--hours", type=float, help="alternative to --until")
    ap.add_argument("--shard", help="i/n: take every n-th dependency chain, e.g. 0/2")
    ap.add_argument("--only", help="regex on job ids; dependencies of matching jobs are kept")
    ap.add_argument("--n-epochs", type=int, help="DEVIATION from the frozen plan: override mgnn n_epochs. Record it in DECISIONS.md")
    ap.add_argument("--data", default=str(ROOT / "data" / "raw"), help="folder holding *_Trans.csv")
    ap.add_argument("--work", default=str(ROOT.parent / "aml_out"), help="output folder (resumes if it already holds a run)")
    ap.add_argument("--cache", default=str(ROOT.parent / "aml_cache"))
    ap.add_argument("--official", default=str(ROOT.parent / "aml_official"))
    ap.add_argument("--no-hash", action="store_true", help="skip SHA-256 of raw files (row counts are still checked)")
    a = ap.parse_args()

    import re

    from amlh import session
    from amlh.plans import plan
    from amlh.runner import code_hash

    tiers = tuple(a.tiers.split(","))
    shard = tuple(int(x) for x in a.shard.split("/")) if a.shard else None
    jobs = plan(a.notebook, shard)
    if a.only:
        keep = closure(jobs, {j.job_id for j in jobs if re.search(a.only, j.job_id)})
        jobs = [j for j in jobs if j.job_id in keep]
    if a.n_epochs:
        for j in jobs:
            if "model" in j.cfg:
                j.cfg = {**j.cfg, "n_epochs": a.n_epochs}
        print(f"WARNING: mgnn n_epochs overridden to {a.n_epochs}. This deviates from the frozen plan.")

    h = code_hash(ROOT)
    print("code hash", h, "OK" if h == EXPECTED_CODE_HASH else f"DIFFERS from the notebooks' {EXPECTED_CODE_HASH}")
    print("GPUs:", session.accelerators() or "NONE FOUND")

    if a.mode == "list":
        for j in jobs:
            if j.tier in tiers:
                print(f"{j.tier}  {j.job_id}  <- {','.join(j.requires) or '-'}")
        print(sum(j.tier in tiers for j in jobs), "jobs")
        return
    if a.mode == "setup":
        print(session.setup_environment(a.notebook, Path(a.official)))
        return

    budget = a.hours if a.hours else (hours_until(a.until) if a.until else 2.5)
    print(f"budget {budget:.2f} h")
    kw = dict(input_root=Path(a.data), working=Path(a.work), cache_root=Path(a.cache), official_root=Path(a.official), verify_hashes=not a.no_hash)
    if a.mode == "pilot":
        report = session.run_pilot(a.notebook, budget_hours=min(budget, 1.0), **kw)
        sys.exit(0 if report["pass"] else 2)
    else:
        summary = session.run_notebook(a.notebook, tiers=tiers, budget_hours=budget, jobs=jobs, shard=shard, restore_roots=[], **kw)
        session.print_summary(Path(a.work))
        print({k: v for k, v in summary.items() if k != "jobs"})


if __name__ == "__main__":
    main()
