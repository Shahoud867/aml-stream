"""Run all 10 notebook shards on this machine's GPU(s), unattended.

  python gpu_session/run_all.py --until 14:30                 # one GPU, defaults
  python gpu_session/run_all.py --gpus 0,1 --until 11:40      # two GPUs on one machine
  python gpu_session/run_all.py --dry-run                     # show the plan only

What it does, in order:
  1. pre-flight check (check_env.py); stops if anything fails
  2. prepares the data once, so the shards never race on the cache
  3. canary pilots: one short pilot per kind (NB1, NB2, NB3). A kind whose pilot fails is NOT launched; the others are
  4. production: every shard, as GPU capacity allows, longest jobs first. Each stops cleanly at --until
Re-running the same command resumes: finished jobs are skipped, interrupted ones continue from their checkpoints.
Output: <work>/<shard name>/aml_run (results) and <work>/<shard name>.log
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "src"))
from make_colab10 import SET  # noqa: E402  (the same 10-way split as the Colab notebooks)

COST = {"NB1": 2, "NB2": 2, "NB3": 1}   # share of one GPU a shard needs; capacity per GPU is --capacity (default 3)


def launch(name: str, notebook: str, shard, mode: str, a, gpu: str, until: str | None) -> subprocess.Popen:
    work = Path(a.work) / name
    work.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(HERE / "run_gpu.py"), "--notebook", notebook, "--mode", mode, "--tiers", "P0", "--shard", f"{shard[0]}/{shard[1]}",
           "--data", a.data, "--work", str(work), "--cache", a.cache, "--official", a.official]
    if until:
        cmd += ["--until", until]
    if a.no_hash:
        cmd += ["--no-hash"]
    if a.mgnn_batch_size and notebook in ("NB1", "NB2"):
        cmd += ["--batch-size", str(a.mgnn_batch_size)]
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu, "PYTHONUNBUFFERED": "1"}
    log = open(Path(a.work) / f"{name}.{mode}.log", "a", encoding="utf8")
    print(f"  start {mode:10s} {name} on GPU {gpu}")
    return subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)


def pilot_passed(a, name: str) -> bool | None:
    f = Path(a.work) / name / "aml_pilot" / "PILOT_REPORT.json"
    return json.loads(f.read_text())["pass"] if f.exists() else None


def run_queue(items, a, until, gpus, capacity):
    """items: (name, notebook, shard, mode). Starts each as soon as some GPU has room. Returns {name: exit code}."""
    used = {g: 0 for g in gpus}
    running: list[tuple[subprocess.Popen, str, str, int]] = []
    codes, pending = {}, list(items)
    while pending or running:
        for g in gpus:   # fill free capacity, in order
            for it in list(pending):
                cost = COST[it[1]] if it[3] == "production" else 1
                if used[g] + cost <= capacity:
                    running.append((launch(it[0], it[1], it[2], it[3], a, g, until), it[0], g, cost))
                    used[g] += cost
                    pending.remove(it)
        time.sleep(15)
        for r in list(running):
            if r[0].poll() is not None:
                codes[r[1]] = r[0].returncode
                used[r[2]] -= r[3]
                running.remove(r)
                print(f"  done  {r[1]} exit code {r[0].returncode}")
    return codes


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpus", default="0", help="comma separated GPU ids, e.g. 0 or 0,1")
    ap.add_argument("--capacity", type=int, default=3, help="per GPU: a Multi-GNN shard costs 2, a TD-GNN shard 1. Lower to 2 if a GPU runs out of memory")
    ap.add_argument("--until", help="stop time HH:MM; default 4 hours from now")
    ap.add_argument("--only", help="comma separated shard numbers, e.g. 2,3,10")
    ap.add_argument("--mgnn-batch-size", type=int, help="DEVIATION from the frozen plan (official 8192): smaller Multi-GNN batches so NB1 and NB2 fit a 10 GB GPU. Output goes to <work>_b<N> so it never mixes with official-setting runs")
    ap.add_argument("--skip-pilots", action="store_true", help="skip the canary pilots (not recommended: the GPU path has never been run)")
    ap.add_argument("--data", default=str(ROOT / "data" / "raw"))
    ap.add_argument("--work", default=str(ROOT.parent / "aml_out"))
    ap.add_argument("--cache", default=str(ROOT.parent / "aml_cache"))
    ap.add_argument("--official", default=str(ROOT.parent / "aml_official"))
    ap.add_argument("--no-hash", action="store_true")
    ap.add_argument("--allow-cpu", action="store_true", help="rehearsal without a GPU")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    if a.mgnn_batch_size:
        a.work = f"{a.work}_b{a.mgnn_batch_size}"
        print(f"NOTE: Multi-GNN batch size {a.mgnn_batch_size} instead of the official 8192. Output folder: {a.work}. Record this in DECISIONS.md.")
    gpus = a.gpus.split(",")
    keep = {int(x) for x in a.only.split(",")} if a.only else set(range(1, 11))
    shards = [(n, *s) for n, s in zip(range(1, 11), SET) if n in keep]
    if a.until is None:
        t = time.localtime(time.time() + 4 * 3600)
        a.until = f"{t.tm_hour:02d}:{t.tm_min:02d}"
    Path(a.work).mkdir(parents=True, exist_ok=True)

    print(f"GPUs {gpus}, capacity {a.capacity} each, stop at {a.until}, output {a.work}")
    for n, name, nb, shard, mode in shards:
        print(f"  {n:2d} {nb} shard {shard[0]}/{shard[1]}  {name}")
    if a.dry_run:
        return

    print("\n[1/4] pre-flight")
    chk = [sys.executable, str(HERE / "check_env.py"), "--official", a.official, "--data", a.data] + (["--allow-cpu"] if a.allow_cpu else [])
    if subprocess.run(chk, cwd=ROOT).returncode != 0:
        sys.exit("pre-flight failed: fix the FAIL lines above, then run again")

    print("\n[2/4] preparing data once")
    from amlh.jobs_gbt import Paths
    from amlh.prepare import ensure_prepared
    from amlh.session import find_raw
    names = ["HI-Small", "LI-Small"]
    raw = find_raw(names, Path(a.data))
    Paths.cache = Path(a.cache)
    Paths.cache.mkdir(parents=True, exist_ok=True)
    for n in names:
        t0 = time.time()
        ensure_prepared(n, raw[n], Paths.cache)
        print(f"  {n} ready in {time.time() - t0:.0f} s")

    blocked: set[str] = set()
    if not a.skip_pilots:
        print("\n[3/4] canary pilots (one per kind)")
        canaries = [s for s in shards if s[4] == "pilot"]
        todo = [(name, nb, sh, "pilot") for n, name, nb, sh, _ in canaries if pilot_passed(a, name) is not True]
        if todo:
            run_queue(todo, a, a.until, gpus, a.capacity)
        for n, name, nb, sh, _ in canaries:
            ok = pilot_passed(a, name)
            print(f"  pilot {nb}: {'PASS' if ok else 'FAIL or missing'}  ({Path(a.work) / name / 'aml_pilot' / 'PILOT_REPORT.md'})")
            if not ok:
                blocked.add(nb)
    print("\n[4/4] production")
    items = [(name, nb, sh, "production") for n, name, nb, sh, _ in shards if nb not in blocked]
    for nb in sorted(blocked):
        print(f"  NOT launching {nb}: its pilot did not pass. Read the report, fix, and run this command again.")
    items.sort(key=lambda it: -COST[it[1]])   # long Multi-GNN runs first
    codes = run_queue(items, a, a.until, gpus, a.capacity)

    print("\n=== summary")
    for n, name, nb, sh, _ in shards:
        s = Path(a.work) / name / "aml_run" / "summary.json"
        info = json.loads(s.read_text()) if s.exists() else {}
        print(f"  {n:2d} {name:38s} exit {codes.get(name, '-')}  {({k: v for k, v in info.items() if k in ('status', 'verified', 'n_verified', 'incomplete')} or 'no summary')}")
    print("Re-run the same command to resume anything that stopped at the time limit.")


if __name__ == "__main__":
    main()
