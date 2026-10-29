"""Service-time profile of the boosted detector (feature extraction + scoring) on this machine, then queueing
behaviour under Poisson load. Latency numbers apply only to the named device, not to a production system."""
import argparse
import json
import platform
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from amlh.features import build_graph, compute_features, feature_names  # noqa: E402
from amlh.queue_sim import max_stable_rate, service_from_profile, simulate  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default="HI-Small")
ap.add_argument("--H", type=int, default=1)
ap.add_argument("--neg-rate", type=float, default=0.05)
ap.add_argument("--repeats", type=int, default=30)
ap.add_argument("--tag", default="service_hi_small")
args = ap.parse_args()

P = ROOT / f"data/processed/{args.dataset}"
card = json.loads((P / "data_card.json").read_text())
ev = pl.read_parquet(P / "events.parquet")
t = ev["t"].to_numpy()
src, dst = ev["src"].to_numpy().astype(np.int64), ev["dst"].to_numpy().astype(np.int64)
usd = ev["usd"].to_numpy()
y = ev["label"].to_numpy()
i_day, j_day = card["split_days"]
day = t // 86400
split = np.where(day < i_day, 0, np.where(day < j_day, 1, 2))
view_end = np.where(split == 0, i_day * 86400 - 1, np.where(split == 1, j_day * 86400 - 1, t.max()))
g = build_graph(t, src, dst, usd)
names = feature_names()
rng = np.random.default_rng(0)
tr = np.where(split == 0)[0]
tr = np.sort(np.r_[tr[y[tr] == 1], rng.choice(tr[y[tr] == 0], int((y[tr] == 0).sum() * args.neg_rate), replace=False)])
booster = lgb.train(dict(objective="binary", num_leaves=63, learning_rate=0.05, verbosity=-1, num_threads=12, seed=0),
                    lgb.Dataset(compute_features(g, tr, args.H, view_end), y[tr].astype(np.float32)), 150)

te = np.where(split == 2)[0]
sizes = [1, 10, 100, 1000, 10000]
profile = []
for n in sizes:
    starts = rng.integers(0, len(te) - n, args.repeats + 5)
    times, feat_times = [], []
    for r, s in enumerate(starts):
        rows = te[s:s + n]
        t0 = time.perf_counter()
        X = compute_features(g, rows, args.H, view_end)
        t1 = time.perf_counter()
        booster.predict(X)
        t2 = time.perf_counter()
        if r >= 5:  # warm-up
            times.append(t2 - t0)
            feat_times.append(t1 - t0)
    profile.append(dict(n=n, median_s=float(np.median(times)), p95_s=float(np.percentile(times, 95)),
                        feat_median_s=float(np.median(feat_times)), events_per_s=float(n / np.median(times))))
    print(profile[-1], flush=True)

svc = service_from_profile([p["n"] for p in profile], [p["median_s"] for p in profile])
rates = [100.0, 300.0, 1000.0, 3000.0, 10000.0, 30000.0, 100000.0]
best, table = max_stable_rate(svc, 1024, rates, duration=20.0)
print("max stable", best)
for row in table:
    print(row)
# Observed mean rate of the benchmark, for scale: core-day traffic of the file.
core_rate = float(np.bincount(day)[:10].mean() / 86400)
peak_minute = float(np.bincount((t // 60)).max() / 60)
out = ROOT / "results" / args.tag
out.mkdir(parents=True, exist_ok=True)
(out / "service.json").write_text(json.dumps(
    dict(device=platform.processor(), node=platform.node() and "HP-15s", H=args.H, profile=profile, max_stable_rate=best,
         table=table, benchmark_mean_rate_eps=core_rate, benchmark_peak_minute_rate_eps=peak_minute), indent=1))
print(f"benchmark mean rate {core_rate:.1f} ev/s, peak minute {peak_minute:.1f} ev/s")
