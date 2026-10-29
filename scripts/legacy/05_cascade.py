"""Adaptive multi-horizon alerting from saved per-horizon scores.

Each transfer is scored at every checkpoint H_1 < ... < H_m (t + H_j). It raises an alert at the first
checkpoint where its score clears tau_j. The daily alert budget B is spent across checkpoints:
fractions alpha_j of the budget go to the checkpoints in order, each tau_j set on validation so the
newly alerting transfers (not already alerted earlier) consume exactly that share.

alpha is chosen on validation only, by exhaustive search on a simplex grid, to maximise the mean
interdictable value share of validation attempts. Test data never influences alpha or tau.
Fixed-horizon detectors are the corners of the simplex, so the policy space contains them.
"""
import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default="HI-Small")
ap.add_argument("--tag", required=True, help="results folder holding scores_H*_s*.npz")
ap.add_argument("--max-day", type=int, default=None)
ap.add_argument("--checkpoints", default="1,600,3600,21600,86400")
ap.add_argument("--seeds", default="0,1,2")
ap.add_argument("--budgets", default="100,500,2000")
ap.add_argument("--grid", type=float, default=0.1)
args = ap.parse_args()

P = ROOT / f"data/processed/{args.dataset}"
card = json.loads((P / "data_card.json").read_text())
ev = pl.read_parquet(P / "events.parquet")
if args.max_day is not None:
    ev = ev.filter(pl.col("t") < 86400 * (args.max_day + 1))
t = ev["t"].to_numpy()
usd = ev["usd"].to_numpy()
y = ev["label"].to_numpy()
N = len(t)
i_day, j_day = card["split_days"]
day = t // 86400
split = np.where(day < i_day, 0, np.where(day < j_day, 1, 2))
val_rows = np.where(split == 1)[0]
te_rows = np.where(split == 2)[0]
mem = pl.read_parquet(P / "pattern_members.parquet").filter(pl.col("pos") < N)
att = pl.read_parquet(P / "attempts.parquet")
typ = dict(zip(att["pattern_id"].to_list(), att["typology"].to_list()))
mem = mem.with_columns(pl.Series("mt", t[mem["pos"].to_numpy()]))
first = mem.group_by("pattern_id").agg(pl.col("mt").min().alias("first"))


def attempts_in(rows):
    lo, hi = t[rows[0]], t[rows[-1]]
    ids = first.filter((pl.col("first") >= lo) & (pl.col("first") <= hi))["pattern_id"].to_list()
    m = mem.filter(pl.col("pattern_id").is_in(ids) & pl.col("mt").is_between(lo, hi))
    idx = np.full(N, -1, dtype=np.int64)
    idx[rows] = np.arange(len(rows))
    return m["pattern_id"].to_numpy(), idx[m["pos"].to_numpy()]


val_id, val_idx = attempts_in(val_rows)
te_id, te_idx = attempts_in(te_rows)
print(f"attempts: val {len(np.unique(val_id))}, test {len(np.unique(te_id))}", flush=True)


def spend(scores: list[np.ndarray], alpha, total_alerts: int):
    """Sequentially spends the budget; returns per-checkpoint thresholds."""
    n = len(scores[0])
    alerted = np.zeros(n, dtype=bool)
    taus = []
    for s, a in zip(scores, alpha):
        k = int(round(a * total_alerts))
        if k <= 0:
            taus.append(np.inf)
            continue
        free = ~alerted
        cand = s[free]
        k = min(k, len(cand))
        tau = float(np.partition(cand, len(cand) - k)[len(cand) - k])
        taus.append(tau)
        alerted |= s >= tau
    return taus


def apply_policy(scores, taus):
    """Returns alert checkpoint index per event (-1 = no alert)."""
    n = len(scores[0])
    first_cp = np.full(n, -1, dtype=np.int64)
    for j, (s, tau) in enumerate(zip(scores, taus)):
        if np.isfinite(tau):
            hit = (first_cp < 0) & (s >= tau)
            first_cp[hit] = j
    return first_cp


def value_shares(first_cp, H, rows, a_id, a_idx):
    ids, inv = np.unique(a_id, return_inverse=True)
    n = len(ids)
    tev = t[rows][a_idx]
    uev = usd[rows][a_idx]
    alert = first_cp[a_idx] >= 0
    dec = tev + np.where(alert, H[np.maximum(first_cp[a_idx], 0)], 0)
    T = np.full(n, np.iinfo(np.int64).max, dtype=np.int64)
    np.minimum.at(T, inv[alert], dec[alert])
    later = tev > T[inv]
    tot = np.bincount(inv, weights=uev, minlength=n)
    lat = np.bincount(inv, weights=uev * later, minlength=n)
    cnt = np.bincount(inv, minlength=n)
    cl = np.bincount(inv, weights=later.astype(float), minlength=n)
    det = T < np.iinfo(np.int64).max
    t0 = np.full(n, np.iinfo(np.int64).max, dtype=np.int64)
    np.minimum.at(t0, inv, tev)
    delay = np.where(det, (T - t0) / 3600.0, np.nan)
    return ids, lat / np.maximum(tot, 1e-12), cl / np.maximum(cnt, 1), det, delay


cps = args.checkpoints.split(",")
H = np.array([int(c) for c in cps], dtype=np.int64)
out_dir = ROOT / "results" / args.tag
grid = np.arange(0, 1 + 1e-9, args.grid)
alphas = [a for a in itertools.product(grid, repeat=len(cps)) if abs(sum(a) - 1) < 1e-9]
print(f"{len(alphas)} budget allocations over checkpoints {cps}", flush=True)
rows = []
val_days = j_day - i_day
te_days = max((t[te_rows].max() - t[te_rows].min()) / 86400, 1)
for seed in [int(s) for s in args.seeds.split(",")]:
    sv, st = [], []
    for c in cps:
        z = np.load(out_dir / f"scores_H{c}_s{seed}.npz")
        sv.append(z["val"].astype(np.float64))
        st.append(z["test"].astype(np.float64))
    for B in [int(b) for b in args.budgets.split(",")]:
        total = int(B * val_days)
        best, best_val = None, -1.0
        fixed_val = []
        for a in alphas:
            taus = spend(sv, a, total)
            _, vs, _, _, _ = value_shares(apply_policy(sv, taus), H, val_rows, val_id, val_idx)
            m = float(vs.mean())
            if max(a) > 1 - 1e-9:
                fixed_val.append((m, a))
            if m > best_val:
                best_val, best = m, a
        fixed_best = max(fixed_val, key=lambda x: x[0])[1]

        def evaluate(a):
            taus = spend(sv, a, total)
            fc = apply_policy(st, taus)
            ids, vs, ts, det, delay = value_shares(fc, H, te_rows, te_id, te_idx)
            flags = fc >= 0
            return {"alpha": [float(x) for x in a], "value_share": float(vs.mean()), "transfer_share": float(ts.mean()),
                    "detected": float(det.mean()), "median_delay_h": float(np.nanmedian(delay)) if det.any() else None,
                    "alerts_per_day": float(flags.sum() / te_days), "precision": float(y[te_rows][flags].mean()) if flags.any() else 0.0,
                    "per_attempt_value": vs.tolist(), "ids": ids.tolist()}

        res_adapt = evaluate(best)
        res_fixed = evaluate(fixed_best)
        per_h = {}
        for j, c in enumerate(cps):
            e = tuple(1.0 if k == j else 0.0 for k in range(len(cps)))
            r = evaluate(e)
            per_h[c] = {k: r[k] for k in ("value_share", "transfer_share", "detected", "median_delay_h", "alerts_per_day", "precision")}
        diff = np.array(res_adapt["per_attempt_value"]) - np.array(res_fixed["per_attempt_value"])
        rg = np.random.default_rng(seed)
        boots = diff[rg.integers(0, len(diff), (5000, len(diff)))].mean(axis=1)
        row = {"seed": seed, "budget": B, "val_adaptive": best_val, "adaptive": res_adapt, "best_fixed_on_val": res_fixed,
               "diff_mean": float(diff.mean()), "diff_ci": [float(np.quantile(boots, .025)), float(np.quantile(boots, .975))],
               "per_h": per_h}
        rows.append(row)
        print(f"seed={seed} B={B}: adaptive value={res_adapt['value_share']:.3f} det={res_adapt['detected']:.2f} "
              f"delay={res_adapt['median_delay_h']} alpha={np.round(best, 2).tolist()} | fixed({fixed_best}) "
              f"value={res_fixed['value_share']:.3f} det={res_fixed['detected']:.2f} delay={res_fixed['median_delay_h']} | "
              f"diff={diff.mean():+.3f} CI[{row['diff_ci'][0]:+.3f},{row['diff_ci'][1]:+.3f}]", flush=True)
(out_dir / "cascade.json").write_text(json.dumps({"args": vars(args), "rows": rows}, indent=1))
