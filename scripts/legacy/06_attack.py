"""Timing-only evasion study. A detector is trained at each horizon on clean data, its alert threshold is
fixed on clean validation, and laundering attempts in the test window are then re-timed. Features are
recomputed from the perturbed stream; thresholds and weights never change."""
import argparse
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from amlh.features import build_graph, compute_features, feature_names  # noqa: E402
from amlh.metrics import threshold_for_budget  # noqa: E402
from amlh.perturb import perturb_stream  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default="HI-Small")
ap.add_argument("--max-day", type=int, default=9)
ap.add_argument("--horizons", default="0,1,3600,86400")
ap.add_argument("--budgets-W", default="3600,21600,86400")
ap.add_argument("--conditions", default="none,burst,stretch,shift_burst,shift_stretch")
ap.add_argument("--seeds", default="0,1,2")
ap.add_argument("--alerts", type=int, default=500)
ap.add_argument("--neg-rate", type=float, default=0.25)
ap.add_argument("--tag", required=True)
args = ap.parse_args()

P = ROOT / f"data/processed/{args.dataset}"
card = json.loads((P / "data_card.json").read_text())
ev = pl.read_parquet(P / "events.parquet")
if args.max_day is not None:
    ev = ev.filter(pl.col("t") < 86400 * (args.max_day + 1))
t0 = ev["t"].to_numpy()
src0, dst0 = ev["src"].to_numpy().astype(np.int64), ev["dst"].to_numpy().astype(np.int64)
usd0 = ev["usd"].to_numpy()
y0 = ev["label"].to_numpy().astype(np.int8)
paid0 = np.log1p(ev["amt_paid"].to_numpy()).astype(np.float32)
recv0 = np.log1p(ev["amt_recv"].to_numpy()).astype(np.float32)
cp0 = ev["cur_paid_id"].to_numpy().astype(np.float32)
cr0 = ev["cur_recv_id"].to_numpy().astype(np.float32)
fm0 = ev["fmt_id"].to_numpy().astype(np.float32)
N = len(t0)
i_day, j_day = card["split_days"]
day = t0 // 86400
split0 = np.where(day < i_day, 0, np.where(day < j_day, 1, 2)).astype(np.int8)
t_lo, t_hi = int(t0[split0 == 2].min()), int(t0.max())
FN = feature_names()
base_names = ["log_usd", "log_paid", "log_recv", "cur_paid", "cur_recv", "fmt", "same_cur", "is_self",
              "usd_vs_uin24", "usd_vs_uout24", "usd_vs_vin24", "usd_vs_vout24",
              "uout_vs_uin24", "vout_vs_vin24", "uoutcp_vs_uincp24", "voutcp_vs_vincp24", "hour"]
names = base_names + FN
ix = {k: FN.index(f"past24h_{k}") for k in ("u_in_amt", "u_out_amt", "v_in_amt", "v_out_amt", "u_in_cnt", "u_out_cnt",
                                            "v_in_cnt", "v_out_cnt", "u_in_ncp", "u_out_ncp", "v_in_ncp", "v_out_ncp")}
cats = [names.index(c) for c in ("cur_paid", "cur_recv", "fmt")]


def design(g, order, rows, H, view_end, t):
    """rows index the (possibly re-sorted) stream; order maps stream position -> original event."""
    orig = order[rows]
    F = compute_features(g, rows, H, view_end)
    lu = np.log1p(usd0[orig]).astype(np.float32)
    B = np.column_stack([lu, paid0[orig], recv0[orig], cp0[orig], cr0[orig], fm0[orig],
                         (cp0[orig] == cr0[orig]).astype(np.float32), (src0[orig] == dst0[orig]).astype(np.float32),
                         lu - F[:, ix["u_in_amt"]], lu - F[:, ix["u_out_amt"]], lu - F[:, ix["v_in_amt"]], lu - F[:, ix["v_out_amt"]],
                         F[:, ix["u_out_cnt"]] - F[:, ix["u_in_cnt"]], F[:, ix["v_out_cnt"]] - F[:, ix["v_in_cnt"]],
                         F[:, ix["u_out_ncp"]] - F[:, ix["u_in_ncp"]], F[:, ix["v_out_ncp"]] - F[:, ix["v_in_ncp"]],
                         ((t[rows] % 86400) // 3600).astype(np.float32)])
    return np.hstack([B, F])


# Attacked attempts: first event inside the test window, at least two events.
mem = pl.read_parquet(P / "pattern_members.parquet").filter(pl.col("pos") < N)
mem = mem.with_columns(pl.Series("mt", t0[mem["pos"].to_numpy()]))
att_tbl = pl.read_parquet(P / "attempts.parquet")
typ = dict(zip(att_tbl["pattern_id"].to_list(), att_tbl["typology"].to_list()))
agg = mem.group_by("pattern_id").agg(pl.col("mt").min().alias("f"), pl.len().alias("n"))
ids = agg.filter((pl.col("f") >= t_lo) & (pl.col("n") >= 2))["pattern_id"].sort().to_list()
attempts = {}
for pid, sub in mem.filter(pl.col("pattern_id").is_in(ids)).sort(["pattern_id", "pos"]).group_by("pattern_id", maintain_order=True):
    attempts[pid[0]] = sub["pos"].to_numpy()
attacked = np.concatenate(list(attempts.values()))
print(f"attacked attempts={len(attempts)} edges={len(attacked)} usd_total={usd0[attacked].sum():.3e}", flush=True)

g0 = build_graph(t0, src0, dst0, usd0)
identity = np.arange(N)
view_end0 = np.where(split0 == 0, i_day * 86400 - 1, np.where(split0 == 1, j_day * 86400 - 1, t_hi))
tr_pos = np.where((split0 == 0) & (y0 == 1))[0]
tr_neg = np.where((split0 == 0) & (y0 == 0))[0]
rng = np.random.default_rng(12345)
tr_rows = np.sort(np.r_[tr_pos, rng.choice(tr_neg, int(len(tr_neg) * args.neg_rate), replace=False)])
val_rows = np.where(split0 == 1)[0]
te_rows0 = np.where(split0 == 2)[0]
results = []
out = ROOT / "results" / args.tag
out.mkdir(parents=True, exist_ok=True)


def outcomes(s_by_event, t_by_event, H, tau):
    """Per attempt: detected, value share still interdictable, edge recall, completion time."""
    res = {}
    for pid, pos in attempts.items():
        flag = s_by_event[pos] >= tau
        ta = t_by_event[pos]
        ua = usd0[pos]
        det = bool(flag.any())
        if det:
            T = (ta[flag] + H).min()
            share = float(ua[ta > T].sum() / max(ua.sum(), 1e-12))
        else:
            share = 0.0
        res[pid] = dict(det=det, share=share, recall=float(flag.mean()), completion_h=float((ta.max() - ta.min()) / 3600))
    return res


for hs in args.horizons.split(","):
    H = int(hs)
    Xtr = design(g0, identity, tr_rows, H, view_end0, t0)
    Xva = design(g0, identity, val_rows, H, view_end0, t0)
    dtrain = lgb.Dataset(Xtr, y0[tr_rows].astype(np.float32), feature_name=names, categorical_feature=cats, free_raw_data=False)
    keep = np.where((y0[val_rows] == 1) | (np.random.default_rng(777).random(len(val_rows)) < 0.2))[0]
    dval = lgb.Dataset(Xva[keep], y0[val_rows][keep].astype(np.float32), feature_name=names, categorical_feature=cats,
                       reference=dtrain, free_raw_data=False)
    for seed in [int(s) for s in args.seeds.split(",")]:
        params = dict(objective="binary", metric="average_precision", learning_rate=0.05, num_leaves=63, min_data_in_leaf=20,
                      feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbosity=-1, num_threads=12,
                      seed=seed, feature_fraction_seed=seed, bagging_seed=seed, force_col_wise=True)
        booster = lgb.train(params, dtrain, num_boost_round=800, valid_sets=[dval], callbacks=[lgb.early_stopping(50, verbose=False)])
        s_val = booster.predict(Xva, num_iteration=booster.best_iteration)
        tau = threshold_for_budget(s_val, j_day - i_day, args.alerts)
        baseline = None
        for Ws in args.budgets_W.split(","):
            W = int(Ws)
            for cond in args.conditions.split(","):
                if cond == "none" and baseline is not None:
                    outc = baseline
                else:
                    t2 = perturb_stream(t0, attempts, W, cond, t_lo, t_hi, seed=seed)
                    order = np.lexsort((np.arange(N), t2))
                    t_s = t2[order]
                    g = build_graph(t_s, src0[order], dst0[order], usd0[order])
                    inv = np.empty(N, dtype=np.int64)
                    inv[order] = np.arange(N)
                    te_rows = np.sort(inv[te_rows0])
                    ve = view_end0[order]
                    # Clean val/train views never change because all attacked events stay in the test window.
                    X = design(g, order, te_rows, H, ve, t_s)
                    s = booster.predict(X, num_iteration=booster.best_iteration)
                    s_ev = np.zeros(N)
                    s_ev[order[te_rows]] = s
                    outc = outcomes(s_ev, t2, H, tau)
                    alerts_day = float((s >= tau).sum() / 2)
                    if cond == "none":
                        baseline = outc
                results.append({"H": hs, "seed": seed, "W": W, "cond": cond, "tau": tau,
                                "per_attempt": {str(k): v for k, v in outc.items()}})
                det = np.mean([v["det"] for v in outc.values()])
                share = np.mean([v["share"] for v in outc.values()])
                rec = np.mean([v["recall"] for v in outc.values()])
                compl = np.median([v["completion_h"] for v in outc.values()])
                print(f"H={hs:>6} s={seed} W={W:>6} {cond:<14} det={det:.3f} recall={rec:.3f} value={share:.3f} "
                      f"completion_med={compl:.1f}h", flush=True)
        (out / "attack.json").write_text(json.dumps({"args": vars(args), "typology": {str(k): typ[k] for k in attempts},
                                                     "results": results}))
print("done", flush=True)
