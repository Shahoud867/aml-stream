"""Decision-horizon sweep with a gradient-boosted detector.

The model is retrained at every horizon: a detector deployed with horizon H is trained on the
same view it will serve with, so a gain or loss cannot be a train/serve mismatch artefact.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from amlh.features import INF_H, build_graph, compute_features, feature_names  # noqa: E402
from amlh.metrics import (attempt_outcomes, best_f1_threshold, bootstrap_mean_ci, event_metrics,  # noqa: E402
                          summarize_attempts, threshold_for_budget)

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default="HI-Small")
ap.add_argument("--max-day", type=int, default=None,
                help="drop events on later days (9 = the documented Sep 1-10 window)")
ap.add_argument("--train-days", type=int, default=None)
ap.add_argument("--val-days", type=int, default=None)
ap.add_argument("--horizons", default="0,60,600,3600,21600,86400,259200,inf")
ap.add_argument("--seeds", default="0,1,2")
ap.add_argument("--neg-rate", type=float, default=0.05)
ap.add_argument("--tag", required=True)
ap.add_argument("--budgets", default="100,500,2000")
ap.add_argument("--save-scores", action="store_true")
ap.add_argument("--drop", default=None, help="regex of feature names to remove (ablations)")
args = ap.parse_args()

P = ROOT / f"data/processed/{args.dataset}"
card = json.loads((P / "data_card.json").read_text())
ev = pl.read_parquet(P / "events.parquet")
if args.max_day is not None:
    ev = ev.filter(pl.col("t") < 86400 * (args.max_day + 1))
mem = pl.read_parquet(P / "pattern_members.parquet")
att = pl.read_parquet(P / "attempts.parquet")

# Positions must stay canonical and contiguous after truncation (the file is already sorted).
t = ev["t"].to_numpy()
src, dst = ev["src"].to_numpy().astype(np.int64), ev["dst"].to_numpy().astype(np.int64)
usd = ev["usd"].to_numpy()
y = ev["label"].to_numpy().astype(np.int8)
N = len(t)
i_day, j_day = card["split_days"]
if args.train_days is not None:
    i_day, j_day = args.train_days, args.train_days + args.val_days
day = t // 86400
split = np.where(day < i_day, 0, np.where(day < j_day, 1, 2)).astype(np.int8)
t_end = t.max()
view_end = np.where(split == 0, i_day * 86400 - 1, np.where(split == 1, j_day * 86400 - 1, t_end))

tr_pos = np.where((split == 0) & (y == 1))[0]
tr_neg = np.where((split == 0) & (y == 0))[0]
val_rows = np.where(split == 1)[0]
te_rows = np.where(split == 2)[0]
rng = np.random.default_rng(12345)
tr_neg_s = rng.choice(tr_neg, int(len(tr_neg) * args.neg_rate), replace=False)
tr_rows = np.sort(np.r_[tr_pos, tr_neg_s])
print(f"N={N} train={tr_rows.size}(pos {tr_pos.size}) val={val_rows.size}(pos {y[val_rows].sum()}) "
      f"test={te_rows.size}(pos {y[te_rows].sum()}) days train<{i_day} val<{j_day} tmax={t_end}", flush=True)

g = build_graph(t, src, dst, usd)
FN = feature_names()
base_names = ["log_usd", "log_paid", "log_recv", "cur_paid", "cur_recv", "fmt", "same_cur", "is_self",
              "usd_vs_uin24", "usd_vs_uout24", "usd_vs_vin24", "usd_vs_vout24",
              "uout_vs_uin24", "vout_vs_vin24", "uoutcp_vs_uincp24", "voutcp_vs_vincp24", "hour"]
CAT = ["cur_paid", "cur_recv", "fmt"]
names = base_names + FN
ix = {k: FN.index(f"past24h_{k}") for k in ("u_in_amt", "u_out_amt", "v_in_amt", "v_out_amt", "u_in_cnt", "u_out_cnt",
                                             "v_in_cnt", "v_out_cnt", "u_in_ncp", "u_out_ncp", "v_in_ncp", "v_out_ncp")}
hour = ((t % 86400) // 3600).astype(np.float32)
log_usd = np.log1p(usd).astype(np.float32)
log_paid = np.log1p(ev["amt_paid"].to_numpy()).astype(np.float32)
log_recv = np.log1p(ev["amt_recv"].to_numpy()).astype(np.float32)
cp = ev["cur_paid_id"].to_numpy().astype(np.float32)
cr = ev["cur_recv_id"].to_numpy().astype(np.float32)
fm = ev["fmt_id"].to_numpy().astype(np.float32)


keep_cols = None
if args.drop:
    import re
    keep_cols = [k for k, n in enumerate(names) if not re.search(args.drop, n)]
    print(f"ablation --drop {args.drop!r}: {len(names) - len(keep_cols)} of {len(names)} features removed", flush=True)
    names_all, names = names, [names[k] for k in keep_cols]
    CAT = [c for c in CAT if c in names]


def design(rows, H):
    F = compute_features(g, rows, H, view_end)
    B = np.column_stack([log_usd[rows], log_paid[rows], log_recv[rows], cp[rows], cr[rows], fm[rows],
                         (cp[rows] == cr[rows]).astype(np.float32), (src[rows] == dst[rows]).astype(np.float32),
                         log_usd[rows] - F[:, ix["u_in_amt"]], log_usd[rows] - F[:, ix["u_out_amt"]],
                         log_usd[rows] - F[:, ix["v_in_amt"]], log_usd[rows] - F[:, ix["v_out_amt"]],
                         F[:, ix["u_out_cnt"]] - F[:, ix["u_in_cnt"]], F[:, ix["v_out_cnt"]] - F[:, ix["v_in_cnt"]],
                         F[:, ix["u_out_ncp"]] - F[:, ix["u_in_ncp"]], F[:, ix["v_out_ncp"]] - F[:, ix["v_in_ncp"]],
                         hour[rows]])
    X = np.hstack([B, F])
    return X if keep_cols is None else X[:, keep_cols]


# Attempts whose first member falls in the test window (all members scored).
pos_max = N
m = mem.filter(pl.col("pos") < pos_max)
m = m.with_columns(pl.Series("mt", t[m["pos"].to_numpy()]))
first = m.group_by("pattern_id").agg(pl.col("mt").min().alias("first"))
test_start = int(t[te_rows[0]])
eligible = first.filter(pl.col("first") >= test_start)["pattern_id"]
m_eval = m.filter(pl.col("pattern_id").is_in(eligible.to_list()))
A_id = m_eval["pattern_id"].to_numpy()
A_pos = m_eval["pos"].to_numpy()
pos_to_te = np.full(N, -1, dtype=np.int64)
pos_to_te[te_rows] = np.arange(te_rows.size)
typ_of = dict(zip(att["pattern_id"].to_list(), att["typology"].to_list()))
print(f"attempts starting in test: {len(np.unique(A_id))}", flush=True)

core_mask = t[te_rows] < 86400 * 10
out_dir = ROOT / "results" / args.tag
out_dir.mkdir(parents=True, exist_ok=True)
seeds = [int(s) for s in args.seeds.split(",")]
budgets = [float(b) for b in args.budgets.split(",")]
results = []
val_days = (j_day - i_day)
for hs in args.horizons.split(","):
    H = INF_H if hs == "inf" else int(hs)
    t0 = time.time()
    Xtr = design(tr_rows, H)
    Xva = design(val_rows, H)
    Xte = design(te_rows, H)
    t_feat = time.time() - t0
    ytr, yva, yte = y[tr_rows], y[val_rows], y[te_rows]
    cats = [names.index(c) for c in CAT]
    dtrain = lgb.Dataset(Xtr, ytr.astype(np.float32), feature_name=names, categorical_feature=cats, free_raw_data=False)
    # Early stopping on all validation positives plus a fixed negative subsample.
    r2 = np.random.default_rng(777)
    keep = np.where((yva == 1) | (r2.random(len(yva)) < 0.2))[0]
    dval = lgb.Dataset(Xva[keep], yva[keep].astype(np.float32), feature_name=names, categorical_feature=cats, reference=dtrain,
                       free_raw_data=False)
    for seed in seeds:
        t1 = time.time()
        params = dict(objective="binary", metric="average_precision", learning_rate=0.05, num_leaves=63,
                      min_data_in_leaf=20, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
                      lambda_l2=1.0, verbosity=-1, num_threads=12, seed=seed, feature_fraction_seed=seed,
                      bagging_seed=seed, force_col_wise=True)
        booster = lgb.train(params, dtrain, num_boost_round=800, valid_sets=[dval],
                            callbacks=[lgb.early_stopping(50, verbose=False)])
        s_val = booster.predict(Xva, num_iteration=booster.best_iteration)
        s_te = booster.predict(Xte, num_iteration=booster.best_iteration)
        tau, f1_val = best_f1_threshold(yva, s_val)
        row = {"H": hs, "seed": seed, "best_iter": booster.best_iteration, "tau": tau, "val_f1": f1_val,
               "val_pr_auc": float(event_metrics(yva, s_val, tau)["pr_auc"]),
               "t_feat_s": round(t_feat, 1), "t_train_s": round(time.time() - t1, 1)}
        row["test"] = event_metrics(yte, s_te, tau)
        if (~core_mask).any():
            row["test_core"] = event_metrics(yte[core_mask], s_te[core_mask], tau)
            row["test_tail"] = event_metrics(yte[~core_mask], s_te[~core_mask], tau)
        # Attempt-level outcomes at F1 threshold and at alert budgets.
        idx = pos_to_te[A_pos]
        ok = idx >= 0
        batch_end = int(t_end) if hs == "inf" else None
        H_eff = 0 if hs == "inf" else H
        res = {"f1_tau": summarize_attempts(attempt_outcomes(A_id[ok], t[A_pos[ok]], usd[A_pos[ok]], s_te[idx[ok]],
                                                            tau, H_eff, batch_end))}
        for b in budgets:
            tb = threshold_for_budget(s_val, val_days, b)
            o = attempt_outcomes(A_id[ok], t[A_pos[ok]], usd[A_pos[ok]], s_te[idx[ok]], tb, H_eff, batch_end)
            summ = summarize_attempts(o)
            summ["value_share_ci"] = bootstrap_mean_ci(o["value_share"], seed=seed)
            summ["alerts_per_day_test"] = float((s_te >= tb).sum() / max((t[te_rows].max() - t[te_rows].min()) / 86400, 1e-9))
            summ["alerts_per_day_core"] = float((s_te[core_mask] >= tb).sum() / 2)
            res[f"budget_{int(b)}"] = summ
            # Per-attempt arrays support paired analysis across horizons.
        row["attempts"] = res
        results.append(row)
        if args.save_scores:
            np.savez_compressed(out_dir / f"scores_H{hs}_s{seed}.npz", val=s_val.astype(np.float32),
                                test=s_te.astype(np.float32))
        tm = row["test"]
        tc = row.get("test_core", tm)
        print(f"H={hs:>7} seed={seed} iter={booster.best_iteration:4d} val_f1={f1_val:.4f} "
              f"test F1={tm['f1']:.4f} PR={tm['pr_auc']:.4f} | core F1={tc['f1']:.4f} PR={tc['pr_auc']:.4f} | "
              f"det@500={res['budget_500']['detected_frac']:.2f} value@500={res['budget_500']['value_share_mean']:.3f} "
              f"feat={t_feat:.0f}s train={row['t_train_s']:.0f}s", flush=True)
        (out_dir / "results.json").write_text(json.dumps(
            {"args": vars(args), "attempts_eval": int(len(np.unique(A_id))), "rows": results}, indent=1))
    del Xtr, Xva, Xte, dtrain, dval
print("done", flush=True)
