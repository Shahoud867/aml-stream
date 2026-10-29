"""Behavioural checks that a leak would fail, and feature attribution for the horizon gain.

  shuffled:  training labels permuted. Test PR-AUC must fall to about prevalence at every H.
  importance: top gain features at each H, split by block, to explain where the gain comes from.
"""
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from amlh.features import INF_H, build_graph, compute_features, feature_names  # noqa: E402

P = ROOT / "data/processed/HI-Small"
card = json.loads((P / "data_card.json").read_text())
ev = pl.read_parquet(P / "events.parquet")
t = ev["t"].to_numpy()
src, dst = ev["src"].to_numpy().astype(np.int64), ev["dst"].to_numpy().astype(np.int64)
usd = ev["usd"].to_numpy()
y = ev["label"].to_numpy().astype(np.int8)
i_d, j_d = card["split_days"]
day = t // 86400
split = np.where(day < i_d, 0, np.where(day < j_d, 1, 2))
view_end = np.where(split == 0, i_d * 86400 - 1, np.where(split == 1, j_d * 86400 - 1, t.max()))
g = build_graph(t, src, dst, usd)
names = feature_names()
rng = np.random.default_rng(12345)
tr = np.where(split == 0)[0]
tr = np.sort(np.r_[tr[y[tr] == 1], rng.choice(tr[y[tr] == 0], int((y[tr] == 0).sum() * 0.25), replace=False)])
te = np.where(split == 2)[0]
core = t[te] < 864000
base_prev = {"test": float(y[te].mean()), "core": float(y[te][core].mean())}
params = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=20, feature_fraction=0.8,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbosity=-1, num_threads=12, seed=0, force_col_wise=True)
out = {"prevalence": base_prev, "shuffled": {}, "importance": {}}


def block(n):
    if n.startswith("past") and "_pair" not in n:
        return "past"
    if n.startswith("fwd") and "_pair" not in n:
        return "forward"
    if "_pair_" in n:
        return "pair"
    return "motif"


for hs, H in [("0", 0), ("1", 1), ("3600", 3600), ("inf", INF_H)]:
    Xtr = compute_features(g, tr, H, view_end)
    Xte = compute_features(g, te, H, view_end)
    # Shuffled-label collapse.
    y_shuf = y[tr].astype(np.float32).copy()
    rng.shuffle(y_shuf)
    b0 = lgb.train(params, lgb.Dataset(Xtr, y_shuf), 150)
    s0 = b0.predict(Xte)
    out["shuffled"][hs] = {"test_pr_auc": float(average_precision_score(y[te], s0)),
                           "core_pr_auc": float(average_precision_score(y[te][core], s0[core]))}
    # Attribution on the true labels (graph features only, so the block shares are comparable across H).
    b1 = lgb.train(params, lgb.Dataset(Xtr, y[tr].astype(np.float32), feature_name=names), 200)
    gain = b1.feature_importance("gain")
    total = gain.sum()
    shares = {}
    for k, n in enumerate(names):
        shares[block(n)] = shares.get(block(n), 0.0) + float(gain[k] / total)
    top = [(names[k], round(float(gain[k] / total), 4)) for k in np.argsort(-gain)[:12]]
    s1 = b1.predict(Xte)
    out["importance"][hs] = {"blocks": shares, "top": top, "graph_only_core_pr_auc": float(average_precision_score(y[te][core], s1[core]))}
    print(hs, out["shuffled"][hs], out["importance"][hs]["blocks"], flush=True)
    print("   top:", top[:8], flush=True)
    (ROOT / "results/sanity.json").write_text(json.dumps(out, indent=1))
print("prevalence", base_prev)
