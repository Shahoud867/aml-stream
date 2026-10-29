"""Model-free audit of the benchmark's temporal structure (no learning involved)."""
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
name = sys.argv[1] if len(sys.argv) > 1 else "HI-Small"
P = ROOT / f"data/processed/{name}"
ev = pl.read_parquet(P / "events.parquet")
att = pl.read_parquet(P / "attempts.parquet")
mem = pl.read_parquet(P / "pattern_members.parquet")
out = {}

day = ev["t"].to_numpy() // 86400
lab = ev["label"].to_numpy()
spl = ev["split"].to_numpy()
rows = []
for d in np.unique(day):
    m = day == d
    rows.append({"day": int(d), "split": int(spl[m][0]), "n": int(m.sum()), "illicit": int(lab[m].sum()),
                 "rate_pct": round(100 * lab[m].mean(), 3)})
print(pl.DataFrame(rows))
out["daily"] = rows

te = spl == 2
t_te = ev["t"].to_numpy()[te]
t_max = t_te.max()
implicit = (t_max - t_te) / 3600
print("implicit horizon in test (hours), all: median %.1f p10 %.1f p90 %.1f" % (
    np.median(implicit), *np.percentile(implicit, [10, 90])))
ill = lab[te] == 1
print("implicit horizon, illicit: median %.1f p10 %.1f p90 %.1f" % (
    np.median(implicit[ill]), *np.percentile(implicit[ill], [10, 90])))
out["implicit_horizon_h"] = {
    "all_median": float(np.median(implicit)), "illicit_median": float(np.median(implicit[ill])),
    "all_p10": float(np.percentile(implicit, 10)), "all_p90": float(np.percentile(implicit, 90)),
    "illicit_p10": float(np.percentile(implicit[ill], 10)), "illicit_p90": float(np.percentile(implicit[ill], 90))}

# How much of the test split's illicit mass sits where background traffic has ended.
core_end = 86400 * 10  # days 0-9 carry the background traffic; verified in the table above
tail = ev.filter((pl.col("split") == 2) & (pl.col("t") >= core_end))
core = ev.filter((pl.col("split") == 2) & (pl.col("t") < core_end))
print(f"test core (d8-9): n={core.height} illicit={core['label'].sum()} rate={100*core['label'].mean():.3f}%")
print(f"test tail (d10+): n={tail.height} illicit={tail['label'].sum()} rate={100*tail['label'].mean():.3f}%")
out["test_core"] = {"n": core.height, "illicit": int(core["label"].sum()), "rate_pct": 100 * float(core["label"].mean())}
out["test_tail"] = {"n": tail.height, "illicit": int(tail["label"].sum()), "rate_pct": 100 * float(tail["label"].mean())}

# Trivial detector on the tail: "flag everything after day 10".
tp = tail["label"].sum()
fp = tail.height - tp
fn = core["label"].sum()
f1 = 2 * tp / (2 * tp + fp + fn)
print(f"trivial 'flag all after day 10' test F1 = {100*f1:.2f}%  (P={100*tp/(tp+fp):.1f}, R={100*tp/(tp+fn):.1f})")
out["trivial_tail_flag_f1_pct"] = 100 * float(f1)

# Attempt timing.
a = att.with_columns(((pl.col("t_last") - pl.col("t_first")) / 3600).alias("span_h"))
print(a.group_by("typology").agg(
    pl.len().alias("attempts"), pl.col("n_matched").median().alias("edges_med"),
    pl.col("span_h").median().round(1).alias("span_h_med"), pl.col("span_h").quantile(0.9).round(1).alias("span_h_p90"),
    (pl.col("split_first") == 2).sum().alias("start_in_test"),
    ((pl.col("split_first") < 2) & (pl.col("split_last") == 2)).sum().alias("straddle_into_test")).sort("typology"))
out["attempts_start_in_test"] = int((a["split_first"] == 2).sum())
out["attempts_straddling_test"] = int(((a["split_first"] < 2) & (a["split_last"] == 2)).sum())

# For each test-split pattern edge: what fraction of its attempt lies strictly in its future?
m = mem.sort(["pattern_id", "t", "pos"])
g = m.with_columns(
    pl.col("t").rank("ordinal").over("pattern_id").alias("r"),
    pl.len().over("pattern_id").alias("n"),
    pl.col("t").gt(pl.col("t").min()).alias("not_first"))
g = g.with_columns(((pl.col("n") - pl.col("r")) / (pl.col("n") - 1).clip(1)).alias("frac_future"))
print("test pattern edges: mean fraction of own attempt still in the future at arrival: %.3f"
      % g.filter(pl.col("split") == 2)["frac_future"].mean())

# Local look-ahead exposure: among edges adjacent to a test edge's endpoints (1-hop, any direction),
# what share occurs after it within the test graph?  This is what an unrestricted sampler can draw on.
src, dst, t = ev["src"].to_numpy(), ev["dst"].to_numpy(), ev["t"].to_numpy()
n_nodes = int(max(src.max(), dst.max())) + 1
deg_total = np.bincount(src, minlength=n_nodes) + np.bincount(dst, minlength=n_nodes)
order = np.arange(len(t))
seen = np.zeros(n_nodes, dtype=np.int64)
past = np.zeros(len(t), dtype=np.int64)
for k in order:  # O(N); canonical order, ties treated as past-or-equal by position
    past[k] = seen[src[k]] + seen[dst[k]]
    seen[src[k]] += 1
    seen[dst[k]] += 1
adj_total = deg_total[src] + deg_total[dst] - 2
future = np.maximum(adj_total - past, 0)
frac = future / np.maximum(adj_total, 1)
for label, mask in (("benign", te & (lab == 0)), ("illicit", te & (lab == 1))):
    print(f"1-hop neighbours in the future, {label}: mean {frac[mask].mean():.3f}, median {np.median(frac[mask]):.3f}")
    out[f"future_1hop_frac_{label}"] = {"mean": float(frac[mask].mean()), "median": float(np.median(frac[mask]))}

# Share of attempts with enough test events for attempt-level analysis.
full = a.filter((pl.col("split_first") == 2) & (pl.col("split_last") == 2))
print(f"attempts fully inside test: {full.height}; by typology:")
print(full.group_by("typology").agg(pl.len().alias("n")).sort("typology"))
out["attempts_full_test"] = full.height
(P / "audit.json").write_text(json.dumps(out, indent=1))
