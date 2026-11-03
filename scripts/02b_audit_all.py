"""Cross-dataset tail audit: per-day volume and the Multi-GNN split, from the raw CSVs (streaming)."""
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from amlh.prepare import multignn_split_days  # noqa: E402

results = {}
for name in sys.argv[1:]:
    q = (pl.scan_csv(ROOT / f"data/raw/{name}_Trans.csv", infer_schema_length=0)
         .select(pl.col("Timestamp").str.slice(0, 10).alias("d"), pl.col("Is Laundering").cast(pl.Int8).alias("y"))
         .group_by("d").agg(pl.len().alias("n"), pl.col("y").sum().alias("illicit")).sort("d"))
    df = q.collect(engine="streaming")
    # Fill any gap days so day indices are contiguous, as in Multi-GNN's day loop.
    dates = pl.date_range(pl.lit(df["d"][0]).str.to_date("%Y/%m/%d"), pl.lit(df["d"][-1]).str.to_date("%Y/%m/%d"),
                          eager=True).dt.strftime("%Y/%m/%d").alias("d").to_frame()
    df = dates.join(df, on="d", how="left").fill_null(0)
    n, ill = df["n"].to_numpy(), df["illicit"].to_numpy()
    i, j = multignn_split_days(n)
    # "Background end" = first day whose volume falls below 1% of the median of the preceding days.
    med = np.array([np.median(n[:k]) if k else n[0] for k in range(len(n))])
    tail_start = next((k for k in range(1, len(n)) if n[k] < 0.01 * med[k]), len(n))
    te_n, te_ill = n[j:].sum(), ill[j:].sum()
    tail_n, tail_ill = n[max(tail_start, j):].sum(), ill[max(tail_start, j):].sum()
    tp, fp, fn = tail_ill, tail_n - tail_ill, te_ill - tail_ill
    f1 = 2 * tp / max(2 * tp + fp + fn, 1)
    results[name] = dict(
        days=len(n), first=df["d"][0], last=df["d"][-1], split_days=[int(i), int(j)],
        n_total=int(n.sum()), n_illicit=int(ill.sum()), illicit_pct=100 * float(ill.sum() / n.sum()),
        train_pct=100 * float(n[:i].sum() / n.sum()), val_pct=100 * float(n[i:j].sum() / n.sum()),
        test_pct=100 * float(te_n / n.sum()),
        test_days=len(n) - int(j), tail_start_day=int(tail_start), tail_start_date=df["d"][min(tail_start, len(n) - 1)],
        test_events=int(te_n), test_illicit=int(te_ill), test_illicit_pct=100 * float(te_ill / te_n),
        tail_events=int(tail_n), tail_illicit=int(tail_ill),
        tail_illicit_rate_pct=100 * float(tail_ill / max(tail_n, 1)),
        tail_share_of_test_illicit_pct=100 * float(tail_ill / max(te_ill, 1)),
        tail_share_of_test_events_pct=100 * float(tail_n / max(te_n, 1)),
        core_illicit_rate_pct=100 * float((te_ill - tail_ill) / max(te_n - tail_n, 1)),
        trivial_flag_tail_f1_pct=100 * float(f1),
        trivial_precision_pct=100 * float(tp / max(tp + fp, 1)), trivial_recall_pct=100 * float(tp / max(tp + fn, 1)),
        daily_n=n.tolist(), daily_illicit=ill.tolist())
    r = results[name]
    print(f"{name}: days={r['days']} ({r['first']}..{r['last']}) split={r['split_days']} "
          f"tr/va/te={r['train_pct']:.1f}/{r['val_pct']:.1f}/{r['test_pct']:.1f} "
          f"tail from {r['tail_start_date']} n={r['tail_events']} rate={r['tail_illicit_rate_pct']:.1f}% "
          f"share_of_test_illicit={r['tail_share_of_test_illicit_pct']:.1f}% core_rate={r['core_illicit_rate_pct']:.3f}% "
          f"trivial_F1={r['trivial_flag_tail_f1_pct']:.2f}%", flush=True)
(ROOT / "results").mkdir(exist_ok=True)
(ROOT / "results/tail_audit.json").write_text(json.dumps(results, indent=1))
