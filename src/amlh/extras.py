"""Two adapters that could not be exercised on the development machine. Both fail loudly if reality differs.

GFP: IBM Snap ML's GraphFeaturePreprocessor, on Linux only (its docs list Linux platforms, and Python support is unclear).
Ethereum: the XBlock MulDiGraph.pkl. Its field names (`isp`, `amount`, `timestamp`) come from memory of the dataset, not from
reading the file, so the loader checks them.
"""
from __future__ import annotations

import io
import json
import os
import pickle
from pathlib import Path

import numpy as np
import polars as pl

from .evaluate import heldout_metrics, validation_metrics
from .jobs_gbt import get_stream, read_scores, write_scores
from .prepare import multignn_split_days
from .runner import Context
from .sealing import is_sealed, score_health

GFP_PARAMS = {  # time windows as stated in Blanusa et al. for the AML datasets; bins are left at library defaults
    "fan": True, "fan_tw": 86400, "degree": True, "degree_tw": 86400, "scatter-gather": True, "scatter-gather_tw": 21600,
    "temp-cycle": True, "temp-cycle_tw": 86400, "lc-cycle": True, "lc-cycle_tw": 86400, "lc-cycle_len": 10,
    "vertex_stats": True, "vertex_stats_tw": 86400, "vertex_stats_cols": [3, 4], "time_window": 86400,
}
XGB_PARAMS = dict(objective="binary:logistic", tree_method="hist", eval_metric="aucpr", max_depth=8, eta=0.1, subsample=0.8,
                  colsample_bytree=0.8, scale_pos_weight=10.0)


def gfp_baseline(ctx: Context, cfg: dict) -> dict:
    """GFP features plus XGBoost, with test transactions fed in batches. Hyperparameters are fixed, not tuned as in the
    paper, so a shortfall against its published value is expected and is not evidence about GFP itself."""
    import snapml
    import xgboost as xgb
    s, _ = get_stream(cfg)
    gfp = snapml.GraphFeaturePreprocessor()
    gfp.set_params({**GFP_PARAMS, "num_threads": os.cpu_count()})
    edges = np.column_stack([np.arange(s.n, dtype=np.float64), s.src, s.dst, s.t, np.log1p(s.usd)]).astype(np.float64)
    base = np.column_stack([np.log1p(s.usd), s.cur_p, s.cur_r, s.fmt])

    def stream(rows, batch):
        out = []
        for i in range(0, len(rows), batch):
            ctx.check(0)
            out.append(gfp.transform(edges[rows[i:i + batch]])[:, 5:])
        return np.hstack([base[rows], np.vstack(out)])
    Xtr = stream(s.tr, cfg.get("train_batch", 2048))
    Xva = stream(s.va, cfg["batch"])
    Xte = stream(s.te, cfg["batch"])
    booster = xgb.train({**XGB_PARAMS, "seed": cfg["seed"], "nthread": os.cpu_count()}, xgb.DMatrix(Xtr, s.y[s.tr]), num_boost_round=cfg.get("rounds", 300),
                        evals=[(xgb.DMatrix(Xva, s.y[s.va]), "val")], early_stopping_rounds=30, verbose_eval=False)
    s_val = booster.predict(xgb.DMatrix(Xva), iteration_range=(0, booster.best_iteration + 1))
    s_te = booster.predict(xgb.DMatrix(Xte), iteration_range=(0, booster.best_iteration + 1))
    write_scores(ctx, "scores_val", s, s.va, s_val)
    write_scores(ctx, "scores_test", s, s.te, s_te)
    m = {"batch": cfg["batch"], "n_features": Xtr.shape[1], "test_health": score_health(s_te), **validation_metrics(s, s_val.astype(np.float64))}
    if not is_sealed(s.name):
        m["test_results"] = heldout_metrics(s, read_scores(ctx.dir, "scores_val", s, s.va), read_scores(ctx.dir, "scores_test", s, s.te), "0")
    return m


class _GraphOnlyUnpickler(pickle.Unpickler):
    """A public pickle can run code when loaded, so only graph containers may be constructed."""
    ALLOWED = {("networkx.classes.multidigraph", "MultiDiGraph"), ("networkx.classes.digraph", "DiGraph"),
               ("networkx.classes.graph", "Graph"), ("networkx.classes.multigraph", "MultiGraph"),
               ("networkx.classes.reportviews", "NodeView"), ("networkx.classes.reportviews", "EdgeView"),
               ("collections", "defaultdict"), ("collections", "OrderedDict"), ("builtins", "dict"), ("builtins", "list")}

    def find_class(self, module, name):
        if (module, name) in self.ALLOWED or module.startswith("networkx.classes.coreviews"):
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"refusing to load {module}.{name}")


def prepare_ethereum(pkl: Path, out: Path) -> dict:
    """Edge label: destination account is labelled phishing, the transaction-level proxy GFP used. No attempts exist,
    so this dataset supports horizon curves only, never stoppable value."""
    out.mkdir(parents=True, exist_ok=True)
    with open(pkl, "rb") as fh:
        graph = _GraphOnlyUnpickler(io.BytesIO(fh.read())).load()
    nodes = list(graph.nodes)
    index = {n: i for i, n in enumerate(nodes)}
    phishing = np.array([int(graph.nodes[n].get("isp", -1)) for n in nodes])
    if (phishing < 0).any():
        raise ValueError("node attribute 'isp' is missing: the file format differs from what this loader expects")
    rows = [(index[u], index[v], float(d["amount"]), int(d["timestamp"])) for u, v, d in graph.edges(data=True)]
    arr = np.array(rows)
    t = arr[:, 3].astype(np.int64)
    t = t - (t.min() // 86400) * 86400
    order = np.lexsort((np.arange(len(t)), t))
    day = (t // 86400).astype(int)
    daily = np.bincount(day)
    i, j = multignn_split_days(daily)
    d = day[order]
    amount = arr[:, 2][order]
    events = pl.DataFrame({
        "pos": np.arange(len(t), dtype=np.int32), "event_id": order.astype(np.int64), "t": t[order], "src": arr[:, 0][order].astype(np.int32),
        "dst": arr[:, 1][order].astype(np.int32), "amt_paid": amount, "amt_recv": amount, "usd": amount, "cur_paid_id": np.zeros(len(t), dtype=np.int16),
        "cur_recv_id": np.zeros(len(t), dtype=np.int16), "fmt_id": np.zeros(len(t), dtype=np.int16),
        "label": phishing[arr[:, 1][order].astype(int)].astype(np.int8), "split": np.where(d < i, 0, np.where(d < j, 1, 2)).astype(np.int8)})
    events.write_parquet(out / "events.parquet")
    pl.DataFrame(schema={"pattern_id": pl.Int32, "event_id": pl.UInt32, "pos": pl.Int32, "t": pl.Int64, "usd": pl.Float64, "label": pl.Int8, "split": pl.Int8}).write_parquet(out / "pattern_members.parquet")
    pl.DataFrame(schema={"pattern_id": pl.Int32, "typology": pl.String, "n_lines": pl.Int32, "n_matched": pl.Int32}).write_parquet(out / "attempts.parquet")
    card = {"n_events": events.height, "n_accounts": len(nodes), "n_illicit": int(events["label"].sum()), "split_days": [int(i), int(j)],
            "documented_days": None, "daily_totals": daily.tolist(), "attempts": 0, "label_is_proxy": True}
    (out / "data_card.json").write_text(json.dumps(card, indent=1))
    return card


REGISTRY = {"gfp_baseline": gfp_baseline}
