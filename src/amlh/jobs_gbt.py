"""Job functions for the boosted detector: horizon sweeps, leakage checks, ablations, alert policy, attacks, service time.

Every job writes scores keyed by event_id. Test metrics are computed only for unsealed datasets, and otherwise
in the merge step. The LightGBM configuration below is frozen by DECISIONS.md D-001.
"""
from __future__ import annotations

import itertools
import json
import os
import time
from pathlib import Path

import numpy as np
import polars as pl

from .datasets import Stream, attempts_in_window, load_stream
from .design import CATEGORICAL, design, horizon_seconds, keep_columns
from .evaluate import equal_alert_threshold, heldout_metrics, validation_metrics
from .features import build_graph
from .metrics import threshold_for_budget
from .perturb import perturb_stream
from .prepare import ensure_prepared
from .queue_sim import max_stable_rate, service_from_profile
from .runner import Context, assert_expected_rows, read_json
from .sealing import is_sealed, score_health

LGB_PARAMS = dict(objective="binary", metric="average_precision", learning_rate=0.05, num_leaves=63, min_data_in_leaf=20,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbosity=-1, force_col_wise=True)
MAX_ROUNDS = 800
EARLY_STOP = 50
SAMPLE_SEED = 12345
VAL_STOP_SEED = 777
VAL_STOP_NEG = 0.2


class Paths:
    """Machine-specific locations. Deliberately outside job configs, so the same job hashes equally anywhere."""
    raw: dict[str, Path] = {}
    cache: Path = Path("cache")


_stream_cache: dict = {}


def get_stream(cfg: dict) -> tuple[Stream, dict]:
    """One dataset is kept in memory at a time; a notebook works through datasets in sequence."""
    key = (cfg["dataset"], cfg.get("protocol", "released"), tuple(cfg["fold"]) if cfg.get("fold") else None)
    if key not in _stream_cache:
        _stream_cache.clear()
        ensure_prepared(cfg["dataset"], Paths.raw[cfg["dataset"]], Paths.cache)
        s = load_stream(Paths.cache, cfg["dataset"], key[1], key[2])
        _stream_cache[key] = (s, build_graph(s.t, s.src, s.dst, s.usd))
    return _stream_cache[key]


def write_scores(ctx: Context, name: str, s: Stream, rows: np.ndarray, scores: np.ndarray):
    assert_expected_rows(len(scores), len(rows), name)
    if np.isnan(scores).any():
        raise ValueError(f"{name}: NaN scores")
    pl.DataFrame({"event_id": s.event_id[rows], "score": np.asarray(scores, dtype=np.float32)}).write_parquet(ctx.dir / f"{name}.parquet")


def read_scores(job_dir: Path, name: str, s: Stream, rows: np.ndarray) -> np.ndarray:
    """Scores aligned to rows, whichever order they were written in. Missing events are an error."""
    df = pl.read_parquet(Path(job_dir) / f"{name}.parquet")
    want = pl.DataFrame({"event_id": s.event_id[rows]}).with_row_index("i")
    joined = want.join(df, on="event_id", how="left").sort("i")
    if joined["score"].null_count():
        raise ValueError(f"{job_dir}/{name}: {joined['score'].null_count()} events have no score")
    return joined["score"].to_numpy().astype(np.float64)


def train_rows(s: Stream, neg_rate: float) -> np.ndarray:
    tr = s.tr
    pos, neg = tr[s.y[tr] == 1], tr[s.y[tr] == 0]
    rng = np.random.default_rng(SAMPLE_SEED)
    return np.sort(np.r_[pos, rng.choice(neg, int(len(neg) * neg_rate), replace=False)])


def fit_lgb(s: Stream, Xtr, ytr, Xva, names, seed: int, threads: int | None = None):
    import lightgbm as lgb
    params = dict(LGB_PARAMS, num_threads=threads or os.cpu_count(), seed=seed, feature_fraction_seed=seed, bagging_seed=seed)
    cats = [names.index(c) for c in CATEGORICAL if c in names]
    yva = s.y[s.va]
    keep = np.where((yva == 1) | (np.random.default_rng(VAL_STOP_SEED).random(len(yva)) < VAL_STOP_NEG))[0]
    dtrain = lgb.Dataset(Xtr, ytr.astype(np.float32), feature_name=names, categorical_feature=cats, free_raw_data=False)
    dval = lgb.Dataset(Xva[keep], yva[keep].astype(np.float32), feature_name=names, categorical_feature=cats, reference=dtrain, free_raw_data=False)
    return lgb.train(params, dtrain, num_boost_round=MAX_ROUNDS, valid_sets=[dval], callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False)])


def gbt_sweep(ctx: Context, cfg: dict) -> dict:
    """One detector at one horizon: train, score validation and test, and write the scores."""
    s, g = get_stream(cfg)
    H = horizon_seconds(cfg["H"])
    family = cfg.get("family", "wall")
    names, keep = keep_columns(family, cfg.get("drop"))
    tr = train_rows(s, cfg.get("neg_rate", 0.25))
    t0 = time.time()
    Xtr = design(s, g, tr, H, s.view_end, s.t, family, keep=keep)
    Xva = design(s, g, s.va, H, s.view_end, s.t, family, keep=keep)
    Xte = design(s, g, s.te, H, s.view_end, s.t, family, keep=keep)
    t_feat = time.time() - t0
    ytr = s.y[tr].copy()
    if cfg.get("shuffle_labels"):
        np.random.default_rng(cfg["seed"] + 1000).shuffle(ytr)
    ctx.check(600)
    t1 = time.time()
    booster = fit_lgb(s, Xtr, ytr, Xva, names, cfg["seed"])
    s_val = booster.predict(Xva, num_iteration=booster.best_iteration)
    s_te = booster.predict(Xte, num_iteration=booster.best_iteration)
    write_scores(ctx, "scores_val", s, s.va, s_val)
    write_scores(ctx, "scores_test", s, s.te, s_te)
    m = {"best_iter": int(booster.best_iteration), "t_feat_s": round(t_feat, 1), "t_train_s": round(time.time() - t1, 1),
         "n_features": len(names), "test_health": score_health(s_te), **validation_metrics(s, s_val)}
    if not is_sealed(s.name):
        m["test_results"] = heldout_metrics(s, s_val, s_te, cfg["H"])
    return m


def _alpha_grid(n: int, step: float) -> list[tuple]:
    grid = np.arange(0, 1 + 1e-9, step)
    return [a for a in itertools.product(grid, repeat=n) if abs(sum(a) - 1) < 1e-9]


def spend(scores: list[np.ndarray], alpha, total_alerts: int) -> list[float]:
    """Spends a budget across checkpoints in order; returns each checkpoint's threshold."""
    alerted = np.zeros(len(scores[0]), dtype=bool)
    taus = []
    for sc, a in zip(scores, alpha):
        k = int(round(a * total_alerts))
        if k <= 0:
            taus.append(float("inf"))
            continue
        cand = sc[~alerted]
        k = min(k, len(cand))
        if k == 0:  # tied scores can alert every row at an earlier checkpoint, leaving nothing to threshold
            taus.append(float("inf"))
            continue
        tau = float(np.partition(cand, len(cand) - k)[len(cand) - k])
        taus.append(tau)
        alerted |= sc >= tau
    return taus


def first_alert(scores: list[np.ndarray], taus: list[float]) -> np.ndarray:
    fc = np.full(len(scores[0]), -1, dtype=np.int64)
    for j, (sc, tau) in enumerate(zip(scores, taus)):
        if np.isfinite(tau):
            fc[(fc < 0) & (sc >= tau)] = j
    return fc


def policy_value(s: Stream, rows: np.ndarray, first: np.ndarray, H: np.ndarray, aid: np.ndarray, apos: np.ndarray) -> np.ndarray:
    idx = np.full(s.n, -1, dtype=np.int64)
    idx[rows] = np.arange(len(rows))
    keep = idx[apos] >= 0
    aid, apos = aid[keep], apos[keep]
    ids, inv = np.unique(aid, return_inverse=True)
    fc = first[idx[apos]]
    tev, usd = s.t[apos], s.usd[apos]
    alert = fc >= 0
    T = np.full(len(ids), np.iinfo(np.int64).max, dtype=np.int64)
    np.minimum.at(T, inv[alert], tev[alert] + H[fc[alert]])
    later = tev > T[inv]
    return np.bincount(inv, weights=usd * later, minlength=len(ids)) / np.maximum(np.bincount(inv, weights=usd, minlength=len(ids)), 1e-12)


def alert_policy(ctx: Context, cfg: dict) -> dict:
    """Chooses the budget allocation on validation only. Test evaluation happens in the merge step."""
    s, _ = get_stream(cfg)
    H = np.array([int(c) for c in cfg["checkpoints"]], dtype=np.int64)
    jobs = ctx.runner.jobs_dir
    s_val = [read_scores(jobs / job, "scores_val", s, s.va) for job in cfg["sweep_jobs"]]
    aid, apos = attempts_in_window(s, s.va, members_inside_only=True)
    alphas = _alpha_grid(len(H), cfg.get("grid", 0.1))
    state = ctx.load_state({"budgets": {}})
    for b in cfg["budgets"]:
        if str(b) in state["budgets"]:
            continue
        ctx.check(30)
        total = int(b * s.val_days)
        best, best_v = None, -1.0
        fixed = {}
        for a in alphas:
            taus = spend(s_val, a, total)
            v = float(policy_value(s, s.va, first_alert(s_val, taus), H, aid, apos).mean())
            if max(a) > 1 - 1e-9:
                fixed[str(int(H[int(np.argmax(a))]))] = (v, [float(x) for x in a])
            if v > best_v:
                best, best_v = a, v
        fixed_best = max(fixed.values(), key=lambda x: x[0])
        state["budgets"][str(b)] = {"alpha": [float(x) for x in best], "taus": spend(s_val, best, total), "val_value": best_v,
                                    "fixed_alpha": fixed_best[1], "fixed_taus": spend(s_val, fixed_best[1], total), "fixed_val_value": fixed_best[0]}
        ctx.save_state(state)
    return {"checkpoints": cfg["checkpoints"], "budgets": state["budgets"]}


def attacked_attempts(s: Stream) -> dict[int, np.ndarray]:
    aid, apos = attempts_in_window(s, s.te)
    out: dict[int, list] = {}
    for a, p in zip(aid.tolist(), apos.tolist()):
        out.setdefault(a, []).append(p)
    return {a: np.sort(np.array(ps, dtype=np.int64)) for a, ps in out.items() if len(ps) >= 2}


def time_grid(s: Stream) -> int:
    return 1 if s.name == "SAML-D" else 60


def retimed(s: Stream, attempts: dict, g: dict, W: int, cond: str, seed: int):
    """A re-timed stream in canonical order. Returns t, order, inverse and graph."""
    te = s.te
    t_lo, t_hi = int(s.t[te].min()), int(s.t.max())
    t2 = perturb_stream(s.t, attempts, W, cond, t_lo, t_hi, seed=seed, g=g, grid=time_grid(s))
    order = np.lexsort((np.arange(s.n), t2))
    inv = np.empty(s.n, dtype=np.int64)
    inv[order] = np.arange(s.n)
    graph = build_graph(t2[order], s.src[order], s.dst[order], s.usd[order])
    return t2, order, inv, graph


def write_attack_scores(ctx: Context, name: str, s: Stream, attempts: dict, t2: np.ndarray, pos: np.ndarray, scores: np.ndarray):
    pid = np.concatenate([np.full(len(v), k, dtype=np.int64) for k, v in attempts.items()])
    pl.DataFrame({"event_id": s.event_id[pos], "pattern_id": pid, "pos": pos, "t": t2[pos], "usd": s.usd[pos],
                  "score": np.asarray(scores, dtype=np.float32)}).write_parquet(ctx.dir / f"{name}.parquet")


def gbt_attack(ctx: Context, cfg: dict) -> dict:
    """A detector trained on clean data, with thresholds fixed on clean validation, scored on re-timed attempts.
    Only attacked rows are rescored, so false alerts elsewhere under attack are not measured."""
    import lightgbm as lgb
    s, g = get_stream(cfg)
    H = horizon_seconds(cfg["H"])
    family = cfg["family"]
    names, keep = keep_columns(family, None)
    booster = ctx.load_checkpoint(lambda p: lgb.Booster(model_file=str(p)), names=("best.bin",))
    state = ctx.load_state({"done": [], "tau": {}})
    if booster is None:
        tr = train_rows(s, cfg.get("neg_rate", 0.25))
        Xtr = design(s, g, tr, H, s.view_end, s.t, family, keep=keep)
        Xva = design(s, g, s.va, H, s.view_end, s.t, family, keep=keep)
        booster = fit_lgb(s, Xtr, s.y[tr], Xva, names, cfg["seed"])
        s_val = booster.predict(Xva, num_iteration=booster.best_iteration)
        state["tau"] = {str(b): threshold_for_budget(s_val, s.val_days, b) for b in cfg.get("budgets", [500])}
        state["val"] = validation_metrics(s, s_val)
        state["best_iter"] = int(booster.best_iteration)
        ctx.save_checkpoint(lambda p: booster.save_model(str(p), num_iteration=state["best_iter"]), name="best.bin")
        ctx.save_state(state)
        del Xtr, Xva
    attempts = attacked_attempts(s)
    pos = np.concatenate(list(attempts.values()))
    for W, cond in itertools.product(cfg["Ws"], cfg["conditions"]):
        key = f"W{W}_{cond}"
        if key in state["done"]:
            continue
        ctx.check(120)
        t2, order, inv, gr = retimed(s, attempts, g, W, cond, cfg["seed"])
        rows = inv[pos]
        X = design(s, gr, rows, H, s.view_end[order], t2[order], family, order=order, keep=keep)
        write_attack_scores(ctx, f"attack_{key}", s, attempts, t2, pos, booster.predict(X))
        state["done"].append(key)
        ctx.save_state(state)
    return {"n_attempts": len(attempts), "n_edges": int(len(pos)), "conditions": state["done"], "tau": state["tau"], "val": state.get("val")}


def service_profile(ctx: Context, cfg: dict) -> dict:
    """Feature extraction plus scoring per batch, on the named hardware, then queueing under Poisson load."""
    import lightgbm as lgb
    s, g = get_stream(cfg)
    H = horizon_seconds(cfg["H"])
    names, keep = keep_columns("wall", None)
    tr = train_rows(s, cfg.get("neg_rate", 0.05))
    booster = lgb.train(dict(LGB_PARAMS, seed=0, num_threads=os.cpu_count()),
                        lgb.Dataset(design(s, g, tr, H, s.view_end, s.t, "wall", keep=keep), s.y[tr].astype(np.float32), feature_name=names), 150)
    rng = np.random.default_rng(0)
    te = s.te
    profile = []
    for n in cfg.get("sizes", [1, 10, 100, 1000, 10000]):
        ctx.check(60)
        times = []
        for r, start in enumerate(rng.integers(0, max(len(te) - n, 1), cfg.get("repeats", 30) + 5)):
            rows = te[start:start + n]
            t0 = time.perf_counter()
            booster.predict(design(s, g, rows, H, s.view_end, s.t, "wall", keep=keep))
            if r >= 5:  # warm-up
                times.append(time.perf_counter() - t0)
        profile.append({"n": n, "median_s": float(np.median(times)), "p95_s": float(np.percentile(times, 95)), "events_per_s": float(n / np.median(times))})
    svc = service_from_profile([p["n"] for p in profile], [p["median_s"] for p in profile])
    best, table = max_stable_rate(svc, 1024, cfg.get("rates", [100.0, 300.0, 1000.0, 3000.0, 10000.0, 30000.0, 100000.0]), duration=20.0)
    return {"device": "cpu", "cores": os.cpu_count(), "H": cfg["H"], "profile": profile, "max_stable_rate": best, "table": table,
            "benchmark_mean_rate_eps": float(s.n / max((s.t.max() - s.t.min()), 1)),
            "benchmark_peak_minute_rate_eps": float(np.bincount(s.t // 60).max() / 60)}


def prepare_check(ctx: Context, cfg: dict) -> dict:
    """Pilot gate for large or unseen files: does preparation work and fit in this session's memory? Reports size only, no label or score."""
    t0 = time.time()
    ensure_prepared(cfg["dataset"], Paths.raw[cfg["dataset"]], Paths.cache)
    seconds = round(time.time() - t0, 1)
    rows = pl.scan_parquet(Paths.cache / cfg["dataset"] / "events.parquet").select(pl.len()).collect().item()
    try:
        import resource
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2)
    except ImportError:
        peak = None
    return {"dataset": cfg["dataset"], "rows": int(rows), "prepare_seconds": seconds, "peak_memory_gb_so_far": peak}


REGISTRY = {"gbt_sweep": gbt_sweep, "alert_policy": alert_policy, "gbt_attack": gbt_attack, "service_profile": service_profile, "prepare_check": prepare_check}


def load_job_metrics(job_dir: Path) -> dict:
    return read_json(Path(job_dir) / "metrics.json", {}) or {}


__all__ = ["Paths", "REGISTRY", "get_stream", "read_scores", "write_scores", "first_alert", "policy_value", "equal_alert_threshold",
           "json"]
