"""Event-level and attempt-level evaluation for decision-horizon experiments."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score


def best_f1_threshold(y: np.ndarray, s: np.ndarray) -> tuple[float, float]:
    """Highest threshold attaining maximal minority F1 (flag if s >= tau)."""
    order = np.argsort(-s, kind="mergesort")
    ss, yy = s[order], y[order]
    tp = np.cumsum(yy)
    k = np.arange(1, len(yy) + 1)
    pos = yy.sum()
    # Only positions where the score actually changes are valid cut points.
    valid = np.r_[ss[1:] != ss[:-1], True]
    f1 = np.where(valid, 2 * tp / (k + pos), -1.0)
    i = int(np.argmax(f1))
    return float(ss[i]), float(f1[i])


def f1_at(y: np.ndarray, s: np.ndarray, tau: float) -> dict:
    flag = s >= tau
    tp = int((flag & (y == 1)).sum())
    fp = int((flag & (y == 0)).sum())
    fn = int(((~flag) & (y == 1)).sum())
    return {"f1": 2 * tp / max(2 * tp + fp + fn, 1), "precision": tp / max(tp + fp, 1),
            "recall": tp / max(tp + fn, 1), "tp": tp, "fp": fp, "fn": fn}


def event_metrics(y, s, tau) -> dict:
    out = f1_at(y, s, tau)
    out["pr_auc"] = float(average_precision_score(y, s)) if y.sum() > 0 else float("nan")
    out["n"] = int(len(y))
    out["n_pos"] = int(y.sum())
    return out


def threshold_for_budget(s_val: np.ndarray, val_days: float, alerts_per_day: float) -> float:
    """Score threshold giving the requested alert rate on validation (labels not needed)."""
    k = int(round(alerts_per_day * val_days))
    k = min(max(k, 1), len(s_val))
    return float(np.partition(s_val, len(s_val) - k)[len(s_val) - k])


def attempt_outcomes(att_ids, ev_t, ev_usd, ev_score, tau, H, t_batch_end=None):
    """Per-attempt detection and interdiction.

    Inputs are aligned arrays with one row per (attempt, member event) scored in the test window.
    A flagged event e raises its alert at d_e = t_e + H; with H = infinity (t_batch_end given) every
    alert is raised at t_batch_end, the moment the whole batch is available.

    Interdictable share of attempt A with first alert time T:
        value:     sum(usd_e : t_e > T) / sum(usd_e)
        transfers: |{e : t_e > T}| / |A|
    Undetected attempts have interdictable share 0.
    Returns dict of arrays ordered by unique attempt id.
    """
    ids, inv = np.unique(att_ids, return_inverse=True)
    n = len(ids)
    T = np.full(n, np.iinfo(np.int64).max, dtype=np.int64)
    alert = ev_score >= tau
    d = ev_t + H if t_batch_end is None else np.full_like(ev_t, t_batch_end)
    np.minimum.at(T, inv[alert], d[alert])
    detected = T < np.iinfo(np.int64).max
    t_first = np.full(n, np.iinfo(np.int64).max, dtype=np.int64)
    np.minimum.at(t_first, inv, ev_t)
    later = ev_t > T[inv]
    usd_total = np.bincount(inv, weights=ev_usd, minlength=n)
    usd_later = np.bincount(inv, weights=ev_usd * later, minlength=n)
    cnt_total = np.bincount(inv, minlength=n)
    cnt_later = np.bincount(inv, weights=later.astype(float), minlength=n)
    delay = np.where(detected, (T - t_first).astype(float), np.nan)
    return {"ids": ids, "detected": detected, "delay_s": delay,
            "value_share": usd_later / np.maximum(usd_total, 1e-12),
            "transfer_share": cnt_later / np.maximum(cnt_total, 1),
            "usd_total": usd_total, "usd_later": usd_later}


def summarize_attempts(o: dict, weights=None) -> dict:
    det = o["detected"]
    return {"attempts": int(len(det)), "detected_frac": float(det.mean()),
            "median_delay_h": float(np.nanmedian(o["delay_s"]) / 3600) if det.any() else float("nan"),
            "value_share_mean": float(o["value_share"].mean()),
            "transfer_share_mean": float(o["transfer_share"].mean())}


def bootstrap_mean_ci(x: np.ndarray, n_boot=2000, seed=0, alpha=0.05) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), (n_boot, len(x)))
    m = x[idx].mean(axis=1)
    return float(np.quantile(m, alpha / 2)), float(np.quantile(m, 1 - alpha / 2))
