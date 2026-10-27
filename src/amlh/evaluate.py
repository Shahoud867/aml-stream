"""Turns saved scores into results. Jobs on sealed datasets never call the test-side functions here."""
from __future__ import annotations

import numpy as np

from .datasets import Stream, attempts_in_window
from .metrics import attempt_outcomes, best_f1_threshold, bootstrap_mean_ci, event_metrics, summarize_attempts, threshold_for_budget
from .sealing import guard_test_metrics

BUDGETS = (100, 500, 2000)


def is_whole_window(H) -> bool:
    return H in ("inf", None)


def _H_args(H, window_end):
    return (0, window_end) if is_whole_window(H) else (int(H), None)


def equal_alert_threshold(scores: np.ndarray, budget: float, days: int) -> float:
    """Top B x D scores on one window: gives every horizon the same alert count (SPEC.md section 7)."""
    k = min(max(int(round(budget * days)), 1), len(scores))
    return float(np.partition(scores, len(scores) - k)[len(scores) - k])


def validation_metrics(s: Stream, s_val: np.ndarray) -> dict:
    y = s.y[s.va]
    tau, f1 = best_f1_threshold(y, s_val)
    m = event_metrics(y, s_val, tau)
    return {"tau": tau, "val_f1": f1, "val_pr_auc": m["pr_auc"], "val_n": m["n"], "val_pos": m["n_pos"]}


def _outcomes(s: Stream, rows: np.ndarray, scores: np.ndarray, H, tau: float, inside_only: bool, window_end: int):
    aid, apos = attempts_in_window(s, rows, members_inside_only=inside_only)
    if len(aid) == 0:
        return None
    idx = np.full(s.n, -1, dtype=np.int64)
    idx[rows] = np.arange(len(rows))
    ok = idx[apos] >= 0
    H_eff, end = _H_args(H, window_end)
    return attempt_outcomes(aid[ok], s.t[apos][ok], s.usd[apos][ok], scores[idx[apos][ok]], tau, H_eff, end)


def heldout_outcomes(s: Stream, s_te: np.ndarray, H, tau: float, unsealed: bool = False):
    guard_test_metrics(s.name, unsealed)
    return _outcomes(s, s.te, s_te, H, tau, False, int(s.t.max()))


def validation_outcomes(s: Stream, s_val: np.ndarray, H, tau: float):
    """Uses validation only, so it is safe on sealed datasets and may drive selection."""
    return _outcomes(s, s.va, s_val, H, tau, True, int(s.t[s.va].max()))


def heldout_metrics(s: Stream, s_val: np.ndarray, s_te: np.ndarray, H, unsealed: bool = False) -> dict:
    """The full results row for one detector at one horizon."""
    guard_test_metrics(s.name, unsealed)
    tau = best_f1_threshold(s.y[s.va], s_val)[0]
    y = s.y[s.te]
    core = s.core_mask
    row = {"tau": tau, "test": event_metrics(y, s_te, tau), "test_core": event_metrics(y[core], s_te[core], tau)}
    if (~core).any():
        row["test_tail"] = event_metrics(y[~core], s_te[~core], tau)
    attempts = {}
    outcome = heldout_outcomes(s, s_te, H, tau, unsealed)
    if outcome is not None:
        attempts["f1_tau"] = summarize_attempts(outcome)
    for b in BUDGETS:
        tb = threshold_for_budget(s_val, s.val_days, b)
        o = heldout_outcomes(s, s_te, H, tb, unsealed)
        if o is None:
            continue
        summ = summarize_attempts(o)
        summ["value_share_ci"] = bootstrap_mean_ci(o["value_share"])
        summ["alerts_per_day_core"] = float((s_te[core] >= tb).sum() / max(s.core_days, 1))
        attempts[f"budget_{b}"] = summ
        te = equal_alert_threshold(s_te, b, s.core_days)
        eq = heldout_outcomes(s, s_te, H, te, unsealed)
        attempts[f"equal_{b}"] = summarize_attempts(eq)
    row["attempts"] = attempts
    return row


def paired_diff_ci(a: np.ndarray, b: np.ndarray, n: int = 4000, seed: int = 0) -> tuple[float, float, float]:
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, len(d), (n, len(d)))].mean(axis=1)
    return float(d.mean()), float(np.quantile(means, .025)), float(np.quantile(means, .975))


def c3b(s: Stream, scores: dict, seeds: list[int], budget: int = 500, candidates=("1", "60", "600", "3600"),
        unsealed: bool = False) -> dict | None:
    """Hypothesis C3b, as frozen in DECISIONS.md D-012.

    scores[(H, seed)] = (s_val, s_te). H* is chosen on validation under the equal-alert rule; the test
    contrast against strict past then uses every seed, paired on attempts. Returns None if data is missing."""
    guard_test_metrics(s.name, unsealed)
    have = [h for h in candidates if all((h, k) in scores for k in seeds)]
    if not have or not all(("0", k) in scores for k in seeds):
        return None

    def val_sv(h):
        vals = []
        for k in seeds:
            sv = scores[(h, k)][0]
            o = validation_outcomes(s, sv, h, equal_alert_threshold(sv, budget, s.val_days))
            vals.append(o["value_share"].mean() if o is not None else 0.0)
        return float(np.mean(vals))

    val_table = {h: val_sv(h) for h in have}
    star = max(have, key=lambda h: val_table[h])

    def test_sv(h):
        per_seed = []
        for k in seeds:
            st = scores[(h, k)][1]
            o = heldout_outcomes(s, st, h, equal_alert_threshold(st, budget, s.core_days), unsealed)
            per_seed.append(o["value_share"])
        return np.mean(per_seed, axis=0)

    wait, strict = test_sv(star), test_sv("0")
    m, lo, hi = paired_diff_ci(wait, strict)
    return {"H_star": star, "val_sv_by_H": val_table, "sv_wait": float(wait.mean()), "sv_strict": float(strict.mean()),
            "diff": m, "ci": [lo, hi], "confirmed": bool(lo > 0), "budget": budget, "attempts": int(len(wait))}


def attack_outcomes(t_attacked: np.ndarray, usd: np.ndarray, attempt_ids: np.ndarray, scores: np.ndarray, tau: float, H, window_end: int) -> dict:
    """Per-attempt detection, recall and stoppable value on a re-timed stream."""
    H_eff, end = _H_args(H, window_end)
    o = attempt_outcomes(attempt_ids, t_attacked, usd, scores, tau, H_eff, end)
    ids, inv = np.unique(attempt_ids, return_inverse=True)
    flagged = np.bincount(inv, weights=(scores >= tau).astype(float), minlength=len(ids))
    counts = np.bincount(inv, minlength=len(ids))
    t_min = np.full(len(ids), np.iinfo(np.int64).max, dtype=np.int64)
    t_max = np.full(len(ids), np.iinfo(np.int64).min, dtype=np.int64)
    np.minimum.at(t_min, inv, t_attacked)
    np.maximum.at(t_max, inv, t_attacked)
    return {"ids": ids, "detected": o["detected"], "value_share": o["value_share"], "recall": flagged / np.maximum(counts, 1),
            "completion_h": (t_max - t_min) / 3600.0}
