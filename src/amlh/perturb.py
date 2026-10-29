"""Timing-only perturbations of laundering attempts.

The adversary changes nothing but submission times: amounts, counterparties, transfer counts and
labels are untouched, and no accounts are injected. Every map below is non-decreasing in (t, pos)
within an attempt, so a transfer never moves ahead of one that preceded it, which preserves
funds-flow order. Budget: |t' - t| <= W. Times stay on the grid (one minute in AMLworld) inside [t_lo, t_hi].

  burst      t' = clip(c, t - W, t + W), c = attempt median: collapses the attempt toward one instant.
  stretch    t' = t + W(2r - 1), r in [0, 1] the rank within the attempt: pushes siblings apart.
  blend      stretches in logical time: places the attempt's transfers so equal numbers of the attempt's
             accounts' other events fall between them (DECISIONS.md D-012). Designed against last-k evidence.
             It reads the realised stream, including events a launderer could not foresee, so it is an
             upper bound on this attack's strength.
  shift_*    rigid shift by the attempt's mean |displacement| under the named attack, random sign.
             Same displacement magnitude, no change to internal gaps: the control that separates the
             effect of the attack's structure from the effect of merely moving transactions.
"""
from __future__ import annotations

import numpy as np

CONDITIONS = ("none", "burst", "stretch", "blend", "shift_burst", "shift_stretch", "shift_blend")


def _grid(x: np.ndarray, grid: int) -> np.ndarray:
    return (x // grid) * grid


def background_times(pos: np.ndarray, t: np.ndarray, g: dict) -> np.ndarray:
    """Sorted times of events touching the attempt's accounts, excluding the attempt's own transfers."""
    accounts = np.unique(np.r_[g["src"][pos], g["dst"][pos]])
    parts = []
    for a in accounts:
        parts.append(g["out_idx"][g["out_ptr"][a]:g["out_ptr"][a + 1]])
        parts.append(g["in_idx"][g["in_ptr"][a]:g["in_ptr"][a + 1]])
    events = np.unique(np.concatenate(parts)) if parts else np.empty(0, dtype=np.int64)
    events = np.setdiff1d(events, pos, assume_unique=False)
    return np.sort(t[events])


def blend_times(t_attempt: np.ndarray, bg: np.ndarray, W: int, t_lo: int, t_hi: int, grid: int = 60) -> np.ndarray:
    """Exact formula in DECISIONS.md D-012. bg holds sorted background event times for this attempt."""
    n = len(t_attempt)
    if n < 2:
        return np.clip(t_attempt.copy(), t_lo, t_hi)
    c0 = np.searchsorted(bg, t_attempt[0] - W, side="right")
    c1 = np.searchsorted(bg, t_attempt[-1] + W, side="right")
    targets = c0 + (np.arange(n) / (n - 1)) * (c1 - c0)
    out = np.empty(n, dtype=np.int64)
    for i, c in enumerate(targets):
        need = int(np.ceil(c))
        if need <= 0:
            cand = t_lo
        elif need > len(bg):
            cand = t_attempt[i] + W
        else:
            cand = bg[need - 1]  # the earliest time at which c(t) >= target
        out[i] = cand
    out = _grid(out, grid)
    out = np.minimum(np.maximum(out, t_attempt - W), t_attempt + W)
    out = np.clip(out, t_lo, t_hi)
    return np.maximum.accumulate(out)


def attack_times(t_attempt: np.ndarray, W: int, how: str, t_lo: int, t_hi: int, grid: int = 60, bg: np.ndarray | None = None) -> np.ndarray:
    """t_attempt sorted ascending in canonical order. how in {burst, stretch, blend}."""
    n = len(t_attempt)
    if how == "burst":
        c = int(_grid(np.array([np.median(t_attempt).astype(np.int64)]), grid)[0])
        out = np.minimum(np.maximum(c, t_attempt - W), t_attempt + W)
    elif how == "stretch":
        r = np.arange(n) / max(n - 1, 1)
        out = _grid((t_attempt + W * (2 * r - 1)).astype(np.int64), grid)
        out = np.minimum(np.maximum(out, t_attempt - W), t_attempt + W)
        out = np.maximum.accumulate(out)
    elif how == "blend":
        if bg is None:
            raise ValueError("blend needs the attempt's background event times")
        return blend_times(t_attempt, bg, W, t_lo, t_hi, grid)
    else:
        raise ValueError(how)
    return np.clip(out, t_lo, t_hi)


def perturb_stream(t: np.ndarray, attempts: dict[int, np.ndarray], W: int, cond: str, t_lo: int, t_hi: int,
                   seed: int = 0, g: dict | None = None, grid: int = 60) -> np.ndarray:
    """Returns a copy of t with the listed attempts re-timed. attempts maps id -> positions sorted by (t, pos).
    g (from features.build_graph on the clean stream) is required for blend and shift_blend."""
    t2 = t.copy()
    if cond == "none":
        return t2
    rng = np.random.default_rng(seed)
    for pid in sorted(attempts):
        pos = attempts[pid]
        ta = t[pos]
        sign = rng.choice([-1, 1])
        base = cond.split("_", 1)[1] if cond.startswith("shift_") else cond
        bg = background_times(pos, t, g) if base == "blend" else None
        moved = attack_times(ta, W, base, t_lo, t_hi, grid, bg)
        if cond.startswith("shift_"):
            shift = int(np.round(np.abs(moved - ta).mean() / grid)) * grid * sign
            t2[pos] = np.clip(ta + shift, t_lo, t_hi)
        else:
            t2[pos] = moved
    return t2
