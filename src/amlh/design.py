"""Design matrices for the boosted detector: one frozen definition for clean runs and attacked streams."""
from __future__ import annotations

import re

import numpy as np

from .features import INF_H, compute_features, feature_names
from .logical import compute_logical, logical_feature_names

BASE_NAMES = ["log_usd", "log_paid", "log_recv", "cur_paid", "cur_recv", "fmt", "same_cur", "is_self",
              "usd_vs_uin24", "usd_vs_uout24", "usd_vs_vin24", "usd_vs_vout24",
              "uout_vs_uin24", "vout_vs_vin24", "uoutcp_vs_uincp24", "voutcp_vs_vincp24", "hour"]
CATEGORICAL = ("cur_paid", "cur_recv", "fmt")
# Logical-clock base: the event's own attributes. No wall-clock window ratios, and no hour of day.
LOGICAL_BASE = ["log_usd", "log_paid", "log_recv", "cur_paid", "cur_recv", "fmt", "same_cur", "is_self"]
# Pair counts and motifs over all visible history are order-based, so they belong to the logical family.
ALL_HISTORY = (10**15,)
_FN = feature_names()
_RATIO_KEYS = ("u_in_amt", "u_out_amt", "v_in_amt", "v_out_amt", "u_in_cnt", "u_out_cnt", "v_in_cnt", "v_out_cnt",
               "u_in_ncp", "u_out_ncp", "v_in_ncp", "v_out_ncp")
_IX = {k: _FN.index(f"past24h_{k}") for k in _RATIO_KEYS}
_PAIR_MOTIF = [i for i, n in enumerate(_FN) if "_pair_" in n or n.startswith("m_")]


def horizon_seconds(H) -> int:
    return INF_H if H == "inf" else int(H)


def names_for(family: str) -> list[str]:
    if family == "wall":
        return BASE_NAMES + _FN
    logical = LOGICAL_BASE + logical_feature_names() + [f"all_{_FN[i]}" for i in _PAIR_MOTIF]
    if family == "logical":
        return logical
    if family == "combined":
        return BASE_NAMES + _FN + logical_feature_names() + [f"all_{_FN[i]}" for i in _PAIR_MOTIF]
    raise ValueError(family)


def _attrs(s, order, rows, t):
    orig = order[rows] if order is not None else rows
    lu = np.log1p(s.usd[orig]).astype(np.float32)
    lp = np.log1p(s.paid[orig]).astype(np.float32)
    lr = np.log1p(s.recv[orig]).astype(np.float32)
    same = (s.cur_p[orig] == s.cur_r[orig]).astype(np.float32)
    selfl = (s.src[orig] == s.dst[orig]).astype(np.float32)
    return orig, lu, lp, lr, same, selfl


def wall_block(s, g, rows, H, view_end, t, order=None) -> np.ndarray:
    orig, lu, lp, lr, same, selfl = _attrs(s, order, rows, t)
    F = compute_features(g, rows, H, view_end)
    ix = _IX
    B = np.column_stack([lu, lp, lr, s.cur_p[orig], s.cur_r[orig], s.fmt[orig], same, selfl,
                         lu - F[:, ix["u_in_amt"]], lu - F[:, ix["u_out_amt"]], lu - F[:, ix["v_in_amt"]], lu - F[:, ix["v_out_amt"]],
                         F[:, ix["u_out_cnt"]] - F[:, ix["u_in_cnt"]], F[:, ix["v_out_cnt"]] - F[:, ix["v_in_cnt"]],
                         F[:, ix["u_out_ncp"]] - F[:, ix["u_in_ncp"]], F[:, ix["v_out_ncp"]] - F[:, ix["v_in_ncp"]],
                         ((t[rows] % 86400) // 3600).astype(np.float32)])
    return np.hstack([B, F])


def logical_extra(g, rows, H, view_end) -> np.ndarray:
    L = compute_logical(g, rows, H, view_end)
    F = compute_features(g, rows, H, view_end, past_windows=ALL_HISTORY)
    return np.hstack([L, F[:, _PAIR_MOTIF_ALL]])


# With one all-history window the feature layout is shorter, so its pair and motif columns sit at the end.
_N_ALL = 1 * 4 * 6 + 4 * 6 + 8 + 5
_PAIR_MOTIF_ALL = list(range(_N_ALL - 13, _N_ALL))


def design(s, g, rows, H, view_end, t, family: str = "wall", order=None, keep: list[int] | None = None) -> np.ndarray:
    """Features for rows of a stream. order maps stream position to original event on a re-timed stream."""
    if family == "wall":
        X = wall_block(s, g, rows, H, view_end, t, order)
    else:
        extra = logical_extra(g, rows, H, view_end)
        if family == "logical":
            orig, lu, lp, lr, same, selfl = _attrs(s, order, rows, t)
            base = np.column_stack([lu, lp, lr, s.cur_p[orig], s.cur_r[orig], s.fmt[orig], same, selfl])
            X = np.hstack([base, extra])
        elif family == "combined":
            X = np.hstack([wall_block(s, g, rows, H, view_end, t, order), extra])
        else:
            raise ValueError(family)
    return X if keep is None else X[:, keep]


def keep_columns(family: str, drop: str | None) -> tuple[list[str], list[int] | None]:
    names = names_for(family)
    if not drop:
        return names, None
    keep = [i for i, n in enumerate(names) if not re.search(drop, n)]
    return [names[i] for i in keep], keep
