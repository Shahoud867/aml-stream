"""Logical-clock evidence: statistics over an account's last k events instead of its last W hours.

This is the timing-invariant feature family proposed for RQ4 (DECISIONS.md D-008, after Lamport's logical clocks).
Nothing here reads a wall-clock gap, so an order-preserving re-timing cannot change the values for an account
with no other traffic. Wall-clock time still decides visibility: which events are in V(e, H).

Blocks, for each k in LOGICAL_KS:
  sets      per (endpoint, direction): count, log-sum USD, distinct counterparties, std of log USD
  flow      per endpoint, over its merged in/out events: was there an inflow in the window, signed log net USD
plus per endpoint: log of all prior events on the account.

At H > 0, same-minute events count as prior (their order is unknown) and a forward block repeats the set statistics
over the next k visible events. At H = 0 same-minute events stay invisible, which is why a burst attack remains a
real threat to a strict-past logical detector: see the module tests.
"""
from __future__ import annotations

import numpy as np
from numba import njit, prange

from .features import SET_NAMES, _lower, _upper

LOGICAL_KS = (8, 32, 128)
FWD_K = 32
SET_STATS = ("cnt", "amt", "ncp", "std_amt")


def logical_feature_names() -> list[str]:
    names = []
    for k in LOGICAL_KS:
        for s in SET_NAMES:
            names += [f"lc{k}_{s}_{st}" for st in SET_STATS]
    for k in LOGICAL_KS:
        for ep in ("u", "v"):
            names += [f"lc{k}_{ep}_inflow", f"lc{k}_{ep}_net"]
    names += ["lc_u_prior", "lc_v_prior"]
    for s in SET_NAMES:
        names += [f"lcfwd_{s}_{st}" for st in SET_STATS]
    return names


N_LOGICAL = len(LOGICAL_KS) * 4 * 4 + len(LOGICAL_KS) * 2 * 2 + 2 + 4 * 4


@njit(cache=True)
def _set_block(idx, a, b, other, usd, e, near_end, k, stamp, token, out, o):
    """Stats over up to k events of idx[a:b], nearest to e. Excludes e."""
    n = 0
    total = 0.0
    s1 = s2 = 0.0
    ndist = 0
    i = b - 1 if near_end else a
    while n < k and a <= i < b:
        p = idx[i]
        if p != e:
            w = other[p]
            if stamp[w] != token:
                stamp[w] = token
                ndist += 1
            total += usd[p]
            la = np.log1p(usd[p])
            s1 += la
            s2 += la * la
            n += 1
        i = i - 1 if near_end else i + 1
    out[o] = np.log1p(n)
    out[o + 1] = np.log1p(total)
    out[o + 2] = np.log1p(ndist)
    if n >= 2:
        out[o + 3] = np.sqrt(max(s2 / n - (s1 / n) ** 2, 0.0))


@njit(cache=True)
def _flow_block(in_idx, ia0, ib, out_idx, oa0, ob, t, usd, e, k, out, o):
    """Walks back through an account's merged events, newest first by (t, position)."""
    i, j = ib - 1, ob - 1
    n = 0
    net = 0.0
    inflow = 0.0
    while n < k and (i >= ia0 or j >= oa0):
        take_in = j < oa0 or (i >= ia0 and (t[in_idx[i]], in_idx[i]) >= (t[out_idx[j]], out_idx[j]))
        p = in_idx[i] if take_in else out_idx[j]
        if take_in:
            i -= 1
        else:
            j -= 1
        if p == e:
            continue
        if take_in:
            net += usd[p]
            inflow = 1.0
        else:
            net -= usd[p]
        n += 1
    out[o] = inflow
    out[o + 1] = np.sign(net) * np.log1p(abs(net))


@njit(cache=True, parallel=True)
def _compute_logical(rows, H, t, src, dst, usd, view_end, n_nodes, out_ptr, out_idx, in_ptr, in_idx, ks, fwd_k, n_features, block_size):
    F = np.zeros((len(rows), n_features), dtype=np.float32)
    h_pos = H > 0
    n_blocks = (len(rows) + block_size - 1) // block_size
    for bk in prange(n_blocks):
        stamp = np.full(n_nodes, -1, dtype=np.int64)
        token = 0
        buf = np.zeros(n_features, dtype=np.float64)
        for r in range(bk * block_size, min((bk + 1) * block_size, len(rows))):
            e = rows[r]
            u, v, t_e = src[e], dst[e], t[e]
            buf[:] = 0.0
            t_end = min(t_e + H, view_end[e]) if h_pos else t_e - 1
            nk = len(ks)
            spans = np.empty((4, 4), dtype=np.int64)  # a0, split, upper, is_out
            for li in range(4):
                ep = u if li < 2 else v
                is_out = li % 2 == 0
                idx = out_idx if is_out else in_idx
                a0, b0 = (out_ptr[ep], out_ptr[ep + 1]) if is_out else (in_ptr[ep], in_ptr[ep + 1])
                # Prior events: strictly earlier at H = 0; same-minute events also count at H > 0.
                split = _upper(idx, a0, b0, t, t_e) if h_pos else _lower(idx, a0, b0, t, t_e)
                upper = _upper(idx, split, b0, t, t_end) if h_pos else split
                spans[li, 0], spans[li, 1], spans[li, 2], spans[li, 3] = a0, split, upper, int(is_out)
            o = 0
            for ki in range(nk):
                for li in range(4):
                    is_out = spans[li, 3] == 1
                    idx = out_idx if is_out else in_idx
                    other = dst if is_out else src
                    token += 1
                    _set_block(idx, spans[li, 0], spans[li, 1], other, usd, e, True, ks[ki], stamp, token, buf, o)
                    o += 4
            for ki in range(nk):
                _flow_block(in_idx, spans[1, 0], spans[1, 1], out_idx, spans[0, 0], spans[0, 1], t, usd, e, ks[ki], buf, o)
                _flow_block(in_idx, spans[3, 0], spans[3, 1], out_idx, spans[2, 0], spans[2, 1], t, usd, e, ks[ki], buf, o + 2)
                o += 4
            # All prior events on the account, excluding e itself when same-minute events are visible.
            prior_u = (spans[0, 1] - spans[0, 0]) + (spans[1, 1] - spans[1, 0])
            prior_v = (spans[2, 1] - spans[2, 0]) + (spans[3, 1] - spans[3, 0])
            if h_pos:
                prior_u -= 1 + (1 if u == v else 0)
                prior_v -= 1 + (1 if u == v else 0)
            buf[o] = np.log1p(max(prior_u, 0))
            buf[o + 1] = np.log1p(max(prior_v, 0))
            o += 2
            for li in range(4):
                if h_pos:
                    is_out = spans[li, 3] == 1
                    idx = out_idx if is_out else in_idx
                    other = dst if is_out else src
                    token += 1
                    _set_block(idx, spans[li, 1], spans[li, 2], other, usd, e, False, fwd_k, stamp, token, buf, o)
                o += 4
            for k2 in range(n_features):
                F[r, k2] = buf[k2]
    return F


def compute_logical(g: dict, rows: np.ndarray, H: int, view_end: np.ndarray, block_size: int = 4096,
                    ks=LOGICAL_KS, fwd_k: int = FWD_K) -> np.ndarray:
    return _compute_logical(np.asarray(rows, dtype=np.int64), int(H), g["t"], g["src"], g["dst"], g["usd"],
                            np.asarray(view_end, dtype=np.int64), g["n_nodes"], g["out_ptr"], g["out_idx"],
                            g["in_ptr"], g["in_idx"], np.asarray(ks, dtype=np.int64), int(fwd_k), N_LOGICAL, block_size)
