"""Horizon-respecting two-hop neighbour sampler for transaction edges.

For a seed e = (u, v, t) and horizon H the sampler only returns events in V(e, H) (see features.py).
Selection is deterministic: nearest in time around t, up to S_past earlier events and, when H > 0,
up to S_fwd later-or-tied ones. Second-hop lists are centred on the seed's time and obey the seed's
view, so a neighbour can never contribute evidence the seed could not have seen at its decision time.

Output layout, for a batch of B seeds:
  hop1[B, 2 endpoints, 2 directions, S1]             event positions, -1 = padding
  hop1_other[B, 2, 2, S1]                            neighbour account of each first-hop event
  hop2[B, 2, 2, S1, 2 directions, S2]                event positions, -1 = padding
with S1 = S_past + S_fwd and S2 = S2_past + S2_fwd. Direction 0 = in, 1 = out.
"""
from __future__ import annotations

import numpy as np
from numba import njit, prange


@njit(cache=True)
def _lower(idx, a, b, t, x):
    while a < b:
        m = (a + b) // 2
        if t[idx[m]] < x:
            a = m + 1
        else:
            b = m
    return a


@njit(cache=True)
def _upper(idx, a, b, t, x):
    while a < b:
        m = (a + b) // 2
        if t[idx[m]] <= x:
            a = m + 1
        else:
            b = m
    return a


@njit(cache=True)
def _take(idx, a0, b0, t, e, t_e, t_end, h_pos, sp, sf, out, o, s_exclude):
    """Writes up to sp past and sf forward event positions to out[o:o+sp+sf].
    Events equal to e or s_exclude are skipped."""
    split = _lower(idx, a0, b0, t, t_e)
    w = o
    n = 0
    i = split - 1
    while i >= a0 and n < sp:
        p = idx[i]
        if p != e and p != s_exclude:
            out[w] = p
            w += 1
            n += 1
        i -= 1
    if h_pos:
        c = _upper(idx, split, b0, t, t_end)
        w = o + sp
        n = 0
        i = split
        while i < c and n < sf:
            p = idx[i]
            if p != e and p != s_exclude:
                out[w] = p
                w += 1
                n += 1
            i += 1


@njit(cache=True, parallel=True)
def sample_batch(seeds, H, t, src, dst, view_end, out_ptr, out_idx, in_ptr, in_idx, sp, sf, s2p, s2f):
    B = len(seeds)
    S1 = sp + sf
    S2 = s2p + s2f
    h_pos = H > 0
    hop1 = np.full((B, 2, 2, S1), -1, dtype=np.int64)
    other = np.full((B, 2, 2, S1), -1, dtype=np.int64)
    hop2 = np.full((B, 2, 2, S1, 2, S2), -1, dtype=np.int64)
    for b in prange(B):
        e = seeds[b]
        t_e = t[e]
        t_end = min(t_e + H, view_end[e]) if h_pos else t_e - 1
        for ep in range(2):
            a = src[e] if ep == 0 else dst[e]
            for d in range(2):
                if d == 0:
                    idx, a0, b0, oth = in_idx, in_ptr[a], in_ptr[a + 1], src
                else:
                    idx, a0, b0, oth = out_idx, out_ptr[a], out_ptr[a + 1], dst
                buf = hop1[b, ep, d]
                _take(idx, a0, b0, t, e, t_e, t_end, h_pos, sp, sf, buf, 0, -1)
                for j in range(S1):
                    p = buf[j]
                    if p < 0:
                        continue
                    x = oth[p]
                    other[b, ep, d, j] = x
                    _take(in_idx, in_ptr[x], in_ptr[x + 1], t, e, t_e, t_end, h_pos, s2p, s2f,
                          hop2[b, ep, d, j, 0], 0, p)
                    _take(out_idx, out_ptr[x], out_ptr[x + 1], t, e, t_e, t_end, h_pos, s2p, s2f,
                          hop2[b, ep, d, j, 1], 0, p)
    return hop1, other, hop2
