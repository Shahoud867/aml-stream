"""Decision-horizon feature engine.

For an event e = (u, v, t) and a decision horizon H the detector's view is

    V(e, H) = { e' != e : t' < t }                         if H == 0  (strict causal)
    V(e, H) = { e' != e : t' <= min(t + H, end(e)) }       if H > 0

where end(e) clips the view at the end of e's split (train and validation graphs
never see later splits, exactly as in the Multi-GNN protocol; H = infinity reproduces
that protocol's "whole split graph" view).

Timestamps have 1-minute resolution, so same-minute events have unknown order.
H == 0 treats them as invisible; any H >= 60 s treats them as visible.

Features fall in three blocks, all computed only from V(e, H):
  past   per (endpoint, direction, W): count, log-sum USD, distinct counterparties, time gap,
         std of log-USD and std of event times (hours)
  fwd    the same statistics over the forward part of the view (all zero when H == 0)
  motif  bounded counts of 2-path, 3-cycle and convergent-sibling structures
Scans over high-degree accounts are capped; the caps are part of the feature definition.
"""
from __future__ import annotations

import numpy as np
from numba import njit, prange

PAST_WINDOWS = (6 * 3600, 24 * 3600, 72 * 3600)
SET_NAMES = ("u_out", "u_in", "v_out", "v_in")
STATS = ("cnt", "amt", "ncp", "gap", "std_amt", "std_t")
CAP_DISTINCT = 256
CAP_PATH = 128
CAP_SIB = 32
CAP_D = 16
CAP_SIB_NBR = 8
INF_H = np.iinfo(np.int64).max // 4


def feature_names() -> list[str]:
    names = []
    for w in PAST_WINDOWS:
        for s in SET_NAMES:
            names += [f"past{w // 3600}h_{s}_{st}" for st in STATS]
    for s in SET_NAMES:
        names += [f"fwd_{s}_{st}" for st in STATS]
    for side in ("past", "fwd"):
        names += [f"{side}_pair_uv_cnt", f"{side}_pair_uv_amt", f"{side}_pair_vu_cnt", f"{side}_pair_vu_amt"]
    names += ["m_path2", "m_cyc3_close", "m_cyc3_open", "m_conv_out", "m_conv_in"]
    return names


N_FEATURES = len(PAST_WINDOWS) * 4 * 6 + 4 * 6 + 8 + 5


@njit(cache=True)
def _csr(keys, n_keys):
    """Stable CSR: positions grouped by key, ascending position inside each group."""
    ptr = np.zeros(n_keys + 1, dtype=np.int64)
    for k in keys:
        ptr[k + 1] += 1
    ptr = np.cumsum(ptr)
    fill = ptr[:-1].copy()
    idx = np.empty(len(keys), dtype=np.int64)
    for p in range(len(keys)):
        k = keys[p]
        idx[fill[k]] = p
        fill[k] += 1
    return ptr, idx


def build_graph(t, src, dst, usd):
    """Adjacency lists plus prefix sums. Input arrays must be in canonical order."""
    n_nodes = int(max(src.max(), dst.max())) + 1
    usd_v = np.asarray(usd, dtype=np.float64)
    out_ptr, out_idx = _csr(src, n_nodes)
    in_ptr, in_idx = _csr(dst, n_nodes)
    out_cum = np.concatenate([[0.0], np.cumsum(usd_v[out_idx])])
    in_cum = np.concatenate([[0.0], np.cumsum(usd_v[in_idx])])
    # Pair lists: key (src, dst) mapped to a dense id; sorted key array supports lookups.
    pair_key = src.astype(np.int64) * n_nodes + dst.astype(np.int64)
    uniq, pair_id = np.unique(pair_key, return_inverse=True)
    pair_ptr, pair_idx = _csr(pair_id.astype(np.int64), len(uniq))
    pair_cum = np.concatenate([[0.0], np.cumsum(usd_v[pair_idx])])
    return dict(t=np.asarray(t, dtype=np.int64), src=np.asarray(src, dtype=np.int64),
                dst=np.asarray(dst, dtype=np.int64), usd=usd_v,
                n_nodes=n_nodes, out_ptr=out_ptr, out_idx=out_idx, out_cum=out_cum,
                in_ptr=in_ptr, in_idx=in_idx, in_cum=in_cum,
                pair_keys=uniq, pair_ptr=pair_ptr, pair_idx=pair_idx, pair_cum=pair_cum)


@njit(cache=True)
def _lower(idx, a, b, t, x):
    """First index in idx[a:b] whose event time is >= x."""
    while a < b:
        m = (a + b) // 2
        if t[idx[m]] < x:
            a = m + 1
        else:
            b = m
    return a


@njit(cache=True)
def _upper(idx, a, b, t, x):
    """First index in idx[a:b] whose event time is > x."""
    while a < b:
        m = (a + b) // 2
        if t[idx[m]] <= x:
            a = m + 1
        else:
            b = m
    return a


@njit(cache=True)
def _set_stats(idx, cum, a, b, t, other, usd, e, t_e, near_end, stamp, token, K, out, o):
    """Stats over idx[a:b] excluding event e. near_end=True scans from b-1 downward (past);
    False scans from a upward (forward). Writes 6 values at out[o:o+6]. Count and sum are exact;
    distinct counterparties and the two standard deviations use the K scanned events."""
    n = b - a
    total = cum[b] - cum[a] if n > 0 else 0.0
    cnt = n
    ndist = 0
    gap = -1.0
    scanned = 0
    m = 0
    s1a = s2a = s1t = s2t = 0.0
    i = b - 1 if near_end else a
    while scanned < K and a <= i < b:
        p = idx[i]
        if p == e:
            cnt -= 1
            total -= _usd_of(cum, i)
        else:
            if gap < 0.0:
                gap = float(abs(t_e - t[p]))
            w = other[p]
            if stamp[w] != token:
                stamp[w] = token
                ndist += 1
            la = np.log1p(usd[p])
            dt = (t_e - t[p]) / 3600.0
            s1a += la
            s2a += la * la
            s1t += dt
            s2t += dt * dt
            m += 1
        scanned += 1
        i = i - 1 if near_end else i + 1
    if scanned < n:  # e may lie beyond the cap; check the rest only for its removal
        j = i
        while a <= j < b and t[idx[j]] == t_e:
            if idx[j] == e:
                cnt -= 1
                total -= _usd_of(cum, j)
                break
            j = j - 1 if near_end else j + 1
    out[o] = np.log1p(max(cnt, 0))
    out[o + 1] = np.log1p(max(total, 0.0))
    out[o + 2] = np.log1p(ndist)
    out[o + 3] = np.log1p(gap) if gap >= 0.0 else -1.0
    if m >= 2:
        out[o + 4] = np.sqrt(max(s2a / m - (s1a / m) ** 2, 0.0))
        out[o + 5] = np.sqrt(max(s2t / m - (s1t / m) ** 2, 0.0))


@njit(cache=True)
def _usd_of(cum, i):
    return cum[i + 1] - cum[i]


@njit(cache=True)
def _pair_lookup(keys, key):
    i = np.searchsorted(keys, key)
    if i < len(keys) and keys[i] == key:
        return i
    return -1


@njit(cache=True)
def _pair_stats(t, pair_ptr, pair_idx, pair_cum, pid, e, t_e, low, t_end, H_pos, out, o):
    """Writes (past cnt, past amt) at out[o:o+2] and (fwd cnt, fwd amt) at out[o+8+...]."""
    if pid < 0:
        return
    a0, b0 = pair_ptr[pid], pair_ptr[pid + 1]
    a = _lower(pair_idx, a0, b0, t, low)
    b = _lower(pair_idx, a0, b0, t, t_e)
    out[o] = np.log1p(b - a)
    out[o + 1] = np.log1p(pair_cum[b] - pair_cum[a])
    if H_pos:
        c = _upper(pair_idx, a0, b0, t, t_end)
        cnt = c - b
        amt = pair_cum[c] - pair_cum[b]
        for k in range(b, c):
            if pair_idx[k] == e:
                cnt -= 1
                amt -= pair_cum[k + 1] - pair_cum[k]
                break
            if t[pair_idx[k]] > t_e:
                break
        out[o + 4] = np.log1p(max(cnt, 0))
        out[o + 5] = np.log1p(max(amt, 0.0))


@njit(cache=True)
def _near_range(a, b, split_at, cap):
    """Indices of up to `cap` list entries nearest to position split_at in [a, b)."""
    lo, hi = split_at, split_at
    n = 0
    while n < cap and (lo > a or hi < b):
        if lo > a:
            lo -= 1
            n += 1
        if n < cap and hi < b:
            hi += 1
            n += 1
    return lo, hi


@njit(cache=True, parallel=True)
def _compute(rows, H, t, src, dst, usd, view_end, n_nodes,
             out_ptr, out_idx, out_cum, in_ptr, in_idx, in_cum,
             pair_keys, pair_ptr, pair_idx, pair_cum, past_windows, n_features, block_size,
             cap_distinct, cap_path, cap_sib, cap_d, cap_sib_nbr):
    F = np.zeros((len(rows), n_features), dtype=np.float32)
    H_pos = H > 0
    n_blocks = (len(rows) + block_size - 1) // block_size
    for bk in prange(n_blocks):
        stamp = np.full(n_nodes, -1, dtype=np.int64)     # distinct counting
        stamp2 = np.full(n_nodes, -1, dtype=np.int64)    # motif membership
        first_t = np.zeros(n_nodes, dtype=np.int64)      # earliest time per stamped node
        token = 0
        buf = np.empty(n_features, dtype=np.float64)
        for r in range(bk * block_size, min((bk + 1) * block_size, len(rows))):
            e = rows[r]
            u, v, t_e = src[e], dst[e], t[e]
            t_end = min(t_e + H, view_end[e]) if H_pos else t_e - 1
            buf[:] = 0.0
            o = 0
            lists = ((out_idx, out_cum, out_ptr[u], out_ptr[u + 1], dst),
                     (in_idx, in_cum, in_ptr[u], in_ptr[u + 1], src),
                     (out_idx, out_cum, out_ptr[v], out_ptr[v + 1], dst),
                     (in_idx, in_cum, in_ptr[v], in_ptr[v + 1], src))
            n_w = len(past_windows)
            fwd_base = n_w * 24
            for li in range(4):
                idx, cum, a0, b0, other = lists[li]
                split_at = _lower(idx, a0, b0, t, t_e)
                for wi in range(n_w):
                    a = _lower(idx, a0, split_at, t, t_e - past_windows[wi])
                    token += 1
                    _set_stats(idx, cum, a, split_at, t, other, usd, e, t_e, True, stamp, token, cap_distinct, buf,
                               wi * 24 + li * 6)
                if H_pos:
                    c = _upper(idx, split_at, b0, t, t_end)
                    token += 1
                    _set_stats(idx, cum, split_at, c, t, other, usd, e, t_e, False, stamp, token, cap_distinct, buf,
                               fwd_base + li * 6)
                else:
                    buf[fwd_base + li * 6 + 3] = -1.0
            o = fwd_base + 24
            low = t_e - past_windows[n_w - 1]
            _pair_stats(t, pair_ptr, pair_idx, pair_cum, _pair_lookup(pair_keys, u * n_nodes + v),
                        e, t_e, low, t_end, H_pos, buf, o)
            _pair_stats(t, pair_ptr, pair_idx, pair_cum, _pair_lookup(pair_keys, v * n_nodes + u),
                        e, t_e, low, t_end, H_pos, buf, o + 2)
            o += 8
            # View bounds per list for motif scans: [t_e - MOTIF, t_end].
            ua0, ub0 = out_ptr[u], out_ptr[u + 1]
            va0, vb0 = in_ptr[v], in_ptr[v + 1]
            # m_path2: distinct w != u, v with u->w then w->v, ordered in time, both in view.
            s_u = _lower(out_idx, ua0, ub0, t, t_e)
            ua = _lower(out_idx, ua0, s_u, t, low)
            ub = _upper(out_idx, s_u, ub0, t, t_end) if H_pos else s_u
            lo, hi = _near_range(ua, ub, s_u, cap_path)
            token += 1
            tk = token
            for i in range(lo, hi):
                p = out_idx[i]
                w = dst[p]
                if p != e and w != u and w != v and stamp2[w] != tk:
                    stamp2[w] = tk
                    first_t[w] = t[p]
            s_v = _lower(in_idx, va0, vb0, t, t_e)
            va = _lower(in_idx, va0, s_v, t, low)
            vb = _upper(in_idx, s_v, vb0, t, t_end) if H_pos else s_v
            lo, hi = _near_range(va, vb, s_v, cap_path)
            n2 = 0
            token += 1
            for i in range(lo, hi):
                p = in_idx[i]
                w = src[p]
                if p != e and w != u and w != v and stamp2[w] == tk and first_t[w] <= t[p] and stamp[w] != token:
                    stamp[w] = token
                    n2 += 1
            buf[o] = np.log1p(n2)
            # m_cyc3_close: v->w->u strictly before e (e closes a 3-cycle).
            # m_cyc3_open:  v->w->u at or after e (e opens one; needs look-ahead).
            voa0, vob0 = out_ptr[v], out_ptr[v + 1]
            uia0, uib0 = in_ptr[u], in_ptr[u + 1]
            s_vo = _lower(out_idx, voa0, vob0, t, t_e)
            s_ui = _lower(in_idx, uia0, uib0, t, t_e)
            n_close = 0
            n_open = 0
            voa = _lower(out_idx, voa0, s_vo, t, low)
            uia = _lower(in_idx, uia0, s_ui, t, low)
            lo, hi = _near_range(voa, s_vo, s_vo, cap_path)
            token += 1
            tk = token
            for i in range(lo, hi):
                p = out_idx[i]
                w = dst[p]
                if w != u and w != v and stamp2[w] != tk:
                    stamp2[w] = tk
                    first_t[w] = t[p]
            lo, hi = _near_range(uia, s_ui, s_ui, cap_path)
            token += 1
            for i in range(lo, hi):
                p = in_idx[i]
                w = src[p]
                if w != u and w != v and stamp2[w] == tk and first_t[w] <= t[p] and stamp[w] != token:
                    stamp[w] = token
                    n_close += 1
            if H_pos:
                vob = _upper(out_idx, s_vo, vob0, t, t_end)
                uib = _upper(in_idx, s_ui, uib0, t, t_end)
                lo, hi = _near_range(s_vo, vob, s_vo, cap_path)
                token += 1
                tk = token
                for i in range(lo, hi):
                    p = out_idx[i]
                    w = dst[p]
                    if p != e and w != u and w != v and stamp2[w] != tk:
                        stamp2[w] = tk
                        first_t[w] = t[p]
                lo, hi = _near_range(s_ui, uib, s_ui, cap_path)
                token += 1
                for i in range(lo, hi):
                    p = in_idx[i]
                    w = src[p]
                    if p != e and w != u and w != v and stamp2[w] == tk and first_t[w] <= t[p] and stamp[w] != token:
                        stamp[w] = token
                        n_open += 1
            buf[o + 1] = np.log1p(n_close)
            buf[o + 2] = np.log1p(n_open)
            # m_conv_out: siblings w' of v under u that share a downstream account with v.
            token += 1
            tkD = token
            voh = _upper(out_idx, s_vo, vob0, t, t_end) if H_pos else s_vo
            lo, hi = _near_range(voa, voh, s_vo, cap_d)
            nD = 0
            for i in range(lo, hi):
                d = dst[out_idx[i]]
                if d != u and d != v:
                    stamp2[d] = tkD
                    nD += 1
            n_conv = 0
            if nD > 0:
                lo, hi = _near_range(ua, ub, s_u, cap_sib)
                token += 1
                for i in range(lo, hi):
                    w = dst[out_idx[i]]
                    if out_idx[i] == e or w == v or w == u or stamp[w] == token:
                        continue
                    stamp[w] = token
                    a0, b0 = out_ptr[w], out_ptr[w + 1]
                    s_w = _lower(out_idx, a0, b0, t, t_e)
                    wa = _lower(out_idx, a0, s_w, t, low)
                    wb = _upper(out_idx, s_w, b0, t, t_end) if H_pos else s_w
                    l2, h2 = _near_range(wa, wb, s_w, cap_sib_nbr)
                    for j in range(l2, h2):
                        if stamp2[dst[out_idx[j]]] == tkD:
                            n_conv += 1
                            break
            buf[o + 3] = np.log1p(n_conv)
            # m_conv_in: siblings w' of u into v that share an upstream account with u.
            token += 1
            tkU = token
            uih = _upper(in_idx, s_ui, uib0, t, t_end) if H_pos else s_ui
            lo, hi = _near_range(uia, uih, s_ui, cap_d)
            nU = 0
            for i in range(lo, hi):
                d = src[in_idx[i]]
                if d != u and d != v:
                    stamp2[d] = tkU
                    nU += 1
            n_conv = 0
            if nU > 0:
                lo, hi = _near_range(va, vb, s_v, cap_sib)
                token += 1
                for i in range(lo, hi):
                    w = src[in_idx[i]]
                    if in_idx[i] == e or w == v or w == u or stamp[w] == token:
                        continue
                    stamp[w] = token
                    a0, b0 = in_ptr[w], in_ptr[w + 1]
                    s_w = _lower(in_idx, a0, b0, t, t_e)
                    wa = _lower(in_idx, a0, s_w, t, low)
                    wb = _upper(in_idx, s_w, b0, t, t_end) if H_pos else s_w
                    l2, h2 = _near_range(wa, wb, s_w, cap_sib_nbr)
                    for j in range(l2, h2):
                        if stamp2[src[in_idx[j]]] == tkU:
                            n_conv += 1
                            break
            buf[o + 4] = np.log1p(n_conv)
            for k in range(n_features):
                F[r, k] = buf[k]
    return F


def compute_features(g: dict, rows: np.ndarray, H: int, view_end: np.ndarray, block_size: int = 4096,
                     caps: tuple = (CAP_DISTINCT, CAP_PATH, CAP_SIB, CAP_D, CAP_SIB_NBR),
                     past_windows: tuple = PAST_WINDOWS) -> np.ndarray:
    """Feature matrix for `rows` (canonical positions) at horizon H seconds (0 = strict causal)."""
    return _compute(np.asarray(rows, dtype=np.int64), int(H), g["t"], g["src"], g["dst"], g["usd"],
                    np.asarray(view_end, dtype=np.int64), g["n_nodes"],
                    g["out_ptr"], g["out_idx"], g["out_cum"], g["in_ptr"], g["in_idx"], g["in_cum"],
                    g["pair_keys"], g["pair_ptr"], g["pair_idx"], g["pair_cum"],
                    np.asarray(past_windows, dtype=np.int64), len(past_windows) * 4 * 6 + 4 * 6 + 8 + 5, block_size, *caps)
