"""Independent O(n^2) reference for the horizon feature engine, written from the module docstring
definition using plain Python sets. Uses large caps so capping semantics do not enter."""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from amlh.features import (PAST_WINDOWS, N_FEATURES, build_graph, compute_features, feature_names)  # noqa: E402

BIG = (10**6,) * 5


def ref_features(t, src, dst, usd, e, H, view_end):
    n = len(t)
    t_e, u, v = t[e], src[e], dst[e]
    t_end = min(t_e + H, view_end[e]) if H > 0 else None
    visible = [k for k in range(n) if k != e and (t[k] < t_e or (H > 0 and t[k] <= t_end))]
    out = []

    def stats(events, other, past):
        if not events:
            return [0.0, 0.0, 0.0, -1.0, 0.0, 0.0]
        gaps = [abs(t_e - t[k]) for k in events]
        la = np.log1p([usd[k] for k in events])
        dt = np.array([(t_e - t[k]) / 3600.0 for k in events])
        stds = [float(la.std()), float(dt.std())] if len(events) >= 2 else [0.0, 0.0]
        return [math.log1p(len(events)), math.log1p(sum(usd[k] for k in events)),
                math.log1p(len({other[k] for k in events})), math.log1p(min(gaps))] + stds

    def members(a, direction):
        return [k for k in visible if (src[k] if direction == "out" else dst[k]) == a]

    sets = [(u, "out"), (u, "in"), (v, "out"), (v, "in")]
    for w in PAST_WINDOWS:
        for a, d in sets:
            ev = [k for k in members(a, d) if t_e - w <= t[k] < t_e]
            out += stats(ev, dst if d == "out" else src, True)
    for a, d in sets:
        ev = [k for k in members(a, d) if t[k] >= t_e]
        out += stats(ev, dst if d == "out" else src, False)
    low = t_e - PAST_WINDOWS[-1]
    for side in ("past", "fwd"):
        for a, b in ((u, v), (v, u)):
            if side == "past":
                ev = [k for k in visible if src[k] == a and dst[k] == b and low <= t[k] < t_e]
            else:
                ev = [k for k in visible if src[k] == a and dst[k] == b and t[k] >= t_e]
            out += [math.log1p(len(ev)), math.log1p(sum(usd[k] for k in ev))]

    view = [k for k in visible if t[k] >= low]
    path2 = set()
    for p in view:
        if src[p] == u and dst[p] not in (u, v):
            w = dst[p]
            for q in view:
                if src[q] == w and dst[q] == v and t[p] <= t[q]:
                    path2.add(w)
    out.append(math.log1p(len(path2)))

    close, opened = set(), set()
    for p in view:
        if src[p] == v and dst[p] not in (u, v):
            w = dst[p]
            for q in view:
                if src[q] == w and dst[q] == u and t[p] <= t[q]:
                    if t[p] < t_e and t[q] < t_e:
                        close.add(w)
                    if t[p] >= t_e and t[q] >= t_e:
                        opened.add(w)
    out += [math.log1p(len(close)), math.log1p(len(opened))]

    # Convergent siblings: D and sibling-neighbour lists use the same view, self-loops excluded by endpoint test.
    D = {dst[k] for k in view if src[k] == v and dst[k] not in (u, v)}
    conv = 0
    for w in {dst[k] for k in view if src[k] == u and dst[k] not in (u, v)}:
        if any(src[k] == w and dst[k] in D for k in view):
            conv += 1
    out.append(math.log1p(conv))
    U = {src[k] for k in view if dst[k] == u and src[k] not in (u, v)}
    conv = 0
    for w in {src[k] for k in view if dst[k] == v and src[k] not in (u, v)}:
        if any(dst[k] == w and src[k] in U for k in view):
            conv += 1
    out.append(math.log1p(conv))
    return np.array(out)


def make_stream(rng, n=70, nodes=10, horizon_steps=40):
    # 1-minute grid with deliberate ties and self-loops.
    t = np.sort(rng.integers(0, horizon_steps, n)) * 3600 // 2 * 2
    t = (t // 60) * 60
    src = rng.integers(0, nodes, n)
    dst = rng.integers(0, nodes, n)
    loops = rng.random(n) < 0.08
    dst[loops] = src[loops]
    usd = rng.integers(1, 1000, n).astype(np.float64)
    return t.astype(np.int64), src.astype(np.int64), dst.astype(np.int64), usd


def check(seed, H, n=70, nodes=10):
    rng = np.random.default_rng(seed)
    t, src, dst, usd = make_stream(rng, n, nodes)
    g = build_graph(t, src, dst, usd)
    view_end = np.full(n, t.max() - 3600, dtype=np.int64)  # a split boundary inside the stream
    rows = np.arange(n)
    F = compute_features(g, rows, H, view_end, caps=BIG)
    names = feature_names()
    for e in rows:
        r = ref_features(t, src, dst, usd, int(e), H, view_end)
        bad = np.where(~np.isclose(F[e], r, rtol=1e-5, atol=1e-5))[0]
        assert len(bad) == 0, (seed, H, int(e), [(names[i], float(F[e, i]), float(r[i])) for i in bad])


def test_names():
    assert len(feature_names()) == N_FEATURES


@pytest.mark.parametrize("H", [0, 60, 3600, 6 * 3600, 24 * 3600, 2**60])
@pytest.mark.parametrize("seed", range(15))
def test_engine_matches_reference(seed, H):
    check(seed, H)


@pytest.mark.parametrize("seed", range(5))
def test_dense_small_graph(seed):
    check(100 + seed, 3600, n=90, nodes=5)


def test_strict_causal_ignores_future_and_ties():
    """H = 0: rewriting all events at or after t_e must not change e's features."""
    rng = np.random.default_rng(7)
    t, src, dst, usd = make_stream(rng, 80, 8)
    view_end = np.full(80, t.max(), dtype=np.int64)
    base = compute_features(build_graph(t, src, dst, usd), np.arange(80), 0, view_end, caps=BIG)
    for e in [20, 40, 60]:
        s2, d2, u2 = src.copy(), dst.copy(), usd.copy()
        later = np.where(t >= t[e])[0]
        later = later[later != e]
        s2[later] = rng.integers(0, 8, len(later))
        d2[later] = rng.integers(0, 8, len(later))
        u2[later] = rng.integers(1, 1000, len(later))
        F = compute_features(build_graph(t, s2, d2, u2), np.array([e]), 0, view_end, caps=BIG)
        assert np.array_equal(F[0], base[e])


def test_horizon_view_ignores_beyond_horizon():
    """Finite H: rewriting events after t_e + H must not change e's features."""
    rng = np.random.default_rng(8)
    t, src, dst, usd = make_stream(rng, 80, 8)
    H = 3600
    view_end = np.full(80, t.max(), dtype=np.int64)
    base = compute_features(build_graph(t, src, dst, usd), np.arange(80), H, view_end, caps=BIG)
    for e in [10, 30, 50]:
        s2, d2, u2 = src.copy(), dst.copy(), usd.copy()
        later = np.where(t > t[e] + H)[0]
        s2[later] = rng.integers(0, 8, len(later))
        d2[later] = rng.integers(0, 8, len(later))
        u2[later] = rng.integers(1, 1000, len(later))
        F = compute_features(build_graph(t, s2, d2, u2), np.array([e]), H, view_end, caps=BIG)
        assert np.array_equal(F[0], base[e])


def test_split_clip():
    """Nothing after view_end may be used, even with infinite horizon."""
    rng = np.random.default_rng(9)
    t, src, dst, usd = make_stream(rng, 80, 8)
    cut = int(np.median(t))
    view_end = np.full(80, cut, dtype=np.int64)
    rows = np.where(t <= cut)[0]
    base = compute_features(build_graph(t, src, dst, usd), rows, 2**60, view_end, caps=BIG)
    s2, d2 = src.copy(), dst.copy()
    later = np.where(t > cut)[0]
    s2[later] = rng.integers(0, 8, len(later))
    d2[later] = rng.integers(0, 8, len(later))
    F = compute_features(build_graph(t, s2, d2, usd), rows, 2**60, view_end, caps=BIG)
    assert np.array_equal(F, base)


def test_sampler_never_exceeds_view():
    from amlh.sampler import sample_batch
    rng = np.random.default_rng(3)
    t, src, dst, usd = make_stream(rng, 300, 15)
    g = build_graph(t, src, dst, usd)
    view_end = np.full(300, t.max() - 1800, dtype=np.int64)
    for H in (0, 60, 3600, 2**60):
        seeds = np.arange(300, dtype=np.int64)
        h1, oth, h2 = sample_batch(seeds, H, g["t"], g["src"], g["dst"], view_end, g["out_ptr"], g["out_idx"],
                                   g["in_ptr"], g["in_idx"], 5, 5, 3, 3)
        for b, e in enumerate(seeds):
            limit = (t[e] - 1) if H == 0 else min(t[e] + H, view_end[e])
            for arr in (h1[b].ravel(), h2[b].ravel()):
                p = arr[arr >= 0]
                assert (t[p] <= limit).all()
                assert (p != e).all()
            if H == 0:
                assert (h1[b][..., 5:] == -1).all() and (h2[b][..., 3:] == -1).all()
        assert (h1 >= 0).any()


def test_sampler_future_invariant():
    from amlh.sampler import sample_batch
    rng = np.random.default_rng(4)
    t, src, dst, usd = make_stream(rng, 300, 15)
    view_end = np.full(300, t.max(), dtype=np.int64)
    e = 150
    H = 3600
    g1 = build_graph(t, src, dst, usd)
    s2, d2 = src.copy(), dst.copy()
    later = np.where(t > t[e] + H)[0]
    s2[later] = rng.integers(0, 15, len(later))
    d2[later] = rng.integers(0, 15, len(later))
    g2 = build_graph(t, s2, d2, usd)
    seeds = np.array([e], dtype=np.int64)
    a = sample_batch(seeds, H, g1["t"], g1["src"], g1["dst"], view_end, g1["out_ptr"], g1["out_idx"], g1["in_ptr"], g1["in_idx"], 5, 5, 3, 3)
    b = sample_batch(seeds, H, g2["t"], g2["src"], g2["dst"], view_end, g2["out_ptr"], g2["out_idx"], g2["in_ptr"], g2["in_idx"], 5, 5, 3, 3)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))
