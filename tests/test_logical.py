"""Brute-force reference for the logical-clock features, and the invariance they exist for."""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from amlh.features import build_graph, compute_features  # noqa: E402
from amlh.logical import FWD_K, LOGICAL_KS, N_LOGICAL, compute_logical, logical_feature_names  # noqa: E402
from amlh.perturb import perturb_stream  # noqa: E402


def stream(seed, n=120, nodes=9):
    rng = np.random.default_rng(seed)
    t = (np.sort(rng.integers(0, 60, n)) * 3600).astype(np.int64)
    src, dst = rng.integers(0, nodes, n).astype(np.int64), rng.integers(0, nodes, n).astype(np.int64)
    usd = rng.integers(1, 1000, n).astype(np.float64)
    return t, src, dst, usd


def ref(t, src, dst, usd, e, H, view_end):
    n = len(t)
    u, v, t_e = src[e], dst[e], t[e]
    prior = [k for k in range(n) if k != e and (t[k] < t_e if H == 0 else t[k] <= t_e)]
    t_end = min(t_e + H, view_end[e])
    forward = [k for k in range(n) if H > 0 and k != e and t_e < t[k] <= t_end]
    out = []

    def stats(ev, other):
        if not ev:
            return [0.0, 0.0, 0.0, 0.0]
        la = np.log1p([usd[k] for k in ev])
        return [math.log1p(len(ev)), math.log1p(sum(usd[k] for k in ev)), math.log1p(len({other[k] for k in ev})),
                float(la.std()) if len(ev) >= 2 else 0.0]

    sets = [(u, src, dst), (u, dst, src), (v, src, dst), (v, dst, src)]
    for k in LOGICAL_KS:
        for a, side, other in sets:
            ev = sorted([p for p in prior if side[p] == a], key=lambda p: (t[p], p), reverse=True)[:k]
            out += stats(ev, other)
    for k in LOGICAL_KS:
        for a in (u, v):
            items = []
            for p in prior:
                if dst[p] == a:
                    items.append((t[p], p, 1))
                if src[p] == a:
                    items.append((t[p], p, -1))
            # Merged walk: on equal (t, pos), the inflow copy of a self-loop is taken first.
            items.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
            items = items[:k]
            net = sum(sg * usd[p] for _, p, sg in items)
            out += [1.0 if any(sg == 1 for *_, sg in items) else 0.0, float(np.sign(net) * math.log1p(abs(net)))]
    for a in (u, v):
        out.append(math.log1p(sum(int(dst[p] == a) + int(src[p] == a) for p in prior)))
    for a, side, other in sets:
        ev = sorted([p for p in forward if side[p] == a], key=lambda p: (t[p], p))[:FWD_K]
        out += stats(ev, other)
    return np.array(out)


def test_names():
    assert len(logical_feature_names()) == N_LOGICAL


@pytest.mark.parametrize("H", [0, 1, 3600, 2**60])
@pytest.mark.parametrize("seed", range(12))
def test_matches_reference(seed, H):
    t, src, dst, usd = stream(seed)
    view_end = np.full(len(t), t.max() - 3600, dtype=np.int64)
    F = compute_logical(build_graph(t, src, dst, usd), np.arange(len(t)), H, view_end)
    names = logical_feature_names()
    for e in range(len(t)):
        r = ref(t, src, dst, usd, e, H, view_end)
        bad = np.where(~np.isclose(F[e], r, rtol=1e-5, atol=1e-5))[0]
        assert len(bad) == 0, (seed, H, e, [(names[i], float(F[e, i]), float(r[i])) for i in bad][:4])


def mule_stream():
    """A dedicated chain of accounts with no other traffic, plus unrelated background."""
    t = np.array([100, 103, 106, 109, 200, 210]) * 3600
    src = np.array([0, 1, 2, 3, 10, 11])
    dst = np.array([1, 2, 3, 4, 11, 12])
    usd = np.array([100.0, 90.0, 80.0, 70.0, 5.0, 6.0])
    return t.astype(np.int64), src.astype(np.int64), dst.astype(np.int64), usd


def sorted_graph(t, src, dst, usd):
    order = np.lexsort((np.arange(len(t)), t))
    inv = np.empty(len(t), dtype=np.int64)
    inv[order] = np.arange(len(t))
    return build_graph(t[order], src[order], dst[order], usd[order]), inv


def strict_past_logical(t, src, dst, usd, rows):
    g, inv = sorted_graph(t, src, dst, usd)
    return compute_logical(g, inv[rows], 0, np.full(len(t), t.max(), dtype=np.int64))


@pytest.mark.parametrize("W", [3600, 21600, 86400])
@pytest.mark.parametrize("cond", ["stretch", "blend", "shift_stretch"])
def test_order_preserving_retiming_changes_nothing_for_a_mule(W, cond):
    """The reason this family exists: stretching a dedicated account cannot remove its evidence."""
    t, src, dst, usd = mule_stream()
    attempt = np.array([0, 1, 2, 3])
    g = build_graph(t, src, dst, usd)
    t2 = perturb_stream(t, {0: attempt}, W, cond, 0, 10**8, seed=1, g=g)
    assert (np.diff(t2[attempt]) > 0).all()
    assert np.array_equal(strict_past_logical(t, src, dst, usd, attempt), strict_past_logical(t2, src, dst, usd, attempt))


def test_wall_clock_features_do_change_under_the_same_stretch():
    """The control for the test above: if this stopped differing, that test would prove nothing."""
    t, src, dst, usd = mule_stream()
    attempt = np.array([0, 1, 2, 3])
    t2 = perturb_stream(t, {0: attempt}, 86400, "stretch", 0, 10**8)
    ve = np.full(len(t), 10**9, dtype=np.int64)
    a = compute_features(build_graph(t, src, dst, usd), attempt, 0, ve)
    g2, inv = sorted_graph(t2, src, dst, usd)
    b = compute_features(g2, inv[attempt], 0, ve)
    assert not np.array_equal(a, b)


def test_burst_is_a_known_limit_at_strict_past():
    """Documented limit: collapsing into one minute hides siblings from a strict-past logical detector."""
    t, src, dst, usd = mule_stream()
    attempt = np.array([0, 1, 2, 3])
    t2 = perturb_stream(t, {0: attempt}, 10**7, "burst", 0, 10**9)
    assert len(set(t2[attempt].tolist())) == 1
    assert not np.array_equal(strict_past_logical(t, src, dst, usd, attempt), strict_past_logical(t2, src, dst, usd, attempt))


def test_future_rewrite_invariant():
    t, src, dst, usd = stream(3)
    n = len(t)
    ve = np.full(n, t.max(), dtype=np.int64)
    H = 3600
    base = compute_logical(build_graph(t, src, dst, usd), np.arange(n), H, ve)
    rng = np.random.default_rng(1)
    for e in (20, 60, 100):
        s2, d2 = src.copy(), dst.copy()
        later = np.where(t > t[e] + H)[0]
        s2[later] = rng.integers(0, 9, len(later))
        d2[later] = rng.integers(0, 9, len(later))
        F = compute_logical(build_graph(t, s2, d2, usd), np.array([e]), H, ve)
        assert np.array_equal(F[0], base[e])
