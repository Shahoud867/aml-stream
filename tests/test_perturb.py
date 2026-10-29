import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from amlh.features import build_graph  # noqa: E402
from amlh.perturb import attack_times, background_times, blend_times, perturb_stream  # noqa: E402


def random_attempt(rng):
    n = rng.integers(2, 40)
    return np.sort(rng.integers(0, 200 * 3600, n) // 60) * 60


@pytest.mark.parametrize("how", ["burst", "stretch"])
@pytest.mark.parametrize("W", [3600, 21600, 86400])
def test_budget_order_grid(how, W):
    rng = np.random.default_rng(0)
    for _ in range(300):
        ta = random_attempt(rng)
        lo, hi = int(ta.min()) - 3600, int(ta.max()) + 3600
        out = attack_times(ta, W, how, lo, hi)
        assert (np.diff(out) >= 0).all(), "funds-flow order violated"
        assert (np.abs(out - ta) <= W).all(), "budget exceeded"
        assert (out % 60 == 0).all(), "off the minute grid"
        assert out.min() >= lo and out.max() <= hi


def test_burst_collapses_when_budget_is_large():
    ta = np.array([0, 3600, 7200, 10800])
    out = attack_times(ta, 86400, "burst", 0, 10**6)
    assert len(set(out.tolist())) == 1


def test_stretch_expands_span():
    ta = np.array([1000 * 60, 1100 * 60, 1200 * 60])
    out = attack_times(ta, 3600, "stretch", 0, 10**8)
    assert out[-1] - out[0] == ta[-1] - ta[0] + 2 * 3600


@pytest.mark.parametrize("cond", ["shift_burst", "shift_stretch"])
def test_controls_keep_internal_gaps(cond):
    rng = np.random.default_rng(1)
    t = np.sort(rng.integers(100000, 1000000, 500) // 60) * 60
    pos = np.array([50, 60, 70, 80])
    t2 = perturb_stream(t, {0: pos}, 21600, cond, 0, 2 * 10**6, seed=3)
    assert np.array_equal(np.diff(t2[pos]), np.diff(t[pos]))
    other = np.setdiff1d(np.arange(500), pos)
    assert np.array_equal(t[other], t2[other])


@pytest.mark.parametrize("W", [3600, 21600, 86400])
def test_blend_invariants(W):
    rng = np.random.default_rng(5)
    for _ in range(300):
        ta = random_attempt(rng)
        bg = np.sort(rng.integers(0, 220 * 3600, rng.integers(0, 400)) // 60) * 60
        lo, hi = int(ta.min()) - 3600, int(ta.max()) + 3600
        out = attack_times(ta, W, "blend", lo, hi, bg=bg)
        assert (np.diff(out) >= 0).all()
        assert (np.abs(out - ta) <= W).all()
        assert (out % 60 == 0).all()
        assert out.min() >= lo and out.max() <= hi


def test_blend_no_background_is_a_rigid_shift():
    """An account with no other traffic has no logical time to stretch into."""
    ta = np.array([1000, 1100, 1300, 1600]) * 60
    out = blend_times(ta, np.empty(0, dtype=np.int64), 3600, 0, 10**8)
    assert np.array_equal(np.diff(out), np.diff(ta))


def test_blend_spaces_transfers_evenly_in_logical_time():
    ta = np.array([100, 110, 120]) * 3600
    bg = (np.arange(0, 400) * 3600 // 2 // 60) * 60
    W = 40 * 3600
    out = blend_times(ta, bg, W, 0, 10**9)
    between = [np.searchsorted(bg, out[i + 1]) - np.searchsorted(bg, out[i]) for i in range(2)]
    assert abs(between[0] - between[1]) <= 2
    assert between[0] > np.searchsorted(bg, ta[1]) - np.searchsorted(bg, ta[0])


def test_background_times_excludes_attempt():
    t = np.arange(10) * 60
    src = np.array([0, 0, 1, 2, 3, 0, 4, 5, 6, 0])
    dst = np.array([1, 2, 3, 3, 4, 5, 5, 6, 7, 7])
    g = build_graph(t, src, dst, np.ones(10))
    pos = np.array([0, 2])  # accounts 0, 1, 3
    assert background_times(pos, t, g).tolist() == [60, 180, 240, 300, 540]


def test_shift_blend_keeps_gaps():
    rng = np.random.default_rng(6)
    t = np.sort(rng.integers(100000, 900000, 400) // 60) * 60
    src, dst = rng.integers(0, 30, 400), rng.integers(0, 30, 400)
    g = build_graph(t, src, dst, np.ones(400))
    pos = np.array([50, 90, 130, 170])
    t2 = perturb_stream(t, {0: pos}, 21600, "shift_blend", 0, 2 * 10**6, seed=2, g=g)
    assert np.array_equal(np.diff(t2[pos]), np.diff(t[pos]))


def test_only_attempt_events_move():
    rng = np.random.default_rng(2)
    t = np.sort(rng.integers(0, 10**6, 200) // 60) * 60
    pos = np.array([10, 20, 30])
    t2 = perturb_stream(t, {0: pos}, 3600, "burst", 0, 2 * 10**6)
    mask = np.ones(200, bool)
    mask[pos] = False
    assert np.array_equal(t[mask], t2[mask])
