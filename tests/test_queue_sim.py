import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from amlh.queue_sim import max_stable_rate, service_from_profile, simulate  # noqa: E402


@pytest.mark.parametrize("lam,mu", [(500.0, 1000.0), (800.0, 1000.0)])
def test_mm1_mean_sojourn(lam, mu):
    """Batch size 1, no batching delay, exponential service: mean sojourn time must match 1 / (mu - lam)."""
    rng = np.random.default_rng(0)
    arr = np.cumsum(rng.exponential(1 / lam, 1_000_000))
    svc = lambda n: rng.exponential(1 / mu)  # noqa: E731
    lat = simulate(arr, svc, max_batch=1, max_wait=0.0)
    theory = 1 / (mu - lam)
    assert abs(lat[100000:].mean() - theory) / theory < 0.05


def test_no_event_lost_and_no_negative_latency():
    rng = np.random.default_rng(1)
    arr = np.cumsum(rng.exponential(1e-3, 5000))
    lat = simulate(arr, lambda n: 1e-3 + 1e-5 * n, 64, 0.005)
    assert len(lat) == len(arr)
    assert (lat > 0).all()


def test_batching_waits_at_most_max_wait_when_idle():
    arr = np.array([0.0, 10.0, 20.0])
    lat = simulate(arr, lambda n: 0.001, 8, 0.005)
    assert np.allclose(lat, 0.006)


def test_full_batch_does_not_wait():
    arr = np.zeros(4)
    lat = simulate(arr, lambda n: 0.001, 4, 1.0)
    assert np.allclose(lat, 0.001)


def test_profile_interpolation_hits_measured_points():
    s = service_from_profile([1, 10, 100], [0.001, 0.002, 0.01])
    assert s(10) == pytest.approx(0.002)
    assert s(1) == pytest.approx(0.001)
    assert 0.002 < s(50) < 0.01


def test_stability_detects_overload():
    best, table = max_stable_rate(lambda n: 0.001 * n, 1, [100.0, 500.0, 5000.0], duration=4.0)
    assert best == 500.0
    assert table[-1]["stable"] is False
