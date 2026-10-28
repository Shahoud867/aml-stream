"""Discrete-event simulation of a single FIFO scoring worker with dynamic batching.

When the worker is free it takes up to `max_batch` waiting events. If fewer are waiting, it waits until
the oldest waiting event has aged `max_wait` seconds or the queue reaches max_batch, whichever is first.
The service time of a batch of n events comes from service(n). Latency = completion - arrival.
"""
from __future__ import annotations

import numpy as np


def simulate(arrivals: np.ndarray, service, max_batch: int = 1024, max_wait: float = 0.005):
    """arrivals sorted ascending (seconds). Returns per-event latency and per-event completion time.
    service(n) -> seconds; may be stochastic. Batches never reorder events."""
    n = len(arrivals)
    latency = np.empty(n)
    i = 0
    free_at = 0.0
    while i < n:
        start = max(free_at, arrivals[i])
        # Events already waiting when the worker can start.
        j = int(np.searchsorted(arrivals, start, side="right"))
        if j - i >= max_batch:
            j = i + max_batch
        else:
            deadline = arrivals[i] + max_wait
            if deadline > start:
                # Wait for more arrivals until the oldest event's deadline or a full batch.
                k = int(np.searchsorted(arrivals, deadline, side="right"))
                j2 = min(k, i + max_batch)
                if j2 == i + max_batch:
                    start = max(start, arrivals[j2 - 1])
                else:
                    start = deadline
                j = j2
        size = j - i
        done = start + service(size)
        latency[i:j] = done - arrivals[i:j]
        free_at = done
        i = j
    return latency


def service_from_profile(sizes, medians):
    """Log-log interpolation of a measured service-time profile."""
    ls, lm = np.log(np.asarray(sizes, float)), np.log(np.asarray(medians, float))
    return lambda n: float(np.exp(np.interp(np.log(max(n, 1)), ls, lm)))


def max_stable_rate(service, max_batch, rates, duration=2.0, seed=0, max_wait=0.005):
    """Largest rate whose median latency in the final fifth of the run is not growing
    relative to the middle fifth (a practical stability criterion)."""
    rng = np.random.default_rng(seed)
    best = None
    table = []
    for lam in rates:
        n = max(int(lam * duration), 2000)
        arr = np.cumsum(rng.exponential(1 / lam, n))
        lat = simulate(arr, service, max_batch, max_wait)
        a, b = np.median(lat[2 * n // 5:3 * n // 5]), np.median(lat[4 * n // 5:])
        stable = bool(b <= 1.5 * a + 1e-3)
        table.append(dict(rate=float(lam), p50=float(np.percentile(lat[n // 10:], 50)),
                          p95=float(np.percentile(lat[n // 10:], 95)), p99=float(np.percentile(lat[n // 10:], 99)),
                          stable=stable))
        if stable:
            best = float(lam)
    return best, table
