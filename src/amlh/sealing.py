"""Sealed scoring (DECISIONS.md D-012).

On confirmatory datasets, jobs write test scores but never compute or display test metrics, because a person
who sees them could tune. Metrics are computed once per dataset in the merge step, and that is logged.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

SEALED = frozenset({"LI-Small", "HI-Medium", "LI-Medium", "SAML-D"})


class SealedError(RuntimeError):
    pass


def is_sealed(dataset: str) -> bool:
    return dataset in SEALED


def guard_test_metrics(dataset: str, unsealed: bool = False):
    """Call before computing anything from test labels. Only the merge step passes unsealed=True."""
    if is_sealed(dataset) and not unsealed:
        raise SealedError(f"{dataset} is sealed: test metrics may only be computed in the merge step")


def score_health(scores) -> dict:
    """Checks that use no labels, safe to display on a sealed dataset."""
    import numpy as np
    s = np.asarray(scores, dtype=float)
    return {"n": int(s.size), "n_nan": int(np.isnan(s).sum()), "min": float(np.nanmin(s)) if s.size else None,
            "max": float(np.nanmax(s)) if s.size else None, "n_distinct_sample": int(np.unique(s[:100000]).size)}


class UnsealLog:
    """Append-only record of every unsealing. A repeat under different code is labelled exploratory."""

    def __init__(self, path: Path):
        self.path = Path(path)

    def entries(self) -> list[dict]:
        if not self.path.exists():
            return []
        return [json.loads(x) for x in self.path.read_text().splitlines() if x.strip()]

    def status(self, dataset: str, code_hash: str) -> str:
        previous = [e for e in self.entries() if e["dataset"] == dataset]
        if not previous:
            return "first"
        return "repeat" if any(e["code_hash"] == code_hash for e in previous) else "exploratory"

    def record(self, dataset: str, code_hash: str, n_jobs: int, complete: bool) -> str:
        status = self.status(dataset, code_hash)
        with open(self.path, "a") as fh:
            fh.write(json.dumps({"ts": time.time(), "dataset": dataset, "code_hash": code_hash, "n_jobs": n_jobs,
                                 "complete": complete, "status": status}) + "\n")
        return status
