"""A small AMLworld-shaped dataset, written as raw files so that preparation is exercised too."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def write_raw(folder: Path, name: str, seed: int = 0, days: int = 12, doc_days: int = 10, per_day: int = 700):
    """Background traffic on documented days, plus planted fan-out attempts that continue into a tail."""
    rng = np.random.default_rng(seed)
    rows, attempts = [], []
    bank = lambda a: f"{(a % 5):03d}"  # noqa: E731

    def stamp(t):
        d, rest = t // 86400, t % 86400
        return f"2022/09/{d + 1:02d} {rest // 3600:02d}:{(rest % 3600) // 60:02d}"

    def line(t, u, v, amt, label, cur="US Dollar", fmt="ACH"):
        return f"{stamp(t)},{bank(u)},{u:08X},{bank(v)},{v:08X},{amt:.2f},{cur},{amt:.2f},{cur},{fmt},{label}"
    for d in range(doc_days):
        for _ in range(per_day):
            t = int(d * 86400 + rng.integers(0, 1440) * 60)
            u, v = rng.integers(0, 300, 2)
            rows.append((t, line(t, int(u), int(v), float(rng.uniform(5, 5000)), 0, str(rng.choice(["US Dollar", "Euro"])), str(rng.choice(["ACH", "Wire", "Cheque"])))))
    hub = 1000
    for k in range(60):
        start = int(rng.integers(0, doc_days - 1)) * 86400 + int(rng.integers(0, 1440)) * 60
        lines = []
        for i in range(int(rng.integers(5, 10))):
            t = min(start + i * int(rng.integers(30, 200)) * 60, (days - 1) * 86400 + 86340)
            t = (t // 60) * 60
            ln = line(t, hub + k, 2000 + k * 20 + i, float(rng.uniform(900, 9000)), 1)
            rows.append((t, ln))
            lines.append(ln)
        attempts.append(lines)
    # Keep the file in shuffled row order, like the real data.
    order = rng.permutation(len(rows))
    folder.mkdir(parents=True, exist_ok=True)
    with open(folder / f"{name}_Trans.csv", "w") as fh:
        fh.write("Timestamp,From Bank,Account,To Bank,Account,Amount Received,Receiving Currency,Amount Paid,Payment Currency,Payment Format,Is Laundering\n")
        for i in order:
            fh.write(rows[i][1] + "\n")
    with open(folder / f"{name}_Patterns.txt", "w") as fh:
        for lines in attempts:
            fh.write("BEGIN LAUNDERING ATTEMPT - FAN-OUT:  test\n")
            fh.write("\n".join(lines) + "\n")
            fh.write("END LAUNDERING ATTEMPT - FAN-OUT\n\n")


@pytest.fixture(scope="session")
def small_data(tmp_path_factory):
    """Prepared data for one unsealed and one sealed dataset name."""
    from amlh.jobs_gbt import Paths
    from amlh.prepare import DOCUMENTED_DAYS
    root = tmp_path_factory.mktemp("aml")
    raw = root / "raw"
    for name, seed in (("HI-Small", 1), ("LI-Small", 2)):
        write_raw(raw, name, seed)
    DOCUMENTED_DAYS.update({"HI-Small": 10, "LI-Small": 10})
    Paths.raw = {"HI-Small": raw, "LI-Small": raw}
    Paths.cache = root / "cache"
    return root
