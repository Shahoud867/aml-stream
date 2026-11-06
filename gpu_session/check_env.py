"""Pre-flight check for a GPU machine. Prints PASS or FAIL for each item and exits non-zero on any failure.

  python gpu_session/check_env.py                 (run with the GPU environment's python)
  --allow-cpu     rehearsal on a machine with no GPU: the GPU items are reported but do not fail
  --official DIR  where setup_windows.ps1 cloned Multi-GNN and FraudGT
"""
from __future__ import annotations

import argparse
import importlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
PINS = {"Multi-GNN": "252b0252afca109d1d216c411c59ff70753b25fc", "FraudGT": "f88db9dc003a6f521a90631c4741541686043119"}
bad: list[str] = []


def report(ok: bool, what: str, detail: str = "", soft: bool = False):
    print(f"[{'PASS' if ok else ('WARN' if soft else 'FAIL')}] {what}{(': ' + detail) if detail else ''}")
    if not ok and not soft:
        bad.append(what)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--allow-cpu", action="store_true")
    ap.add_argument("--official", default=str(ROOT.parent / "aml_official"))
    ap.add_argument("--data", default=str(ROOT / "data" / "raw"))
    a = ap.parse_args()

    v = sys.version_info
    report((3, 10) <= v[:2] <= (3, 12), "Python 3.10 to 3.12", sys.version.split()[0])
    try:
        import torch
        report(torch.__version__.startswith("2.5.1"), "torch 2.5.1", torch.__version__)
        cuda = torch.cuda.is_available()
        report(cuda, "CUDA is usable by torch", f"{torch.version.cuda}" if cuda else "this is a CPU build or the driver is not seen", soft=a.allow_cpu)
        if cuda:
            p = torch.cuda.get_device_properties(0)
            report(p.total_memory / 2**30 > 7.5, "GPU memory", f"{torch.cuda.get_device_name(0)}, {p.total_memory / 2**30:.1f} GB")
            x = torch.randn(2048, 2048, device="cuda")
            report(bool(torch.isfinite((x @ x).sum()).item()), "a matrix multiply runs on the GPU")
    except Exception as e:
        report(False, "torch imports", repr(e))
    for m in ("torch_scatter", "torch_sparse", "torch_geometric", "polars", "numba", "lightgbm", "pyarrow", "sklearn", "pandas", "numpy", "tqdm", "wandb"):
        try:
            mod = importlib.import_module(m)
            report(True, f"import {m}", getattr(mod, "__version__", ""))
        except Exception as e:
            report(False, f"import {m}", repr(e)[:200])
    try:
        import numpy
        report(numpy.__version__.startswith("1."), "numpy below 2 (the official code needs it)", numpy.__version__)
        import torch_geometric
        report(torch_geometric.__version__ == "2.7.0", "torch-geometric 2.7.0 (2.6.1 breaks the official model builder)", torch_geometric.__version__)
        import torch_scatter  # noqa: F401  (import order: torch first)
        from torch_geometric.nn import GINEConv  # noqa: F401
    except Exception as e:
        report(False, "PyG stack works together", repr(e)[:300])

    for name, commit in PINS.items():
        d = Path(a.official) / name
        head = subprocess.run(["git", "-C", str(d), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip() if d.exists() else ""
        report(head == commit, f"{name} at the pinned commit", head[:10] or "not found")

    from amlh.runner import code_hash
    from amlh.session import EXPECTED_RAW, verify_raw
    h = code_hash(ROOT)
    nb = json.loads((ROOT / "notebooks" / "nb1.ipynb").read_text(encoding="utf8"))
    code = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    want = re.search(r"EXPECTED_CODE_HASH = '([0-9a-f]+)'", code).group(1)
    report(h == want, "code hash equals the one the notebooks expect", f"{h} (notebooks expect {want})")
    for ds in ("HI-Small", "LI-Small"):
        try:
            verify_raw(ds, Path(a.data), check_hash=True)
            report(True, f"raw {ds}_Trans.csv is whole (row count and SHA-256)", f"{EXPECTED_RAW[ds][0]} rows")
        except Exception as e:
            report(False, f"raw {ds}_Trans.csv", str(e)[:200])
    free = shutil.disk_usage(ROOT).free / 2**30
    report(free > 40, "free disk space above 40 GB", f"{free:.0f} GB")
    try:
        import ctypes

        class MS(ctypes.Structure):
            _fields_ = [("l", ctypes.c_ulong), ("m", ctypes.c_ulong), ("total", ctypes.c_ulonglong), ("avail", ctypes.c_ulonglong), ("a", ctypes.c_ulonglong), ("b", ctypes.c_ulonglong), ("c", ctypes.c_ulonglong), ("d", ctypes.c_ulonglong), ("e", ctypes.c_ulonglong)]
        s = MS(); s.l = ctypes.sizeof(MS); ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(s))
        report(s.total / 2**30 >= 16, "RAM of at least 16 GB (data preparation peaks near 12 GB for HI-Small-sized work)", f"{s.total / 2**30:.0f} GB", soft=True)
    except Exception:
        pass
    print("\nALL CHECKS PASSED" if not bad else f"\n{len(bad)} CHECK(S) FAILED: " + "; ".join(bad))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
