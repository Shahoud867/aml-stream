"""Verify every job folder under results/ against its DONE marker, without trusting any summary text.

For each job (a folder holding a DONE file) it reports: OK, or how many listed files are missing / hash-mismatched.
A mismatch is classified as 'line-ending only' when converting LF to CRLF in the file reproduces the recorded sha256
(the signature of git's autocrlf having rewritten a log that the PC wrote with CRLF).

  python gpu_session/consolidation/verify_results.py                 # report
  python gpu_session/consolidation/verify_results.py --restore-crlf  # rewrite ONLY files whose CRLF form reproduces the recorded hash
  python gpu_session/consolidation/verify_results.py --json out.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def sha_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ROOT / "results"))
    ap.add_argument("--restore-crlf", action="store_true")
    ap.add_argument("--json")
    a = ap.parse_args()
    rows, restored = [], 0
    for done in sorted(Path(a.root).rglob("DONE")):
        job = done.parent
        try:
            files = json.loads(done.read_text(encoding="utf8"))["files"]
        except Exception as e:
            rows.append({"job": str(job.relative_to(a.root)), "state": f"UNREADABLE DONE: {e}", "listed": 0, "missing": [], "mismatch": [], "crlf_only": []})
            continue
        missing, mismatch, crlf_only = [], [], []
        for rel, want in files.items():
            p = job / rel
            if not p.exists():
                missing.append(rel)
                continue
            data = p.read_bytes()
            if sha_bytes(data) == want:
                continue
            as_crlf = data.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")
            if sha_bytes(as_crlf) == want:
                if a.restore_crlf:
                    p.write_bytes(as_crlf)
                    restored += 1
                else:
                    crlf_only.append(rel)
            else:
                mismatch.append(rel)
        state = "OK" if not (missing or mismatch or crlf_only) else "INCOMPLETE" if missing else "MISMATCH" if mismatch else "LINE-ENDINGS"
        rows.append({"job": str(job.relative_to(a.root)).replace("\\", "/"), "state": state, "listed": len(files), "missing": missing, "mismatch": mismatch, "crlf_only": crlf_only})
    by = defaultdict(lambda: defaultdict(int))
    for r in rows:
        top = r["job"].split("/")[0] if not r["job"].startswith("pc17/") else "/".join(r["job"].split("/")[:2])
        by[top][r["state"]] += 1
    print(f"{'folder':58s} jobs  " + "  ".join(f"{s:>12s}" for s in ("OK", "INCOMPLETE", "MISMATCH", "LINE-ENDINGS")))
    tot = defaultdict(int)
    for k in sorted(by):
        print(f"{k:58s} {sum(by[k].values()):4d}  " + "  ".join(f"{by[k].get(s, 0):12d}" for s in ("OK", "INCOMPLETE", "MISMATCH", "LINE-ENDINGS")))
        for s, n in by[k].items():
            tot[s] += n
    print(f"{'TOTAL':58s} {sum(tot.values()):4d}  " + "  ".join(f"{tot.get(s, 0):12d}" for s in ("OK", "INCOMPLETE", "MISMATCH", "LINE-ENDINGS")))
    if a.restore_crlf:
        print("files rewritten with their original CRLF bytes (each proven by the recorded sha256):", restored)
    if a.json:
        Path(a.json).write_text(json.dumps(rows, indent=1), encoding="utf8")


if __name__ == "__main__":
    main()
