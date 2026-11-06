"""Import raw result folders into results/ byte for byte, without ever overwriting different data.

Rules, per file (checked in this order):
  destination missing                         -> copy
  destination identical (sha256)              -> skip
  destination equals source after CR/LF only  -> replace with the source bytes (git's autocrlf had rewritten the destination;
                                                  the source is the original that the DONE hash was computed on)
  anything else                               -> CONFLICT: nothing is written for that file, it is reported, exit code 1

Every copied file is re-read and its sha256 compared with the source. A manifest (TSV: action, source, destination, sha256, bytes)
is written so the origin of every imported file stays auditable.

  python gpu_session/consolidation/import_results.py [--dry-run]
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BACKUP = Path(r"C:\dev\aml-backups\2026-11-06")
EXTRACT = Path(r"C:\dev\aml-extract\PC-18")

# (label, source folder, destination under results/)
PLAN = [
    ("PC-8 shard 03 (NB1 Multi-GIN seed 2)", BACKUP / "PC-8" / "aml_run", "shard03_NB1_gin_s2"),
    ("PC-8 console log", BACKUP / "PC-8" / "aml_out_nb1_s2.log", "shard03_NB1_gin_s2/aml_out_nb1_s2.log"),
    ("PC-18 shard 08 production (NB3 part 1)", EXTRACT / "08_NB3_tdgnn_part1_PILOT" / "aml_run", "shard08_NB3_tdgnn_part1"),
    ("PC-18 shard 08 canary pilot", EXTRACT / "08_NB3_tdgnn_part1_PILOT" / "aml_pilot", "shard08_NB3_tdgnn_part1_pilot"),
    ("PC-18 shard 09 (NB3 part 2)", EXTRACT / "09_NB3_tdgnn_part2" / "aml_run", "shard09_NB3_tdgnn_part2"),
    ("PC-18 shard 10 (NB3 part 3)", EXTRACT / "10_NB3_tdgnn_part3" / "aml_run", "shard10_NB3_tdgnn_part3"),
    ("PC-18 shard 05 PARTIAL (Multi-PNA seed 2, batch 1024, 14 of 100 epochs, no DONE)", EXTRACT / "05_NB2_pna_HI_s2" / "aml_run", "shard05_NB2_pna_HI_s2_PARTIAL_bs1024"),
    ("PC-18 shard 05 FAILED (Multi-PNA seed 2, batch 8192, out of memory)", EXTRACT / "05_NB2_pna_HI_s2" / "aml_run_failed_oom_batch8192", "shard05_NB2_pna_HI_s2_FAILED_oom_bs8192"),
    ("PC-17 raw NB2 pilot (batch 512)", BACKUP / "PC-17" / "aml_out_nb2_pilot_bs512", "pc17/aml_out_nb2_pilot_bs512"),
]
for name in ("probe_nb1.log", "probe_nb1b.log", "probe_nb1c.log", "probe_pna.log", "run_all.log", "run_all_4.log", "run_all_9_10.log", "run_all_10_retry.log", "setup.log"):
    PLAN.append((f"PC-18 operational log {name}", EXTRACT / name, f"pc18_logs/{name}"))


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def same_but_line_endings(a: Path, b: Path) -> bool:
    if a.stat().st_size > 50_000_000 or b.stat().st_size > 50_000_000:
        return False
    return a.read_bytes().replace(b"\r\n", b"\n") == b.read_bytes().replace(b"\r\n", b"\n")


def files_of(src: Path):
    if src.is_file():
        yield src, Path(src.name)
    else:
        for p in sorted(src.rglob("*")):
            if p.is_file():
                yield p, p.relative_to(src)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    manifest, conflicts, counts = [], [], {"copy": 0, "skip": 0, "replace_line_endings": 0}
    for label, src, dest in PLAN:
        if not src.exists():
            print(f"MISSING SOURCE: {label}: {src}")
            conflicts.append(f"missing source {src}")
            continue
        dest_root = ROOT / "results" / dest
        n_before = sum(counts.values())
        for p, rel in files_of(src):
            d = dest_root if src.is_file() else dest_root / rel
            h = sha(p)
            if not d.exists():
                action = "copy"
            elif sha(d) == h:
                action = "skip"
            elif same_but_line_endings(p, d):
                action = "replace_line_endings"
            else:
                conflicts.append(f"CONFLICT (different content, left untouched): {d}  <-  {p}")
                continue
            if action != "skip" and not a.dry_run:
                d.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(p, d)
                if sha(d) != h:
                    conflicts.append(f"COPY VERIFY FAILED: {d}")
                    continue
            counts[action] += 1
            manifest.append(f"{action}\t{p}\t{d.relative_to(ROOT)}\t{h}\t{p.stat().st_size}")
        print(f"{label}: {sum(counts.values()) - n_before} files")
    print("totals:", counts)
    if not a.dry_run:
        out = ROOT / "results" / "IMPORT_MANIFEST.tsv"
        out.write_text("action\tsource\tdestination\tsha256\tbytes\n" + "\n".join(manifest) + "\n", encoding="utf8", newline="\n")
        print("manifest:", out)
    for c in conflicts:
        print(c)
    sys.exit(1 if conflicts else 0)


if __name__ == "__main__":
    main()
