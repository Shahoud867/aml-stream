"""Split the three GPU notebooks (NB1, NB2, NB3) into 10 Colab notebooks that can run at the same time.

It copies notebooks/nb1|nb2|nb3.ipynb and edits only the settings cell (SHARD, TIERS, MODE) and adds a header line, so the
code hash and every job definition stay exactly as frozen. Sharding is the built-in one: shard (i, n) takes every n-th
dependency chain, so no two notebooks run the same job and a replay always runs beside its checkpoint.

Output: gpu_session/colab10/*.ipynb. Run:  python gpu_session/make_colab10.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from amlh.plans import plan  # noqa: E402

# (file name, source notebook, shard, MODE). The first notebook of each kind is the canary: it runs the pilot, the rest go straight to production.
SET = [
    ("01_NB1_gin_s3_plus_formatcheck_PILOT", "NB1", (0, 3), "pilot"),
    ("02_NB1_gin_s1", "NB1", (1, 3), "production"),
    ("03_NB1_gin_s2", "NB1", (2, 3), "production"),
    ("04_NB2_pna_HI_s1_PILOT", "NB2", (0, 4), "pilot"),
    ("05_NB2_pna_HI_s2", "NB2", (1, 4), "production"),
    ("06_NB2_gin_LI_s1", "NB2", (2, 4), "production"),
    ("07_NB2_gin_LI_s2", "NB2", (3, 4), "production"),
    ("08_NB3_tdgnn_part1_PILOT", "NB3", (0, 3), "pilot"),
    ("09_NB3_tdgnn_part2", "NB3", (1, 3), "production"),
    ("10_NB3_tdgnn_part3", "NB3", (2, 3), "production"),
]


def edit(nb: dict, k: int, name: str, notebook: str, shard, mode: str, p0_jobs: list[str]):
    done = 0
    for c in nb["cells"]:
        src = "".join(c["source"])
        if c["cell_type"] == "code" and "# Settings." in src:
            for pat, new in [(r"^MODE = '[a-z]+'", f"MODE = '{mode}'"), (r"^SHARD = None", f"SHARD = {shard}"),
                             (r"^TIERS = .*?(?=   #)", "TIERS = ('P0',)")]:
                src, n = re.subn(pat, new, src, count=1, flags=re.M)
                assert n == 1, (name, pat)
            c["source"] = src.splitlines(keepends=True)
            done += 1
    assert done == 1, name
    head = (f"> **Parallel set, notebook {k} of 10.** Shard {shard[0]} of {shard[1]} of {notebook}; P0 only (add `'P1'` to `TIERS` once P0 is finished). "
            f"{'**Canary: runs the pilot.** Read `aml_pilot/PILOT_REPORT.md`; the other notebooks of this kind should start once it says PASS.' if mode == 'pilot' else 'Production run.'}\n"
            f">\n> P0 jobs in this notebook: " + ", ".join(f"`{j}`" for j in p0_jobs) + "\n\n")
    first = nb["cells"][0]
    first["source"] = (head + "".join(first["source"])).splitlines(keepends=True)


def main():
    out = ROOT / "gpu_session" / "colab10"
    out.mkdir(exist_ok=True)
    seen: dict[str, list[str]] = {}
    for k, (name, notebook, shard, mode) in enumerate(SET, 1):
        nb = json.loads((ROOT / "notebooks" / f"{notebook.lower()}.ipynb").read_text(encoding="utf8"))
        p0 = [j.job_id for j in plan(notebook, shard) if j.tier == "P0"]
        edit(nb, k, name, notebook, shard, mode, p0)
        (out / f"{name}.ipynb").write_text(json.dumps(nb, indent=1), encoding="utf8")
        seen.setdefault(notebook, []).extend(p0)
        print(f"{k:2d} {name:38s} {mode:10s} {len(p0):2d} P0 jobs")
    for notebook, jobs in seen.items():   # every P0 job exactly once
        everything = [j.job_id for j in plan(notebook) if j.tier == "P0"]
        assert sorted(jobs) == sorted(everything), f"{notebook}: jobs missing or duplicated"
    print("every P0 job of NB1, NB2 and NB3 appears exactly once")


if __name__ == "__main__":
    main()
