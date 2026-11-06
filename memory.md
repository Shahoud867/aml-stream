# Session memory: GPU planning and 10-way split

**Date:** 4 November 2026. **Repo:** `C:\dev\aml-stream` (branch `main`, last commit `c15fea1`). **Nothing from this session is committed;** the new files are untracked in `gpu_session/` and this file.

## 1. Project context (what the session established)

- Phase 1 (`docs/course/CN_Assignment_1.pdf`): topic "Directed Multigraph GNNs for Real-Time AML on Cryptocurrency Transaction Networks". Four gaps: directed multigraph model, real-time/streaming evaluation, generalization, explainability.
- Phase 2 Revision 3 (`paper/cn_phase2.tex`, 1 Nov 2026) **pivoted** to "Decision Horizons in Graph-Based AML". The four gaps became G1 (directed multigraph, now a gated extension, not built), G2 (decision horizon, primary), G3 (one chronological protocol), G4 (explainability, deferred), plus new G5 (benchmark integrity) and G6 (timing robustness). Reasons are in `docs/DECISIONS.md` D-002 to D-005.
- Phase 3 brief (`docs/course/Phase 03 - Implementation.pdf`): seven components (implementation, dataset, baselines, graphs, comparison, ablations, discussion). Mapping and status in `docs/ROADMAP.md` section 4. Progress milestone: **15 November 2026**.
- **Assessment given:** the Revision 3 direction (benchmark audit, horizon curves, timing-invariant detection) has a better publication chance than DMT-GNN, which is incremental next to MEGA, DIAM and Multi-GNN. It depends on whether the official Multi-GNN shows the same tail and horizon effects as LightGBM (Gate G1). Evidence is HI-Small only, synthetic data, 86 test attempts. Venue dates in the plan are unconfirmed.
- **Risks flagged:** no record of the instructor's reply to the direction change; Phase 2 class feedback missing; the discussion section must explain why Gaps 1 and 4 were demoted.

## 2. Compute situation

- Three RTX 3080s (10 GB each, Windows/WDDM, driver 610.60, CUDA 13.3): two until about 12:00, one until 15:00. A later message mentioned four PCs; unconfirmed. Each PC shows one GPU.
- This laptop (HP, user `shaho`) has **no GPU**. The 3080 PCs are university machines (user `i232515`).
- Plan estimates: NB1 30-65 GPU h, NB2 30-60, NB3 20-40. Only anchor: about 3.6 h for one 100-epoch Multi-GIN run on an RTX 3090. A full run does not fit a 3-hour GPU.
- **The GPU code path has never run on a GPU, and the official Multi-GNN code has never been executed** (ROADMAP, NOTEBOOK_METHODOLOGY section 8). Treat the first pilots as testing.
- Colab: a free account gives few concurrent GPU sessions; using several accounts to get around limits breaks the terms and was not assisted.

## 3. What was built (all untracked, code hash unchanged at `9686ce184d525c0a`)

`gpu_session/` sits outside `src/amlh` and `scripts/`, the only folders the code hash covers.

| File | Purpose |
|---|---|
| `gpu_session/run_gpu.py` | Drives one NB1/NB2/NB3 notebook from a shell: modes `list`, `setup`, `pilot`, `production`; `--shard i/n`, `--only REGEX`, `--until HH:MM`, `--n-epochs` (a deviation from the frozen 100 epochs, must be logged). Pilot mode exits non-zero on failure. |
| `gpu_session/make_colab10.py` | Generates the 10 Colab notebooks from `notebooks/nb1..3.ipynb` by editing only `MODE`, `SHARD`, `TIERS` and adding a header. Asserts every P0 job appears exactly once. |
| `gpu_session/colab10/*.ipynb` | The 10 notebooks (all compile; not run on Colab). |
| `gpu_session/setup_windows.ps1` | One-command Windows environment: Python 3.10-3.12, venv `.venv-gpu`, torch 2.5.1 CUDA 12.1 build (default Windows pip build has no GPU), PyG wheels, official repos cloned from local `third_party/` at pinned commits, then `check_env.py`. `-Cpu` for rehearsal. |
| `gpu_session/check_env.py` | Pre-flight: torch/CUDA, GPU memory, imports, numpy<2, torch-geometric 2.7.0, pinned commits, code hash, raw CSV row counts and SHA-256, disk, RAM. |
| `gpu_session/run_all.py` | Unattended launcher for all 10 shards: pre-flight, prepare data once, canary pilots (a failed kind is not launched), then production as GPU capacity allows (Multi-GNN shard costs 2 of 3 capacity units per GPU, TD-GNN 1), stops at `--until`. Resumes on re-run. |
| `gpu_session/README.md` | The four-step instructions for the 3080 PC. |

### The 10-way split (same shards for Colab and `run_all.py`)

| # | Shard | P0 jobs |
|---|---|---|
| 01 | NB1 shard 0/3: format check + Multi-GIN seed 3 (**pilot canary**) | 5 |
| 02 | NB1 shard 1/3: Multi-GIN seed 1 (the E1/Gate G1 result) | 4 |
| 03 | NB1 shard 2/3: Multi-GIN seed 2 | 4 |
| 04 | NB2 shard 0/4: Multi-PNA HI-Small seed 1 (**pilot canary**) | 3 |
| 05 | NB2 shard 1/4: Multi-PNA HI-Small seed 2 | 3 |
| 06 | NB2 shard 2/4: Multi-GIN LI-Small seed 1 | 3 |
| 07 | NB2 shard 3/4: Multi-GIN LI-Small seed 2 | 3 |
| 08 | NB3 shard 0/3: TD-GNN part 1, includes leakage check (**pilot canary**) | 14 |
| 09 | NB3 shard 1/3: TD-GNN part 2 | 14 |
| 10 | NB3 shard 2/3: TD-GNN part 3 | 13 |

All tiers set to P0 only; add `'P1'` after P0 finishes. P1/P2 (FraudGT, ablations, other horizons, latency profile) are not covered.

## 4. Verification status (be honest about this)

- **Done:** job lists print correctly for NB1/NB2/NB3; the 10 notebooks compile; `run_all.py --dry-run` prints the plan; code hash matches the notebooks.
- **In progress when this file was written:** the full test suite (about 70% done, no failure so far); the real `setup_windows.ps1 -Cpu` rehearsal installing into a scratch venv (at the PyG wheel step).
- **Not done:** the planned CPU rehearsal of the NB3, NB1 and NB2 pilots through `run_all.py`; nothing has run on a GPU or on Colab; `setup_windows.ps1` has not completed once. The setup script had two Windows edge-case fixes made while it was running (no `py` launcher); the fixed version has not been run.
- Known mismatch: `.venv` here has torch 2.15 CPU and polars etc.; `.venv-mgnn` has torch 2.5.1 CPU but lacks polars, numba, lightgbm, pyarrow, pytest. The GPU environment must contain both sets, which is what the setup script installs.
- **"Error free" cannot be promised.** The canary pilots on the PC are designed to catch GPU-specific failures first.

## 5. Advice already given to the user

- Start the three pilots first (01, 04, 08), then 02 right after the NB1 pilot passes. Suggested commands: `run_all.py --only 1,2 --until 14:30` on the long-window PC, `--only 4,5` and `--only 8,9` / `--only 9,10` (or `10`) on the others, each with `--until 11:40`. Do not let two PCs run the same shard. These commands were suggestions and are untested.
- Running shards 4, 5, 6 together on one 3080 is not recommended: all are Multi-GNN, VRAM unmeasured, an out-of-memory failure is deliberate and loud, three runs share one GPU so each is about 3x slower, and 4 is the pilot. Run 4 alone, watch `nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv -l 5`, add another only below about 4 GB and 60% utilization. `--capacity 6` forces it.
- For a 3-hour GPU prefer the lighter TD-GNN shards (8, 9, 10).
- Raw data for the PCs: only `HI-Small_Trans.csv` (476 MB) and `LI-Small_Trans.csv` (650 MB) plus the `*_Patterns.txt` files; `data/raw` here also holds the Medium files (about 6 GB).

## 6. Open items and next steps

1. Finish the CPU rehearsal: wait for the setup run, then run the NB3, NB1 and NB2 pilots on CPU in the rehearsal environment and fix what breaks. Note D-007: local execution of third-party Multi-GNN was previously blocked by the permission policy and must not be worked around.
2. Read the full test-suite result in `%TEMP%\tests_full.log`.
3. Confirm with the user: how many PCs and GPUs, internet and git on them, Colab plan concurrency.
4. Decide whether to use `--n-epochs` (shorter than official 100 epochs) for more Multi-GNN runs; if so, add a DECISIONS.md entry and keep those runs separate from the E1 headline.
5. Not GPU work but gating: the leakage check at H > 0 (NB4 stage 0) is CPU-only and should run first on this laptop. It was offered, not started.
6. Roadmap items still open: tell the instructor about the direction change (task 3), Phase 2 class feedback, SAML-D, Phase 3 progress report by 12 Nov, submit by 14 Nov.
7. Commit the `gpu_session/` files only when the user asks.

## 7. Update — 6 November 2026 (PC 18 status, written from Claude's own memory file)

This section mirrors `project_aml_gpu_shard_run.md` kept in Claude's persistent memory for this project
(`C:\Users\i232515\.claude\projects\d--CNET-PROJ-aml-stream\memory\`), added to the workspace copy on request.

**Shard map recap:** 01-03 = NB1 Multi-GIN HI-Small; 04-07 = NB2 Multi-PNA/Multi-GIN; 08-10 = NB3 TD-GNN.

On this machine (**PC 18**):
- **Shard 09, 10**: fully DONE (30 jobs each). Committed to git as part of `aafef70`, reported pushed.
- **Shard 08**: only a pilot ran here; full production is elsewhere.
- **Shard 05** (Multi-PNA HI-Small seed 2): actively training, resumed from checkpoint, `batch_size` forced to 1024 (see D-020 below). ~15.3-15.8 min/epoch, 100 epochs total; last check was around epoch 31/100.
- **Shard 04**: only a pilot ran on PC 18; full shard 04 assigned to PC 17, which hit the same OOM and also had to drop to batch_size 1024.

Elsewhere (per the user's own status reports, not independently verified):
- Shard 01 failed on PC 17 ("correlation gate didn't clear") — unresolved, no diagnostic detail obtained yet.
- Shard 02 done on PC 9.
- Shard 03 partially done (2/3) on PC 8.
- Shards 06, 07 (Multi-GIN LI-Small) unclaimed/unattempted anywhere as of last check.

**D-020 deviation (logged in `docs/DECISIONS.md`):** Multi-PNA's official `batch_size` (8192) does not fit a 10 GB RTX 3080 — confirmed real CUDA OOM at ~8.8 GB allocated, not the Windows memory-spill bug. Fix: `--batch-size 1024` flag on `run_gpu.py`. This changes gradient noise/batchnorm stats vs. the frozen 8192 default, so it's a scientific deviation, not a pure bug fix, and must be reported wherever shards 04/05 run with it.

A Kaggle-style environment (4 CPU, 32 GB RAM, RTX Pro 6000 Blackwell, 96 GB VRAM, 12 h session cap) was discussed for re-running shard 04 at the *true* official batch_size 8192 (96 GB easily covers the ~8.8 GB need, so no deviation needed there) — but real per-epoch timing on that hardware was never measured, so whether 100 epochs fits one 12 h session is unconfirmed. A short probe run was recommended before committing a full session.

**Git status:** commit `aafef70` (fix_pyg.py sitecustomize hook + run_gpu.py `--batch-size` flag + DECISIONS.md D-020) exists locally on PC 18 but could not be pushed to `origin/main` — the configured Git Credential Manager needs interactive browser auth, unavailable in this non-interactive session. Needs either a manual push by the user or a PAT.

See also the operational lessons captured in Claude's memory file `feedback_pc18_session_ops.md` (lock vs. sign-out, the GPU memory-spill fix, confirming GPU power/clocks are already at hardware max on this card, safe checkpoint-copy timing, no git/admin by default on this PC).
