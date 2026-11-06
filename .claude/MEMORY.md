# PC-17 Session Memory

This file tracks everything done **on this specific machine (PC-17)** during Claude Code sessions working on
the AML look-ahead research project (`aml-stream`). It is local history/status, not the cross-session user-preference
memory system — purpose is so anyone (or any future session) picking up this PC knows exactly what ran, what broke,
what got fixed, and what's currently going.

Last updated: 2026-11-06, ~late morning session.

---

## 1. What this PC's job has been

Standing instruction for this session: autonomously diagnose and fix errors in the project's GPU notebooks run
locally, run notebooks to completion where reasonable, and act as a general problem-solver for the multi-PC
campaign (the user runs the same notebooks in parallel across ~5-6 machines: PC8, PC9, PC17 = this one, PC18, and
a university RTX 4000 Ada 20GB machine).

Campaign context (per `docs/PHASE3_PLAN.md` / `docs/ROADMAP.md`):
- Goal is **not** to beat published AML-detection accuracy. It's to test whether published numbers depend on
  unrealistic look-ahead (decision horizon) that real-time detectors wouldn't have.
- Current phase across the whole fleet: **"Gate G1" / baseline reproduction** — running the official Multi-GNN,
  FraudGT and TD-GNN code as published, to confirm they actually work and reproduce published-style numbers,
  before the real horizon-sweep experiments (replay jobs) can mean anything.
- `docs/PHASE3_PLAN.md` explicitly flags this as the single most important open item: *"The biggest open question:
  the official Multi-GNN has not been run. Everything said about published models is an inference from their code
  until it has."* So this GPU work is not optional busywork — it's the empirical core of the paper's main claim.
- Deadline: 15 November 2026 progress milestone (not a finished paper) — professor explicitly said a presentable
  partial milestone is fine, full runs can continue after.

---

## 2. Notebook numbering used by the user (fleet-wide, their own labels)

The user tracks progress across PCs with simple numbers 01-10. Confirmed mappings so far:
- **01 = NB1** (official Multi-GIN+EU on HI-Small)
- **04 = NB2** (Multi-GNN PNA + FraudGT, HI-Small/LI-Small/LI-Medium/HI-Medium)
- 02, 03, 05-10: not confirmed in detail on this PC; per the user's own fleet status report (2026-11-06 morning):
  - 02: done on PC9
  - 03: shard 2/3 done on PC8, other shard not done
  - 05, 06, 07: not done anywhere yet
  - 08, 09, 10: done on PC18

## 3. What has actually run on THIS PC (PC-17)

### 01 / NB1 (Multi-GIN+EU, HI-Small) — attempted, abandoned (superseded)
- Ran format-check (passed) and started GIN training seed 1.
- Stopped by the user at ~11:30 (2026-11-05) after epoch 1 of 4, once an equivalent run completed on another
  machine. Logged as **D-021** in `docs/DECISIONS.md`.
- Resume-correlation gate failure that originally motivated re-running this was **never resolved** on this
  machine's NB1 run specifically (see batch_size finding below for a possible but unconfirmed explanation).
- State left on disk: `C:\Users\i232515\aml_prod_s1` — only a passed format-check + ~1 epoch partial GIN
  checkpoint (1.6 MB). Low value; flagged to the user as not worth backing up (superseded by the other PC's run).

### 04 / NB2 (PNA + FraudGT) — pilot fully diagnosed and fixed; production in progress

**Pilot diagnosis (2026-11-05), all bugs found and fixed — logged as D-022 in DECISIONS.md:**
1. **PNA VRAM overflow.** At the frozen official `batch_size=8192`, PNA needed ~13.7 GB (card is a 10 GB
   RTX 3080) → silent Windows WDDM shared-memory spill (invisible to `nvidia-smi`, visible only via
   `Get-Counter '\GPU Adapter Memory(*)\...'`), causing ~61s/batch instead of normal speed. Swept batch sizes:
   8192→13.70GB, 4096→8.45GB, 2048→5.03GB (still spilling), 1024→2.86GB (partial spill), 512→clean fit, no spill.
   **Fix:** added `--batch-size` CLI flag to `gpu_session/run_gpu.py`; PNA jobs run at `batch_size=512`.
2. **FraudGT missing dependencies**, found one at a time on first-ever local run: `yacs`, `ogb`, `gdown`,
   `matplotlib`, `torchmetrics` (the last only found via a deep reproduction of the real subprocess invocation).
   All pip-installed into `.venv-gpu`.
3. **FraudGT's `AddEgoIdsForLinkNeighbor`** — same PyG 2.7.0 `BaseTransform.forward`-abstract incompatibility as
   Multi-GNN's own `AddEgoIds` (D-020). Fixed in `src/amlh/fraudgt_driver.py`: `__abstractmethods__ = frozenset()`.
4. **FraudGT Windows DataLoader crash** — official default `num_workers=2` crashed on Windows multiprocessing
   (`DataLoader worker exited unexpectedly`). Fixed in `src/amlh/fraudgt.py`'s `overrides()`: `num_workers=0` on
   Windows only, keeps official default of 2 elsewhere.
5. A runner retry-cap caching quirk (`src/amlh/runner.py` `run_job()`) was discovered along the way: once a job
   hits its retry cap while FAILED, re-running the pilot returns the cached FAILED status *without re-invoking the
   job*. Caused a lot of confusing "still failing" reports for already-fixed bugs. Had to delete the stale job
   folder under `aml_pilot/jobs/<id>/` to force a genuine retry. Good to know if a fix "doesn't seem to work."

Result: a from-scratch pilot run completed clean, every job (format-check, PNA training, replay, FraudGT) passed,
and the resume-correlation gate passed for the first time ever on this project (0.951/0.962 correlation, real
forced kill at exit 137). Evidence saved to `C:\Users\i232515\aml_out_nb2_pilot_bs512` and backed up to
**`G:\PC-17\aml_out_nb2_pilot_bs512`** (USB, 59.5 MB, 62 files) on 2026-11-06.

**Production run (started 2026-11-06):**
- Scoped to **P0 + P1 tiers only** on this PC — **P2 (the single `nb2-mgnn-gin-HI-Medium-s1` job, ~201.8h
  projected) cannot run here**: `HI-Medium_Trans.csv` is not present in `data/raw/` on this machine (only
  `HI-Small_Trans.csv` and `LI-Small_Trans.csv` are). Needs to come from whichever PC has it (PC18 is a
  candidate, per fleet status) or be run on the 20GB Ada machine once that file is copied there.
- Decided (with the user) to parallelize the 7 real training jobs (PNA/GIN main runs; replays are cheap,
  ~10s-few min, not a bottleneck) across every available PC instead of running everything serially on one
  machine, **and** cap `n_epochs` per machine (a logged deviation, same category as the batch-size one) to fit
  roughly a 24h window, reasoned to be low-cost because the training code already tracks and scores from the
  **best validation epoch's checkpoint**, not the last one (`mgnn.py` lines ~317-329) — so stopping early mostly
  only costs you something if the model was still improving.
- This PC's assignment: `nb2-mgnn-pna-HI-Small-s1` (n_epochs=40 cap) and `nb2-mgnn-gin-LI-Small-s1`
  (n_epochs=30 cap originally; see bug below), both at `batch_size=512`, work folder
  `C:\Users\i232515\aml_out_nb2_prod`.
- **Environment quirk discovered:** on this machine, launching `.venv-gpu\Scripts\python.exe` always shows up as
  a parent→child chain (`.venv-gpu\python.exe` → `C:\Python311\python.exe` → `.venv-gpu\python.exe -c isolate...`
  → `C:\Python311\python.exe -c isolate...`), **not two competing duplicate runs**. Mistook this for a duplicate
  launch bug once, killed what looked like "the duplicate," which was actually a legitimate child — broke the
  `pna-HI-Small-s1` job's attempt mid-run (no permanent harm: checkpointing is atomic and per-epoch, so it just
  needs to resume). **Lesson: always check `ParentProcessId` lineage before killing anything that looks like a
  duplicate `run_gpu.py` process on this PC.**

**Real bug found and being fixed right now: GIN/LI-Small training collapse.**
- At `batch_size=512` (applied blindly to *every* mgnn job via the CLI override, not just PNA), `gin-LI-Small-s1`
  showed `val_f1 = 0.0` for 10 straight epochs (0 through 9) with flat/non-decreasing loss — classic model
  collapse to always predicting the majority (non-fraud) class.
- **Root cause:** LI-Small's fraud rate is only **0.0515%** (3,565 / 6,924,049 edges) — the official loader
  (`Multi-GNN/train_util.py`) does plain random shuffled batching, no stratified positive sampling. At
  `batch_size=512`, expected fraud edges per batch ≈ 0.26 — most batches never see a single positive example.
  HI-Small's fraud rate is 0.102% (5,177 / 5,078,345) — only ~2x higher, so the batch-size cut hurt LI-Small much
  worse even though it isn't an order of magnitude sparser.
- PNA/HI-Small by contrast showed real learning at the same `batch_size=512` (val_f1 0.25 → 0.52 across its first
  two epochs in the pilot) — so the blanket batch-size override was the mistake, not a general batch_size=512
  problem.
- Confirmed official `batch_size=8192` **also overflows VRAM for GIN/LI-Small** on this 10GB card (10.5GB
  dedicated + 4.3GB shared spill measured).
- `batch_size=4096` fits VRAM cleanly (8.26GB dedicated, ~0.08GB shared — negligible spill) but **still showed
  val_f1=0.0 for epochs 0 and 1** in a 2-epoch test, even though loss was decreasing in a healthier way
  (0.0174 → 0.0128, vs. flat ~0.019 at batch_size=512). Not yet conclusive whether this is still partly a
  density problem or whether GIN/LI-Small genuinely just needs more epochs before any positive signal appears
  (LI = "Low Illicit" is, by the AMLworld dataset design, meant to be the *harder* detection scenario, not just a
  rarer one).
- **Currently running (as of this file's last update):** an 8-epoch extended test at `batch_size=4096`,
  work folder `C:\Users\i232515\aml_out_nb2_ginli_test3`, to see whether `val_f1` ever turns nonzero with more
  training time before concluding anything further. Epoch 0 of this run logged at `val_f1=0.0` (took unusually
  long, ~1793s, vs ~850s in the earlier 2-epoch test — not yet explained, possibly background load).
  Task id: `b48djd2vq`.

---

## 4. Currently running / outstanding on this PC (live state — check actual job status files for ground truth)

- **`nb2-mgnn-gin-LI-Small-s1` diagnostic run** (batch_size=4096, n_epochs=8) — `C:\Users\i232515\aml_out_nb2_ginli_test3`
  — in progress, investigating the val_f1=0.0 collapse described above.
- **`nb2-mgnn-pna-HI-Small-s1`** (batch_size=512, n_epochs=40 cap) — `C:\Users\i232515\aml_out_nb2_prod` — **not
  currently running**, was interrupted mid-attempt-2 by an accidental process kill (see lesson above). Has a
  resumable checkpoint from attempt 1/2; needs to be resumed once the GPU is free again (not in parallel with the
  GIN diagnostic — one job on this 10GB card at a time).
- Once the GIN/LI-Small root cause is actually fixed (right batch size / right epoch count), the production
  command for it still needs to be re-run for real (the earlier `aml_out_nb2_prod` attempt is the one that
  collapsed and should NOT be trusted/reused without the fix).

To check live status at any time:
```powershell
Get-Content "C:\Users\i232515\aml_out_nb2_prod\aml_run\jobs\<job-id>\status.json"
Get-Content "C:\Users\i232515\aml_out_nb2_prod\aml_run\jobs\<job-id>\log.jsonl" -Tail 10
Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -match 'run_gpu|isolate' } | Select ProcessId,ParentProcessId,CommandLine
```

---

## 5. Data present on this PC

`data/raw/`: `HI-Small_Trans.csv` (454 MB), `LI-Small_Trans.csv` (620 MB), plus all four `*_Patterns.txt` files.
**`HI-Medium_Trans.csv` and `LI-Medium_Trans.csv` are NOT present** — this PC cannot run P2 (`HI-Medium` GIN job)
or any LI-Medium/HI-Medium work until one of those is copied over from a PC that has it.

---

## 6. Files backed up off this PC

- `G:\PC-17\aml_out_nb2_pilot_bs512` (USB drive) — the clean-passing NB2 pilot evidence, copied 2026-11-06.

---

## 7. Key files changed on this PC this session

- `gpu_session/run_gpu.py` — added `--batch-size` CLI override (deviation flag, mirrors `--n-epochs`), applied to
  both production job list and pilot-mode job specs.
- `src/amlh/fraudgt.py` — `overrides()` now defaults `num_workers=0` on Windows (`sys.platform == "win32"`), 2
  elsewhere.
- `src/amlh/fraudgt_driver.py` — added `AddEgoIdsForLinkNeighbor.__abstractmethods__ = frozenset()` right after
  import.
- `docs/DECISIONS.md` — D-021 (NB1 scope cut) and D-022 (PNA memory fix + FraudGT dependency/bug fixes, batch-size
  sweep measurements, retry-cap caching note) added.

---

## 8. Open questions / things to revisit

- Why did `batch_size=512` happen to make NB1's/NB2's resume-correlation gate pass for the first time ever on
  this project? Hypothesis only (smaller batch → less GPU non-determinism?), never confirmed with a controlled
  experiment.
- GIN/LI-Small collapse — root cause understood (class imbalance + batch composition), fix not yet confirmed
  (waiting on the 8-epoch test at batch_size=4096).
- Whether P2 (HI-Medium) should run on this PC at all (needs the raw CSV copied in) or stay on whichever PC
  already has it / the 20GB Ada machine.
- Massive compute-plan overrun vs. `ROADMAP.md`'s own stale 26-51 GPU-hour estimate: real NB2 P0/P1/P2 projections
  (before epoch-capping) were ~151.9h / ~107.6h / ~201.8h respectively. Epoch-capping to fit ~24h windows per
  machine is the agreed mitigation, not yet fully re-measured with real per-machine speeds.
