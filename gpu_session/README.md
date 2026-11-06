# Running the GPU work on a Windows machine with an RTX 3080 (or on Colab)

The scripts here read the expected code hash from the notebooks, so they never go stale after a rebuild. The current hash is `39a6ee5169e533f6`; the package to upload is `dist/aml-code-39a6ee5169e533f6.zip`.

**`dist/aml-code-9686ce184d525c0a.zip` is kept on purpose.** Every finished result (shards 02, 03, 06, 08, 09, 10) and PC-18's unfinished shard 05 were produced under hash `9686ce184d525c0a`, and the runner blocks resuming a job under a different code hash. Finish any in-flight `9686` job with that package. Do not upload both packages to Colab: it needs exactly one.

What differs between `9686…` and `39a6…` (DECISIONS.md D-024): the format-check gate tolerates float-printing noise in the two amount columns, FraudGT gets Windows-safe defaults, two idempotent fixes for the PyG 2.7 `AddEgoIds` fault live in `src/`, and TF32 exists but is **off unless `AMLH_TF32=1`**. Numerics of Multi-GNN and TD-GNN training are otherwise identical.

## On the RTX 3080 machine: four steps

You need Python 3.10 to 3.12 (3.11 is best), git, and the NVIDIA driver (already there if `nvidia-smi` works).

1. **Copy** the repository folder to the machine. Include `third_party/` and `data/raw/HI-Small_Trans.csv`, `data/raw/LI-Small_Trans.csv`.
   Leave out `.venv*`, `data/processed`, the Medium CSVs and `.git` if you like.
2. **Set up** (about 10 to 20 minutes, needs internet for pip, not for git):
   ```
   powershell -ExecutionPolicy Bypass -File gpu_session\setup_windows.ps1
   ```
   It builds `.venv-gpu`, installs the CUDA build of torch (the default Windows pip build has no GPU support), the PyG wheels, and the official Multi-GNN and FraudGT at their pinned commits, then runs `check_env.py`. Fix any FAIL line it prints.
3. **Run everything**, stopping cleanly at the time you give (24 h clock):
   ```
   .venv-gpu\Scripts\python gpu_session\run_all.py --until 14:30
   ```
   Two or three GPUs in one machine: `--gpus 0,1,2`. To split by machine instead, give each machine different shards: `--only 2,5` and so on.
4. **Collect** the `aml_out` folder (next to the repository) before your access ends.

Run the same command again to resume. Finished jobs are skipped and interrupted ones continue from their checkpoints.

### What `run_all.py` does
Pre-flight check, then prepares the data once, then runs one short pilot per kind (NB1, NB2, NB3). **A kind whose pilot fails is not launched**; the other kinds are. Then it runs all 10 shards as GPU capacity allows, longest first.
Logs are in `aml_out\<shard>.production.log`; results in `aml_out\<shard>\aml_run`; a failed pilot's reason is in `aml_out\<shard>\aml_pilot\PILOT_REPORT.md`.

### Compatibility layer
`compat/sitecustomize.py` lets official `AddEgoIds` be created under torch-geometric 2.7.0, which marks `BaseTransform.forward` abstract. `run_gpu.py` applies it to job processes. The official transform runs unchanged. Since code hash `39a6ee5169e533f6` the same fault is also handled inside `src/amlh/mgnn.py`, so Colab and Kaggle notebooks no longer need this layer for `AddEgoIds`. The layer is still what provides the Windows GPU-memory limit and the `np.load` fix for replay jobs. `fix_pyg.py` (PC-18) is superseded by this layer: do not run both, because Python imports only one `sitecustomize`.

It also limits each job to 95% of GPU memory. Without a limit, PyTorch reserved 16.5 GB for 8.3 GB of real use and Windows silently paged the excess to RAM, making training about 5 times slower. **The limit assumes one job per GPU:** two jobs sharing a card would each be allowed 95%. To share one, set `AMLH_GPU_MEMORY_FRACTION` lower, but Multi-GIN on HI-Small needs more than 85% of a 10 GB card, so it does not fit alongside another job there. Use `--capacity 2` for one Multi-GNN shard per GPU.

### If a GPU runs out of memory
Lower `--capacity` to 2, so a Multi-GNN shard runs with nothing beside it.

## On Colab
Run `python gpu_session/make_colab10.py` to regenerate `gpu_session/colab10/*.ipynb`. Start notebooks 01, 04 and 08 first (they run the pilots) and the others when those pass. Details are in the notebooks' first cell.

## Recording deviations
`--n-epochs` and `--batch-size` (and `run_all.py --mgnn-batch-size`) are deviations from the frozen plan. Use a separate `--work` folder for them and record each in `docs/DECISIONS.md`. Job ids do not encode batch size or epoch count; only `config.json` and the DONE `config_hash` do, so never merge two runs of the same job id without comparing those.

## Result files and line endings
Each job's `DONE` marker stores sha256 hashes of its files. `.gitattributes` marks `results/**` as `-text` so git never converts line endings there. Copy output folders byte for byte (do not re-save or re-zip text files through an editor), and verify with `verify_job` or `gpu_session/consolidation/verify_results.py` after any copy.
