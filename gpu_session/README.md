# Running the GPU work on a Windows machine with an RTX 3080 (or on Colab)

Nothing here changes the frozen code: the code hash is still `9686ce184d525c0a`.

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

### If a GPU runs out of memory
Lower `--capacity` to 2, so a Multi-GNN shard runs with nothing beside it.

## On Colab
Run `python gpu_session/make_colab10.py` to regenerate `gpu_session/colab10/*.ipynb`. Start notebooks 01, 04 and 08 first (they run the pilots) and the others when those pass. Details are in the notebooks' first cell.
