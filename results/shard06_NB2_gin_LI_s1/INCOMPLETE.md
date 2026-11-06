# INCOMPLETE and a DEVIATION: shard 06 (NB2 Multi-GIN LI-Small seed 1 plus replays), produced on PC-9

**Do not read `epoch: 100` in `metrics.json`, `isolated_result.json` or `summary.md` as an epoch count.** The job trained about 84 of the planned 100
epochs (best validation F1 0.2663 at epoch 68, no improvement afterwards). The maintainer then edited the epoch counter inside `ckpt/latest.bin`
so the job would finish; this leaves no trace in the ledger and is recorded only in `memory_pc9.md` section 4.4 and the commit message.
It is logged in `docs/DECISIONS.md` D-024. The stopping decision used LI-Small validation F1 only (LI-Small is sealed, so no test metric exists here); label this run
'early-stopped at about 84 of 100 epochs, validation-informed, single seed' and never average it with full-protocol seeds.
The edited checkpoint is `ckpt/latest.bin`; model selection uses `ckpt/best.bin` (epoch 68), which predates the edit. The replays at 3600 s and 86400 s load `best.bin` by the code in `mgnn_replay` (a reading of the code, not re-verified from the missing logs), so they inherit this truncated training run.

## Why this folder is incomplete

The PC-9 branch (`pc9`) contains only the small files of each job. The per-edge score files (`scores_*.parquet`), per-epoch logs
(`log.jsonl`), replay state and leftover replay bucket files were never `git add`-ed (nothing in `.gitignore` excludes them), and the raw
output folder stayed on PC-9. Each job's `DONE` marker lists every file with its sha256, so the gaps are exact:

| job | files listed in DONE | present here | missing | what is missing |
|---|---|---|---|---|
| `nb2-mgnn-gin-LI-Small-s1` | 10 | 5 | 5 | 1 x log.jsonl, 1 x scores_test.parquet, 1 x scores_test_eval.parquet, 1 x scores_val.parquet, 1 x scores_val_eval.parquet |
| `nb2-mgnn-gin-LI-Small-s1-replay3600` | 7 | 3 | 4 | 1 x log.jsonl, 1 x scores_test.parquet, 1 x scores_val.parquet, 1 x state.json |
| `nb2-mgnn-gin-LI-Small-s1-replay86400` | 7 | 3 | 4 | 1 x log.jsonl, 1 x scores_test.parquet, 1 x scores_val.parquet, 1 x state.json |

Every file that **is** present here matches its recorded sha256 byte for byte (checked on the git blobs; `-text` in `.gitattributes` keeps it that way).
`verify_job` therefore reports these jobs as incomplete, not corrupted.

## To complete it

Copy the original output folder off PC-9 in full (it is believed to be `E:\aml_out\06_NB2_gin_LI_s1\aml_run`; this path is inferred from `memory_pc9.md` and
the run conventions, not verified), then run `python gpu_session/consolidation/import_results.py` after adding it to `PLAN`, and
`python gpu_session/consolidation/verify_results.py`. Copy the whole folder, because the replay `DONE` markers also list leftover bucket files.
Without the scores, tail, horizon and stoppable-value metrics cannot be recomputed for this shard.
