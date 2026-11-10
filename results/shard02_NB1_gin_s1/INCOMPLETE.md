# INCOMPLETE: shard 02 (NB1 Multi-GIN HI-Small seed 1 plus replays), produced on PC-9

Code hash `9686ce184d525c0a`, 100 epochs, batch 8192, neighbours [100, 100], no config overrides. The committed `metrics.json` reports
`best_val_f1 0.6099` at epoch 47, `official_style_test_f1 0.6257` (eval mode 0.6426), against the published 64.79%.
The 100-epoch count cannot be re-derived from this folder because `log.jsonl` is missing; ledger running time (19.3 h over about nine resumes)
is consistent with it. The `8.44 h` in `summary.md` is only the last session.
The 'sitecustomize' and 'WinError 32' text in `summary.md` and `status.json` is stale: the replay failures were Windows file-handle errors that the
`np.load` shim fixed, and the retries finished (ledger).

## Why this folder is incomplete

The PC-9 branch (`pc9`) contains only the small files of each job. The per-edge score files (`scores_*.parquet`), per-epoch logs
(`log.jsonl`), replay state and leftover replay bucket files were never `git add`-ed (nothing in `.gitignore` excludes them), and the raw
output folder stayed on PC-9. Each job's `DONE` marker lists every file with its sha256, so the gaps are exact:

| job | files listed in DONE | present here | missing | what is missing |
|---|---|---|---|---|
| `nb1-mgnn-gin-HI-Small-s1` | 10 | 5 | 5 | 1 x log.jsonl, 1 x scores_test.parquet, 1 x scores_test_eval.parquet, 1 x scores_val.parquet, 1 x scores_val_eval.parquet |
| `nb1-mgnn-gin-HI-Small-s1-replay21600` | 15 | 3 | 12 | 8 x bucket_scores_*.npz (leftover replay buckets), 1 x log.jsonl, 1 x scores_test.parquet, 1 x scores_val.parquet, 1 x state.json |
| `nb1-mgnn-gin-HI-Small-s1-replay3600` | 55 | 3 | 52 | 48 x bucket_scores_*.npz (leftover replay buckets), 1 x log.jsonl, 1 x scores_test.parquet, 1 x scores_val.parquet, 1 x state.json |
| `nb1-mgnn-gin-HI-Small-s1-replay86400` | 19 | 3 | 16 | 12 x bucket_scores_*.npz (leftover replay buckets), 1 x log.jsonl, 1 x scores_test.parquet, 1 x scores_val.parquet, 1 x state.json |

Every file that **is** present here matches its recorded sha256 byte for byte (checked on the git blobs; `-text` in `.gitattributes` keeps it that way).
`verify_job` therefore reports these jobs as incomplete, not corrupted.

## To complete it

Copy the original output folder off PC-9 in full (it is believed to be `E:\aml_out\02_NB1_gin_s1\aml_run`; this path is inferred from `docs/history/memory_pc9.md` and
the run conventions, not verified), then run `python gpu_session/consolidation/import_results.py` after adding it to `PLAN`, and
`python gpu_session/consolidation/verify_results.py`. Copy the whole folder, because the replay `DONE` markers also list leftover bucket files.
Without the scores, tail, horizon and stoppable-value metrics cannot be recomputed for this shard.
