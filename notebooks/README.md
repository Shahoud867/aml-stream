# Running the notebooks

There are five notebooks and one merge notebook. Each runs on **Kaggle or Google Colab** and detects which. Each can run in a separate account, and none reads another's output while it runs.

| Notebook | Job | Accelerator |
|---|---|---|
| `nb1.ipynb` | Official Multi-GIN+EU on HI-Small | GPU (Kaggle T4 x2, Colab T4) |
| `nb2.ipynb` | Multi-PNA+EU, LI-Small, FraudGT | GPU |
| `nb3.ipynb` | TD-GNN across horizons | GPU |
| `nb4.ipynb` | Boosted trees on unseen datasets. **Run this one first** | None |
| `nb5.ipynb` | Timing attacks, logical-clock detector, GFP | None |
| `merge.ipynb` | Verifies everything and computes the verdicts | None |

## 1a. Set up once on Colab

1. In Google Drive, make a folder named `aml` and put `dist/aml-code-<hash>.zip` in it.
2. Open Colab, then **File, Upload notebook**, and upload one `.ipynb`.
3. **Runtime, Change runtime type:** T4 GPU for NB1 to NB3, and None (CPU) for NB4 and NB5.
4. Run all cells. The first cell asks for permission to mount Drive.

What the notebook then does on Colab:
- It downloads only the raw files it needs, from the public Kaggle download address, and checks each file is whole. If that download is blocked, put the raw files in Drive `aml/data` and it copies them from there.
- It keeps its output in Drive `aml/out_nbX`. If the session disconnects, use **Runtime, Run all** again. Finished jobs are not repeated, and interrupted ones continue from their checkpoints.

**Limits to know about on Colab:**
- A free runtime has about 12 GB of RAM. Preparing a Medium file needed 12.1 GB here, and running out of memory kills the whole session. So on Colab **NB4 and NB5 run only their P0 jobs by default**, and the pilot skips the Medium memory check. Add `'P1'` to `TIERS` only on a runtime with 32 GB or more.
- The default time budget on Colab is 5 hours, because sessions are shorter and less predictable than Kaggle's. Change `BUDGET_HOURS` once you see how long yours last.
- Keep the tab open. An idle or closed tab ends the session.
- Free GPU time is not published and varies by day and account. The GPU notebooks may be refused a GPU at busy times.
- Each notebook downloads the raw files again at the start of a new session. The preparation of Multi-GNN's data is also repeated each session, because it lives on the local disk.

## 1b. Set up once on Kaggle

1. Upload `dist/aml-code-<hash>.zip` as a **private** Kaggle dataset named `aml-code`, and share it with everyone who will run a notebook.
2. In Kaggle, choose **Create, Import Notebook** and upload each `.ipynb`.
3. In each notebook, attach the code package and the inputs listed at the top of that notebook. Internet must be on.

Kaggle needs a phone-verified account for GPU and internet access.

Every notebook stops at once if its code package is a different version. If you see that message, rebuild the notebooks and the package together.

## 2. Pilot every notebook first

Leave `MODE = 'pilot'` and run all cells. It takes up to 2 hours and uses HI-Small only. Sealed datasets are only loaded, to check that they fit in memory.

Read `aml_pilot/PILOT_REPORT.md` in the output. It shows:

- whether each job finished;
- whether a job killed on purpose resumed and agreed with an uninterrupted run;
- how long each job took, and a rough projection for the full notebook;
- which checks were skipped, and why.

**For the GPU notebooks, also note how much GPU quota the pilot used.** Nobody has yet confirmed whether Kaggle's T4 x2 counts double. If it does not, setting `GPUS = 2` runs two jobs at once.

**Do not start production until the pilot says PASS.** The official-model code has never been run, so the first pilot of NB1 and NB2 may well fail. A failed pilot costs far less than a failed production run.

## 3. Run production

1. Run **NB4 first**. Its first jobs are the leakage check. If a shuffled-label run scores above prevalence, stop everything and investigate: it would invalidate the positive-horizon results.
2. Set `MODE = 'production'` and run all cells. The other notebooks can start at the same time.
3. Each session stops cleanly at its time budget. If the last cell says jobs remain, run again. On Kaggle, first attach this notebook's output as an input (**Add Input, Notebook Output**). On Colab, just run all cells again.
4. Repeat until the last cell passes.

To share a heavy notebook across two machines, set `SHARD = (0, 2)` on one and `SHARD = (1, 2)` on the other. Jobs that depend on each other stay together. On Colab each shard writes to its own folder, `out_nbX_shard0of2` and so on.

## 4. Publish and merge

**Kaggle:** publish each notebook's final `aml_run` folder as a private dataset named `aml-out-nbX-v1` (with `GPUS = 2`, both `gpu0/aml_run` and `gpu1/aml_run`). Attach all of them, the raw datasets and the code package to `merge.ipynb`.

**Colab:** copy every notebook's `out_nbX` folder into one Drive folder, `aml/collect`, in the account that runs the merge. Sharing a folder and adding it as a Drive shortcut may also work, but that has not been tested, so copy if it does not.

Run the merge on CPU. On a free Colab runtime it merges only HI-Small and LI-Small by default, because the Medium datasets need more RAM. Read `merged/MERGE_REPORT.md`. It lists what is complete, partial, failed or missing, and gives each hypothesis a verdict.

**Do not look at sealed test metrics any other way, and do not rerun with changed settings and report the new numbers as confirmation.** The merge step logs each unsealing.

## If something goes wrong

| Message | What to do |
|---|---|
| `this notebook expects code ...` | The notebook and the package are different versions. Rebuild both |
| `attach or upload exactly one code package` | Kaggle: attach one `aml-code`. Colab: keep exactly one `aml-code-*.zip` in Drive `aml` |
| `attach these files as notebook inputs` | Kaggle: attach the dataset it names. Colab: the download failed, so put the file in Drive `aml/data` |
| `needs a GPU` | Kaggle: set Accelerator to GPU T4 x2. Colab: Runtime, Change runtime type, T4 GPU |
| `expected N rows` or `SHA-256 ... differs` | The input file is not the verified one. Delete it and download again. Do not continue |
| `still incomplete after 8 attempts` | The download keeps being cut. Download the file by hand and put it in Drive `aml/data` |
| `BLOCKED` in the summary | The code or a job's settings changed after that job started. Restore them, or start that job again on purpose and record why |
| `FAILED` in the summary | The job failed three times and the queue moved on. The reason is in `jobs/<id>/status.json` |
| The Colab session restarts with no message | It probably ran out of memory. Use P0 only and no Medium datasets on a free runtime |
| `no matching torch-scatter` | The platform's torch has no matching PyG wheels, and the torch 2.5.1 fallback also failed. Report the printed versions |

## Rules

- Never put a token in a notebook. On Kaggle use Kaggle Secrets. Colab needs none.
- Do not edit the code or job settings in a notebook. Change them in the repository, then rebuild.
- Rebuild with `python scripts/build_notebooks.py`, then `python scripts/make_code_dataset.py`. Upload the new package and the new notebooks together.
