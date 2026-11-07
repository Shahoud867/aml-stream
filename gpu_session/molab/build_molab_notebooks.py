"""Writes the three Molab notebooks (01_gnn_baselines, 02_confirmatory_sweep, 03_attacks_and_merge.ipynb).

Like scripts/build_notebooks.py, these only orchestrate: the logic lives in amlh, which is tested (303 passing tests
at the time this was written). Unlike the Kaggle/Colab notebooks, a Molab notebook is genuinely self-contained: it
`git clone`s this repository's `main` branch itself, so it carries no hard-coded expected code hash and never goes
stale. Rebuild after any change to this file:

    python gpu_session/molab/build_molab_notebooks.py

Design note (read this before editing): the three notebooks are a division of the five original notebooks (NB1-NB5)
plus the merge step, chosen because Molab notebooks "run independently" (separate sessions, not guaranteed to share
storage) and each needed to be comprehensive on its own:
  01 = NB1 + NB2 + NB3's remaining P1 work (everything GPU-bound: official Multi-GNN, FraudGT, TD-GNN)
  02 = NB4 (everything CPU-bound and *gating*: the leakage check, the LightGBM confirmatory sweep, the alert policy)
  03 = NB5 (timing attacks, logical-clock features) + the merge step (unseal, verdicts, benchmark tables, figures)
Each notebook also runs its own mid-run analysis (reusing amlh.merge's functions directly) so it is useful standalone,
and 03's merge step defaults to also reading the freshly cloned repo's own committed results/, so even a lone run of
03 produces a real benchmark table from whatever is already on `main` (today: PC-8's and PC-18's finished results).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
REPO_URL = "https://github.com/Shahoud867/aml-stream.git"

# ---------------------------------------------------------------------------------------------------------------- #
# Cell builders
# ---------------------------------------------------------------------------------------------------------------- #


def md(src: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)}


def code(src: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": src.splitlines(keepends=True)}


def notebook(cells: list[dict]) -> dict:
    return {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"}}, "nbformat": 4, "nbformat_minor": 5}


# ---------------------------------------------------------------------------------------------------------------- #
# Shared bootstrap: identical first few cells in every notebook (clone, data, install, tests). Edit once, used thrice.
# ---------------------------------------------------------------------------------------------------------------- #

def title_cell(name: str, goal: str, scope_md: str, hours: str, gpu_line: str) -> dict:
    return md(f"""# Molab {name}

**Goal:** {goal}

{scope_md}

**Expected wall time on the stated hardware:** {hours}. {gpu_line}

## What makes this notebook self-contained
Every cell below is idempotent and safe to re-run. On a fresh Molab session it, in order: (1) clones `aml-stream`'s
`main` branch from GitHub (public, no token needed — set `GITHUB_TOKEN` as an environment variable/secret only if the
repository has since been made private); (2) downloads the AMLworld CSVs it needs directly from Kaggle's public
download endpoint (no `kaggle.json` needed, with a hash check against `docs/SPEC.md`'s verified values, because one
download endpoint has silently truncated a file before); (3) installs the exact pinned dependencies; (4) runs the
test suite as a gate — **a failing test stops the notebook before it spends any compute**; (5) reuses whatever
earlier, *verified* output already exists in the cloned repository (so PC-8's and PC-18's already-finished jobs are
never repeated); (6) runs its job list under a wall-clock budget that always stops cleanly before the 12 h session
limit, in a way that is safe to interrupt and resume (`Run all` again after a restart continues from the last
checkpoint, it does not start over); (7) writes a human-readable analysis with tables and plots; (8) zips the output
folder so it survives the session ending.

**Nothing here needs you to upload or attach anything by hand.**""")


def settings_cell(nb_label: str, needs_gpu: bool, budget_hours: float, tiers_expr: str, extra_lines: str = "") -> dict:
    gpu_block = """
# --- GPU ---
import subprocess
_GPU = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True)
HAS_GPU = _GPU.returncode == 0 and bool(_GPU.stdout.strip())
print("GPU:", _GPU.stdout.strip() or "none found")
""" if needs_gpu else ""
    return code(f"""# ============================= Settings: the only cell you would normally change =============================
import asyncio, os, sys, time   # asyncio is used by later cells via asyncio.to_thread(...), so it is imported once, here
NOTEBOOK = '{nb_label}'
REPO_URL = os.environ.get('AML_REPO_URL', '{REPO_URL}')
REPO_BRANCH = os.environ.get('AML_REPO_BRANCH', 'main')   # pin a commit instead by setting this to a SHA
BUDGET_HOURS = float(os.environ.get('AML_BUDGET_HOURS', {budget_hours}))   # stops cleanly well inside Molab's 12 h cap
TIERS = {tiers_expr}
RUN_TESTS = True     # a failing test stops the notebook before it spends any compute
INSTALL = True
{extra_lines}
SESSION_START = time.time()
def hours_left():
    return BUDGET_HOURS - (time.time() - SESSION_START) / 3600
{gpu_block}""")


def paths_cell() -> dict:
    return code("""# ============================= Where things live =============================
# A plain, dependency-free layout that works on any Jupyter kernel (Molab, a local machine, Colab, ...): everything
# lives under the user's home directory, so no platform-specific mount point is assumed.
from pathlib import Path

BASE = Path(os.environ.get('AML_WORKDIR', Path.home() / 'aml_molab'))
BASE.mkdir(parents=True, exist_ok=True)
CODE = BASE / 'aml-stream'               # the cloned repository
INPUT = BASE / 'data'                    # raw AMLworld files
CACHE = BASE / 'cache'                   # prepared (parquet) data, rebuilt once and reused across cells
WORKING = BASE / f'out_{NOTEBOOK.lower()}'
WORKING.mkdir(parents=True, exist_ok=True)
for p in (INPUT, CACHE):
    p.mkdir(parents=True, exist_ok=True)
print('workdir', BASE)
print('  code   ->', CODE)
print('  data   ->', INPUT)
print('  cache  ->', CACHE)
print('  output ->', WORKING)""")


def clone_cell() -> dict:
    return code("""# ============================= Get the code =============================
# Clones (or fast-forwards) the repository. The expected code hash is computed from whatever we actually cloned,
# never hard-coded, so this notebook can never go stale the way a fixed-hash code package would.
#
# git clone/fetch run via asyncio.to_thread: a plain blocking subprocess.run() call here (or anywhere else long-
# running in this notebook) blocks this kernel's own async event loop for its whole duration. Confirmed live,
# pairing this exact notebook: that is enough for something outside this project's code (consistent with an
# unresponsiveness watchdog) to interrupt the cell outright. to_thread moves the blocking wait off the event
# loop's own thread, so the loop stays free to answer whatever is checking on it, the whole time.
import subprocess

def run(cmd, **kw):
    r = subprocess.run(cmd, text=True, capture_output=True, **kw)
    if r.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)} failed:\\n{r.stdout}\\n{r.stderr}")
    return r.stdout

token = os.environ.get('GITHUB_TOKEN') or os.environ.get('GH_TOKEN')
clone_url = REPO_URL.replace('https://', f'https://{token}@') if token else REPO_URL
if not (CODE / '.git').exists():
    print('cloning', REPO_URL, '...')
    await asyncio.to_thread(run, ['git', 'clone', '--branch', REPO_BRANCH, clone_url, str(CODE)])
else:
    print('updating existing clone ...')
    await asyncio.to_thread(run, ['git', '-C', str(CODE), 'fetch', 'origin', REPO_BRANCH])
    run(['git', '-C', str(CODE), 'checkout', REPO_BRANCH])        # local, fast: no network wait worth threading
    run(['git', '-C', str(CODE), 'reset', '--hard', f'origin/{REPO_BRANCH}'])

sys.path.insert(0, str(CODE / 'src'))
from amlh.runner import code_hash
CODE_HASH = code_hash(CODE)
print('HEAD', run(['git', '-C', str(CODE), 'rev-parse', 'HEAD']).strip())
print('code hash', CODE_HASH)""")


def install_cell(needs_gnn_stack: bool) -> dict:
    gnn = """
    # torch + torch-geometric + the official Multi-GNN and FraudGT repos at their pinned commits.
    from amlh.session import setup_environment
    if HAS_GPU:
        info = await asyncio.to_thread(setup_environment, 'NB2', official_root=BASE / 'official')   # 'NB2' installs everything NB1 needs too
        print(info)
    else:
        print("No GPU: skipping the official Multi-GNN/FraudGT/torch install. This notebook cannot do its GPU work")
        print("without a GPU; see the next cell, which stops here rather than pretending to proceed.")
""" if needs_gnn_stack else ""
    return code(f"""# ============================= Install dependencies =============================
# Off the event loop via asyncio.to_thread -- see the note in the "Get the code" cell above.
if INSTALL:
    await asyncio.to_thread(run, [sys.executable, '-m', 'pip', 'install', '-q', '-r', str(CODE / 'requirements-kaggle.txt')])
    print('requirements-kaggle.txt installed')
{gnn}""")


def gpu_gate_cell() -> dict:
    return code("""# ============================= GPU gate =============================
# This notebook trains official Multi-GNN models and TD-GNN, which need a GPU to finish in any reasonable time (the
# project's own measurements: roughly 1 to 1.3 GPU-hours per 100-epoch Multi-GIN seed on a much smaller card than
# Molab's; on CPU the same run would be impractical). Rather than silently running a misleading, barely-started
# fraction of the work, this stops here with a clear message if no GPU is attached — "fail loudly, not silently"
# is this project's own stated rule for exactly this situation (see gpu_session/README.md).
assert HAS_GPU, (
    "This notebook needs a GPU (Molab: attach the RTX PRO 6000 Blackwell and re-run). "
    "Everything up to here (code, data, environment, tests) is done and verified, so re-running after "
    "attaching a GPU will not repeat any of it."
)""")


def data_cell(datasets_expr: str) -> dict:
    return code(f"""# ============================= Raw data =============================
# Downloaded directly from Kaggle's public endpoint (no account needed) and checked against the row counts and
# SHA-256 hashes in docs/SPEC.md -- the same check every notebook in this project runs, because one Kaggle download
# endpoint has silently truncated a file before while still reporting success.
# Off the event loop via asyncio.to_thread -- see the note in the "Get the code" cell above. HI-Medium and
# LI-Medium are several GB each; without this, a download that size is exactly the kind of long blocking call
# that gets this cell interrupted outright (confirmed live: this happened before to_thread was added here).
from amlh.fetch import ensure_raw
from amlh.session import verify_raw

DATASETS = {datasets_expr}
print('fetching', DATASETS)
await asyncio.to_thread(ensure_raw, DATASETS, INPUT)
for d in DATASETS:
    print(d, verify_raw(d, INPUT))""")


def tests_cell() -> dict:
    return code("""# ============================= Tests: the gate =============================
# Off the event loop via asyncio.to_thread -- see the note in the "Get the code" cell above. The full suite
# takes several minutes; run synchronously that is easily long enough to get this cell interrupted outright.
from amlh.session import run_preflight_tests
if RUN_TESTS:
    await asyncio.to_thread(run_preflight_tests, CODE)
    print('all tests passed')""")


def restore_cell(extra_note: str = "") -> dict:
    return code(f"""# ============================= Reuse prior verified work =============================
# The freshly cloned repository's own results/ folder holds every job this project has finished so far, in exactly
# the job-folder layout the runner understands. Pointing it there for free: a job already verified elsewhere is
# never repeated, and a job that was interrupted partway (with a usable checkpoint) resumes instead of restarting.
PRIOR_RESULTS = [CODE / 'results']
print('will reuse any verified job found under', [str(p) for p in PRIOR_RESULTS]){extra_note}""")


def sliced_run_helper_cell() -> dict:
    return code('''# ============================= Run the job queue off the event loop =============================
# amlh.session.run_notebook's own job loop is a plain synchronous `for` loop: gbt_sweep (LightGBM) and the
# polling loop around an isolated mgnn_train/tdgnn_train subprocess both run entirely in-process, with no
# `await` anywhere inside them. A single call that runs for a long time therefore blocks this kernel's own
# async event loop for that whole time. Observed live, pairing this exact notebook with the marimo-pair skill:
# a long call like this gets interrupted outright, confirmed from the runner's own ledger to be `stop_requested`
# becoming true (not this project's own time-budget check: remaining() was still far above what the next step
# needed) -- consistent with something outside this project's code deciding the kernel looked unresponsive and
# signalling it. The same thing happened to the clone, install, data-fetch and test cells above, which is why
# they are wrapped in asyncio.to_thread too.
#
# An earlier version of this fix chopped run_notebook into many short, repeated slices with a sleep in between,
# to force the event loop to get a turn. That introduced its own bug: run_notebook() restores from every prior
# result and re-verifies the raw files on every call, and repeating that every few minutes (instead of once)
# consumed each slice's entire budget before a new job could ever start -- confirmed live, DONE froze for dozens
# of consecutive slices while the budget drained. asyncio.to_thread is the more direct fix for the actual
# problem: it moves run_notebook's blocking work onto its own OS thread, so the event loop stays free for the
# whole call, in one piece, not an artificially chopped-up one -- no slicing, no redundant re-verification.
from amlh.session import run_notebook


async def run_notebook_sliced(notebook, budget_hours, **kw):
    return await asyncio.to_thread(run_notebook, notebook, budget_hours=budget_hours, **kw)''')


def wrapup_cell(label: str) -> dict:
    return code(f"""# ============================= Package the output =============================
# Molab sessions are not guaranteed to persist, so the output is zipped here (and the notebook's own cell outputs,
# saved with the .ipynb itself, carry the printed tables below even if the zip is never downloaded).
import shutil
archive = shutil.make_archive(str(BASE / f'{label}_{{NOTEBOOK.lower()}}'), 'zip', WORKING)
print('wrote', archive, f'({{Path(archive).stat().st_size / 1e6:.1f}} MB)')
print()
print('Download this file before your Molab session ends. If you plan to run another of the three notebooks next,')
print('the most useful thing to carry over is the whole', WORKING, 'folder (or this zip, unpacked next to the new')
print("notebook's own working directory) -- the next notebook's merge step will pick up anything placed under")
print("its PRIOR_RESULTS / EXTRA_ROOTS list.")""")




# ---------------------------------------------------------------------------------------------------------------- #
# Notebook-specific cell lists
# ---------------------------------------------------------------------------------------------------------------- #
def build_notebook_1():
    cells = [
        title_cell(
            "01: GNN Baselines (Official Multi-GNN + TD-GNN) — completing Gate G1",
            "Finish the official-model work the whole project's main claim depends on: does the published Multi-GNN "
            "result (and TD-GNN, our own view-restricted model) actually depend on look-ahead and on the benchmark's "
            "post-simulation tail? `docs/PHASE3_PLAN.md` calls this \"the biggest open question\": the official "
            "Multi-GNN had never been run at all before the GPU campaign this notebook completes.",
            "**Covers:** NB1 (official Multi-GIN+EU, HI-Small, 3 seeds + replays), NB2 (official Multi-PNA+EU HI-Small "
            "2 seeds, Multi-GIN+EU LI-Small 2 seeds + replays, FraudGT audit), and NB3's remaining P1 work (the full "
            "horizon grid, 3 ablations, LI-Small attacks, GPU service time — NB3's P0 is already 100% complete "
            "and verified, so none of it is repeated here).\n\n"
            "**A real change from the plan's own assumptions, worth noting up front:** every GPU run so far used a "
            "10 GB RTX 3080, where the official Multi-PNA setting (`batch_size=8192`) does not fit and needed a "
            "documented deviation (`docs/DECISIONS.md` D-022/D-023, batch 512 or 1024). Molab's 96 GB card has "
            "**13x** that memory, so this notebook runs Multi-PNA at the frozen **official batch size, unmodified** "
            "— the deviation this project has been carrying simply does not apply here.",
            "3 to 10 hours, depending heavily on how much of NB1's 3 full 100-epoch seeds remain after reuse (one is "
            "already finished and verified in `main`'s `results/`, so it is skipped automatically)",
            "**Requires a GPU.** The next cell stops cleanly with a clear message if none is attached, after "
            "finishing setup so nothing is repeated once you attach one and re-run.",
        ),
        settings_cell("NB1", True, 11.5, "('P0', 'P1')"),
        paths_cell(),
        clone_cell(),
        install_cell(needs_gnn_stack=True),
        gpu_gate_cell(),
        data_cell("['HI-Small', 'LI-Small']"),
        tests_cell(),
        restore_cell(),
        sliced_run_helper_cell(),
        md("""## Run NB1, NB2, and NB3's remaining work

Each of the three keeps using `amlh.session.run_notebook`, the exact tested orchestrator the project's Kaggle/Colab
notebooks use: per-epoch checkpointing, resumable replays, a wall-clock budget that stops cleanly, and jobs that
fail loudly (not silently) on a real error. The budget below is split across the three, biased toward NB1 since it
carries the heaviest remaining work (up to 3 full 100-epoch seeds); each stops early and hands its leftover time to
the next if it finishes sooner than its share."""),
        code('''from amlh.session import print_summary, final_checklist

# A fixed split of this session's remaining budget. NB1 (Multi-GIN, up to 3 full seeds) gets the largest share;
# NB2 (Multi-PNA + Multi-GIN LI-Small + FraudGT) the next; NB3 (only P1 remains) the smallest, since its P0 is done.
SHARES = {'NB1': 0.45, 'NB2': 0.40, 'NB3': 0.15}
summaries = {}
for nb, share in SHARES.items():
    left = hours_left()
    if left <= 0.1:
        print(f'{nb}: no time left in this session, skipping (re-run this cell after extending BUDGET_HOURS or in a new session)')
        continue
    budget = min(left, max(left * share / sum(s for n, s in SHARES.items() if n not in summaries), 0.25))
    print(f'=== {nb}: {budget:.2f} h budget, {left:.2f} h left in the session ===')
    nb_work = WORKING / nb.lower()
    summaries[nb] = await run_notebook_sliced(nb, budget_hours=budget, tiers=TIERS, input_root=INPUT, working=nb_work,
                                               restore_roots=PRIOR_RESULTS, cache_root=CACHE, run_name='aml_run',
                                               official_root=BASE / 'official')
    print_summary(nb_work)
    final_checklist(nb, TIERS, nb_work, platform='molab')
'''),
        md("""## Gate G1: does the official model show the look-ahead / tail effect?

Reuses `amlh.merge`'s own functions directly — the same code the final merge step (notebook 03) and
`scripts/08_analyze.py` use — so this table is computed the identical, tested way, not re-derived by hand."""),
        code('''import json
import numpy as np
import matplotlib.pyplot as plt
from amlh.merge import collect, official_audit, leakage_check
from amlh.jobs_gbt import get_stream, Paths

Paths.raw = {d: INPUT for d in DATASETS}
Paths.cache = CACHE

roots = [WORKING] + PRIOR_RESULTS
c = collect(roots)
print(f"{len(c.jobs)} verified jobs found across this session's output and the reused prior work; {len(c.problems)} problems")
for p in c.problems[:10]:
    print(' problem:', p)

rows = []
for ds in ('HI-Small',):   # LI-Small is sealed: this notebook must not compute or display its test metrics
    s, _ = get_stream({'dataset': ds, 'protocol': 'released'})
    audit = official_audit(c, s)
    for jid, a in sorted(audit.items()):
        if 'problem' in a:
            continue
        r = a['row']
        rows.append({'job': jid, 'dataset': ds, 'H': a['H'], 'official_style_f1': a['official_style_f1'],
                     'published_reference': a['published_reference'], 'core_f1': r['test_core']['f1'],
                     'tail_f1': r.get('test_tail', {}).get('f1'), 'core_pr_auc': r['test_core']['pr_auc']})
    leak = leakage_check(c, s)
    for jid, v in leak.items():
        print(f"leakage check {jid}: core PR-AUC {v['core_pr_auc']:.4f} vs prevalence {v['prevalence']:.5f} "
              f"-> {'PASS' if v['pass'] else 'FAIL -- investigate before trusting any positive-horizon result'}")

if rows:
    print(f"{'job':45s} {'H':>8s} {'official F1':>12s} {'published':>10s} {'core F1':>8s} {'tail F1':>8s}")
    for r in rows:
        pub = f"{r['published_reference']:.4f}" if r['published_reference'] is not None else '-'
        tail = f"{r['tail_f1']:.4f}" if r['tail_f1'] is not None else '-'
        print(f"{r['job']:45s} {str(r['H']):>8s} {r['official_style_f1']:>12.4f} {pub:>10s} {r['core_f1']:>8.4f} {tail:>8s}")
    (WORKING / 'gate_g1_audit.json').write_text(json.dumps(rows, indent=1))
    labels = [r['job'][:30] for r in rows]
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(max(6, len(rows) * 0.6), 4))
    ax.bar(x - 0.2, [r['official_style_f1'] for r in rows], 0.4, label='reproduced (official-style F1)')
    ax.bar(x + 0.2, [r['published_reference'] or 0 for r in rows], 0.4, label='published reference')
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=60, ha='right'); ax.set_ylabel('minority-class F1')
    ax.set_title('Gate G1: reproduced vs published, by job and cadence'); ax.legend(); plt.tight_layout()
    plt.savefig(WORKING / 'gate_g1.png', dpi=130); plt.show()
else:
    print('No official-model jobs verified yet in this session or in the reused prior results -- Gate G1 remains open.')
'''),
        md("""## TD-GNN: core PR-AUC by decision horizon (RQ1)

The same computation used earlier in this project's own analysis, now reading whatever this session produced plus
whatever NB3's already-complete P0 output (reused above) contributes — should reproduce it closely on HI-Small,
and extend it with the P1 horizons / ablations this session ran."""),
        code('''import re
from collections import defaultdict
import statistics as st

td_rows = []
for root in roots:
    for jd in Path(root).rglob('nb3-tdgnn-*'):
        if not jd.is_dir() or not (jd / 'metrics.json').exists():
            continue
        m = re.match(r'nb3-tdgnn-(HI-Small|LI-Small)-H(\\w+)-s(\\d+)$', jd.name)
        if not m:
            continue
        ds, h, seed = m.groups()
        met = json.loads((jd / 'metrics.json').read_text())
        td_rows.append((ds, h, int(seed), met))

by = defaultdict(list)
for ds, h, seed, met in td_rows:
    by[(ds, h)].append(met)

HORDER = ['0', '1', '60', '600', '3600', '21600', '86400', '259200', 'inf']
print(f"{'H':>8s} {'n':>2s} {'val PR-AUC':>11s} {'test PR-AUC':>12s} {'core PR-AUC':>12s} {'tail PR-AUC':>12s}")
curve = []
for h in HORDER:
    if ('HI-Small', h) not in by:
        continue
    ms = by[('HI-Small', h)]
    def avg(path):
        vals = [path(m) for m in ms if path(m) is not None]
        return st.mean(vals) if vals else None
    v = avg(lambda m: m.get('val_pr_auc'))
    t = avg(lambda m: m.get('test_results', {}).get('test', {}).get('pr_auc'))
    co = avg(lambda m: m.get('test_results', {}).get('test_core', {}).get('pr_auc'))
    ta = avg(lambda m: m.get('test_results', {}).get('test_tail', {}).get('pr_auc'))
    print(f"{h:>8s} {len(ms):>2d} {v or 0:>11.4f} {t or 0:>12.4f} {co or 0:>12.4f} {ta or 0:>12.4f}")
    curve.append((h, co))
if curve:
    fig, ax = plt.subplots(figsize=(6, 4))
    xs = range(len(curve))
    ax.plot(xs, [c[1] or 0 for c in curve], 'o-')
    ax.set_xticks(xs); ax.set_xticklabels([c[0] for c in curve], rotation=45)
    ax.set_xlabel('decision horizon H'); ax.set_ylabel('core PR-AUC'); ax.set_title('TD-GNN, HI-Small: does waiting help?')
    plt.tight_layout(); plt.savefig(WORKING / 'td_horizon_curve.png', dpi=130); plt.show()
'''),
        wrapup_cell("gnn_baselines"),
        md("""## What to do next

- If you still have session time: increase `BUDGET_HOURS` and `Run all` again. Nothing above repeats finished work.
- Otherwise: download the zip this notebook wrote, or push `results/` back to a branch of this repository, and run
  **Notebook 03** next — point its `EXTRA_ROOTS` at this notebook's output folder (or its unzipped contents) to
  fold these results into the final merge and benchmark report."""),
    ]
    return notebook(cells)

def build_notebook_2():
    cells = [
        title_cell(
            "02: Confirmatory Sweep, Leakage Check & Alert Policy (NB4)",
            "Run the boosted-tree confirmatory sweep that tests the frozen hypotheses C1-C5 on data that did not "
            "inform them, and that is the precondition for `LI-Small` (and every other sealed dataset) to be "
            "unsealed at all. **This notebook has never been run anywhere in this project before this** — the "
            "audit found zero NB4 evidence in any of the four PCs' work.",
            "**Covers:** NB4 in full — the shuffled-label leakage check (runs *first*, and the notebook stops "
            "and warns loudly if it fails, exactly as the project's own rule requires), the 9-horizon x 2-protocol x "
            "3-seed LightGBM sweep on HI-Small and LI-Small (tier P0), then SAML-D / HI-Medium / LI-Medium "
            "(tier P1) with whatever time remains. Tier P2 (rolling-origin folds plus the Ethereum dataset) is "
            "not run by default: P2 also includes the Ethereum sweep, which needs `MulDiGraph.pkl` — not "
            "downloadable by `amlh.fetch.ensure_raw` (see `fetch.py`), so it would break this notebook's "
            "self-contained design. Add `'P2'` to `TIERS` and attach that file by hand if you want those jobs too.\n\n"
            "**Why HI-Small's own numbers here are not a new confirmatory result:** `docs/DECISIONS.md` D-001 is "
            "explicit — \"HI-Small is the discovery set. A claim counts as confirmed only on data not modelled "
            "when the hypotheses were frozen.\" This notebook's HI-Small sweep is a faithful reproduction check "
            "(useful for catching regressions, and it is shown below), not evidence for C1-C5. **LI-Small is the "
            "real target**: it is sealed (`amlh.sealing.SEALED`), so this notebook deliberately never computes or "
            "displays its test metrics — only validation metrics and label-free health checks, exactly as "
            "`gbt_sweep`'s own code already enforces. Completing LI-Small's full sweep here is what lets notebook "
            "03's merge step unseal it and compute this project's **first real confirmatory verdicts**.",
            "CPU-only: LightGBM at the frozen, un-tuned hyperparameters (`amlh.jobs_gbt.LGB_PARAMS`, fixed by "
            "DECISIONS.md D-001) on 4 cores. 2 to 5 hours for the full HI-Small + LI-Small P0 sweep (114 jobs total; "
            "the project's own measurement on a similar-class CPU was 1 to 2 minutes per HI-Small job including "
            "features), leaving most of a 12 h session for the P1 datasets",
            "**No GPU needed or used** — the frozen protocol is CPU LightGBM by design (DECISIONS.md D-001: "
            "switching LightGBM to a GPU build would change a frozen detector). Runs correctly on Molab with the "
            "GPU detached.",
        ),
        settings_cell("NB4", False, 11.5, "('P0', 'P1')"),  # P2 includes rolling-origin folds (fine, reuse HI-Small) but also Ethereum jobs, which need MulDiGraph.pkl -- not fetchable by amlh.fetch.ensure_raw (see fetch.py's sources()), so excluded from this self-contained notebook's default; add it back and attach that file by hand if you want those jobs too
        paths_cell(),
        clone_cell(),
        install_cell(needs_gnn_stack=False),
        data_cell("['HI-Small', 'LI-Small', 'SAML-D', 'HI-Medium', 'LI-Medium']  # every P1 dataset; TIERS below stops at P1, not P2, because Ethereum (P2) needs a file this notebook cannot fetch on its own"),
        tests_cell(),
        restore_cell(),
        sliced_run_helper_cell(),
        md("""## Stage 0: the leakage check — runs before anything else

A detector trained on **shuffled labels** must score at prevalence (it has learned nothing real). If it scores well
above prevalence instead, the feature pipeline is leaking the label through some other column, and **every
positive-horizon result in this whole project would be invalid** until that is fixed. This is deliberately the very
first thing this notebook computes, with no other work started first."""),
        code('''from amlh.runner import Runner
from amlh.plans import plan, registry
from amlh.jobs_gbt import Paths

Paths.raw = {'HI-Small': INPUT, 'LI-Small': INPUT}
Paths.cache = CACHE

leakage_jobs = [j for j in plan('NB4') if 'shuffled' in j.job_id]
print(f'{len(leakage_jobs)} leakage-check jobs:', [j.job_id for j in leakage_jobs])

leak_run = WORKING / 'leakage_check'
r = Runner(leak_run, 'NB4-leakage', 3600, code_version=CODE_HASH, registry=registry())
r.install_signal_handlers()
r.restore(PRIOR_RESULTS)
r.write_manifest({'stage': 'leakage_check'})
summary = r.run(leakage_jobs, tiers=('P0',))
print(summary)
'''),
        code('''import numpy as np
from amlh.merge import collect, leakage_check
from amlh.jobs_gbt import get_stream

c = collect([leak_run] + PRIOR_RESULTS)
failed = []
for ds in ('HI-Small',):   # the frozen leakage-check jobs are defined on HI-Small (plans.py nb4())
    s, _ = get_stream({'dataset': ds, 'protocol': 'released'})
    res = leakage_check(c, s)
    for jid, v in res.items():
        status = 'PASS' if v['pass'] else 'FAIL'
        print(f"{jid}: core PR-AUC {v['core_pr_auc']:.5f} vs {10 * v['prevalence']:.5f} (10x prevalence) -> {status}")
        if not v['pass']:
            failed.append(jid)

assert not failed, (
    f"LEAKAGE CHECK FAILED for {failed}. Per this project's own rule, do not run anything below this cell. "
    "Investigate the feature pipeline (src/amlh/design.py, src/amlh/features.py) before trusting any positive-"
    "horizon result anywhere in this project."
)
print()
print('Leakage check passed. Continuing to the main sweep.')
'''),
        md("""## The main sweep, alert policy, and (time permitting) the Medium/SAML-D/fold tiers

Reuses `amlh.session.run_notebook` exactly as the Kaggle/Colab NB4 does: per-job checkpointing (a LightGBM fit that
is interrupted restarts that one fit, not the whole sweep), a wall-clock budget that stops cleanly, and automatic
tier ordering (P0 before P1 before P2) so the most important deliverable — the full HI-Small + LI-Small P0
sweep and alert policy — is always attempted first."""),
        code('''from amlh.session import print_summary, final_checklist

nb4_work = WORKING / 'nb4'
summary = await run_notebook_sliced('NB4', budget_hours=hours_left(), tiers=TIERS, input_root=INPUT, working=nb4_work,
                                     restore_roots=[leak_run] + PRIOR_RESULTS, cache_root=CACHE, run_name='aml_run')
print_summary(nb4_work)
final_checklist('NB4', TIERS, nb4_work, platform='molab')
'''),
        md("""## Results: hypothesis verdicts

**HI-Small** (exploratory / reproduction check only, per D-001 above). **LI-Small**: validation metrics and
label-free health only — if you see no test numbers for LI-Small below, that is the sealing rule working as
designed, not a bug."""),
        code('''import json
from amlh.merge import collect, sweep_results, verdicts_for
from amlh.jobs_gbt import get_stream

c = collect([nb4_work, leak_run] + PRIOR_RESULTS)
print(f"{len(c.jobs)} verified jobs total; {len(c.problems)} problems")

s, _ = get_stream({'dataset': 'HI-Small', 'protocol': 'released'})
rows = sweep_results(c, s)
if rows:
    v = verdicts_for(c, s, rows)
    print('HI-Small (exploratory reproduction, NOT a confirmatory result -- see markdown above):')
    for hid, val in sorted(v.items()):
        value = val.get('value')
        print(f"  {hid}: {value:.4f} " + ('PASS' if val.get('pass') else 'FAIL') if value is not None else f"  {hid}: {val}")
    (WORKING / 'hi_small_verdicts.json').write_text(json.dumps(v, indent=1, default=str))

    import matplotlib.pyplot as plt
    import numpy as np
    HORDER = ['0', '1', '60', '600', '3600', '21600', '86400', '259200', 'inf']
    seeds = sorted({k for _, k in rows})
    curve = [(h, np.mean([rows[(h, k)]['row']['test_core']['pr_auc'] for k in seeds if (h, k) in rows]))
             for h in HORDER if any((h, k) in rows for k in seeds)]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(range(len(curve)), [c[1] for c in curve], 'o-', color='tab:orange')
    ax.set_xticks(range(len(curve))); ax.set_xticklabels([c[0] for c in curve], rotation=45)
    ax.set_xlabel('decision horizon H'); ax.set_ylabel('core PR-AUC')
    ax.set_title('LightGBM, HI-Small: core PR-AUC vs horizon (reproduction check)')
    plt.tight_layout(); plt.savefig(WORKING / 'hi_small_horizon_curve.png', dpi=130); plt.show()
else:
    print('HI-Small sweep not yet complete in this session.')

li, _ = get_stream({'dataset': 'LI-Small', 'protocol': 'released'})
li_jobs = [jid for jid in c.jobs if '-gbt-wall-LI-Small-released-H' in jid and 'fold' not in jid and 'shuffled' not in jid]
print()
print(f"LI-Small: {len(li_jobs)} of 54 planned P0 sweep jobs verified (needed for notebook 03 to unseal it).")
print('No test metric for LI-Small is computed here -- that only ever happens once, in the merge step.')
'''),
        wrapup_cell("confirmatory_sweep"),
        md("""## What to do next

- If `hours_left()` is still positive and jobs remain (check the summary above), `Run all` again: nothing finished
  is repeated.
- Once LI-Small's P0 sweep (54 jobs) and its alert policy (3 jobs) are complete, run **Notebook 03** — its
  merge step will unseal LI-Small for the first time in this project and report the real C1-C5 verdicts. Point its
  `EXTRA_ROOTS` at this notebook's output (the zip above, unzipped) so it can see this work."""),
    ]
    return notebook(cells)

def build_notebook_3():
    cells = [
        title_cell(
            "03: Timing Attacks, Logical-Clock Defense (NB5) & Final Merge",
            "Test RQ4 — can an adversary who only controls *when* transactions are submitted suppress "
            "detection, and does evidence built on logical-clock order (what re-timing cannot change) resist this "
            "where wall-clock evidence does not? Then run the project's final merge step: unseal every dataset "
            "whose confirmatory sweep is complete, compute every frozen hypothesis's verdict, and produce the "
            "benchmark comparison against the published baseline.",
            "**Covers:** NB5 in full (logical/combined detectors, burst/stretch/blend attacks with matched "
            "displacement controls, 6 feature ablations, GFP baseline where `snapml` installs, CPU service time), "
            "**and** the merge step (`amlh.merge.merge`, the same function `notebooks/merge.ipynb` runs) — "
            "this is the one place sealed test metrics are ever computed, and every unsealing is permanently logged.\n\n"
            "**This notebook is useful standalone, and more complete when chained after 01 and 02.** By default its "
            "merge step reads this session's own NB5 output plus whatever is already committed in `main`'s "
            "`results/` (today: PC-8's finished NB1 seed and all of PC-18's finished NB3 TD-GNN work), so even a "
            "lone run produces a real benchmark table. Set `EXTRA_ROOTS` below to also fold in notebooks 01's and "
            "02's fresh output (copy their unzipped working folders next to this one, or push them to a branch).",
            "1 to 4 hours for NB5's P0 (roughly 150 LightGBM fits/attacks on HI-Small, similar per-job cost to "
            "notebook 02), then minutes for the merge step regardless of how much data feeds it",
            "**No GPU needed or used.**",
        ),
        settings_cell("NB5", False, 11.0, "('P0', 'P1')"),
        paths_cell(),
        clone_cell(),
        install_cell(needs_gnn_stack=False),
        data_cell("['HI-Small', 'LI-Small']"),
        tests_cell(),
        restore_cell(),
        sliced_run_helper_cell(),
        md("""## Extra output to fold into the merge (optional)

If you ran notebooks 01 and/or 02 first and copied or unzipped their output next to this notebook's own working
directory, list those folders here. Leave empty to merge only this session's own NB5 output plus whatever is
already committed in the cloned repository — the notebook still runs correctly either way."""),
        code('''EXTRA_ROOTS = []   # e.g. [BASE / 'out_nb1', BASE / 'out_nb4'] if you copied them in
EXTRA_ROOTS = [Path(p) for p in EXTRA_ROOTS if Path(p).exists()]
print('extra roots found:', EXTRA_ROOTS or 'none (this is fine; the merge below still runs)')'''),
        md("""## Run NB5

Attacks are built on top of a clean-data detector's own fixed thresholds, with a **matched-displacement control**
for every attack condition (`shift_burst`, `shift_stretch`, `shift_blend`) — so a suppression in detection can
be attributed to the *timing pattern itself*, not merely to moving events around at all."""),
        code('''from amlh.session import print_summary, final_checklist

nb5_work = WORKING / 'nb5'
summary = await run_notebook_sliced('NB5', budget_hours=hours_left(), tiers=TIERS, input_root=INPUT, working=nb5_work,
                                     restore_roots=PRIOR_RESULTS, cache_root=CACHE, run_name='aml_run')
print_summary(nb5_work)
final_checklist('NB5', TIERS, nb5_work, platform='molab')
'''),
        md("""## Timing-robustness verdicts (C6, C8) and attack plots

C6: does a burst evade a strict-past detector, and does a same-minute view blunt this? C8: does logical-clock
evidence resist `stretch` and `blend` where wall-clock evidence does not, at an acceptable clean-data cost?"""),
        code('''import json
import numpy as np
import matplotlib.pyplot as plt
from amlh.merge import collect, timing_verdicts, attack_table
from amlh.jobs_gbt import get_stream

c = collect([nb5_work] + PRIOR_RESULTS + EXTRA_ROOTS)
s, _ = get_stream({'dataset': 'HI-Small', 'protocol': 'released'})
tv = timing_verdicts(c, s)
print('Timing-robustness verdicts (HI-Small, exploratory -- see notebook 02 for why HI-Small is not confirmatory):')
for hid, v in sorted(tv.items()):
    print(f"  {hid}: {'PASS' if v.get('pass') else 'FAIL'}  {json.dumps({k: v[k] for k in v if k != 'pass'}, default=str)[:200]}")
(WORKING / 'timing_verdicts.json').write_text(json.dumps(tv, indent=1, default=str))

wall = attack_table(c, s, 'wall', '0', 21600)
if wall:
    conds = sorted(wall)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(conds, [wall[cd]['detected'].mean() for cd in conds])
    ax.set_ylabel('fraction of attempts detected'); ax.set_title('Wall-clock detector under each attack condition, H=0')
    plt.xticks(rotation=30, ha='right'); plt.tight_layout(); plt.savefig(WORKING / 'attack_detection.png', dpi=130); plt.show()
'''),
        md("""## Final merge: unseal, compute every verdict, build the benchmark report

This is the one place in the whole project sealed test metrics are computed, and it happens **at most once per
dataset per code version** — `amlh.sealing.UnsealLog` permanently records every unsealing, and a repeat under
different code is labelled `exploratory`, never silently treated as a fresh confirmation."""),
        code('''from amlh.merge import merge

ALL_ROOTS = [nb5_work] + PRIOR_RESULTS + EXTRA_ROOTS
DATASETS = ['HI-Small', 'LI-Small']   # add 'SAML-D', 'HI-Medium', 'LI-Medium' here once notebook 02 has swept them

MERGE_OUT = WORKING / 'merged'
report = merge(ALL_ROOTS, MERGE_OUT, DATASETS, analysis_root=CODE)
print((MERGE_OUT / 'MERGE_REPORT.md').read_text())
'''),
        md("""## Benchmark comparison against the published baseline

`published_reference` below is the official Multi-GNN paper's own reported minority-class F1 on HI-Small (as cited
in this project's literature review, `paper/cn_phase2.tex`), via `amlh.merge.PUBLISHED_F1`. Read this table with
the project's actual goal in mind: **this project is not trying to out-accuracy the baseline model** —
`docs/PHASE3_PLAN.md` is explicit that reproducing accuracy is not among its claims. Matching the published number
under full look-ahead (large H) is an expected sanity check. The actual contribution is the *gap* as H shrinks
toward 0: a genuine, measured answer to "how much of the published number is unrealistic look-ahead", which the
original paper's own evaluation protocol never reports."""),
        code('''from amlh.merge import official_audit
Paths_report = []
for ds in ('HI-Small',):
    s, _ = get_stream({'dataset': ds, 'protocol': 'released'})
    audit = official_audit(c, s)
    for jid, a in sorted(audit.items()):
        if 'problem' in a or a.get('published_reference') is None:
            continue
        r = a['row']
        gap = a['official_style_f1'] - a['published_reference']
        print(f"{jid:42s} H={str(a['H']):>8s}  reproduced={a['official_style_f1']:.4f}  "
              f"published={a['published_reference']:.4f}  gap={gap:+.4f}")
print()
print('A full-window (H=inf) job close to the published number, alongside an H=0 (strict-past) job well below it,')
print('is this project\\'s central empirical finding: direct evidence that the published number embeds look-ahead.')
'''),
        md("""## Regenerate the paper's tables and figures

`scripts/08_analyze.py` turns the exported results above into exactly the tables and figures `paper/cn_phase2.tex`
includes — nothing in the paper's results is typed by hand."""),
        code('''import subprocess, shutil, os as _os
link = CODE / 'data' / 'processed'
link.parent.mkdir(parents=True, exist_ok=True)
if not link.exists():
    try:
        _os.symlink(CACHE, link)
    except OSError:
        shutil.copytree(CACHE, link, dirs_exist_ok=True)   # symlinks need a privilege Molab may not grant; copy instead
result = subprocess.run([sys.executable, str(CODE / 'scripts' / '08_analyze.py'), str(MERGE_OUT)], capture_output=True, text=True)
print(result.stdout[-2000:])
if result.returncode != 0:
    print('--- stderr ---'); print(result.stderr[-2000:])
if (CODE / 'results' / 'report').exists():
    shutil.copytree(CODE / 'results' / 'report', MERGE_OUT / 'report', dirs_exist_ok=True)
    print('tables and figures copied to', MERGE_OUT / 'report')
'''),
        wrapup_cell("attacks_and_merge"),
        md("""## Completion summary

Read `MERGE_REPORT.md` above (and `merged.json` for the full machine-readable detail) for the authoritative status:
which hypotheses have a verdict, which datasets are unsealed, and the plan-completion counts for every one of the
five original notebooks. Download the zip this notebook wrote — it is this project's most complete single
artifact so far."""),
    ]
    return notebook(cells)


if __name__ == "__main__":
    for fn, name in [(build_notebook_1, "01_gnn_baselines.ipynb"),
                      (build_notebook_2, "02_confirmatory_sweep.ipynb"),
                      (build_notebook_3, "03_attacks_and_merge.ipynb")]:
        nb = fn()
        path = OUT / name
        path.write_text(json.dumps(nb, indent=1), encoding="utf8")
        n_code = sum(1 for c in nb["cells"] if c["cell_type"] == "code")
        n_md = sum(1 for c in nb["cells"] if c["cell_type"] == "markdown")
        print(f"wrote {path} ({n_code} code cells, {n_md} markdown cells)")
