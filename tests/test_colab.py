"""Colab: no attached datasets, so the raw files are downloaded and the output lives in Drive."""
import http.server
import shutil
import sys
import threading
from pathlib import Path

import pytest

import amlh.fetch as fetch
import amlh.session as sess
from amlh.runner import JobSpec, verify_job
from test_notebooks import SWEEP, _common, _package, _small_input, cells, restore_globals, source  # noqa: F401 (restore_globals is an autouse fixture)


class _Server:
    """Serves bytes with Range support. Its first full response per file is silently cut short, as the real endpoint has done."""

    def __init__(self, files: dict[str, bytes]):
        self.files, self.requests, cut = files, [], set()
        outer = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                data = outer.files.get(self.path.rsplit("/", 1)[-1])
                rng = self.headers.get("Range")
                outer.requests.append((self.path, rng))
                if data is None:
                    self.send_error(404)
                    return
                start = int(rng.split("=")[1].rstrip("-")) if rng else 0
                if start >= len(data):
                    self.send_error(416)
                    return
                body = data[start:]
                if not rng and self.path not in cut:
                    cut.add(self.path)
                    body = body[:len(body) * 2 // 5]
                self.send_response(206 if rng else 200)
                if rng:
                    self.send_header("Content-Range", f"bytes {start}-{len(data) - 1}/{len(data)}")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}/{{slug}}/{{file}}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()


def _raw_bytes():
    trans = "h,a,b\n" + "".join(f"{i},x{i},y{i}\n" for i in range(1000))
    patterns = "BEGIN LAUNDERING ATTEMPT - FAN-OUT:  t\n1,2,3\nEND LAUNDERING ATTEMPT - FAN-OUT\n"
    saml = "t,a,b\n" + "".join(f"{i},p{i},q{i}\n" for i in range(500))
    return {"HI-Small_Trans.csv": trans.encode(), "HI-Small_Patterns.txt": patterns.encode(), "SAML-D.csv": saml.encode()}


def test_fetch_resumes_a_silently_truncated_download(tmp_path, monkeypatch):
    monkeypatch.setitem(sess.EXPECTED_RAW, "HI-Small", (1000, "unused"))
    files = _raw_bytes()
    server = _Server(files)
    try:
        fetch.ensure_raw(["HI-Small", "SAML-D"], tmp_path / "d", url_template=server.url, log=lambda *a: None)
        for name, data in files.items():
            assert (tmp_path / "d" / name).read_bytes() == data
        assert any(rng for _, rng in server.requests), "an incomplete file must be resumed, not accepted"
        seen = len(server.requests)
        fetch.ensure_raw(["HI-Small", "SAML-D"], tmp_path / "d", url_template=server.url, log=lambda *a: None)
        assert len(server.requests) == seen, "whole files are not downloaded again"
    finally:
        server.close()


def test_fetch_prefers_a_whole_copy_in_drive_and_refuses_a_dead_end(tmp_path, monkeypatch):
    monkeypatch.setitem(sess.EXPECTED_RAW, "HI-Small", (1000, "unused"))
    backup = tmp_path / "backup"
    backup.mkdir()
    for name, data in _raw_bytes().items():
        (backup / name).write_bytes(data)
    dead = "http://127.0.0.1:9/{slug}/{file}"
    fetch.ensure_raw(["HI-Small"], tmp_path / "d", backup=backup, url_template=dead, log=lambda *a: None)
    assert (tmp_path / "d" / "HI-Small_Trans.csv").read_bytes() == _raw_bytes()["HI-Small_Trans.csv"]
    with pytest.raises(RuntimeError, match="incomplete|no progress"):
        fetch.ensure_raw(["SAML-D"], tmp_path / "e", url_template=dead, attempts=2, fetch=lambda url, path: 7, log=lambda *a: None)
    with pytest.raises(ValueError, match="cannot be downloaded"):
        fetch.ensure_raw(["Ethereum"], tmp_path / "f")


def test_fetch_rejects_a_file_with_the_wrong_row_count(tmp_path, monkeypatch):
    monkeypatch.setitem(sess.EXPECTED_RAW, "HI-Small", (1000, "unused"))
    path = tmp_path / "HI-Small_Trans.csv"
    path.write_bytes(_raw_bytes()["HI-Small_Trans.csv"][:-40])
    assert not fetch.complete("HI-Small", path.name, path)
    path.write_bytes(_raw_bytes()["HI-Small_Trans.csv"])
    assert fetch.complete("HI-Small", path.name, path)


def test_pilot_skips_medium_checks_on_a_small_machine():
    keep, skipped = sess.pilot_specs("NB4", ram_gb=12)
    assert [j for j, _ in skipped] == ["pilot-nb4-prepare-HI-Medium"] and "pilot-nb4-prepare-HI-Medium" not in {j.job_id for j in keep}
    assert not sess.pilot_specs("NB4", ram_gb=32)[1]
    assert not sess.pilot_specs("NB1", ram_gb=12)[1]


def test_pilot_skips_a_check_whose_file_is_missing_and_says_so(small_data, tmp_path):
    specs = [JobSpec("pilot-sweep", "gbt_sweep", SWEEP), JobSpec("pilot-nb4-prepare-SAML-D", "prepare_check", {"dataset": "SAML-D"})]
    report = sess.run_pilot("NB4", specs=specs, kill_point=("pilot-sweep", 1), budget_hours=1.0, **_common(_small_input(small_data, tmp_path), tmp_path))
    assert report["pass"]
    assert any("Skipped" in g[0] and "SAML-D" in g[0] and g[1] is None for g in report["gates"])


def test_final_checklist_colab_points_at_drive_not_kaggle(small_data, tmp_path, monkeypatch, capsys):
    spec = JobSpec("nb4-gbt-wall-HI-Small-released-H3600-s0", "gbt_sweep", SWEEP)
    sess.run_notebook("NB4", tiers=("P0",), budget_hours=1.0, jobs=[spec], restore_roots=[], **_common(_small_input(small_data, tmp_path), tmp_path))
    monkeypatch.setattr(sess, "plan", lambda nb, shard=None: [spec])
    capsys.readouterr()
    assert sess.final_checklist("NB4", ("P0",), tmp_path / "w", platform="colab")
    out = capsys.readouterr().out
    assert "Drive" in out and "aml-out" not in out
    monkeypatch.setattr(sess, "plan", lambda nb, shard=None: [spec, JobSpec("not-run-yet", "gbt_sweep", SWEEP)])
    assert not sess.final_checklist("NB4", ("P0",), tmp_path / "w", platform="colab")
    assert "Run this cell again" in capsys.readouterr().out


def test_notebook_runs_on_colab_with_drive_output_and_downloaded_data(small_data, tmp_path, monkeypatch):
    content = tmp_path / "content"
    home = content / "drive" / "MyDrive" / "aml"
    home.mkdir(parents=True)
    staging = tmp_path / "staging"
    _package(staging)
    shutil.copy(next((staging / "code").glob("aml-code-*.zip")), home)
    raw = _small_input(small_data, tmp_path) / "raw"
    asked = {}

    def fake_ensure_raw(names, dest, backup=None, **kw):
        asked.update(names=list(names), dest=Path(dest), backup=backup)
        shutil.copytree(raw, dest, dirs_exist_ok=True)
        return dest

    spec = JobSpec("nb4-gbt-wall-HI-Small-released-H3600-s0", "gbt_sweep", SWEEP)
    monkeypatch.setattr(fetch, "ensure_raw", fake_ensure_raw)
    monkeypatch.setattr(sess, "plan", lambda nb, shard=None: [spec])
    monkeypatch.setattr(sess, "EXPECTED_RAW", {})
    monkeypatch.setattr(sess, "OFFICIAL_NEEDED", {})
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setenv("AMLH_PLATFORM", "colab")
    ns = {}
    for c in cells("nb4"):
        if c["cell_type"] != "code":
            continue
        text = (source(c).replace("Path('/content')", f"Path({content.as_posix()!r})").replace("/tmp/aml_code", (tmp_path / "code").as_posix())
                .replace("MODE = 'pilot'", "MODE = 'production'").replace("INSTALL = True", "INSTALL = False").replace("RUN_TESTS = True", "RUN_TESTS = False")
                .replace("shard=SHARD,", f"shard=SHARD, cache_root=Path({(tmp_path / 'c').as_posix()!r}), official_root=Path({(tmp_path / 'o').as_posix()!r}),"))
        exec(compile(text, "nb4", "exec"), ns)
    assert ns["PLATFORM"] == "colab" and ns["TIERS"] == ("P0",) and ns["BUDGET_HOURS"] == 5.0
    assert ns["WORKING"] == home / "out_nb4"
    assert asked["names"] == ["HI-Small"] and asked["dest"] == content / "aml_data" and asked["backup"] == home / "data"
    assert verify_job(home / "out_nb4" / "aml_run" / "jobs" / spec.job_id)[0], "output must land in Drive, where a disconnect cannot lose it"


def test_install_gnn_stack_falls_back_through_known_torch_combos_in_order(monkeypatch):
    """Found live, on a Molab RTX PRO 6000 Blackwell session (compute capability sm_120, needs CUDA >= 12.8): the
    session's own torch had no PyG wheels, and the single hard-coded fallback (2.5.1+cu121, written for the RTX
    3080 fleet) also has none for that card -- PyG's wheel index stopped publishing torch-scatter/torch-sparse
    builds after torch-2.9.1+cu130. Verified end to end there (torch, torch-scatter, torch-sparse, and a real
    GINEConv forward pass, all on CUDA) that 2.5.1+cu121, 2.9.1+cu130 is a real, tried-in-order list that must stop
    at the first combo with wheels, never silently pick a later one when an earlier one already has what it needs."""
    calls = []

    def fake_pip(*args):
        calls.append(args)
        if "torch-scatter" in args:  # the `wheels()` probe: only the second candidate's URL has wheels
            return args[-1] == "https://data.pyg.org/whl/torch-2.9.1+cu130.html"
        return True  # every `torch==` install itself succeeds

    monkeypatch.setattr(sess, "_pip", fake_pip)
    monkeypatch.setattr("importlib.metadata.version", lambda pkg: "2.15.0")  # the session's own torch (no matching wheels either); install_gnn_stack imports `version` locally, so the patch target is importlib.metadata itself
    result = sess.install_gnn_stack()
    assert result == "2.9.1+cu130"
    installs = [c for c in calls if c and c[0].startswith("torch==")]
    assert [c[0] for c in installs] == ["torch==2.5.1", "torch==2.9.1"], "must try the known-good RTX 3080 combo before any newer one"


def test_install_gnn_stack_raises_with_every_combo_tried_when_none_have_wheels(monkeypatch):
    monkeypatch.setattr(sess, "_pip", lambda *a: "torch-scatter" not in a)  # every torch install succeeds; no wheel probe ever does
    monkeypatch.setattr("importlib.metadata.version", lambda pkg: "2.15.0")
    with pytest.raises(RuntimeError, match=r"2\.5\.1\+cu121.*2\.9\.1\+cu130"):
        sess.install_gnn_stack()
