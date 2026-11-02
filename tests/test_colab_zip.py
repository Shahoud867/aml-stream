"""Regression tests for the first Colab pilot: SAML-D arrived as a zip, and one bad optional file crashed the whole session."""
import io
import zipfile

import pytest

import amlh.fetch as fetch
import amlh.session as sess
from amlh.runner import JobSpec
from test_colab import _Server, _raw_bytes
from test_notebooks import _common, _small_input, restore_globals  # noqa: F401 (restore_globals is an autouse fixture)


def _zipped(name: str, data: bytes) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(name, data)
    return buf.getvalue()


def test_a_file_delivered_as_a_truncated_zip_is_resumed_and_unpacked(tmp_path):
    original = _raw_bytes()["SAML-D.csv"] * 40
    server = _Server({"SAML-D.csv": _zipped("SAML-D.csv", original)})
    try:
        fetch.ensure_raw(["SAML-D"], tmp_path / "d", url_template=server.url, log=lambda *a: None)
    finally:
        server.close()
    assert (tmp_path / "d" / "SAML-D.csv").read_bytes() == original
    assert any(rng for _, rng in server.requests), "the cut zip must be resumed"
    assert not list((tmp_path / "d").glob("*.unzipped"))


def test_a_complete_zip_already_on_disk_is_unpacked_without_downloading(tmp_path):
    original = _raw_bytes()["SAML-D.csv"]
    dest = tmp_path / "d"
    dest.mkdir()
    (dest / "SAML-D.csv").write_bytes(_zipped("SAML-D.csv", original))
    fetch.ensure_raw(["SAML-D"], dest, url_template="http://127.0.0.1:9/{slug}/{file}", log=lambda *a: None)
    assert (dest / "SAML-D.csv").read_bytes() == original


def test_a_truncated_zip_is_never_treated_as_complete(tmp_path):
    path = tmp_path / "SAML-D.csv"
    path.write_bytes(_zipped("SAML-D.csv", _raw_bytes()["SAML-D.csv"] * 40)[:200])
    assert fetch.is_zip(path) and not fetch.unpack_if_zip(path, lambda *a: None) and not fetch.complete("SAML-D", path.name, path)


def test_a_zip_with_the_wrong_contents_is_reported(tmp_path):
    path = tmp_path / "SAML-D.csv"
    path.write_bytes(_zipped("a.csv", b"x,y\n1,2\n") + b"")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("a.csv", b"x\n")
        z.writestr("b.csv", b"y\n")
    path.write_bytes(buf.getvalue())
    with pytest.raises(ValueError, match="none of them is the file"):
        fetch.unpack_if_zip(path, lambda *a: None)


def test_a_bad_file_that_only_a_prepare_check_touches_fails_one_job_not_the_session(small_data, tmp_path):
    inp = _small_input(small_data, tmp_path)
    (inp / "raw" / "SAML-D.csv").write_bytes(b"PK\x03\x04 this is not a real archive")
    sweep = JobSpec("pilot-sweep", "gbt_sweep", {"dataset": "HI-Small", "protocol": "released", "seed": 0, "family": "wall", "neg_rate": 0.25, "H": "3600"})
    check = JobSpec("pilot-nb4-prepare-SAML-D", "prepare_check", {"dataset": "SAML-D"})
    summary = sess.run_notebook("NB4", tiers=("P0",), budget_hours=1.0, jobs=[sweep, check], restore_roots=[], **_common(inp, tmp_path))
    status = {r["job_id"]: r["status"] for r in summary["jobs"]}
    assert status == {"pilot-sweep": "DONE", "pilot-nb4-prepare-SAML-D": "FAILED"}, "the sweep must still run"
    assert not summary["complete"]


def test_prepare_check_reports_size_and_time_for_a_good_file(small_data, tmp_path):
    inp = _small_input(small_data, tmp_path)
    check = JobSpec("pilot-nb4-prepare-LI-Small", "prepare_check", {"dataset": "LI-Small"})
    summary = sess.run_notebook("NB4", tiers=("P0",), budget_hours=1.0, jobs=[check], restore_roots=[], **_common(inp, tmp_path))
    assert summary["complete"]
    import json
    m = json.loads((tmp_path / "w/aml_run/jobs/pilot-nb4-prepare-LI-Small/metrics.json").read_text())
    assert m["rows"] > 0 and m["prepare_seconds"] >= 0
