"""Gets the raw files onto a machine that has no attached datasets (Colab).

The download endpoint has silently truncated files with a success exit code, so a file counts only when it passes its own
completeness check, and an incomplete one is resumed. Hashes are still checked afterwards by session.verify_raw.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import zipfile
from pathlib import Path

AML_SLUG = "ealtman2019/ibm-transactions-for-anti-money-laundering-aml"
SAML_SLUG = "berkanoztas/synthetic-transaction-monitoring-dataset-aml"
URL = "https://www.kaggle.com/api/v1/datasets/download/{slug}/{file}"
PATTERNS_END = b"END LAUNDERING ATTEMPT"
ZIP_MAGIC = b"PK\x03\x04"


def sources(name: str) -> list[tuple[str, str]]:
    from .session import EXPECTED_RAW
    if name in EXPECTED_RAW:
        return [(AML_SLUG, f"{name}_Trans.csv"), (AML_SLUG, f"{name}_Patterns.txt")]
    if name == "SAML-D":
        return [(SAML_SLUG, "SAML-D.csv")]
    raise ValueError(f"{name} cannot be downloaded: put its file in the data folder by hand")


def _tail(path: Path, n: int = 4096) -> bytes:
    size = path.stat().st_size
    with open(path, "rb") as fh:
        fh.seek(max(size - n, 0))
        return fh.read()


def is_zip(path: Path) -> bool:
    """The endpoint sometimes returns a file as a zip archive (SAML-D did), whatever the requested name says."""
    if not path.exists() or path.stat().st_size < 4:
        return False
    with open(path, "rb") as fh:
        return fh.read(4) == ZIP_MAGIC


def unpack_if_zip(path: Path, log=print) -> bool:
    """Replaces a complete zip with the one file inside it. A truncated zip is left alone, so a resume can finish it."""
    if not is_zip(path) or not zipfile.is_zipfile(path):
        return False
    tmp = path.with_name(path.name + ".unzipped")
    with zipfile.ZipFile(path) as z:
        members = [n for n in z.namelist() if not n.endswith("/")]
        member = next((n for n in members if Path(n).name == path.name), members[0] if len(members) == 1 else None)
        if member is None:
            raise ValueError(f"{path.name} is a zip holding {members}, and none of them is the file")
        with z.open(member) as src, open(tmp, "wb") as dst:
            shutil.copyfileobj(src, dst, 1 << 22)
    os.replace(tmp, path)
    log(f"{path.name}: was delivered as a zip, unpacked")
    return True


def complete(name: str, file: str, path: Path) -> bool:
    """True only when the file is whole. Row counts come from SPEC.md, and for files with none the last row must match the header's width."""
    from .session import EXPECTED_RAW
    if not path.exists() or path.stat().st_size == 0 or is_zip(path):
        return False
    if file.endswith("_Patterns.txt"):
        return PATTERNS_END in _tail(path)
    if name in EXPECTED_RAW:
        with open(path, "rb") as fh:
            counted = sum(buf.count(b"\n") for buf in iter(lambda: fh.read(1 << 22), b"")) - 1
        return counted == EXPECTED_RAW[name][0]
    with open(path, "rb") as fh:
        header = fh.readline()
    tail = _tail(path)
    last = [line for line in tail.splitlines() if line.strip()][-1]
    # Catches a cut inside a row, not one exactly between rows. A file with no final newline is treated as cut.
    return tail.endswith(b"\n") and last.count(b",") == header.count(b",")


def _curl(url: str, path: Path) -> int:
    return subprocess.run(["curl", "-sS", "-L", "-C", "-", "--retry", "3", "--retry-delay", "5", "-o", str(path), url]).returncode


def ensure_raw(names: list[str], dest: Path, backup: Path | None = None, url_template: str = URL, fetch=_curl, attempts: int = 8, log=print) -> Path:
    """Makes every named dataset's files whole in dest. A whole copy in `backup` (for example on Drive) is used before downloading."""
    from .session import EXPECTED_RAW
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    for name in names:
        for slug, file in sources(name):
            target = dest / file
            unpack_if_zip(target, log)
            if complete(name, file, target):
                continue
            if backup is not None and complete(name, file, Path(backup) / file):
                log(f"copying {file} from {backup}")
                shutil.copyfile(Path(backup) / file, target)
                continue
            restarted = False
            for attempt in range(1, attempts + 1):
                before = target.stat().st_size if target.exists() else 0
                code = fetch(url_template.format(slug=slug, file=file), target)
                unpack_if_zip(target, log)
                if complete(name, file, target):
                    log(f"{file}: whole after {attempt} attempt(s)")
                    break
                after = target.stat().st_size if target.exists() else 0
                if after == before > 0 and code == 0 and name not in EXPECTED_RAW and file.endswith(".csv") and not is_zip(target):
                    log(f"{file}: the server has nothing more to send, so it is treated as whole")  # curl treats a resume past the end as success
                    break
                log(f"{file}: incomplete ({after} bytes, curl exit {code}), resuming")
                if after <= before and target.exists():  # no progress: a corrupt partial file would be resumed forever
                    if restarted:
                        raise RuntimeError(f"{file}: no progress and still incomplete. Put a whole copy in {backup or dest} by hand")
                    target.unlink()
                    restarted = True
            else:
                raise RuntimeError(f"{file}: still incomplete after {attempts} attempts")
    return dest
