"""Packs the research code into one zip for a private Kaggle dataset, so all five notebooks run the same version.

Usage: python scripts/make_code_dataset.py <output_dir>
Prints the code hash that every notebook's manifest must show.
"""
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from amlh.runner import code_hash  # noqa: E402

out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "dist"
out_dir.mkdir(parents=True, exist_ok=True)
version = code_hash(ROOT)
target = out_dir / f"aml-code-{version}.zip"
INCLUDE = [("src/amlh", "*.py"), ("scripts", "*.py"), ("tests", "*.py"), ("notebooks", "*.ipynb")]
FILES = ["requirements-kaggle.txt", "requirements-mgnn.txt"]
with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as z:
    for sub, pattern in INCLUDE:
        for f in sorted((ROOT / sub).glob(pattern)):
            z.write(f, f.relative_to(ROOT))
    for name in FILES:
        if (ROOT / name).exists():
            z.write(ROOT / name, name)
print(f"code hash: {version}\nwrote {target} ({target.stat().st_size / 1024:.0f} KB)")
