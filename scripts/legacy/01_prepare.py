import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from amlh.prepare import prepare  # noqa: E402

name = sys.argv[1] if len(sys.argv) > 1 else "HI-Small"
card = prepare(ROOT / f"data/raw/{name}_Trans.csv", ROOT / f"data/raw/{name}_Patterns.txt",
               ROOT / f"data/processed/{name}")
print(json.dumps({k: v for k, v in card.items() if k not in ("usd_rates", "usd_rate_cv", "vocab")}, indent=1))
print("currencies with missing rate:", card["rate_missing"])
print("max rate CV:", max((v, k) for k, v in card["usd_rate_cv"].items() if v == v))
print("vocab fmt:", card["vocab"]["fmt"])
