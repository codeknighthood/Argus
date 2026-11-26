"""Phase 7: is this computer ready to run ARGUS?

Run from the project root:
    python scripts/check_setup.py            # quick checks
    python scripts/check_setup.py --live     # also asks the model one real question

Prints [ OK ], [WARN] or [FAIL] for Python, the packages, config.yaml, Ollama and the model, and the data, and for
anything that is not OK it prints the exact command that fixes it. Exit code 1 if something must be fixed.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from argus.setup_check import format_results, run_checks  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--live", action="store_true", help="also make one real model call")
    args = ap.parse_args()
    results = run_checks(ROOT, live=args.live)
    print(format_results(results))
    sys.exit(1 if any(r["status"] == "fail" for r in results) else 0)


if __name__ == "__main__":
    main()
