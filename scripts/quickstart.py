"""Run the whole ARGUS workflow in order, stopping at the first problem with a message that says what to do.

Run from the project root, with Ollama running:
    python scripts/quickstart.py                  # about 20 to 40 minutes on an RTX 4060
    python scripts/quickstart.py --limit 30       # quicker look
    python scripts/quickstart.py --skip-eval      # only check, download, prepare and run the pipeline

Steps:  1 check the setup (including one real model call)   2 download the VAST data   3 prepare the dev and test samples
        4 compare the three stance strategies on dev        5 run the full pipeline on the sample transcript

Safe to run again: downloaded files are kept and model answers are cached, so a second run only redoes what is missing.
What to do with the results is in INSTALL.md.
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def run_step(number: int, total: int, title: str, command: list[str]) -> None:
    print(f"\n===== Step {number} of {total}: {title} =====", flush=True)
    started = time.time()
    code = subprocess.run([sys.executable, *command], cwd=ROOT).returncode
    if code != 0:
        sys.exit(f"\nStep {number} ({title}) did not finish. Fix what is printed above and run this again; "
                 "finished steps will not be repeated in full.")
    print(f"(step {number} took {time.time() - started:.0f} s)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=60, help="dev sentences used in step 4 (default 60)")
    ap.add_argument("--skip-eval", action="store_true", help="skip step 4")
    ap.add_argument("--skip-pipeline", action="store_true", help="skip step 5")
    ap.add_argument("--base-url", default=None, help="download from here instead of GitHub (testing only)")
    args = ap.parse_args()

    download = ["scripts/download_data.py"] + (["--base-url", args.base_url] if args.base_url else [])
    steps = [("check the setup", ["scripts/check_setup.py", "--live"]),
             ("download the VAST data", download),
             ("prepare the dev and test samples", ["scripts/prepare_data.py"])]
    if not args.skip_eval:
        steps.append((f"compare the stance strategies on {args.limit} dev sentences",
                      ["scripts/evaluate.py", "--limit", str(args.limit), "--bootstrap", "200"]))
    if not args.skip_pipeline:
        steps.append(("run the full pipeline on the sample transcript", ["scripts/run_pipeline.py"]))
    for i, (title, command) in enumerate(steps, start=1):
        run_step(i, len(steps), title, command)

    print("\nAll steps finished.")
    if not args.skip_eval:
        print("  Strategy comparison:  results/evaluation/summary.md")
    if not args.skip_pipeline:
        print("  Pipeline output:      results/sample_transcript.csv  (open in Excel)")
    print("  Try the demo:         streamlit run app.py")


if __name__ == "__main__":
    main()
