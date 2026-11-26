"""Download the VAST dataset (no account, no registration, about 11 MB).

Run from the project root:
    python scripts/download_data.py
    python scripts/download_data.py --with-train     # also the 30 MB training file (ARGUS does not need it)

VAST ("Varied Stance Topics", Allaway and McKeown, 2020) is published on GitHub:
    https://github.com/emilyallaway/zero-shot-stance/tree/master/data/VAST
Files go to data/raw/vast/. A file that is already there is kept unless you pass --force.

If this fails (a firewall, no internet): open the two links the script prints in your browser, press Ctrl+S to save
each page as vast_dev.csv and vast_test.csv into data/raw/vast/, and carry on with prepare_data.py.
"""
import argparse
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

BASE_URL = "https://raw.githubusercontent.com/emilyallaway/zero-shot-stance/master/data/VAST"
FILES = {"vast_dev.csv": 2_000_000, "vast_test.csv": 3_000_000, "vast_train.csv": 10_000_000}   # rough minimum sizes
REQUIRED_HEADER = ("post", "new_topic", "label")


def download(url: str, dest: Path, retries: int = 3) -> int:
    """Download to a temporary file, check it looks like a VAST file, then move it into place. Returns the size."""
    tmp = dest.with_suffix(dest.suffix + ".part")
    last = None
    for attempt in range(1, retries + 1):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "argus-data-download"})
            with urllib.request.urlopen(request, timeout=60) as resp, open(tmp, "wb") as out:
                total = int(resp.headers.get("Content-Length") or 0)
                done = 0
                while True:
                    chunk = resp.read(1 << 16)
                    if not chunk:
                        break
                    out.write(chunk)
                    done += len(chunk)
                    shown = f"{done / 1e6:.1f} of {total / 1e6:.1f} MB" if total else f"{done / 1e6:.1f} MB"
                    sys.stdout.write(f"\r  {dest.name}: {shown}   ")
                    sys.stdout.flush()
            sys.stdout.write("\n")
            with open(tmp, "r", encoding="utf-8-sig", errors="replace") as fh:
                header = fh.readline()
            if not all(col in header for col in REQUIRED_HEADER):
                raise ValueError(f"the downloaded file does not look like VAST (first line: {header[:80]!r})")
            tmp.replace(dest)
            return dest.stat().st_size
        except (urllib.error.URLError, OSError, ValueError) as e:
            last = e
            sys.stdout.write("\n")
            if attempt < retries:
                print(f"  attempt {attempt} failed ({e}); retrying ...")
                time.sleep(2 * attempt)
    tmp.unlink(missing_ok=True)
    raise RuntimeError(str(last))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "raw" / "vast")
    ap.add_argument("--with-train", action="store_true", help="also download vast_train.csv (not needed)")
    ap.add_argument("--force", action="store_true", help="download again even if the file exists")
    ap.add_argument("--base-url", default=BASE_URL, help="where to fetch from (change only for testing)")
    args = ap.parse_args()

    names = [n for n in FILES if args.with_train or n != "vast_train.csv"]
    args.out.mkdir(parents=True, exist_ok=True)
    failed = []
    for name in names:
        dest = args.out / name
        if dest.exists() and dest.stat().st_size > FILES[name] and not args.force:
            print(f"  {name}: already there ({dest.stat().st_size / 1e6:.1f} MB), skipping")
            continue
        url = f"{args.base_url.rstrip('/')}/{name}"
        try:
            size = download(url, dest)
            print(f"  {name}: saved ({size / 1e6:.1f} MB)")
        except RuntimeError as e:
            failed.append((name, url, e))

    if failed:
        lines = ["Could not download:"]
        for name, url, e in failed:
            lines.append(f"  {name}: {e}\n    open {url.replace('raw.githubusercontent.com', 'github.com').replace('/master/', '/blob/master/')}"
                         f"\n    (or the raw link {url}) and save it as {args.out / name}")
        lines.append("Then run:  python scripts/prepare_data.py")
        sys.exit("\n".join(lines))
    print(f"\nDone. Next:  python scripts/prepare_data.py")


if __name__ == "__main__":
    main()
