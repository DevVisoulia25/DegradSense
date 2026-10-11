"""Download the NASA-linked PHM archive and extract only C-MAPSS text files."""

import argparse
import hashlib
import io
import json
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

URL = "https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip"
ROOT = Path(__file__).resolve().parents[1]


def extract(payload, destination, depth=0):
    if depth > 3:
        raise ValueError("Too many nested ZIP archives")
    with ZipFile(io.BytesIO(payload)) as archive:
        for entry in archive.infolist():
            name = Path(entry.filename).name
            if entry.file_size > 150_000_000:
                raise ValueError("Unexpectedly large archive entry")
            if name.lower().endswith(".zip"):
                extract(archive.read(entry), destination, depth + 1)
            elif name.lower().endswith(".txt") and (
                name.startswith(("train_FD00", "test_FD00", "RUL_FD00")) or "readme" in name.lower()
            ):
                # Only basenames are used, so archive paths cannot escape destination.
                target = destination / name
                if not target.exists():
                    target.write_bytes(archive.read(entry))
                    print(f"Extracted {name}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, help="Use an already downloaded ZIP")
    args = parser.parse_args()
    destination = ROOT / "data" / "raw"
    destination.mkdir(parents=True, exist_ok=True)
    if args.archive:
        payload = args.archive.read_bytes()
    else:
        print("Downloading NASA-linked C-MAPSS archive...", flush=True)
        with urlopen(URL, timeout=60) as response:
            payload = response.read(150_000_001)
        if len(payload) > 150_000_000:
            raise ValueError("Archive exceeds expected size")
    extract(payload, destination)
    required = [destination / f"{prefix}_FD001.txt" for prefix in ("train", "test", "RUL")]
    if not all(path.exists() for path in required):
        raise RuntimeError("Archive did not contain the required FD001 files")
    manifest = {
        "source_url": URL,
        "archive_sha256": hashlib.sha256(payload).hexdigest(),
        "citation": "A. Saxena and K. Goebel (2008), Turbofan Engine Degradation Simulation Data Set, NASA Ames.",
        "files": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(destination.glob("*.txt"))
        },
    }
    (destination / "source.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("Dataset ready in data/raw", flush=True)


if __name__ == "__main__":
    main()
