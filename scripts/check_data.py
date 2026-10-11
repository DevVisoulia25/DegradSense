"""Dependency-free verification of downloaded FD001 files."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    folder = ROOT / "data" / "raw"
    for kind in ("train", "test"):
        engines, count = {}, 0
        with (folder / f"{kind}_FD001.txt").open() as stream:
            for row in stream:
                values = row.split()
                if len(values) != 26:
                    raise ValueError("Expected 26 values per row")
                engine, cycle = int(values[0]), int(values[1])
                if cycle != engines.get(engine, 0) + 1:
                    raise ValueError("Engine cycles must be consecutive")
                engines[engine] = cycle
                count += 1
        if len(engines) != 100:
            raise ValueError("FD001 should contain 100 engines per split")
        print(f"{kind}: {count:,} rows, {len(engines)} engines")
    targets = [
        float(row.strip()) for row in (folder / "RUL_FD001.txt").read_text().splitlines() if row.strip()
    ]
    if len(targets) != 100 or min(targets) < 0:
        raise ValueError("Expected 100 nonnegative RUL targets")
    print("RUL vector: 100 valid targets. Raw-file structure verified.")


if __name__ == "__main__":
    main()
