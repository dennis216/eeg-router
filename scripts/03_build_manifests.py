from __future__ import annotations

import argparse
from pathlib import Path

from bfa.data.scan import scan_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("manifests"))
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    recordings, seizures = scan_dataset(args.root, args.out / "data_audit.json")
    recordings.to_parquet(args.out / "recordings.parquet", index=False)
    seizures.to_parquet(args.out / "seizures.parquet", index=False)
    print(
        f"recordings={len(recordings)} patients={recordings.patient_id.nunique()} "
        f"seizures={len(seizures)}"
    )


if __name__ == "__main__":
    main()
