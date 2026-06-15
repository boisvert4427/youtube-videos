from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from scraper.demography import build_france_female_first_names_timeseries_csv as base


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_RAW_DIR = base.DEFAULT_RAW_DIR
DEFAULT_ARCHIVE = base.DEFAULT_ARCHIVE
DEFAULT_SOURCE_CSV = base.DEFAULT_SOURCE_CSV
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "demography"
    / "france_male_first_names"
    / "france_male_first_names_1900_2024.csv"
)

START_YEAR = 1900
END_YEAR = 2024
MALE_CODE = "1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the official annual French male first-name timeseries."
    )
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--source-csv", type=Path, default=DEFAULT_SOURCE_CSV)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--start-year", type=int, default=START_YEAR)
    parser.add_argument("--end-year", type=int, default=END_YEAR)
    parser.add_argument("--keep-top", type=int, default=30)
    parser.add_argument("--refresh", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.start_year > args.end_year:
        raise ValueError("--start-year must be less than or equal to --end-year.")
    if args.keep_top < 12:
        raise ValueError("--keep-top must be at least 12.")

    previous_code = base.FEMALE_CODE
    base.FEMALE_CODE = MALE_CODE
    try:
        if args.refresh or not args.archive.exists():
            base._download(base.INSEE_ARCHIVE_URL, args.archive)
        if args.refresh or not args.source_csv.exists():
            base._extract_csv(args.archive, args.source_csv)

        by_year = base._load_female_values(args.source_csv, args.start_year, args.end_year)
        rows = base.build_rows(by_year, args.start_year, args.end_year, args.keep_top)
        output = base.write_csv(rows, args.output)
    finally:
        base.FEMALE_CODE = previous_code

    print(f"[scraper] France male first-name CSV generated -> {output}")
    print(
        f"[scraper] {len(rows)} rows, {args.start_year}-{args.end_year}, "
        f"top {args.keep_top} retained per year"
    )


if __name__ == "__main__":
    main()
