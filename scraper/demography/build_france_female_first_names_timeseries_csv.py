from __future__ import annotations

import argparse
import csv
import shutil
import urllib.request
import zipfile
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw" / "demography" / "france_first_names"
DEFAULT_ARCHIVE = DEFAULT_RAW_DIR / "prenoms-2024-nat_csv.zip"
DEFAULT_SOURCE_CSV = DEFAULT_RAW_DIR / "prenoms-2024-nat.csv"
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "demography"
    / "france_female_first_names"
    / "france_female_first_names_1900_2024.csv"
)

INSEE_ARCHIVE_URL = (
    "https://www.insee.fr/fr/statistiques/fichier/8595130/"
    "prenoms-2024-nat_csv.zip"
)
USER_AGENT = "Mozilla/5.0 (compatible; Codex France Female First Names Builder)"
START_YEAR = 1900
END_YEAR = 2024
FEMALE_CODE = "2"


def _download(url: str, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".download")
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=120) as response, temporary_path.open("wb") as output:
        shutil.copyfileobj(response, output)
    temporary_path.replace(output_path)
    return output_path


def _extract_csv(archive_path: Path, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        csv_members = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        if not csv_members:
            raise RuntimeError("The Insee archive does not contain a CSV file.")
        with archive.open(csv_members[0]) as source, output_path.open("wb") as output:
            shutil.copyfileobj(source, output)
    return output_path


def _display_name(value: str) -> str:
    return value.strip().title()


def _load_female_values(
    source_csv: Path,
    start_year: int,
    end_year: int,
) -> dict[int, list[tuple[str, int]]]:
    by_year: dict[int, list[tuple[str, int]]] = defaultdict(list)
    with source_csv.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file, delimiter=";")
        for row in reader:
            if row.get("sexe", "").strip() != FEMALE_CODE:
                continue
            year_text = row.get("periode", "").strip()
            raw_name = row.get("prenom", "").strip()
            if not year_text.isdigit() or raw_name.startswith("_"):
                continue
            year = int(year_text)
            if not start_year <= year <= end_year:
                continue
            try:
                births = int(row.get("valeur", "0"))
            except ValueError:
                continue
            if births <= 0:
                continue
            by_year[year].append((_display_name(raw_name), births))

    missing = [year for year in range(start_year, end_year + 1) if year not in by_year]
    if missing:
        raise RuntimeError(f"Missing female first-name data for years: {missing}")
    return by_year


def _format_births(value: int) -> str:
    return f"{value:,}".replace(",", " ")


def build_rows(
    by_year: dict[int, list[tuple[str, int]]],
    start_year: int,
    end_year: int,
    keep_top: int,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    previous_values: dict[str, int] = {}

    for year in range(start_year, end_year + 1):
        ranked = sorted(by_year[year], key=lambda item: (-item[1], item[0]))
        leader_name, leader_births = ranked[0]
        changes = sorted(
            (
                (births - previous_values.get(name, births), name)
                for name, births in ranked
                if name in previous_values
            ),
            reverse=True,
        )
        if changes and changes[0][0] > 0:
            gain, rising_name = changes[0]
            summary = (
                f"N°1 : {leader_name} ({_format_births(leader_births)})"
                f"|Plus forte hausse : {rising_name} (+{_format_births(gain)})"
            )
        else:
            summary = (
                f"N°1 : {leader_name} ({_format_births(leader_births)})"
                f"|Classement annuel {year}"
            )

        for rank, (name, births) in enumerate(ranked[:keep_top], start=1):
            previous = previous_values.get(name)
            rows.append(
                {
                    "ranking_date": f"{year}-12-31",
                    "first_name": name,
                    "births": str(births),
                    "annual_change": "" if previous is None else str(births - previous),
                    "annual_rank": str(rank),
                    "season_summary": summary,
                    "data_source": "Insee - Fichier des prénoms 2024",
                }
            )
        previous_values = dict(ranked)
    return rows


def write_csv(rows: list[dict[str, str]], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "ranking_date",
                "first_name",
                "births",
                "annual_change",
                "annual_rank",
                "season_summary",
                "data_source",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the official annual French female first-name timeseries."
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

    if args.refresh or not args.archive.exists():
        _download(INSEE_ARCHIVE_URL, args.archive)
    if args.refresh or not args.source_csv.exists():
        _extract_csv(args.archive, args.source_csv)

    by_year = _load_female_values(args.source_csv, args.start_year, args.end_year)
    rows = build_rows(by_year, args.start_year, args.end_year, args.keep_top)
    output = write_csv(rows, args.output)
    print(f"[scraper] France female first-name CSV generated -> {output}")
    print(
        f"[scraper] {len(rows)} rows, {args.start_year}-{args.end_year}, "
        f"top {args.keep_top} retained per year"
    )


if __name__ == "__main__":
    main()
