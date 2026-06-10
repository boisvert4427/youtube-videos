from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import urllib.request
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw" / "demography" / "world_bank_population"
DEFAULT_WORLD_BANK_JSON = DEFAULT_RAW_DIR / "world_bank_population_1960_2024.json"
DEFAULT_METADATA = DEFAULT_RAW_DIR / "world_bank_country_metadata.json"
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "demography"
    / "world_population"
    / "world_population_1960_2024.csv"
)
DEFAULT_FLAGS_DIR = PROJECT_ROOT / "data" / "raw" / "flags"

WORLD_BANK_POPULATION_URL = (
    "https://api.worldbank.org/v2/country/all/indicator/SP.POP.TOTL"
    "?date=1960:2024&format=json&per_page=25000"
)
WORLD_BANK_METADATA_URL = "https://api.worldbank.org/v2/country?format=json&per_page=400"
FLAG_URL_TEMPLATE = "https://flagcdn.com/w80/{alpha2}.png"
USER_AGENT = "Mozilla/5.0 (compatible; Codex World Population Race Builder)"

START_YEAR = 1960
END_YEAR = 2024


def _download(url: str, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".download")
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=120) as response, temporary_path.open("wb") as output:
        shutil.copyfileobj(response, output)
    temporary_path.replace(output_path)
    return output_path


def _ensure_json(url: str, path: Path, refresh: bool) -> Path:
    if refresh or not path.exists():
        _download(url, path)
    return path


def _load_country_metadata(metadata_path: Path) -> dict[str, tuple[str, str]]:
    payload = json.loads(metadata_path.read_text(encoding="utf-8-sig"))
    records = payload[1] if isinstance(payload, list) and len(payload) > 1 else []
    countries: dict[str, tuple[str, str]] = {}
    for record in records:
        region = record.get("region") or {}
        if region.get("id") == "NA":
            continue
        alpha3 = str(record.get("id", "")).strip().upper()
        alpha2 = str(record.get("iso2Code", "")).strip().upper()
        name = str(record.get("name", "")).strip()
        if len(alpha3) == 3 and len(alpha2) == 2:
            countries[alpha3] = (alpha2, name)
    if not countries:
        raise RuntimeError("World Bank metadata did not contain any countries.")
    return countries


def _load_world_bank_values(
    json_path: Path,
    countries: dict[str, tuple[str, str]],
) -> dict[tuple[int, str], tuple[str, str, float, str]]:
    payload = json.loads(json_path.read_text(encoding="utf-8-sig"))
    records = payload[1] if isinstance(payload, list) and len(payload) > 1 else []
    values: dict[tuple[int, str], tuple[str, str, float, str]] = {}
    for record in records:
        alpha3 = str(record.get("countryiso3code", "")).strip().upper()
        if alpha3 not in countries or record.get("value") is None:
            continue
        try:
            year = int(record.get("date", ""))
            population = float(record["value"])
        except (TypeError, ValueError):
            continue
        if not math.isfinite(population) or population <= 0:
            continue
        alpha2, metadata_name = countries[alpha3]
        country = record.get("country") or {}
        country_name = str(country.get("value", "")).strip() or metadata_name
        values[(year, alpha3)] = (country_name, alpha2, population, "World Bank")
    return values


def _format_population(value: float) -> str:
    return f"{value / 1_000_000:,.1f}M"


def build_rows(
    values: dict[tuple[int, str], tuple[str, str, float, str]],
    start_year: int,
    end_year: int,
) -> list[dict[str, str]]:
    by_year: dict[int, list[tuple[str, str, str, float, str]]] = defaultdict(list)
    for (year, alpha3), (country_name, alpha2, population, source) in values.items():
        if start_year <= year <= end_year:
            by_year[year].append((alpha3, country_name, alpha2, population, source))

    missing_years = [year for year in range(start_year, end_year + 1) if not by_year[year]]
    if missing_years:
        raise RuntimeError(f"Missing population data for years: {missing_years}")

    rows: list[dict[str, str]] = []
    previous_values: dict[str, float] = {}
    for year in range(start_year, end_year + 1):
        ranked = sorted(by_year[year], key=lambda item: (-item[3], item[1]))
        leader = ranked[0]

        growth: list[tuple[float, str]] = []
        for alpha3, country_name, _, population, _ in ranked:
            previous = previous_values.get(alpha3)
            if previous is not None:
                growth.append((population - previous, country_name))
        growth.sort(reverse=True)
        if growth:
            change, fastest_name = growth[0]
            summary = (
                f"Leader: {leader[1]} {_format_population(leader[3])}"
                f"|Fastest growth: {fastest_name} +{_format_population(change)}"
            )
        else:
            summary = f"Leader: {leader[1]} {_format_population(leader[3])}|Dataset begins in {year}"

        for alpha3, country_name, alpha2, population, source in ranked:
            previous = previous_values.get(alpha3)
            rows.append(
                {
                    "ranking_date": f"{year}-12-31",
                    "country_name": country_name,
                    "country_code": alpha2,
                    "country_iso3": alpha3,
                    "population": str(int(round(population))),
                    "yearly_change": "" if previous is None else str(int(round(population - previous))),
                    "season_summary": summary,
                    "data_source": source,
                }
            )
        previous_values = {alpha3: population for alpha3, _, _, population, _ in ranked}

    return rows


def write_csv(rows: list[dict[str, str]], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "ranking_date",
                "country_name",
                "country_code",
                "country_iso3",
                "population",
                "yearly_change",
                "season_summary",
                "data_source",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def download_top_flags(rows: list[dict[str, str]], flags_dir: Path, top_n: int) -> list[Path]:
    by_date: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_date[row["ranking_date"]].append(row)

    alpha2_codes: set[str] = set()
    for year_rows in by_date.values():
        ranked = sorted(year_rows, key=lambda row: -float(row["population"]))[:top_n]
        alpha2_codes.update(row["country_code"].lower() for row in ranked if row["country_code"])

    downloaded: list[Path] = []
    flags_dir.mkdir(parents=True, exist_ok=True)
    for alpha2 in sorted(alpha2_codes):
        output_path = flags_dir / f"{alpha2}.png"
        if output_path.exists():
            continue
        try:
            _download(FLAG_URL_TEMPLATE.format(alpha2=alpha2), output_path)
            downloaded.append(output_path)
        except Exception as error:
            print(f"[scraper] flag download skipped for {alpha2}: {error}")
    return downloaded


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the official 1960-2024 World Bank population timeseries.")
    parser.add_argument("--world-bank-json", type=Path, default=DEFAULT_WORLD_BANK_JSON)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--flags-dir", type=Path, default=DEFAULT_FLAGS_DIR)
    parser.add_argument("--start-year", type=int, default=START_YEAR)
    parser.add_argument("--end-year", type=int, default=END_YEAR)
    parser.add_argument("--top-n-assets", type=int, default=12)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--skip-flags", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.start_year > args.end_year:
        raise ValueError("--start-year must be less than or equal to --end-year.")

    metadata_path = _ensure_json(WORLD_BANK_METADATA_URL, args.metadata, args.refresh)
    world_bank_path = _ensure_json(WORLD_BANK_POPULATION_URL, args.world_bank_json, args.refresh)
    countries = _load_country_metadata(metadata_path)
    values = _load_world_bank_values(world_bank_path, countries)
    rows = build_rows(values, args.start_year, args.end_year)
    output = write_csv(rows, args.output)
    downloaded_flags = [] if args.skip_flags else download_top_flags(rows, args.flags_dir, args.top_n_assets)

    print(f"[scraper] World population CSV generated -> {output}")
    print(f"[scraper] {len(rows)} rows, {args.start_year}-{args.end_year}, {len(downloaded_flags)} flags downloaded")


if __name__ == "__main__":
    main()
