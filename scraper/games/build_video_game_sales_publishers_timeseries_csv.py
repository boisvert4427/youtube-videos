from __future__ import annotations

import argparse
import csv
import io
import re
from collections import defaultdict
from pathlib import Path

import requests


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_RAW_CSV = PROJECT_ROOT / "data" / "raw" / "video_game_sales" / "vgsales.csv"
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "video_game_sales"
    / "video_game_sales_publishers_1980_2017.csv"
)

SOURCE_URL = "https://raw.githubusercontent.com/raghav-19/Video-Games-Sales-Data-Analysis/master/vgsales.csv"
DATA_SOURCE = "Kaggle Video Game Sales"
START_YEAR = 1980
END_YEAR = 2017
DEFAULT_LOGOS_DIR = PROJECT_ROOT / "data" / "raw" / "video_game_sales" / "logos"

LOGO_DOMAINS: dict[str, str] = {
    "activision": "activision.com",
    "atari": "atari.com",
    "bethesda_softworks": "bethesda.net",
    "capcom": "capcom.com",
    "disney_interactive_studios": "disney.com",
    "eidos_interactive": "eidosinteractive.com",
    "electronic_arts": "ea.com",
    "konami_digital_entertainment": "konami.com",
    "lucasarts": "lucasfilm.com",
    "mattel_interactive": "mattel.com",
    "microsoft_game_studios": "microsoft.com",
    "namco_bandai_games": "bandainamcoent.com",
    "nintendo": "nintendo.com",
    "sony_computer_entertainment": "playstation.com",
    "square_enix": "square-enix.com",
    "take_two_interactive": "take2games.com",
    "ubisoft": "ubisoft.com",
    "warner_bros_interactive_entertainment": "wbgames.com",
    "sega": "sega.com",
    "acclaim_entertainment": "acclaim.com",
    "enix_corporation": "square-enix.com",
    "squaresoft": "square-enix.com",
    "virgin_interactive": "virgin.com",
    "parker_bros": "hasbro.com",
    "coleco": "coleco.com",
    "hudson_soft": "hudsonsoft.jp",
    "midway_games": "midway.com",
    "thq": "thq.com",
    "20th_century_fox_video_games": "20thcenturystudios.com",
}


def _normalize_key(value: str) -> str:
    key = value.strip().lower()
    key = re.sub(r"[^a-z0-9]+", "_", key)
    return re.sub(r"_+", "_", key).strip("_") or "unknown"


def _parse_year(raw_year: str) -> int | None:
    raw_year = raw_year.strip()
    if not raw_year or raw_year.lower() == "n/a":
        return None
    try:
        year = int(float(raw_year))
    except ValueError:
        return None
    if year < START_YEAR or year > END_YEAR:
        return None
    return year


def _parse_sales(raw_value: str) -> float:
    try:
        return float(raw_value.strip())
    except ValueError:
        return 0.0


def _download_file(url: str, output_path: Path) -> None:
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    output_path.write_bytes(response.content)


def download_raw_csv(raw_csv: Path, refresh: bool = False) -> Path:
    raw_csv.parent.mkdir(parents=True, exist_ok=True)
    if raw_csv.exists() and not refresh:
        return raw_csv
    response = requests.get(SOURCE_URL, timeout=60)
    response.raise_for_status()
    raw_csv.write_text(response.text, encoding="utf-8")
    return raw_csv


def download_logos(logos_dir: Path, refresh: bool) -> list[Path]:
    logos_dir.mkdir(parents=True, exist_ok=True)
    downloaded: list[Path] = []
    for key, domain in LOGO_DOMAINS.items():
        output_path = logos_dir / f"{key}.png"
        if output_path.exists() and not refresh:
            continue
        try:
            url = f"https://www.google.com/s2/favicons?domain={domain}&sz=128"
            _download_file(url, output_path)
            downloaded.append(output_path)
        except Exception as error:
            print(f"[scraper] logo download skipped for {key}: {error}")
    return downloaded


def build_timeseries(raw_csv: Path, output_csv: Path) -> list[dict[str, str]]:
    with raw_csv.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        rows = list(reader)

    records_by_year: dict[int, list[tuple[str, float]]] = defaultdict(list)
    for row in rows:
        year = _parse_year(row.get("Year", ""))
        publisher = row.get("Publisher", "").strip()
        if year is None or not publisher or publisher.lower() == "unknown":
            continue
        sales = _parse_sales(row.get("Global_Sales", "0"))
        if sales <= 0:
            continue
        records_by_year[year].append((publisher, sales))

    cumulative: defaultdict[str, float] = defaultdict(float)
    previous_snapshot: dict[str, float] = {}
    output_rows: list[dict[str, str]] = []

    for year in range(START_YEAR, END_YEAR + 1):
        for publisher, sales in records_by_year.get(year, []):
            cumulative[publisher] += sales

        current_sorted = sorted(
            cumulative.items(),
            key=lambda item: (-item[1], item[0].lower()),
        )
        current_snapshot = dict(current_sorted)
        if not current_sorted:
            continue

        top_publisher, top_sales = current_sorted[0]
        summary = (
            f"Top publisher: {top_publisher} | "
            f"cumulative global sales: {top_sales:.1f}M | "
            f"{len(current_sorted)} publishers tracked"
        )
        ranking_date = f"{year}-01-01"

        for publisher, sales in current_sorted:
            previous_sales = previous_snapshot.get(publisher, 0.0)
            output_rows.append(
                {
                    "ranking_date": ranking_date,
                    "browser_name": publisher,
                    "browser_key": _normalize_key(publisher),
                    "market_share": f"{sales:.3f}",
                    "monthly_change": f"{sales - previous_sales:.3f}",
                    "season_summary": summary,
                    "data_source": DATA_SOURCE,
                    "source_url": SOURCE_URL,
                }
            )

        previous_snapshot = current_snapshot

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "ranking_date",
                "browser_name",
                "browser_key",
                "market_share",
                "monthly_change",
                "season_summary",
                "data_source",
                "source_url",
            ],
        )
        writer.writeheader()
        writer.writerows(output_rows)

    return output_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a cumulative video game publisher sales timeseries CSV.")
    parser.add_argument("--raw-csv", type=Path, default=DEFAULT_RAW_CSV)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--logos-dir", type=Path, default=DEFAULT_LOGOS_DIR)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--skip-logos", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_csv = download_raw_csv(args.raw_csv, refresh=args.refresh)
    rows = build_timeseries(raw_csv, args.output)
    downloaded = [] if args.skip_logos else download_logos(args.logos_dir, args.refresh)
    years = sorted({row["ranking_date"] for row in rows})
    print(f"[scraper] Video game sales CSV generated -> {args.output}")
    print(
        f"[scraper] Rows: {len(rows)} | snapshots: {len(years)} | first: {years[0]} | "
        f"last: {years[-1]} | {len(downloaded)} logos downloaded"
    )


if __name__ == "__main__":
    main()
