from __future__ import annotations

import argparse
import csv
import io
import json
import re
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "cycling" / "cycling_monuments_timeseries_1892_2025.csv"

API_URL = "https://en.wikipedia.org/w/api.php"
USER_AGENT = "youtube-videos/1.0 (https://example.com; contact: local-script)"
SUMMARY_ORDER = [
    ("msr", "MSR"),
    ("rvv", "RVV"),
    ("pr", "PR"),
    ("lbl", "LBL"),
    ("lom", "LOM"),
]
COLUMN_ALIASES = {
    "Year": "year",
    "Milan–San Remo": "msr",
    "Milan-San Remo": "msr",
    "Tour of Flanders": "rvv",
    "Paris–Roubaix": "pr",
    "Paris-Roubaix": "pr",
    "Liège–Bastogne–Liège": "lbl",
    "Liège-Bastogne-Liège": "lbl",
    "Ličge–Bastogne–Ličge": "lbl",
    "Ličge-Bastogne-Ličge": "lbl",
    "Giro di Lombardia": "lom",
    "Il Lombardia": "lom",
}


def _fetch_json(params: dict[str, str]) -> dict:
    query = urllib.parse.urlencode(params)
    request = urllib.request.Request(f"{API_URL}?{query}", headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def _get_section_index(page_title: str, section_name: str) -> int:
    payload = _fetch_json(
        {
            "action": "parse",
            "page": page_title,
            "prop": "sections",
            "format": "json",
            "redirects": "1",
        }
    )
    sections = payload.get("parse", {}).get("sections", [])
    for section in sections:
        if section.get("line", "").strip().lower() == section_name.strip().lower():
            return int(section["index"])
    raise RuntimeError(f"Section '{section_name}' not found on page '{page_title}'.")


def _fetch_section_html(page_title: str, section_index: int) -> str:
    payload = _fetch_json(
        {
            "action": "parse",
            "page": page_title,
            "prop": "text",
            "section": str(section_index),
            "format": "json",
            "redirects": "1",
        }
    )
    return payload["parse"]["text"]["*"]


def _normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    renamed = {}
    for column in frame.columns:
        label = str(column).strip()
        normalized = COLUMN_ALIASES.get(label)
        if normalized:
            renamed[column] = normalized
    frame = frame.rename(columns=renamed)
    required = ["year", "msr", "rvv", "pr", "lbl", "lom"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise RuntimeError(f"Missing expected columns in monuments table: {missing}")
    return frame[required].copy()


def _extract_rider_name(cell: object) -> tuple[str | None, str]:
    raw = "" if cell is None else str(cell)
    raw = raw.replace("\xa0", " ").strip()
    if not raw or "not contested" in raw.lower():
        return None, ""
    raw = re.sub(r"\[[^\]]+\]", "", raw).strip()
    country_match = re.search(r"\(([A-Z]{3})\)", raw)
    country_code = country_match.group(1) if country_match else ""
    rider_name = re.sub(r"\([A-Z]{3}\)", "", raw)
    rider_name = re.sub(r"\(\d+/\d+\)", "", rider_name)
    rider_name = re.sub(r"\(\d+\)", "", rider_name)
    rider_name = re.sub(r"\s+", " ", rider_name).strip(" -*")
    if not rider_name:
        return None, country_code
    return rider_name, country_code


def _parse_monuments_table() -> pd.DataFrame:
    section_index = _get_section_index("Cycling monument", "Monuments winners")
    html = _fetch_section_html("Cycling monument", section_index)
    frames = pd.read_html(io.StringIO(html))
    if not frames:
        raise RuntimeError("No tables found in Cycling monument winners section.")
    return _normalize_columns(frames[0])


def build_rows(frame: pd.DataFrame, start_year: int | None = None, end_year: int | None = None) -> list[dict[str, str]]:
    monuments = [key for key, _ in SUMMARY_ORDER]
    cumulative_titles: dict[str, int] = defaultdict(int)
    rider_country: dict[str, str] = {}
    rows: list[dict[str, str]] = []

    frame["year"] = frame["year"].astype(str).str.extract(r"(\d{4})").astype(int)
    frame = frame.sort_values("year")
    if start_year is not None:
        frame = frame[frame["year"] >= start_year]
    if end_year is not None:
        frame = frame[frame["year"] <= end_year]

    for row in frame.itertuples(index=False):
        summary_parts: list[str] = []
        year = int(row.year)
        for key, label in SUMMARY_ORDER:
            rider_name, country_code = _extract_rider_name(getattr(row, key))
            if rider_name is None:
                continue
            cumulative_titles[rider_name] += 1
            if country_code and not rider_country.get(rider_name):
                rider_country[rider_name] = country_code
            summary_parts.append(f"{label} {rider_name}")

        for rider_name, points in sorted(cumulative_titles.items(), key=lambda item: (-item[1], item[0])):
            rows.append(
                {
                    "ranking_date": f"{year}-12-31",
                    "player_name": rider_name,
                    "country_code": rider_country.get(rider_name, ""),
                    "points": str(points),
                    "season_summary": "|".join(summary_parts),
                }
            )
    return rows


def write_csv(rows: list[dict[str, str]], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["ranking_date", "player_name", "country_code", "points", "season_summary"],
        )
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build cumulative cycling Monuments wins timeseries CSV.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--start-year", type=int, default=None)
    parser.add_argument("--end-year", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    table = _parse_monuments_table()
    rows = build_rows(table, start_year=args.start_year, end_year=args.end_year)
    output = write_csv(rows, args.output)
    print(f"[scraper] Cycling Monuments timeseries generated -> {output}")


if __name__ == "__main__":
    main()
