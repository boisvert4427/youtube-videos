from __future__ import annotations

import argparse
import csv
import html
import re
import time
import urllib.request
from pathlib import Path

from bs4 import BeautifulSoup


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "cycling" / "tour_de_france" / "tour_de_france_through_the_years_1947_2025.csv"
START_YEAR = 1947
END_YEAR = 2025
USER_AGENT = "Mozilla/5.0 (compatible; Codex Tour de France builder)"
PAGE_TEMPLATE = "https://en.wikipedia.org/wiki/{year}_Tour_de_France"

COUNTRY_NAME_TO_ALPHA3 = {
    "Australia": "AUS",
    "Austria": "AUT",
    "Belgium": "BEL",
    "Canada": "CAN",
    "Colombia": "COL",
    "Czech Republic": "CZE",
    "Czechoslovakia": "TCH",
    "Denmark": "DEN",
    "Ecuador": "ECU",
    "Estonia": "EST",
    "France": "FRA",
    "Germany": "GER",
    "Great Britain": "GBR",
    "Ireland": "IRL",
    "Italy": "ITA",
    "Luxembourg": "LUX",
    "Netherlands": "NED",
    "Norway": "NOR",
    "Poland": "POL",
    "Portugal": "POR",
    "Russia": "RUS",
    "Slovakia": "SVK",
    "Slovenia": "SLO",
    "South Africa": "RSA",
    "Spain": "ESP",
    "Switzerland": "SUI",
    "Ukraine": "UKR",
    "United Kingdom": "GBR",
    "United States": "USA",
    "West Germany": "FRG",
}

FIELDNAMES = [
    "year",
    "winner_name",
    "winner_country",
    "winner_team",
    "winner_time",
    "gc2_name",
    "gc2_country",
    "gc2_team",
    "gc2_gap",
    "gc3_name",
    "gc3_country",
    "gc3_team",
    "gc3_gap",
    "points_name",
    "points_country",
    "points_team",
    "mountains_name",
    "mountains_country",
    "mountains_team",
    "badge_label",
    "card_bg_color",
    "accent_color",
]


def _fetch_page(url: str) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read().decode("utf-8", "ignore")


def _clean_text(text: str) -> str:
    text = html.unescape(text or "")
    text = re.sub(r"<ref[^>]*>.*?</ref>", "", text, flags=re.S)
    text = re.sub(r"<ref[^/]*/>", "", text)
    text = re.sub(r"\[[^\]]*\]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _parse_name_country(raw: str) -> tuple[str, str]:
    text = _clean_text(raw)
    match = re.match(r"^(.*?)\s*\(\s*([A-Za-z]{2,3})\s*\)$", text)
    if not match:
        return text, ""
    name = _clean_text(match.group(1))
    country = match.group(2).upper()
    if len(country) == 2:
        return name, country
    return name, country


def _normalize_country(country: str) -> str:
    raw = _clean_text(country).strip()
    if not raw:
        return ""
    raw = raw.replace(".", "").replace("(", "").replace(")", "")
    if len(raw) == 3 and raw.isalpha():
        return raw.upper()
    if len(raw) == 2 and raw.isalpha():
        return raw.upper()
    return COUNTRY_NAME_TO_ALPHA3.get(raw, raw.upper())


def _extract_section_table(soup: BeautifulSoup, section_id: str):
    node = soup.find(id=section_id)
    if node is None:
        return None
    parent = node.parent
    if parent is None:
        return None
    return parent.find_next_sibling("table")


def _parse_ranked_table(table, limit: int | None) -> list[dict[str, str]]:
    if table is None:
        return []
    rows = table.select("tr")
    if len(rows) < 2:
        return []
    parsed: list[dict[str, str]] = []
    for row in rows[1:]:
        cells = [cell.get_text(" ", strip=True) for cell in row.find_all(["th", "td"])]
        if len(cells) < 2:
            continue
        rank_text = _clean_text(cells[0])
        if not rank_text.isdigit():
            continue
        rider_name, rider_country = _parse_name_country(cells[1])
        team = _clean_text(cells[2]) if len(cells) > 2 else ""
        value = _clean_text(cells[3]) if len(cells) > 3 else ""
        parsed.append(
            {
                "rank": rank_text,
                "name": rider_name,
                "country": _normalize_country(rider_country),
                "team": team,
                "value": value,
            }
        )
        if limit is not None and len(parsed) >= limit:
            break
    return parsed


def _build_year_row(year: int) -> dict[str, str]:
    html_text = _fetch_page(PAGE_TEMPLATE.format(year=year))
    soup = BeautifulSoup(html_text, "lxml")

    gc_rows = _parse_ranked_table(_extract_section_table(soup, "General_classification"), limit=3)
    points_rows = _parse_ranked_table(_extract_section_table(soup, "Points_classification"), limit=1)
    mountains_rows = _parse_ranked_table(_extract_section_table(soup, "Mountains_classification"), limit=1)

    winner = gc_rows[0] if gc_rows else {"name": "", "country": "", "team": "", "value": ""}
    gc2 = gc_rows[1] if len(gc_rows) > 1 else {"name": "", "country": "", "team": "", "value": ""}
    gc3 = gc_rows[2] if len(gc_rows) > 2 else {"name": "", "country": "", "team": "", "value": ""}
    points = points_rows[0] if points_rows else {"name": "", "country": "", "team": "", "value": ""}
    mountains = mountains_rows[0] if mountains_rows else {"name": "", "country": "", "team": "", "value": ""}

    return {
        "year": str(year),
        "winner_name": winner["name"],
        "winner_country": winner["country"],
        "winner_team": winner["team"],
        "winner_time": winner["value"],
        "gc2_name": gc2["name"],
        "gc2_country": gc2["country"],
        "gc2_team": gc2["team"],
        "gc2_gap": gc2["value"],
        "gc3_name": gc3["name"],
        "gc3_country": gc3["country"],
        "gc3_team": gc3["team"],
        "gc3_gap": gc3["value"],
        "points_name": points["name"],
        "points_country": points["country"],
        "points_team": points["team"],
        "mountains_name": mountains["name"],
        "mountains_country": mountains["country"],
        "mountains_team": mountains["team"],
        "badge_label": "WINNER",
        "card_bg_color": "#050505",
        "accent_color": "#f4c319",
    }


def build_rows(start_year: int = START_YEAR, end_year: int = END_YEAR) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for year in range(start_year, end_year + 1):
        rows.append(_build_year_row(year))
        time.sleep(0.1)
    return rows


def write_csv(rows: list[dict[str, str]], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a Tour de France through-the-years cards CSV.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--start-year", type=int, default=START_YEAR)
    parser.add_argument("--end-year", type=int, default=END_YEAR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = build_rows(start_year=args.start_year, end_year=args.end_year)
    output = write_csv(rows, args.output)
    print(f"[scraper] Tour de France through-the-years CSV generated -> {output}")


if __name__ == "__main__":
    main()
