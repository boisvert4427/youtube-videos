from __future__ import annotations

import argparse
import csv
import html
import re
import urllib.request
from collections import defaultdict
from pathlib import Path

from bs4 import BeautifulSoup


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "cycling"
    / "tour_de_france"
    / "tour_de_france_yellow_jersey_days_postwar_1947_2025.csv"
)

START_YEAR = 1947
END_YEAR = 2025
USER_AGENT = "Mozilla/5.0 (compatible; Codex Tour de France builder)"
PAGE_TEMPLATE = "https://en.wikipedia.org/wiki/{year}_Tour_de_France"
STATS_URL = "https://en.wikipedia.org/wiki/Yellow_jersey_statistics"
STAGE_WINS_CSV = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "cycling"
    / "tour_de_france"
    / "tour_de_france_stage_wins_postwar_1947_2025.csv"
)


COUNTRY_NAME_TO_ALPHA2 = {
    "Australia": "au",
    "Austria": "at",
    "Belgium": "be",
    "Canada": "ca",
    "Colombia": "co",
    "Denmark": "dk",
    "Ecuador": "ec",
    "Estonia": "ee",
    "France": "fr",
    "Germany": "de",
    "Ireland": "ie",
    "Italy": "it",
    "Luxembourg": "lu",
    "Netherlands": "nl",
    "Norway": "no",
    "Poland": "pl",
    "Portugal": "pt",
    "Russia": "ru",
    "Slovakia": "sk",
    "Slovenia": "si",
    "South Africa": "za",
    "Spain": "es",
    "Switzerland": "ch",
    "Ukraine": "ua",
    "United Kingdom": "gb",
    "United States": "us",
}


def _fetch_page(url: str) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read().decode("utf-8")


def _strip_tags(text: str) -> str:
    text = re.sub(r"<sup[^>]*>.*?</sup>", "", text, flags=re.S)
    text = re.sub(r"<span[^>]*class=\"mw-ref\"[^>]*>.*?</span>", "", text, flags=re.S)
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    text = re.sub(r"\[[^\]]*\]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _clean_cell_text(text: str) -> str:
    text = html.unescape(text)
    text = re.sub(r"\s*\[\s*[a-z0-9]+\s*\]", "", text, flags=re.I)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _canonical_country_code(country_name: str) -> str:
    return COUNTRY_NAME_TO_ALPHA2.get(country_name.strip(), "")


def _parse_rider_country_map() -> dict[str, str]:
    soup = BeautifulSoup(_fetch_page(STATS_URL), "lxml")
    tables = soup.select("table.wikitable")
    if len(tables) < 2:
        raise RuntimeError("Could not find yellow jersey statistics table.")

    rider_map: dict[str, str] = {}
    for row in tables[1].select("tr")[1:]:
        cells = [cell.get_text(" ", strip=True) for cell in row.find_all(["th", "td"])]
        if len(cells) < 3:
            continue
        name = _clean_cell_text(cells[1])
        country = _clean_cell_text(cells[2])
        if not name:
            continue
        rider_map[name] = _canonical_country_code(country)

    if STAGE_WINS_CSV.exists():
        with STAGE_WINS_CSV.open("r", newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            for row in reader:
                name = _clean_cell_text(row.get("player_name", ""))
                country_code = _clean_cell_text(row.get("country_code", ""))
                if name and country_code and not rider_map.get(name):
                    rider_map[name] = country_code
    return rider_map


def _extract_table_rows(table) -> list[list[str]]:
    rows = table.select("tr")
    if not rows:
        return []

    header_cells = rows[0].find_all(["th", "td"])
    width = len(header_cells)
    active: list[dict[str, object] | None] = [None] * width
    parsed_rows: list[list[str]] = []

    for row in rows[1:]:
        cells = row.find_all(["th", "td"])
        cell_index = 0
        parsed: list[str] = []
        col = 0
        while col < width:
            carried = active[col]
            if carried is not None:
                parsed.append(str(carried["text"]))
                remaining = int(carried["remaining"]) - 1
                active[col] = None if remaining <= 0 else {"text": carried["text"], "remaining": remaining}
                col += 1
                continue

            if cell_index >= len(cells):
                parsed.append("")
                col += 1
                continue

            cell = cells[cell_index]
            cell_index += 1
            text = _clean_cell_text(cell.get_text(" ", strip=True))
            rowspan = int(cell.get("rowspan", 1))
            colspan = int(cell.get("colspan", 1))
            for offset in range(colspan):
                parsed.append(text)
                if rowspan > 1:
                    active[col + offset] = {"text": text, "remaining": rowspan - 1}
            col += colspan

        parsed_rows.append(parsed)

    return parsed_rows


def _extract_classification_leaders(page_html: str) -> list[str]:
    soup = BeautifulSoup(page_html, "lxml")
    for table in soup.select("table.wikitable"):
        caption = table.find("caption")
        caption_text = caption.get_text(" ", strip=True) if caption else ""
        if "Classification leadership" not in caption_text:
            continue
        leaders: list[str] = []
        for row in table.select("tr")[1:]:
            for cell in row.find_all(["th", "td"]):
                style = cell.get("style", "").replace(" ", "").lower()
                if "background:#ffeb64" in style:
                    gc_leader = _clean_cell_text(cell.get_text(" ", strip=True))
                    rowspan = int(cell.get("rowspan", 1))
                    if "not awarded" in gc_leader.lower():
                        break
                    leaders.extend([gc_leader] * max(1, rowspan))
                    break
        if leaders:
            return leaders
    raise RuntimeError("No classification leadership table could be found on the page.")


def build_rows(start_year: int = START_YEAR, end_year: int = END_YEAR) -> list[dict[str, str]]:
    rider_country = _parse_rider_country_map()
    cumulative_counts: dict[str, int] = defaultdict(int)
    country_by_rider: dict[str, str] = {}
    rows: list[dict[str, str]] = []

    for year in range(start_year, end_year + 1):
        page_html = _fetch_page(PAGE_TEMPLATE.format(year=year))
        leaders = _extract_classification_leaders(page_html)

        year_counts: dict[str, int] = defaultdict(int)
        for rider_name in leaders:
            cumulative_counts[rider_name] += 1
            year_counts[rider_name] += 1
            if rider_name not in country_by_rider:
                country_by_rider[rider_name] = rider_country.get(rider_name, "")

        top_year = sorted(year_counts.items(), key=lambda item: (-item[1], item[0]))[:3]
        summary_parts = ["Top yellow jersey days"]
        summary_parts.extend(f"{name} x{count}" for name, count in top_year)
        season_summary = "|".join(summary_parts)

        for rider_name, days in sorted(cumulative_counts.items(), key=lambda item: (-item[1], item[0])):
            rows.append(
                {
                    "ranking_date": f"{year}-12-31",
                    "player_name": rider_name,
                    "country_code": country_by_rider.get(rider_name, ""),
                    "points": str(days),
                    "season_summary": season_summary,
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
    parser = argparse.ArgumentParser(description="Build post-war Tour de France yellow jersey days timeseries CSV.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--start-year", type=int, default=START_YEAR)
    parser.add_argument("--end-year", type=int, default=END_YEAR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = build_rows(start_year=args.start_year, end_year=args.end_year)
    output = write_csv(rows, args.output)
    print(f"[scraper] Tour de France yellow jersey days CSV generated -> {output}")


if __name__ == "__main__":
    main()
