from __future__ import annotations

import argparse
import csv
import html
import re
import urllib.request
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "cycling"
    / "tour_de_france"
    / "tour_de_france_stage_wins_postwar_1947_2025.csv"
)

START_YEAR = 1947
END_YEAR = 2025
USER_AGENT = "Mozilla/5.0 (compatible; Codex Tour de France builder)"
PAGE_TEMPLATE = "https://en.wikipedia.org/wiki/{year}_Tour_de_France"


def _fetch_page(year: int) -> str:
    request = urllib.request.Request(PAGE_TEMPLATE.format(year=year), headers={"User-Agent": USER_AGENT})
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


def _parse_winner(winner: str) -> tuple[str, str] | None:
    winner = winner.strip()
    if not winner:
        return None
    if "not contested" in winner.lower():
        return None
    match = re.match(r"^(.*?)(?:\s*\(([^()]+)\))?$", winner)
    if not match:
        return None
    name = re.sub(r"\s+", " ", match.group(1)).strip(" -*")
    country = (match.group(2) or "").strip().upper()
    if not name:
        return None
    return name, country


def _extract_stage_winners(page_html: str) -> list[tuple[str, str]]:
    tables = re.findall(r'<table[^>]*class="[^"]*wikitable[^"]*"[^>]*>.*?</table>', page_html, flags=re.S | re.I)
    for table_html in tables:
        rows = re.findall(r"<tr[^>]*>(.*?)</tr>", table_html, flags=re.S | re.I)
        header: list[str] = []
        for row_html in rows[:3]:
            if "<th" not in row_html.lower():
                continue
            header = [_strip_tags(cell_html) for cell_html in re.findall(r"<t[hd][^>]*>(.*?)</t[hd]>", row_html, flags=re.S | re.I)]
            break
        if not header or "Stage" not in header or "Winner" not in header:
            continue

        winners: list[tuple[str, str]] = []
        for row_html in rows[1:]:
            cells = [_strip_tags(cell_html) for cell_html in re.findall(r"<t[hd][^>]*>(.*?)</t[hd]>", row_html, flags=re.S | re.I)]
            if len(cells) < 2:
                continue

            stage = cells[0].strip()
            winner_text = cells[-1].strip()
            stage_type = cells[-2].strip() if len(cells) >= 2 else ""

            if not stage or not winner_text:
                continue
            if "team time trial" in stage_type.lower():
                continue
            if winner_text.lower().startswith("team "):
                continue

            parsed = _parse_winner(winner_text)
            if parsed is None:
                continue
            winners.append(parsed)

        return winners

    raise RuntimeError("No Tour de France stage table could be found on the page.")


def build_rows(start_year: int = START_YEAR, end_year: int = END_YEAR) -> list[dict[str, str]]:
    cumulative_counts: dict[str, int] = defaultdict(int)
    country_by_rider: dict[str, str] = {}
    rows: list[dict[str, str]] = []

    for year in range(start_year, end_year + 1):
        stage_winners = _extract_stage_winners(_fetch_page(year))
        year_counts: dict[str, int] = defaultdict(int)

        for rider_name, country_code in stage_winners:
            cumulative_counts[rider_name] += 1
            year_counts[rider_name] += 1
            if country_code and not country_by_rider.get(rider_name):
                country_by_rider[rider_name] = country_code

        top_year_winners = sorted(year_counts.items(), key=lambda item: (-item[1], item[0]))[:3]
        summary_parts = ["Top stage winners"]
        summary_parts.extend(f"{name} x{count}" for name, count in top_year_winners)
        season_summary = "|".join(summary_parts)

        for rider_name, points in sorted(cumulative_counts.items(), key=lambda item: (-item[1], item[0])):
            rows.append(
                {
                    "ranking_date": f"{year}-12-31",
                    "player_name": rider_name,
                    "country_code": country_by_rider.get(rider_name, ""),
                    "points": str(points),
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
    parser = argparse.ArgumentParser(description="Build post-war Tour de France stage wins timeseries CSV.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--start-year", type=int, default=START_YEAR)
    parser.add_argument("--end-year", type=int, default=END_YEAR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = build_rows(start_year=args.start_year, end_year=args.end_year)
    output = write_csv(rows, args.output)
    print(f"[scraper] Tour de France post-war stage wins CSV generated -> {output}")


if __name__ == "__main__":
    main()
