from __future__ import annotations

import argparse
import csv
import re
from html import unescape
from pathlib import Path
from urllib.request import Request, urlopen


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "basketball" / "nba_points_by_age_lebron_jordan_kobe.csv"

PLAYERS = {
    "LeBron James": "jamesle01",
    "Michael Jordan": "jordami01",
    "Kobe Bryant": "bryanko01",
}


def _fetch_player_page(slug: str) -> str:
    url = f"https://www.basketball-reference.com/players/{slug[0]}/{slug}.html"
    request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(request, timeout=30) as response:
        return response.read().decode("utf-8", "replace")


def _extract_totals_table(html: str) -> str:
    match = re.search(r'<table[^>]+id="totals_stats"[^>]*>(.*?)</table>', html, flags=re.S)
    if not match:
        raise RuntimeError("Could not find totals_stats table.")
    return match.group(1)


def _cell(row_html: str, stat: str) -> str:
    match = re.search(rf'<(?:td|th)[^>]+data-stat="{re.escape(stat)}"[^>]*>(.*?)</(?:td|th)>', row_html, flags=re.S)
    if not match:
        return ""
    value = re.sub(r"<.*?>", "", match.group(1))
    return unescape(value).strip()


def _season_points_by_age(slug: str) -> dict[int, int]:
    table = _extract_totals_table(_fetch_player_page(slug))
    by_age: dict[int, int] = {}
    for row_match in re.finditer(r"<tr[^>]*>(.*?)</tr>", table, flags=re.S):
        row = row_match.group(1)
        row_class = row_match.group(0)
        if "thead" in row_class:
            continue
        age_text = _cell(row, "age")
        team = _cell(row, "team_name_abbr")
        pts_text = _cell(row, "pts")
        season = _cell(row, "year_id")
        if not age_text or not pts_text or not season:
            continue
        if team == "TOT":
            continue
        try:
            age = int(age_text)
            pts = int(pts_text.replace(",", ""))
        except ValueError:
            continue
        by_age[age] = by_age.get(age, 0) + pts
    return by_age


def build_rows() -> list[dict[str, str | int]]:
    season_by_player = {player: _season_points_by_age(slug) for player, slug in PLAYERS.items()}
    min_age = min(min(values) for values in season_by_player.values())
    max_age = max(max(values) for values in season_by_player.values())

    rows: list[dict[str, str | int]] = []
    for player, values in season_by_player.items():
        cumulative = 0
        for age in range(min_age, max_age + 1):
            season_points = values.get(age, 0)
            cumulative += season_points
            rows.append(
                {
                    "player": player,
                    "age": age,
                    "season_points": season_points,
                    "cumulative_points": cumulative,
                    "source": f"https://www.basketball-reference.com/players/{PLAYERS[player][0]}/{PLAYERS[player]}.html",
                }
            )
    return rows


def write_csv(rows: list[dict[str, str | int]], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["player", "age", "season_points", "cumulative_points", "source"],
        )
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build LeBron/Jordan/Kobe NBA regular season points by age CSV.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = write_csv(build_rows(), args.output)
    print(f"[scraper] wrote NBA points by age CSV -> {output}")


if __name__ == "__main__":
    main()
