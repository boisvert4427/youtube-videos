from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

from scraper.tennis.build_grand_slam_titles_timeseries_csv import PLAYER_COUNTRIES, SEASONS


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "tennis" / "roland_garros_titles_top12_cards.csv"

ISO3_TO_ISO2 = {
    "ARG": "ar",
    "AUS": "au",
    "AUT": "at",
    "BRA": "br",
    "CRO": "hr",
    "CZE": "cz",
    "ECU": "ec",
    "ESP": "es",
    "FRA": "fr",
    "GBR": "gb",
    "GER": "de",
    "ITA": "it",
    "NED": "nl",
    "ROU": "ro",
    "RSA": "za",
    "RUS": "ru",
    "SRB": "rs",
    "SUI": "ch",
    "SWE": "se",
    "USA": "us",
}

CARD_COLORS = [
    ("#26150f", "#f6b15c"),
    ("#1b2d25", "#ffd87a"),
    ("#341f14", "#ff9b5f"),
    ("#152521", "#8de1c0"),
    ("#2d1e12", "#ffd06a"),
    ("#202739", "#96baff"),
    ("#30191f", "#ffb2cf"),
    ("#1d2f37", "#84e7ff"),
    ("#3c2214", "#f3a862"),
    ("#16221f", "#b7f47e"),
    ("#241b34", "#c7adff"),
    ("#212a35", "#f1d7a6"),
]


def _country_code(player_name: str) -> str:
    iso3 = PLAYER_COUNTRIES.get(player_name, "")
    return ISO3_TO_ISO2.get(iso3, "")


def build_rows(top_n: int) -> list[dict[str, str]]:
    wins_by_player: dict[str, list[int]] = defaultdict(list)
    for year, winners in SEASONS:
        winner = winners.get("RG", "")
        if winner:
            wins_by_player[winner].append(year)

    ordered = sorted(
        wins_by_player.items(),
        key=lambda item: (
            len(item[1]),
            max(item[1]),
            item[0],
        ),
    )[-top_n:]
    ordered.sort(
        key=lambda item: (
            len(item[1]),
            max(item[1]),
            item[0],
        )
    )

    rows: list[dict[str, str]] = []
    total = len(ordered)
    for idx, (player_name, years) in enumerate(ordered, start=1):
        sorted_years = sorted(years, reverse=True)
        bg_color, accent_color = CARD_COLORS[(idx - 1) % len(CARD_COLORS)]
        rows.append(
            {
                "rank": str(total - idx + 1),
                "player_name": player_name,
                "country_code": _country_code(player_name),
                "titles": str(len(years)),
                "years_won": " / ".join(str(year) for year in sorted_years),
                "first_title": str(min(years)),
                "last_title": str(max(years)),
                "badge_label": f"ROLAND-GARROS x{len(years)}",
                "card_bg_color": bg_color,
                "accent_color": accent_color,
            }
        )
    return rows


def run(output_csv: Path = DEFAULT_OUTPUT, top_n: int = 12) -> Path:
    rows = build_rows(top_n=top_n)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "rank",
                "player_name",
                "country_code",
                "titles",
                "years_won",
                "first_title",
                "last_title",
                "badge_label",
                "card_bg_color",
                "accent_color",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return output_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Roland-Garros top winners cards CSV.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--top-n", type=int, default=12)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    output = run(output_csv=args.output, top_n=args.top_n)
    print(f"[scraper] Roland-Garros titles cards CSV generated -> {output}")

