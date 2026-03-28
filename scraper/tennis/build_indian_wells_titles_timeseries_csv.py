from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_INPUT = PROJECT_ROOT / "data" / "processed" / "atp_vertical_timeline_indian_wells_winners_1976_2025.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "tennis" / "indian_wells_titles_timeseries_1976_2025.csv"

PLAYER_COUNTRIES = {
    "Alex Corretja": "ESP",
    "Andre Agassi": "USA",
    "Boris Becker": "GER",
    "Brian Gottfried": "USA",
    "Carlos Alcaraz": "ESP",
    "Cameron Norrie": "GBR",
    "Dominic Thiem": "AUT",
    "Hubert Hurkacz": "POL",
    "Ivan Ljubicic": "CRO",
    "Jim Courier": "USA",
    "Jimmy Connors": "USA",
    "Joakim Nystrom": "SWE",
    "John Isner": "USA",
    "Jose Higueras": "ESP",
    "Juan Martin del Potro": "ARG",
    "Larry Stefanki": "USA",
    "Lleyton Hewitt": "AUS",
    "Marcelo Rios": "CHI",
    "Mark Philippoussis": "AUS",
    "Michael Chang": "USA",
    "Miloslav Mecir Sr.": "SVK",
    "Novak Djokovic": "SRB",
    "Pete Sampras": "USA",
    "Rafael Nadal": "ESP",
    "Roger Federer": "SUI",
    "Roscoe Tanner": "USA",
    "Stefan Edberg": "SWE",
    "Taylor Fritz": "USA",
    "Yannick Noah": "FRA",
}


@dataclass(frozen=True)
class SnapshotRow:
    ranking_date: str
    player_name: str
    country_code: str
    points: int
    season_summary: str


def _extract_final_summary(player_name: str, results: str) -> str:
    for part in reversed([item.strip() for item in results.split("|") if item.strip()]):
        if not part.startswith("F "):
            continue
        payload = part[2:].strip()
        if " " not in payload:
            return player_name
        score_start = payload.find(" ")
        opponent = payload[:score_start].strip()
        score = payload[score_start:].strip()
        return f"{player_name} def. {opponent} | {score}"
    return player_name


def build_rows(input_csv: Path) -> list[SnapshotRow]:
    champions_by_year: dict[int, tuple[str, str]] = {}
    with input_csv.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            year = int(row["year"])
            player_name = row["player_name"].strip()
            results = row.get("results", "").strip()
            champions_by_year[year] = (player_name, _extract_final_summary(player_name, results))

    cumulative: defaultdict[str, int] = defaultdict(int)
    rows: list[SnapshotRow] = []
    for year in sorted(champions_by_year):
        player_name, season_summary = champions_by_year[year]
        cumulative[player_name] += 1
        ranking_date = f"{year}-03-31"
        ranked_players = sorted(cumulative.items(), key=lambda item: (-item[1], item[0]))
        for ranked_name, titles in ranked_players:
            rows.append(
                SnapshotRow(
                    ranking_date=ranking_date,
                    player_name=ranked_name,
                    country_code=PLAYER_COUNTRIES.get(ranked_name, ""),
                    points=titles,
                    season_summary=season_summary,
                )
            )
    return rows


def write_rows(rows: list[SnapshotRow], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["ranking_date", "player_name", "country_code", "points", "season_summary"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "ranking_date": row.ranking_date,
                    "player_name": row.player_name,
                    "country_code": row.country_code,
                    "points": row.points,
                    "season_summary": row.season_summary,
                }
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Indian Wells cumulative titles timeseries CSV.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = build_rows(args.input)
    write_rows(rows, args.output)
    print(f"[scraper] Indian Wells titles timeseries CSV generated -> {args.output}")


if __name__ == "__main__":
    main()
