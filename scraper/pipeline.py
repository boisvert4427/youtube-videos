from __future__ import annotations

import csv
from dataclasses import dataclass, asdict
from datetime import date
from pathlib import Path
from typing import Iterable, List


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_CSV = PROJECT_ROOT / "data" / "processed" / "player_game_stats_v1.csv"


@dataclass(frozen=True)
class PlayerGameStat:
    game_date: str
    league: str
    season: str
    team: str
    opponent: str
    player_name: str
    minutes: int
    points: int
    rebounds: int
    assists: int


def fetch_demo_stats() -> List[dict]:
    """Temporary local source to validate the CSV contract end-to-end."""
    today = date.today().isoformat()
    return [
        {
            "game_date": today,
            "league": "nba",
            "season": "2025-26",
            "team": "LAL",
            "opponent": "BOS",
            "player_name": "Player A",
            "minutes": 34,
            "points": 28,
            "rebounds": 8,
            "assists": 7,
        },
        {
            "game_date": today,
            "league": "nba",
            "season": "2025-26",
            "team": "GSW",
            "opponent": "DEN",
            "player_name": "Player B",
            "minutes": 31,
            "points": 22,
            "rebounds": 5,
            "assists": 10,
        },
    ]


def normalize(rows: Iterable[dict]) -> List[PlayerGameStat]:
    normalized: List[PlayerGameStat] = []
    for row in rows:
        normalized.append(
            PlayerGameStat(
                game_date=str(row["game_date"]),
                league=str(row["league"]).lower(),
                season=str(row["season"]),
                team=str(row["team"]).upper(),
                opponent=str(row["opponent"]).upper(),
                player_name=str(row["player_name"]).strip(),
                minutes=int(row["minutes"]),
                points=int(row["points"]),
                rebounds=int(row["rebounds"]),
                assists=int(row["assists"]),
            )
        )
    return normalized


def write_csv(rows: Iterable[PlayerGameStat], output_path: Path = OUTPUT_CSV) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [field.name for field in PlayerGameStat.__dataclass_fields__.values()]
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))
    return output_path


def run() -> Path:
    raw = fetch_demo_stats()
    cleaned = normalize(raw)
    return write_csv(cleaned)


if __name__ == "__main__":
    path = run()
    print(f"CSV generated: {path}")
