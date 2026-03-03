from __future__ import annotations

import csv
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_CSV = PROJECT_ROOT / "data" / "processed" / "player_game_stats_v1.csv"


def load_rows(path: Path = INPUT_CSV) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")

    with path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        return list(reader)


if __name__ == "__main__":
    rows = load_rows()
    print(f"[video_generator] loaded {len(rows)} rows from {INPUT_CSV}")
    if rows:
        print(f"sample row: {rows[0]}")
