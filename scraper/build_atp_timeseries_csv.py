from __future__ import annotations

import csv
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "atp_rankings"
OUTPUT_CSV = PROJECT_ROOT / "data" / "processed" / "atp_ranking_timeseries_v1.csv"


@dataclass(frozen=True)
class AtpRankingRow:
    ranking_date: str
    player_name: str
    country_code: str
    points: int


def _normalize_row(row: dict) -> AtpRankingRow:
    return AtpRankingRow(
        ranking_date=str(row["ranking_date"]).strip(),
        player_name=str(row["player_name"]).strip(),
        country_code=str(row["country_code"]).strip().upper(),
        points=int(row["points"]),
    )


def load_raw_rows(raw_dir: Path = RAW_DIR) -> list[AtpRankingRow]:
    if not raw_dir.exists():
        return []

    rows: list[AtpRankingRow] = []
    for csv_file in sorted(raw_dir.glob("*.csv")):
        with csv_file.open("r", newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            for row in reader:
                rows.append(_normalize_row(row))
    return rows


def write_output(rows: Iterable[AtpRankingRow], output_csv: Path = OUTPUT_CSV) -> Path:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [field.name for field in AtpRankingRow.__dataclass_fields__.values()]
    with output_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))
    return output_csv


def run() -> Path:
    rows = load_raw_rows()
    if not rows:
        raise RuntimeError(
            f"No ATP raw snapshots found in {RAW_DIR}. "
            "Add CSV files with columns: ranking_date,player_name,country_code,points"
        )
    return write_output(rows)


if __name__ == "__main__":
    output = run()
    print(f"[scraper] ATP timeseries generated -> {output}")
