from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_KAGGLE_RANKINGS = Path.home() / "Downloads" / "rankings.csv"
DEFAULT_KAGGLE_PLAYERS = Path.home() / "Downloads" / "players.csv"
DEFAULT_OFFICIAL_CSV = PROJECT_ROOT / "data" / "processed" / "tennis" / "wta_ranking_points_timeseries_1990_2026.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "tennis" / "wta_rankings_kaggle_official_1984_2026.csv"


@dataclass(frozen=True)
class RankingRow:
    ranking_date: str
    ranking: int
    player_id: str
    player_name: str
    country_code: str
    ranking_points: int
    source: str


def _read_csv(path: Path) -> list[dict[str, str]]:
    encodings = ["utf-8-sig", "latin-1", "cp1252", "utf-8"]
    last_error: Exception | None = None
    for encoding in encodings:
        try:
            with path.open("r", newline="", encoding=encoding) as file:
                return list(csv.DictReader(file))
        except Exception as exc:
            last_error = exc
    raise last_error or RuntimeError(f"Could not read {path}")


def _normalize_player_id(value: str) -> str:
    value = (value or "").strip()
    if not value:
        return ""
    try:
        return str(int(float(value)))
    except ValueError:
        return value


def _format_date(raw: str) -> str:
    raw = (raw or "").strip()
    if len(raw) == 8 and raw.isdigit():
        return datetime.strptime(raw, "%Y%m%d").date().isoformat()
    if len(raw) == 10 and raw[4] == "-" and raw[7] == "-":
        return raw
    return raw


def _player_name(row: dict[str, str]) -> str:
    first = (row.get("first_name") or "").strip()
    last = (row.get("last_name") or "").strip()
    name = " ".join(part for part in [first, last] if part)
    return name.strip()


def _build_from_kaggle(rankings_csv: Path, players_csv: Path) -> list[RankingRow]:
    players = { _normalize_player_id(row.get("player_id", "")): row for row in _read_csv(players_csv) }
    rankings = _read_csv(rankings_csv)
    rows: list[RankingRow] = []
    for row in rankings:
        player_id = _normalize_player_id(row.get("player_id", ""))
        if not player_id:
            continue
        player = players.get(player_id)
        if player is None:
            continue
        player_name = _player_name(player)
        if not player_name:
            continue
        country_code = (player.get("country_code") or "").strip().upper()
        ranking_date = _format_date(row.get("ranking_date", ""))
        try:
            ranking = int(float(row.get("ranking", "0")))
            ranking_points = int(round(float(row.get("ranking_points", "0"))))
        except ValueError:
            continue
        rows.append(
            RankingRow(
                ranking_date=ranking_date,
                ranking=ranking,
                player_id=player_id,
                player_name=player_name,
                country_code=country_code,
                ranking_points=ranking_points,
                source="kaggle",
            )
        )
    return rows


def _build_from_official(official_csv: Path, cutoff_date: str = "2017-09-11") -> list[RankingRow]:
    rows: list[RankingRow] = []
    if not official_csv.exists():
        return rows
    with official_csv.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            ranking_date = (row.get("ranking_date") or "").strip()
            if not ranking_date or ranking_date <= cutoff_date:
                continue
            try:
                ranking = int(float(row.get("rank", "0")))
                ranking_points = int(round(float(row.get("points", "0"))))
            except ValueError:
                continue
            player_name = (row.get("player_name") or "").strip()
            country_code = (row.get("country_code") or "").strip().upper()
            rows.append(
                RankingRow(
                    ranking_date=ranking_date,
                    ranking=ranking,
                    player_id="",
                    player_name=player_name,
                    country_code=country_code,
                    ranking_points=ranking_points,
                    source="official_wta",
                )
            )
    return rows


def build_csv(
    rankings_csv: Path = DEFAULT_KAGGLE_RANKINGS,
    players_csv: Path = DEFAULT_KAGGLE_PLAYERS,
    official_csv: Path = DEFAULT_OFFICIAL_CSV,
    output_csv: Path = DEFAULT_OUTPUT,
) -> Path:
    rows = _build_from_kaggle(rankings_csv, players_csv)
    rows.extend(_build_from_official(official_csv))
    rows = sorted(rows, key=lambda item: (item.ranking_date, item.ranking, item.player_name, item.source))

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "ranking_date",
                "ranking",
                "player_id",
                "player_name",
                "country_code",
                "ranking_points",
                "source",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "ranking_date": row.ranking_date,
                    "ranking": row.ranking,
                    "player_id": row.player_id,
                    "player_name": row.player_name,
                    "country_code": row.country_code,
                    "ranking_points": row.ranking_points,
                    "source": row.source,
                }
            )
    return output_csv


def main() -> None:
    output = build_csv()
    print(f"[wta-kaggle] wrote {output}")


if __name__ == "__main__":
    main()
