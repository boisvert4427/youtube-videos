from __future__ import annotations

import argparse
import csv
import json
import os
import time
import urllib.parse
import urllib.request
import urllib.error
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path


DEFAULT_KAGGLE_MIN_DATE = "1986-12-22"
DEFAULT_API_START_DATE = "2000-11-27"
DEFAULT_END_DATE = "2026-06-29"
DEFAULT_PAGE_SIZE = 12
DEFAULT_SLEEP_SECONDS = 5.0
DEFAULT_429_SLEEP_SECONDS = 300.0
DEFAULT_MAX_NEW_SNAPSHOTS = 5

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_KAGGLE_RANKINGS = Path.home() / "Downloads" / "rankings.csv"
DEFAULT_KAGGLE_PLAYERS = Path.home() / "Downloads" / "players.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "tennis" / "wta_rankings_weekly_top12_1986_2026.csv"
DEFAULT_API_ONLY_OUTPUT = PROJECT_ROOT / "data" / "processed" / "tennis" / "wta_rankings_weekly_top12_api_only_2000_2026.csv"

API_URL = "https://api.wtatennis.com/tennis/players/ranked"
USER_AGENT = "Mozilla/5.0 (compatible; Codex WTA rankings scraper)"


@dataclass(frozen=True)
class RankingRow:
    ranking_date: str
    ranking: int
    player_id: str
    player_name: str
    country_code: str
    ranking_points: int
    source: str


def _request_json(url: str) -> list[dict]:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    delay = 1.0
    for attempt in range(10):
        try:
            with urllib.request.urlopen(request, timeout=45) as response:
                return json.loads(response.read().decode("utf-8"))
        except (urllib.error.HTTPError, urllib.error.URLError) as exc:
            code = getattr(exc, "code", None)
            if code not in {429, 500, 502, 503, 504} or attempt == 9:
                raise
            time.sleep(max(delay, DEFAULT_429_SLEEP_SECONDS) if code == 429 else delay)
            delay *= 2.0
    raise RuntimeError("unreachable")


def _date_range(start_date: str, end_date: str) -> list[str]:
    start = date.fromisoformat(start_date)
    end = date.fromisoformat(end_date)
    values: list[str] = []
    current = start
    while current <= end:
        values.append(current.isoformat())
        current += timedelta(days=7)
    return values


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    for encoding in ["utf-8-sig", "latin-1", "cp1252", "utf-8"]:
        try:
            with path.open("r", newline="", encoding=encoding) as file:
                return list(csv.DictReader(file))
        except Exception:
            continue
    raise RuntimeError(f"Could not read {path}")


def _normalize_player_id(value: str) -> str:
    value = (value or "").strip()
    if not value:
        return ""
    try:
        return str(int(float(value)))
    except ValueError:
        return value


def _format_kaggle_date(raw: str) -> str:
    raw = (raw or "").strip()
    if len(raw) == 8 and raw.isdigit():
        return f"{raw[:4]}-{raw[4:6]}-{raw[6:8]}"
    return raw


def _build_kaggle_rows(rankings_csv: Path, players_csv: Path) -> list[RankingRow]:
    players = {_normalize_player_id(row.get("player_id", "")): row for row in _read_csv_rows(players_csv)}
    rankings = _read_csv_rows(rankings_csv)
    rows: list[RankingRow] = []
    for row in rankings:
        ranking_date = _format_kaggle_date(row.get("ranking_date", ""))
        if not ranking_date or ranking_date < DEFAULT_KAGGLE_MIN_DATE:
            continue
        player_id = _normalize_player_id(row.get("player_id", ""))
        player = players.get(player_id)
        if player is None:
            continue
        if not row.get("ranking_points", "").strip():
            continue
        first = (player.get("first_name") or "").strip()
        last = (player.get("last_name") or "").strip()
        player_name = " ".join(part for part in [first, last] if part)
        if not player_name:
            continue
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
                country_code=(player.get("country_code") or "").strip().upper(),
                ranking_points=ranking_points,
                source="kaggle",
            )
        )
    return rows


def _fetch_snapshot(at_date: str, page_size: int) -> list[RankingRow] | None:
    params = {
        "page": 0,
        "pageSize": page_size,
        "type": "rankSingles",
        "sort": "asc",
        "name": "",
        "metric": "SINGLES",
        "at": at_date,
        "nationality": "",
    }
    url = API_URL + "?" + urllib.parse.urlencode(params)
    payload = _request_json(url)
    if not payload:
        return None
    rows: list[RankingRow] = []
    for item in payload[:page_size]:
        player = item.get("player") or {}
        ranked_at = (item.get("rankedAt") or at_date).split("T", 1)[0]
        full_name = (player.get("fullName") or "").strip()
        if not full_name:
            first = (player.get("firstName") or "").strip()
            last = (player.get("lastName") or "").strip()
            full_name = " ".join(part for part in [first, last] if part).strip()
        if not full_name:
            continue
        rows.append(
            RankingRow(
                ranking_date=ranked_at,
                ranking=int(item.get("ranking") or 0),
                player_id=str(player.get("id") or "").strip(),
                player_name=full_name,
                country_code=str(player.get("countryCode") or "").strip().upper(),
                ranking_points=int(round(float(item.get("points") or 0))),
                source="wta_api",
            )
        )
    return rows


def _write_rows(path: Path, rows: list[RankingRow], append: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append and path.exists() else "w"
    write_header = mode == "w"
    with path.open(mode, newline="", encoding="utf-8") as file:
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
        if write_header:
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


def build_csv(
    start_date: str,
    end_date: str,
    page_size: int,
    output_csv: Path,
    sleep_seconds: float,
    max_new_snapshots: int,
    seed_from_kaggle: bool,
) -> Path:
    existing_dates: set[str] = set()
    if output_csv.exists():
        try:
            with output_csv.open("r", newline="", encoding="utf-8") as file:
                reader = csv.DictReader(file)
                existing_dates = {row["ranking_date"] for row in reader if row.get("ranking_date")}
        except Exception:
            existing_dates = set()

    if seed_from_kaggle and (not output_csv.exists() or not existing_dates):
        kaggle_rows = _build_kaggle_rows(DEFAULT_KAGGLE_RANKINGS, DEFAULT_KAGGLE_PLAYERS)
        kaggle_rows = sorted(kaggle_rows, key=lambda item: (item.ranking_date, item.ranking, item.player_name))
        _write_rows(output_csv, kaggle_rows, append=False)
        existing_dates.update(row.ranking_date for row in kaggle_rows)

    processed = 0
    for at_date in _date_range(start_date, end_date):
        if at_date in existing_dates:
            continue
        try:
            snapshot = _fetch_snapshot(at_date, page_size)
        except Exception as exc:
            if getattr(exc, "code", None) == 429:
                print(f"[wta-api] 429 on {at_date}, sleeping {DEFAULT_429_SLEEP_SECONDS}s")
                time.sleep(DEFAULT_429_SLEEP_SECONDS)
                continue
            print(f"[wta-api] skip {at_date}: {type(exc).__name__}")
            continue
        if not snapshot:
            continue
        ranking_date = snapshot[0].ranking_date
        if ranking_date in existing_dates:
            continue
        existing_dates.add(ranking_date)
        _write_rows(output_csv, snapshot, append=True)
        processed += 1
        print(f"[wta-api] wrote {ranking_date} ({processed} new snapshots)")
        time.sleep(sleep_seconds)
        if processed >= max_new_snapshots:
            print(f"[wta-api] batch limit reached ({processed}); exiting so you can resume safely")
            break

    return output_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build WTA weekly ranking timeseries from the WTA API.")
    parser.add_argument("--start-date", default=DEFAULT_API_START_DATE)
    parser.add_argument("--end-date", default=DEFAULT_END_DATE)
    parser.add_argument("--page-size", type=int, default=DEFAULT_PAGE_SIZE)
    parser.add_argument("--sleep-seconds", type=float, default=DEFAULT_SLEEP_SECONDS)
    parser.add_argument("--max-new-snapshots", type=int, default=DEFAULT_MAX_NEW_SNAPSHOTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--api-only", action="store_true", help="Do not seed the CSV from Kaggle data; build WTA API rows only.")
    parser.add_argument("--api-only-output", type=Path, default=DEFAULT_API_ONLY_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = build_csv(
        args.start_date,
        args.end_date,
        args.page_size,
        args.api_only_output if args.api_only else args.output,
        args.sleep_seconds,
        args.max_new_snapshots,
        seed_from_kaggle=not args.api_only,
    )
    print(f"[wta-api] wrote {output}")


if __name__ == "__main__":
    main()
