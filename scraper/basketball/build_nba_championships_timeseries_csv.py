from __future__ import annotations

import argparse
import csv
import html as html_module
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from urllib.request import Request, urlopen


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SOURCE_URL = "https://www.nba.com/news/history-nba-champions"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "basketball" / "nba_championships_timeseries_1947_2025.csv"


FRANCHISE_MAP = {
    "Baltimore Bullets": "Baltimore Bullets",
    "Boston Celtics": "Boston Celtics",
    "Chicago Bulls": "Chicago Bulls",
    "Cleveland Cavaliers": "Cleveland Cavaliers",
    "Dallas Mavericks": "Dallas Mavericks",
    "Denver Nuggets": "Denver Nuggets",
    "Detroit Pistons": "Detroit Pistons",
    "Fort Wayne Pistons": "Detroit Pistons",
    "Ft. Wayne Pistons": "Detroit Pistons",
    "Golden State Warriors": "Golden State Warriors",
    "Houston Rockets": "Houston Rockets",
    "Los Angeles Lakers": "Los Angeles Lakers",
    "Miami Heat": "Miami Heat",
    "Milwaukee Bucks": "Milwaukee Bucks",
    "Minneapolis Lakers": "Los Angeles Lakers",
    "New Jersey Nets": "Brooklyn Nets",
    "New York Knicks": "New York Knicks",
    "Oklahoma City Thunder": "Oklahoma City Thunder",
    "Orlando Magic": "Orlando Magic",
    "Philadelphia 76ers": "Philadelphia 76ers",
    "Philadelphia Warriors": "Golden State Warriors",
    "Phoenix Suns": "Phoenix Suns",
    "Portland Trail Blazers": "Portland Trail Blazers",
    "Rochester Royals": "Sacramento Kings",
    "San Antonio Spurs": "San Antonio Spurs",
    "San Francisco Warriors": "Golden State Warriors",
    "Seattle SuperSonics": "Oklahoma City Thunder",
    "St. Louis Hawks": "Atlanta Hawks",
    "Syracuse Nationals": "Philadelphia 76ers",
    "Toronto Raptors": "Toronto Raptors",
    "Utah Jazz": "Utah Jazz",
    "Washington Bullets": "Washington Wizards",
}


TEAM_ABBR = {
    "Atlanta Hawks": "ATL",
    "Baltimore Bullets": "BLT",
    "Boston Celtics": "BOS",
    "Brooklyn Nets": "BKN",
    "Chicago Bulls": "CHI",
    "Cleveland Cavaliers": "CLE",
    "Dallas Mavericks": "DAL",
    "Denver Nuggets": "DEN",
    "Detroit Pistons": "DET",
    "Golden State Warriors": "GSW",
    "Houston Rockets": "HOU",
    "Los Angeles Lakers": "LAL",
    "Miami Heat": "MIA",
    "Milwaukee Bucks": "MIL",
    "New York Knicks": "NYK",
    "Oklahoma City Thunder": "OKC",
    "Orlando Magic": "ORL",
    "Philadelphia 76ers": "PHI",
    "Phoenix Suns": "PHX",
    "Portland Trail Blazers": "POR",
    "Sacramento Kings": "SAC",
    "San Antonio Spurs": "SAS",
    "Toronto Raptors": "TOR",
    "Utah Jazz": "UTA",
    "Washington Wizards": "WAS",
}


ARTICLE_JSON_RE = re.compile(r'<script[^>]+type="application/ld\+json"[^>]*>(.*?)</script>', re.IGNORECASE | re.DOTALL)
SEASON_RE = re.compile(
    r'(?P<season>\d{4}-\d{2})\s+--\s+(?P<champion>.+?)\s+def\.\s+(?P<runner_up>.+?),\s*(?P<result>\d-\d)',
    re.DOTALL,
)


@dataclass(frozen=True)
class NbaRow:
    ranking_date: str
    team_name: str
    team_abbr: str
    titles: int
    season_summary: str


@dataclass(frozen=True)
class SeasonChampion:
    ranking_date: str
    champion: str
    runner_up: str
    result: str


def _fetch_html(url: str) -> str:
    request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(request) as response:
        return response.read().decode("utf-8", errors="ignore")


def _extract_article_body(html_text: str) -> str:
    for raw_json in ARTICLE_JSON_RE.findall(html_text):
        candidate = html_module.unescape(raw_json)
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        objects = payload if isinstance(payload, list) else [payload]
        for item in objects:
            if isinstance(item, dict) and item.get("@type") == "Article" and item.get("articleBody"):
                return str(item["articleBody"])
    raise RuntimeError("Unable to locate NBA.com articleBody in the page HTML")


def _normalize_team_name(team_name: str) -> str:
    return FRANCHISE_MAP.get(team_name, team_name)


def _team_abbr(team_name: str) -> str:
    return TEAM_ABBR.get(team_name, "".join(part[0] for part in team_name.split()[:3]).upper())


def parse_champions(article_body: str) -> list[SeasonChampion]:
    seasons: list[SeasonChampion] = []
    normalized = article_body.replace("\xa0", " ")
    for match in SEASON_RE.finditer(normalized):
        season = match.group("season")
        end_year = int(season[:4]) + 1
        seasons.append(
            SeasonChampion(
                ranking_date=f"{end_year}-06-30",
                champion=match.group("champion").strip(),
                runner_up=match.group("runner_up").strip(),
                result=match.group("result").strip(),
            )
        )
    if not seasons:
        raise RuntimeError("No NBA seasons parsed from the official article")
    seasons.sort(key=lambda item: item.ranking_date)
    return seasons


def build_rows(seasons: list[SeasonChampion]) -> list[NbaRow]:
    cumulative: dict[str, int] = defaultdict(int)
    rows: list[NbaRow] = []

    for season in seasons:
        champion_franchise = _normalize_team_name(season.champion)
        cumulative[champion_franchise] += 1
        season_summary = f"{season.champion} def. {season.runner_up} | {season.result}"
        ordered = sorted(cumulative.items(), key=lambda item: (-item[1], item[0]))
        for team_name, titles in ordered:
            rows.append(
                NbaRow(
                    ranking_date=season.ranking_date,
                    team_name=team_name,
                    team_abbr=_team_abbr(team_name),
                    titles=titles,
                    season_summary=season_summary,
                )
            )
    return rows


def run(output_csv: Path = DEFAULT_OUTPUT, source_url: str = SOURCE_URL) -> Path:
    html_text = _fetch_html(source_url)
    article_body = _extract_article_body(html_text)
    seasons = parse_champions(article_body)
    rows = build_rows(seasons)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["ranking_date", "team_name", "team_abbr", "titles", "season_summary"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "ranking_date": row.ranking_date,
                    "team_name": row.team_name,
                    "team_abbr": row.team_abbr,
                    "titles": row.titles,
                    "season_summary": row.season_summary,
                }
            )
    return output_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch NBA champions history from NBA.com and build a titles timeseries CSV.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--source-url", default=SOURCE_URL)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = run(output_csv=args.output, source_url=args.source_url)
    print(f"[scraper] NBA champions timeseries generated -> {output}")


if __name__ == "__main__":
    main()
