from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_CSV = PROJECT_ROOT / "data" / "processed" / "basketball" / "nba_titles_timeseries_1947_2025.csv"


SEASONS = [
    (1947, "Philadelphia Warriors", "Chicago Stags", "4-1"),
    (1948, "Baltimore Bullets", "Philadelphia Warriors", "4-2"),
    (1949, "Minneapolis Lakers", "Washington Capitols", "4-2"),
    (1950, "Minneapolis Lakers", "Syracuse Nationals", "4-2"),
    (1951, "Rochester Royals", "New York Knicks", "4-3"),
    (1952, "Minneapolis Lakers", "New York Knicks", "4-3"),
    (1953, "Minneapolis Lakers", "New York Knicks", "4-1"),
    (1954, "Minneapolis Lakers", "Syracuse Nationals", "4-3"),
    (1955, "Syracuse Nationals", "Fort Wayne Pistons", "4-3"),
    (1956, "Philadelphia Warriors", "Fort Wayne Pistons", "4-1"),
    (1957, "Boston Celtics", "St. Louis Hawks", "4-3"),
    (1958, "St. Louis Hawks", "Boston Celtics", "4-2"),
    (1959, "Boston Celtics", "Minneapolis Lakers", "4-0"),
    (1960, "Boston Celtics", "St. Louis Hawks", "4-3"),
    (1961, "Boston Celtics", "St. Louis Hawks", "4-1"),
    (1962, "Boston Celtics", "Los Angeles Lakers", "4-3"),
    (1963, "Boston Celtics", "Los Angeles Lakers", "4-2"),
    (1964, "Boston Celtics", "San Francisco Warriors", "4-1"),
    (1965, "Boston Celtics", "Los Angeles Lakers", "4-1"),
    (1966, "Boston Celtics", "Los Angeles Lakers", "4-3"),
    (1967, "Philadelphia 76ers", "San Francisco Warriors", "4-2"),
    (1968, "Boston Celtics", "Los Angeles Lakers", "4-2"),
    (1969, "Boston Celtics", "Los Angeles Lakers", "4-3"),
    (1970, "New York Knicks", "Los Angeles Lakers", "4-3"),
    (1971, "Milwaukee Bucks", "Baltimore Bullets", "4-0"),
    (1972, "Los Angeles Lakers", "New York Knicks", "4-1"),
    (1973, "New York Knicks", "Los Angeles Lakers", "4-1"),
    (1974, "Boston Celtics", "Milwaukee Bucks", "4-3"),
    (1975, "Golden State Warriors", "Washington Bullets", "4-0"),
    (1976, "Boston Celtics", "Phoenix Suns", "4-2"),
    (1977, "Portland Trail Blazers", "Philadelphia 76ers", "4-2"),
    (1978, "Washington Bullets", "Seattle SuperSonics", "4-3"),
    (1979, "Seattle SuperSonics", "Washington Bullets", "4-1"),
    (1980, "Los Angeles Lakers", "Philadelphia 76ers", "4-2"),
    (1981, "Boston Celtics", "Houston Rockets", "4-2"),
    (1982, "Los Angeles Lakers", "Philadelphia 76ers", "4-2"),
    (1983, "Philadelphia 76ers", "Los Angeles Lakers", "4-0"),
    (1984, "Boston Celtics", "Los Angeles Lakers", "4-3"),
    (1985, "Los Angeles Lakers", "Boston Celtics", "4-2"),
    (1986, "Boston Celtics", "Houston Rockets", "4-2"),
    (1987, "Los Angeles Lakers", "Boston Celtics", "4-2"),
    (1988, "Los Angeles Lakers", "Detroit Pistons", "4-3"),
    (1989, "Detroit Pistons", "Los Angeles Lakers", "4-0"),
    (1990, "Detroit Pistons", "Portland Trail Blazers", "4-1"),
    (1991, "Chicago Bulls", "Los Angeles Lakers", "4-1"),
    (1992, "Chicago Bulls", "Portland Trail Blazers", "4-2"),
    (1993, "Chicago Bulls", "Phoenix Suns", "4-2"),
    (1994, "Houston Rockets", "New York Knicks", "4-3"),
    (1995, "Houston Rockets", "Orlando Magic", "4-0"),
    (1996, "Chicago Bulls", "Seattle SuperSonics", "4-2"),
    (1997, "Chicago Bulls", "Utah Jazz", "4-2"),
    (1998, "Chicago Bulls", "Utah Jazz", "4-2"),
    (1999, "San Antonio Spurs", "New York Knicks", "4-1"),
    (2000, "Los Angeles Lakers", "Indiana Pacers", "4-2"),
    (2001, "Los Angeles Lakers", "Philadelphia 76ers", "4-1"),
    (2002, "Los Angeles Lakers", "New Jersey Nets", "4-0"),
    (2003, "San Antonio Spurs", "New Jersey Nets", "4-2"),
    (2004, "Detroit Pistons", "Los Angeles Lakers", "4-1"),
    (2005, "San Antonio Spurs", "Detroit Pistons", "4-3"),
    (2006, "Miami Heat", "Dallas Mavericks", "4-2"),
    (2007, "San Antonio Spurs", "Cleveland Cavaliers", "4-0"),
    (2008, "Boston Celtics", "Los Angeles Lakers", "4-2"),
    (2009, "Los Angeles Lakers", "Orlando Magic", "4-1"),
    (2010, "Los Angeles Lakers", "Boston Celtics", "4-3"),
    (2011, "Dallas Mavericks", "Miami Heat", "4-2"),
    (2012, "Miami Heat", "Oklahoma City Thunder", "4-1"),
    (2013, "Miami Heat", "San Antonio Spurs", "4-3"),
    (2014, "San Antonio Spurs", "Miami Heat", "4-1"),
    (2015, "Golden State Warriors", "Cleveland Cavaliers", "4-2"),
    (2016, "Cleveland Cavaliers", "Golden State Warriors", "4-3"),
    (2017, "Golden State Warriors", "Cleveland Cavaliers", "4-1"),
    (2018, "Golden State Warriors", "Cleveland Cavaliers", "4-0"),
    (2019, "Toronto Raptors", "Golden State Warriors", "4-2"),
    (2020, "Los Angeles Lakers", "Miami Heat", "4-2"),
    (2021, "Milwaukee Bucks", "Phoenix Suns", "4-2"),
    (2022, "Golden State Warriors", "Boston Celtics", "4-2"),
    (2023, "Denver Nuggets", "Miami Heat", "4-1"),
    (2024, "Boston Celtics", "Dallas Mavericks", "4-1"),
    (2025, "Oklahoma City Thunder", "Indiana Pacers", "4-3"),
]


FRANCHISE_MAP = {
    "Baltimore Bullets": "Baltimore Bullets",
    "Boston Celtics": "Boston Celtics",
    "Chicago Bulls": "Chicago Bulls",
    "Cleveland Cavaliers": "Cleveland Cavaliers",
    "Dallas Mavericks": "Dallas Mavericks",
    "Denver Nuggets": "Denver Nuggets",
    "Detroit Pistons": "Detroit Pistons",
    "Fort Wayne Pistons": "Detroit Pistons",
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
    "Washington Capitols": "Washington Capitols",
}


TEAM_ABBREVIATIONS = {
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
    "Washington Capitols": "WAS",
    "Washington Wizards": "WAS",
}


@dataclass(frozen=True)
class NbaRow:
    ranking_date: str
    team_name: str
    team_abbr: str
    titles: int
    season_summary: str


def _franchise_name(team_name: str) -> str:
    return FRANCHISE_MAP.get(team_name, team_name)


def _abbr(team_name: str) -> str:
    return TEAM_ABBREVIATIONS.get(team_name, "".join(part[0] for part in team_name.split()[:3]).upper())


def build_rows(start_year: int = 1947) -> list[NbaRow]:
    cumulative: dict[str, int] = {}
    rows: list[NbaRow] = []
    for year, champion, runner_up, result in SEASONS:
        franchise = _franchise_name(champion)
        cumulative[franchise] = cumulative.get(franchise, 0) + 1
        if year < start_year:
            continue
        ranking_date = f"{year}-06-30"
        summary = f"{franchise} def. {_franchise_name(runner_up)} | {result}"
        ordered = sorted(cumulative.items(), key=lambda item: (-item[1], item[0]))
        for team_name, titles in ordered:
            rows.append(
                NbaRow(
                    ranking_date=ranking_date,
                    team_name=team_name,
                    team_abbr=_abbr(team_name),
                    titles=titles,
                    season_summary=summary,
                )
            )
    return rows


def run(output_csv: Path = OUTPUT_CSV, start_year: int = 1947) -> Path:
    rows = build_rows(start_year=start_year)
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
    parser = argparse.ArgumentParser(description="Build NBA franchise titles timeseries CSV.")
    parser.add_argument("--output", type=Path, default=OUTPUT_CSV)
    parser.add_argument("--start-year", type=int, default=1947)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = run(output_csv=args.output, start_year=args.start_year)
    print(f"[scraper] NBA titles timeseries generated -> {output}")


if __name__ == "__main__":
    main()
