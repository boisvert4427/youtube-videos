from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from urllib.request import Request, urlopen


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_URL = "https://api-hub.nba.com/news/kia-mvp-ladder-updates-2025-26"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "basketball" / "nba_kia_mvp_ladder_2025_26_weekly.csv"

SEASON = "2025-26"
RANK_SCORES = {
    1: 100,
    2: 80,
    3: 65,
    4: 50,
    5: 40,
}
PLAYER_NAMES = {
    "Antetokounmpo": "Giannis Antetokounmpo",
    "Brown": "Jaylen Brown",
    "Brunson": "Jalen Brunson",
    "Cunningham": "Cade Cunningham",
    "Dončić": "Luka Doncic",
    "Gilgeous-Alexander": "Shai Gilgeous-Alexander",
    "Jokić": "Nikola Jokic",
    "Maxey": "Tyrese Maxey",
    "Wembanyama": "Victor Wembanyama",
}


class ScriptParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._in_script = False
        self._attrs: dict[str, str] = {}
        self._buffer: list[str] = []
        self.scripts: list[tuple[dict[str, str], str]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "script":
            self._in_script = True
            self._attrs = {key: value or "" for key, value in attrs}
            self._buffer = []

    def handle_data(self, data: str) -> None:
        if self._in_script:
            self._buffer.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "script" and self._in_script:
            self.scripts.append((self._attrs, "".join(self._buffer)))
            self._in_script = False


@dataclass(frozen=True)
class MvpLadderRow:
    season: str
    week: int
    rank: int
    player: str
    player_short: str
    rank_score: int
    source: str


def fetch_article_body(source_url: str) -> str:
    request = Request(source_url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(request, timeout=30) as response:
        html = response.read().decode("utf-8", "replace")

    parser = ScriptParser()
    parser.feed(html)

    for attrs, data in parser.scripts:
        if attrs.get("type") != "application/ld+json" or "articleBody" not in data:
            continue
        payload = json.loads(data)
        article_body = payload.get("articleBody")
        if isinstance(article_body, str):
            return article_body

    raise RuntimeError("Could not find NBA.com articleBody JSON-LD payload.")


def parse_weekly_ladder(article_body: str, source_url: str) -> list[MvpLadderRow]:
    lines = [line.strip() for line in article_body.splitlines() if line.strip()]
    rows: list[MvpLadderRow] = []
    index = 0

    while index < len(lines):
        week_match = re.fullmatch(r"Week\s+(\d+)", lines[index])
        if not week_match:
            index += 1
            continue

        week = int(week_match.group(1))
        player_shorts = lines[index + 1 : index + 6]
        if len(player_shorts) != 5:
            raise RuntimeError(f"Incomplete Top 5 for Week {week}.")

        for rank, player_short in enumerate(player_shorts, start=1):
            rows.append(
                MvpLadderRow(
                    season=SEASON,
                    week=week,
                    rank=rank,
                    player=PLAYER_NAMES.get(player_short, player_short),
                    player_short=player_short,
                    rank_score=RANK_SCORES[rank],
                    source=source_url,
                )
            )
        index += 6

    if not rows:
        raise RuntimeError("No weekly MVP Ladder rows parsed.")

    return sorted(rows, key=lambda row: (row.week, row.rank))


def write_csv(rows: list[MvpLadderRow], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "season",
                "week",
                "rank",
                "player",
                "player_short",
                "rank_score",
                "source",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build NBA Kia MVP Ladder weekly CSV from NBA.com.")
    parser.add_argument("--source-url", default=DEFAULT_SOURCE_URL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    article_body = fetch_article_body(args.source_url)
    rows = parse_weekly_ladder(article_body, args.source_url)
    output_path = write_csv(rows, args.output)
    print(f"[scraper] wrote {len(rows)} MVP Ladder rows -> {output_path}")


if __name__ == "__main__":
    main()
