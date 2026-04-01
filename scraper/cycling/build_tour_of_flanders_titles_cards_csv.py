from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path
from urllib.request import Request, urlopen


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "cycling" / "tour_of_flanders_titles_top10_cards.csv"
HISTORY_URL = "https://www.rondevanvlaanderen.be/en/race/men-elite/history"
USER_AGENT = "Mozilla/5.0 (compatible; Codex Tour of Flanders builder)"

CARD_COLORS = [
    ("#14243d", "#ffd24a"),
    ("#2d1b12", "#ff8f5a"),
    ("#172f22", "#9be564"),
    ("#25183e", "#6cf0ff"),
    ("#332015", "#ffd76a"),
    ("#1d233f", "#8db8ff"),
    ("#2f1a2f", "#ff9ad9"),
    ("#14313b", "#7ee7ff"),
]

EDITION_PATTERN = re.compile(
    r'<div class="accordion__panel" id="edition-(\d{4})">(.*?)</div>\s*</div>',
    re.S,
)
RESULT_PATTERN = re.compile(
    r'<span class="participants-list__position-item">([123])\.</span>\s*'
    r'<img[^>]+country-flags/([a-z]{2})\.[^"]+"[^>]*>\s*'
    r"<span>([^<]+)</span>",
    re.S,
)


def _fetch_html(url: str) -> str:
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=30) as response:
        return response.read().decode("utf-8", "ignore")


def _load_results() -> list[tuple[int, int, str, str]]:
    html = _fetch_html(HISTORY_URL)
    results: list[tuple[int, int, str, str]] = []
    for year, panel_html in EDITION_PATTERN.findall(html):
        top_three = RESULT_PATTERN.findall(panel_html)
        for position, country_code, rider_name in top_three[:3]:
            results.append((int(year), int(position), rider_name.strip(), country_code.strip().lower()))
    if not results:
        raise RuntimeError("No Tour of Flanders results found on official history page.")
    return results


def build_rows(top_n: int) -> list[dict[str, str]]:
    wins_by_rider: dict[str, list[tuple[int, str]]] = defaultdict(list)
    podiums_by_rider: dict[str, list[int]] = defaultdict(list)
    country_by_rider: dict[str, str] = {}
    for year, position, rider_name, country_code in _load_results():
        country_by_rider[rider_name] = country_code
        podiums_by_rider[rider_name].append(year)
        if position == 1:
            wins_by_rider[rider_name].append((year, country_code))

    ordered = sorted(
        wins_by_rider.items(),
        key=lambda item: (
            len(item[1]),
            len(podiums_by_rider[item[0]]),
            max(year for year, _ in item[1]),
            item[0],
        ),
    )[-top_n:]
    ordered.sort(
        key=lambda item: (
            len(item[1]),
            len(podiums_by_rider[item[0]]),
            max(year for year, _ in item[1]),
            item[0],
        )
    )

    rows: list[dict[str, str]] = []
    total = len(ordered)
    for idx, (rider_name, wins) in enumerate(ordered, start=1):
        years = sorted((year for year, _ in wins), reverse=True)
        country_code = country_by_rider[rider_name]
        podiums = len(podiums_by_rider[rider_name])
        bg_color, accent_color = CARD_COLORS[(idx - 1) % len(CARD_COLORS)]
        rows.append(
            {
                "rank": str(total - idx + 1),
                "player_name": rider_name,
                "country_code": country_code,
                "titles": str(len(wins)),
                "podiums": str(podiums),
                "years_won": " / ".join(str(year) for year in years),
                "first_title": str(min(years)),
                "last_title": str(max(years)),
                "badge_label": f"FLANDERS x{len(wins)}",
                "card_bg_color": bg_color,
                "accent_color": accent_color,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Tour of Flanders top winners cards CSV.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--top-n", type=int, default=10)
    args = parser.parse_args()

    rows = build_rows(args.top_n)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "rank",
                "player_name",
                "country_code",
                "titles",
                "podiums",
                "years_won",
                "first_title",
                "last_title",
                "badge_label",
                "card_bg_color",
                "accent_color",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"[scraper] Tour of Flanders titles cards CSV generated -> {args.output}")


if __name__ == "__main__":
    main()
