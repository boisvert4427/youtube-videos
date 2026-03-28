from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "tennis" / "miami_titles_top20_cards.csv"

MIAMI_WINNERS = [
    (2025, "Jakub Mensik", "CZ"),
    (2024, "Jannik Sinner", "IT"),
    (2023, "Daniil Medvedev", "RU"),
    (2022, "Carlos Alcaraz", "ES"),
    (2021, "Hubert Hurkacz", "PL"),
    (2019, "Roger Federer", "CH"),
    (2018, "John Isner", "US"),
    (2017, "Roger Federer", "CH"),
    (2016, "Novak Djokovic", "RS"),
    (2015, "Novak Djokovic", "RS"),
    (2014, "Novak Djokovic", "RS"),
    (2013, "Andy Murray", "GB"),
    (2012, "Novak Djokovic", "RS"),
    (2011, "Novak Djokovic", "RS"),
    (2010, "Andy Roddick", "US"),
    (2009, "Andy Murray", "GB"),
    (2008, "Nikolay Davydenko", "RU"),
    (2007, "Novak Djokovic", "RS"),
    (2006, "Roger Federer", "CH"),
    (2005, "Roger Federer", "CH"),
    (2004, "Andy Roddick", "US"),
    (2003, "Andre Agassi", "US"),
    (2002, "Andre Agassi", "US"),
    (2001, "Andre Agassi", "US"),
    (2000, "Pete Sampras", "US"),
    (1999, "Richard Krajicek", "NL"),
    (1998, "Marcelo Rios", "CL"),
    (1997, "Thomas Muster", "AT"),
    (1996, "Andre Agassi", "US"),
    (1995, "Andre Agassi", "US"),
    (1994, "Pete Sampras", "US"),
    (1993, "Pete Sampras", "US"),
    (1992, "Michael Chang", "US"),
    (1991, "Jim Courier", "US"),
    (1990, "Andre Agassi", "US"),
    (1989, "Ivan Lendl", "US"),
    (1988, "Mats Wilander", "SE"),
    (1987, "Miloslav Mecir", "SK"),
    (1986, "Ivan Lendl", "US"),
    (1985, "Tim Mayotte", "US"),
]


CARD_COLORS = [
    ("#0b2a4a", "#4fd1ff"),
    ("#1f2746", "#7ddc5b"),
    ("#3f1837", "#ffb347"),
    ("#102c36", "#ffe066"),
    ("#2d1844", "#6cf0ff"),
    ("#3a2512", "#ff8c69"),
    ("#162c1f", "#9be564"),
    ("#2a1d12", "#ffda79"),
]

FINAL_RESULTS = {
    1986: "1986 d. Wilander 3-6 6-1 7-6 6-4",
    1989: "1989 d. Muster walkover",
    1993: "1993 d. Washington 6-3 6-2",
    1994: "1994 d. Agassi 5-7 6-3 6-3",
    1995: "1995 d. Sampras 3-6 6-2 7-6",
    1996: "1996 d. Ivanisevic 3-0 ret.",
    2000: "2000 d. Kuerten 6-1 6-7 7-6 7-6",
    2001: "2001 d. Gambill 7-6 6-1 6-0",
    2002: "2002 d. Federer 6-3 6-3 3-6 6-4",
    2003: "2003 d. Moya 6-3 6-3",
    2004: "2004 d. Coria 6-7 6-3 6-1 7-6",
    2005: "2005 d. Nadal 2-6 6-7 7-6 6-3 6-1",
    2006: "2006 d. Ljubicic 7-6 7-6 7-6",
    2007: "2007 d. Canas 6-3 6-2 6-4",
    2009: "2009 d. Djokovic 6-2 7-5",
    2010: "2010 d. Berdych 7-5 6-4",
    2011: "2011 d. Nadal 4-6 6-3 7-6",
    2012: "2012 d. Murray 6-1 7-6",
    2013: "2013 d. Ferrer 2-6 6-4 7-6",
    2014: "2014 d. Nadal 6-3 6-3",
    2015: "2015 d. Murray 7-6 4-6 6-0",
    2016: "2016 d. Nishikori 6-3 6-3",
    2017: "2017 d. Nadal 6-3 6-4",
    2019: "2019 d. Isner 6-1 6-4",
    2023: "2023 d. Sinner 7-5 6-3",
    2024: "2024 d. Dimitrov 6-3 6-1",
    2025: "2025 d. Djokovic 7-6 7-6",
}


def build_rows(top_n: int) -> list[dict[str, str]]:
    grouped: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for year, player, country in MIAMI_WINNERS:
        grouped[player].append((year, country))

    ordered = sorted(
        grouped.items(),
        key=lambda item: (len(item[1]), max(year for year, _ in item[1]), item[0]),
    )[-top_n:]
    ordered.sort(key=lambda item: (len(item[1]), max(year for year, _ in item[1]), item[0]))

    rows: list[dict[str, str]] = []
    for idx, (player, wins) in enumerate(ordered, start=1):
        years = sorted((year for year, _ in wins), reverse=True)
        country = wins[0][1]
        bg_color, accent_color = CARD_COLORS[(idx - 1) % len(CARD_COLORS)]
        rows.append(
            {
                "rank": str(idx),
                "player_name": player,
                "country_code": country.lower(),
                "titles": str(len(wins)),
                "years_won": " / ".join(str(year) for year in years),
                "first_title": str(min(years)),
                "last_title": str(max(years)),
                "badge_label": f"MIAMI x{len(wins)}",
                "final_results": " / ".join(FINAL_RESULTS.get(year, str(year)) for year in years),
                "card_bg_color": bg_color,
                "accent_color": accent_color,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Miami Open men's singles top winners cards CSV.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--top-n", type=int, default=20)
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
                "years_won",
                "first_title",
                "last_title",
                "badge_label",
                "final_results",
                "card_bg_color",
                "accent_color",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"[scraper] Miami titles cards CSV generated -> {args.output}")


if __name__ == "__main__":
    main()
