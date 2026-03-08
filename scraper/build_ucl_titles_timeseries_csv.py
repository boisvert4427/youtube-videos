from __future__ import annotations

import csv
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_CSV = PROJECT_ROOT / "data" / "processed" / "ucl_titles_timeseries_1956_2025.csv"


# European Cup / UEFA Champions League winners by final year.
WINNERS_BY_YEAR: list[tuple[int, str, str]] = [
    (1956, "Real Madrid", "ESP"),
    (1957, "Real Madrid", "ESP"),
    (1958, "Real Madrid", "ESP"),
    (1959, "Real Madrid", "ESP"),
    (1960, "Real Madrid", "ESP"),
    (1961, "Benfica", "POR"),
    (1962, "Benfica", "POR"),
    (1963, "AC Milan", "ITA"),
    (1964, "Inter Milan", "ITA"),
    (1965, "Inter Milan", "ITA"),
    (1966, "Real Madrid", "ESP"),
    (1967, "Celtic", "GBR"),
    (1968, "Manchester United", "GBR"),
    (1969, "AC Milan", "ITA"),
    (1970, "Feyenoord", "NED"),
    (1971, "Ajax", "NED"),
    (1972, "Ajax", "NED"),
    (1973, "Ajax", "NED"),
    (1974, "Bayern Munich", "GER"),
    (1975, "Bayern Munich", "GER"),
    (1976, "Bayern Munich", "GER"),
    (1977, "Liverpool", "GBR"),
    (1978, "Liverpool", "GBR"),
    (1979, "Nottingham Forest", "GBR"),
    (1980, "Nottingham Forest", "GBR"),
    (1981, "Liverpool", "GBR"),
    (1982, "Aston Villa", "GBR"),
    (1983, "Hamburger SV", "GER"),
    (1984, "Liverpool", "GBR"),
    (1985, "Juventus", "ITA"),
    (1986, "Steaua Bucuresti", "ROU"),
    (1987, "Porto", "POR"),
    (1988, "PSV Eindhoven", "NED"),
    (1989, "AC Milan", "ITA"),
    (1990, "AC Milan", "ITA"),
    (1991, "Red Star Belgrade", "SRB"),
    (1992, "Barcelona", "ESP"),
    (1993, "Marseille", "FRA"),
    (1994, "AC Milan", "ITA"),
    (1995, "Ajax", "NED"),
    (1996, "Juventus", "ITA"),
    (1997, "Borussia Dortmund", "GER"),
    (1998, "Real Madrid", "ESP"),
    (1999, "Manchester United", "GBR"),
    (2000, "Real Madrid", "ESP"),
    (2001, "Bayern Munich", "GER"),
    (2002, "Real Madrid", "ESP"),
    (2003, "AC Milan", "ITA"),
    (2004, "Porto", "POR"),
    (2005, "Liverpool", "GBR"),
    (2006, "Barcelona", "ESP"),
    (2007, "AC Milan", "ITA"),
    (2008, "Manchester United", "GBR"),
    (2009, "Barcelona", "ESP"),
    (2010, "Inter Milan", "ITA"),
    (2011, "Barcelona", "ESP"),
    (2012, "Chelsea", "GBR"),
    (2013, "Bayern Munich", "GER"),
    (2014, "Real Madrid", "ESP"),
    (2015, "Barcelona", "ESP"),
    (2016, "Real Madrid", "ESP"),
    (2017, "Real Madrid", "ESP"),
    (2018, "Real Madrid", "ESP"),
    (2019, "Liverpool", "GBR"),
    (2020, "Bayern Munich", "GER"),
    (2021, "Chelsea", "GBR"),
    (2022, "Real Madrid", "ESP"),
    (2023, "Manchester City", "GBR"),
    (2024, "Real Madrid", "ESP"),
    (2025, "Paris Saint-Germain", "FRA"),
]


def build_rows() -> list[dict]:
    titles_by_club: dict[str, int] = {}
    country_by_club: dict[str, str] = {}
    rows: list[dict] = []
    first_15_unique_winners: list[str] = []
    seen_winners: set[str] = set()
    start_year: int | None = None

    for year, winner, _ in WINNERS_BY_YEAR:
        if winner not in seen_winners:
            seen_winners.add(winner)
            first_15_unique_winners.append(winner)
            if len(first_15_unique_winners) == 15 and start_year is None:
                start_year = year
                break

    if start_year is None:
        return rows
    first_competition_year = WINNERS_BY_YEAR[0][0]

    for club in first_15_unique_winners:
        titles_by_club[club] = 0

    # Initial frame: all selected clubs start at 0.
    initial_date = f"{first_competition_year - 1}-06-01"
    initial_ranked = sorted(
        ((club, titles_by_club.get(club, 0)) for club in first_15_unique_winners),
        key=lambda item: (-item[1], item[0]),
    )
    for club, titles in initial_ranked:
        country = next((c for _, w, c in WINNERS_BY_YEAR if w == club), "")
        country_by_club[club] = country
        rows.append(
            {
                "ranking_date": initial_date,
                "player_name": club,
                "country_code": country,
                "points": titles,
            }
        )

    for year, winner, country_code in WINNERS_BY_YEAR:
        if winner in titles_by_club:
            titles_by_club[winner] = titles_by_club.get(winner, 0) + 1
        country_by_club[winner] = country_code
        if year < first_competition_year:
            continue

        ranking_date = f"{year}-06-01"
        ranked = sorted(
            ((club, titles_by_club.get(club, 0)) for club in first_15_unique_winners),
            key=lambda item: (-item[1], item[0]),
        )
        for club, titles in ranked:
            rows.append(
                {
                    "ranking_date": ranking_date,
                    "player_name": club,
                    "country_code": country_by_club[club],
                    "points": titles,
                }
            )
    return rows


def write_csv(rows: list[dict]) -> Path:
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["ranking_date", "player_name", "country_code", "points"]
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return OUTPUT_CSV


def run() -> Path:
    rows = build_rows()
    return write_csv(rows)


if __name__ == "__main__":
    output = run()
    print(f"[scraper] UCL timeseries generated -> {output}")
