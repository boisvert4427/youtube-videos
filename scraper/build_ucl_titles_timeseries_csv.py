from __future__ import annotations

import csv
import urllib.request
from io import StringIO
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_CSV = PROJECT_ROOT / "data" / "processed" / "ucl_titles_timeseries_1956_2025.csv"
FINALS_URL = "https://en.wikipedia.org/wiki/List_of_European_Cup_and_UEFA_Champions_League_finals"


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


def _load_finals_by_year() -> dict[int, dict[str, str]]:
    req = urllib.request.Request(FINALS_URL, headers={"User-Agent": "youtube-videos-local/1.0"})
    with urllib.request.urlopen(req, timeout=60) as response:
        html = response.read().decode("utf-8", errors="ignore")
    tables = pd.read_html(StringIO(html))
    finals = tables[2]
    results: dict[int, dict[str, str]] = {}
    for _, row in finals.iterrows():
        season = str(row["Season"])
        score = str(row["Score"])
        if "–" not in season or score.lower() == "v":
            continue
        start_year_text, end_year_text = season.split("–", 1)
        start_year = int(start_year_text)
        end_suffix = end_year_text.strip()
        if len(end_suffix) == 2:
            century = (start_year // 100) * 100
            year = century + int(end_suffix)
            if year < start_year:
                year += 100
        else:
            year = int(end_suffix)
        winner = str(row["Winners"]).strip()
        runner_up = str(row["Runners-up"]).strip()
        results[year] = {
            "winner": winner,
            "runner_up": runner_up,
            "score": score,
            "final_score_line": f"{winner} {score} {runner_up}",
        }
    return results


def build_rows() -> list[dict[str, str | int]]:
    finals_by_year = _load_finals_by_year()
    clubs = sorted({winner for _, winner, _ in WINNERS_BY_YEAR})
    titles_by_club = {club: 0 for club in clubs}
    country_by_club = {winner: country for _, winner, country in WINNERS_BY_YEAR}
    rows: list[dict[str, str | int]] = []
    previous_top10: set[str] = set()

    initial_date = f"{WINNERS_BY_YEAR[0][0] - 1}-06-01"
    for club in sorted(clubs):
        rows.append(
            {
                "ranking_date": initial_date,
                "club_name": club,
                "country_code": country_by_club.get(club, ""),
                "titles": 0,
                "won_this_year": 0,
                "entered_top10": 0,
                "final_score": "",
                "final_runner_up": "",
                "final_score_line": "",
            }
        )

    for year, winner, country_code in WINNERS_BY_YEAR:
        titles_by_club[winner] += 1
        country_by_club[winner] = country_code
        ranking_date = f"{year}-06-01"
        ranked = sorted(
            ((club, titles_by_club[club]) for club in clubs),
            key=lambda item: (-item[1], item[0]),
        )
        top10 = {club for club, _ in ranked[:10]}
        final_meta = finals_by_year.get(year, {"score": "", "runner_up": "", "final_score_line": ""})
        for club, titles in ranked:
            rows.append(
                {
                    "ranking_date": ranking_date,
                    "club_name": club,
                    "country_code": country_by_club.get(club, ""),
                    "titles": titles,
                    "won_this_year": 1 if club == winner else 0,
                    "entered_top10": 1 if titles > 0 and club in top10 and club not in previous_top10 else 0,
                    "final_score": final_meta["score"],
                    "final_runner_up": final_meta["runner_up"],
                    "final_score_line": final_meta["final_score_line"],
                }
            )
        previous_top10 = top10
    return rows


def write_csv(rows: list[dict[str, str | int]]) -> Path:
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "ranking_date",
        "club_name",
        "country_code",
        "titles",
        "won_this_year",
        "entered_top10",
        "final_score",
        "final_runner_up",
        "final_score_line",
    ]
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
