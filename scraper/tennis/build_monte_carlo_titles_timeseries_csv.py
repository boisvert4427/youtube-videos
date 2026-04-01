from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "tennis" / "monte_carlo_titles_timeseries_1968_2025.csv"

TOURNAMENT_YEARS = [
    (1968, ["Nicola Pietrangeli"], "Nicola Pietrangeli | wins Monte-Carlo"),
    (1969, ["Tom Okker"], "Tom Okker | wins Monte-Carlo"),
    (1970, ["Zeljko Franulovic"], "Zeljko Franulovic | wins Monte-Carlo"),
    (1971, ["Ilie Nastase"], "Ilie Nastase | wins Monte-Carlo"),
    (1972, ["Ilie Nastase"], "Ilie Nastase | wins Monte-Carlo"),
    (1973, ["Ilie Nastase"], "Ilie Nastase | wins Monte-Carlo"),
    (1974, ["Andrew Pattison"], "Andrew Pattison | wins Monte-Carlo"),
    (1975, ["Manuel Orantes"], "Manuel Orantes | wins Monte-Carlo"),
    (1976, ["Guillermo Vilas"], "Guillermo Vilas | wins Monte-Carlo"),
    (1977, ["Bjorn Borg"], "Bjorn Borg | wins Monte-Carlo"),
    (1978, ["Raul Ramirez"], "Raul Ramirez | wins Monte-Carlo"),
    (1979, ["Bjorn Borg"], "Bjorn Borg | wins Monte-Carlo"),
    (1980, ["Bjorn Borg"], "Bjorn Borg | wins Monte-Carlo"),
    (1981, ["Guillermo Vilas", "Jimmy Connors"], "Guillermo Vilas & Jimmy Connors | co-champions"),
    (1982, ["Guillermo Vilas"], "Guillermo Vilas | wins Monte-Carlo"),
    (1983, ["Mats Wilander"], "Mats Wilander | wins Monte-Carlo"),
    (1984, ["Henrik Sundstrom"], "Henrik Sundstrom | wins Monte-Carlo"),
    (1985, ["Ivan Lendl"], "Ivan Lendl | wins Monte-Carlo"),
    (1986, ["Joakim Nystrom"], "Joakim Nystrom | wins Monte-Carlo"),
    (1987, ["Mats Wilander"], "Mats Wilander | wins Monte-Carlo"),
    (1988, ["Ivan Lendl"], "Ivan Lendl | wins Monte-Carlo"),
    (1989, ["Alberto Mancini"], "Alberto Mancini | wins Monte-Carlo"),
    (1990, ["Andrei Chesnokov"], "Andrei Chesnokov | wins Monte-Carlo"),
    (1991, ["Sergi Bruguera"], "Sergi Bruguera | wins Monte-Carlo"),
    (1992, ["Thomas Muster"], "Thomas Muster | wins Monte-Carlo"),
    (1993, ["Sergi Bruguera"], "Sergi Bruguera | wins Monte-Carlo"),
    (1994, ["Andrei Medvedev"], "Andrei Medvedev | wins Monte-Carlo"),
    (1995, ["Thomas Muster"], "Thomas Muster | wins Monte-Carlo"),
    (1996, ["Thomas Muster"], "Thomas Muster | wins Monte-Carlo"),
    (1997, ["Marcelo Rios"], "Marcelo Rios | wins Monte-Carlo"),
    (1998, ["Carlos Moya"], "Carlos Moya | wins Monte-Carlo"),
    (1999, ["Gustavo Kuerten"], "Gustavo Kuerten | wins Monte-Carlo"),
    (2000, ["Cedric Pioline"], "Cedric Pioline | wins Monte-Carlo"),
    (2001, ["Gustavo Kuerten"], "Gustavo Kuerten | wins Monte-Carlo"),
    (2002, ["Juan Carlos Ferrero"], "Juan Carlos Ferrero | wins Monte-Carlo"),
    (2003, ["Juan Carlos Ferrero"], "Juan Carlos Ferrero | wins Monte-Carlo"),
    (2004, ["Guillermo Coria"], "Guillermo Coria | wins Monte-Carlo"),
    (2005, ["Rafael Nadal"], "Rafael Nadal | starts Monte-Carlo streak"),
    (2006, ["Rafael Nadal"], "Rafael Nadal | extends streak to 2"),
    (2007, ["Rafael Nadal"], "Rafael Nadal | extends streak to 3"),
    (2008, ["Rafael Nadal"], "Rafael Nadal | extends streak to 4"),
    (2009, ["Rafael Nadal"], "Rafael Nadal | extends streak to 5"),
    (2010, ["Rafael Nadal"], "Rafael Nadal | extends streak to 6"),
    (2011, ["Rafael Nadal"], "Rafael Nadal | extends streak to 7"),
    (2012, ["Rafael Nadal"], "Rafael Nadal | extends streak to 8"),
    (2013, ["Novak Djokovic"], "Novak Djokovic | ends Nadal streak"),
    (2014, ["Stan Wawrinka"], "Stan Wawrinka | wins Monte-Carlo"),
    (2015, ["Novak Djokovic"], "Novak Djokovic | wins Monte-Carlo"),
    (2016, ["Rafael Nadal"], "Rafael Nadal | wins Monte-Carlo again"),
    (2017, ["Rafael Nadal"], "Rafael Nadal | reaches 10 titles"),
    (2018, ["Rafael Nadal"], "Rafael Nadal | reaches 11 titles"),
    (2019, ["Fabio Fognini"], "Fabio Fognini | wins Monte-Carlo"),
    (2021, ["Stefanos Tsitsipas"], "Stefanos Tsitsipas | wins Monte-Carlo"),
    (2022, ["Stefanos Tsitsipas"], "Stefanos Tsitsipas | wins back-to-back"),
    (2023, ["Andrey Rublev"], "Andrey Rublev | wins Monte-Carlo"),
    (2024, ["Stefanos Tsitsipas"], "Stefanos Tsitsipas | reaches 3 titles"),
    (2025, ["Carlos Alcaraz"], "Carlos Alcaraz | wins Monte-Carlo"),
]

PLAYER_COUNTRIES = {
    "Alberto Mancini": "ARG",
    "Andrei Chesnokov": "RUS",
    "Andrei Medvedev": "UKR",
    "Andrew Pattison": "ZAF",
    "Andrey Rublev": "RUS",
    "Bjorn Borg": "SWE",
    "Carlos Alcaraz": "ESP",
    "Carlos Moya": "ESP",
    "Cedric Pioline": "FRA",
    "Fabio Fognini": "ITA",
    "Gustavo Kuerten": "BRA",
    "Guillermo Coria": "ARG",
    "Guillermo Vilas": "ARG",
    "Henrik Sundstrom": "SWE",
    "Ilie Nastase": "ROU",
    "Ivan Lendl": "CZE",
    "Jimmy Connors": "USA",
    "Joakim Nystrom": "SWE",
    "Juan Carlos Ferrero": "ESP",
    "Manuel Orantes": "ESP",
    "Marcelo Rios": "CHI",
    "Mats Wilander": "SWE",
    "Nicola Pietrangeli": "ITA",
    "Novak Djokovic": "SRB",
    "Rafael Nadal": "ESP",
    "Raul Ramirez": "MEX",
    "Sergi Bruguera": "ESP",
    "Stan Wawrinka": "SUI",
    "Stefanos Tsitsipas": "GRE",
    "Thomas Muster": "AUT",
    "Tom Okker": "NED",
    "Zeljko Franulovic": "CRO",
}


@dataclass(frozen=True)
class SnapshotRow:
    ranking_date: str
    player_name: str
    country_code: str
    points: int
    season_summary: str


def build_rows() -> list[SnapshotRow]:
    cumulative: defaultdict[str, int] = defaultdict(int)
    rows: list[SnapshotRow] = []

    for year, champions, season_summary in TOURNAMENT_YEARS:
        for champion in champions:
            cumulative[champion] += 1
        ranking_date = f"{year}-04-30"
        ranked_players = sorted(cumulative.items(), key=lambda item: (-item[1], item[0]))
        for player_name, titles in ranked_players:
            rows.append(
                SnapshotRow(
                    ranking_date=ranking_date,
                    player_name=player_name,
                    country_code=PLAYER_COUNTRIES.get(player_name, ""),
                    points=titles,
                    season_summary=season_summary,
                )
            )

    return rows


def write_rows(rows: list[SnapshotRow], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["ranking_date", "player_name", "country_code", "points", "season_summary"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "ranking_date": row.ranking_date,
                    "player_name": row.player_name,
                    "country_code": row.country_code,
                    "points": row.points,
                    "season_summary": row.season_summary,
                }
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Monte-Carlo cumulative titles timeseries CSV.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = build_rows()
    write_rows(rows, args.output)
    print(f"[scraper] Monte-Carlo titles timeseries CSV generated -> {args.output}")


if __name__ == "__main__":
    main()
