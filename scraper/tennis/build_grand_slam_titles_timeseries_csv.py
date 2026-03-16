from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_CSV = PROJECT_ROOT / "data" / "processed" / "tennis" / "grand_slam_titles_timeseries_2000_2025.csv"
FULL_ERA_OUTPUT_CSV = PROJECT_ROOT / "data" / "processed" / "tennis" / "grand_slam_titles_timeseries_1968_2025.csv"


PLAYER_COUNTRIES = {
    "Adriano Panatta": "ITA",
    "Albert Costa": "ESP",
    "Andre Agassi": "USA",
    "Andres Gimeno": "ESP",
    "Andres Gomez": "ECU",
    "Andy Murray": "GBR",
    "Andy Roddick": "USA",
    "Arthur Ashe": "USA",
    "Bjorn Borg": "SWE",
    "Boris Becker": "GER",
    "Brian Teacher": "USA",
    "Carlos Alcaraz": "ESP",
    "Carlos Moya": "ESP",
    "Daniil Medvedev": "RUS",
    "Dominic Thiem": "AUT",
    "Gaston Gaudio": "ARG",
    "Goran Ivanisevic": "CRO",
    "Gustavo Kuerten": "BRA",
    "Guillermo Vilas": "ARG",
    "Ilie Nastase": "ROU",
    "Ivan Lendl": "USA",
    "Jan Kodes": "CZE",
    "Jannik Sinner": "ITA",
    "Johan Kriek": "RSA",
    "John McEnroe": "USA",
    "John Newcombe": "AUS",
    "Jim Courier": "USA",
    "Jimmy Connors": "USA",
    "Juan Carlos Ferrero": "ESP",
    "Juan Martin del Potro": "ARG",
    "Ken Rosewall": "AUS",
    "Lleyton Hewitt": "AUS",
    "Mark Edmondson": "AUS",
    "Manuel Orantes": "ESP",
    "Marat Safin": "RUS",
    "Marin Cilic": "CRO",
    "Mats Wilander": "SWE",
    "Michael Chang": "USA",
    "Michael Stich": "GER",
    "Novak Djokovic": "SRB",
    "Pat Cash": "AUS",
    "Patrick Rafter": "AUS",
    "Pete Sampras": "USA",
    "Petr Korda": "CZE",
    "Rafael Nadal": "ESP",
    "Richard Krajicek": "NED",
    "Rod Laver": "AUS",
    "Roger Federer": "SUI",
    "Roscoe Tanner": "USA",
    "Sergi Bruguera": "ESP",
    "Stan Smith": "USA",
    "Stan Wawrinka": "SUI",
    "Stefan Edberg": "SWE",
    "Thomas Johansson": "SWE",
    "Thomas Muster": "AUT",
    "Yannick Noah": "FRA",
    "Yevgeny Kafelnikov": "RUS",
}


SEASONS = [
    (1968, {"AO": "", "RG": "Ken Rosewall", "WIM": "Rod Laver", "USO": "Arthur Ashe"}),
    (1969, {"AO": "Rod Laver", "RG": "Rod Laver", "WIM": "Rod Laver", "USO": "Rod Laver"}),
    (1970, {"AO": "Arthur Ashe", "RG": "Jan Kodes", "WIM": "John Newcombe", "USO": "Ken Rosewall"}),
    (1971, {"AO": "Ken Rosewall", "RG": "Jan Kodes", "WIM": "John Newcombe", "USO": "Stan Smith"}),
    (1972, {"AO": "Ken Rosewall", "RG": "Andres Gimeno", "WIM": "Stan Smith", "USO": "Ilie Nastase"}),
    (1973, {"AO": "John Newcombe", "RG": "Ilie Nastase", "WIM": "Jan Kodes", "USO": "John Newcombe"}),
    (1974, {"AO": "Jimmy Connors", "RG": "Bjorn Borg", "WIM": "Jimmy Connors", "USO": "Jimmy Connors"}),
    (1975, {"AO": "John Newcombe", "RG": "Bjorn Borg", "WIM": "Arthur Ashe", "USO": "Manuel Orantes"}),
    (1976, {"AO": "Mark Edmondson", "RG": "Adriano Panatta", "WIM": "Bjorn Borg", "USO": "Jimmy Connors"}),
    (1977, {"AO": "Roscoe Tanner", "RG": "Guillermo Vilas", "WIM": "Bjorn Borg", "USO": "Guillermo Vilas"}),
    (1978, {"AO": "Guillermo Vilas", "RG": "Bjorn Borg", "WIM": "Bjorn Borg", "USO": "Jimmy Connors"}),
    (1979, {"AO": "Guillermo Vilas", "RG": "Bjorn Borg", "WIM": "Bjorn Borg", "USO": "John McEnroe"}),
    (1980, {"AO": "Brian Teacher", "RG": "Bjorn Borg", "WIM": "Bjorn Borg", "USO": "John McEnroe"}),
    (1981, {"AO": "Johan Kriek", "RG": "Bjorn Borg", "WIM": "John McEnroe", "USO": "John McEnroe"}),
    (1982, {"AO": "Johan Kriek", "RG": "Mats Wilander", "WIM": "Jimmy Connors", "USO": "Jimmy Connors"}),
    (1983, {"AO": "Mats Wilander", "RG": "Yannick Noah", "WIM": "John McEnroe", "USO": "Jimmy Connors"}),
    (1984, {"AO": "Mats Wilander", "RG": "Ivan Lendl", "WIM": "John McEnroe", "USO": "John McEnroe"}),
    (1985, {"AO": "Stefan Edberg", "RG": "Mats Wilander", "WIM": "Boris Becker", "USO": "Ivan Lendl"}),
    (1986, {"AO": "", "RG": "Ivan Lendl", "WIM": "Boris Becker", "USO": "Ivan Lendl"}),
    (1987, {"AO": "Stefan Edberg", "RG": "Ivan Lendl", "WIM": "Pat Cash", "USO": "Ivan Lendl"}),
    (1988, {"AO": "Mats Wilander", "RG": "Mats Wilander", "WIM": "Stefan Edberg", "USO": "Mats Wilander"}),
    (1989, {"AO": "Ivan Lendl", "RG": "Michael Chang", "WIM": "Boris Becker", "USO": "Boris Becker"}),
    (1990, {"AO": "Ivan Lendl", "RG": "Andres Gomez", "WIM": "Stefan Edberg", "USO": "Pete Sampras"}),
    (1991, {"AO": "Boris Becker", "RG": "Jim Courier", "WIM": "Michael Stich", "USO": "Stefan Edberg"}),
    (1992, {"AO": "Jim Courier", "RG": "Jim Courier", "WIM": "Andre Agassi", "USO": "Stefan Edberg"}),
    (1993, {"AO": "Jim Courier", "RG": "Sergi Bruguera", "WIM": "Pete Sampras", "USO": "Pete Sampras"}),
    (1994, {"AO": "Pete Sampras", "RG": "Sergi Bruguera", "WIM": "Pete Sampras", "USO": "Andre Agassi"}),
    (1995, {"AO": "Andre Agassi", "RG": "Thomas Muster", "WIM": "Pete Sampras", "USO": "Pete Sampras"}),
    (1996, {"AO": "Boris Becker", "RG": "Yevgeny Kafelnikov", "WIM": "Richard Krajicek", "USO": "Pete Sampras"}),
    (1997, {"AO": "Pete Sampras", "RG": "Gustavo Kuerten", "WIM": "Pete Sampras", "USO": "Patrick Rafter"}),
    (1998, {"AO": "Petr Korda", "RG": "Carlos Moya", "WIM": "Pete Sampras", "USO": "Patrick Rafter"}),
    (1999, {"AO": "Yevgeny Kafelnikov", "RG": "Andre Agassi", "WIM": "Pete Sampras", "USO": "Andre Agassi"}),
    (2000, {"AO": "Andre Agassi", "RG": "Gustavo Kuerten", "WIM": "Pete Sampras", "USO": "Marat Safin"}),
    (2001, {"AO": "Andre Agassi", "RG": "Gustavo Kuerten", "WIM": "Goran Ivanisevic", "USO": "Lleyton Hewitt"}),
    (2002, {"AO": "Thomas Johansson", "RG": "Albert Costa", "WIM": "Lleyton Hewitt", "USO": "Pete Sampras"}),
    (2003, {"AO": "Andre Agassi", "RG": "Juan Carlos Ferrero", "WIM": "Roger Federer", "USO": "Andy Roddick"}),
    (2004, {"AO": "Roger Federer", "RG": "Gaston Gaudio", "WIM": "Roger Federer", "USO": "Roger Federer"}),
    (2005, {"AO": "Marat Safin", "RG": "Rafael Nadal", "WIM": "Roger Federer", "USO": "Roger Federer"}),
    (2006, {"AO": "Roger Federer", "RG": "Rafael Nadal", "WIM": "Roger Federer", "USO": "Roger Federer"}),
    (2007, {"AO": "Roger Federer", "RG": "Rafael Nadal", "WIM": "Roger Federer", "USO": "Roger Federer"}),
    (2008, {"AO": "Novak Djokovic", "RG": "Rafael Nadal", "WIM": "Rafael Nadal", "USO": "Roger Federer"}),
    (2009, {"AO": "Rafael Nadal", "RG": "Roger Federer", "WIM": "Roger Federer", "USO": "Juan Martin del Potro"}),
    (2010, {"AO": "Roger Federer", "RG": "Rafael Nadal", "WIM": "Rafael Nadal", "USO": "Rafael Nadal"}),
    (2011, {"AO": "Novak Djokovic", "RG": "Rafael Nadal", "WIM": "Novak Djokovic", "USO": "Novak Djokovic"}),
    (2012, {"AO": "Novak Djokovic", "RG": "Rafael Nadal", "WIM": "Roger Federer", "USO": "Andy Murray"}),
    (2013, {"AO": "Novak Djokovic", "RG": "Rafael Nadal", "WIM": "Andy Murray", "USO": "Rafael Nadal"}),
    (2014, {"AO": "Stan Wawrinka", "RG": "Rafael Nadal", "WIM": "Novak Djokovic", "USO": "Marin Cilic"}),
    (2015, {"AO": "Novak Djokovic", "RG": "Stan Wawrinka", "WIM": "Novak Djokovic", "USO": "Novak Djokovic"}),
    (2016, {"AO": "Novak Djokovic", "RG": "Novak Djokovic", "WIM": "Andy Murray", "USO": "Stan Wawrinka"}),
    (2017, {"AO": "Roger Federer", "RG": "Rafael Nadal", "WIM": "Roger Federer", "USO": "Rafael Nadal"}),
    (2018, {"AO": "Roger Federer", "RG": "Rafael Nadal", "WIM": "Novak Djokovic", "USO": "Novak Djokovic"}),
    (2019, {"AO": "Novak Djokovic", "RG": "Rafael Nadal", "WIM": "Novak Djokovic", "USO": "Rafael Nadal"}),
    (2020, {"AO": "Novak Djokovic", "RG": "Rafael Nadal", "WIM": "", "USO": "Dominic Thiem"}),
    (2021, {"AO": "Novak Djokovic", "RG": "Novak Djokovic", "WIM": "Novak Djokovic", "USO": "Daniil Medvedev"}),
    (2022, {"AO": "Rafael Nadal", "RG": "Rafael Nadal", "WIM": "Novak Djokovic", "USO": "Carlos Alcaraz"}),
    (2023, {"AO": "Novak Djokovic", "RG": "Novak Djokovic", "WIM": "Carlos Alcaraz", "USO": "Novak Djokovic"}),
    (2024, {"AO": "Jannik Sinner", "RG": "Carlos Alcaraz", "WIM": "Carlos Alcaraz", "USO": "Jannik Sinner"}),
    (2025, {"AO": "Jannik Sinner", "RG": "Carlos Alcaraz", "WIM": "Jannik Sinner", "USO": "Carlos Alcaraz"}),
]


@dataclass(frozen=True)
class SlamRow:
    ranking_date: str
    player_name: str
    country_code: str
    points: int
    season_summary: str


def build_rows(start_year: int = 2000) -> list[SlamRow]:
    cumulative: dict[str, int] = {}
    rows: list[SlamRow] = []
    for year, winners in SEASONS:
        for winner in winners.values():
            if winner:
                cumulative[winner] = cumulative.get(winner, 0) + 1
        if year < start_year:
            continue
        summary = " | ".join(f"{slam} {winner or 'Canceled'}" for slam, winner in winners.items())
        ranking_date = f"{year}-12-31"
        ordered = sorted(cumulative.items(), key=lambda item: (-item[1], item[0]))
        for player_name, points in ordered:
            rows.append(
                SlamRow(
                    ranking_date=ranking_date,
                    player_name=player_name,
                    country_code=PLAYER_COUNTRIES.get(player_name, ""),
                    points=points,
                    season_summary=summary,
                )
            )
    return rows


def run(output_csv: Path = OUTPUT_CSV, start_year: int = 2000) -> Path:
    rows = build_rows(start_year=start_year)
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
    return output_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Grand Slam titles timeseries CSV.")
    parser.add_argument("--output", type=Path, default=OUTPUT_CSV)
    parser.add_argument("--start-year", type=int, default=2000)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    output = run(output_csv=args.output, start_year=args.start_year)
    print(f"[scraper] Grand Slam titles timeseries generated -> {output}")
