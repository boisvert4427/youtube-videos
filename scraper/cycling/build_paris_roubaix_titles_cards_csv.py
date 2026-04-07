from __future__ import annotations

import argparse
import csv
import html as html_lib
import re
from collections import defaultdict
from pathlib import Path
from urllib.request import Request, urlopen


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "cycling" / "paris_roubaix_titles_top10_cards.csv"
HISTORY_URL = "https://www.paris-roubaix.fr/en/history"
USER_AGENT = "Mozilla/5.0 (compatible; Codex Paris-Roubaix builder)"

CARD_COLORS = [
    ("#1c1f28", "#d7b36a"),
    ("#24201a", "#e49c5c"),
    ("#1f2530", "#9ec3ff"),
    ("#2a1e1f", "#ff9b7d"),
    ("#1e2621", "#9be564"),
    ("#241e2f", "#a593ff"),
    ("#1a2c33", "#72e3ff"),
    ("#2c2218", "#ffd86f"),
]

TABLE_BODY_PATTERN = re.compile(r"<h2>Race winners since 1896</h2>\s*<table[^>]*>(.*?)</table>", re.S)
ROW_PATTERN = re.compile(
    r"<tr>\s*<td(?: rowspan=\"\d+\")?[^>]*>(\d{4})</td>\s*<td[^>]*>(.*?)</td>\s*<td(?: rowspan=\"\d+\")?[^>]*>(.*?)</td>\s*</tr>",
    re.S,
)
SECOND_WINNER_PATTERN = re.compile(r"<tr>\s*<td[^>]*>(.*?)</td>\s*</tr>", re.S)

COUNTRY_MAP = {
    "Mathieu VAN DER POEL": "nl",
    "Dylan VAN BAARLE": "nl",
    "Sonny COLBRELLI": "it",
    "Philippe GILBERT": "be",
    "Peter SAGAN": "sk",
    "Greg VAN AVERMAET": "be",
    "Mathew HAYMAN": "au",
    "John DEGENKOLB": "de",
    "Niki TERPSTRA": "nl",
    "Fabian CANCELLARA": "ch",
    "Tom BOONEN": "be",
    "Johan VAN SUMMEREN": "be",
    "Stuart O’GRADY": "au",
    "Magnus BACKSTEDT": "se",
    "Peter VAN PETEGEM": "be",
    "Johan MUSEEUW": "be",
    "Servais KNAVEN": "nl",
    "Andrea TAFI": "it",
    "Franco BALLERINI": "it",
    "Frédéric GUESDON": "fr",
    "Andreï TCHMIL": "be",
    "Gilbert DUCLOS-LASSALLE": "fr",
    "Marc MADIOT": "fr",
    "Eddy PLANCKERT": "be",
    "Jean-Marie WAMPERS": "be",
    "Dirk DE MOL": "be",
    "Eric VANDERAERDEN": "be",
    "Sean KELLY": "ie",
    "Hennie KUIPER": "nl",
    "Jan RAAS": "nl",
    "Bernard HINAULT": "fr",
    "Francesco MOSER": "it",
    "Roger DE VLAEMINCK": "be",
    "Marc DE MEYER": "be",
    "Eddy MERCKX": "be",
    "Roger ROSIERS": "be",
    "Walter GODEFROOT": "be",
    "Jan JANSSEN": "nl",
    "Felice GIMONDI": "it",
    "Rik VAN LOOY": "be",
    "Peter POST": "nl",
    "Émile DAEMS": "be",
    "Pino CERAMI": "be",
    "Noël FORE": "fr",
    "Léon VAN DAELE": "be",
    "Fred DE BRUYNE": "be",
    "Louison BOBET": "fr",
    "Jean FORESTIER": "fr",
    "Raymond IMPANIS": "be",
    "Germain DERYCKE": "be",
    "Rik VAN STEENBERGEN": "be",
    "Antonio BEVILACQUA": "it",
    "Fausto COPPI": "it",
    "André MAHE": "fr",
    "Serse COPPI": "it",
    "Georges CLAES": "be",
    "Paul MAYE": "fr",
    "Maurice DE SIMPELAERE": "be",
    "Marcel KINT": "be",
    "Émile MASSON": "be",
    "Lucien STORME": "be",
    "Jules ROSSI": "fr",
    "Georges SPEICHER": "fr",
    "Gaston REBRY": "be",
    "Sylvère MAES": "be",
    "Romain GIJSSELS": "be",
    "Julien VERVAECKE": "be",
    "Charles MEUNIER": "fr",
    "André LEDUCQ": "fr",
    "Georges RONSSE": "be",
    "Julien DELBECQUE": "be",
    "Felix SELLIER": "be",
    "Jules VAN HEVEL": "be",
    "Henri SUTER": "ch",
    "Albert DEJONGHE": "be",
    "Henri PELISSIER": "fr",
    "Paul DEMAN": "be",
    "Charles CRUPELANDT": "fr",
    "François FABER": "lu",
    "Octave LAPIZE": "fr",
    "Cyrille VAN HAUWAERT": "be",
    "Georges PASSERIEU": "fr",
    "Henri CORNET": "fr",
    "Louis TROUSSELIER": "fr",
    "Hippolyte AUCOUTURIER": "fr",
    "Lucien LESNA": "fr",
    "Emile BOUHOURS": "fr",
    "Albert CHAMPION": "fr",
    "Maurice GARIN": "fr",
    "Josef FISCHER": "de",
}


def _fetch_html(url: str) -> str:
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=30) as response:
        return response.read().decode("utf-8", "ignore")


def _load_winners() -> list[tuple[int, str, str]]:
    html = _fetch_html(HISTORY_URL)
    results: list[tuple[int, str, str]] = []
    table_match = TABLE_BODY_PATTERN.search(html)
    if not table_match:
        raise RuntimeError("Paris-Roubaix history table not found on official history page.")
    table_html = table_match.group(1)
    rows = list(ROW_PATTERN.finditer(table_html))
    for idx, match in enumerate(rows):
        year, name_html, _km_html = match.groups()
        if year == "2020":
            continue
        rider = " ".join(html_lib.unescape(re.sub(r"<[^>]+>", " ", name_html)).split()).strip()
        if rider.upper() == "CANCELED":
            continue
        country_code = COUNTRY_MAP.get(rider, "be")
        results.append((int(year), rider, country_code))
        next_start = match.end()
        next_end = rows[idx + 1].start() if idx + 1 < len(rows) else len(table_html)
        between = table_html[next_start:next_end]
        for extra_match in SECOND_WINNER_PATTERN.finditer(between):
            extra_name = " ".join(html_lib.unescape(re.sub(r"<[^>]+>", " ", extra_match.group(1))).split()).strip()
            if not extra_name or extra_name.upper() == "CANCELED":
                continue
            extra_name = extra_name.replace("(ex-æquo)", "").replace("(ex-aequo)", "").strip()
            country_code = COUNTRY_MAP.get(extra_name, "be")
            results.append((int(year), extra_name, country_code))
    deduped = sorted(set(results))
    if not deduped:
        raise RuntimeError("No Paris-Roubaix winners found on official history page.")
    return deduped


def build_rows(top_n: int) -> list[dict[str, str]]:
    wins_by_rider: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for year, rider_name, country_code in _load_winners():
        wins_by_rider[rider_name].append((year, country_code))

    ordered = sorted(
        wins_by_rider.items(),
        key=lambda item: (
            len(item[1]),
            max(year for year, _ in item[1]),
            item[0],
        ),
    )[-top_n:]
    ordered.sort(
        key=lambda item: (
            len(item[1]),
            max(year for year, _ in item[1]),
            item[0],
        )
    )

    rows: list[dict[str, str]] = []
    total = len(ordered)
    for idx, (rider_name, wins) in enumerate(ordered, start=1):
        years = sorted((year for year, _ in wins), reverse=True)
        country_code = wins[-1][1]
        bg_color, accent_color = CARD_COLORS[(idx - 1) % len(CARD_COLORS)]
        rows.append(
            {
                "rank": str(total - idx + 1),
                "player_name": rider_name,
                "country_code": country_code,
                "titles": str(len(wins)),
                "years_won": " / ".join(str(year) for year in years),
                "first_title": str(min(years)),
                "last_title": str(max(years)),
                "badge_label": f"PARIS-ROUBAIX x{len(wins)}",
                "card_bg_color": bg_color,
                "accent_color": accent_color,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Paris-Roubaix top winners cards CSV.")
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

    print(f"[scraper] Paris-Roubaix titles cards CSV generated -> {args.output}")


if __name__ == "__main__":
    main()
