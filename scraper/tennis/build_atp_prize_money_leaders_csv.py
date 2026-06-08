from __future__ import annotations

import argparse
import csv
import re
import urllib.request
from datetime import datetime
from pathlib import Path

import fitz


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SOURCE_URL = "https://www.protennislive.com/posting/ramr/career_prize.pdf"
DEFAULT_PDF_CACHE = PROJECT_ROOT / "data" / "raw" / "tennis_prize_money" / "atp_career_prize.pdf"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "tennis" / "atp_prize_money_leaders_current.csv"
DEFAULT_TOP_N = 10


COUNTRY_BY_PLAYER = {
    "Novak Djokovic": "RS",
    "Rafael Nadal": "ES",
    "Roger Federer": "CH",
    "Carlos Alcaraz": "ES",
    "Andy Murray": "GB",
    "Jannik Sinner": "IT",
    "Alexander Zverev": "DE",
    "Daniil Medvedev": "RU",
    "Pete Sampras": "US",
    "Stan Wawrinka": "CH",
}


def _download_pdf(url: str, cache_path: Path, refresh: bool) -> Path:
    if cache_path.exists() and not refresh:
        return cache_path

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "youtube-videos-local/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        cache_path.write_bytes(response.read())
    return cache_path


def _parse_money(raw_value: str) -> int:
    cleaned = raw_value.strip().replace("$", "").replace(",", "")
    if not cleaned or cleaned == "-":
        return 0
    return int(float(cleaned))


def _normalize_player_name(raw_name: str) -> str:
    text = raw_name.strip()
    if "," not in text:
        return text
    surname, given = [part.strip() for part in text.split(",", 1)]
    return f"{given} {surname}".strip()


def _parse_ranking_date(text: str) -> str:
    match = re.search(r"Rankings Date:\s*([A-Za-z]+\s+\d{1,2},\s+\d{4})", text)
    if not match:
        raise RuntimeError("Impossible d'extraire la date du classement ATP depuis le PDF.")
    parsed = datetime.strptime(match.group(1), "%B %d, %Y")
    return parsed.date().isoformat()


def _extract_rows_from_page(page_text: str, top_n: int) -> list[dict[str, str | int]]:
    lines = [line.strip() for line in page_text.splitlines() if line.strip()]
    if "Doubles" not in lines:
        raise RuntimeError("Le tableau ATP n'a pas ete trouve dans le PDF.")

    start_idx = lines.index("Doubles") + 1
    rows: list[dict[str, str | int]] = []
    i = start_idx
    while i + 5 < len(lines) and len(rows) < top_n:
        player_raw = lines[i]
        rank_raw = lines[i + 1]
        career_raw = lines[i + 2]
        ytd_raw = lines[i + 3]
        singles_raw = lines[i + 4]
        doubles_raw = lines[i + 5]

        if not rank_raw.isdigit():
            break

        player_name = _normalize_player_name(player_raw)
        rows.append(
            {
                "rank": int(rank_raw),
                "player_name": player_name,
                "country_code": COUNTRY_BY_PLAYER.get(player_name, ""),
                "career_usd": _parse_money(career_raw),
                "ytd_usd": _parse_money(ytd_raw),
                "singles_usd": _parse_money(singles_raw),
                "doubles_usd": _parse_money(doubles_raw),
            }
        )
        i += 6
    return rows


def build_rows(pdf_path: Path, top_n: int) -> list[dict[str, str | int]]:
    doc = fitz.open(pdf_path)
    if doc.page_count < 1:
        raise RuntimeError(f"Le PDF ATP est vide: {pdf_path}")

    first_page_text = doc.load_page(0).get_text("text")
    ranking_date = _parse_ranking_date(first_page_text)
    rows = _extract_rows_from_page(first_page_text, top_n=top_n)
    for row in rows:
        row["ranking_date"] = ranking_date
        row["tour"] = "ATP"
    return rows


def write_csv(output: Path, rows: list[dict[str, str | int]]) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "ranking_date",
                "tour",
                "rank",
                "player_name",
                "country_code",
                "career_usd",
                "ytd_usd",
                "singles_usd",
                "doubles_usd",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    return output


def run(output: Path, pdf_path: Path, top_n: int, refresh_pdf: bool) -> Path:
    cached_pdf = _download_pdf(SOURCE_URL, pdf_path, refresh=refresh_pdf)
    rows = build_rows(cached_pdf, top_n=top_n)
    return write_csv(output, rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the ATP prize money leaders CSV from the official PDF.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF_CACHE, help="Local cache path for the ATP PDF.")
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N, help="Number of leaders to keep.")
    parser.add_argument("--refresh-pdf", action="store_true", help="Force a re-download of the ATP PDF.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = run(
        output=args.output,
        pdf_path=args.pdf,
        top_n=args.top_n,
        refresh_pdf=args.refresh_pdf,
    )
    print(f"[scraper] ATP prize money leaders CSV generated -> {output}")


if __name__ == "__main__":
    main()
