from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import fitz
import requests


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "tennis" / "wta_rankings_official"
OUTPUT_CSV = PROJECT_ROOT / "data" / "processed" / "tennis" / "wta_ranking_points_timeseries_1990_2026.csv"

ARCHIVE_URL = "https://wtafiles.wtatennis.com/pdf/rankings/RankingArchive/Singles_Numeric_{year}.pdf"
CURRENT_URL = "https://wtafiles.wtatennis.com/pdf/rankings/Singles_Numeric.pdf"

AVAILABLE_ARCHIVE_YEARS = [*range(1990, 2020), *range(2021, 2026)]
EXTRA_LOCAL_PDFS = ["1975.pdf", "1980.pdf", "1985.pdf", "1990.pdf", "1995.pdf", "2000.pdf", "2005.pdf", "2008.pdf", "2010.pdf", "2025.pdf", "current.pdf"]

COUNTRY_OVERRIDES = {
    "Aryna Sabalenka": "BLR",
    "Victoria Azarenka": "BLR",
    "Mirra Andreeva": "RUS",
    "Ekaterina Alexandrova": "RUS",
    "Liudmila Samsonova": "RUS",
    "Diana Shnaider": "RUS",
    "Anna Kalinskaya": "RUS",
    "Veronika Kudermetova": "RUS",
    "Anastasia Pavlyuchenkova": "RUS",
    "Daria Kasatkina": "AUS",
    "Amanda Anisimova": "USA",
}

COUNTRY_NORMALIZATION = {
    "FRG": "GER",
    "GDR": "GER",
    "URS": "RUS",
    "TCH": "CZE",
    "YUG": "SRB",
    "SCG": "SRB",
    "ROM": "ROU",
    "RSA": "ZAF",
    "SLO": "SLO",
}


@dataclass(frozen=True)
class RankingRow:
    ranking_date: str
    player_name: str
    country_code: str
    points: int
    rank: int


def _download_pdf(year: int | None) -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if year is None:
        path = RAW_DIR / "Singles_Numeric_current.pdf"
        url = CURRENT_URL
    else:
        path = RAW_DIR / f"Singles_Numeric_{year}.pdf"
        url = ARCHIVE_URL.format(year=year)
    if path.exists() and path.stat().st_size > 1000:
        return path

    response = requests.get(url, headers={"User-Agent": "youtube-videos-local/1.0"}, timeout=60)
    response.raise_for_status()
    if response.content[:4] != b"%PDF":
        raise RuntimeError(f"Not a PDF response for {url}")
    path.write_bytes(response.content)
    return path


def _fallback_year_from_path(pdf_path: Path) -> int | None:
    stem = pdf_path.stem
    match = re.search(r"(19\d{2}|20\d{2})", stem)
    if match:
        return int(match.group(1))
    if stem.isdigit() and len(stem) == 4:
        return int(stem)
    return None


def _discover_source_pdfs() -> list[tuple[Path, int | None]]:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for year in AVAILABLE_ARCHIVE_YEARS:
        _download_pdf(year)
    _download_pdf(None)

    paths = {path.resolve() for path in RAW_DIR.glob("*.pdf") if path.is_file()}
    for filename in EXTRA_LOCAL_PDFS:
        candidate = RAW_DIR / filename
        if candidate.exists():
            paths.add(candidate.resolve())
    ordered = sorted(paths, key=lambda item: item.name.lower())
    return [(path, _fallback_year_from_path(path)) for path in ordered]


def _extract_text(pdf_path: Path) -> str:
    doc = fitz.open(pdf_path)
    return "\n".join(page.get_text("text") for page in doc)


def _extract_ranking_date(text: str, fallback_year: int | None) -> str:
    patterns = [
        r"As of:\s*(\d{1,2})\s+([A-Za-z]+)\s+(\d{4})",
        r"RANK DATE:\s*(\d{1,2})\s+([A-Za-z]+)\s+(\d{4})",
        r"For:\s*(\d{1,2})\s+([A-Za-z]+)\s+(\d{4})",
        r"Printed:\s*(\d{1,2})\s+([A-Za-z]+)\s+(\d{4})",
        r"(\d{1,2})\s+([A-Za-z]+)\s+(\d{4})",
    ]
    month_map = {
        "january": 1,
        "february": 2,
        "march": 3,
        "april": 4,
        "may": 5,
        "june": 6,
        "july": 7,
        "august": 8,
        "september": 9,
        "october": 10,
        "november": 11,
        "december": 12,
    }
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.I)
        if not match:
            continue
        day = int(match.group(1))
        month = month_map.get(match.group(2).lower())
        year = int(match.group(3))
        if month:
            return date(year, month, day).isoformat()
    if fallback_year is not None:
        return f"{fallback_year}-12-31"
    return "2026-06-29"


def _title_name(raw_name: str) -> str:
    name = " ".join(raw_name.replace(".", " ").split())
    if "," in name:
        last, first = [part.strip() for part in name.split(",", 1)]
        name = f"{first} {last}"
    name = name.title()
    replacements = {
        "Arantx Sanchez Vicario": "Arantxa Sanchez Vicario",
        "Aran Sanchez Vicario": "Arantxa Sanchez Vicario",
        "Arantx Sanchez-Vicario": "Arantxa Sanchez Vicario",
        "Sanchez-Vicario Arantx": "Arantxa Sanchez Vicario",
        "Henin-Hardenne Justine": "Justine Henin",
        "Justine Henin-Hardenne": "Justine Henin",
        "Na Li": "Li Na",
        "Date Kimiko": "Kimiko Date",
        "Date Krumm Kimiko": "Kimiko Date-Krumm",
        "Garbi�E Muguruza": "Garbine Muguruza",
        "Carla Su�Rez Navarro": "Carla Suarez Navarro",
        "Bianca Vanessa Andreescu": "Bianca Andreescu",
        "Brend Schultz-Mccarthy": "Brenda Schultz-McCarthy",
        "Manu Maleeva-Fragniere": "Manuela Maleeva-Fragniere",
    }
    return replacements.get(name, name)


def _clean_country(country: str, player_name: str) -> str:
    code = country.strip().upper()
    code = COUNTRY_NORMALIZATION.get(code, code)
    return COUNTRY_OVERRIDES.get(player_name, code)


def _is_number(value: str) -> bool:
    return bool(re.fullmatch(r"\.?\d+(?:\.\d+)?", value.strip()))


def _parse_fixed_width(text: str, ranking_date: str) -> list[RankingRow]:
    rows: list[RankingRow] = []
    pattern = re.compile(
        r"^\*?\s*(\d+)\s+\([^)]+\)\s+(.+?)\s+([A-Z]{3})\s+"
        r"(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)",
        flags=re.M,
    )
    for match in pattern.finditer(text):
        rank = int(match.group(1))
        if rank > 80:
            continue
        player = _title_name(match.group(2))
        country = _clean_country(match.group(3), player)
        # In fixed-width WTA lists, the first numeric column is the sorting score.
        points = int(round(float(match.group(4))))
        rows.append(RankingRow(ranking_date, player, country, points, rank))
    return _dedupe_rows(rows)


def _parse_tokenized(text: str, ranking_date: str) -> list[RankingRow]:
    tokens = [line.strip() for line in text.splitlines() if line.strip()]
    rows: list[RankingRow] = []
    i = 0
    while i < len(tokens) - 4:
        if not re.fullmatch(r"\d{1,4}", tokens[i]):
            i += 1
            continue
        rank = int(tokens[i])
        if rank > 100:
            i += 1
            continue
        if not re.fullmatch(r"\(\d+[A-Z]?\)", tokens[i + 1]):
            i += 1
            continue

        name_token = tokens[i + 2]
        if not re.search(r"[A-Z]{2,}", name_token):
            i += 1
            continue
        player = _title_name(name_token)

        j = i + 3
        country = ""
        if j < len(tokens) and re.fullmatch(r"[A-Z]{3}", tokens[j]):
            country = tokens[j]
            j += 1

        if j >= len(tokens) or not _is_number(tokens[j]):
            i += 1
            continue
        points = int(round(float(tokens[j].replace(",", ""))))
        j += 1

        # In several WTA PDFs, RUS/BLR flags are intentionally omitted.
        if not country:
            for lookahead in range(j, min(j + 7, len(tokens))):
                if re.fullmatch(r"[A-Z]{3}", tokens[lookahead]):
                    country = tokens[lookahead]
                    break
        country = _clean_country(country, player)
        rows.append(RankingRow(ranking_date, player, country, points, rank))
        i = j
    return _dedupe_rows(rows)


def _parse_columnar(text: str, ranking_date: str) -> list[RankingRow]:
    tokens = [line.strip() for line in text.splitlines() if line.strip()]
    rows: list[RankingRow] = []
    row_start = 0
    for i in range(len(tokens) - 3):
        if not re.fullmatch(r"[A-Z]{3}", tokens[i]):
            continue
        if "," not in tokens[i + 1] or not re.search(r"[A-Z]{2,}", tokens[i + 1]):
            continue
        if not re.fullmatch(r"\d{1,4}", tokens[i + 2]):
            continue
        if not re.fullmatch(r"\(\d+[A-Z]?\)", tokens[i + 3]):
            continue

        rank = int(tokens[i + 2])
        if rank > 100:
            continue
        leading_numbers = [token for token in tokens[row_start:i] if _is_number(token)]
        if not leading_numbers:
            continue
        points = int(round(float(leading_numbers[0].replace(",", ""))))
        player = _title_name(tokens[i + 1])
        country = _clean_country(tokens[i], player)
        rows.append(RankingRow(ranking_date, player, country, points, rank))
        row_start = i + 4
    return _dedupe_rows(rows)


def _dedupe_rows(rows: list[RankingRow]) -> list[RankingRow]:
    best: dict[tuple[str, int, str], RankingRow] = {}
    for row in rows:
        key = (row.ranking_date, row.rank, row.player_name)
        if key not in best:
            best[key] = row
    return sorted(best.values(), key=lambda item: (item.ranking_date, item.rank, -item.points, item.player_name))


def _parse_pdf(pdf_path: Path, fallback_year: int | None) -> list[RankingRow]:
    text = _extract_text(pdf_path)
    ranking_date = _extract_ranking_date(text, fallback_year)
    rows = _parse_fixed_width(text, ranking_date)
    tokenized = _parse_tokenized(text, ranking_date)
    columnar = _parse_columnar(text, ranking_date)
    rows = max([rows, tokenized, columnar], key=len)
    rows = [row for row in rows if row.rank <= 20 and row.points > 0]
    if len(rows) < 10:
        raise RuntimeError(f"Could not parse enough ranking rows from {pdf_path}")
    return rows


def build_csv(output_csv: Path = OUTPUT_CSV) -> Path:
    rows: list[RankingRow] = []
    skipped: list[Path] = []
    for pdf_path, fallback_year in _discover_source_pdfs():
        try:
            rows.extend(_parse_pdf(pdf_path, fallback_year))
        except Exception as exc:
            skipped.append(pdf_path)
            print(f"[wta-ranking] skip {pdf_path.name}: {type(exc).__name__}")
    rows = _normalize_duplicate_early_ranks(rows)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["ranking_date", "player_name", "country_code", "points", "rank"])
        writer.writeheader()
        for row in sorted(rows, key=lambda item: (item.ranking_date, item.rank, item.player_name)):
            writer.writerow(
                {
                    "ranking_date": row.ranking_date,
                    "player_name": row.player_name,
                    "country_code": row.country_code,
                    "points": row.points,
                    "rank": row.rank,
                }
            )
    if skipped:
        print(f"[wta-ranking] skipped {len(skipped)} pdfs that were not parseable")
    return output_csv


def _normalize_duplicate_early_ranks(rows: list[RankingRow]) -> list[RankingRow]:
    grouped: dict[tuple[str, int], list[RankingRow]] = {}
    for row in rows:
        grouped.setdefault((row.ranking_date, row.rank), []).append(row)

    adjusted: list[RankingRow] = []
    for row in rows:
        peers = grouped.get((row.ranking_date, row.rank), [])
        if int(row.ranking_date[:4]) < 2000 and len(peers) > 1:
            target = min(peer.points for peer in peers)
            adjusted.append(RankingRow(row.ranking_date, row.player_name, row.country_code, target, row.rank))
        else:
            adjusted.append(row)
    return adjusted


def main() -> None:
    output = build_csv()
    print(f"[wta-ranking] wrote {output}")


if __name__ == "__main__":
    main()
