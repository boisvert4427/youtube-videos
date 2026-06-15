from __future__ import annotations

import argparse
import csv
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

import requests
from bs4 import BeautifulSoup, Tag


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "finance"
    / "public_corporations_market_capitalization"
    / "public_corporations_market_capitalization_2000_2026.csv"
)
SOURCE_URL = "https://en.wikipedia.org/wiki/List_of_public_corporations_by_market_capitalization"
USER_AGENT = "Mozilla/5.0 (compatible; Codex Market Cap Builder)"

ANNUAL_YEAR_DATES = {
    2000: "2000-12-31",
    2001: "2001-12-31",
    2002: "2002-12-31",
    2003: "2003-12-31",
    2004: "2004-12-31",
    2005: "2005-12-31",
}

COUNTRY_ALIASES = {
    "brunei darussalam": "Brunei",
    "china": "China",
    "hong kong": "Hong Kong",
    "japan": "Japan",
    "netherlands": "Netherlands",
    "netherlands united kingdom": "Netherlands / United Kingdom",
    "saudi arabia": "Saudi Arabia",
    "south korea": "South Korea",
    "switzerland": "Switzerland",
    "taiwan": "Taiwan",
    "united kingdom": "United Kingdom",
    "united states": "United States",
}


def _normalize(text: str) -> str:
    cleaned = unicodedata.normalize("NFKD", text or "")
    cleaned = "".join(character for character in cleaned if not unicodedata.combining(character))
    cleaned = cleaned.lower().strip()
    cleaned = re.sub(r"[^a-z0-9]+", " ", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def _clean_text(value: str) -> str:
    value = (value or "").replace("\xa0", " ")
    value = re.sub(r"\[\s*\d+\s*\]", "", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def _extract_number(text: str) -> float:
    cleaned = _clean_text(text)
    matches = re.findall(r"[\d,]+(?:\.\d+)?", cleaned)
    if not matches:
        raise ValueError(f"Could not parse numeric value from: {text!r}")
    return float(matches[-1].replace(",", ""))


def _extract_company_name(cell: Tag) -> str:
    anchor = cell.find("a")
    if anchor is not None:
        name = _clean_text(anchor.get_text(" ", strip=True))
        if name:
            return name
    text = _clean_text(cell.get_text(" ", strip=True))
    text = re.sub(r"[\d,]+(?:\.\d+)?", "", text)
    text = text.replace("|", " ")
    text = re.sub(r"\s+", " ", text).strip(" -")
    return text


def _extract_country_name(cell: Tag) -> str:
    anchor = cell.find("a")
    if anchor is not None:
        title = anchor.get("title") or anchor.get_text(" ", strip=True)
    else:
        title = cell.get_text(" ", strip=True)
    title = _clean_text(title)
    return COUNTRY_ALIASES.get(_normalize(title), title)


def _quarter_date(year: int, quarter_index: int) -> str:
    month_day = {1: "03-31", 2: "06-30", 3: "09-30", 4: "12-31"}
    return f"{year:04d}-{month_day[quarter_index]}"


def _fetch_page() -> BeautifulSoup:
    response = requests.get(
        SOURCE_URL,
        headers={"User-Agent": USER_AGENT},
        timeout=120,
    )
    response.raise_for_status()
    return BeautifulSoup(response.text, "html.parser")


def _section_table(year_heading: Tag) -> Tag:
    table = year_heading.find_next("table")
    if table is None:
        raise RuntimeError(f"Could not find a table after year heading {year_heading.get_text(strip=True)!r}.")
    return table


def _parse_annual_section(year: int, table: Tag) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for tr in table.find_all("tr")[1:]:
        cells = tr.find_all(["th", "td"], recursive=False)
        if len(cells) < 5:
            continue
        try:
            rank = int(_clean_text(cells[0].get_text(" ", strip=True)))
            company_name = _extract_company_name(cells[1])
            country_name = _extract_country_name(cells[2])
            industry = _clean_text(cells[3].get_text(" ", strip=True))
            market_cap = _extract_number(cells[4].get_text(" ", strip=True))
        except ValueError:
            continue

        rows.append(
            {
                "ranking_date": ANNUAL_YEAR_DATES.get(year, f"{year:04d}-12-31"),
                "period_label": f"{year}",
                "rank": str(rank),
                "company_name": company_name,
                "country_name": country_name,
                "industry": industry,
                "market_cap_usd_m": str(int(round(market_cap))),
                "data_source": "Wikipedia / Financial Times",
            }
        )
    return rows


def _parse_quarterly_section(year: int, table: Tag) -> list[dict[str, str]]:
    header_cells = table.find_all("tr")[0].find_all(["th", "td"], recursive=False)
    quarter_count = max(0, len(header_cells) - 1)
    rows: list[dict[str, str]] = []

    for tr in table.find_all("tr")[1:]:
        cells = tr.find_all(["th", "td"], recursive=False)
        if len(cells) < 1 + quarter_count * 2:
            continue
        try:
            rank = int(_clean_text(cells[0].get_text(" ", strip=True)))
        except ValueError:
            continue

        for quarter_index in range(quarter_count):
            flag_cell = cells[1 + quarter_index * 2]
            company_cell = cells[2 + quarter_index * 2]
            company_text = _clean_text(company_cell.get_text(" ", strip=True))
            if not company_text:
                continue
            try:
                company_name = _extract_company_name(company_cell)
                market_cap = _extract_number(company_cell.get_text(" ", strip=True))
            except ValueError:
                continue

            country_name = _extract_country_name(flag_cell)
            rows.append(
                {
                    "ranking_date": _quarter_date(year, quarter_index + 1),
                    "period_label": f"{year} Q{quarter_index + 1}",
                    "rank": str(rank),
                    "company_name": company_name,
                    "country_name": country_name,
                    "industry": "",
                    "market_cap_usd_m": str(int(round(market_cap))),
                    "data_source": "Wikipedia / Financial Times",
                }
            )

    return rows


def build_rows() -> list[dict[str, str]]:
    soup = _fetch_page()
    rows: list[dict[str, str]] = []

    for heading in soup.find_all("h3"):
        year_text = _clean_text(heading.get_text(" ", strip=True))
        if not re.fullmatch(r"\d{4}", year_text):
            continue
        year = int(year_text)
        table = _section_table(heading)
        header_text = _clean_text(table.find_all("tr")[0].get_text(" ", strip=True)).lower()
        if "quarter" in header_text:
            rows.extend(_parse_quarterly_section(year, table))
        else:
            rows.extend(_parse_annual_section(year, table))

    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["ranking_date"]].append(row)

    final_rows: list[dict[str, str]] = []
    for ranking_date in sorted(grouped):
        ranked = sorted(grouped[ranking_date], key=lambda item: (int(item["rank"]), item["company_name"]))
        if ranked:
            leader = ranked[0]
            leader_cap = int(leader["market_cap_usd_m"])
            summary = f"Leader: {leader['company_name']} ${leader_cap:,}M|Wikipedia market cap list"
        else:
            summary = "Wikipedia market cap list"
        for row in ranked:
            row["season_summary"] = summary
            final_rows.append(row)

    return final_rows


def write_csv(rows: list[dict[str, str]], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "ranking_date",
                "period_label",
                "rank",
                "company_name",
                "country_name",
                "industry",
                "market_cap_usd_m",
                "season_summary",
                "data_source",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a market cap timeseries from the Wikipedia public corporations list.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = build_rows()
    output = write_csv(rows, args.output)
    snapshots = len({row["ranking_date"] for row in rows})
    print(f"[scraper] Wikipedia market cap CSV generated -> {output}")
    print(f"[scraper] {len(rows)} rows, {snapshots} snapshots")


if __name__ == "__main__":
    main()
