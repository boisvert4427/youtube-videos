from __future__ import annotations

import csv
import json
import re
import urllib.parse
import urllib.request
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_CSV = PROJECT_ROOT / "data" / "processed" / "atp_ranking_timeseries_2025_top15.csv"
PHOTOS_DIR = PROJECT_ROOT / "data" / "raw" / "player_photos"
ATP_API_BASE = "https://atp-rankings-data-visualization.onrender.com/api"
WIKIDATA_SPARQL = "https://query.wikidata.org/sparql"
USER_AGENT = "youtube-videos-local/1.0 (data build)"
WIKIPEDIA_TITLE_OVERRIDES = {
    "Felix Auger-Aliassime": "Félix Auger-Aliassime",
}

COUNTRY_CODE_NORMALIZATION = {
    "DEU": "GER",
    "YUG": "SRB",
}


def _fetch_json(url: str) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def _slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")


def _to_int_points(value: str) -> int:
    return int(value.replace(",", "").strip())


def _escape_sparql_string(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return escaped


def load_2025_top15_rows() -> list[dict]:
    weeks_payload = _fetch_json(f"{ATP_API_BASE}/weeks")
    weeks = weeks_payload.get("weeks", []) if isinstance(weeks_payload, dict) else weeks_payload
    week_dates: list[str] = []
    for item in weeks:
        if isinstance(item, dict):
            date_value = str(item.get("date", "")).strip()
        else:
            date_value = str(item).strip()
        if date_value.startswith("2025-"):
            week_dates.append(date_value)
    week_dates.sort()

    rows: list[dict] = []
    for ranking_date in week_dates:
        weekly = _fetch_json(f"{ATP_API_BASE}/week/{ranking_date}")
        rankings = weekly.get("rankings", [])[:15]
        for rank_row in rankings:
            rows.append(
                {
                    "ranking_date": ranking_date,
                    "rank": int(rank_row["rank"]),
                    "player_name": str(rank_row["name"]).strip(),
                    "points": _to_int_points(str(rank_row["points"])),
                }
            )
    return rows


def fetch_player_country_and_photo(player_names: list[str]) -> dict[str, dict]:
    data: dict[str, dict] = {}
    chunk_size = 20

    for i in range(0, len(player_names), chunk_size):
        chunk = player_names[i : i + chunk_size]
        values = " ".join(f'"{_escape_sparql_string(name)}"@en' for name in chunk)
        query = f"""
SELECT ?playerLabel ?countryCode ?image WHERE {{
  VALUES ?playerLabel {{ {values} }}
  ?player rdfs:label ?playerLabel .
  ?player wdt:P31 wd:Q5 .
  ?player wdt:P106 wd:Q10833314 .
  OPTIONAL {{ ?player wdt:P27 ?country . ?country wdt:P298 ?countryCode . }}
  OPTIONAL {{ ?player wdt:P18 ?image . }}
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
}}
"""
        url = f"{WIKIDATA_SPARQL}?{urllib.parse.urlencode({'query': query, 'format': 'json'})}"
        payload = _fetch_json(url)
        for item in payload.get("results", {}).get("bindings", []):
            name = item.get("playerLabel", {}).get("value", "").strip()
            if not name:
                continue
            data[name] = {
                "country_code": item.get("countryCode", {}).get("value", "").upper(),
                "image_url": item.get("image", {}).get("value", ""),
            }
    return data


def _download_file(url: str, path: Path) -> None:
    parsed = urllib.parse.urlsplit(url)
    safe_path = urllib.parse.quote(urllib.parse.unquote(parsed.path), safe="/%")
    safe_query = urllib.parse.quote_plus(urllib.parse.unquote_plus(parsed.query), safe="=&%")
    safe_url = urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, safe_path, safe_query, parsed.fragment))
    request = urllib.request.Request(safe_url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:
        content = response.read()
    path.write_bytes(content)


def _wikipedia_thumbnail_url(player_name: str) -> str:
    title = WIKIPEDIA_TITLE_OVERRIDES.get(player_name, player_name)
    params = urllib.parse.urlencode(
        {
            "action": "query",
            "titles": title,
            "prop": "pageimages",
            "format": "json",
            "pithumbsize": "900",
        }
    )
    payload = _fetch_json(f"https://en.wikipedia.org/w/api.php?{params}")
    pages = payload.get("query", {}).get("pages", {})
    for _, page in pages.items():
        thumb = page.get("thumbnail", {}).get("source", "")
        if thumb:
            return str(thumb)
    return ""


def save_photos(player_names: list[str], player_metadata: dict[str, dict]) -> None:
    PHOTOS_DIR.mkdir(parents=True, exist_ok=True)
    for player_name in player_names:
        meta = player_metadata.get(player_name, {})
        slug = _slugify(player_name)
        out_path = PHOTOS_DIR / f"{slug}.jpg"
        if out_path.exists():
            continue

        candidates: list[str] = []
        primary = str(meta.get("image_url", "")).strip()
        if primary:
            candidates.append(primary)
        fallback = _wikipedia_thumbnail_url(player_name)
        if fallback:
            candidates.append(fallback)

        for image_url in candidates:
            try:
                _download_file(image_url.replace("http://", "https://"), out_path)
                break
            except Exception:
                # Keep build resilient when one image fails.
                continue


def build_output_rows(raw_rows: list[dict], player_metadata: dict[str, dict]) -> list[dict]:
    country_fallback = {
        "Alex de Minaur": "AUS",
        "Andrey Rublev": "RUS",
        "Arthur Fils": "FRA",
        "Ben Shelton": "USA",
        "Casper Ruud": "NOR",
        "Carlos Alcaraz": "ESP",
        "Daniil Medvedev": "RUS",
        "Felix Auger-Aliassime": "CAN",
        "Frances Tiafoe": "USA",
        "Grigor Dimitrov": "BUL",
        "Holger Rune": "DEN",
        "Hubert Hurkacz": "POL",
        "Jack Draper": "GBR",
        "Jannik Sinner": "ITA",
        "Lorenzo Musetti": "ITA",
        "Novak Djokovic": "SRB",
        "Stefanos Tsitsipas": "GRE",
        "Taylor Fritz": "USA",
        "Tommy Paul": "USA",
        "Ugo Humbert": "FRA",
        "Alexander Zverev": "GER",
    }

    rows: list[dict] = []
    for item in raw_rows:
        name = item["player_name"]
        meta = player_metadata.get(name, {})
        country_code = str(meta.get("country_code", "")).upper() or country_fallback.get(name, "")
        country_code = COUNTRY_CODE_NORMALIZATION.get(country_code, country_code)
        rows.append(
            {
                "ranking_date": item["ranking_date"],
                "player_name": name,
                "country_code": country_code,
                "points": item["points"],
                "rank": item["rank"],
            }
        )
    rows.sort(key=lambda r: (r["ranking_date"], int(r["rank"])))
    return rows


def write_csv(rows: list[dict]) -> Path:
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["ranking_date", "player_name", "country_code", "points", "rank"]
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return OUTPUT_CSV


def run() -> tuple[Path, int, int]:
    raw_rows = load_2025_top15_rows()
    unique_players = sorted({row["player_name"] for row in raw_rows})
    metadata = fetch_player_country_and_photo(unique_players)
    save_photos(unique_players, metadata)
    output_rows = build_output_rows(raw_rows, metadata)
    output_path = write_csv(output_rows)
    return output_path, len(unique_players), len(raw_rows)


if __name__ == "__main__":
    output, players_count, rows_count = run()
    print(f"[scraper] generated -> {output}")
    print(f"[scraper] players: {players_count} | rows: {rows_count}")
