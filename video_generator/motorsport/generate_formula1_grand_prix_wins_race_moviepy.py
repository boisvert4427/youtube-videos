from __future__ import annotations

import argparse
import csv
import html
import json
import re
import sys
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from bs4 import BeautifulSoup


if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from video_generator.tennis import generate_wta_ranking_points_race_moviepy as base


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_INPUT = PROJECT_ROOT / "data" / "processed" / "motorsport" / "formula1_grand_prix_wins_1950_2026.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "motorsport" / "formula1_grand_prix_wins_race_1950_2026_3min.mp4"
DEFAULT_PREVIEW = PROJECT_ROOT / "data" / "processed" / "motorsport" / "formula1_grand_prix_wins_preview.png"
DEFAULT_BACKGROUND_IMAGE = Path(r"C:\Users\leona\Downloads\ChatGPT Image 13 juil. 2026, 14_59_48.png")
DEFAULT_PHOTOS_DIR = PROJECT_ROOT / "data" / "raw" / "formula1_driver_photos"
FORIX_URL = "https://8w.forix.com/6thgear/gp-yby.html"
WIKIPEDIA_URL = "https://en.wikipedia.org/wiki/List_of_Formula_One_Grand_Prix_winners"
USER_AGENT = "Mozilla/5.0 (compatible; Codex Formula1 photo downloader)"

TITLE = "FORMULA 1 GRAND PRIX WINS RACE"
SUBTITLE = "F1 DRIVERS - CUMULATIVE GRAND PRIX WINS | 1950-2026"
SEASON_SUMMARY = "World Championship era|Cumulative Grand Prix wins|1950-2026"

COUNTRY_NAME_TO_ALPHA3 = {
    "Argentina": "ARG",
    "Australia": "AUS",
    "Austria": "AUT",
    "Belgium": "BEL",
    "Brazil": "BRA",
    "Canada": "CAN",
    "Chile": "CHL",
    "China": "CHN",
    "Colombia": "COL",
    "Czech Republic": "CZE",
    "Denmark": "DEN",
    "Finland": "FIN",
    "France": "FRA",
    "Germany": "GER",
    "Hungary": "HUN",
    "Italy": "ITA",
    "Japan": "JPN",
    "Mexico": "MEX",
    "Monaco": "MON",
    "Netherlands": "NED",
    "New Zealand": "NZL",
    "Poland": "POL",
    "Portugal": "POR",
    "South Africa": "ZAF",
    "Spain": "ESP",
    "Sweden": "SWE",
    "Switzerland": "SUI",
    "United Kingdom": "GBR",
    "United States": "USA",
    "Uruguay": "URU",
    "Venezuela": "VEN",
}

DISPLAY_NAME_OVERRIDES = {
    "jose froilan gonzalez": "Jose Froilan Gonzalez",
    "emerson fittipaldi": "Emerson Fittipaldi",
    "jacques villeneuve": "Jacques Villeneuve",
    "michael schumacher": "Michael Schumacher",
    "nigel mansell": "Nigel Mansell",
    "alain prost": "Alain Prost",
    "max verstappen": "Max Verstappen",
    "lewis hamilton": "Lewis Hamilton",
    "lando norris": "Lando Norris",
    "oscar piastri": "Oscar Piastri",
    "fernando alonso": "Fernando Alonso",
    "charles leclerc": "Charles Leclerc",
    "george russell": "George Russell",
    "carlos sainz jr": "Carlos Sainz Jr",
    "carlos sainz jr.": "Carlos Sainz Jr",
    "ayrton senna": "Ayrton Senna",
    "nelson piquet": "Nelson Piquet",
}

MANUAL_DRIVER_COUNTRIES = {
    "alain prost": "FRA",
    "alberto ascari": "ITA",
    "ayrton senna": "BRA",
    "carlos sainz": "ESP",
    "carlos sainz jr": "ESP",
    "carlos sainz jr.": "ESP",
    "charles leclerc": "MON",
    "damon hill": "GBR",
    "eduardo regazzoni": "SUI",
    "emerson fittipaldi": "BRA",
    "fernando alonso": "ESP",
    "felipe massa": "BRA",
    "george russell": "GBR",
    "graham hill": "GBR",
    "jack brabham": "AUS",
    "jackie stewart": "GBR",
    "jacky ickx": "BEL",
    "jenson button": "GBR",
    "jim clark": "GBR",
    "john surtees": "GBR",
    "lee wallard": "USA",
    "jose froilan gonzalez": "ARG",
    "juan manuel fangio": "ARG",
    "juan pablo montoya": "COL",
    "kevin magnussen": "DEN",
    "kimi raikkonen": "FIN",
    "lando norris": "GBR",
    "lewis hamilton": "GBR",
    "max verstappen": "NED",
    "michael schumacher": "GER",
    "nelson piquet": "BRA",
    "niki lauda": "AUT",
    "nigel mansell": "GBR",
    "nico rosberg": "GER",
    "oscar piastri": "AUS",
    "pierre gasly": "FRA",
    "rubens barrichello": "BRA",
    "sebastian vettel": "GER",
    "sergio perez": "MEX",
    "stirling moss": "GBR",
    "stefan johansson": "SWE",
    "valtteri bottas": "FIN",
}


def _normalize_key(value: str) -> str:
    value = unicodedata.normalize("NFKD", value)
    value = value.encode("ascii", "ignore").decode("ascii")
    value = re.sub(r"[^a-zA-Z0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip().lower()


def _clean_name(value: str) -> str:
    value = html.unescape(value or "")
    value = re.sub(r"[\u2020\u2021\u2022]", "", value)
    value = re.sub(r"\s+", " ", value).strip()
    return DISPLAY_NAME_OVERRIDES.get(_normalize_key(value), value)


def _country_to_alpha3(country_name: str) -> str:
    return COUNTRY_NAME_TO_ALPHA3.get(country_name.strip(), "")


def _request_json(url: str) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    delay = 1.0
    for attempt in range(6):
        try:
            with urllib.request.urlopen(request, timeout=45) as response:
                return json.loads(response.read().decode("utf-8"))
        except (urllib.error.HTTPError, urllib.error.URLError):
            if attempt == 5:
                raise
            time.sleep(delay)
            delay *= 2.0


def _request_bytes(url: str) -> tuple[bytes, str]:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    delay = 1.0
    for attempt in range(6):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return response.read(), response.headers.get_content_type()
        except (urllib.error.HTTPError, urllib.error.URLError):
            if attempt == 5:
                raise
            time.sleep(delay)
            delay *= 2.0


def _driver_country_map_from_wikipedia() -> dict[str, str]:
    response = requests.get(WIKIPEDIA_URL, headers={"User-Agent": "Mozilla/5.0"}, timeout=60)
    response.raise_for_status()
    response.encoding = response.apparent_encoding or response.encoding or "utf-8"
    soup = BeautifulSoup(response.text, "lxml")
    for table in soup.find_all("table", class_="wikitable"):
        headers = [th.get_text(" ", strip=True) for th in table.find_all("th")]
        if "Driver" not in headers or "Country" not in headers:
            continue
        driver_col = headers.index("Driver")
        country_col = headers.index("Country")
        mapping: dict[str, str] = {}
        for row in table.find_all("tr")[1:]:
            cells = row.find_all(["th", "td"])
            if len(cells) <= max(driver_col, country_col):
                continue
            driver = _clean_name(cells[driver_col].get_text(" ", strip=True))
            country_name = cells[country_col].get_text(" ", strip=True)
            alpha3 = _country_to_alpha3(country_name)
            if driver and alpha3:
                mapping[_normalize_key(driver)] = alpha3
        if mapping:
            return mapping
    return {}


def _build_input_rows() -> list[dict[str, str]]:
    response = requests.get(FORIX_URL, headers={"User-Agent": "Mozilla/5.0"}, timeout=60)
    response.raise_for_status()
    response.encoding = response.apparent_encoding or response.encoding or "iso-8859-1"
    soup = BeautifulSoup(response.text, "lxml")
    driver_country_map = _driver_country_map_from_wikipedia()

    wins: dict[str, int] = defaultdict(int)
    display_names: dict[str, str] = {}
    driver_order: list[str] = []
    rows: list[dict[str, str]] = []

    for h2 in soup.find_all("h2"):
        anchor = h2.find("a", attrs={"name": re.compile(r"^\d{4}$")})
        if anchor is None:
            continue
        year = int(anchor.get("name", "0"))
        if year < 1950 or year > 2026:
            continue

        node = h2.find_next_sibling()
        while node is not None and node.name != "h2":
            if node.name == "ul":
                for li in node.find_all("li", recursive=False):
                    if li.find("strong") is None:
                        continue
                    text = li.get_text(" ", strip=True)
                    match = re.search(r",\s*(\d{1,2})-(\d{1,2}):\s*(.+?),\s*[^,]+$", text)
                    if not match:
                        continue
                    day = int(match.group(1))
                    month = int(match.group(2))
                    winner = _clean_name(match.group(3))
                    if "/" in winner:
                        continue
                    key = _normalize_key(winner)
                    display_names.setdefault(key, winner)
                    if key not in driver_order:
                        driver_order.append(key)
                    wins[key] += 1

                    race_date = f"{year:04d}-{month:02d}-{day:02d}"
                    snapshot_drivers = [k for k in driver_order if wins.get(k, 0) > 0]
                    snapshot_drivers.sort(key=lambda item: (-wins[item], display_names[item]))
                    for driver_key in snapshot_drivers:
                        rows.append(
                            {
                                "ranking_date": race_date,
                                "player_name": display_names[driver_key],
                                "country_code": driver_country_map.get(driver_key, MANUAL_DRIVER_COUNTRIES.get(driver_key, "")),
                                "points": str(wins[driver_key]),
                                "season_summary": SEASON_SUMMARY,
                            }
                        )
            node = node.find_next_sibling()

    return rows


def build_input_csv(input_csv: Path) -> Path:
    input_csv.parent.mkdir(parents=True, exist_ok=True)
    rows = _build_input_rows()
    with input_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["ranking_date", "player_name", "country_code", "points", "season_summary"],
        )
        writer.writeheader()
        writer.writerows(rows)
    return input_csv


def _load_top_names(input_csv: Path, top_n: int = 12) -> list[str]:
    with input_csv.open("r", newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    unique: dict[str, None] = {}
    for ranking_date in sorted({row["ranking_date"] for row in rows}):
        year_rows = [row for row in rows if row["ranking_date"] == ranking_date][:top_n]
        for row in year_rows:
            name = row["player_name"].strip()
            if name:
                unique.setdefault(name, None)
    return list(unique.keys())


def _load_snapshots(input_csv: Path) -> list[tuple[str, list[dict[str, str]]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    with input_csv.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            grouped[row["ranking_date"].strip()].append(row)
    snapshots: list[tuple[str, list[dict[str, str]]]] = []
    for ranking_date in sorted(grouped.keys()):
        states = sorted(
            grouped[ranking_date],
            key=lambda row: (-float(row["points"]), row["player_name"].strip()),
        )
        snapshots.append((ranking_date, states))
    return snapshots


def _search_wikidata(name: str) -> dict | None:
    queries = [name]
    ascii_name = _normalize_key(name)
    if ascii_name and ascii_name != _normalize_key(name):
        queries.insert(0, ascii_name)
    for query in queries:
        search_url = "https://www.wikidata.org/w/api.php?" + urllib.parse.urlencode(
            {
                "action": "wbsearchentities",
                "search": query,
                "language": "en",
                "format": "json",
                "limit": 10,
            }
        )
        data = _request_json(search_url)
        results = data.get("search", [])
        if not results:
            continue
        preferred = []
        for item in results:
            description = (item.get("description") or "").lower()
            if any(token in description for token in ("formula one driver", "racing driver", "motor racing driver")):
                preferred.append(item)
        return preferred[0] if preferred else results[0]
    return None


def _get_enwiki_title(qid: str) -> str | None:
    entity_url = f"https://www.wikidata.org/wiki/Special:EntityData/{qid}.json"
    data = _request_json(entity_url)
    entity = data.get("entities", {}).get(qid, {})
    enwiki = entity.get("sitelinks", {}).get("enwiki")
    if enwiki:
        return enwiki.get("title")
    return None


def _get_wikidata_image_filename(qid: str) -> str | None:
    entity_url = f"https://www.wikidata.org/wiki/Special:EntityData/{qid}.json"
    data = _request_json(entity_url)
    entity = data.get("entities", {}).get(qid, {})
    p18 = entity.get("claims", {}).get("P18", [])
    if not p18:
        return None
    try:
        value = p18[0]["mainsnak"]["datavalue"]["value"]
    except (KeyError, IndexError, TypeError):
        return None
    return value.strip() if isinstance(value, str) and value.strip() else None


def _get_thumbnail_url(title: str) -> str | None:
    summary_url = "https://en.wikipedia.org/api/rest_v1/page/summary/" + urllib.parse.quote(title.replace(" ", "_"))
    data = _request_json(summary_url)
    for key in ("thumbnail", "originalimage"):
        image = data.get(key) or {}
        source = image.get("source")
        if source:
            return source
    return None


def _get_commons_file_url(filename: str) -> str:
    return "https://commons.wikimedia.org/wiki/Special:FilePath/" + urllib.parse.quote(filename)


def _download_driver_photo(name: str, output_dir: Path) -> tuple[str, str]:
    slug = re.sub(r"[^a-z0-9]+", "_", _normalize_key(name)).strip("_")
    existing = next((path for path in output_dir.glob(f"{slug}.*") if path.is_file()), None)
    if existing is not None:
        return name, f"exists:{existing.name}"
    try:
        search = _search_wikidata(name)
        if not search:
            return name, "missing:wikidata"
        qid = search["id"]
        title = _get_enwiki_title(qid)
        image_url = None
        if title:
            try:
                image_url = _get_thumbnail_url(title)
            except Exception:
                image_url = None
        if not image_url:
            filename = _get_wikidata_image_filename(qid)
            if filename:
                image_url = _get_commons_file_url(filename)
        if not image_url:
            return name, f"missing:image:{qid}"
        image_bytes, content_type = _request_bytes(image_url)
        ext = Path(urllib.parse.urlparse(image_url).path).suffix.lower()
        if ext not in {".jpg", ".jpeg", ".png", ".webp"}:
            ext = ".jpg" if content_type == "image/jpeg" else ".png"
        output_path = output_dir / f"{slug}{ext}"
        output_path.write_bytes(image_bytes)
        return name, f"downloaded:{output_path.name}"
    except Exception as exc:
        return name, f"error:{type(exc).__name__}"


def ensure_driver_photos(names: list[str], photos_dir: Path, workers: int = 3) -> None:
    photos_dir.mkdir(parents=True, exist_ok=True)
    names = [name for name in names if name.strip()]
    results: list[tuple[str, str]] = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {executor.submit(_download_driver_photo, name, photos_dir): name for name in names}
        for future in as_completed(futures):
            results.append(future.result())
    downloaded = sum(1 for _, status in results if status.startswith("downloaded:"))
    cached = sum(1 for _, status in results if status.startswith("exists:"))
    print(f"[photos] drivers={len(names)} downloaded={downloaded} cached={cached}")


def _infer_duration_months(input_csv: Path, seconds_per_month: float) -> float:
    with input_csv.open("r", newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    dates = sorted({row["ranking_date"] for row in rows})
    if len(dates) < 2:
        return 180.0
    first_year, first_month = map(int, dates[0].split("-")[:2])
    last_year, last_month = map(int, dates[-1].split("-")[:2])
    months = (last_year - first_year) * 12 + (last_month - first_month) + 1
    return max(30.0, months * seconds_per_month)


def render_video(
    input_csv: Path,
    output_path: Path,
    preview_path: Path | None,
    preview_time: float,
    flags_dir: Path,
    photos_dir: Path,
    background_image: Path | None,
    duration: float,
    fps: int,
    top_n: int,
    seconds_per_month: float,
) -> Path:
    snapshots = _load_snapshots(input_csv)
    if duration <= 0:
        duration = _infer_duration_months(input_csv, seconds_per_month)

    if preview_path is not None and snapshots:
        periods = len(snapshots)
        seconds_per_period = duration / max(periods, 1)
        period_index = min(int(preview_time / max(seconds_per_period, 1e-6)), periods - 1)
        preview_names = {
            row["player_name"]
            for _, states in snapshots[max(period_index - 1, 0) : min(period_index + 2, periods)]
            for row in states[:top_n]
        }
        ensure_driver_photos(sorted(preview_names), photos_dir)
    else:
        ensure_driver_photos(_load_top_names(input_csv, top_n=top_n), photos_dir)

    base.TITLE = TITLE
    base.SUBTITLE = SUBTITLE
    return base.render_video(
        input_csv=input_csv,
        output_path=output_path,
        preview_path=preview_path,
        preview_time=preview_time,
        flags_dir=flags_dir,
        photos_dir=photos_dir,
        background_image=background_image,
        wimbledon_logo=None,
        audio_path=base.DEFAULT_AUDIO,
        duration=duration,
        fps=fps,
        top_n=top_n,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a Formula 1 Grand Prix wins bar chart race video.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--preview", type=Path, default=None)
    parser.add_argument("--preview-time", type=float, default=120.0)
    parser.add_argument("--flags-dir", type=Path, default=base.DEFAULT_FLAGS_DIR)
    parser.add_argument("--photos-dir", type=Path, default=DEFAULT_PHOTOS_DIR)
    parser.add_argument("--background-image", type=Path, default=DEFAULT_BACKGROUND_IMAGE)
    parser.add_argument("--duration", type=float, default=0.0)
    parser.add_argument("--seconds-per-month", type=float, default=2.0)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--top-n", type=int, default=12)
    parser.add_argument("--refresh-data", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_csv = args.input
    if args.refresh_data or not input_csv.exists():
        input_csv = build_input_csv(input_csv)

    output = render_video(
        input_csv=input_csv,
        output_path=args.output,
        preview_path=args.preview,
        preview_time=args.preview_time,
        flags_dir=args.flags_dir,
        photos_dir=args.photos_dir,
        background_image=args.background_image,
        duration=args.duration,
        fps=args.fps,
        top_n=args.top_n,
        seconds_per_month=args.seconds_per_month,
    )
    if args.preview is not None:
        print(f"[video_generator] Formula 1 preview generated -> {output}")
    else:
        print(f"[video_generator] Formula 1 race generated -> {output}")


if __name__ == "__main__":
    main()
