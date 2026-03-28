from __future__ import annotations

import csv
import json
import re
import time
import unicodedata
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
INPUT_CSV = PROJECT_ROOT / "data" / "processed" / "cycling" / "cycling_monuments_timeseries_1892_2025.csv"
PHOTOS_DIR = PROJECT_ROOT / "data" / "raw" / "player_photos"
USER_AGENT = "Mozilla/5.0 (compatible; Codex Cycling Monuments photo downloader)"
TOP_N = 12


MANUAL_TITLES = {
    "Albert Champion": "Albert Champion",
    "Alejandro Valverde": "Alejandro Valverde",
    "Alfons Schepers": "Alfons Schepers",
    "Alfredo Binda": "Alfredo Binda",
    "Bernard Hinault": "Bernard Hinault",
    "Carlo Oriani": "Carlo Oriani",
    "Charles Crupelandt": "Charles Crupelandt",
    "Costante Girardengo": "Costante Girardengo",
    "Erik Zabel": "Erik Zabel",
    "Eugene Christophe": "Eugène Christophe",
    "Fabian Cancellara": "Fabian Cancellara",
    "Fausto Coppi": "Fausto Coppi",
    "Felice Gimondi": "Felice Gimondi",
    "Fiorenzo Magni": "Fiorenzo Magni",
    "Francesco Moser": "Francesco Moser",
    "Gaston Rebry": "Gaston Rebry",
    "Gaetano Belloni": "Gaetano Belloni",
    "Giovanni Cuniolo": "Giovanni Cuniolo",
    "Gino Bartali": "Gino Bartali",
    "Hennie Kuiper": "Hennie Kuiper",
    "Henri Pelissier": "Henri Pélissier",
    "Jan Raas": "Jan Raas",
    "Johan Museeuw": "Johan Museeuw",
    "Louis Mottiat": "Louis Mottiat",
    "Mathieu van der Poel": "Mathieu van der Poel",
    "Michele Bartoli": "Michele Bartoli",
    "Moreno Argentin": "Moreno Argentin",
    "Philippe Gilbert": "Philippe Gilbert",
    "Rene Vermandel": "René Vermandel",
    "Rik Van Looy": "Rik Van Looy",
    "Rik Van Steenbergen": "Rik Van Steenbergen",
    "Romain Gijssels": "Romain Gijssels",
    "Roger De Vlaeminck": "Roger De Vlaeminck",
    "Sean Kelly": "Sean Kelly",
    "Tadej Pogacar": "Tadej Pogačar",
    "Tom Boonen": "Tom Boonen",
}


def _slugify(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", normalized.strip().lower())
    return cleaned.strip("_")


def _ascii_name(value: str) -> str:
    return unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii").strip()


def _fetch_json(url: str) -> dict:
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8", "ignore"))


def _page_image_for_title(title: str) -> str | None:
    summary_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{quote(title, safe='')}"
    try:
        summary = _fetch_json(summary_url)
        thumbnail = summary.get("thumbnail", {}).get("source")
        if thumbnail:
            return thumbnail
    except Exception:
        pass
    url = (
        "https://en.wikipedia.org/w/api.php?action=query&prop=pageimages&piprop=original|thumbnail&pithumbsize=1600"
        f"&titles={quote(title)}&format=json&formatversion=2"
    )
    data = _fetch_json(url)
    page = data.get("query", {}).get("pages", [{}])[0]
    return page.get("thumbnail", {}).get("source") or page.get("original", {}).get("source")


def _search_titles(query: str, limit: int = 6) -> list[str]:
    url = (
        "https://en.wikipedia.org/w/api.php?action=query&list=search"
        f"&srsearch={quote(query)}&format=json&utf8=1&srlimit={limit}"
    )
    data = _fetch_json(url)
    return [item["title"] for item in data.get("query", {}).get("search", [])]


def _download(url: str, output_path: Path) -> None:
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=60) as response:
        output_path.write_bytes(response.read())


def _top_riders(input_csv: Path, top_n: int) -> list[str]:
    snapshot_rows: dict[str, list[tuple[str, int]]] = {}
    with input_csv.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            ranking_date = row["ranking_date"].strip()
            snapshot_rows.setdefault(ranking_date, []).append((row["player_name"].strip(), int(float(row["points"]))))
    riders: list[str] = []
    seen: set[str] = set()
    for ranking_date in sorted(snapshot_rows.keys()):
        ranked = sorted(snapshot_rows[ranking_date], key=lambda item: (-item[1], item[0]))[:top_n]
        for rider_name, _ in ranked:
            if rider_name not in seen:
                seen.add(rider_name)
                riders.append(rider_name)
    return riders


def _candidate_titles(name: str) -> list[str]:
    ascii_name = _ascii_name(name)
    manual = MANUAL_TITLES.get(name) or MANUAL_TITLES.get(ascii_name)
    candidates: list[str] = []
    if manual:
        candidates.append(manual)
    for value in (
        name,
        ascii_name,
        f"{name} cyclist",
        f"{ascii_name} cyclist",
        f"{name} (cyclist)",
        f"{ascii_name} (cyclist)",
        f"{name} road cyclist",
        f"{ascii_name} road cyclist",
    ):
        if value and value not in candidates:
            candidates.append(value)
    return candidates


def main() -> None:
    PHOTOS_DIR.mkdir(parents=True, exist_ok=True)
    riders = _top_riders(INPUT_CSV, TOP_N)

    already = 0
    downloaded = 0
    missing: list[str] = []

    for name in riders:
        output_path = PHOTOS_DIR / f"{_slugify(name)}.jpg"
        if output_path.exists():
            already += 1
            continue

        image_url = None
        for candidate in _candidate_titles(name):
            try:
                image_url = _page_image_for_title(candidate)
            except Exception:
                image_url = None
            if image_url:
                break
            time.sleep(0.4)
            try:
                for result_title in _search_titles(candidate):
                    image_url = _page_image_for_title(result_title)
                    if image_url:
                        break
                    time.sleep(0.3)
            except Exception:
                image_url = None
            if image_url:
                break
            time.sleep(0.6)

        if not image_url:
            missing.append(name)
            continue

        try:
            _download(image_url, output_path)
            downloaded += 1
        except Exception:
            missing.append(name)
        time.sleep(1.0)

    print(f"[scraper] Cycling Monuments rider photos: already={already} downloaded={downloaded} missing={len(missing)}")
    if missing:
        print("[scraper] missing riders:")
        for name in missing:
            print(f" - {name}")


if __name__ == "__main__":
    main()
