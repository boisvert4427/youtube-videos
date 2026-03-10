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
INPUT_CSV = PROJECT_ROOT / "data" / "processed" / "cycling" / "paris_nice" / "paris_nice_timeline_postwar_template.csv"
PHOTOS_DIR = PROJECT_ROOT / "data" / "raw" / "player_photos"
USER_AGENT = "Mozilla/5.0 (compatible; Codex Paris-Nice photo downloader)"


MANUAL_QUERIES = {
    "Primož Roglič": ["Primoz Roglic", "Primož Roglič"],
    "Tadej Pogačar": ["Tadej Pogacar", "Tadej Pogačar"],
    "Michał Kwiatkowski": ["Michal Kwiatkowski", "Michał Kwiatkowski"],
    "Luis León Sánchez": ["Luis Leon Sanchez", "Luis León Sánchez"],
    "Jörg Jaksche": ["Jorg Jaksche", "Jörg Jaksche"],
    "Andreas Klöden": ["Andreas Kloden", "Andreas Klöden"],
    "Alex Zülle": ["Alex Zulle", "Alex Zülle"],
}


def _slugify(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", normalized.strip().lower())
    return cleaned.strip("_")


def _fetch_json(url: str) -> dict:
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8", "ignore"))


def _page_image_for_title(title: str) -> str | None:
    url = (
        "https://en.wikipedia.org/w/api.php?action=query&prop=pageimages&piprop=original"
        f"&titles={quote(title)}&format=json&formatversion=2"
    )
    data = _fetch_json(url)
    page = data.get("query", {}).get("pages", [{}])[0]
    return page.get("original", {}).get("source")


def _search_titles(query: str, limit: int = 5) -> list[str]:
    url = (
        "https://en.wikipedia.org/w/api.php?action=query&list=search"
        f"&srsearch={quote(query)}&format=json&utf8=1&srlimit={limit}"
    )
    data = _fetch_json(url)
    return [item["title"] for item in data.get("query", {}).get("search", [])]


def _candidate_queries(name: str) -> list[str]:
    if name in MANUAL_QUERIES:
        return MANUAL_QUERIES[name]
    ascii_name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii").strip()
    queries = [name]
    if ascii_name and ascii_name != name:
        queries.insert(0, ascii_name)
    return queries


def _download(url: str, output_path: Path) -> None:
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=60) as response:
        output_path.write_bytes(response.read())


def unique_winners(input_csv: Path) -> list[str]:
    winners: list[str] = []
    seen: set[str] = set()
    with input_csv.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            name = row.get("winner_name", "").strip()
            if not name or name == "-" or name in seen:
                continue
            seen.add(name)
            winners.append(name)
    return winners


def main() -> None:
    PHOTOS_DIR.mkdir(parents=True, exist_ok=True)
    winners = unique_winners(INPUT_CSV)
    downloaded = 0
    already = 0
    missing: list[str] = []

    for name in winners:
        output_path = PHOTOS_DIR / f"{_slugify(name)}.jpg"
        if output_path.exists():
            already += 1
            continue

        image_url = None
        for query in _candidate_queries(name):
            try:
                image_url = _page_image_for_title(query)
            except Exception:
                image_url = None
            if image_url:
                break
            time.sleep(0.8)
            try:
                for title in _search_titles(query):
                    try:
                        image_url = _page_image_for_title(title)
                    except Exception:
                        image_url = None
                    if image_url:
                        break
                    time.sleep(0.6)
            except Exception:
                image_url = None
            if image_url:
                break
            time.sleep(1.2)

        if not image_url:
            missing.append(name)
            continue

        try:
            _download(image_url, output_path)
            downloaded += 1
        except Exception:
            missing.append(name)
        time.sleep(1.5)

    print(f"[scraper] Paris-Nice winner photos: already={already} downloaded={downloaded} missing={len(missing)}")
    if missing:
        print("[scraper] missing winners:")
        for name in missing:
            print(f" - {name}")


if __name__ == "__main__":
    main()
