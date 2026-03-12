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
    "Primoz Roglic": ["Primoz Roglic", "Primoz Roglic cyclist", "Primoz Roglic (cyclist)"],
    "Tadej Pogacar": ["Tadej Pogacar", "Tadej Pogacar cyclist", "Tadej Pogacar (cyclist)"],
    "Michal Kwiatkowski": ["Michal Kwiatkowski", "Michal Kwiatkowski cyclist"],
    "Luis Leon Sanchez": ["Luis Leon Sanchez", "Luis Leon Sanchez cyclist", "Luis Leon Sanchez (cyclist)"],
    "Jorg Jaksche": ["Jorg Jaksche", "Jorg Jaksche cyclist"],
    "Andreas Kloden": ["Andreas Kloden", "Andreas Kloden cyclist"],
    "Alex Zulle": ["Alex Zulle", "Alex Zulle cyclist"],
}

MANUAL_TITLES = {
    "Gilbert Duclos-Lassalle": "Gilbert Duclos-Lassalle",
    "Jorg Jaksche": "Jörg Jaksche",
    "Floyd Landis": "Floyd Landis",
    "Alberto Contador": "Alberto Contador",
    "Luis Leon Sanchez": "Luis León Sánchez",
    "Tony Martin": "Tony Martin (cyclist)",
    "Bradley Wiggins": "Bradley Wiggins",
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


def _candidate_queries(name: str) -> list[str]:
    ascii_name = _ascii_name(name)
    queries: list[str] = []
    manual = MANUAL_QUERIES.get(ascii_name) or MANUAL_QUERIES.get(name) or []
    for candidate in manual + [ascii_name, name]:
        if not candidate:
            continue
        for query in (candidate, f"{candidate} cyclist", f"{candidate} (cyclist)", f"{candidate} cycling"):
            if query not in queries:
                queries.append(query)
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
        forced_title = MANUAL_TITLES.get(_ascii_name(name))
        if forced_title:
            try:
                image_url = _page_image_for_title(forced_title)
            except Exception:
                image_url = None
        for query in _candidate_queries(name):
            if image_url:
                break
            try:
                image_url = _page_image_for_title(query)
            except Exception:
                image_url = None
            if image_url:
                break
            time.sleep(1.2)
            try:
                for title in _search_titles(query):
                    try:
                        image_url = _page_image_for_title(title)
                    except Exception:
                        image_url = None
                    if image_url:
                        break
                    time.sleep(0.9)
            except Exception:
                image_url = None
            if image_url:
                break
            time.sleep(1.5)

        if not image_url:
            missing.append(name)
            continue

        try:
            _download(image_url, output_path)
            downloaded += 1
        except Exception:
            missing.append(name)
        time.sleep(3.5)

    print(f"[scraper] Paris-Nice winner photos: already={already} downloaded={downloaded} missing={len(missing)}")
    if missing:
        print("[scraper] missing winners:")
        for name in missing:
            print(f" - {name}")


if __name__ == "__main__":
    main()
