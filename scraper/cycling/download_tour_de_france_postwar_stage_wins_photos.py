from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
import unicodedata
from io import BytesIO
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen

from PIL import Image, ImageOps


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_INPUT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "cycling"
    / "tour_de_france"
    / "tour_de_france_stage_wins_postwar_1947_2025.csv"
)
DEFAULT_PHOTOS_DIR = PROJECT_ROOT / "data" / "raw" / "player_photos"
USER_AGENT = "Mozilla/5.0 (compatible; Codex Tour de France photo downloader)"
TOP_N = 12
MANUAL_IMAGE_URLS = {
    "Aldo Ronconi": "https://blogger.googleusercontent.com/img/b/R29vZ2xl/AVvXsEiRwhHRxhhoVIjyevM-Iw0VsV17he7TgixdsQR3F5CCPEwcrGmZSOhduyMdSGnHNfeTfyu042wZl61IeJ7JadqMZ30UAwdGy7eW3MCgAalfW9mJc0GqMMZVh5xgcLhrtbRJR9WZ166h9cAL/s1600/AldoRonconi.jpg",
    "Bernard Gauthier": "https://www.ledicodutour.com/images/coureurs/a_gauthier.jpg",
    "Giovanni Corrieri": "https://commons.wikimedia.org/wiki/Special:FilePath/Giovanni_Corrieri.jpg?width=640",
    "Guy Lapébie": "https://www.ledicodutour.com/images/coureurs/a_lapebie.jpg",
    "Giuseppe Tacca": "https://www.ledicodutour.com/images/coureurs/a_tacca.jpg",
    "Nino Defilippis": "https://www.ledicodutour.com/images/coureurs/defilippis.jpg",
    "Pietro Tarchini": "https://www.ledicodutour.com/images/coureurs/tarchini48.jpg",
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


def _fetch_text(url: str) -> str | None:
    try:
        request = Request(url, headers={"User-Agent": USER_AGENT})
        with urlopen(request, timeout=30) as response:
            return response.read().decode("utf-8", "ignore")
    except Exception:
        return None


def _download_bytes(url: str) -> bytes | None:
    try:
        if url.startswith("//"):
            url = "https:" + url
        thumb_url = _to_wikimedia_thumb_url(url)
        if thumb_url is not None:
            url = thumb_url
        request = Request(url, headers={"User-Agent": USER_AGENT})
        with urlopen(request, timeout=60) as response:
            return response.read()
    except Exception:
        return None


def _to_wikimedia_thumb_url(image_url: str, size: int = 640) -> str | None:
    marker = "/wikipedia/commons/"
    if marker not in image_url or "/wikipedia/commons/thumb/" in image_url:
        return None
    base, _, tail = image_url.partition(marker)
    parts = tail.split("/")
    if len(parts) < 3:
        return None
    filename = parts[-1]
    thumb_tail = "thumb/" + "/".join(parts) + f"/{size}px-{filename}"
    return base + marker + thumb_tail


def _wikipedia_search_title(query: str, language: str) -> str | None:
    url = (
        f"https://{language}.wikipedia.org/w/api.php?action=opensearch&limit=5&namespace=0&format=json"
        f"&search={quote(query)}"
    )
    try:
        data = _fetch_json(url)
        candidates = data[1] if len(data) > 1 else []
        return str(candidates[0]).strip() if candidates else None
    except Exception:
        return None


def _wikipedia_summary_thumbnail(title: str, language: str) -> bytes | None:
    if not title:
        return None
    try:
        url = f"https://{language}.wikipedia.org/api/rest_v1/page/summary/{quote(title, safe='')}"
        data = _fetch_json(url)
        thumb = (data.get("thumbnail") or {}).get("source")
        if not thumb:
            return None
        return _download_bytes(thumb)
    except Exception:
        return None


def _wikipedia_page_image(title: str, language: str) -> bytes | None:
    if not title:
        return None
    safe_title = title.replace(" ", "_")
    page_url = f"https://{language}.wikipedia.org/wiki/{quote(safe_title)}"
    html = _fetch_text(page_url)
    if not html:
        return None
    match = re.search(r'<meta property="og:image" content="([^"]+)"', html)
    if not match:
        return None
    return _download_bytes(match.group(1))


def _wikipedia_thumbnail(title: str, language: str) -> bytes | None:
    if not title:
        return None
    try:
        url = (
            f"https://{language}.wikipedia.org/w/api.php?action=query&prop=pageimages"
            f"&titles={quote(title)}&pithumbsize=900&format=json"
        )
        data = _fetch_json(url)
        page = next(iter(data["query"]["pages"].values()))
        thumb = (page.get("thumbnail") or {}).get("source")
        if not thumb:
            return None
        return _download_bytes(thumb)
    except Exception:
        return None


def _wikidata_id(title: str, language: str) -> str | None:
    if not title:
        return None
    try:
        url = (
            f"https://{language}.wikipedia.org/w/api.php?action=query&titles={quote(title)}"
            "&prop=pageprops&format=json"
        )
        data = _fetch_json(url)
        page = next(iter(data["query"]["pages"].values()))
        return (page.get("pageprops") or {}).get("wikibase_item")
    except Exception:
        return None


def _wikidata_p18(qid: str) -> bytes | None:
    if not qid:
        return None
    try:
        url = f"https://www.wikidata.org/w/api.php?action=wbgetentities&ids={quote(qid)}&props=claims&format=json"
        data = _fetch_json(url)
        claims = data["entities"][qid].get("claims") or {}
        p18 = claims.get("P18")
        if not p18:
            return None
        filename = p18[0]["mainsnak"]["datavalue"]["value"]
        commons_title = f"File:{filename}"
        info_url = (
            "https://commons.wikimedia.org/w/api.php?action=query"
            f"&titles={quote(commons_title)}&prop=imageinfo&iiprop=url&iiurlwidth=900&format=json"
        )
        info = _fetch_json(info_url)
        page = next(iter(info["query"]["pages"].values()))
        image_info = (page.get("imageinfo") or [{}])[0]
        img_url = image_info.get("thumburl") or image_info.get("url")
        if not img_url:
            return None
        return _download_bytes(img_url)
    except Exception:
        return None


def _commons_search(query: str) -> bytes | None:
    if not query:
        return None
    try:
        url = (
            "https://commons.wikimedia.org/w/api.php?action=query&generator=search&gsrnamespace=6"
            f"&gsrsearch={quote(query)}&prop=imageinfo&iiprop=url&iiurlwidth=900&format=json"
        )
        data = _fetch_json(url)
        pages = (data.get("query") or {}).get("pages") or {}
        for page in pages.values():
            image_info = (page.get("imageinfo") or [{}])[0]
            img_url = image_info.get("thumburl") or image_info.get("url")
            if img_url:
                result = _download_bytes(img_url)
                if result is not None:
                    return result
    except Exception:
        return None
    return None


def _candidate_queries(name: str) -> list[str]:
    ascii_name = _ascii_name(name)
    queries: list[str] = []
    for candidate in (
        name,
        ascii_name,
        f"{name} cyclist",
        f"{ascii_name} cyclist",
        f"{name} road cyclist",
        f"{ascii_name} road cyclist",
        f"{name} (cyclist)",
        f"{ascii_name} (cyclist)",
    ):
        if candidate and candidate not in queries:
            queries.append(candidate)
    return queries


def _try_remote_image(name: str) -> bytes | None:
    manual_url = MANUAL_IMAGE_URLS.get(name) or MANUAL_IMAGE_URLS.get(_ascii_name(name))
    if manual_url:
        data = _download_bytes(manual_url)
        if data is not None:
            return data

    languages = ("en", "fr", "it", "de", "nl", "es")
    for query in _candidate_queries(name):
        for language in languages:
            data = _wikipedia_summary_thumbnail(query, language)
            if data is not None:
                return data

            data = _wikipedia_page_image(query, language)
            if data is not None:
                return data

            data = _wikipedia_thumbnail(query, language)
            if data is not None:
                return data

            qid = _wikidata_id(query, language)
            if qid:
                data = _wikidata_p18(qid)
                if data is not None:
                    return data

            data = _commons_search(query)
            if data is not None:
                return data

        for language in languages:
            resolved = _wikipedia_search_title(query, language) or query

            data = _wikipedia_summary_thumbnail(resolved, language)
            if data is not None:
                return data

            data = _wikipedia_page_image(resolved, language)
            if data is not None:
                return data

            data = _wikipedia_thumbnail(resolved, language)
            if data is not None:
                return data

            qid = _wikidata_id(resolved, language)
            if qid:
                data = _wikidata_p18(qid)
                if data is not None:
                    return data

            data = _commons_search(resolved)
            if data is not None:
                return data
    return None


def _save_jpeg(data: bytes, output_path: Path) -> None:
    with Image.open(BytesIO(data)) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(output_path, format="JPEG", quality=94, optimize=True)


def _top_riders(input_csv: Path, top_n: int) -> list[str]:
    snapshots: dict[str, list[tuple[str, float]]] = {}
    with input_csv.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            ranking_date = row["ranking_date"].strip()
            snapshots.setdefault(ranking_date, []).append((row["player_name"].strip(), float(row["points"])))
    riders: list[str] = []
    seen: set[str] = set()
    for ranking_date in sorted(snapshots):
        ranked = sorted(snapshots[ranking_date], key=lambda item: (-item[1], item[0]))[:top_n]
        for rider_name, _ in ranked:
            if rider_name not in seen:
                seen.add(rider_name)
                riders.append(rider_name)
    return riders


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download portraits for riders that appear in the post-war Tour de France top 12."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--photos-dir", type=Path, default=DEFAULT_PHOTOS_DIR)
    parser.add_argument("--top-n", type=int, default=TOP_N)
    parser.add_argument("--refresh", action="store_true", help="Redownload portraits even when a file already exists.")
    return parser.parse_args()


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()
    riders = _top_riders(args.input, args.top_n)
    args.photos_dir.mkdir(parents=True, exist_ok=True)

    already = 0
    downloaded = 0
    missing: list[str] = []

    for index, name in enumerate(riders, start=1):
        output_path = args.photos_dir / f"{_slugify(name)}.jpg"
        if output_path.exists() and not args.refresh:
            already += 1
            continue

        image_bytes = _try_remote_image(name)
        if image_bytes is None:
            missing.append(name)
            continue

        try:
            _save_jpeg(image_bytes, output_path)
            downloaded += 1
        except Exception:
            missing.append(name)

        if index % 3 == 0:
            time.sleep(0.35)

    print(
        f"[scraper] Tour de France post-war top {args.top_n} photos: already={already} downloaded={downloaded} missing={len(missing)}"
    )
    if missing:
        print("[scraper] missing riders:")
        for name in missing:
            print(f" - {name}")


if __name__ == "__main__":
    main()
