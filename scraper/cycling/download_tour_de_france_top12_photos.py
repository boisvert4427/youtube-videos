from __future__ import annotations

import argparse
import csv
import html
import json
import re
import unicodedata
import time
import urllib.parse
import urllib.request
import urllib.error
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CSV = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "cycling"
    / "tour_de_france"
    / "tour_de_france_yellow_jersey_days_postwar_1947_2025.csv"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "raw" / "player_photos"
USER_AGENT = "Mozilla/5.0 (compatible; Codex Tour de France photo downloader)"


def _ascii_key(value: str) -> str:
    return unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii").strip().lower()


def _slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", _ascii_key(value)).strip("_")


def _clean_text(value: str) -> str:
    value = html.unescape(value or "")
    value = re.sub(r"\s+", " ", value).strip()
    return value


def _request_json(url: str) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    delay = 1.0
    for attempt in range(6):
        try:
            with urllib.request.urlopen(request, timeout=45) as response:
                return json.loads(response.read().decode("utf-8"))
        except (urllib.error.HTTPError, urllib.error.URLError) as exc:
            code = getattr(exc, "code", None)
            if code not in {429, 500, 502, 503, 504} or attempt == 5:
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
        except (urllib.error.HTTPError, urllib.error.URLError) as exc:
            code = getattr(exc, "code", None)
            if code not in {429, 500, 502, 503, 504} or attempt == 5:
                raise
            time.sleep(delay)
            delay *= 2.0


def _load_top12_names(input_csv: Path) -> list[str]:
    with input_csv.open("r", newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    unique: "OrderedDict[str, None]" = OrderedDict()
    years = sorted({row["ranking_date"][:4] for row in rows})
    for year in years:
        year_rows = [row for row in rows if row["ranking_date"].startswith(year)][:12]
        for row in year_rows:
            unique.setdefault(_clean_text(row["player_name"]), None)
    return list(unique.keys())


def _search_wikidata(name: str) -> dict | None:
    queries = []
    ascii_name = _clean_text(_ascii_key(name).replace("_", " "))
    if ascii_name and ascii_name != name.lower():
        queries.append(ascii_name)
    queries.append(name)
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
            description = _clean_text(item.get("description", "")).lower()
            if any(token in description for token in ("cyclist", "racing cyclist", "bicycle racer", "road cyclist")):
                preferred.append(item)
        return (preferred[0] if preferred else results[0])
    return None


def _get_enwiki_title(qid: str) -> str | None:
    entity_url = f"https://www.wikidata.org/wiki/Special:EntityData/{qid}.json"
    data = _request_json(entity_url)
    entity = data["entities"].get(qid, {})
    sitelinks = entity.get("sitelinks", {})
    enwiki = sitelinks.get("enwiki")
    if enwiki:
        return enwiki.get("title")
    return None


def _get_wikidata_image_filename(qid: str) -> str | None:
    entity_url = f"https://www.wikidata.org/wiki/Special:EntityData/{qid}.json"
    data = _request_json(entity_url)
    entity = data["entities"].get(qid, {})
    claims = entity.get("claims", {})
    p18 = claims.get("P18", [])
    if not p18:
        return None
    try:
        value = p18[0]["mainsnak"]["datavalue"]["value"]
    except (KeyError, IndexError, TypeError):
        return None
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


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


def _download_photo(name: str, output_dir: Path) -> tuple[str, str]:
    slug = _slugify(name)
    existing = next((path for path in output_dir.glob(f"{slug}.*") if path.is_file()), None)
    if existing is not None:
        return name, f"exists:{existing.name}"

    last_error: str | None = None
    for attempt in range(3):
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
                if title:
                    return name, f"missing:image:{title}"
                return name, f"missing:image:{qid}"

            image_bytes, content_type = _request_bytes(image_url)
            ext = Path(urllib.parse.urlparse(image_url).path).suffix.lower()
            if ext not in {".jpg", ".jpeg", ".png", ".webp"}:
                ext = ".jpg" if content_type == "image/jpeg" else ".png"
            output_path = output_dir / f"{slug}{ext}"
            output_path.write_bytes(image_bytes)
            return name, f"downloaded:{output_path.name}"
        except Exception as exc:
            last_error = f"{type(exc).__name__}"
            if attempt < 2:
                time.sleep(2.0 * (attempt + 1))
                continue
            return name, f"error:{last_error}"
    return name, f"error:{last_error or 'UnknownError'}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download portraits for the Tour de France yearly top 12 riders.")
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--workers", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    names = _load_top12_names(args.input_csv)
    print(f"[photos] unique names: {len(names)}")
    results: list[tuple[str, str]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {executor.submit(_download_photo, name, args.output_dir): name for name in names}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            name, status = result
            print(f"[photos] {status} | {name}")

    downloaded = sum(1 for _, status in results if status.startswith("downloaded:"))
    skipped = sum(1 for _, status in results if status.startswith("exists:"))
    missing = [name for name, status in results if status.startswith("missing:")]
    errors = [name for name, status in results if status.startswith("error:")]
    print(f"[photos] downloaded={downloaded} skipped={skipped} missing={len(missing)}")
    for name in missing:
        print(f"[photos] missing -> {name}")
    if errors:
        print(f"[photos] errors={len(errors)}")
        for name in errors:
            print(f"[photos] error -> {name}")


if __name__ == "__main__":
    main()
