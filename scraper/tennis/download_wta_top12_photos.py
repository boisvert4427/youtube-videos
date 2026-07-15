from __future__ import annotations

import argparse
import csv
import html
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import unicodedata
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CSV = PROJECT_ROOT / "data" / "processed" / "tennis" / "wta_ranking_points_timeseries_1990_2026.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "raw" / "player_photos"
USER_AGENT = "Mozilla/5.0 (compatible; Codex WTA photo downloader)"
WTA_HEADSHOT_URL = "https://wtafiles.blob.core.windows.net/images/headshots/{player_id}.jpg"
WTA_PLAYER_URL = "https://www.wtatennis.com/players/{player_id}/{slug}"


def _ascii_key(value: str) -> str:
    return unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii").strip().lower()


def _slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", _ascii_key(value)).strip("_")


def _clean_text(value: str) -> str:
    value = html.unescape(value or "")
    return re.sub(r"\s+", " ", value).strip()


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
    for ranking_date in sorted({row["ranking_date"] for row in rows}):
        year_rows = [row for row in rows if row["ranking_date"] == ranking_date][:12]
        for row in year_rows:
            unique.setdefault(_clean_text(row["player_name"]), None)
    return list(unique.keys())


def _load_top12_players(input_csv: Path) -> list[tuple[str, str]]:
    with input_csv.open("r", newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    unique: "OrderedDict[str, str]" = OrderedDict()
    for ranking_date in sorted({row["ranking_date"] for row in rows}):
        year_rows = [row for row in rows if row["ranking_date"] == ranking_date][:12]
        for row in year_rows:
            name = _clean_text(row["player_name"])
            player_id = _clean_text(row.get("player_id", ""))
            if name and name not in unique:
                unique[name] = player_id
    return list(unique.items())


def _search_wikidata(name: str) -> dict | None:
    queries = [name]
    ascii_name = _clean_text(_ascii_key(name).replace("_", " "))
    if ascii_name and ascii_name != name.lower():
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
        tennis = []
        for item in results:
            description = _clean_text(item.get("description", "")).lower()
            if any(token in description for token in ("tennis player", "professional tennis player")):
                tennis.append(item)
        return tennis[0] if tennis else results[0]
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


def _get_wta_headshot_url(player_id: str) -> str | None:
    player_id = _clean_text(player_id)
    if not player_id:
        return None
    return WTA_HEADSHOT_URL.format(player_id=urllib.parse.quote(player_id))


def _get_wta_profile_image_url(player_id: str, slug: str) -> str | None:
    player_id = _clean_text(player_id)
    if not player_id:
        return None
    url = WTA_PLAYER_URL.format(player_id=urllib.parse.quote(player_id), slug=urllib.parse.quote(slug or "player"))
    try:
        html_text = _request_bytes(url)[0].decode("utf-8", errors="ignore")
    except Exception:
        return None
    match = re.search(r'"image"\s*:\s*"([^"]+)"', html_text)
    if match:
        return html.unescape(match.group(1))
    match = re.search(r'property="og:image"\s+content="([^"]+)"', html_text)
    if match:
        return html.unescape(match.group(1))
    return None


def _download_photo(name: str, player_id: str, output_dir: Path) -> tuple[str, str]:
    slug = _slugify(name)
    existing = next((path for path in output_dir.glob(f"{slug}.*") if path.is_file()), None)
    if existing is not None:
        return name, f"exists:{existing.name}"

    for attempt in range(3):
        try:
            image_url = _get_wta_headshot_url(player_id)
            if image_url:
                try:
                    image_bytes, content_type = _request_bytes(image_url)
                    if len(image_bytes) > 1024:
                        ext = Path(urllib.parse.urlparse(image_url).path).suffix.lower()
                        if ext not in {".jpg", ".jpeg", ".png", ".webp"}:
                            ext = ".jpg" if content_type == "image/jpeg" else ".png"
                        output_path = output_dir / f"{slug}{ext}"
                        output_path.write_bytes(image_bytes)
                        return name, f"downloaded:{output_path.name}"
                except Exception:
                    image_url = None

            if not image_url:
                profile_url = _get_wta_profile_image_url(player_id, slug)
                if profile_url:
                    try:
                        image_bytes, content_type = _request_bytes(profile_url)
                        if len(image_bytes) > 1024:
                            ext = Path(urllib.parse.urlparse(profile_url).path).suffix.lower()
                            if ext not in {".jpg", ".jpeg", ".png", ".webp"}:
                                ext = ".jpg" if content_type == "image/jpeg" else ".png"
                            output_path = output_dir / f"{slug}{ext}"
                            output_path.write_bytes(image_bytes)
                            return name, f"downloaded:{output_path.name}"
                    except Exception:
                        pass

            search = _search_wikidata(name)
            if not search:
                return name, "missing:wta-wikidata"

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
                return name, f"missing:image:{title or qid}"

            image_bytes, content_type = _request_bytes(image_url)
            ext = Path(urllib.parse.urlparse(image_url).path).suffix.lower()
            if ext not in {".jpg", ".jpeg", ".png", ".webp"}:
                ext = ".jpg" if content_type == "image/jpeg" else ".png"
            output_path = output_dir / f"{slug}{ext}"
            output_path.write_bytes(image_bytes)
            return name, f"downloaded:{output_path.name}"
        except Exception as exc:
            if attempt < 2:
                time.sleep(2.0 * (attempt + 1))
                continue
            return name, f"error:{type(exc).__name__}"
    return name, "error:unknown"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download portraits for WTA players appearing in the top 12.")
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--workers", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    players = _load_top12_players(args.input_csv)
    print(f"[wta-photos] unique top-12 names: {len(players)}")
    results: list[tuple[str, str]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {executor.submit(_download_photo, name, player_id, args.output_dir): name for name, player_id in players}
        for future in as_completed(futures):
            name, status = future.result()
            results.append((name, status))
            print(f"[wta-photos] {status} | {name}")

    downloaded = sum(1 for _, status in results if status.startswith("downloaded:"))
    skipped = sum(1 for _, status in results if status.startswith("exists:"))
    missing = [name for name, status in results if status.startswith("missing:")]
    errors = [name for name, status in results if status.startswith("error:")]
    print(f"[wta-photos] downloaded={downloaded} skipped={skipped} missing={len(missing)} errors={len(errors)}")
    for name in missing:
        print(f"[wta-photos] missing -> {name}")
    for name in errors:
        print(f"[wta-photos] error -> {name}")


if __name__ == "__main__":
    main()
