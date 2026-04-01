from __future__ import annotations

import csv
import json
import re
import unicodedata
import urllib.parse
import urllib.request
from html import unescape
from argparse import ArgumentParser
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_CSV = PROJECT_ROOT / "data" / "raw" / "france_kings_reigns.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "raw" / "portraits"
HEADERS = {"User-Agent": "history-timelines-local/1.0"}
MANUAL_COMMONS_FILES = {
    "Robert II le Pieux": "File:Blondel - Robert II of France.jpg",
    "Philippe I": "File:Philippe Ier.jpg",
    "Louis VI le Gros": "File:Louis VI le Gros.jpg",
}


def slugify(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", "_", normalized.lower()).strip("_")


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def fetch_json(url: str) -> dict:
    request = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def download_bytes(url: str) -> bytes:
    request = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def fetch_text(url: str) -> str | None:
    try:
        request = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(request, timeout=60) as response:
            return response.read().decode("utf-8", errors="ignore")
    except Exception:
        return None


def resolve_search_title(title: str, language: str) -> str | None:
    if not title:
        return None
    try:
        url = (
            f"https://{language}.wikipedia.org/w/api.php?action=opensearch&limit=5&namespace=0&format=json"
            f"&search={urllib.parse.quote(title)}"
        )
        data = fetch_json(url)
        candidates = data[1]
        if candidates:
            return str(candidates[0]).strip()
    except Exception:
        return None
    return None


def try_wikipedia_thumbnail(title: str, language: str) -> bytes | None:
    if not title:
        return None
    try:
        url = (
            f"https://{language}.wikipedia.org/w/api.php?action=query&prop=pageimages"
            f"&titles={urllib.parse.quote(title)}&pithumbsize=900&format=json"
        )
        data = fetch_json(url)
        page = next(iter(data["query"]["pages"].values()))
        thumb = (page.get("thumbnail") or {}).get("source")
        if not thumb:
            return None
        return download_bytes(thumb)
    except Exception:
        return None


def try_wikipedia_page_image(title: str, language: str) -> bytes | None:
    if not title:
        return None
    safe_title = title.replace(" ", "_")
    page_url = f"https://{language}.wikipedia.org/wiki/{urllib.parse.quote(safe_title)}"
    html = fetch_text(page_url)
    if not html:
        return None
    match = re.search(r'<meta property="og:image" content="([^"]+)"', html)
    if not match:
        return None
    image_url = unescape(match.group(1))
    try:
        return download_bytes(image_url)
    except Exception:
        return None


def get_wikidata_id(title: str, language: str) -> str | None:
    if not title:
        return None
    try:
        url = (
            f"https://{language}.wikipedia.org/w/api.php?action=query&titles={urllib.parse.quote(title)}"
            "&prop=pageprops&format=json"
        )
        data = fetch_json(url)
        page = next(iter(data["query"]["pages"].values()))
        return (page.get("pageprops") or {}).get("wikibase_item")
    except Exception:
        return None


def try_wikidata_p18(qid: str) -> bytes | None:
    if not qid:
        return None
    try:
        url = f"https://www.wikidata.org/w/api.php?action=wbgetentities&ids={urllib.parse.quote(qid)}&props=claims&format=json"
        data = fetch_json(url)
        claims = data["entities"][qid].get("claims") or {}
        p18 = claims.get("P18")
        if not p18:
            return None
        filename = p18[0]["mainsnak"]["datavalue"]["value"]
        commons_title = f"File:{filename}"
        info_url = (
            "https://commons.wikimedia.org/w/api.php?action=query"
            f"&titles={urllib.parse.quote(commons_title)}&prop=imageinfo&iiprop=url&iiurlwidth=900&format=json"
        )
        info = fetch_json(info_url)
        page = next(iter(info["query"]["pages"].values()))
        image_info = (page.get("imageinfo") or [{}])[0]
        img_url = image_info.get("thumburl") or image_info.get("url")
        if not img_url:
            return None
        return download_bytes(img_url)
    except Exception:
        return None


def try_commons_search(title: str) -> bytes | None:
    if not title:
        return None
    try:
        url = (
            "https://commons.wikimedia.org/w/api.php?action=query&generator=search&gsrnamespace=6"
            f"&gsrsearch={urllib.parse.quote(title)}&prop=imageinfo&iiprop=url&iiurlwidth=900&format=json"
        )
        data = fetch_json(url)
        pages = (data.get("query") or {}).get("pages") or {}
        for page in pages.values():
            image_info = (page.get("imageinfo") or [{}])[0]
            img_url = image_info.get("thumburl") or image_info.get("url")
            if img_url:
                return download_bytes(img_url)
    except Exception:
        return None
    return None


def try_commons_file(file_title: str) -> bytes | None:
    if not file_title:
        return None
    try:
        url = (
            "https://commons.wikimedia.org/w/api.php?action=query"
            f"&titles={urllib.parse.quote(file_title)}&prop=imageinfo&iiprop=url&iiurlwidth=900&format=json"
        )
        data = fetch_json(url)
        page = next(iter(data["query"]["pages"].values()))
        image_info = (page.get("imageinfo") or [{}])[0]
        img_url = image_info.get("thumburl") or image_info.get("url")
        if img_url:
            return download_bytes(img_url)
    except Exception:
        return None
    return None


def try_best_remote_image(display_name: str, wiki_title: str) -> bytes | None:
    manual = try_commons_file(MANUAL_COMMONS_FILES.get(display_name, ""))
    if manual is not None:
        return manual
    search_titles = [
        wiki_title.strip(),
        display_name.strip(),
    ]
    for title in search_titles:
        if not title:
            continue
        for language in ("fr", "en"):
            data = try_wikipedia_page_image(title, language)
            if data is not None:
                return data
            resolved = resolve_search_title(title, language) or title
            data = try_wikipedia_page_image(resolved, language)
            if data is not None:
                return data
            qid = get_wikidata_id(resolved, language)
            if qid:
                data = try_wikidata_p18(qid)
                if data is not None:
                    return data
            data = try_wikipedia_thumbnail(resolved, language)
            if data is not None:
                return data
        data = try_commons_search(resolve_search_title(title, "fr") or title)
        if data is not None:
            return data
        data = try_commons_search(resolve_search_title(title, "en") or title)
        if data is not None:
            return data
    return None


def build_placeholder(display_name: str, dynasty: str, color: str) -> bytes:
    size = 900
    image = Image.new("RGB", (size, size), "#0f2236")
    overlay = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")
    rgb = tuple(int(color[i : i + 2], 16) for i in (1, 3, 5))
    draw.ellipse((40, 30, size - 40, 620), fill=(*rgb, 255))
    draw.ellipse((190, 120, size - 190, 530), fill=(250, 240, 224, 42))
    draw.rounded_rectangle((110, 555, size - 110, size - 90), radius=42, fill=(8, 18, 32, 218), outline=(255, 255, 255, 30), width=2)
    draw.ellipse((260, 150, 640, 530), fill=(244, 233, 220, 255))
    draw.rounded_rectangle((210, 455, 690, 750), radius=180, fill=(28, 46, 78, 255))
    initials = "".join(part[0] for part in display_name.split()[:2]).upper()[:2]
    font_big = ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf", 170)
    font_small = ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf", 52)
    font_micro = ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf", 36)
    initials_box = draw.textbbox((0, 0), initials, font=font_big)
    draw.text(((size - (initials_box[2] - initials_box[0])) // 2, 240), initials, font=font_big, fill=(15, 33, 56, 140))
    name_box = draw.textbbox((0, 0), display_name.upper(), font=font_small)
    draw.text(((size - (name_box[2] - name_box[0])) // 2, 622), display_name.upper(), font=font_small, fill=(244, 247, 251, 255))
    dyn_box = draw.textbbox((0, 0), dynasty.upper(), font=font_micro)
    draw.text(((size - (dyn_box[2] - dyn_box[0])) // 2, 704), dynasty.upper(), font=font_micro, fill=(193, 220, 241, 255))
    overlay = overlay.filter(ImageFilter.GaussianBlur(radius=0.2))
    image.paste(overlay, (0, 0), overlay)
    out = BytesIO()
    image.save(out, format="PNG")
    return out.getvalue()


def parse_args() -> ArgumentParser:
    parser = ArgumentParser(description="Télécharge ou fabrique les portraits des rois de France.")
    parser.add_argument("--refresh", action="store_true", help="Retente un téléchargement même si un portrait existe déjà.")
    return parser


def main() -> None:
    args = parse_args().parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_rows(INPUT_CSV)
    seen: set[str] = set()
    for row in rows:
        display_name = row["display_name"].strip()
        if display_name in seen:
            continue
        seen.add(display_name)
        out_path = OUTPUT_DIR / f"{slugify(display_name)}.png"
        if out_path.exists() and not args.refresh:
            continue
        data = try_best_remote_image(display_name, row["wiki_title"].strip())
        if data is None:
            data = build_placeholder(display_name, row["dynasty"].strip(), row["house_color"].strip())
        out_path.write_bytes(data)
        print(f"[history] portrait ready -> {out_path.name}")


if __name__ == "__main__":
    main()
