from __future__ import annotations

import argparse
import csv
import functools
import json
import math
import re
import sys
import time
import unicodedata
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen

import numpy as np
from moviepy import AudioFileClip, CompositeAudioClip, VideoClip
from moviepy.audio.fx import AudioFadeIn, AudioFadeOut
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageOps

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from video_generator.generate_ucl_barchart_race_moviepy import DEFAULT_AUDIO, build_audio_track


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_INPUT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "cycling"
    / "tour_de_france"
    / "tour_de_france_through_the_years_1947_2025.csv"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "cycling"
    / "tour_de_france"
    / "tour_de_france_through_the_years_1947_2025_4min_60fps.mp4"
)
DEFAULT_FLAGS_DIR = PROJECT_ROOT / "data" / "raw" / "flags"
DEFAULT_PHOTOS_DIR = PROJECT_ROOT / "data" / "raw" / "player_photos" / "tour_de_france_through_the_years"
DEFAULT_JERSEYS_DIR = PROJECT_ROOT / "data" / "raw" / "cycling" / "tour_de_france_jerseys"

WIDTH = 1920
HEIGHT = 1080
FPS = 60
TOTAL_DURATION = 240.0
HOLD_START = 4.0
HOLD_END = 6.0

TITLE_LEFT = "T O U R  D E  F R A N C E"
TITLE_RIGHT = "T H R O U G H  T H E  Y E A R S"

ALPHA3_TO_ALPHA2 = {
    "ARG": "ar",
    "AUS": "au",
    "AUT": "at",
    "BEL": "be",
    "CAN": "ca",
    "COL": "co",
    "CZE": "cz",
    "DEN": "dk",
    "ECU": "ec",
    "EST": "ee",
    "FRA": "fr",
    "FRG": "de",
    "GBR": "gb",
    "GER": "de",
    "GDR": "de",
    "IRL": "ie",
    "ITA": "it",
    "LUX": "lu",
    "NED": "nl",
    "NOR": "no",
    "POL": "pl",
    "POR": "pt",
    "RSA": "za",
    "RUS": "ru",
    "SLO": "si",
    "SVK": "sk",
    "SUI": "ch",
    "SWE": "se",
    "TCH": "cz",
    "UKR": "ua",
    "USA": "us",
}


@dataclass(frozen=True)
class TourCardEntry:
    year: int
    winner_name: str
    winner_country: str
    winner_team: str
    winner_time: str
    gc2_name: str
    gc2_country: str
    gc2_team: str
    gc2_gap: str
    gc3_name: str
    gc3_country: str
    gc3_team: str
    gc3_gap: str
    points_name: str
    points_country: str
    points_team: str
    mountains_name: str
    mountains_country: str
    mountains_team: str
    badge_label: str
    card_bg_color: str
    accent_color: str


def _load_font(size: int, bold: bool = False, italic: bool = False):
    if bold and italic:
        candidates = [
            "C:/Windows/Fonts/arialbi.ttf",
            "C:/Windows/Fonts/segoeuiz.ttf",
        ]
    elif bold:
        candidates = [
            "C:/Windows/Fonts/arialbd.ttf",
            "C:/Windows/Fonts/segoeuib.ttf",
        ]
    elif italic:
        candidates = [
            "C:/Windows/Fonts/ariali.ttf",
            "C:/Windows/Fonts/segoeuii.ttf",
        ]
    else:
        candidates = [
            "C:/Windows/Fonts/arial.ttf",
            "C:/Windows/Fonts/segoeui.ttf",
        ]
    for font_path in candidates:
        path = Path(font_path)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def _fit_font(draw: ImageDraw.ImageDraw, text: str, max_width: int, max_size: int, min_size: int, bold: bool = False, italic: bool = False):
    size = max_size
    while size >= min_size:
        font = _load_font(size=size, bold=bold, italic=italic)
        bbox = draw.textbbox((0, 0), text, font=font)
        if bbox[2] - bbox[0] <= max_width:
            return font
        size -= 1
    return _load_font(size=min_size, bold=bold, italic=italic)


def _truncate(draw: ImageDraw.ImageDraw, text: str, font, max_width: int) -> str:
    if max_width <= 0:
        return ""
    if draw.textbbox((0, 0), text, font=font)[2] <= max_width:
        return text
    suffix = "..."
    lo, hi = 0, len(text)
    best = suffix
    while lo <= hi:
        mid = (lo + hi) // 2
        candidate = text[:mid].rstrip() + suffix
        if draw.textbbox((0, 0), candidate, font=font)[2] <= max_width:
            best = candidate
            lo = mid + 1
        else:
            hi = mid - 1
    return best


def _slugify(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", normalized.strip().lower())
    return cleaned.strip("_")


def _ascii_key(value: str) -> str:
    return unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii").strip().lower()


def _hex_to_rgb(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    return tuple(int(value[i : i + 2], 16) for i in (0, 2, 4))


def _mix_rgb(color: str | tuple[int, int, int], target: tuple[int, int, int], amount: float) -> tuple[int, int, int]:
    r, g, b = _hex_to_rgb(color) if isinstance(color, str) else color
    amount = min(max(amount, 0.0), 1.0)
    return (
        int(r + (target[0] - r) * amount),
        int(g + (target[1] - g) * amount),
        int(b + (target[2] - b) * amount),
    )


def _text_on(color: str) -> str:
    r, g, b = _hex_to_rgb(color)
    luminance = (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255.0
    return "#0f1f2d" if luminance > 0.68 else "#f7fbff"


def _to_alpha2(country_code: str) -> str:
    code = _clean_country(country_code).upper()
    if len(code) == 2 and code.isalpha():
        return code.lower()
    if len(code) == 3 and code.isalpha():
        return ALPHA3_TO_ALPHA2.get(code.upper(), "")
    return ""


def _clean_country(value: str) -> str:
    value = value.strip().replace("(", "").replace(")", "").replace(".", "")
    return re.sub(r"\s+", "", value)


def _short_rider_name(full_name: str) -> str:
    tokens = [token for token in full_name.strip().split() if token]
    if len(tokens) <= 1:
        return full_name.strip()
    particles = {"de", "del", "da", "di", "du", "la", "le", "van", "von", "st", "saint"}
    surname = [tokens[-1]]
    i = len(tokens) - 2
    while i >= 1 and tokens[i].lower() in particles:
        surname.insert(0, tokens[i])
        i -= 1
    return f"{tokens[0][0]}. {' '.join(surname)}"


def _split_display_name(full_name: str) -> tuple[str, str]:
    tokens = [token for token in full_name.strip().split() if token]
    if not tokens:
        return "", ""
    if len(tokens) == 1:
        return tokens[0].upper(), ""
    particles = {"de", "del", "da", "di", "du", "la", "le", "van", "von", "st", "saint"}
    surname = [tokens[-1]]
    i = len(tokens) - 2
    while i >= 1 and tokens[i].lower() in particles:
        surname.insert(0, tokens[i])
        i -= 1
    given = " ".join(tokens[: i + 1]) or tokens[0]
    return given.upper(), " ".join(surname).upper()


def _parse_csv_rows(input_csv: Path) -> list[TourCardEntry]:
    entries: list[TourCardEntry] = []
    with input_csv.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            entries.append(
                TourCardEntry(
                    year=int(row["year"]),
                    winner_name=row.get("winner_name", "").strip(),
                    winner_country=row.get("winner_country", "").strip(),
                    winner_team=row.get("winner_team", "").strip(),
                    winner_time=row.get("winner_time", "").strip(),
                    gc2_name=row.get("gc2_name", "").strip(),
                    gc2_country=row.get("gc2_country", "").strip(),
                    gc2_team=row.get("gc2_team", "").strip(),
                    gc2_gap=row.get("gc2_gap", "").strip(),
                    gc3_name=row.get("gc3_name", "").strip(),
                    gc3_country=row.get("gc3_country", "").strip(),
                    gc3_team=row.get("gc3_team", "").strip(),
                    gc3_gap=row.get("gc3_gap", "").strip(),
                    points_name=row.get("points_name", "").strip(),
                    points_country=row.get("points_country", "").strip(),
                    points_team=row.get("points_team", "").strip(),
                    mountains_name=row.get("mountains_name", "").strip(),
                    mountains_country=row.get("mountains_country", "").strip(),
                    mountains_team=row.get("mountains_team", "").strip(),
                    badge_label=row.get("badge_label", "WINNER").strip() or "WINNER",
                    card_bg_color=row.get("card_bg_color", "#050505").strip() or "#050505",
                    accent_color=row.get("accent_color", "#f4c319").strip() or "#f4c319",
                )
            )
    entries.sort(key=lambda item: item.year)
    return entries


def _request_json(url: str) -> dict:
    request = Request(url, headers={"User-Agent": "Mozilla/5.0 (compatible; Codex Tour de France photo fetcher)"})
    with urlopen(request, timeout=45) as response:
        return json.loads(response.read().decode("utf-8", "ignore"))


def _request_bytes(url: str) -> tuple[bytes, str]:
    request = Request(url, headers={"User-Agent": "Mozilla/5.0 (compatible; Codex Tour de France photo fetcher)"})
    with urlopen(request, timeout=60) as response:
        return response.read(), response.headers.get_content_type()


def _wikipedia_summary_thumbnail(title: str) -> bytes | None:
    if not title:
        return None
    try:
        data = _request_json(f"https://en.wikipedia.org/api/rest_v1/page/summary/{quote(title.replace(' ', '_'), safe='')}")
        thumb = (data.get("thumbnail") or {}).get("source")
        if not thumb:
            return None
        return _request_bytes(thumb)[0]
    except Exception:
        return None


def _wikidata_id(title: str) -> str | None:
    if not title:
        return None
    try:
        data = _request_json(
            "https://www.wikidata.org/w/api.php?action=wbsearchentities"
            f"&search={quote(title)}&language=en&format=json&limit=5"
        )
        results = data.get("search", [])
        if not results:
            return None
        return results[0].get("id")
    except Exception:
        return None


def _wikidata_p18(qid: str) -> bytes | None:
    if not qid:
        return None
    try:
        data = _request_json(f"https://www.wikidata.org/wiki/Special:EntityData/{qid}.json")
        entity = data["entities"].get(qid, {})
        claims = entity.get("claims") or {}
        p18 = claims.get("P18")
        if not p18:
            return None
        filename = p18[0]["mainsnak"]["datavalue"]["value"]
        commons = _request_json(
            "https://commons.wikimedia.org/w/api.php?action=query&titles="
            f"{quote('File:' + filename)}&prop=imageinfo&iiprop=url&iiurlwidth=1200&format=json"
        )
        page = next(iter(commons["query"]["pages"].values()))
        image_info = (page.get("imageinfo") or [{}])[0]
        img_url = image_info.get("thumburl") or image_info.get("url")
        if not img_url:
            return None
        return _request_bytes(img_url)[0]
    except Exception:
        return None


def _commons_search(query: str) -> bytes | None:
    if not query:
        return None
    try:
        data = _request_json(
            "https://commons.wikimedia.org/w/api.php?action=query&generator=search&gsrnamespace=6"
            f"&gsrsearch={quote(query)}&prop=imageinfo&iiprop=url&iiurlwidth=1200&format=json"
        )
        pages = (data.get("query") or {}).get("pages") or {}
        for page in pages.values():
            image_info = (page.get("imageinfo") or [{}])[0]
            img_url = image_info.get("thumburl") or image_info.get("url")
            if img_url:
                try:
                    return _request_bytes(img_url)[0]
                except Exception:
                    continue
    except Exception:
        return None
    return None


def _candidate_queries(name: str) -> list[str]:
    ascii_name = _ascii_key(name)
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


@functools.lru_cache(maxsize=256)
def _fetch_winner_photo_bytes(name: str) -> bytes | None:
    for query in _candidate_queries(name):
        data = _wikipedia_summary_thumbnail(query)
        if data is not None:
            return data
        qid = _wikidata_id(query)
        if qid:
            data = _wikidata_p18(qid)
            if data is not None:
                return data
        data = _commons_search(query)
        if data is not None:
            return data
    return None


def _resolve_photo_path(player_name: str, photos_dir: Path) -> Path | None:
    photos_dir.mkdir(parents=True, exist_ok=True)
    slug = _slugify(player_name)
    for ext in (".jpg", ".jpeg", ".png", ".webp"):
        candidate = photos_dir / f"{slug}{ext}"
        if candidate.exists():
            return candidate
    data = _fetch_winner_photo_bytes(player_name)
    if data is None:
        return None
    output = photos_dir / f"{slug}.jpg"
    try:
        output.write_bytes(data)
        return output
    except Exception:
        return None


def _resolve_flag(country_code: str, flags_dir: Path) -> Image.Image | None:
    alpha2 = _to_alpha2(country_code)
    if not alpha2:
        return None
    path = flags_dir / f"{alpha2}.png"
    if not path.exists():
        return None
    try:
        return Image.open(path).convert("RGBA")
    except Exception:
        return None


def _background() -> Image.Image:
    xx = np.linspace(0, 1, WIDTH, dtype=np.float32)
    yy = np.linspace(0, 1, HEIGHT, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(xx, yy)
    night = np.array([4, 8, 16], dtype=np.float32)
    steel = np.array([14, 27, 42], dtype=np.float32)
    gold = np.array([241, 195, 25], dtype=np.float32)
    ember = np.array([196, 145, 42], dtype=np.float32)

    mix = np.clip(0.46 * grid_y + 0.18 * grid_x, 0.0, 1.0)
    top_glow = np.exp(-(((grid_x - 0.72) / 0.28) ** 2 + ((grid_y - 0.09) / 0.12) ** 2))
    side_glow = np.exp(-(((grid_x - 0.12) / 0.16) ** 2 + ((grid_y - 0.76) / 0.25) ** 2))
    far_glow = np.exp(-(((grid_x - 0.90) / 0.12) ** 2 + ((grid_y - 0.72) / 0.18) ** 2))

    bg = np.clip(
        night[None, None, :] * (1.0 - mix[..., None])
        + steel[None, None, :] * (0.82 * mix[..., None])
        + gold[None, None, :] * (0.08 * top_glow[..., None] + 0.04 * side_glow[..., None])
        + ember[None, None, :] * (0.04 * far_glow[..., None]),
        0,
        255,
    ).astype(np.uint8)
    frame = Image.fromarray(bg, mode="RGB").convert("RGBA")

    overlay = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")
    draw.rounded_rectangle((28, 22, WIDTH - 28, HEIGHT - 28), radius=42, outline=(255, 255, 255, 16), width=2)
    draw.line((80, 948, WIDTH - 80, 948), fill=(255, 255, 255, 9), width=2)
    draw.line((120, 930, WIDTH - 120, 930), fill=(255, 255, 255, 5), width=1)
    draw.ellipse((50, 120, 320, 390), outline=(241, 195, 25, 18), width=3)
    draw.ellipse((1550, 60, 1910, 420), outline=(241, 195, 25, 18), width=3)
    draw.line((100, 80, 560, 80), fill=(241, 195, 25, 120), width=5)
    draw.line((1360, 80, 1820, 80), fill=(241, 195, 25, 120), width=5)
    overlay = overlay.filter(ImageFilter.GaussianBlur(radius=3))
    frame.alpha_composite(overlay)
    return frame


@functools.lru_cache(maxsize=12)
def _jersey_asset(kind: str, width: int, height: int) -> Image.Image:
    source_path = DEFAULT_JERSEYS_DIR / f"{kind}.png"
    if source_path.exists():
        source = ImageOps.exif_transpose(Image.open(source_path)).convert("RGBA")
        source = ImageOps.contain(source, (width, height), Image.Resampling.LANCZOS)
        canvas = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        canvas.alpha_composite(source, ((width - source.width) // 2, height - source.height))
        return canvas

    scale = 4
    w = width * scale
    h = height * scale
    jersey = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    jd = ImageDraw.Draw(jersey, "RGBA")

    if kind == "yellow":
        base = (244, 207, 23)
        sponsor_fill = (24, 76, 154)
        sponsor_text = (255, 226, 35)
    elif kind == "green":
        base = (76, 190, 62)
        sponsor_fill = (238, 245, 238)
        sponsor_text = (35, 130, 45)
    else:
        base = (246, 246, 240)
        sponsor_fill = (22, 74, 150)
        sponsor_text = (245, 245, 245)

    light = _mix_rgb(base, (255, 255, 255), 0.35)
    dark = _mix_rgb(base, (0, 0, 0), 0.35)
    outline = (255, 255, 255, 210) if kind == "polka" else (*_mix_rgb(base, (255, 255, 255), 0.18), 255)
    cx = w // 2
    top = int(h * 0.05)
    shoulder = int(h * 0.18)
    chest_top = int(h * 0.25)
    hem = int(h * 0.93)
    chest_l = int(w * 0.28)
    chest_r = int(w * 0.72)
    waist_l = int(w * 0.31)
    waist_r = int(w * 0.69)

    shadow = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    sd = ImageDraw.Draw(shadow, "RGBA")
    sd.ellipse((int(w * 0.18), int(h * 0.88), int(w * 0.82), int(h * 1.00)), fill=(0, 0, 0, 130))
    shadow = shadow.filter(ImageFilter.GaussianBlur(radius=max(2, int(w * 0.035))))
    jersey.alpha_composite(shadow)

    sleeve_l = [
        (chest_l + int(w * 0.02), shoulder),
        (int(w * 0.10), int(h * 0.27)),
        (int(w * 0.10), int(h * 0.52)),
        (int(w * 0.24), int(h * 0.56)),
        (chest_l + int(w * 0.04), int(h * 0.47)),
    ]
    sleeve_r = [
        (chest_r - int(w * 0.02), shoulder),
        (int(w * 0.90), int(h * 0.27)),
        (int(w * 0.90), int(h * 0.52)),
        (int(w * 0.76), int(h * 0.56)),
        (chest_r - int(w * 0.04), int(h * 0.47)),
    ]
    torso = [(chest_l, shoulder), (chest_r, shoulder), (waist_r, hem), (waist_l, hem)]

    jd.polygon(sleeve_l, fill=(*base, 255), outline=outline)
    jd.polygon(sleeve_r, fill=(*base, 255), outline=outline)
    jd.polygon(torso, fill=(*base, 255), outline=outline)

    shine = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    sd = ImageDraw.Draw(shine, "RGBA")
    sd.ellipse((int(w * 0.16), int(h * 0.08), int(w * 0.55), int(h * 0.80)), fill=(*light, 65))
    sd.ellipse((int(w * 0.52), int(h * 0.10), int(w * 0.88), int(h * 0.78)), fill=(*dark, 45))
    shine.putalpha(shine.getchannel("A").filter(ImageFilter.GaussianBlur(radius=int(w * 0.12))))
    jersey.alpha_composite(shine)
    jd = ImageDraw.Draw(jersey, "RGBA")

    collar = (cx - int(w * 0.16), top, cx + int(w * 0.16), chest_top)
    jd.rounded_rectangle(collar, radius=int(w * 0.035), fill=(*_mix_rgb(base, (255, 255, 255), 0.12), 255), outline=outline, width=max(1, scale))
    neck = (cx - int(w * 0.07), top + int(h * 0.08), cx + int(w * 0.07), chest_top - int(h * 0.01))
    jd.pieslice(neck, 0, 180, fill=(7, 8, 7, 255))

    jd.line((cx, chest_top - int(h * 0.02), cx, hem - int(h * 0.03)), fill=(*_mix_rgb(base, (0, 0, 0), 0.42), 185), width=max(2, int(w * 0.022)))
    jd.line((cx + int(w * 0.018), chest_top, cx + int(w * 0.018), hem - int(h * 0.04)), fill=(255, 255, 255, 70), width=max(1, int(w * 0.008)))
    for offset, alpha in [(-0.16, 55), (0.16, 38), (-0.27, 34), (0.27, 28)]:
        jd.line((cx + int(w * offset), chest_top + int(h * 0.05), cx + int(w * offset * 0.55), hem - int(h * 0.05)), fill=(255, 255, 255, alpha), width=max(1, int(w * 0.01)))
    jd.line((waist_l, hem - int(h * 0.04), waist_r, hem - int(h * 0.04)), fill=(*dark, 130), width=max(2, int(w * 0.018)))

    if kind == "polka":
        dot = (224, 35, 45, 255)
        radius = max(5, int(w * 0.045))
        ys = [int(h * v) for v in (0.31, 0.43, 0.55, 0.67, 0.79)]
        xs = [int(w * v) for v in (0.34, 0.50, 0.66)]
        for row, yy in enumerate(ys):
            for xx in xs:
                shift = int(w * 0.08) if row % 2 else 0
                jd.ellipse((xx + shift - radius, yy - radius, xx + shift + radius, yy + radius), fill=dot)
        for xx in (int(w * 0.17), int(w * 0.83)):
            for yy in (int(h * 0.34), int(h * 0.47)):
                jd.ellipse((xx - radius, yy - radius, xx + radius, yy + radius), fill=dot)
        patch = (cx + int(w * 0.07), int(h * 0.30), cx + int(w * 0.29), int(h * 0.43))
        jd.rounded_rectangle(patch, radius=int(w * 0.025), fill=(16, 19, 22, 220), outline=(255, 255, 255, 160), width=max(1, scale))
    else:
        sponsor = (cx + int(w * 0.05), int(h * 0.31), cx + int(w * 0.30), int(h * 0.43))
        jd.rounded_rectangle(sponsor, radius=int(w * 0.03), fill=(*sponsor_fill, 245), outline=(255, 255, 255, 150), width=max(1, scale))
        sponsor_font = _load_font(max(9, int(h * 0.095)), bold=True)
        label = "LCL"
        bbox = jd.textbbox((0, 0), label, font=sponsor_font)
        jd.text((sponsor[0] + (sponsor[2] - sponsor[0] - (bbox[2] - bbox[0])) // 2, sponsor[1] - int(h * 0.005)), label, font=sponsor_font, fill=(*sponsor_text, 255))

    logo_x = cx - int(w * 0.24)
    logo_y = int(h * 0.31)
    jd.polygon(
        [
            (logo_x, logo_y + int(h * 0.08)),
            (logo_x + int(w * 0.04), logo_y),
            (logo_x + int(w * 0.08), logo_y + int(h * 0.08)),
        ],
        fill=(12, 18, 20, 135),
    )
    jd.rounded_rectangle((cx - int(w * 0.28), int(h * 0.46), cx - int(w * 0.09), int(h * 0.51)), radius=int(w * 0.012), fill=(255, 255, 255, 80))

    jersey = jersey.filter(ImageFilter.UnsharpMask(radius=1.0, percent=120, threshold=3))
    return jersey.resize((width, height), Image.Resampling.LANCZOS)


def _paste_jersey_icon(card: Image.Image, rect: tuple[int, int, int, int], kind: str) -> None:
    x0, y0, x1, y1 = rect
    asset = _jersey_asset(kind, x1 - x0, y1 - y0)
    card.alpha_composite(asset, (x0, y0))


def _draw_golden_mountain_scene(card: Image.Image, rect: tuple[int, int, int, int], accent_rgb: tuple[int, int, int], seed: int) -> None:
    x0, y0, x1, y1 = rect
    w = x1 - x0
    h = y1 - y0
    scene = Image.new("RGBA", (w, h), (4, 5, 5, 255))
    sd = ImageDraw.Draw(scene, "RGBA")

    for y in range(h):
        t = y / max(1, h - 1)
        r = int(4 + 34 * (1.0 - t))
        g = int(5 + 27 * (1.0 - t))
        b = int(5 + 6 * (1.0 - t))
        sd.line((0, y, w, y), fill=(r, g, b, 255))

    sun_x = int(w * (0.42 + 0.08 * ((seed % 5) / 4)))
    sun_y = int(h * 0.50)
    glow = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    gd = ImageDraw.Draw(glow, "RGBA")
    gd.ellipse((sun_x - 210, sun_y - 170, sun_x + 210, sun_y + 170), fill=(*accent_rgb, 74))
    gd.ellipse((sun_x - 100, sun_y - 78, sun_x + 100, sun_y + 78), fill=(*accent_rgb, 86))
    glow = glow.filter(ImageFilter.GaussianBlur(radius=46))
    scene.alpha_composite(glow)

    left_peak = [(0, h - 58), (int(w * 0.22), int(h * 0.22)), (int(w * 0.48), h - 42)]
    mid_peak = [(int(w * 0.20), h - 54), (int(w * 0.55), int(h * 0.30)), (int(w * 0.88), h - 50)]
    right_peak = [(int(w * 0.52), h - 56), (w, int(h * 0.16)), (w, h - 36)]
    sd.polygon(left_peak, fill=(84, 68, 21, 180))
    sd.polygon(mid_peak, fill=(49, 43, 20, 205))
    sd.polygon(right_peak, fill=(92, 75, 20, 165))
    sd.line((int(w * 0.20), int(h * 0.22), int(w * 0.10), int(h * 0.46)), fill=(250, 214, 76, 70), width=3)
    sd.line((int(w * 0.55), int(h * 0.30), int(w * 0.42), int(h * 0.56)), fill=(250, 214, 76, 54), width=3)
    sd.line((w - 2, int(h * 0.16), int(w * 0.76), int(h * 0.52)), fill=(250, 214, 76, 62), width=3)

    if seed % 3 == 0:
        monument = (int(w * 0.11), int(h * 0.56), int(w * 0.34), int(h * 0.79))
        sd.rectangle((monument[0], monument[3] - 18, monument[2], monument[3]), fill=(5, 6, 6, 135))
        sd.rounded_rectangle(monument, radius=4, outline=(216, 168, 38, 54), width=3)
        sd.arc((monument[0] + 44, monument[1] + 42, monument[2] - 44, monument[3] + 40), 180, 360, fill=(216, 168, 38, 54), width=3)

    vignette = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    vd = ImageDraw.Draw(vignette, "RGBA")
    vd.rectangle((0, 0, w, 42), fill=(0, 0, 0, 92))
    vd.rectangle((0, h - 115, w, h), fill=(0, 0, 0, 126))
    vd.rectangle((0, 0, 42, h), fill=(0, 0, 0, 82))
    vd.rectangle((w - 42, 0, w, h), fill=(0, 0, 0, 82))
    vignette = vignette.filter(ImageFilter.GaussianBlur(radius=18))
    scene.alpha_composite(vignette)

    mask = Image.new("L", (w, h), 0)
    md = ImageDraw.Draw(mask)
    md.rounded_rectangle((0, 0, w - 1, h - 1), radius=24, fill=255)
    card.paste(scene, (x0, y0), mask)


def _make_card_frame(entry: TourCardEntry, card_w: int, card_h: int, photos_dir: Path, flags_dir: Path) -> Image.Image:
    card = Image.new("RGBA", (card_w, card_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(card, "RGBA")
    card_rgb = _hex_to_rgb(entry.card_bg_color)
    accent_rgb = _hex_to_rgb(entry.accent_color)

    shadow = Image.new("RGBA", card.size, (0, 0, 0, 0))
    sdraw = ImageDraw.Draw(shadow, "RGBA")
    sdraw.rounded_rectangle((12, 12, card_w - 12, card_h - 12), radius=38, fill=(0, 0, 0, 110))
    shadow = shadow.filter(ImageFilter.GaussianBlur(radius=18))
    card.alpha_composite(shadow)

    draw.rounded_rectangle((0, 0, card_w - 1, card_h - 1), radius=36, fill=entry.card_bg_color, outline=(*accent_rgb, 255), width=4)
    draw.rounded_rectangle((12, 12, card_w - 13, card_h - 13), radius=30, outline=(255, 255, 255, 18), width=1)

    glow = Image.new("RGBA", card.size, (0, 0, 0, 0))
    gdraw = ImageDraw.Draw(glow, "RGBA")
    gdraw.ellipse((card_w - 280, -130, card_w + 80, 180), fill=(*accent_rgb, 52))
    gdraw.ellipse((-140, 220, 220, 620), fill=(*card_rgb, 36))
    gdraw.ellipse((card_w - 420, card_h - 320, card_w + 40, card_h + 120), fill=(*accent_rgb, 26))
    glow = glow.filter(ImageFilter.GaussianBlur(radius=46))
    card.alpha_composite(glow)

    year_box = (card_w // 2 - 88, 0, card_w // 2 + 88, 84)
    draw.rounded_rectangle(year_box, radius=18, fill=(245, 197, 30, 255))
    year_font = _fit_font(draw, str(entry.year), year_box[2] - year_box[0] - 20, 54, 30, bold=True)
    year_bbox = draw.textbbox((0, 0), str(entry.year), font=year_font)
    draw.text(
        (year_box[0] + (year_box[2] - year_box[0]) // 2 - (year_bbox[2] - year_bbox[0]) // 2, 18),
        str(entry.year),
        font=year_font,
        fill="#111111",
    )

    photo_rect = (26, 96, card_w - 26, 450)
    draw.rounded_rectangle(photo_rect, radius=26, fill=(4, 8, 15, 255))
    photo_path = _resolve_photo_path(entry.winner_name, photos_dir)
    if photo_path is not None:
        try:
            photo = ImageOps.exif_transpose(Image.open(photo_path)).convert("RGB")
            photo = ImageOps.fit(
                photo,
                (photo_rect[2] - photo_rect[0], photo_rect[3] - photo_rect[1]),
                method=Image.Resampling.LANCZOS,
                centering=(0.5, 0.18),
            )
            photo_rgba = photo.convert("RGBA")
            portrait_mask = Image.new("L", photo_rgba.size, 0)
            md = ImageDraw.Draw(portrait_mask)
            md.rounded_rectangle((0, 0, photo_rgba.size[0] - 1, photo_rgba.size[1] - 1), radius=22, fill=178)
            md.rectangle((0, int(photo_rgba.size[1] * 0.62), photo_rgba.size[0], photo_rgba.size[1]), fill=142)
            portrait_mask = portrait_mask.filter(ImageFilter.GaussianBlur(radius=3))
            card.paste(photo_rgba, (photo_rect[0], photo_rect[1]), portrait_mask)
        except Exception:
            pass
    else:
        initials = "".join(part[0] for part in entry.winner_name.split()[:2]).upper()
        initials_font = _fit_font(draw, initials, 160, 120, 56, bold=True)
        bbox = draw.textbbox((0, 0), initials, font=initials_font)
        draw.text(
            (card_w // 2 - (bbox[2] - bbox[0]) // 2, photo_rect[1] + 144),
            initials,
            font=initials_font,
            fill="#d8d0bc",
        )

    winner_panel = (18, 416, card_w - 18, 554)
    draw.rounded_rectangle(winner_panel, radius=16, fill=(5, 7, 10, 232), outline=(255, 255, 255, 22), width=1)
    flag = _resolve_flag(entry.winner_country, flags_dir)
    name_left = winner_panel[0] + 20
    if flag is not None:
        try:
            flag = flag.resize((64, 42), Image.Resampling.LANCZOS)
            card.alpha_composite(flag, (name_left, winner_panel[1] + 27))
            name_left += 82
        except Exception:
            pass

    name_line1, name_line2 = _split_display_name(entry.winner_name)
    title_font = _fit_font(draw, name_line1, card_w - name_left - 44, 24, 16, bold=True)
    surname_font = _fit_font(draw, name_line2 or name_line1, card_w - name_left - 44, 42, 24, bold=True, italic=True)
    team_font = _load_font(18, bold=False)

    draw.text((name_left, winner_panel[1] + 18), name_line1, font=title_font, fill="#ffffff")
    if name_line2:
        draw.text((name_left, winner_panel[1] + 50), name_line2, font=surname_font, fill="#f4c319")
    else:
        draw.text((name_left, winner_panel[1] + 42), name_line1, font=surname_font, fill="#f4c319")

    team_line = " | ".join(part for part in [entry.winner_team, entry.winner_country] if part)
    if team_line:
        team_line = _truncate(draw, team_line.upper(), team_font, card_w - name_left - 44)
        draw.text((name_left, winner_panel[1] + 96), team_line, font=team_font, fill="#f0f0ee")

    ribbon = (34, 554, card_w - 34, 598)
    draw.rounded_rectangle(ribbon, radius=8, fill=(241, 195, 25, 255))
    ribbon_font = _fit_font(draw, entry.badge_label, ribbon[2] - ribbon[0] - 24, 28, 18, bold=True, italic=True)
    bbox = draw.textbbox((0, 0), entry.badge_label, font=ribbon_font)
    draw.text(
        (ribbon[0] + (ribbon[2] - ribbon[0]) // 2 - (bbox[2] - bbox[0]) // 2, ribbon[1] + 8),
        entry.badge_label,
        font=ribbon_font,
        fill="#121212",
    )

    draw.line((34, 610, card_w - 34, 610), fill=(241, 195, 25, 84), width=2)
    podium_box = (20, 618, card_w - 20, 782)
    draw.rounded_rectangle(podium_box, radius=12, fill=(4, 7, 11, 176), outline=(241, 195, 25, 40), width=1)
    podium_font = _load_font(22, bold=True)
    draw.text((podium_box[0] + 14, podium_box[1] + 12), "PODIUM", font=podium_font, fill="#f4c319")

    row_y = podium_box[1] + 44
    row_h = 34
    row_gap = 8
    podium_rows = [
        (1, entry.winner_name, entry.winner_country, entry.winner_time),
        (2, entry.gc2_name, entry.gc2_country, entry.gc2_gap),
        (3, entry.gc3_name, entry.gc3_country, entry.gc3_gap),
    ]
    rank_font = _load_font(20, bold=True)
    rider_font = _load_font(19, bold=True)
    value_font = _load_font(18, bold=True)
    for idx, rider_name, country, value in podium_rows:
        y0 = row_y + (idx - 1) * (row_h + row_gap)
        y1 = y0 + row_h
        if idx > 1:
            draw.line((podium_box[0] + 12, y0 - 5, podium_box[2] - 12, y0 - 5), fill=(255, 255, 255, 16), width=1)
        draw.rounded_rectangle((podium_box[0] + 10, y0, podium_box[0] + 38, y1), radius=8, fill=(241, 195, 25, 255))
        bbox = draw.textbbox((0, 0), str(idx), font=rank_font)
        draw.text((podium_box[0] + 24 - (bbox[2] - bbox[0]) // 2, y0 + 6), str(idx), font=rank_font, fill="#121212")

        rider_short = _short_rider_name(rider_name)
        rider_line = rider_short
        if country:
            rider_line = f"{rider_short} ({country})"
        rider_line = _truncate(draw, rider_line, rider_font, podium_box[2] - podium_box[0] - 160)
        draw.text((podium_box[0] + 52, y0 + 5), rider_line, font=rider_font, fill="#edf2f7")

        value_text = _truncate(draw, value, value_font, 150)
        vbox = draw.textbbox((0, 0), value_text, font=value_font)
        draw.text((podium_box[2] - 18 - (vbox[2] - vbox[0]), y0 + 6), value_text, font=value_font, fill="#f1d88b")

    draw.line((34, 788, card_w - 34, 788), fill=(241, 195, 25, 62), width=1)
    jersey_box = (22, 794, card_w - 22, card_h - 18)
    section_font = _load_font(16, bold=True)
    small_font = _load_font(15, bold=True)
    labels = [
        ("YELLOW\nJERSEY", entry.winner_name, "yellow"),
        ("GREEN\nJERSEY", entry.points_name or "-", "green"),
        ("POLKA\nDOTS", entry.mountains_name or "-", "polka"),
    ]
    col_w = (jersey_box[2] - jersey_box[0]) // 3
    for idx, (label, rider_name, kind) in enumerate(labels):
        x0 = jersey_box[0] + idx * col_w
        x1 = x0 + col_w
        if idx > 0:
            draw.line((x0, jersey_box[1] + 16, x0, jersey_box[3] - 12), fill=(241, 195, 25, 32), width=1)
        label_x = x0 + col_w // 2
        draw.multiline_text(
            (label_x, jersey_box[1] + 0),
            label,
            font=section_font,
            fill="#f4c319" if kind == "yellow" else "#f0fff0" if kind == "green" else "#fff0f0",
            anchor="ma",
            align="center",
            spacing=0,
        )
        jersey_w = min(180, col_w - 2)
        jersey_x0 = label_x - jersey_w // 2
        _paste_jersey_icon(card, (jersey_x0, jersey_box[1] + 28, jersey_x0 + jersey_w, jersey_box[1] + 162), kind)
        rider_value = _truncate(draw, rider_name.upper(), small_font, x1 - x0 - 18)
        rider_bbox = draw.textbbox((0, 0), rider_value, font=small_font)
        draw.text((label_x - (rider_bbox[2] - rider_bbox[0]) // 2, jersey_box[1] + 158), rider_value, font=small_font, fill="#f5f7fb")

    return card


def render_video(
    input_csv: Path,
    output_path: Path,
    photos_dir: Path,
    flags_dir: Path,
    audio_path: Path,
    duration: float,
    fps: int,
    timeline_duration: float | None = None,
    timeline_start: float = 0.0,
    preview_path: Path | None = None,
    preview_year: int | None = None,
) -> Path:
    entries = _parse_csv_rows(input_csv)
    if not entries:
        raise RuntimeError("No Tour de France entries to render.")

    gap = 18
    side_padding = 28
    card_w = int((WIDTH - 2 * side_padding - gap * 3) / 4)
    card_h = 990
    card_y = 88
    pitch = card_w + gap
    total_shift = max(0.0, (len(entries) - 4) * pitch)
    cards = [Image.fromarray(np.array(_make_card_frame(entry, card_w, card_h, photos_dir, flags_dir).convert("RGB"))).convert("RGBA") for entry in entries]
    cards_layer_width = int(WIDTH + total_shift + side_padding * 2)
    cards_layer = Image.new("RGBA", (cards_layer_width, HEIGHT), (0, 0, 0, 0))
    for idx, card in enumerate(cards):
        cards_layer.alpha_composite(card, (int(side_padding + idx * pitch), card_y))

    bg = _background()
    full_timeline_duration = max(duration, timeline_duration or duration)

    header = Image.new("RGBA", (WIDTH, 110), (0, 0, 0, 0))
    hd = ImageDraw.Draw(header, "RGBA")
    hd.line((164, 42, 392, 42), fill=(241, 195, 25, 210), width=3)
    hd.line((1528, 42, 1756, 42), fill=(241, 195, 25, 210), width=3)
    title_font = _load_font(38, bold=True, italic=True)
    subtitle_font = _load_font(34, bold=True, italic=True)
    title_left_bbox = hd.textbbox((0, 0), TITLE_LEFT, font=title_font)
    title_right_bbox = hd.textbbox((0, 0), TITLE_RIGHT, font=subtitle_font)
    total_w = (title_left_bbox[2] - title_left_bbox[0]) + 18 + (title_right_bbox[2] - title_right_bbox[0])
    start_x = WIDTH // 2 - total_w // 2
    hd.text((start_x, 26), TITLE_LEFT, font=title_font, fill="#f7fbff")
    hd.text((start_x + (title_left_bbox[2] - title_left_bbox[0]) + 24, 28), TITLE_RIGHT, font=subtitle_font, fill="#f4c319")
    header = header.filter(ImageFilter.GaussianBlur(radius=0.3))

    transition_duration = max(0.1, full_timeline_duration - max(0.0, HOLD_START + HOLD_END))
    scroll_duration = max(0.1, transition_duration)

    def make_frame(t: float) -> np.ndarray:
        frame = bg.copy()
        timeline_t = min(max(timeline_start + t, 0.0), full_timeline_duration)
        if timeline_t <= HOLD_START:
            shift = 0.0
        elif timeline_t >= full_timeline_duration - HOLD_END:
            shift = total_shift
        else:
            progress = (timeline_t - HOLD_START) / scroll_duration
            shift = total_shift * min(max(progress, 0.0), 1.0)

        canvas = frame.copy()
        canvas.alpha_composite(header)
        crop_left = int(round(shift))
        crop_right = crop_left + WIDTH
        viewport = cards_layer.crop((crop_left, 0, crop_right, HEIGHT))
        canvas.alpha_composite(viewport, (0, 0))
        return np.array(canvas.convert("RGB"))

    clip = VideoClip(make_frame, duration=duration)

    audio_clip = None
    keep_alive: list[object] = []
    if audio_path.exists():
        audio_clip, keep_alive = build_audio_track(audio_path, duration)
        clip = clip.with_audio(audio_clip)

    if preview_path is not None:
        preview_path.parent.mkdir(parents=True, exist_ok=True)
        preview_frame = Image.fromarray(make_frame(0.0), mode="RGB")
        preview_frame.save(preview_path)
        return preview_path

    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_kwargs = {
        "fps": fps,
        "codec": "libx264",
        "ffmpeg_params": ["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
    }
    if audio_clip is not None:
        write_kwargs["audio_codec"] = "aac"
    else:
        write_kwargs["audio"] = False
    clip.write_videofile(str(output_path), **write_kwargs)

    clip.close()
    if audio_clip is not None:
        audio_clip.close()
    for item in keep_alive:
        close = getattr(item, "close", None)
        if callable(close):
            close()
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a Tour de France through-the-years cards video.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--photos-dir", type=Path, default=DEFAULT_PHOTOS_DIR)
    parser.add_argument("--flags-dir", type=Path, default=DEFAULT_FLAGS_DIR)
    parser.add_argument("--audio", type=Path, default=DEFAULT_AUDIO)
    parser.add_argument("--duration", type=float, default=TOTAL_DURATION)
    parser.add_argument("--fps", type=int, default=FPS)
    parser.add_argument("--timeline-duration", type=float, default=None)
    parser.add_argument("--timeline-start", type=float, default=0.0)
    parser.add_argument("--preview", type=Path, default=None)
    parser.add_argument("--preview-year", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = render_video(
        input_csv=args.input,
        output_path=args.output,
        photos_dir=args.photos_dir,
        flags_dir=args.flags_dir,
        audio_path=args.audio,
        duration=args.duration,
        fps=args.fps,
        timeline_duration=args.timeline_duration,
        timeline_start=args.timeline_start,
        preview_path=args.preview,
        preview_year=args.preview_year,
    )
    if args.preview is not None:
        print(f"[video_generator] Tour de France through-the-years preview generated -> {output}")
    else:
        print(f"[video_generator] Tour de France through-the-years video generated -> {output}")


if __name__ == "__main__":
    main()
