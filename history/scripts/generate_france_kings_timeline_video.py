from __future__ import annotations

import argparse
import csv
import math
import re
import unicodedata
from pathlib import Path

import numpy as np
from moviepy import AudioFileClip, CompositeAudioClip, VideoClip
from moviepy.audio.fx import AudioFadeIn, AudioFadeOut
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps


PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_CSV = PROJECT_ROOT / "data" / "raw" / "france_kings_reigns.csv"
PORTRAITS_DIR = PROJECT_ROOT / "data" / "raw" / "portraits"
OUTPUT_MP4 = PROJECT_ROOT / "data" / "processed" / "france_kings_timeline_481_1870_300s_60fps_audio.mp4"
DEFAULT_AUDIO = PROJECT_ROOT / "data" / "raw" / "audio" / "audio.mp3"

WIDTH = 1920
HEIGHT = 1080
FPS = 60
DURATION = 300.0
DISPLAY_START_YEAR = 400
DISPLAY_END_YEAR = 1870
EPISODE_LABEL = "FRANCE ROYALE ET IMPÉRIALE"
FINAL_AUDIO_FADE_OUT = 8.0
LOOP_CROSSFADE = 5.0

CAPITAL_BY_DYNASTY = {
    "Merovingiens": "Soissons → Paris",
    "Transition": "Pouvoir fragmenté",
    "Carolingiens": "Aix-la-Chapelle",
    "Robertiens": "Paris",
    "Capetiens": "Paris",
    "Valois": "Paris",
    "Bourbons": "Versailles → Paris",
    "Révolution et Empire": "Paris",
    "Empire": "Paris",
    "Orleans": "Paris",
    "Deuxième République": "Paris",
    "Second Empire": "Paris",
}

TERRITORY_BY_DYNASTY = {
    "Merovingiens": "Royaume des Francs\nExtension majeure",
    "Transition": "Transition politique\nRoyaume instable",
    "Carolingiens": "Empire carolingien\nRayonnement européen",
    "Robertiens": "Royaume de France\nAncrage capétien",
    "Capetiens": "Royaume de France\nCentralisation",
    "Valois": "Royaume de France\nGuerre de Cent Ans",
    "Bourbons": "Royaume de France\nMonarchie absolue",
    "Révolution et Empire": "France révolutionnaire\nRépublique",
    "Empire": "Premier Empire\nExpansion européenne",
    "Orleans": "Royaume de France\nMonarchie de Juillet",
    "Deuxième République": "République française\nSuffrage universel masculin",
    "Second Empire": "Empire français\nModernisation du pays",
}

FEATURE_OVERRIDES = {
    "Clovis I": ("Unification des Francs", "Conversion au christianisme"),
    "Napoléon Ier": ("Premier Empire", "Empereur des Français"),
    "Napoléon III": ("Second Empire", "Modernisation de la France"),
}

PORTRAIT_CROPS = {
    "Napoléon III": (0.18, 0.0, 0.82, 0.64),
}


def slugify(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", "_", normalized.lower()).strip("_")


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf",
    ]
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def hex_to_rgb(color: str) -> tuple[int, int, int]:
    color = color.lstrip("#")
    return tuple(int(color[i : i + 2], 16) for i in (0, 2, 4))


def make_background() -> Image.Image:
    xx = np.linspace(0, 1, WIDTH, dtype=np.float32)
    yy = np.linspace(0, 1, HEIGHT, dtype=np.float32)
    gx, gy = np.meshgrid(xx, yy)
    top = np.array([6, 18, 34], dtype=np.float32)
    bottom = np.array([19, 10, 12], dtype=np.float32)
    ember = np.array([95, 59, 28], dtype=np.float32)
    gold = np.array([233, 193, 110], dtype=np.float32)
    mix = np.clip(0.58 * gy + 0.12 * gx, 0, 1)
    glow = np.exp(-(((gx - 0.67) / 0.18) ** 2 + ((gy - 0.13) / 0.11) ** 2))
    glow2 = np.exp(-(((gx - 0.55) / 0.22) ** 2 + ((gy - 0.08) / 0.09) ** 2))
    img = np.clip(
        top[None, None, :] * (1.0 - mix[..., None])
        + bottom[None, None, :] * mix[..., None]
        + ember[None, None, :] * (0.06 * glow2[..., None])
        + gold[None, None, :] * (0.10 * glow[..., None]),
        0,
        255,
    ).astype(np.uint8)
    frame = Image.fromarray(img).convert("RGBA")
    overlay = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")
    draw.rounded_rectangle((28, 28, WIDTH - 28, HEIGHT - 28), radius=40, outline=(233, 193, 110, 22), width=2)

    map_layer = Image.new("RGBA", (WIDTH, 280), (0, 0, 0, 0))
    map_draw = ImageDraw.Draw(map_layer, "RGBA")
    map_draw.ellipse((250, 10, 760, 180), fill=(122, 91, 46, 26))
    map_draw.ellipse((530, -10, 1040, 165), fill=(130, 98, 50, 22))
    map_draw.ellipse((890, 0, 1470, 185), fill=(112, 82, 38, 26))
    map_draw.ellipse((1110, 20, 1650, 175), fill=(105, 75, 34, 18))
    for y in (26, 58, 90, 126, 154):
        map_draw.line((420, y, 1500, y + 10), fill=(233, 193, 110, 14), width=2)
    for x in (520, 740, 920, 1140, 1360):
        map_draw.line((x, 8, x + 64, 180), fill=(233, 193, 110, 10), width=2)
    map_draw.arc((360, -90, 1530, 240), start=205, end=340, fill=(233, 193, 110, 40), width=3)
    map_layer = map_layer.filter(ImageFilter.GaussianBlur(radius=8))
    overlay.alpha_composite(map_layer, (0, 0))

    draw.ellipse((845, 16, 1235, 208), fill=(233, 193, 110, 28))
    draw.ellipse((915, 42, 1105, 172), fill=(255, 217, 133, 18))
    draw.arc((470, -80, 1590, 260), start=208, end=339, fill=(233, 193, 110, 78), width=4)
    draw.arc((520, -30, 1520, 220), start=208, end=338, fill=(255, 217, 133, 22), width=2)
    draw.line((82, 892, WIDTH - 82, 892), fill=(255, 255, 255, 12), width=1)
    overlay = overlay.filter(ImageFilter.GaussianBlur(radius=3))
    frame.alpha_composite(overlay)
    return frame


def build_segments(rows: list[dict[str, str]]) -> list[dict[str, str | int]]:
    segments = []
    for row in rows:
        start_year = int(row["start_year"])
        if start_year >= DISPLAY_END_YEAR:
            continue
        end_year = min(int(row["end_year"]), DISPLAY_END_YEAR)
        segments.append(
            {
                "start_year": start_year,
                "end_year": end_year,
                "display_name": row["display_name"],
                "dynasty": row["dynasty"],
                "house_color": row["house_color"],
                "notes": row["notes"],
                "fait_1": row.get("fait_1", ""),
                "fait_2": row.get("fait_2", ""),
                "fait_3": row.get("fait_3", ""),
            }
        )
    segments.extend(
        [
            {
                "start_year": 738,
                "end_year": 742,
                "display_name": "Sans roi",
                "dynasty": "Transition",
                "house_color": "#6c757d",
                "notes": "Interrègne avant Childéric III",
                "fait_1": "La succession monarchique est interrompue pendant cet interrègne.",
                "fait_2": "Le pouvoir réel est alors dominé par les maires du palais.",
                "fait_3": "Cette transition prépare le retour d'un roi mérovingien en 743.",
            },
            {
                "start_year": 841,
                "end_year": 842,
                "display_name": "Sans roi",
                "dynasty": "Transition",
                "house_color": "#6c757d",
                "notes": "Conflit de partage après Louis le Pieux",
                "fait_1": "La guerre entre héritiers suit la mort de Louis le Pieux.",
                "fait_2": "Cette crise mène au partage de Verdun en 843.",
                "fait_3": "La Francie occidentale émerge alors comme cadre de la future France.",
            },
            {
                "start_year": 1793,
                "end_year": 1803,
                "display_name": "Sans roi",
                "dynasty": "Révolution et Empire",
                "house_color": "#6c757d",
                "notes": "Monarchie interrompue",
                "fait_1": "La monarchie est abolie après la Révolution française.",
                "fait_2": "La Première République puis le Consulat transforment le régime.",
                "fait_3": "Le Consulat prépare l'avènement du Premier Empire.",
            },
            {
                "start_year": 1848,
                "end_year": 1851,
                "display_name": "Deuxième République",
                "dynasty": "Deuxième République",
                "house_color": "#6c757d",
                "notes": "Régime républicain",
                "fait_1": "La révolution de 1848 renverse la monarchie de Juillet.",
                "fait_2": "Le suffrage universel masculin est instauré.",
                "fait_3": "Louis-Napoléon Bonaparte devient président de la République.",
            },
        ]
    )
    segments.sort(key=lambda item: int(item["start_year"]))
    return segments


def find_segment(segments: list[dict[str, str | int]], year: int) -> dict[str, str | int]:
    for segment in segments:
        if int(segment["start_year"]) <= year <= int(segment["end_year"]):
            return segment
    return segments[-1]


def load_portrait(name: str) -> Image.Image | None:
    path = PORTRAITS_DIR / f"{slugify(name)}.png"
    if not path.exists():
        return None
    image = Image.open(path).convert("RGBA")
    crop = PORTRAIT_CROPS.get(name)
    if crop:
        width, height = image.size
        image = image.crop(
            (
                int(width * crop[0]),
                int(height * crop[1]),
                int(width * crop[2]),
                int(height * crop[3]),
            )
        )
    return ImageOps.fit(image, (360, 430), method=Image.Resampling.LANCZOS, centering=(0.5, 0.34))


def draw_glow_text(frame: Image.Image, position: tuple[int, int], text: str, font: ImageFont.ImageFont, fill: tuple[int, int, int], glow: tuple[int, int, int]) -> None:
    x, y = position
    glow_layer = Image.new("RGBA", frame.size, (0, 0, 0, 0))
    glow_draw = ImageDraw.Draw(glow_layer, "RGBA")
    for dx, dy in ((-3, 0), (3, 0), (0, -3), (0, 3), (0, 0)):
        glow_draw.text((x + dx, y + dy), text, font=font, fill=(*glow, 70))
    glow_layer = glow_layer.filter(ImageFilter.GaussianBlur(radius=4))
    frame.alpha_composite(glow_layer)
    ImageDraw.Draw(frame, "RGBA").text((x, y), text, font=font, fill=fill)


def wrap_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int, max_lines: int) -> list[str]:
    words = text.split()
    if not words:
        return []
    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        if draw.textbbox((0, 0), candidate, font=font)[2] <= max_width:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def text_width(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> int:
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0]


def text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def draw_centered_text(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    text: str,
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int, int] | tuple[int, int, int] | str,
) -> None:
    width, height = text_size(draw, text, font)
    x0, y0, x1, y1 = box
    draw.text(((x0 + x1 - width) / 2, (y0 + y1 - height) / 2 - 1), text, font=font, fill=fill)


def fit_fact_layout(
    draw: ImageDraw.ImageDraw,
    facts: list[str],
    max_width: int,
    max_height: int,
    *,
    base_size: int = 17,
    min_size: int = 11,
    max_lines: int = 4,
) -> tuple[ImageFont.ImageFont, int, int, list[list[str]]]:
    for font_size in range(base_size, min_size - 1, -1):
        font = load_font(font_size, bold=False)
        line_step = max(10, text_size(draw, "Ag", font)[1] + 1)
        fact_gap = 4
        wrapped_facts: list[list[str]] = []
        total_height = 0

        for fact in facts:
            text = fact.strip()
            if not text:
                continue
            wrapped = wrap_text(draw, text, font, max_width, max_lines)
            wrapped_facts.append(wrapped)
            if total_height:
                total_height += fact_gap
            total_height += len(wrapped) * line_step

        if total_height <= max_height:
            return font, line_step, fact_gap, wrapped_facts

    font = load_font(min_size, bold=False)
    line_step = max(10, text_size(draw, "Ag", font)[1] + 1)
    fact_gap = 4
    wrapped_facts = []
    for fact in facts:
        text = fact.strip()
        if not text:
            continue
        wrapped_facts.append(wrap_text(draw, text, font, max_width, max_lines))
    return font, line_step, fact_gap, wrapped_facts


def unique_dynasty_items(segments: list[dict[str, str | int]]) -> list[tuple[str, tuple[int, int, int]]]:
    seen: set[str] = set()
    items: list[tuple[str, tuple[int, int, int]]] = []
    for segment in segments:
        dynasty = str(segment["dynasty"])
        if dynasty in seen:
            continue
        seen.add(dynasty)
        items.append((dynasty, hex_to_rgb(str(segment["house_color"]))))
    return items


def compact_phrase(text: str, max_words: int = 5, max_chars: int = 34) -> str:
    cleaned = re.sub(r"\s+", " ", text).strip()
    if not cleaned:
        return ""
    cleaned = re.sub(r"[«»\"]", "", cleaned)
    words = cleaned.split()
    if len(words) > max_words:
        cleaned = " ".join(words[:max_words])
    if len(cleaned) > max_chars:
        cleaned = cleaned[:max_chars].rstrip(" ,;:.")
    else:
        cleaned = cleaned.rstrip(" ,;:.")
    return cleaned


SUMMARY_REWRITES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"unification des francs", re.IGNORECASE), "Unification des Francs"),
    (re.compile(r"roi de tous les francs", re.IGNORECASE), "Roi de tous les Francs"),
    (re.compile(r"roi des francs saliens", re.IGNORECASE), "Roi des Francs saliens"),
    (re.compile(r"unifier une grande partie des (?:royaumes?\s+)?francs", re.IGNORECASE), "Unification des Francs"),
    (re.compile(r"conversion au christianisme", re.IGNORECASE), "Conversion au christianisme"),
    (re.compile(r"rattachement.*principaut", re.IGNORECASE), "Rattachement des principautés au domaine royal"),
    (re.compile(r"renforcement de l['’]autorité royale", re.IGNORECASE), "Autorité royale renforcée"),
    (re.compile(r"guerre de cent ans", re.IGNORECASE), "Fin de la guerre de Cent Ans"),
    (re.compile(r"empire carolingien", re.IGNORECASE), "Empire carolingien menacé"),
    (re.compile(r"maires du palais", re.IGNORECASE), "Pouvoir des maires du palais"),
    (re.compile(r"monarchie est abolie|monarchie abolie", re.IGNORECASE), "Monarchie abolie"),
    (re.compile(r"premi[èe]re r[ée]publique", re.IGNORECASE), "Première République"),
    (re.compile(r"consulat", re.IGNORECASE), "Consulat"),
    (re.compile(r"premier empire", re.IGNORECASE), "Premier Empire"),
    (re.compile(r"cent[- ]jours", re.IGNORECASE), "Cent-Jours"),
    (re.compile(r"territoires perdus|récup[ée]r", re.IGNORECASE), "Territoires récupérés"),
    (re.compile(r"familles aristocratiques", re.IGNORECASE), "Aristocratie montante"),
    (re.compile(r"pouvoir imp[ée]rial", re.IGNORECASE), "Pouvoir impérial fragilisé"),
    (re.compile(r"guerre entre héritiers", re.IGNORECASE), "Guerre entre héritiers"),
    (re.compile(r"succession monarchique est interrompue|succession monarchique", re.IGNORECASE), "Interrègne"),
    (re.compile(r"issu de la dynastie des ([^.,;:]+)", re.IGNORECASE), "Dynastie des \\1"),
)


def normalize_summary_text(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", text).strip()
    if not cleaned:
        return ""
    cleaned = re.sub(r"[«»\"]", "", cleaned)
    return cleaned


def shorten_summary_phrase(text: str, max_words: int) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]).rstrip(" ,;:.")


def essential_summary_phrase(text: str, max_words: int = 12, max_chars: int = 72) -> str:
    cleaned = normalize_summary_text(text)
    if not cleaned:
        return ""

    lower = cleaned.lower()
    for pattern, replacement in SUMMARY_REWRITES:
        match = pattern.search(lower)
        if match:
            if "\\1" in replacement and match.groups():
                return replacement.replace("\\1", match.group(1).strip().title())
            return replacement

    reign_match = re.search(
        r"^Règne de ([^,]+?) de (\d{3,4}) à (\d{3,4})\.?$",
        cleaned,
        flags=re.IGNORECASE,
    )
    if reign_match:
        name = reign_match.group(1).strip()
        start_year = reign_match.group(2)
        end_year = reign_match.group(3)
        return f"Règne de {name} {start_year} - {end_year}"

    royal_match = re.search(
        r"roi de france de (\d{3,4}) à (\d{3,4}|sa mort)\b",
        lower,
        flags=re.IGNORECASE,
    )
    if royal_match:
        return f"Roi de France de {royal_match.group(1)} à {royal_match.group(2)}"

    francs_match = re.search(
        r"roi des francs(?: saliens)?(?: de| jusqu['’]en)? (\d{3,4}) (?:à|jusqu['’]en) (\d{3,4}|sa mort)\b",
        lower,
        flags=re.IGNORECASE,
    )
    if francs_match:
        return f"Roi des Francs de {francs_match.group(1)} à {francs_match.group(2)}"

    aquitaine_match = re.search(
        r"roi d['’]aquitaine(?: jusqu['’]en)? (\d{3,4})",
        lower,
        flags=re.IGNORECASE,
    )
    if aquitaine_match:
        return f"Roi d'Aquitaine jusqu'en {aquitaine_match.group(1)}"

    empereur_match = re.search(
        r"empereur d['’]occident(?: de)? (\d{3,4})?(?: à (\d{3,4}))?",
        lower,
        flags=re.IGNORECASE,
    )
    if empereur_match and empereur_match.group(1):
        start_year = empereur_match.group(1)
        end_year = empereur_match.group(2) or "sa mort"
        return f"Empereur d'Occident de {start_year} à {end_year}"

    emperor_france_match = re.search(
        r"empereur des français de (\d{3,4}) à (\d{3,4}|sa mort)\b",
        lower,
        flags=re.IGNORECASE,
    )
    if emperor_france_match:
        return f"Empereur des Français de {emperor_france_match.group(1)} à {emperor_france_match.group(2)}"

    emperor_france_title_match = re.search(
        r"empereur des français\b",
        lower,
        flags=re.IGNORECASE,
    )
    if emperor_france_title_match:
        return "Empereur des Français"

    dynasty_match = re.search(
        r"appartient à la dynastie des ([^.,;:]+)",
        lower,
        flags=re.IGNORECASE,
    )
    if dynasty_match:
        dynasty = dynasty_match.group(1).strip()
        return f"Dynastie des {dynasty}"

    clause = re.split(r"(?<=[.!?])\s+", cleaned, maxsplit=1)[0]
    clause = re.sub(r"^([^,]+,\s+){1,2}", "", clause)
    clause = clause.strip(" ,;:.")
    clause = shorten_summary_phrase(clause, max_words=max_words)
    if max_chars > 0 and len(clause) > max_chars:
        clause = shorten_summary_phrase(clause, max_words=max(4, max_words - 2))
    if clause:
        return clause[:1].upper() + clause[1:]
    return ""


def build_card_context(segment: dict[str, str | int]) -> dict[str, str | tuple[str, str]]:
    display_name = str(segment["display_name"])
    dynasty = str(segment["dynasty"])
    notes = str(segment.get("notes", "")).strip()
    highlight_1, highlight_2 = FEATURE_OVERRIDES.get(display_name, ("", ""))
    if not highlight_1:
        source_texts = [notes, str(segment.get("fait_1", "")).strip(), str(segment.get("fait_2", "")).strip()]
        short_phrases = [essential_summary_phrase(text, max_words=8, max_chars=48) for text in source_texts if text]
        short_phrases = [phrase for phrase in short_phrases if phrase]
        if short_phrases:
            highlight_1 = short_phrases[0]
        if len(short_phrases) > 1:
            highlight_2 = short_phrases[1]
    if not highlight_1:
        highlight_1 = essential_summary_phrase(str(segment.get("fait_1", "")), max_words=8, max_chars=48)
    if not highlight_2:
        highlight_2 = essential_summary_phrase(str(segment.get("fait_2", "")), max_words=8, max_chars=48)
    return {
        "display_name": display_name,
        "dynasty": dynasty,
        "capital": CAPITAL_BY_DYNASTY.get(dynasty, "Paris"),
        "territory": TERRITORY_BY_DYNASTY.get(dynasty, "Royaume de France"),
        "highlights": (highlight_1 or "Règne structurant", highlight_2 or "Chronologie consolidée"),
    }


def draw_compass_rose(
    draw: ImageDraw.ImageDraw,
    center: tuple[int, int],
    radius: int,
    fill: tuple[int, int, int, int],
    accent: tuple[int, int, int, int],
) -> None:
    cx, cy = center
    outer = (cx - radius, cy - radius, cx + radius, cy + radius)
    draw.ellipse(outer, outline=accent, width=max(2, radius // 10))
    for angle, long_r, short_r in ((0, radius, radius // 2), (90, radius, radius // 2), (180, radius, radius // 2), (270, radius, radius // 2)):
        if angle == 0:
            points = [(cx, cy - long_r), (cx - 5, cy - short_r // 2), (cx + 5, cy - short_r // 2)]
        elif angle == 90:
            points = [(cx + long_r, cy), (cx + short_r // 2, cy - 5), (cx + short_r // 2, cy + 5)]
        elif angle == 180:
            points = [(cx, cy + long_r), (cx - 5, cy + short_r // 2), (cx + 5, cy + short_r // 2)]
        else:
            points = [(cx - long_r, cy), (cx - short_r // 2, cy - 5), (cx - short_r // 2, cy + 5)]
        draw.polygon(points, fill=fill)
    diag = int(radius * 0.72)
    for points in (
        [(cx, cy - diag), (cx - 4, cy - 4), (cx + 4, cy - 4)],
        [(cx + diag, cy), (cx + 4, cy - 4), (cx + 4, cy + 4)],
        [(cx, cy + diag), (cx - 4, cy + 4), (cx + 4, cy + 4)],
        [(cx - diag, cy), (cx - 4, cy - 4), (cx - 4, cy + 4)],
    ):
        draw.polygon(points, fill=accent)
    inner = (cx - radius // 4, cy - radius // 4, cx + radius // 4, cy + radius // 4)
    draw.ellipse(inner, fill=accent)


def draw_crown_icon(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], color: tuple[int, int, int, int]) -> None:
    x0, y0, x1, y1 = box
    w = x1 - x0
    h = y1 - y0
    base_y = y0 + h * 0.70
    points = [
        (x0 + w * 0.06, base_y),
        (x0 + w * 0.18, y0 + h * 0.30),
        (x0 + w * 0.34, y0 + h * 0.58),
        (x0 + w * 0.50, y0 + h * 0.18),
        (x0 + w * 0.66, y0 + h * 0.58),
        (x0 + w * 0.82, y0 + h * 0.30),
        (x0 + w * 0.94, base_y),
        (x0 + w * 0.94, y0 + h * 0.88),
        (x0 + w * 0.06, y0 + h * 0.88),
    ]
    draw.polygon(points, fill=color)
    for px in (0.18, 0.50, 0.82):
        draw.ellipse(
            (
                x0 + w * px - w * 0.05,
                y0 + h * 0.18 - w * 0.05,
                x0 + w * px + w * 0.05,
                y0 + h * 0.18 + w * 0.05,
            ),
            fill=color,
        )


def draw_pin_icon(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], color: tuple[int, int, int, int]) -> None:
    x0, y0, x1, y1 = box
    cx = (x0 + x1) / 2
    cy = y0 + (y1 - y0) * 0.44
    r = (x1 - x0) * 0.26
    draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=color)
    draw.polygon([(cx, y1 - 2), (cx - r * 0.90, cy + r * 0.20), (cx + r * 0.90, cy + r * 0.20)], fill=color)
    draw.ellipse((cx - r * 0.34, cy - r * 0.34, cx + r * 0.34, cy + r * 0.34), fill=(8, 18, 32, 255))


def draw_person_icon(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], color: tuple[int, int, int, int]) -> None:
    x0, y0, x1, y1 = box
    w = x1 - x0
    h = y1 - y0
    draw.ellipse((x0 + w * 0.30, y0 + h * 0.08, x0 + w * 0.70, y0 + h * 0.42), fill=color)
    draw.rounded_rectangle((x0 + w * 0.20, y0 + h * 0.42, x0 + w * 0.80, y0 + h * 0.92), radius=int(min(w, h) * 0.26), fill=color)


def draw_swords_icon(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], color: tuple[int, int, int, int]) -> None:
    x0, y0, x1, y1 = box
    draw.line((x0 + 4, y0 + 5, x1 - 4, y1 - 5), fill=color, width=4)
    draw.line((x0 + 4, y1 - 5, x1 - 4, y0 + 5), fill=color, width=4)
    draw.rectangle((x0 + 8, y0 + 18, x0 + 16, y0 + 28), fill=color)
    draw.rectangle((x1 - 16, y1 - 28, x1 - 8, y1 - 18), fill=color)
    draw.polygon([(x0 + 3, y0 + 3), (x0 + 8, y0 + 6), (x0 + 6, y0 + 11)], fill=color)
    draw.polygon([(x1 - 3, y1 - 3), (x1 - 8, y1 - 6), (x1 - 6, y1 - 11)], fill=color)


def draw_globe_icon(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], color: tuple[int, int, int, int]) -> None:
    x0, y0, x1, y1 = box
    draw.ellipse((x0 + 2, y0 + 2, x1 - 2, y1 - 2), outline=color, width=4)
    cx = (x0 + x1) / 2
    cy = (y0 + y1) / 2
    rx = (x1 - x0) / 2 - 4
    ry = (y1 - y0) / 2 - 4
    draw.ellipse((cx - rx * 0.68, cy - ry, cx + rx * 0.68, cy + ry), outline=color, width=2)
    draw.ellipse((cx - rx, cy - ry * 0.66, cx + rx, cy + ry * 0.66), outline=color, width=2)
    draw.line((cx - rx, cy, cx + rx, cy), fill=color, width=2)


def draw_header_logo(frame: Image.Image, position: tuple[int, int]) -> None:
    draw = ImageDraw.Draw(frame, "RGBA")
    draw_compass_rose(draw, (position[0] + 38, position[1] + 38), 32, (233, 193, 110, 255), (233, 193, 110, 230))
    # subtle stem to the left of the title
    draw.line((position[0] + 76, position[1] + 38, position[0] + 76, position[1] + 38), fill=(0, 0, 0, 0), width=0)


def tracked_text_width(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, tracking: int = 2) -> int:
    if not text:
        return 0
    width = 0
    for index, char in enumerate(text):
        width += text_width(draw, char, font)
        if index < len(text) - 1:
            width += tracking
    return width


def draw_tracked_text(
    draw: ImageDraw.ImageDraw,
    position: tuple[int, int],
    text: str,
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int, int] | tuple[int, int, int] | str,
    tracking: int = 2,
) -> None:
    x, y = position
    for index, char in enumerate(text):
        draw.text((x, y), char, font=font, fill=fill)
        x += text_width(draw, char, font)
        if index < len(text) - 1:
            x += tracking


def draw_shadowed_round_rect(
    frame: Image.Image,
    box: tuple[int, int, int, int],
    *,
    radius: int,
    fill: tuple[int, int, int, int],
    outline: tuple[int, int, int, int],
    width: int = 2,
    shadow_color: tuple[int, int, int, int] = (0, 0, 0, 120),
    shadow_offset: tuple[int, int] = (0, 12),
    shadow_blur: int = 16,
) -> None:
    shadow = Image.new("RGBA", frame.size, (0, 0, 0, 0))
    shadow_draw = ImageDraw.Draw(shadow, "RGBA")
    x0, y0, x1, y1 = box
    dx, dy = shadow_offset
    shadow_draw.rounded_rectangle((x0 + dx, y0 + dy, x1 + dx, y1 + dy), radius=radius, fill=shadow_color)
    shadow = shadow.filter(ImageFilter.GaussianBlur(radius=shadow_blur))
    frame.alpha_composite(shadow)
    draw = ImageDraw.Draw(frame, "RGBA")
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def fit_lines_to_width(
    draw: ImageDraw.ImageDraw,
    text: str,
    max_width: int,
    *,
    font_sizes: tuple[int, ...] = (22, 20, 18),
    max_lines: int = 2,
    bold: bool = False,
) -> tuple[ImageFont.ImageFont, list[str]]:
    best_font = load_font(font_sizes[-1], bold=bold)
    best_lines: list[str] = [text]
    for font_size in font_sizes:
        font = load_font(font_size, bold=bold)
        lines = wrap_text(draw, text, font, max_width, max_lines)
        if not lines:
            continue
        if len(lines) <= max_lines and all(text_width(draw, line, font) <= max_width for line in lines):
            return font, lines
        best_font = font
        best_lines = lines
    return best_font, best_lines


def build_timeline_bands(segments: list[dict[str, str | int]]) -> list[dict[str, str | int]]:
    bands: list[dict[str, str | int]] = []
    for segment in segments:
        start_year = int(segment["start_year"])
        end_year = int(segment["end_year"])
        dynasty = str(segment["dynasty"])
        color = str(segment["house_color"])
        if bands and bands[-1]["dynasty"] == dynasty and bands[-1]["color"] == color and start_year <= int(bands[-1]["end_year"]) + 1:
            bands[-1]["end_year"] = max(int(bands[-1]["end_year"]), end_year)
        else:
            bands.append(
                {
                    "start_year": start_year,
                    "end_year": end_year,
                    "dynasty": dynasty,
                    "color": color,
                }
            )
    if bands and bands[0]["dynasty"] == "Merovingiens" and int(bands[0]["start_year"]) > DISPLAY_START_YEAR:
        bands[0]["start_year"] = DISPLAY_START_YEAR
    return bands


def draw_rounded_image(frame: Image.Image, image: Image.Image, box: tuple[int, int, int, int], radius: int) -> None:
    width = box[2] - box[0]
    height = box[3] - box[1]
    if image.size != (width, height):
        image = ImageOps.fit(image, (width, height), method=Image.Resampling.LANCZOS, centering=(0.5, 0.34))
    mask = Image.new("L", (width, height), 0)
    mask_draw = ImageDraw.Draw(mask)
    mask_draw.rounded_rectangle((0, 0, width - 1, height - 1), radius=radius, fill=255)
    frame.paste(image, (box[0], box[1]), mask)


def draw_placeholder_portrait(
    frame: Image.Image,
    box: tuple[int, int, int, int],
    *,
    title: str,
    dynasty: str,
    accent: tuple[int, int, int, int],
) -> None:
    draw = ImageDraw.Draw(frame, "RGBA")
    x0, y0, x1, y1 = box
    draw.rounded_rectangle(box, radius=28, fill=(63, 45, 24, 255), outline=accent, width=2)
    for offset in range(0, x1 - x0, 28):
        shade = 18 if offset % 56 == 0 else 9
        draw.line((x0 + offset, y0 + 8, x0 + offset + 12, y1 - 8), fill=(233, 193, 110, shade), width=2)
    draw.ellipse((x0 + 18, y0 + 18, x0 + 88, y0 + 88), outline=(233, 193, 110, 60), width=2)
    draw.rounded_rectangle((x0 + 56, y0 + 115, x1 - 56, y1 - 38), radius=32, fill=(0, 0, 0, 30))
    draw.text((x0 + 28, y1 - 74), title, font=load_font(28, bold=True), fill="#f4efe7")
    draw.text((x0 + 28, y1 - 42), dynasty.upper(), font=load_font(18, bold=True), fill=(233, 193, 110, 220))


def draw_info_card(
    frame: Image.Image,
    box: tuple[int, int, int, int],
    *,
    title: str,
    icon: str,
    icon_color: tuple[int, int, int, int],
    value_lines: list[str],
) -> None:
    draw = ImageDraw.Draw(frame, "RGBA")
    x0, y0, x1, y1 = box
    draw.rounded_rectangle(box, radius=18, fill=(10, 24, 40, 198), outline=(233, 193, 110, 150), width=2)
    draw.rounded_rectangle((x0 + 1, y0 + 1, x1 - 1, y1 - 1), radius=17, outline=(255, 255, 255, 10), width=1)

    icon_box = (x0 + 18, y0 + 22, x0 + 66, y0 + 70)
    if icon == "crown":
        draw_crown_icon(draw, icon_box, icon_color)
    elif icon == "pin":
        draw_pin_icon(draw, icon_box, icon_color)
    elif icon == "person":
        draw_person_icon(draw, icon_box, icon_color)
    elif icon == "swords":
        draw_swords_icon(draw, icon_box, icon_color)
    elif icon == "globe":
        draw_globe_icon(draw, icon_box, icon_color)

    title_font = load_font(16, bold=True)
    title_x = x0 + 82
    title_y = y0 + 18
    draw.text((title_x, title_y), title, font=title_font, fill=(233, 193, 110, 255))

    value_area_width = max(60, x1 - title_x - 22)
    value_y = y0 + 44
    content_lines: list[str] = []
    candidate_fonts: list[ImageFont.ImageFont] = []
    for line in value_lines:
        if not line:
            continue
        candidate_font, wrapped = fit_lines_to_width(
            draw,
            line,
            value_area_width,
            font_sizes=(16, 14, 12, 11, 10),
            max_lines=4,
            bold=False,
        )
        candidate_fonts.append(candidate_font)
        content_lines.extend(wrapped)
    if not content_lines:
        content_lines = [""]
    content_font = min(candidate_fonts, key=lambda font: getattr(font, "size", 16)) if candidate_fonts else load_font(10, bold=False)
    line_step = max(10, text_size(draw, "Ag", content_font)[1] + 1)
    available_height = max(1, y1 - value_y - 4)
    total_height = len(content_lines) * line_step
    if total_height > available_height:
        line_step = max(8, available_height // max(1, len(content_lines)))
        total_height = len(content_lines) * line_step
    current_y = value_y + max(0, (available_height - total_height) // 2)
    for line in content_lines:
        draw.text((title_x, current_y), line, font=content_font, fill="#f5f1e8")
        current_y += line_step


def draw_legend_panel(
    frame: Image.Image,
    box: tuple[int, int, int, int],
    items: list[tuple[str, tuple[int, int, int]]],
) -> None:
    draw = ImageDraw.Draw(frame, "RGBA")
    x0, y0, x1, y1 = box
    draw_shadowed_round_rect(
        frame,
        box,
        radius=22,
        fill=(10, 24, 40, 205),
        outline=(233, 193, 110, 155),
        width=2,
        shadow_color=(0, 0, 0, 95),
        shadow_offset=(0, 10),
        shadow_blur=14,
    )
    draw.rounded_rectangle((x0 + 1, y0 + 1, x1 - 1, y1 - 1), radius=21, outline=(255, 255, 255, 12), width=1)
    title_font = load_font(17, bold=True)
    title = "DYNASTIES / ÉVÉNEMENTS"
    draw_tracked_text(draw, (x0 + 22, y0 + 18), title, title_font, (233, 193, 110, 255), tracking=1)
    content_top = y0 + 62
    content_bottom = y1 - 22
    row_height = max(24, (content_bottom - content_top) // max(1, len(items)))
    chip_size = min(26, max(16, row_height - 8))
    label_x = x0 + 62
    label_max_width = x1 - label_x - 18

    for index, (label, color) in enumerate(items):
        row_top = content_top + index * row_height
        chip_y = row_top + (row_height - chip_size) // 2
        draw.rounded_rectangle(
            (x0 + 22, chip_y, x0 + 22 + chip_size, chip_y + chip_size),
            radius=8,
            fill=(*color, 255),
        )
        item_font, label_lines = fit_lines_to_width(
            draw,
            label,
            label_max_width,
            font_sizes=(18, 17, 16, 15, 14, 13, 12),
            max_lines=1,
            bold=False,
        )
        label_height = text_size(draw, label_lines[0], item_font)[1]
        label_y = row_top + max(0, (row_height - label_height) // 2 - 1)
        draw.text((label_x, label_y), label_lines[0], font=item_font, fill="#f5f1e8")


def draw_header_block(frame: Image.Image, start_year: int, end_year: int) -> None:
    draw = ImageDraw.Draw(frame, "RGBA")
    brand_font = load_font(58, bold=True)
    brand_sub_font = load_font(18, bold=False)
    header_font = load_font(20, bold=False)
    header_year_font = load_font(27, bold=True)
    gold = (233, 193, 110, 255)
    cream = (245, 241, 232, 255)

    draw_compass_rose(draw, (72, 72), 34, gold, (255, 227, 160, 245))
    draw_tracked_text(draw, (118, 46), "HISTOVISION", brand_font, gold, tracking=1)
    draw.text((120, 104), "L'HISTOIRE EN CARTES ET EN CHIFFRES", font=brand_sub_font, fill=(220, 194, 134, 230))

    right_edge = WIDTH - 80
    label = EPISODE_LABEL
    draw.text((right_edge - text_width(draw, label, header_font), 54), label, font=header_font, fill=cream)
    year_range = f"{start_year} - {end_year}"
    draw.text((right_edge - text_width(draw, year_range, header_year_font), 84), year_range, font=header_year_font, fill=gold)


def draw_footer_block(frame: Image.Image) -> None:
    draw = ImageDraw.Draw(frame, "RGBA")
    y = 1000
    gold = (233, 193, 110, 120)
    draw.line((52, y, 770, y), fill=gold, width=2)
    draw.line((1158, y, WIDTH - 52, y), fill=gold, width=2)
    draw_compass_rose(draw, (WIDTH // 2 - 72, y + 7), 18, (233, 193, 110, 255), (255, 225, 150, 240))
    footer_font = load_font(24, bold=True)
    draw_tracked_text(draw, (WIDTH // 2 - 40, y - 5), "HISTOVISION", footer_font, (233, 193, 110, 255), tracking=3)


def build_audio_track(audio_path: Path, duration: float):
    base = AudioFileClip(str(audio_path))
    fade_out = max(0.0, min(FINAL_AUDIO_FADE_OUT, duration))
    if base.duration >= duration:
        clip = base.subclipped(0, duration)
        if fade_out > 0:
            clip = clip.with_effects([AudioFadeOut(fade_out)])
        return clip, [base]

    clips = []
    keep_alive = [base]
    step = max(0.1, base.duration - LOOP_CROSSFADE)
    loops = int(math.ceil(max(0.0, duration - LOOP_CROSSFADE) / step))
    for index in range(loops):
        segment = base.with_start(index * step)
        if LOOP_CROSSFADE > 0:
            segment = segment.with_effects([AudioFadeIn(LOOP_CROSSFADE), AudioFadeOut(LOOP_CROSSFADE)])
        clips.append(segment)
    mixed = CompositeAudioClip(clips).with_duration(duration)
    if fade_out > 0:
        mixed = mixed.with_effects([AudioFadeOut(fade_out)])
    return mixed, keep_alive



def render_video(rows: list[dict[str, str]], output_path: Path, duration: float, fps: int) -> Path:
    segments = build_segments(rows)
    background = make_background()
    title_font = load_font(58, bold=True)
    subtitle_font = load_font(24, bold=False)
    year_font = load_font(112, bold=True)
    name_font = load_font(46, bold=True)
    meta_font = load_font(28, bold=True)
    axis_font = load_font(18, bold=True)
    legend_title_font = load_font(18, bold=True)
    legend_font = load_font(17, bold=True)

    start_year = int(segments[0]["start_year"])
    end_year = int(segments[-1]["end_year"])
    total_years = end_year - start_year

    portraits: dict[str, Image.Image | None] = {}
    for segment in segments:
        name = str(segment["display_name"])
        if name not in portraits:
            portraits[name] = load_portrait(name)

    timeline_left = 100
    timeline_right = WIDTH - 100
    timeline_top = 780
    timeline_height = 74
    timeline_width = timeline_right - timeline_left

    def year_to_x(year: float) -> int:
        ratio = (year - start_year) / max(1, total_years)
        return timeline_left + int(ratio * timeline_width)

    def make_frame(t: float) -> np.ndarray:
        year = start_year + int(round((t / duration) * total_years))
        year = max(start_year, min(end_year, year))
        segment = find_segment(segments, year)
        name = str(segment["display_name"])
        dynasty = str(segment["dynasty"])
        color = str(segment["house_color"])
        rgb = hex_to_rgb(color)

        frame = background.copy()
        draw = ImageDraw.Draw(frame, "RGBA")

        draw.text((92, 68), "ROIS ET EMPEREURS DE FRANCE", font=title_font, fill="#f4efe7")
        draw.text((96, 136), "Timeline annuelle canonique de Clovis Ier à Napoléon III", font=subtitle_font, fill="#d6dfeb")

        card = (84, 196, WIDTH - 84, 724)
        draw.rounded_rectangle(card, radius=34, fill=(8, 18, 32, 196), outline=(255, 255, 255, 24), width=2)
        draw.rounded_rectangle((110, 228, 468, 586), radius=34, fill=(*rgb, 255), outline=(255, 255, 255, 22), width=2)
        portrait = portraits.get(name)
        if portrait is not None:
            frame.alpha_composite(portrait, (119, 237))

        draw_glow_text(frame, (540, 244), str(year), year_font, (244, 239, 231), (231, 189, 103))
        draw.text((540, 390), name, font=name_font, fill="#f4efe7")
        draw.text((544, 458), dynasty.upper(), font=meta_font, fill=(rgb[0], rgb[1], rgb[2], 255))
        reign_text = f"{segment['start_year']} - {segment['end_year']}"
        draw.text((544, 506), reign_text, font=meta_font, fill="#d5e0eb")
        facts = [
            str(segment.get("fait_1", "")).strip(),
            str(segment.get("fait_2", "")).strip(),
            str(segment.get("fait_3", "")).strip(),
        ]
        fact_y = 554
        fact_text_x = 576
        fact_text_width = card[2] - fact_text_x - 84
        fact_font, fact_line_step, fact_gap, wrapped_facts = fit_fact_layout(
            draw,
            facts,
            fact_text_width,
            164,
            base_size=17,
            min_size=11,
            max_lines=4,
        )
        bullet_size = 16
        for wrapped in wrapped_facts:
            bullet_top = fact_y + max(4, (fact_line_step - bullet_size) // 2)
            draw.rounded_rectangle(
                (544, bullet_top, 544 + bullet_size, bullet_top + bullet_size),
                radius=6,
                fill=(rgb[0], rgb[1], rgb[2], 255),
            )
            line_y = fact_y
            for line in wrapped:
                draw.text((fact_text_x, line_y), line, font=fact_font, fill="#d5e0eb")
                line_y += fact_line_step
            fact_y = line_y + fact_gap

        legend_box = (84, 734, WIDTH - 84, 772)
        draw.rounded_rectangle(legend_box, radius=18, fill=(8, 18, 32, 132), outline=(255, 255, 255, 18), width=1)
        legend_items = unique_dynasty_items(segments)
        legend_label = "LÉGENDE DE LA FRISE"
        legend_label_x = 108
        legend_label_y = 742
        draw.text((legend_label_x, legend_label_y), legend_label, font=legend_title_font, fill="#f4efe7")
        legend_x = legend_label_x + text_width(draw, legend_label, legend_title_font) + 30
        legend_chip_size = 18
        legend_chip_gap = 14
        for dynasty_name, dynasty_color in legend_items:
            draw.rounded_rectangle(
                (legend_x, legend_label_y + 2, legend_x + legend_chip_size, legend_label_y + 2 + legend_chip_size),
                radius=6,
                fill=dynasty_color,
            )
            draw.text((legend_x + 26, legend_label_y + 1), dynasty_name, font=legend_font, fill="#f4efe7")
            legend_x += 26 + text_width(draw, dynasty_name, legend_font) + legend_chip_gap

        for segment_item in segments:
            sx = year_to_x(int(segment_item["start_year"]))
            ex = year_to_x(int(segment_item["end_year"]) + 1)
            draw.rounded_rectangle((sx, timeline_top, max(sx + 4, ex), timeline_top + timeline_height), radius=14, fill=hex_to_rgb(str(segment_item["house_color"])))

        for tick in (500, 800, 1000, 1200, 1400, 1600, 1800):
            if start_year <= tick <= end_year:
                x = year_to_x(tick)
                draw.line((x, timeline_top - 24, x, timeline_top + timeline_height + 24), fill=(255, 255, 255, 22), width=2)
                draw.text((x - 24, timeline_top + 90), str(tick), font=axis_font, fill="#d6dfeb")

        marker_x = year_to_x(year)
        draw.rounded_rectangle((marker_x - 6, timeline_top - 30, marker_x + 6, timeline_top + timeline_height + 30), radius=6, fill=(244, 239, 231, 255))
        draw.ellipse((marker_x - 16, timeline_top + 18, marker_x + 16, timeline_top + 50), fill=(244, 239, 231, 255))

        return np.array(frame.convert("RGB"))

    clip = VideoClip(make_frame, duration=duration)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    clip.write_videofile(str(output_path), fps=fps, codec="libx264", audio=False)
    clip.close()
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a France kings timeline video.")
    parser.add_argument("--input", type=Path, default=INPUT_CSV)
    parser.add_argument("--output", type=Path, default=OUTPUT_MP4)
    parser.add_argument("--duration", type=float, default=DURATION)
    parser.add_argument("--fps", type=int, default=FPS)
    parser.add_argument("--audio", type=Path, default=DEFAULT_AUDIO)
    return parser.parse_args()


def render_video(rows: list[dict[str, str]], output_path: Path, duration: float, fps: int, audio_path: Path) -> Path:
    segments = build_segments(rows)
    timeline_bands = build_timeline_bands(segments)
    background = make_background()
    year_font = load_font(112, bold=True)
    name_font = load_font(60, bold=True)
    dynasty_font = load_font(28, bold=True)
    reign_font = load_font(26, bold=True)
    axis_font = load_font(18, bold=True)
    bubble_font = load_font(20, bold=True)

    animation_start_year = int(segments[0]["start_year"])
    animation_end_year = int(segments[-1]["end_year"])
    total_years = animation_end_year - animation_start_year

    portraits: dict[str, Image.Image | None] = {}
    for segment in segments:
        name = str(segment["display_name"])
        if name not in portraits:
            portraits[name] = load_portrait(name)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    audio_clip, keep_alive = build_audio_track(audio_path, duration)

    timeline_left = 52
    timeline_right = WIDTH - 52
    timeline_top = 650
    timeline_height = 64
    timeline_width = timeline_right - timeline_left
    axis_start_year = DISPLAY_START_YEAR
    axis_end_year = DISPLAY_END_YEAR
    last_year: int | None = None
    last_frame: np.ndarray | None = None

    def year_to_x(year: float) -> int:
        ratio = (year - axis_start_year) / max(1, axis_end_year - axis_start_year)
        ratio = max(0.0, min(1.0, ratio))
        return timeline_left + int(ratio * timeline_width)

    def make_frame(t: float) -> np.ndarray:
        year = animation_start_year + int(round((t / duration) * total_years))
        year = max(animation_start_year, min(animation_end_year, year))
        nonlocal last_year, last_frame
        if last_year == year and last_frame is not None:
            return last_frame
        segment = find_segment(segments, year)
        name = str(segment["display_name"])
        dynasty = str(segment["dynasty"])
        color = str(segment["house_color"])
        rgb = hex_to_rgb(color)
        context = build_card_context(segment)

        frame = background.copy()
        draw = ImageDraw.Draw(frame, "RGBA")

        draw_header_block(frame, animation_start_year, animation_end_year)

        main_box = (52, 146, WIDTH - 52, 624)
        draw_shadowed_round_rect(
            frame,
            main_box,
            radius=24,
            fill=(8, 20, 35, 222),
            outline=(233, 193, 110, 180),
            width=2,
            shadow_color=(0, 0, 0, 90),
            shadow_offset=(0, 10),
            shadow_blur=16,
        )
        draw.rounded_rectangle((main_box[0] + 1, main_box[1] + 1, main_box[2] - 1, main_box[3] - 1), radius=23, outline=(255, 255, 255, 12), width=1)

        portrait_outer = (80, 172, 455, 600)
        portrait_inner = (84, 176, 451, 596)
        draw.rounded_rectangle((portrait_outer[0] - 2, portrait_outer[1] - 2, portrait_outer[2] + 2, portrait_outer[3] + 2), radius=28, outline=(120, 84, 170, 220), width=4)
        draw.rounded_rectangle(portrait_outer, radius=26, outline=(233, 193, 110, 180), width=2, fill=(34, 21, 16, 140))
        portrait = portraits.get(name)
        if portrait is not None:
            draw_rounded_image(frame, portrait, portrait_inner, radius=24)
        else:
            draw_placeholder_portrait(frame, portrait_inner, title=name, dynasty=dynasty, accent=(233, 193, 110, 160))
        draw.rounded_rectangle((portrait_outer[0], portrait_outer[1], portrait_outer[2], portrait_outer[3]), radius=26, outline=(255, 255, 255, 12), width=1)

        draw_glow_text(frame, (548, 190), str(year), year_font, (247, 242, 233), (233, 193, 110))
        draw.text((548, 314), name, font=name_font, fill="#f7f1e8")
        draw.text((548, 384), dynasty.upper(), font=dynasty_font, fill=rgb + (255,))
        reign_text = f"{segment['start_year']} - {segment['end_year']}"
        draw.text((548, 431), reign_text, font=reign_font, fill=(233, 193, 110, 255))

        summary_candidates = [
            essential_summary_phrase(str(segment.get("fait_1", "")).strip(), max_words=12, max_chars=84),
            essential_summary_phrase(str(segment.get("fait_2", "")).strip(), max_words=12, max_chars=84),
            essential_summary_phrase(str(segment.get("fait_3", "")).strip(), max_words=12, max_chars=84),
            context["highlights"][0],
            context["highlights"][1],
        ]

        def is_weak_summary(value: str) -> bool:
            lower_value = value.lower()
            return (
                lower_value.startswith("règne de ")
                or lower_value.startswith("roi ")
                or lower_value.startswith("dynastie des ")
                or lower_value.startswith("né ")
                or lower_value.startswith("mort ")
                or "appartient à la dynastie" in lower_value
                or "figure dans la succession" in lower_value
                or lower_value.startswith("issu de la dynastie")
            )

        summary_lines: list[str] = []
        weak_candidates: list[str] = []
        seen_candidates: set[str] = set()
        for candidate in summary_candidates:
            if not candidate:
                continue
            key = candidate.lower()
            if key in seen_candidates:
                continue
            seen_candidates.add(key)
            if is_weak_summary(candidate):
                weak_candidates.append(candidate)
                continue
            summary_lines.append(candidate)
            if len(summary_lines) >= 2:
                break
        if len(summary_lines) < 2:
            for candidate in weak_candidates:
                if candidate.lower() in {line.lower() for line in summary_lines}:
                    continue
                summary_lines.append(candidate)
                if len(summary_lines) >= 2:
                    break
        if not summary_lines:
            summary_lines = ["", ""]
        fact_x = 580
        fact_y = 456
        fact_max_width = 840
        fact_font, fact_line_step, fact_gap, wrapped_facts = fit_fact_layout(
            draw,
            summary_lines[:2],
            fact_max_width,
            148,
            base_size=24,
            min_size=18,
            max_lines=2,
        )
        bullet_size = 18
        for wrapped in wrapped_facts:
            bullet_top = fact_y + max(4, (fact_line_step - bullet_size) // 2)
            draw.rounded_rectangle(
                (548, bullet_top, 548 + bullet_size, bullet_top + bullet_size),
                radius=6,
                fill=(rgb[0], rgb[1], rgb[2], 255),
            )
            line_y = fact_y
            for line in wrapped:
                draw.text((fact_x, line_y), line, font=fact_font, fill="#d8e1eb")
                line_y += fact_line_step
            fact_y = line_y + fact_gap

        legend_box = (1495, 175, 1848, 595)
        draw_legend_panel(frame, legend_box, unique_dynasty_items(segments))

        timeline_base = (timeline_left, timeline_top, timeline_right, timeline_top + timeline_height)
        draw.rounded_rectangle(timeline_base, radius=20, fill=(11, 24, 41, 180), outline=(255, 255, 255, 18), width=1)
        draw.rounded_rectangle((timeline_left + 1, timeline_top + 1, timeline_right - 1, timeline_top + timeline_height - 1), radius=19, outline=(233, 193, 110, 20), width=1)

        for band in timeline_bands:
            sx = year_to_x(int(band["start_year"]))
            ex = year_to_x(int(band["end_year"]) + 1)
            if ex <= sx:
                ex = sx + 4
            draw.rounded_rectangle((sx, timeline_top + 4, ex, timeline_top + timeline_height - 4), radius=14, fill=hex_to_rgb(str(band["color"])) + (230,))

        tick_years = (400, 800, 1000, 1200, 1400, 1600, 1800, 1870)
        for tick in tick_years:
            x = year_to_x(tick)
            draw.line((x, timeline_top - 18, x, timeline_top + timeline_height + 18), fill=(255, 255, 255, 80), width=2)
            draw.text((x - 18, timeline_top + 72), str(tick), font=axis_font, fill=(216, 225, 235, 255))

        start_x = year_to_x(DISPLAY_START_YEAR)
        draw.line((start_x, timeline_top - 14, start_x, timeline_top + timeline_height + 14), fill=(243, 236, 225, 220), width=5)
        draw.ellipse((start_x - 8, timeline_top + 17, start_x + 8, timeline_top + 33), fill=(243, 236, 225, 240))

        marker_x = year_to_x(year)
        bubble_text = str(year)
        bubble_width = text_width(draw, bubble_text, bubble_font) + 34
        bubble_height = 34
        bubble_x0 = marker_x - bubble_width // 2
        bubble_y0 = timeline_top - 54
        bubble_box = (bubble_x0, bubble_y0, bubble_x0 + bubble_width, bubble_y0 + bubble_height)
        draw.rounded_rectangle(bubble_box, radius=13, fill=(10, 24, 40, 240), outline=(233, 193, 110, 255), width=2)
        draw.text(
            (
                bubble_box[0] + (bubble_width - text_width(draw, bubble_text, bubble_font)) / 2,
                bubble_box[1] + 5,
            ),
            bubble_text,
            font=bubble_font,
            fill=(247, 241, 232, 255),
        )
        draw.line((marker_x, bubble_box[3], marker_x, timeline_top + 14), fill=(233, 193, 110, 240), width=2)
        draw.ellipse((marker_x - 6, timeline_top + 17, marker_x + 6, timeline_top + 29), fill=(233, 193, 110, 255))

        card_y = 757
        card_h = 104
        card_gap = 18
        card_w = (WIDTH - 2 * 52 - 4 * card_gap) // 5
        card_specs = [
            ("RÈGNE", "crown", [f"{segment['start_year']} - {segment['end_year']}"]),
            ("CAPITALE", "pin", [str(context["capital"])]),
            ("DYNASTIE", "person", [dynasty]),
            ("FAITS MARQUANTS", "swords", [str(context["highlights"][0]), str(context["highlights"][1])]),
            ("TERRITOIRE", "globe", [line for line in str(context["territory"]).splitlines() if line.strip()]),
        ]
        card_x = 52
        for title, icon, lines in card_specs:
            draw_info_card(
                frame,
                (card_x, card_y, card_x + card_w, card_y + card_h),
                title=title,
                icon=icon,
                icon_color=(233, 193, 110, 255),
                value_lines=lines,
            )
            card_x += card_w + card_gap

        draw_footer_block(frame)

        last_year = year
        last_frame = np.array(frame.convert("RGB"))
        return last_frame

    clip = VideoClip(make_frame, duration=duration)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    clip = clip.with_audio(audio_clip)
    clip.write_videofile(str(output_path), fps=fps, codec="libx264", audio_codec="aac")
    clip.close()
    audio_clip.close()
    for item in keep_alive:
        item.close()
    return output_path


def main() -> None:
    args = parse_args()
    rows = load_rows(args.input)
    output = render_video(rows, args.output, args.duration, args.fps, args.audio)
    print(f"[history] timeline video generated -> {output}")


if __name__ == "__main__":
    main()
