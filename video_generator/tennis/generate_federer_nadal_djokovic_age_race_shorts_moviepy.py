from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from moviepy import VideoClip
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont, ImageOps

from video_generator.generate_ucl_barchart_race_moviepy import (
    DEFAULT_AUDIO,
    _fit_font_size,
    _load_font,
    build_audio_track,
)


DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "tennis" / "nadal_djokovic_federer_grand_slam_titles_by_age.mp4"
DEFAULT_PHOTOS_DIR = PROJECT_ROOT / "data" / "raw" / "player_photos"
DEFAULT_LOGOS_DIR = PROJECT_ROOT / "data" / "raw" / "tennis_logos"
DEFAULT_PREVIEW = PROJECT_ROOT / "tmp_frames" / "federer_nadal_djokovic_age_race_preview.png"

WIDTH = 1080
HEIGHT = 1920
FPS = 60

AGE_MIN = 18
AGE_MAX = 37
INTRO_HOLD = 1.15
AGE_STEP_DURATION = 0.82
OUTRO_HOLD = 1.35
TOTAL_DURATION = 40.0
VALUE_SCALE_MAX = 24.0

TITLE = "NADAL vs DJOKOVIC vs FEDERER"
SUBTITLE = "Grand Slam Titles by Age"
FOOTER = "Tennis Clash"

TICK_AGES = [18, 20, 23, 26, 29, 32, 35]
VERTICAL_SHIFT = 16
BLOCK_SHIFT = 110
CHART_LEFT = 358
CARD_X = 68
ROW_CENTER_OFFSET = 0

FEDERER_COLOR = "#F4F1E8"
NADAL_COLOR = "#F11D2A"
DJOKOVIC_COLOR = "#F3C94F"

PLAYER_TIE_ORDER = {
    "Rafael Nadal": 0,
    "Novak Djokovic": 1,
    "Roger Federer": 2,
}

TOURNAMENT_LOGO_FILES = {
    "AO": "australian_open.png",
    "RG": "roland_garros.jpg",
    "WIM": "wimbledon.png",
    "USO": "us_open.png",
}
_LOGO_TILE_CACHE: dict[tuple[int, int], Image.Image] = {}


@dataclass(frozen=True)
class Player:
    name: str
    short: str
    photo_name: str
    color: str
    counts: dict[int, int]


PLAYERS = [
    Player(
        name="Rafael Nadal",
        short="NADAL",
        photo_name="rafael_nadal.jpg",
        color=NADAL_COLOR,
        counts={
            17: 0,
            18: 0,
            19: 1,
            20: 2,
            21: 3,
            22: 5,
            23: 6,
            24: 9,
            25: 10,
            26: 11,
            27: 13,
            28: 14,
            29: 14,
            30: 14,
            31: 16,
            32: 17,
            33: 19,
            34: 20,
            35: 20,
            36: 22,
            37: 22,
        },
    ),
    Player(
        name="Novak Djokovic",
        short="DJOKOVIC",
        photo_name="novak_djokovic.jpg",
        color=DJOKOVIC_COLOR,
        counts={
            17: 0,
            18: 0,
            19: 0,
            20: 1,
            21: 1,
            22: 1,
            23: 1,
            24: 4,
            25: 5,
            26: 6,
            27: 7,
            28: 10,
            29: 12,
            30: 12,
            31: 14,
            32: 16,
            33: 17,
            34: 20,
            35: 21,
            36: 24,
            37: 24,
        },
    ),
    Player(
        name="Roger Federer",
        short="FEDERER",
        photo_name="roger_federer.jpg",
        color=FEDERER_COLOR,
        counts={
            17: 0,
            18: 0,
            19: 0,
            20: 0,
            21: 1,
            22: 3,
            23: 6,
            24: 7,
            25: 10,
            26: 12,
            27: 13,
            28: 16,
            29: 16,
            30: 16,
            31: 17,
            32: 17,
            33: 17,
            34: 17,
            35: 18,
            36: 20,
            37: 20,
        },
    ),
]

SEGMENT_SLAMS: dict[str, dict[int, list[str]]] = {
    "Rafael Nadal": {
        18: ["RG"],
        19: ["RG"],
        20: ["RG"],
        21: ["RG", "WIM"],
        22: ["AO"],
        23: ["RG", "WIM", "USO"],
        24: ["RG"],
        25: ["RG"],
        26: ["RG", "USO"],
        27: ["RG"],
        30: ["RG", "USO"],
        31: ["RG"],
        32: ["RG", "USO"],
        33: ["RG"],
        35: ["AO", "RG"],
    },
    "Novak Djokovic": {
        19: ["AO"],
        23: ["AO", "WIM", "USO"],
        24: ["AO"],
        25: ["AO"],
        26: ["WIM"],
        27: ["AO", "WIM", "USO"],
        28: ["AO", "RG"],
        30: ["WIM", "USO"],
        31: ["AO", "WIM"],
        32: ["AO"],
        33: ["AO", "RG", "WIM"],
        34: ["WIM"],
        35: ["AO", "RG", "USO"],
    },
    "Roger Federer": {
        20: ["WIM"],
        21: ["AO", "WIM"],
        22: ["USO", "WIM", "USO"],
        23: ["AO"],
        24: ["WIM", "USO", "AO"],
        25: ["WIM", "USO"],
        26: ["USO"],
        27: ["RG", "WIM", "AO"],
        30: ["WIM"],
        34: ["AO"],
        35: ["WIM", "AO"],
    },
}


def _hex_to_rgb(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    return tuple(int(value[index : index + 2], 16) for index in (0, 2, 4))


def _mix_rgb(color: str, target: tuple[int, int, int], amount: float) -> tuple[int, int, int]:
    amount = max(0.0, min(1.0, amount))
    r, g, b = _hex_to_rgb(color)
    return (
        int(r + (target[0] - r) * amount),
        int(g + (target[1] - g) * amount),
        int(b + (target[2] - b) * amount),
    )


def _darken(color: str, amount: float) -> tuple[int, int, int]:
    r, g, b = _hex_to_rgb(color)
    return (
        int(r * (1.0 - amount)),
        int(g * (1.0 - amount)),
        int(b * (1.0 - amount)),
    )


def _text_on(color: str) -> str:
    r, g, b = _hex_to_rgb(color)
    luminance = (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255.0
    return "#0D1421" if luminance > 0.66 else "#F7F8FC"


def _smoothstep(value: float) -> float:
    value = max(0.0, min(1.0, value))
    return value * value * (3.0 - 2.0 * value)


def _truncate_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int) -> str:
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


def _player_initials(name: str) -> str:
    parts = [part for part in name.replace("-", " ").split() if part]
    if len(parts) == 1:
        return parts[0][:2].upper()
    return (parts[0][0] + parts[-1][0]).upper()


def _count_for_age(player: Player, age: float) -> float:
    low = int(math.floor(age))
    high = min(AGE_MAX, low + 1)
    alpha = _smoothstep(age - low)
    low_count = player.counts[low]
    high_count = player.counts[high]
    return low_count + (high_count - low_count) * alpha


def _gain_for_segment(player: Player, age: int) -> int:
    current = max(AGE_MIN, min(AGE_MAX - 1, age))
    return player.counts[current + 1] - player.counts[current]


def _slam_logos_for_segment(player: Player, age: int) -> list[str]:
    gain = _gain_for_segment(player, age)
    slams = SEGMENT_SLAMS.get(player.name, {}).get(age, [])
    if len(slams) >= gain:
        return slams[:gain]
    return slams + ["AO"] * max(0, gain - len(slams))


def _rank_map_at_age(age: int) -> dict[str, int]:
    ranked = sorted(
        PLAYERS,
        key=lambda player: (-player.counts[age], PLAYER_TIE_ORDER[player.name], player.name),
    )
    return {player.name: index for index, player in enumerate(ranked)}


def _interpolated_rank_map(age: float) -> dict[str, float]:
    low = int(math.floor(age))
    high = min(AGE_MAX, low + 1)
    alpha = _smoothstep(age - low)
    low_ranks = _rank_map_at_age(low)
    high_ranks = _rank_map_at_age(high)
    return {
        player.name: low_ranks[player.name] + (high_ranks[player.name] - low_ranks[player.name]) * alpha
        for player in PLAYERS
    }


def _visible_age_labels(current_age: int) -> list[int]:
    first = max(AGE_MIN, current_age - 1)
    last = min(AGE_MAX, current_age + 4)
    return list(range(first, last + 1))


def _make_background() -> Image.Image:
    xx = np.linspace(0, 1, WIDTH, dtype=np.float32)
    yy = np.linspace(0, 1, HEIGHT, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(xx, yy)

    top_left = np.array([132, 95, 255], dtype=np.float32)
    top_right = np.array([92, 196, 255], dtype=np.float32)
    bottom_left = np.array([217, 106, 255], dtype=np.float32)
    bottom_right = np.array([126, 118, 255], dtype=np.float32)

    base = (
        top_left[None, None, :] * (1.0 - grid_x[..., None]) * (1.0 - grid_y[..., None])
        + top_right[None, None, :] * grid_x[..., None] * (1.0 - grid_y[..., None])
        + bottom_left[None, None, :] * (1.0 - grid_x[..., None]) * grid_y[..., None]
        + bottom_right[None, None, :] * grid_x[..., None] * grid_y[..., None]
    )

    glow_top = np.exp(-(((grid_x - 0.58) / 0.28) ** 2 + ((grid_y - 0.12) / 0.10) ** 2))
    glow_left = np.exp(-(((grid_x - 0.18) / 0.22) ** 2 + ((grid_y - 0.42) / 0.24) ** 2))
    glow_right = np.exp(-(((grid_x - 0.88) / 0.20) ** 2 + ((grid_y - 0.28) / 0.22) ** 2))
    glow_bottom = np.exp(-(((grid_x - 0.50) / 0.45) ** 2 + ((grid_y - 0.88) / 0.12) ** 2))

    image = np.clip(
        base
        + np.array([255, 255, 255], dtype=np.float32)[None, None, :] * (0.08 * glow_top[..., None])
        + np.array([186, 106, 255], dtype=np.float32)[None, None, :] * (0.20 * glow_left[..., None])
        + np.array([94, 201, 255], dtype=np.float32)[None, None, :] * (0.18 * glow_right[..., None])
        + np.array([255, 255, 255], dtype=np.float32)[None, None, :] * (0.05 * glow_bottom[..., None]),
        0,
        255,
    ).astype(np.uint8)

    frame = Image.fromarray(image).convert("RGBA")
    overlay = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")
    draw.rounded_rectangle((28, 24, WIDTH - 28, HEIGHT - 24), radius=38, outline=(255, 255, 255, 22), width=2)
    draw.rounded_rectangle((42, 38, WIDTH - 42, HEIGHT - 38), radius=34, outline=(255, 255, 255, 10), width=1)
    overlay = overlay.filter(ImageFilter.GaussianBlur(radius=1.5))
    frame.alpha_composite(overlay)
    return frame


def _load_player_photo(photos_dir: Path, player: Player, size: int) -> Image.Image | None:
    path = photos_dir / player.photo_name
    if not path.exists():
        return None
    try:
        image = ImageOps.exif_transpose(Image.open(path)).convert("RGBA")
        image = ImageOps.fit(image, (size, size), method=Image.Resampling.LANCZOS, centering=(0.5, 0.22))
        return image
    except Exception:
        return None


def _build_player_card(player: Player, photos_dir: Path) -> Image.Image:
    card_size = 250
    photo_size = 224
    canvas = Image.new("RGBA", (card_size, card_size), (0, 0, 0, 0))

    shadow = Image.new("RGBA", (card_size, card_size), (0, 0, 0, 0))
    shadow_draw = ImageDraw.Draw(shadow, "RGBA")
    shadow_draw.rounded_rectangle((10, 12, card_size - 10, card_size - 8), radius=40, fill=(0, 0, 0, 72))
    shadow = shadow.filter(ImageFilter.GaussianBlur(radius=8))
    canvas.alpha_composite(shadow)

    draw = ImageDraw.Draw(canvas, "RGBA")
    draw.rounded_rectangle((0, 0, card_size - 18, card_size - 18), radius=40, fill=(255, 255, 255, 245))
    draw.rounded_rectangle((5, 5, card_size - 23, card_size - 23), radius=34, fill=(255, 255, 255, 255))

    photo = _load_player_photo(photos_dir, player, photo_size)
    if photo is not None:
        photo_mask = Image.new("L", (photo_size, photo_size), 0)
        ImageDraw.Draw(photo_mask).rounded_rectangle((0, 0, photo_size - 1, photo_size - 1), radius=24, fill=255)
        canvas.paste(photo, (11, 11), photo_mask)
    else:
        fallback = Image.new("RGBA", (photo_size, photo_size), (255, 255, 255, 255))
        fd = ImageDraw.Draw(fallback)
        initials_font = _load_font(28, bold=True)
        initials = _player_initials(player.name)
        bbox = fd.textbbox((0, 0), initials, font=initials_font)
        fd.text(
            ((photo_size - (bbox[2] - bbox[0])) // 2, (photo_size - (bbox[3] - bbox[1])) // 2 - 2),
            initials,
            font=initials_font,
            fill="#1A2233",
        )
        canvas.paste(fallback, (11, 11))

    draw.rounded_rectangle((0, 0, card_size - 18, card_size - 18), radius=40, outline=(255, 255, 255, 255), width=3)
    draw.rounded_rectangle((6, 6, card_size - 24, card_size - 24), radius=34, outline=(255, 255, 255, 220), width=1)
    return canvas


def _load_tournament_logos(logos_dir: Path) -> dict[str, Image.Image]:
    logos: dict[str, Image.Image] = {}
    for key, filename in TOURNAMENT_LOGO_FILES.items():
        path = logos_dir / filename
        if not path.exists():
            continue
        try:
            logo = ImageOps.exif_transpose(Image.open(path)).convert("RGBA")
            logos[key] = logo
        except Exception:
            continue
    return logos


def _logo_tile(logo: Image.Image, size: int) -> Image.Image:
    cache_key = (id(logo), size)
    cached = _LOGO_TILE_CACHE.get(cache_key)
    if cached is not None:
        return cached

    canvas = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    if logo.width > logo.height * 2.4:
        fitted = ImageOps.contain(logo, (size - 3, max(10, size - 8)), method=Image.Resampling.LANCZOS)
    else:
        fitted = ImageOps.contain(logo, (size - 5, size - 5), method=Image.Resampling.LANCZOS)
    x = (size - fitted.width) // 2
    y = (size - fitted.height) // 2
    canvas.alpha_composite(fitted, (x, y))
    _LOGO_TILE_CACHE[cache_key] = canvas
    return canvas


def _draw_slam_logo_cluster(
    frame: Image.Image,
    draw: ImageDraw.ImageDraw,
    center: tuple[int, int],
    slams: list[str],
    logos: dict[str, Image.Image],
) -> None:
    slams = slams[:3]
    if not slams:
        draw.ellipse((center[0] - 5, center[1] - 5, center[0] + 5, center[1] + 5), fill=(15, 20, 32, 42))
        return

    if len(slams) == 1:
        positions = [center]
        icon_size = 62
    elif len(slams) == 2:
        positions = [(center[0] - 16, center[1]), (center[0] + 16, center[1])]
        icon_size = 40
    else:
        positions = [(center[0], center[1] - 15), (center[0] - 17, center[1] + 13), (center[0] + 17, center[1] + 13)]
        icon_size = 33

    for slam, position in zip(slams, positions):
        logo = logos.get(slam)
        if logo is None:
            fallback_font = _load_font(12, bold=True)
            _draw_text_center(draw, position, slam, fallback_font, "#111827", "#FFFFFF", stroke_width=1)
            continue
        tile = _logo_tile(logo, icon_size)
        frame.alpha_composite(tile, (position[0] - icon_size // 2, position[1] - icon_size // 2))


def _draw_text_center(
    draw: ImageDraw.ImageDraw,
    center: tuple[int, int],
    text: str,
    font: ImageFont.ImageFont,
    fill: str,
    stroke_fill: str,
    stroke_width: int = 3,
) -> None:
    draw.text(center, text, font=font, fill=fill, anchor="mm", stroke_width=stroke_width, stroke_fill=stroke_fill)


def _draw_text_left(
    draw: ImageDraw.ImageDraw,
    left_center: tuple[int, int],
    text: str,
    font: ImageFont.ImageFont,
    fill: str,
    stroke_fill: str,
    stroke_width: int = 3,
) -> None:
    bbox = draw.textbbox((0, 0), text, font=font, stroke_width=stroke_width)
    text_h = bbox[3] - bbox[1]
    draw.text(
        (left_center[0], int(left_center[1] - text_h / 2)),
        text,
        font=font,
        fill=fill,
        stroke_width=stroke_width,
        stroke_fill=stroke_fill,
    )


def _draw_grand_slam_mark(
    draw: ImageDraw.ImageDraw,
    center: tuple[int, int],
    scale: int,
    accent: tuple[int, int, int],
) -> None:
    cx, cy = center
    gold = (255, 210, 72, 255)
    dark_gold = (190, 129, 20, 255)
    shine = (255, 248, 184, 255)
    shadow = (17, 20, 32, 90)
    rim = (*accent, 210)

    draw.ellipse((cx - scale - 1, cy + scale - 4, cx + scale + 1, cy + scale + 3), fill=shadow)
    draw.rounded_rectangle(
        (cx - scale // 2, cy - scale, cx + scale // 2, cy + scale // 3),
        radius=max(3, scale // 5),
        fill=gold,
        outline=dark_gold,
        width=1,
    )
    draw.arc((cx - scale, cy - scale + 1, cx - scale // 5, cy + scale // 4), 285, 78, fill=dark_gold, width=2)
    draw.arc((cx + scale // 5, cy - scale + 1, cx + scale, cy + scale // 4), 102, 255, fill=dark_gold, width=2)
    draw.rectangle((cx - scale // 5, cy + scale // 3, cx + scale // 5, cy + scale // 2 + 1), fill=dark_gold)
    draw.rounded_rectangle((cx - scale // 2, cy + scale // 2, cx + scale // 2, cy + scale // 2 + 4), radius=2, fill=dark_gold)
    draw.line((cx - scale // 4, cy - scale + 4, cx + scale // 4, cy - scale + 4), fill=shine, width=2)
    draw.arc((cx - scale, cy - scale, cx + scale, cy + scale), 205, 330, fill=rim, width=1)


def _draw_grand_slam_logos(
    draw: ImageDraw.ImageDraw,
    center: tuple[int, int],
    count: int,
    accent: tuple[int, int, int],
) -> None:
    count = max(0, min(3, count))
    if count <= 0:
        empty_font = _load_font(22, bold=True)
        _draw_text_center(draw, center, "-", empty_font, "#0E1422", "#FFFFFF", stroke_width=1)
        return

    if count == 1:
        positions = [center]
        scale = 15
    elif count == 2:
        positions = [(center[0] - 11, center[1]), (center[0] + 11, center[1])]
        scale = 11
    else:
        positions = [(center[0], center[1] - 10), (center[0] - 12, center[1] + 9), (center[0] + 12, center[1] + 9)]
        scale = 10

    for position in positions:
        _draw_grand_slam_mark(draw, position, scale, accent)


def _visible_circle_fraction(circle_x: int, bar_right: int, radius: int) -> float:
    visible_width = circle_x + radius - bar_right
    return max(0.0, min(1.0, visible_width / max(1, radius * 2)))


def _draw_clipped_gain_circle(
    frame: Image.Image,
    circle_x: int,
    row_center: int,
    bar_right: int,
    radius: int,
    fill: tuple[int, int, int, int],
    outline: tuple[int, int, int, int],
    text: str,
    font: ImageFont.ImageFont,
) -> None:
    visible_fraction = _visible_circle_fraction(circle_x, bar_right, radius)
    if visible_fraction <= 0.0:
        return

    circle_layer = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    circle_draw = ImageDraw.Draw(circle_layer, "RGBA")
    circle_bbox = (
        circle_x - radius,
        int(row_center - radius),
        circle_x + radius,
        int(row_center + radius),
    )
    circle_draw.ellipse(circle_bbox, fill=fill, outline=outline, width=3)

    text_alpha = int(round(255 * _smoothstep(min(1.0, visible_fraction * 1.45))))
    if text_alpha > 0:
        _draw_text_center(
            circle_draw,
            (circle_x, row_center + 1),
            text,
            font,
            (14, 20, 34, text_alpha),
            (255, 255, 255, text_alpha),
            stroke_width=1,
        )

    clip_mask = Image.new("L", (WIDTH, HEIGHT), 0)
    mask_draw = ImageDraw.Draw(clip_mask)
    mask_draw.rectangle((bar_right + 1, 0, WIDTH, HEIGHT), fill=255)
    circle_layer.putalpha(ImageChops.multiply(circle_layer.getchannel("A"), clip_mask))
    frame.alpha_composite(circle_layer)


def _circle_visible_radius(circle_x: int, bar_right: int, radius: int) -> int | None:
    if _visible_circle_fraction(circle_x, bar_right, radius) <= 0.0:
        return None
    return radius


def _render_scene(
    age_value: float,
    background: Image.Image,
    player_cards: dict[str, Image.Image],
    title_font: ImageFont.ImageFont,
    subtitle_font: ImageFont.ImageFont,
    age_font: ImageFont.ImageFont,
    age_active_font: ImageFont.ImageFont,
    value_font: ImageFont.ImageFont,
    footer_font: ImageFont.ImageFont,
) -> Image.Image:
    frame = background.copy()
    draw = ImageDraw.Draw(frame, "RGBA")

    title_center = (WIDTH // 2, 96 + BLOCK_SHIFT)
    _draw_text_center(draw, title_center, TITLE, title_font, "#F7FBFF", "#172033", stroke_width=4)

    pill_w = 334
    pill_h = 60
    pill_rect = (WIDTH // 2 - pill_w // 2, 155 + BLOCK_SHIFT, WIDTH // 2 + pill_w // 2, 155 + BLOCK_SHIFT + pill_h)
    draw.rounded_rectangle(pill_rect, radius=28, fill=(170, 102, 255, 215), outline=(255, 255, 255, 48), width=1)
    _draw_text_center(
        draw,
        (WIDTH // 2, pill_rect[1] + pill_h // 2 + 1),
        SUBTITLE,
        subtitle_font,
        "#FBFCFF",
        "#6E2FD6",
        stroke_width=2,
    )

    chart_left = CHART_LEFT
    chart_right = 1038
    grid_top = 300 + BLOCK_SHIFT
    grid_bottom = 1420 + BLOCK_SHIFT
    tick_origin_x = chart_left + 205
    age_step_px = 332
    fade_px = 760.0
    circle_radius = 34

    draw.line((chart_left, grid_top - 6, chart_left, grid_bottom), fill=(255, 255, 255, 255), width=4)

    current_age = max(AGE_MIN, min(AGE_MAX, age_value))
    current_tick_age = int(math.floor(current_age))
    visible_age_ticks = _visible_age_labels(current_tick_age)

    for age in visible_age_ticks:
        for part in (1, 2, 3):
            minor_x = tick_origin_x + (age + part / 4.0 - current_age) * age_step_px
            if chart_left < minor_x <= WIDTH + 30:
                draw.line((minor_x, grid_top + 96, minor_x, grid_bottom - 32), fill=(255, 255, 255, 22), width=1)

    for age in visible_age_ticks:
        x = tick_origin_x + (age - current_age) * age_step_px
        if x <= chart_left or x > WIDTH + 90:
            continue
        distance = abs(age - current_age) * age_step_px
        fade = max(0.0, min(1.0, 1.0 - distance / fade_px))
        opacity = int(178 * fade)
        if opacity <= 0:
            continue
        width = 2 if age == current_tick_age else 1
        draw.line((x, grid_top, x, grid_bottom), fill=(255, 255, 255, opacity), width=width)

        if opacity > 12:
            label = f"Age {age}"
            font = age_active_font if age == current_tick_age else age_font
            fill = (247, 251, 255, opacity)
            stroke = (31, 36, 86, opacity)
            label_y = grid_top - 28
            bbox = draw.textbbox((0, 0), label, font=font)
            label_w = bbox[2] - bbox[0]
            label_h = bbox[3] - bbox[1]
            label_x = int(x - label_w / 2)
            draw.text(
                (label_x, label_y - label_h / 2),
                label,
                font=font,
                fill=fill,
                stroke_width=2,
                stroke_fill=stroke,
            )

    row_centers = [
        520 + BLOCK_SHIFT + ROW_CENTER_OFFSET,
        820 + BLOCK_SHIFT + ROW_CENTER_OFFSET,
        1120 + BLOCK_SHIFT + ROW_CENTER_OFFSET,
    ]
    low_age = int(math.floor(current_age))
    rank_map = _interpolated_rank_map(current_age)
    previous_rank_map = _rank_map_at_age(low_age)

    rows: list[tuple[float, Player, float, int]] = []
    for player in PLAYERS:
        count = _count_for_age(player, current_age)
        gain = _gain_for_segment(player, low_age)
        rows.append((rank_map[player.name], player, count, gain))
    rows.sort(key=lambda item: (item[0], PLAYER_TIE_ORDER[item[1].name]))

    min_bar_w = 70
    max_bar_w = 430
    bar_area_left = chart_left + 8
    bar_area_right = chart_right - 48
    _ = bar_area_right  # keep the intended spacing explicit.
    row_gap = row_centers[1] - row_centers[0]
    render_rows = sorted(
        rows,
        key=lambda item: (
            previous_rank_map[item[1].name] - item[0],
            item[0],
            PLAYER_TIE_ORDER[item[1].name],
        ),
    )

    # Keep the age connector lines behind every foreground element so they
    # never cut through a bar value or a circle value during rank crossings.
    for rank_position, _player, count, _gain in render_rows:
        row_center = int(round(row_centers[0] + rank_position * row_gap))
        bar_w = int(min_bar_w + (count / VALUE_SCALE_MAX) * (max_bar_w - min_bar_w))
        bar_w = max(min_bar_w, min(max_bar_w, bar_w))
        bar_right = bar_area_left + bar_w
        line_start = bar_right + 8
        if line_start < WIDTH:
            draw.line((line_start, row_center, WIDTH, row_center), fill=(255, 255, 255, 58), width=2)

    for rank_position, player, count, gain in render_rows:
        row_center = int(round(row_centers[0] + rank_position * row_gap))
        card = player_cards[player.name]
        card_x = CARD_X
        card_y = int(row_center - card.height / 2)
        frame.alpha_composite(card, (card_x, card_y))

        bar_h = 176
        bar_w = int(min_bar_w + (count / VALUE_SCALE_MAX) * (max_bar_w - min_bar_w))
        bar_w = max(min_bar_w, min(max_bar_w, bar_w))
        bar_top = int(row_center - bar_h / 2)
        bar_bottom = bar_top + bar_h
        bar_color = player.color
        text_color = _text_on(bar_color)
        shadow_rect = (bar_area_left + 8, bar_top + 8, bar_area_left + bar_w + 8, bar_bottom + 8)
        draw.rounded_rectangle(shadow_rect, radius=18, fill=(0, 0, 0, 72))
        draw.rounded_rectangle(
            (bar_area_left, bar_top, bar_area_left + bar_w, bar_bottom),
            radius=18,
            fill=bar_color,
            outline=_mix_rgb(bar_color, (255, 255, 255), 0.18),
            width=2,
        )
        draw.rounded_rectangle(
            (bar_area_left + 8, bar_top + 8, bar_area_left + max(70, int(bar_w * 0.58)), min(bar_bottom - 8, bar_top + 24)),
            radius=10,
            fill=(255, 255, 255, 44),
        )
        draw.line(
            (bar_area_left + 18, bar_bottom - 8, bar_area_left + max(32, int(bar_w * 0.8)), bar_bottom - 8),
            fill=(*_darken(bar_color, 0.18), 84),
            width=3,
        )

        value_text = str(int(round(count)))
        value_x = bar_area_left + min(max(int(bar_w * 0.42), 42), max(42, bar_w - 34))
        _draw_text_center(
            draw,
            (value_x, row_center + 1),
            value_text,
            value_font,
            text_color,
            "#10131E" if text_color == "#F7F8FC" else "#FFFFFF",
            stroke_width=2,
        )

        for gain_age in visible_age_ticks:
            circle_x = int(round(tick_origin_x + (gain_age - current_age) * age_step_px))
            bar_right = bar_area_left + bar_w
            display_radius = _circle_visible_radius(circle_x, bar_right, circle_radius)
            if display_radius is None or circle_x > WIDTH + circle_radius:
                continue

            gain_text = str(int(round(_gain_for_segment(player, gain_age))))
            circle_fill = (249, 249, 252, 255)
            circle_outline_base = _mix_rgb(bar_color, (255, 255, 255), 0.15)
            circle_outline = (*circle_outline_base, 255)
            gain_font_fit = _fit_font_size(draw, gain_text, display_radius * 2 - 14, 28, 18, bold=True)
            _draw_clipped_gain_circle(
                frame,
                circle_x,
                row_center,
                bar_right,
                display_radius,
                circle_fill,
                circle_outline,
                gain_text,
                gain_font_fit,
            )

    return frame


def render_video(
    output_path: Path,
    photos_dir: Path,
    audio_path: Path,
    duration: float,
    fps: int,
    preview_image: Path | None = None,
    preview_age: float = 29.0,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    photos_dir.mkdir(parents=True, exist_ok=True)

    background = _make_background()
    player_cards = {
        player.name: _build_player_card(player, photos_dir)
        for player in PLAYERS
    }

    title_font = _load_font(50, bold=True)
    subtitle_font = _load_font(26, bold=True)
    age_font = _load_font(34, bold=True)
    age_active_font = _load_font(38, bold=True)
    value_font = _load_font(64, bold=True)
    footer_font = _load_font(22, bold=True)

    def make_frame(t: float) -> np.ndarray:
        if t <= INTRO_HOLD:
            age_value = AGE_MIN
        elif t >= duration - OUTRO_HOLD:
            age_value = AGE_MAX
        else:
            progress = (t - INTRO_HOLD) / max(1e-6, duration - INTRO_HOLD - OUTRO_HOLD)
            age_value = AGE_MIN + (AGE_MAX - AGE_MIN) * max(0.0, min(1.0, progress))
        frame = _render_scene(
            age_value,
            background,
            player_cards,
            title_font,
            subtitle_font,
            age_font,
            age_active_font,
            value_font,
            footer_font,
        )
        return np.array(frame.convert("RGB"))

    if preview_image is not None:
        preview_image.parent.mkdir(parents=True, exist_ok=True)
        preview_frame = _render_scene(
            max(AGE_MIN, min(AGE_MAX, preview_age)),
            background,
            player_cards,
            title_font,
            subtitle_font,
            age_font,
            age_active_font,
            value_font,
            footer_font,
        )
        preview_frame.save(preview_image)

    clip = VideoClip(make_frame, duration=duration)
    audio_clip, keep_alive = build_audio_track(audio_path, duration)
    clip = clip.with_audio(audio_clip)
    clip.write_videofile(str(output_path), fps=fps, codec="libx264", audio_codec="aac")
    clip.close()
    audio_clip.close()
    for item in keep_alive:
        item.close()
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a tennis Grand Slam age race Shorts video.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--photos-dir", type=Path, default=DEFAULT_PHOTOS_DIR)
    parser.add_argument("--audio", type=Path, default=DEFAULT_AUDIO)
    parser.add_argument("--duration", type=float, default=TOTAL_DURATION)
    parser.add_argument("--fps", type=int, default=FPS)
    parser.add_argument("--preview-image", type=Path, default=DEFAULT_PREVIEW)
    parser.add_argument("--preview-age", type=float, default=29.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = render_video(
        output_path=args.output,
        photos_dir=args.photos_dir,
        audio_path=args.audio,
        duration=args.duration,
        fps=args.fps,
        preview_image=args.preview_image,
        preview_age=args.preview_age,
    )
    print(f"[video_generator] Tennis age race generated -> {output}")


if __name__ == "__main__":
    main()
