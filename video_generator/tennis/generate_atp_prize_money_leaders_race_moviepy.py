from __future__ import annotations

import argparse
import csv
import math
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from moviepy import AudioFileClip, CompositeAudioClip, VideoClip
from moviepy.audio.fx import AudioFadeIn, AudioFadeOut
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_INPUT = PROJECT_ROOT / "data" / "processed" / "tennis" / "atp_prize_money_leaders_current.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "tennis" / "atp_prize_money_leaders_current.mp4"
DEFAULT_PHOTOS_DIR = PROJECT_ROOT / "data" / "raw" / "player_photos"
DEFAULT_FLAGS_DIR = PROJECT_ROOT / "data" / "raw" / "flags"
DEFAULT_AUDIO = PROJECT_ROOT / "data" / "raw" / "audio" / "audio.mp3"

WIDTH = 1920
HEIGHT = 1080
FPS = 60
TOTAL_DURATION = 40.0
TOP_N = 10

BAR_WIDTH = 272
BAR_DEPTH_X = 72
BAR_DEPTH_Y = 24
BAR_GAP = 388
LEFT_MARGIN = 560
GROUND_Y = 1086
MIN_BAR_HEIGHT = 430
MAX_BAR_HEIGHT = 770
PORTRAIT_SIZE = 178
FLAG_SIZE = (110, 73)
CAMERA_EASE_POWER = 1.12

TITLE = "Globe"
SUBTITLE = "Prize money leaders"


@dataclass(frozen=True)
class PrizeMoneyEntry:
    ranking_date: str
    tour: str
    rank: int
    player_name: str
    country_code: str
    career_usd: int
    ytd_usd: int
    singles_usd: int
    doubles_usd: int


@dataclass(frozen=True)
class BarLayout:
    entry: PrizeMoneyEntry
    x: int
    bar_height: int
    phase: float


def _slugify(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", "_", normalized.lower()).strip("_")


def _hex_to_rgb(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    return tuple(int(value[i : i + 2], 16) for i in (0, 2, 4))


def _mix_rgb(color: str, target: tuple[int, int, int], amount: float) -> tuple[int, int, int]:
    r, g, b = _hex_to_rgb(color)
    amount = min(max(amount, 0.0), 1.0)
    return (
        int(r + (target[0] - r) * amount),
        int(g + (target[1] - g) * amount),
        int(b + (target[2] - b) * amount),
    )


def _lighten(color: str, amount: float) -> tuple[int, int, int]:
    return _mix_rgb(color, (255, 255, 255), amount)


def _darken(color: str, amount: float) -> tuple[int, int, int]:
    return _mix_rgb(color, (0, 0, 0), amount)


def _text_on(color: str) -> str:
    r, g, b = _hex_to_rgb(color)
    luminance = (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255.0
    return "#132033" if luminance > 0.63 else "#f5f8fd"


def _load_font(size: int, bold: bool = False, serif: bool = False) -> ImageFont.ImageFont:
    if serif:
        candidates = [
            "C:/Windows/Fonts/georgiab.ttf" if bold else "C:/Windows/Fonts/georgia.ttf",
            "C:/Windows/Fonts/GeorgiaPro-Bold.ttf" if bold else "C:/Windows/Fonts/GeorgiaPro-Regular.ttf",
            "C:/Windows/Fonts/timesbd.ttf" if bold else "C:/Windows/Fonts/times.ttf",
        ]
    else:
        candidates = [
            "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
            "C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf",
            "C:/Windows/Fonts/arialnb.ttf" if bold else "C:/Windows/Fonts/arialn.ttf",
        ]
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def _fit_font(
    draw: ImageDraw.ImageDraw,
    text: str,
    max_width: int,
    start_size: int,
    min_size: int,
    *,
    bold: bool = True,
    serif: bool = False,
) -> ImageFont.ImageFont:
    size = start_size
    while size >= min_size:
        font = _load_font(size, bold=bold, serif=serif)
        if draw.textbbox((0, 0), text, font=font)[2] <= max_width:
            return font
        size -= 1
    return _load_font(min_size, bold=bold, serif=serif)


def _truncate(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int) -> str:
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


def _format_money(value: int) -> str:
    return f"${value / 1_000_000:.1f}M"


def _format_date(iso_date: str) -> str:
    return f"{iso_date[8:10]} {datetime_from_iso(iso_date).strftime('%b %Y')}".upper()


def datetime_from_iso(iso_date: str):
    from datetime import datetime

    return datetime.strptime(iso_date, "%Y-%m-%d")


def _load_entries(input_csv: Path) -> list[PrizeMoneyEntry]:
    entries: list[PrizeMoneyEntry] = []
    with input_csv.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            entries.append(
                PrizeMoneyEntry(
                    ranking_date=row["ranking_date"].strip(),
                    tour=row.get("tour", "ATP").strip(),
                    rank=int(row["rank"]),
                    player_name=row["player_name"].strip(),
                    country_code=row.get("country_code", "").strip().upper(),
                    career_usd=int(float(row["career_usd"])),
                    ytd_usd=int(float(row.get("ytd_usd", "0") or 0)),
                    singles_usd=int(float(row.get("singles_usd", "0") or 0)),
                    doubles_usd=int(float(row.get("doubles_usd", "0") or 0)),
                )
            )
    return entries


def _resolve_photo(player_name: str, photos_dir: Path) -> Path | None:
    slug = _slugify(player_name)
    for ext in (".jpg", ".jpeg", ".png", ".webp"):
        candidate = photos_dir / f"{slug}{ext}"
        if candidate.exists():
            return candidate
    return None


def _build_avatar(player_name: str, photos_dir: Path, size: int) -> Image.Image:
    photo_path = _resolve_photo(player_name, photos_dir)
    if photo_path is not None:
        try:
            img = Image.open(photo_path).convert("RGBA")
            img = ImageOps.exif_transpose(img)
            fitted = ImageOps.fit(img, (size, size), method=Image.Resampling.LANCZOS, centering=(0.5, 0.22))
        except Exception:
            fitted = None
    else:
        fitted = None

    avatar = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    mask = Image.new("L", (size, size), 0)
    mask_draw = ImageDraw.Draw(mask)
    mask_draw.ellipse((0, 0, size - 1, size - 1), fill=255)

    if fitted is not None:
        avatar.paste(fitted, (0, 0), mask)
    else:
        draw = ImageDraw.Draw(avatar, "RGBA")
        for y in range(size):
            alpha = int(255 * (0.72 + 0.28 * (y / max(1, size - 1))))
            draw.line((0, y, size, y), fill=(20, 44, 77, alpha))
        draw.ellipse((0, 0, size - 1, size - 1), outline=(255, 255, 255, 120), width=4)
        initials = "".join(part[0] for part in player_name.split()[:2]).upper()
        font = _fit_font(draw, initials, size - 24, 50, 28, bold=True, serif=False)
        draw.text((size // 2, size // 2), initials, font=font, fill=(246, 248, 252, 255), anchor="mm")

    bordered = Image.new("RGBA", (size + 8, size + 8), (0, 0, 0, 0))
    border_draw = ImageDraw.Draw(bordered, "RGBA")
    border_draw.ellipse((0, 0, size + 7, size + 7), fill=(255, 255, 255, 235))
    bordered.paste(avatar, (4, 4), mask)
    return bordered


def _load_flag(flag_dir: Path, country_code: str, size: tuple[int, int]) -> Image.Image | None:
    if not country_code:
        return None
    flag_path = flag_dir / f"{country_code.lower()}.png"
    if not flag_path.exists():
        return None
    try:
        flag = Image.open(flag_path).convert("RGBA")
        flag = ImageOps.contain(flag, size, method=Image.Resampling.LANCZOS)
        return flag
    except Exception:
        return None


def _wave_flag(flag: Image.Image) -> Image.Image:
    width, height = flag.size
    amplitude = max(2, width // 28)
    waved = Image.new("RGBA", (width + amplitude * 2, height + 4), (0, 0, 0, 0))
    for y in range(height):
        phase = (y / max(1, height - 1)) * math.tau * 1.18 + 0.75
        offset = amplitude + int(round(math.sin(phase) * amplitude))
        waved.alpha_composite(flag.crop((0, y, width, y + 1)), (offset, y + 2))
    return waved


def _unit_phase(value: str) -> float:
    total = 0
    for idx, ch in enumerate(value):
        total += (idx + 1) * ord(ch)
    return (total % 997) / 997.0


def _smoothstep(value: float) -> float:
    value = min(max(value, 0.0), 1.0)
    return value * value * (3.0 - 2.0 * value)


def _prepare_layouts(entries: list[PrizeMoneyEntry]) -> list[BarLayout]:
    max_value = max(entry.career_usd for entry in entries)
    layouts: list[BarLayout] = []
    for idx, entry in enumerate(entries):
        bar_height = int(MIN_BAR_HEIGHT + ((entry.career_usd / max_value) ** 0.78) * (MAX_BAR_HEIGHT - MIN_BAR_HEIGHT))
        layouts.append(
            BarLayout(
                entry=entry,
                x=LEFT_MARGIN + idx * (BAR_WIDTH + BAR_GAP),
                bar_height=bar_height,
                phase=_unit_phase(entry.player_name),
            )
        )
    return layouts


def _gradient_overlay(
    size: tuple[int, int],
    rgb: tuple[int, int, int],
    *,
    axis: str,
    alpha_start: int,
    alpha_end: int,
) -> Image.Image:
    width, height = max(1, size[0]), max(1, size[1])
    if axis == "horizontal":
        ramp = np.linspace(0.0, 1.0, width, dtype=np.float32)[None, :]
        alpha = np.repeat(ramp, height, axis=0)
    else:
        ramp = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
        alpha = np.repeat(ramp, width, axis=1)
    alpha = np.clip(alpha_start + (alpha_end - alpha_start) * alpha, 0, 255).astype(np.uint8)
    data = np.zeros((height, width, 4), dtype=np.uint8)
    data[..., 0] = rgb[0]
    data[..., 1] = rgb[1]
    data[..., 2] = rgb[2]
    data[..., 3] = alpha
    return Image.fromarray(data, mode="RGBA")


def _make_background(width: int, height: int) -> Image.Image:
    xx = np.linspace(0, 1, width, dtype=np.float32)
    yy = np.linspace(0, 1, height, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(xx, yy)

    sky_top = np.array([86, 116, 154], dtype=np.float32)
    sky_bottom = np.array([126, 166, 205], dtype=np.float32)
    glow = np.exp(-(((grid_x - 0.56) / 0.26) ** 2 + ((grid_y - 0.22) / 0.18) ** 2))
    haze = np.exp(-(((grid_x - 0.32) / 0.36) ** 2 + ((grid_y - 0.46) / 0.24) ** 2))

    mix = np.clip(grid_y * 1.02, 0, 1)
    rgb = sky_top[None, None, :] * (1.0 - mix[..., None]) + sky_bottom[None, None, :] * mix[..., None]
    rgb += glow[..., None] * np.array([8, 9, 12], dtype=np.float32)
    rgb += haze[..., None] * np.array([5, 7, 9], dtype=np.float32)
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)

    frame = Image.fromarray(rgb, mode="RGB").convert("RGBA")
    overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")

    skyline_base = 510
    cursor = 0
    pattern = [70, 118, 90, 142, 96, 168, 80, 130, 104, 154, 88, 122, 76, 176, 94, 140]
    idx = 0
    while cursor < width:
        block = pattern[idx % len(pattern)]
        h = 44 + ((idx * 37) % 120)
        if idx % 5 == 0:
            h += 34
        if idx % 7 == 0:
            h += 52
        x0 = cursor
        x1 = min(width, cursor + block)
        draw.rectangle((x0, skyline_base - h, x1, skyline_base), fill=(64, 64, 68, 255))
        if idx % 3 == 0:
            draw.rectangle((x0 + 12, skyline_base - h - 38, x0 + 28, skyline_base - h), fill=(61, 61, 66, 255))
        cursor += block - 10
        idx += 1

    draw.rectangle((0, skyline_base, width, height), fill=(68, 70, 73, 255))
    ground_glow = np.exp(-(((grid_x - 0.5) / 0.72) ** 2 + ((grid_y - 0.9) / 0.12) ** 2))
    floor = np.zeros((height, width, 4), dtype=np.uint8)
    floor[..., 3] = np.clip(ground_glow * 28, 0, 28).astype(np.uint8)
    overlay.alpha_composite(Image.fromarray(floor, mode="RGBA"))
    overlay = overlay.filter(ImageFilter.GaussianBlur(radius=0.8))
    frame.alpha_composite(overlay)

    haze_overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    haze_draw = ImageDraw.Draw(haze_overlay, "RGBA")
    haze_draw.rectangle((0, 620, width, height), fill=(39, 42, 48, 34))
    haze_draw.rectangle((0, 760, width, height), fill=(27, 30, 36, 42))
    haze_overlay = haze_overlay.filter(ImageFilter.GaussianBlur(radius=12))
    frame.alpha_composite(haze_overlay)
    return frame


def _draw_text_with_shadow(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int],
    shadow_fill: tuple[int, int, int] = (0, 0, 0),
    anchor: str = "la",
    stroke_width: int = 0,
    stroke_fill: tuple[int, int, int] | None = None,
) -> None:
    x, y = xy
    draw.text((x + 2, y + 3), text, font=font, fill=(*shadow_fill, 120), anchor=anchor)
    draw.text(
        (x, y),
        text,
        font=font,
        fill=(*fill, 255),
        anchor=anchor,
        stroke_width=stroke_width,
        stroke_fill=stroke_fill,
    )


def _draw_bar(world: Image.Image, entry: PrizeMoneyEntry, x: int, bar_height: int, photos_dir: Path, flags_dir: Path) -> None:
    draw = ImageDraw.Draw(world, "RGBA")

    front_left = x
    front_right = x + BAR_WIDTH
    front_top = GROUND_Y - bar_height
    front_bottom = GROUND_Y
    side_poly = [
        (front_right, front_top),
        (front_right + BAR_DEPTH_X, front_top - BAR_DEPTH_Y),
        (front_right + BAR_DEPTH_X, front_bottom - BAR_DEPTH_Y),
        (front_right, front_bottom),
    ]
    top_poly = [
        (front_left, front_top),
        (front_right, front_top),
        (front_right + BAR_DEPTH_X, front_top - BAR_DEPTH_Y),
        (front_left + BAR_DEPTH_X, front_top - BAR_DEPTH_Y),
    ]

    shadow = Image.new("RGBA", world.size, (0, 0, 0, 0))
    shadow_draw = ImageDraw.Draw(shadow, "RGBA")
    shadow_draw.ellipse(
        (front_left - 70, front_bottom - 12, front_right + BAR_DEPTH_X + 132, front_bottom + 54),
        fill=(0, 0, 0, 108),
    )
    shadow_draw.ellipse(
        (front_left + 18, front_bottom - 2, front_right + BAR_DEPTH_X + 34, front_bottom + 18),
        fill=(0, 0, 0, 78),
    )
    shadow = shadow.filter(ImageFilter.GaussianBlur(radius=18))
    world.alpha_composite(shadow)

    front_color = "#d8c9aa"
    side_color = "#c0af8d"
    top_color = "#e7dcc7"
    label_color = "#432a37"
    value_color = "#7c0c0a"
    front_fill = _hex_to_rgb(front_color)
    side_fill = _hex_to_rgb(side_color)
    top_fill = _hex_to_rgb(top_color)
    value_fill = _hex_to_rgb(value_color)

    draw.polygon(side_poly, fill=side_fill)
    draw.polygon(top_poly, fill=top_fill)
    draw.rectangle((front_left, front_top, front_right, front_bottom), fill=front_fill)
    draw.line((front_left, front_top, front_right, front_top), fill=(255, 255, 255, 95), width=2)
    draw.line((front_left, front_top, front_left, front_bottom), fill=(255, 255, 255, 36), width=2)
    draw.line((front_right, front_top, front_right, front_bottom), fill=(0, 0, 0, 26), width=2)
    draw.line((front_right + BAR_DEPTH_X, front_top - BAR_DEPTH_Y, front_right + BAR_DEPTH_X, front_bottom - BAR_DEPTH_Y), fill=(0, 0, 0, 44), width=2)
    draw.line((front_left + BAR_DEPTH_X, front_top - BAR_DEPTH_Y, front_right + BAR_DEPTH_X, front_top - BAR_DEPTH_Y), fill=(0, 0, 0, 34), width=2)
    draw.line((front_left, front_bottom, front_right, front_bottom), fill=(108, 85, 60, 48), width=2)

    gradient = Image.new("RGBA", (BAR_WIDTH, bar_height), (0, 0, 0, 0))
    gradient_draw = ImageDraw.Draw(gradient, "RGBA")
    gradient_draw.rectangle((0, 0, BAR_WIDTH, max(1, int(bar_height * 0.22))), fill=(255, 255, 255, 28))
    gradient_draw.rectangle((0, int(bar_height * 0.55), BAR_WIDTH, bar_height), fill=(0, 0, 0, 10))
    world.alpha_composite(gradient, (front_left, front_top))

    world.alpha_composite(_gradient_overlay((BAR_WIDTH, bar_height), (255, 255, 255), axis="vertical", alpha_start=34, alpha_end=0), (front_left, front_top))
    world.alpha_composite(_gradient_overlay((BAR_WIDTH, bar_height), (0, 0, 0), axis="vertical", alpha_start=0, alpha_end=18), (front_left, front_top))
    world.alpha_composite(_gradient_overlay((BAR_WIDTH, bar_height), (255, 255, 255), axis="horizontal", alpha_start=18, alpha_end=0), (front_left, front_top))
    world.alpha_composite(_gradient_overlay((BAR_WIDTH, bar_height), (0, 0, 0), axis="horizontal", alpha_start=0, alpha_end=24), (front_left, front_top))

    label_font = _fit_font(draw, entry.player_name, BAR_WIDTH - 42, 38, 22, bold=True, serif=True)
    value_font = _fit_font(draw, _format_money(entry.career_usd), BAR_WIDTH - 30, 74, 40, bold=True, serif=True)
    hud_font = _load_font(22, bold=True, serif=False)

    label_rect = (front_left + 18, front_top + 80, front_right - 18, front_top + 146)
    draw.rounded_rectangle(label_rect, radius=12, fill=_hex_to_rgb(label_color))
    draw.line((label_rect[0] + 6, label_rect[1] + 5, label_rect[2] - 6, label_rect[1] + 5), fill=(255, 255, 255, 32), width=2)
    draw.line((label_rect[0] + 6, label_rect[3] - 4, label_rect[2] - 6, label_rect[3] - 4), fill=(0, 0, 0, 36), width=2)
    draw.text(
        ((label_rect[0] + label_rect[2]) // 2, (label_rect[1] + label_rect[3]) // 2 - 1),
        entry.player_name,
        font=label_font,
        fill=(248, 247, 243, 255),
        anchor="mm",
        stroke_width=1,
        stroke_fill=(19, 14, 18, 160),
    )

    money_text = _format_money(entry.career_usd)
    draw.text(
        ((front_left + front_right) // 2, front_top + 230),
        money_text,
        font=value_font,
        fill=(*value_fill, 255),
        anchor="mm",
        stroke_width=2,
        stroke_fill=(116, 8, 8, 130),
    )

    country_code = entry.country_code.upper()
    avatar = _build_avatar(entry.player_name, photos_dir, PORTRAIT_SIZE)
    portrait_w = avatar.width
    portrait_h = avatar.height
    portrait_x = front_left + (BAR_WIDTH - portrait_w) // 2 + BAR_DEPTH_X // 2
    portrait_y = front_top - portrait_h + 8

    portrait_shadow = Image.new("RGBA", world.size, (0, 0, 0, 0))
    portrait_shadow_draw = ImageDraw.Draw(portrait_shadow, "RGBA")
    portrait_shadow_draw.ellipse(
        (portrait_x + 8, portrait_y + 14, portrait_x + portrait_w + 10, portrait_y + portrait_h + 18),
        fill=(0, 0, 0, 128),
    )
    portrait_shadow = portrait_shadow.filter(ImageFilter.GaussianBlur(radius=10))
    world.alpha_composite(portrait_shadow)

    draw.ellipse((portrait_x - 4, portrait_y - 4, portrait_x + portrait_w + 3, portrait_y + portrait_h + 3), fill=(255, 255, 255, 234))
    world.alpha_composite(avatar, (portrait_x, portrait_y))

    flag = _load_flag(flags_dir, country_code, FLAG_SIZE)
    flag_paste_x = portrait_x + portrait_w - 12
    flag_paste_y = portrait_y + 44
    pole_start = (flag_paste_x + 8, flag_paste_y + FLAG_SIZE[1] - 3)
    pole_end = (flag_paste_x + 34, flag_paste_y + 8)
    draw.line((pole_start[0], pole_start[1], pole_end[0], pole_end[1]), fill=(72, 54, 32, 180), width=4)
    if flag is not None:
        rotated_flag = _wave_flag(flag).rotate(-10, expand=True, resample=Image.Resampling.BICUBIC)
        shadow_alpha = rotated_flag.getchannel("A").filter(ImageFilter.GaussianBlur(radius=4))
        flag_shadow = Image.new("RGBA", rotated_flag.size, (0, 0, 0, 95))
        flag_shadow.putalpha(shadow_alpha)
        world.alpha_composite(flag_shadow, (flag_paste_x + 13, flag_paste_y + 8))
        world.alpha_composite(rotated_flag, (flag_paste_x + 8, flag_paste_y))
    else:
        fallback = Image.new("RGBA", FLAG_SIZE, (0, 0, 0, 0))
        fallback_draw = ImageDraw.Draw(fallback, "RGBA")
        fallback_draw.rounded_rectangle((0, 0, FLAG_SIZE[0] - 1, FLAG_SIZE[1] - 1), radius=6, fill=(255, 255, 255, 230))
        fallback_draw.text((FLAG_SIZE[0] // 2, FLAG_SIZE[1] // 2), country_code or "?", font=hud_font, fill=(29, 41, 61, 255), anchor="mm")
        world.alpha_composite(fallback, (flag_paste_x + 8, flag_paste_y))


def _draw_world_subscribe(world: Image.Image, layouts: list[BarLayout]) -> None:
    if not layouts:
        return

    anchor = layouts[min(5, len(layouts) - 1)]
    draw = ImageDraw.Draw(world, "RGBA")
    font = _load_font(28, bold=True, serif=False)
    x = anchor.x + BAR_WIDTH + 110
    y = GROUND_Y - anchor.bar_height + 245
    draw.text((x + 2, y + 3), "Subscribe", font=font, fill=(0, 0, 0, 120), anchor="mm")
    draw.text((x, y), "Subscribe", font=font, fill=(255, 211, 48, 255), anchor="mm")


def _build_world(layouts: list[BarLayout], photos_dir: Path, flags_dir: Path) -> Image.Image:
    if not layouts:
        raise RuntimeError("Impossible de construire la scene sans barres.")

    world_width = max(
        WIDTH + 480,
        layouts[-1].x + BAR_WIDTH + BAR_DEPTH_X + PORTRAIT_SIZE + FLAG_SIZE[0] + 220,
    )

    world = _make_background(world_width, HEIGHT)

    floor_overlay = Image.new("RGBA", (world_width, HEIGHT), (0, 0, 0, 0))
    floor_draw = ImageDraw.Draw(floor_overlay, "RGBA")
    floor_draw.rectangle((0, 610, world_width, HEIGHT), fill=(30, 32, 38, 48))
    floor_draw.line((0, HEIGHT - 5, world_width, HEIGHT - 5), fill=(88, 80, 65, 38), width=2)
    floor_overlay = floor_overlay.filter(ImageFilter.GaussianBlur(radius=12))
    world.alpha_composite(floor_overlay)

    for layout in layouts:
        _draw_bar(world, layout.entry, layout.x, layout.bar_height, photos_dir, flags_dir)
    _draw_world_subscribe(world, layouts)
    return world


def _add_motion_overlays(
    frame: Image.Image,
    layouts: list[BarLayout],
    camera_x: int,
    t: float,
) -> None:
    motion = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    motion_draw = ImageDraw.Draw(motion, "RGBA")

    sweep_pos = (t * 0.025) % 1.0
    sweep_x = int(-260 + sweep_pos * (WIDTH + 520))
    motion_draw.polygon(
        [
            (sweep_x - 150, 250),
            (sweep_x + 10, 232),
            (sweep_x + 190, 900),
            (sweep_x + 30, 928),
        ],
        fill=(255, 255, 255, 4),
    )

    motion = motion.filter(ImageFilter.GaussianBlur(radius=28))
    frame.alpha_composite(motion)

    accents = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    accents_draw = ImageDraw.Draw(accents, "RGBA")
    for layout in layouts:
        screen_left = layout.x - camera_x
        if screen_left > WIDTH + 220 or screen_left + BAR_WIDTH + BAR_DEPTH_X < -220:
            continue

        front_top = GROUND_Y - layout.bar_height
        pulse = 0.35 + 0.25 * math.sin(math.tau * (t * 0.32 + layout.phase))
        bob = math.sin(math.tau * (t * 0.42 + layout.phase)) * 0.9

        avatar_size = PORTRAIT_SIZE + 8
        portrait_x = screen_left + (BAR_WIDTH - avatar_size) // 2 + BAR_DEPTH_X // 2
        portrait_y = front_top - avatar_size + 8 + int(round(bob))
        portrait_box = (portrait_x - 8, portrait_y - 8, portrait_x + avatar_size + 8, portrait_y + avatar_size + 8)
        accents_draw.ellipse(portrait_box, outline=(255, 255, 255, int(16 + 18 * pulse)), width=2)
        accents_draw.ellipse(
            (portrait_x + 10, portrait_y + 16, portrait_x + avatar_size + 14, portrait_y + avatar_size + 20),
            fill=(0, 0, 0, int(18 + 20 * pulse)),
        )

        flag_x = portrait_x + avatar_size + 2
        flag_y = portrait_y + 28 + int(round(0.8 * math.sin(math.tau * (t * 0.62 + layout.phase))))
        accents_draw.rounded_rectangle(
            (flag_x + 6, flag_y + 6, flag_x + FLAG_SIZE[0] + 13, flag_y + FLAG_SIZE[1] + 9),
            radius=7,
            outline=(255, 255, 255, int(14 + 12 * pulse)),
            width=1,
        )

        label_x0 = screen_left + 18
        label_y0 = front_top + 70
        label_y = label_y0 + 8 + int(round(1.5 * math.sin(math.tau * (t * 0.36 + layout.phase))))
        accents_draw.line(
            (label_x0 + 12, label_y, label_x0 + BAR_WIDTH - 12, label_y),
            fill=(255, 255, 255, int(8 + 8 * pulse)),
            width=2,
        )

        shimmer = (t * 0.08 + layout.phase) % 1.0
        sheen_y = front_top + int(0.15 * layout.bar_height + shimmer * 0.58 * layout.bar_height)
        accents_draw.line(
            (screen_left + 28, sheen_y - 10, screen_left + BAR_WIDTH - 30, sheen_y + 8),
            fill=(255, 255, 255, int(6 + 6 * pulse)),
            width=6,
        )
        accents_draw.line(
            (screen_left + 42, sheen_y + 8, screen_left + BAR_WIDTH - 38, sheen_y + 18),
            fill=(38, 28, 24, int(8 + 8 * pulse)),
            width=4,
        )

    accents = accents.filter(ImageFilter.GaussianBlur(radius=0.8))
    frame.alpha_composite(accents)


def build_audio_track(audio_path: Path, duration: float):
    base = AudioFileClip(str(audio_path))
    if base.duration >= duration:
        return base.subclipped(0, duration).with_effects([AudioFadeOut(6.0)]), [base]

    clips = []
    keep_alive = [base]
    step = max(0.1, base.duration - 4.0)
    loops = int(math.ceil(max(0.0, duration - 4.0) / step))
    for index in range(loops):
        segment = (
            base.with_start(index * step)
            .with_effects([AudioFadeIn(4.0), AudioFadeOut(4.0)])
        )
        clips.append(segment)
    mixed = CompositeAudioClip(clips).with_duration(duration).with_effects([AudioFadeOut(6.0)])
    return mixed, keep_alive


def render_video(
    input_csv: Path,
    output_path: Path,
    photos_dir: Path,
    flags_dir: Path,
    audio_path: Path | None,
    duration: float,
    fps: int,
    top_n: int,
) -> Path:
    entries = _load_entries(input_csv)
    if not entries:
        raise RuntimeError(f"No data found in {input_csv}")

    selected = sorted(entries, key=lambda item: (-item.career_usd, item.player_name))[:top_n]
    selected = sorted(selected, key=lambda item: (item.career_usd, item.player_name))
    layouts = _prepare_layouts(selected)
    world = _build_world(layouts, photos_dir, flags_dir)
    max_camera_x = max(0, world.width - WIDTH)

    def make_frame(t: float) -> np.ndarray:
        progress = min(max(t / duration, 0.0), 1.0)
        eased = progress ** CAMERA_EASE_POWER
        camera_drift = 2.0 * math.sin(math.tau * (t * 0.05 + 0.13))
        camera_x = int(round(min(max_camera_x, max(0.0, eased * max_camera_x + camera_drift))))
        frame = world.crop((camera_x, 0, camera_x + WIDTH, HEIGHT)).convert("RGBA")
        _add_motion_overlays(frame, layouts, camera_x, t)
        hud = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
        hud_draw = ImageDraw.Draw(hud, "RGBA")

        watermark_font = _load_font(58, bold=True, serif=False)
        hud_draw.text((62, 45), TITLE, font=watermark_font, fill=(0, 0, 0, 80))
        hud_draw.text((58, 40), TITLE, font=watermark_font, fill=(238, 245, 252, 235))

        frame.alpha_composite(hud)
        return np.array(frame.convert("RGB"))

    clip = VideoClip(make_frame, duration=duration)
    audio_clip = None
    keep_alive: list[AudioFileClip] = []
    if audio_path is not None and audio_path.exists():
        audio_clip, keep_alive = build_audio_track(audio_path, duration)
        clip = clip.with_audio(audio_clip)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    clip.write_videofile(
        str(output_path),
        fps=fps,
        codec="libx264",
        audio_codec="aac" if audio_clip is not None else None,
        ffmpeg_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
    )
    clip.close()
    if audio_clip is not None:
        audio_clip.close()
    for item in keep_alive:
        item.close()
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a landscape ATP prize money leaders race video.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--photos-dir", type=Path, default=DEFAULT_PHOTOS_DIR)
    parser.add_argument("--flags-dir", type=Path, default=DEFAULT_FLAGS_DIR)
    parser.add_argument("--audio", type=Path, default=DEFAULT_AUDIO)
    parser.add_argument("--duration", type=float, default=TOTAL_DURATION)
    parser.add_argument("--fps", type=int, default=FPS)
    parser.add_argument("--top-n", type=int, default=TOP_N)
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
        top_n=args.top_n,
    )
    print(f"[video_generator] ATP prize money leaders race generated -> {output}")


if __name__ == "__main__":
    main()
