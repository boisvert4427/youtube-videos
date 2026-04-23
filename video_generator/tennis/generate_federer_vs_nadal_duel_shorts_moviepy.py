from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from moviepy import VideoClip
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageOps

from video_generator.generate_ucl_barchart_race_moviepy import (
    DEFAULT_AUDIO,
    _fit_font_size,
    _load_font,
    build_audio_track,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PHOTOS_DIR = PROJECT_ROOT / "data" / "raw" / "player_photos"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "tennis" / "federer_vs_nadal_duel_shorts_optimized.mp4"

WIDTH = 1080
HEIGHT = 1920
FPS = 60
TOTAL_DURATION = 11.5
MUSIC_VOLUME = 0.44

FEDERER = {
    "name": "FEDERER",
    "full_name": "Roger Federer",
    "photo": "roger_federer.jpg",
    "accent": (116, 198, 255),
    "accent_soft": (170, 224, 255),
    "dark": (9, 28, 52),
}
NADAL = {
    "name": "NADAL",
    "full_name": "Rafael Nadal",
    "photo": "rafael_nadal.jpg",
    "accent": (255, 118, 72),
    "accent_soft": (255, 193, 143),
    "dark": (58, 18, 20),
}

STATS = [
    {
        "label": "GRAND SLAMS",
        "left_value": 20,
        "right_value": 22,
        "note": "Close battle",
        "winner": "right",
        "tag": "1 / 4",
    },
    {
        "label": "HEAD-TO-HEAD",
        "left_value": 16,
        "right_value": 24,
        "note": "Nadal leads",
        "winner": "right",
        "tag": "2 / 4",
    },
    {
        "label": "CLAY TITLES",
        "left_value": 11,
        "right_value": 63,
        "note": "On clay? Over.",
        "winner": "right",
        "tag": "3 / 4",
    },
    {
        "label": "GRASS TITLES",
        "left_value": 19,
        "right_value": 4,
        "note": "Federer answers back",
        "winner": "left",
        "tag": "4 / 4",
    },
]

SCENES = [
    ("flash", 0.00, 0.35),
    ("hook", 0.35, 0.90),
    ("install", 0.90, 1.40),
    ("stat_1", 1.40, 2.20),
    ("stat_2", 2.20, 3.00),
    ("pause_clay", 3.00, 3.80),
    ("stat_3", 3.80, 5.10),
    ("hold", 5.10, 5.80),
    ("twist_grass", 5.80, 6.60),
    ("stat_4", 6.60, 7.80),
    ("final_board", 7.80, 8.50),
    ("climax", 8.50, 9.40),
    ("goat_twist", 9.40, 10.20),
    ("ending", 10.20, TOTAL_DURATION),
]


@dataclass(frozen=True)
class SceneState:
    name: str
    start: float
    end: float
    progress: float
    index: int


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * _clamp(t)


def _ease_out(value: float) -> float:
    value = _clamp(value)
    return 1.0 - (1.0 - value) ** 3


def _ease_in_out(value: float) -> float:
    value = _clamp(value)
    return value * value * (3.0 - 2.0 * value)


def _hex_to_rgb(hex_value: str) -> tuple[int, int, int]:
    hex_value = hex_value.lstrip("#")
    return tuple(int(hex_value[i : i + 2], 16) for i in (0, 2, 4))


def _mix_rgb(color_a: tuple[int, int, int], color_b: tuple[int, int, int], amount: float) -> tuple[int, int, int]:
    amount = _clamp(amount)
    return (
        int(color_a[0] + (color_b[0] - color_a[0]) * amount),
        int(color_a[1] + (color_b[1] - color_a[1]) * amount),
        int(color_a[2] + (color_b[2] - color_a[2]) * amount),
    )


def _scene_for_time(t: float) -> SceneState:
    for index, (name, start, end) in enumerate(SCENES):
        if start <= t < end or (index == len(SCENES) - 1 and t >= start):
            span = max(1e-6, end - start)
            return SceneState(name=name, start=start, end=end, progress=_clamp((t - start) / span), index=index)
    name, start, end = SCENES[-1]
    return SceneState(name=name, start=start, end=end, progress=1.0, index=len(SCENES) - 1)


def _load_portrait(path: Path, initials: str) -> Image.Image:
    if path.exists():
        img = ImageOps.exif_transpose(Image.open(path)).convert("RGBA")
        return img
    placeholder = Image.new("RGBA", (720, 720), (8, 12, 20, 255))
    draw = ImageDraw.Draw(placeholder)
    draw.rounded_rectangle((24, 24, 696, 696), radius=64, fill=(18, 26, 40, 255), outline=(255, 255, 255, 72), width=4)
    draw.text((360, 356), initials, font=_load_font(72, bold=True), fill=(245, 248, 252, 255), anchor="mm")
    return placeholder


def _circle_tile(source: Image.Image, ring_color: tuple[int, int, int], size: int = 186) -> Image.Image:
    portrait = ImageOps.fit(source, (size, size), method=Image.Resampling.LANCZOS, centering=(0.5, 0.24))
    portrait = ImageEnhance.Brightness(portrait).enhance(1.08)
    portrait = ImageEnhance.Contrast(portrait).enhance(1.10)
    mask = Image.new("L", (size, size), 0)
    ImageDraw.Draw(mask).ellipse((0, 0, size - 1, size - 1), fill=255)
    tile = Image.new("RGBA", (size + 28, size + 28), (0, 0, 0, 0))
    td = ImageDraw.Draw(tile, "RGBA")
    td.ellipse((0, 0, size + 27, size + 27), fill=(0, 0, 0, 90))
    td.ellipse((4, 4, size + 23, size + 23), fill=(*ring_color, 255))
    td.ellipse((11, 11, size + 16, size + 16), fill=(12, 18, 30, 255))
    circle = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    circle.paste(portrait, (0, 0), mask)
    tile.alpha_composite(circle, (14, 14))
    return tile


def _draw_glow_text(
    frame: Image.Image,
    position: tuple[int, int],
    text: str,
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int],
    glow: tuple[int, int, int],
    anchor: str = "mm",
    stroke_width: int = 2,
    stroke_fill: tuple[int, int, int] = (0, 0, 0),
) -> None:
    blur_layer = Image.new("RGBA", frame.size, (0, 0, 0, 0))
    blur_draw = ImageDraw.Draw(blur_layer, "RGBA")
    blur_draw.text(position, text, font=font, fill=(*glow, 130), anchor=anchor)
    blur_layer = blur_layer.filter(ImageFilter.GaussianBlur(radius=10))
    frame.alpha_composite(blur_layer)
    draw = ImageDraw.Draw(frame, "RGBA")
    draw.text(
        position,
        text,
        font=font,
        fill=(*fill, 255),
        anchor=anchor,
        stroke_width=stroke_width,
        stroke_fill=(*stroke_fill, 190),
    )


def _animated_value(target: float, progress: float) -> str:
    reveal = _ease_out(_clamp(progress * 1.15))
    current = target * reveal
    if abs(target - round(target)) < 1e-9:
        return str(int(round(current)))
    return f"{current:.1f}".rstrip("0").rstrip(".")


def _score_at_time(t: float) -> tuple[float, float]:
    left = 0.0
    right = 0.0
    for stat, (name, start, end) in zip(STATS, SCENES[3:7]):
        if t >= end:
            if stat["winner"] == "left":
                left += 1.0
            elif stat["winner"] == "right":
                right += 1.0
            continue
        if start <= t < end:
            local = _ease_out((t - start) / max(1e-6, end - start))
            if stat["winner"] == "left":
                left += local
            elif stat["winner"] == "right":
                right += local
            break
    return left, right


def _score_text(score: tuple[float, float]) -> str:
    return f"{int(round(score[0]))} - {int(round(score[1]))}"


def _background_kind(scene: str) -> str:
    if scene in {"pause_clay", "stat_3", "hold"}:
        return "clay"
    if scene in {"twist_grass", "stat_4", "ending"}:
        return "grass"
    return "split"


def _make_background(kind: str) -> Image.Image:
    xx = np.linspace(0, 1, WIDTH, dtype=np.float32)
    yy = np.linspace(0, 1, HEIGHT, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(xx, yy)

    if kind == "clay":
        base = np.array([14, 10, 14], dtype=np.float32)
        left_glow = np.array(FEDERER["accent"], dtype=np.float32)
        right_glow = np.array((255, 150, 92), dtype=np.float32)
        court = np.array((145, 34, 35), dtype=np.float32)
    elif kind == "grass":
        base = np.array([8, 16, 18], dtype=np.float32)
        left_glow = np.array((120, 214, 142), dtype=np.float32)
        right_glow = np.array(NADAL["accent"], dtype=np.float32)
        court = np.array((12, 72, 58), dtype=np.float32)
    else:
        base = np.array([7, 12, 20], dtype=np.float32)
        left_glow = np.array(FEDERER["accent"], dtype=np.float32)
        right_glow = np.array(NADAL["accent"], dtype=np.float32)
        court = np.array((18, 34, 54), dtype=np.float32)

    left_hot = np.exp(-(((grid_x - 0.18) / 0.22) ** 2 + ((grid_y - 0.34) / 0.26) ** 2))
    right_hot = np.exp(-(((grid_x - 0.82) / 0.22) ** 2 + ((grid_y - 0.34) / 0.26) ** 2))
    center_hot = np.exp(-(((grid_x - 0.50) / 0.32) ** 2 + ((grid_y - 0.56) / 0.26) ** 2))
    vertical = np.clip(1.0 - 0.72 * grid_y, 0.0, 1.0)

    img = np.clip(
        base[None, None, :] * (1.0 - 0.70 * vertical[..., None])
        + court[None, None, :] * (0.55 * vertical[..., None])
        + left_glow[None, None, :] * (0.18 * left_hot[..., None])
        + right_glow[None, None, :] * (0.20 * right_hot[..., None])
        + np.array((255, 255, 255), dtype=np.float32)[None, None, :] * (0.02 * center_hot[..., None]),
        0,
        255,
    ).astype(np.uint8)

    frame = Image.fromarray(img, mode="RGB").convert("RGBA")
    overlay = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")
    draw.rounded_rectangle((38, 56, WIDTH - 38, HEIGHT - 56), radius=54, outline=(255, 255, 255, 16), width=2)
    draw.line((92, 298, WIDTH - 92, 298), fill=(255, 255, 255, 10), width=2)
    draw.line((92, 1560, WIDTH - 92, 1560), fill=(255, 255, 255, 8), width=2)
    draw.ellipse((0, 168, 340, 560), fill=(*FEDERER["accent"], 18))
    draw.ellipse((740, 168, 1080, 560), fill=(*NADAL["accent"], 18))
    if kind == "clay":
        draw.line((110, 820, 970, 820), fill=(255, 150, 92, 20), width=6)
        draw.line((110, 860, 970, 860), fill=(255, 150, 92, 14), width=3)
    elif kind == "grass":
        draw.line((110, 820, 970, 820), fill=(120, 214, 142, 20), width=6)
        draw.line((110, 860, 970, 860), fill=(120, 214, 142, 14), width=3)
    overlay = overlay.filter(ImageFilter.GaussianBlur(radius=10))
    frame.alpha_composite(overlay)
    return frame


def _apply_motion(frame: Image.Image, scene: SceneState, t: float) -> None:
    draw = ImageDraw.Draw(frame, "RGBA")
    pulse = 0.5 + 0.5 * math.sin(t * 2.4)
    sweep_x = int(_lerp(120, WIDTH - 120, 0.5 + 0.5 * math.sin(t * 1.2)))
    sweep_alpha = int(26 + 18 * pulse)
    draw.ellipse((sweep_x - 210, 120, sweep_x + 210, 540), fill=(255, 255, 255, sweep_alpha))
    if scene.name == "climax":
        draw.ellipse((530, 780, 980, 1230), fill=(*NADAL["accent"], 24))
    elif scene.name == "twist_grass":
        draw.ellipse((120, 760, 540, 1180), fill=(*FEDERER["accent"], 20))
    elif scene.name == "pause_clay":
        draw.ellipse((500, 760, 980, 1260), fill=(255, 144, 90, 22))
    draw.rectangle((0, 0, WIDTH, 36), fill=(0, 0, 0, 60))
    draw.rectangle((0, HEIGHT - 36, WIDTH, HEIGHT), fill=(0, 0, 0, 84))


def _draw_header(
    frame: Image.Image,
    scene: SceneState,
    hook_text: str,
    sub_text: str,
    top_scale: float,
) -> None:
    title_font = _fit_font_size(ImageDraw.Draw(frame), hook_text, 920, 84, 28, bold=True)
    sub_font = _load_font(22, bold=True)
    x = WIDTH // 2
    y = 116
    if scene.name in {"flash", "hook"}:
        jitter = int(4 * math.sin(scene.progress * 30.0))
        y += jitter
    _draw_glow_text(frame, (x, y), hook_text, title_font, (255, 248, 236), (255, 200, 140), anchor="ma", stroke_width=3)
    if top_scale > 0:
        badge = Image.new("RGBA", (WIDTH, 120), (0, 0, 0, 0))
        bd = ImageDraw.Draw(badge, "RGBA")
        badge_alpha = int(255 * top_scale)
        bd.rounded_rectangle((262, 64, 818, 110), radius=20, fill=(10, 16, 26, int(170 * top_scale)), outline=(255, 255, 255, int(34 * top_scale)), width=2)
        bd.text((540, 87), sub_text, font=sub_font, fill=(230, 237, 246, badge_alpha), anchor="mm")
        frame.alpha_composite(badge, (0, 0))


def _draw_duel_cards(frame: Image.Image, scene: SceneState, left_img: Image.Image, right_img: Image.Image) -> None:
    draw = ImageDraw.Draw(frame, "RGBA")
    intro = _ease_out(_clamp((scene.progress if scene.name in {"hook", "install", "ending"} else 1.0)))
    if scene.name == "hook":
        left_slide = int(_lerp(-340, 0, intro))
        right_slide = int(_lerp(340, 0, intro))
    elif scene.name == "ending":
        left_slide = int(_lerp(0, -80, intro))
        right_slide = int(_lerp(0, 80, intro))
    else:
        left_slide = 0
        right_slide = 0

    emphasis = {
        "clay": "right",
        "grass": "left",
        "split": None,
    }
    accent_kind = _background_kind(scene.name)
    dominant = emphasis[accent_kind]
    left_scale = 1.0
    right_scale = 1.0
    if dominant == "right":
        right_scale = 1.08
        left_scale = 0.94
    elif dominant == "left":
        left_scale = 1.08
        right_scale = 0.94

    card_top = 332
    card_bottom = 766
    left_box = (42 + left_slide, card_top, 512 + left_slide, card_bottom)
    right_box = (568 + right_slide, card_top, 1038 + right_slide, card_bottom)

    draw.rounded_rectangle(left_box, radius=44, fill=(10, 24, 40, 228), outline=(*FEDERER["accent"], 170), width=2)
    draw.rounded_rectangle(right_box, radius=44, fill=(42, 18, 20, 228), outline=(*NADAL["accent"], 180), width=2)
    draw.rounded_rectangle((436, 460, 644, 616), radius=36, fill=(11, 18, 28, 246), outline=(255, 221, 154, 200), width=4)

    left_tile = _circle_tile(left_img, FEDERER["accent"], size=int(186 * left_scale))
    right_tile = _circle_tile(right_img, NADAL["accent"], size=int(186 * right_scale))
    left_pos = (left_box[0] + 108 - (left_tile.width - 214) // 2, card_top + 52 - (left_tile.height - 214) // 2)
    right_pos = (right_box[0] + 108 - (right_tile.width - 214) // 2, card_top + 52 - (right_tile.height - 214) // 2)
    frame.alpha_composite(left_tile, left_pos)
    frame.alpha_composite(right_tile, right_pos)

    name_font = _load_font(30, bold=True)
    sub_font = _load_font(18, bold=True)
    _draw_glow_text(frame, (left_box[0] + 234, 616), FEDERER["name"], name_font, (245, 248, 252), FEDERER["accent"], anchor="mm", stroke_width=2)
    _draw_glow_text(frame, (right_box[0] + 234, 616), NADAL["name"], name_font, (245, 248, 252), NADAL["accent"], anchor="mm", stroke_width=2)
    draw.text((left_box[0] + 234, 652), "ROGER", font=sub_font, fill="#d8e8f7", anchor="mm")
    draw.text((right_box[0] + 234, 652), "RAFAEL", font=sub_font, fill="#ffe1d4", anchor="mm")
    _draw_glow_text(frame, (540, 532), "VS", _load_font(42, bold=True), (255, 226, 162), (255, 220, 120), anchor="mm", stroke_width=3)
    draw.text((540, 596), "4 STATS. 1 WINNER.", font=_load_font(20, bold=True), fill="#e8edf4", anchor="mm")


def _draw_score_chip(frame: Image.Image, score: tuple[float, float], scene: SceneState) -> None:
    draw = ImageDraw.Draw(frame, "RGBA")
    value = _score_text(score)
    pulse = _ease_out(scene.progress if scene.name in {"stat_1", "stat_2", "stat_3", "stat_4", "climax", "final_board"} else 0.0)
    box = (358, 810, 722, 936)
    draw.rounded_rectangle(box, radius=34, fill=(9, 18, 30, 234), outline=(255, 255, 255, 24), width=2)
    draw.rounded_rectangle((372, 826, 708, 898), radius=28, fill=(13, 28, 44, 222))
    _draw_glow_text(
        frame,
        (540, 860),
        value,
        _load_font(40, bold=True),
        (255, 248, 236),
        (255, 220, 160),
        anchor="mm",
        stroke_width=3,
    )
    draw.text((540, 904), "STAT BATTLE SCORE", font=_load_font(18, bold=True), fill=(214, 226, 239, int(255 * (0.5 + 0.5 * pulse))), anchor="mm")


def _draw_stat_card(
    frame: Image.Image,
    stat: dict,
    progress: float,
    score: tuple[float, float],
    scene: SceneState,
    left_name: str,
    right_name: str,
) -> None:
    draw = ImageDraw.Draw(frame, "RGBA")
    panel = (56, 980, 1024, 1726)
    draw.rounded_rectangle(panel, radius=50, fill=(8, 16, 28, 226), outline=(255, 255, 255, 20), width=2)
    draw.rounded_rectangle((74, 998, 1006, 1052), radius=26, fill=(15, 26, 42, 240), outline=(255, 255, 255, 16), width=1)
    draw.text((140, 1025), stat["tag"], font=_load_font(18, bold=True), fill="#f4f7fb", anchor="mm")
    draw.text((540, 1025), "DUEL DATA", font=_load_font(20, bold=True), fill="#dce7f2", anchor="mm")
    draw.text((942, 1025), "1 POINT", font=_load_font(18, bold=True), fill="#f4f7fb", anchor="mm")

    label_font = _fit_font_size(draw, stat["label"], 800, 76, 28, bold=True)
    _draw_glow_text(
        frame,
        (540, 1148),
        stat["label"],
        label_font,
        (245, 248, 252),
        (255, 210, 160) if stat["winner"] == "right" else (170, 224, 255),
        anchor="mm",
        stroke_width=3,
    )

    left_target = float(stat["left_value"])
    right_target = float(stat["right_value"])
    left_value = _animated_value(left_target, progress)
    right_value = _animated_value(right_target, progress)
    value_font_left = _fit_font_size(draw, left_value, 250, 108, 40, bold=True)
    value_font_right = _fit_font_size(draw, right_value, 250, 108, 40, bold=True)

    left_box = (112, 1238, 428, 1538)
    right_box = (652, 1238, 968, 1538)
    draw.rounded_rectangle(left_box, radius=42, fill=(14, 28, 48, 224), outline=(*FEDERER["accent"], 145), width=2)
    draw.rounded_rectangle(right_box, radius=42, fill=(48, 20, 22, 224), outline=(*NADAL["accent"], 150), width=2)
    draw.text((270, 1290), left_name, font=_load_font(20, bold=True), fill="#dce7f2", anchor="mm")
    draw.text((810, 1290), right_name, font=_load_font(20, bold=True), fill="#ffe1d4", anchor="mm")
    _draw_glow_text(frame, (270, 1382), left_value, value_font_left, (248, 250, 252), FEDERER["accent"], anchor="mm", stroke_width=3)
    _draw_glow_text(frame, (810, 1382), right_value, value_font_right, (248, 250, 252), NADAL["accent"], anchor="mm", stroke_width=3)

    bar_y = 1500
    bar_left = 150
    bar_right = 930
    total = max(left_target, right_target, 1.0)
    left_width = int(330 * (_clamp(progress) * 0.18 + (left_target / total) * 0.82))
    right_width = int(330 * (_clamp(progress) * 0.18 + (right_target / total) * 0.82))
    draw.rounded_rectangle((bar_left, bar_y, bar_left + 330, bar_y + 18), radius=9, fill=(255, 255, 255, 20))
    draw.rounded_rectangle((bar_right - 330, bar_y, bar_right, bar_y + 18), radius=9, fill=(255, 255, 255, 20))
    draw.rounded_rectangle((bar_left, bar_y, bar_left + left_width, bar_y + 18), radius=9, fill=(*FEDERER["accent"], 235))
    draw.rounded_rectangle((bar_right - right_width, bar_y, bar_right, bar_y + 18), radius=9, fill=(*NADAL["accent"], 235))

    winner = stat["winner"]
    flash = _ease_out(_clamp((progress - 0.68) / 0.32))
    if flash > 0.0:
        glow_color = FEDERER["accent"] if winner == "left" else NADAL["accent"]
        target_x = 270 if winner == "left" else 810
        target_y = 1382
        radius = int(_lerp(40, 118, flash))
        overlay = Image.new("RGBA", frame.size, (0, 0, 0, 0))
        od = ImageDraw.Draw(overlay, "RGBA")
        od.ellipse((target_x - radius, target_y - radius, target_x + radius, target_y + radius), outline=(*glow_color, int(180 * flash)), width=max(4, int(10 * flash)))
        overlay = overlay.filter(ImageFilter.GaussianBlur(radius=10))
        frame.alpha_composite(overlay)

    note_font = _fit_font_size(draw, stat["note"], 620, 36, 22, bold=True)
    note_color = (255, 242, 220) if winner == "right" else (232, 242, 255)
    _draw_glow_text(frame, (540, 1624), stat["note"], note_font, note_color, (255, 220, 160) if winner == "right" else (170, 224, 255), anchor="mm", stroke_width=2)

    score_line = _score_text(score)
    draw.text((540, 1566), score_line, font=_load_font(30, bold=True), fill="#f4f7fb", anchor="mm")


def _draw_transition_text(frame: Image.Image, text: str, font_size: int, y: int, fill: tuple[int, int, int], glow: tuple[int, int, int]) -> None:
    font = _load_font(font_size, bold=True)
    _draw_glow_text(frame, (WIDTH // 2, y), text, font, fill, glow, anchor="mm", stroke_width=3)


def render_video(output_path: Path, audio_path: Path, duration: float, fps: int) -> Path:
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    portraits = {
        "left": _load_portrait(PHOTOS_DIR / FEDERER["photo"], "RF"),
        "right": _load_portrait(PHOTOS_DIR / NADAL["photo"], "RN"),
    }

    backgrounds = {
        "split": _make_background("split"),
        "clay": _make_background("clay"),
        "grass": _make_background("grass"),
    }

    hook_texts = {
        "flash": "FEDERER vs NADAL",
        "hook": "LE DUEL QUI DIVISE TOUT LE MONDE",
        "install": "4 STATS. 1 VAINQUEUR.",
        "pause_clay": "ET SUR TERRE BATTUE...",
        "stat_3": "UNPLAYABLE",
        "hold": "NADAL EST MONSTRUEUX ICI",
        "twist_grass": "MAIS SUR GAZON...",
        "stat_4": "FEDERER STRIKES BACK",
        "final_board": "SCORE FINAL",
        "climax": "NADAL WINS THIS ONE",
        "goat_twist": "MAIS EST-CE ASSEZ POUR ETRE LE GOAT ?",
        "ending": "TU CHOISIS QUI ?",
    }

    def make_frame(t: float) -> np.ndarray:
        scene = _scene_for_time(t)
        bg = backgrounds[_background_kind(scene.name)].copy()
        _apply_motion(bg, scene, t)

        left_score, right_score = _score_at_time(t)
        draw = ImageDraw.Draw(bg, "RGBA")

        # Scene-specific top copy keeps the narrative moving without overcrowding the screen.
        if scene.name == "flash":
            flash = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
            fd = ImageDraw.Draw(flash, "RGBA")
            alpha = int(240 * _ease_out(scene.progress))
            fd.rectangle((0, 0, WIDTH, HEIGHT), fill=(0, 0, 0, alpha))
            fd.rectangle((448, 848, 632, 1004), fill=(255, 255, 255, int(28 * scene.progress)))
            fd.text((540, 926), "FEDERER vs NADAL", font=_fit_font_size(fd, "FEDERER vs NADAL", 900, 86, 32, bold=True), fill=(255, 248, 236, 255), anchor="mm", stroke_width=3, stroke_fill=(0, 0, 0, 210))
            bg.alpha_composite(flash)
            return np.array(bg.convert("RGB"))

        hook_text = hook_texts.get(scene.name, "FEDERER vs NADAL")
        subtitle = "4 STATS. 1 WINNER."
        if scene.name == "hook":
            scale = 0.55 + 0.45 * _ease_out(scene.progress)
        elif scene.name == "install":
            scale = 1.0
        else:
            scale = 0.0
        _draw_header(bg, scene, hook_text, subtitle, scale)
        _draw_duel_cards(bg, scene, portraits["left"], portraits["right"])
        _draw_score_chip(bg, (left_score, right_score), scene)

        if scene.name in {"stat_1", "stat_2", "stat_3", "stat_4"}:
            stat_index = {"stat_1": 0, "stat_2": 1, "stat_3": 2, "stat_4": 3}[scene.name]
            _draw_stat_card(
                bg,
                STATS[stat_index],
                scene.progress,
                (left_score, right_score),
                scene,
                FEDERER["name"],
                NADAL["name"],
            )

        if scene.name == "hook":
            _draw_transition_text(bg, "LE VRAI BOSS ?", 44, 176, (255, 242, 220), (255, 190, 120))
        elif scene.name == "install":
            _draw_transition_text(bg, _score_text((0, 0)), 40, 790, (255, 248, 236), (255, 220, 160))
        elif scene.name == "pause_clay":
            _draw_transition_text(bg, "ET SUR TERRE BATTUE...", 44, 930, (255, 244, 236), (255, 150, 90))
        elif scene.name == "stat_3":
            _draw_transition_text(bg, "UNPLAYABLE", 58, 948, (255, 242, 236), (255, 154, 86))
        elif scene.name == "hold":
            _draw_transition_text(bg, "63 SUR TERRE BATTUE.", 42, 946, (255, 244, 236), (255, 160, 92))
        elif scene.name == "twist_grass":
            _draw_transition_text(bg, "MAIS SUR GAZON...", 46, 936, (234, 244, 252), (120, 214, 142))
        elif scene.name == "stat_4":
            _draw_transition_text(bg, "FEDERER STRIKES BACK", 52, 946, (241, 247, 252), (170, 224, 255))
        elif scene.name == "final_board":
            _draw_transition_text(bg, _score_text((1, 3)), 98, 1040, (255, 248, 236), (255, 206, 153))
            draw.text((540, 1122), "STAT BATTLE WINNER", font=_load_font(22, bold=True), fill="#dbe6f0", anchor="mm")
        elif scene.name == "climax":
            final_box = (182, 1002, 898, 1328)
            draw.rounded_rectangle(final_box, radius=44, fill=(118, 22, 30, 184), outline=(255, 122, 122, 230), width=10)
            _draw_glow_text(bg, (540, 1158), "NADAL WINS THIS ONE", _fit_font_size(draw, "NADAL WINS THIS ONE", 620, 78, 30, bold=True), (255, 240, 240), NADAL["accent"], anchor="mm", stroke_width=3)
        elif scene.name == "goat_twist":
            _draw_transition_text(bg, "MAIS EST-CE ASSEZ POUR ETRE LE GOAT ?", 34, 1478, (245, 248, 252), (255, 220, 160))
        elif scene.name == "ending":
            end_box = (216, 1530, 864, 1710)
            draw.rounded_rectangle(end_box, radius=40, fill=(10, 18, 30, 220), outline=(255, 255, 255, 24), width=2)
            draw.text((540, 1594), "TU CHOISIS QUI ?", font=_load_font(30, bold=True), fill="#f4f7fb", anchor="mm")
            draw.text((540, 1642), "TEAM FEDERER OU TEAM NADAL ?", font=_load_font(22, bold=True), fill="#dce7f2", anchor="mm")

        return np.array(bg.convert("RGB"))

    clip = VideoClip(make_frame, duration=duration)
    audio_track, keep_alive = build_audio_track(audio_path, duration)
    clip = clip.with_audio(audio_track.with_volume_scaled(MUSIC_VOLUME))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_output = output_path.with_name(f"{output_path.stem}.render.mp4")
    tmp_audio = output_path.with_name(f"{output_path.stem}.temp_audio.m4a")
    try:
        clip.write_videofile(
            str(tmp_output),
            fps=fps,
            codec="libx264",
            audio_codec="aac",
            bitrate="12000k",
            preset="slow",
            temp_audiofile=str(tmp_audio),
            remove_temp=False,
        )
        if output_path.exists():
            output_path.unlink()
        tmp_output.replace(output_path)
    finally:
        clip.close()
        audio_track.close()
        for item in keep_alive:
            item.close()
        if tmp_audio.exists():
            try:
                tmp_audio.unlink()
            except OSError:
                pass
        if tmp_output.exists() and tmp_output != output_path:
            try:
                tmp_output.unlink()
            except OSError:
                pass
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate an optimized Federer vs Nadal duel Shorts video.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--audio", type=Path, default=DEFAULT_AUDIO)
    parser.add_argument("--duration", type=float, default=TOTAL_DURATION)
    parser.add_argument("--fps", type=int, default=FPS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = render_video(args.output, args.audio, args.duration, args.fps)
    print(f"[video_generator] Federer vs Nadal duel Shorts generated -> {output}")


if __name__ == "__main__":
    main()
