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
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "basketball" / "sga_vs_jokic_stats_shorts.mp4"
DEFAULT_PREVIEW_OUTPUT = PROJECT_ROOT / "data" / "processed" / "basketball" / "sga_vs_jokic_stats_shorts_preview.mp4"
DEFAULT_AUDIO_PATH = DEFAULT_AUDIO
ASSETS_DIR = PROJECT_ROOT / "data" / "raw" / "nba_mvp_assets"

WIDTH = 1080
HEIGHT = 1920
FPS = 60
TOTAL_DURATION = 50.0
PREVIEW_DURATION = 15.0
MUSIC_VOLUME = 0.42

SGA = {
    "name": "SGA",
    "full_name": "Shai Gilgeous-Alexander",
    "photo_candidates": ["sga.png", "shai.png", "shai_gilgeous_alexander.png"],
    "accent": (0, 122, 193),
    "accent_soft": (94, 183, 255),
    "dark": (11, 16, 28),
}

JOKIC = {
    "name": "JOKIC",
    "full_name": "Nikola Jokic",
    "photo_candidates": ["jokic.png", "nikola_jokic.png", "nikola.png"],
    "accent": (253, 185, 39),
    "accent_soft": (255, 226, 153),
    "dark": (18, 28, 44),
}


@dataclass(frozen=True)
class DuelStat:
    label: str
    left_main: str
    left_sub: str
    right_main: str
    right_sub: str
    winner: str
    note: str
    tag: str


STATS = [
    DuelStat("POINTS / MATCH", "31.1", "PPG", "27.7", "PPG", "left", "SGA scores more", "1 / 8"),
    DuelStat("ASSISTS / MATCH", "6.6", "AST", "10.7", "AST", "right", "Jokic is the engine", "2 / 8"),
    DuelStat("REBOUNDS / MATCH", "4.3", "RPG", "12.9", "RPG", "right", "No contest inside", "3 / 8"),
    DuelStat("FG% / TS%", "55.3%", "TS 66.5%", "56.9%", "TS 67.0%", "right", "Jokic is just cleaner", "4 / 8"),
    DuelStat("MATCHS JOUES", "68", "games", "65", "games", "left", "SGA stayed on the floor", "5 / 8"),
    DuelStat("VICTOIRES / WIN %", "64-18", ".780", "54-28", ".659", "left", "Team success goes OKC", "6 / 8"),
    DuelStat("30+ PTS GAMES", "40", "games", "22", "games", "left", "SGA gets hot more often", "7 / 8"),
    DuelStat("TRIPLE-DOUBLES", "0", "this season", "34", "this season", "right", "Jokic owns the all-around game", "8 / 8"),
]

SCENES = [
    ("flash", 0.00, 0.35),
    ("hook", 0.35, 3.10),
    ("install", 3.10, 5.10),
    ("stat_1", 5.10, 9.25),
    ("stat_2", 9.25, 13.40),
    ("stat_3", 13.40, 17.55),
    ("stat_4", 17.55, 21.70),
    ("stat_5", 21.70, 25.85),
    ("stat_6", 25.85, 30.00),
    ("stat_7", 30.00, 34.15),
    ("stat_8", 34.15, 38.30),
    ("final_board", 38.30, 43.20),
    ("ending", 43.20, TOTAL_DURATION),
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


def _resolve_portrait(player: dict) -> Image.Image:
    candidates = [ASSETS_DIR / name for name in player["photo_candidates"]]
    for candidate in candidates:
        if candidate.exists():
            return _load_portrait(candidate, player["name"][:2].upper())
    return _load_portrait(candidates[0], player["name"][:2].upper())


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
        stroke_fill=(0, 0, 0, 180),
    )


def _make_background(kind: str) -> Image.Image:
    xx = np.linspace(0, 1, WIDTH, dtype=np.float32)
    yy = np.linspace(0, 1, HEIGHT, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(xx, yy)
    if kind == "clutch":
        base = np.array([8, 10, 18], dtype=np.float32)
        left_glow = np.array(SGA["accent"], dtype=np.float32)
        right_glow = np.array(JOKIC["accent"], dtype=np.float32)
        center = np.array([18, 30, 48], dtype=np.float32)
    elif kind == "jokic":
        base = np.array([9, 12, 18], dtype=np.float32)
        left_glow = np.array([72, 174, 255], dtype=np.float32)
        right_glow = np.array(JOKIC["accent"], dtype=np.float32)
        center = np.array([20, 24, 36], dtype=np.float32)
    else:
        base = np.array([8, 11, 20], dtype=np.float32)
        left_glow = np.array(SGA["accent"], dtype=np.float32)
        right_glow = np.array([255, 132, 74], dtype=np.float32)
        center = np.array([18, 24, 38], dtype=np.float32)

    left_hot = np.exp(-(((grid_x - 0.18) / 0.22) ** 2 + ((grid_y - 0.34) / 0.26) ** 2))
    right_hot = np.exp(-(((grid_x - 0.82) / 0.22) ** 2 + ((grid_y - 0.34) / 0.26) ** 2))
    center_hot = np.exp(-(((grid_x - 0.50) / 0.32) ** 2 + ((grid_y - 0.56) / 0.26) ** 2))
    vertical = np.clip(1.0 - 0.72 * grid_y, 0.0, 1.0)

    img = np.clip(
        base[None, None, :] * (1.0 - 0.70 * vertical[..., None])
        + center[None, None, :] * (0.46 * vertical[..., None])
        + left_glow[None, None, :] * (0.18 * left_hot[..., None])
        + right_glow[None, None, :] * (0.18 * right_hot[..., None])
        + np.array((255, 255, 255), dtype=np.float32)[None, None, :] * (0.03 * center_hot[..., None]),
        0,
        255,
    ).astype(np.uint8)

    frame = Image.fromarray(img, mode="RGB").convert("RGBA")
    overlay = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")
    draw.rounded_rectangle((38, 56, WIDTH - 38, HEIGHT - 56), radius=54, outline=(255, 255, 255, 16), width=2)
    draw.line((92, 298, WIDTH - 92, 298), fill=(255, 255, 255, 10), width=2)
    draw.line((92, 1560, WIDTH - 92, 1560), fill=(255, 255, 255, 8), width=2)
    draw.ellipse((0, 168, 340, 560), fill=(*SGA["accent"], 18))
    draw.ellipse((740, 168, 1080, 560), fill=(*JOKIC["accent"], 18))
    overlay = overlay.filter(ImageFilter.GaussianBlur(radius=10))
    frame.alpha_composite(overlay)
    return frame


def _apply_motion(frame: Image.Image, scene: SceneState, t: float) -> None:
    draw = ImageDraw.Draw(frame, "RGBA")
    pulse = 0.5 + 0.5 * math.sin(t * 2.4)
    sweep_x = int(_lerp(120, WIDTH - 120, 0.5 + 0.5 * math.sin(t * 1.2)))
    sweep_alpha = int(20 + 16 * pulse)
    draw.ellipse((sweep_x - 210, 120, sweep_x + 210, 540), fill=(255, 255, 255, sweep_alpha))
    if scene.name in {"stat_2", "stat_3", "stat_4", "stat_8"}:
        draw.ellipse((500, 760, 980, 1260), fill=(*JOKIC["accent"], 18))
    elif scene.name in {"stat_1", "stat_5", "stat_6", "stat_7"}:
        draw.ellipse((120, 760, 540, 1180), fill=(*SGA["accent"], 18))
    draw.rectangle((0, 0, WIDTH, 36), fill=(0, 0, 0, 60))
    draw.rectangle((0, HEIGHT - 36, WIDTH, HEIGHT), fill=(0, 0, 0, 84))


def _load_player_portrait(player: dict) -> Image.Image:
    return _resolve_portrait(player)


def _draw_header(frame: Image.Image, scene: SceneState, hook_text: str, sub_text: str, top_scale: float) -> None:
    title_font = _fit_font_size(ImageDraw.Draw(frame), hook_text, 920, 84, 28, bold=True)
    sub_font = _load_font(22, bold=True)
    x = WIDTH // 2
    y = 116
    if scene.name in {"flash", "hook"}:
        y += int(4 * math.sin(scene.progress * 30.0))
    _draw_glow_text(frame, (x, y), hook_text, title_font, (255, 248, 236), (255, 214, 134), anchor="ma", stroke_width=3)
    if top_scale > 0:
        badge = Image.new("RGBA", (WIDTH, 120), (0, 0, 0, 0))
        bd = ImageDraw.Draw(badge, "RGBA")
        bd.rounded_rectangle((250, 64, 830, 110), radius=20, fill=(10, 16, 26, int(170 * top_scale)), outline=(255, 255, 255, int(34 * top_scale)), width=2)
        bd.text((540, 87), sub_text, font=sub_font, fill=(230, 237, 246, int(255 * top_scale)), anchor="mm")
        frame.alpha_composite(badge, (0, 0))


def _score_at_time(t: float) -> tuple[int, int]:
    left = 0
    right = 0
    for stat, (_, start, end) in zip(STATS, SCENES[3:11]):
        if t >= end:
            if stat.winner == "left":
                left += 1
            elif stat.winner == "right":
                right += 1
            continue
        if start <= t < end:
            local = _ease_out((t - start) / max(1e-6, end - start))
            if local > 0.72:
                if stat.winner == "left":
                    left += 1
                elif stat.winner == "right":
                    right += 1
            break
    return left, right


def _score_text(score: tuple[int, int]) -> str:
    return f"{score[0]} - {score[1]}"


def _draw_duel_cards(frame: Image.Image, scene: SceneState, left_img: Image.Image, right_img: Image.Image) -> None:
    draw = ImageDraw.Draw(frame, "RGBA")
    intro = _ease_out(_clamp(scene.progress if scene.name in {"hook", "install", "ending"} else 1.0))
    if scene.name == "hook":
        left_slide = int(_lerp(-340, 0, intro))
        right_slide = int(_lerp(340, 0, intro))
    elif scene.name == "ending":
        left_slide = int(_lerp(0, -80, intro))
        right_slide = int(_lerp(0, 80, intro))
    else:
        left_slide = 0
        right_slide = 0

    left_box = (42 + left_slide, 332, 512 + left_slide, 766)
    right_box = (568 + right_slide, 332, 1038 + right_slide, 766)

    draw.rounded_rectangle(left_box, radius=44, fill=(10, 24, 40, 228), outline=(*SGA["accent"], 170), width=2)
    draw.rounded_rectangle(right_box, radius=44, fill=(42, 18, 20, 228), outline=(*JOKIC["accent"], 180), width=2)
    draw.rounded_rectangle((436, 460, 644, 616), radius=36, fill=(11, 18, 28, 246), outline=(255, 221, 154, 200), width=4)

    left_tile = _circle_tile(left_img, SGA["accent"], size=186)
    right_tile = _circle_tile(right_img, JOKIC["accent"], size=186)
    left_pos = (left_box[0] + 108 - (left_tile.width - 214) // 2, 332 + 52 - (left_tile.height - 214) // 2)
    right_pos = (right_box[0] + 108 - (right_tile.width - 214) // 2, 332 + 52 - (right_tile.height - 214) // 2)
    frame.alpha_composite(left_tile, left_pos)
    frame.alpha_composite(right_tile, right_pos)

    name_font = _load_font(30, bold=True)
    sub_font = _load_font(18, bold=True)
    _draw_glow_text(frame, (left_box[0] + 234, 616), SGA["name"], name_font, (245, 248, 252), SGA["accent"], anchor="mm", stroke_width=2)
    _draw_glow_text(frame, (right_box[0] + 234, 616), JOKIC["name"], name_font, (245, 248, 252), JOKIC["accent"], anchor="mm", stroke_width=2)
    draw.text((left_box[0] + 234, 652), "OKLAHOMA CITY", font=sub_font, fill="#d8e8f7", anchor="mm")
    draw.text((right_box[0] + 234, 652), "DENVER", font=sub_font, fill="#ffe1d4", anchor="mm")
    _draw_glow_text(frame, (540, 532), "VS", _load_font(42, bold=True), (255, 226, 162), (255, 220, 120), anchor="mm", stroke_width=3)
    draw.text((540, 596), "8 STATS. 1 DECISION.", font=_load_font(20, bold=True), fill="#e8edf4", anchor="mm")


def _draw_score_chip(frame: Image.Image, score: tuple[int, int], scene: SceneState) -> None:
    draw = ImageDraw.Draw(frame, "RGBA")
    pulse = _ease_out(scene.progress if scene.name.startswith("stat_") or scene.name in {"final_board", "ending"} else 0.0)
    box = (358, 810, 722, 936)
    draw.rounded_rectangle(box, radius=34, fill=(9, 18, 30, 234), outline=(255, 255, 255, 24), width=2)
    draw.rounded_rectangle((372, 826, 708, 898), radius=28, fill=(13, 28, 44, 222))
    _draw_glow_text(
        frame,
        (540, 860),
        _score_text(score),
        _load_font(40, bold=True),
        (255, 248, 236),
        (255, 220, 160),
        anchor="mm",
        stroke_width=3,
    )
    draw.text((540, 904), "BATTLE SCORE", font=_load_font(18, bold=True), fill=(214, 226, 239, int(255 * (0.5 + 0.5 * pulse))), anchor="mm")


def _draw_stat_card(frame: Image.Image, stat: DuelStat, progress: float, score: tuple[int, int], scene: SceneState) -> None:
    draw = ImageDraw.Draw(frame, "RGBA")
    panel = (56, 980, 1024, 1726)
    draw.rounded_rectangle(panel, radius=50, fill=(8, 16, 28, 226), outline=(255, 255, 255, 20), width=2)
    draw.rounded_rectangle((74, 998, 1006, 1052), radius=26, fill=(15, 26, 42, 240), outline=(255, 255, 255, 16), width=1)
    draw.text((140, 1025), stat.tag, font=_load_font(18, bold=True), fill="#f4f7fb", anchor="mm")
    draw.text((540, 1025), "DUEL DATA", font=_load_font(20, bold=True), fill="#dce7f2", anchor="mm")
    draw.text((942, 1025), "1 POINT", font=_load_font(18, bold=True), fill="#f4f7fb", anchor="mm")

    label_font = _fit_font_size(draw, stat.label, 820, 76, 28, bold=True)
    _draw_glow_text(
        frame,
        (540, 1148),
        stat.label,
        label_font,
        (245, 248, 252),
        (255, 210, 160) if stat.winner == "right" else (170, 224, 255),
        anchor="mm",
        stroke_width=3,
    )

    left_box = (112, 1238, 428, 1538)
    right_box = (652, 1238, 968, 1538)
    draw.rounded_rectangle(left_box, radius=42, fill=(14, 28, 48, 224), outline=(*SGA["accent"], 145), width=2)
    draw.rounded_rectangle(right_box, radius=42, fill=(48, 20, 22, 224), outline=(*JOKIC["accent"], 150), width=2)

    left_title = _load_font(20, bold=True)
    left_value_font = _fit_font_size(draw, stat.left_main, 250, 108, 40, bold=True)
    right_value_font = _fit_font_size(draw, stat.right_main, 250, 108, 40, bold=True)
    sub_font = _load_font(18, bold=True)
    draw.text((270, 1290), SGA["name"], font=left_title, fill="#dce7f2", anchor="mm")
    draw.text((810, 1290), JOKIC["name"], font=left_title, fill="#ffe1d4", anchor="mm")
    _draw_glow_text(frame, (270, 1378), stat.left_main, left_value_font, (248, 250, 252), SGA["accent"], anchor="mm", stroke_width=3)
    _draw_glow_text(frame, (810, 1378), stat.right_main, right_value_font, (248, 250, 252), JOKIC["accent"], anchor="mm", stroke_width=3)
    draw.text((270, 1454), stat.left_sub, font=sub_font, fill="#c7d7e5", anchor="mm")
    draw.text((810, 1454), stat.right_sub, font=sub_font, fill="#f1ddd2", anchor="mm")

    bar_y = 1516
    if stat.winner == "left":
        left_width = int(330 * (_clamp(progress) * 0.18 + 1.0 * 0.82))
        right_width = int(330 * (_clamp(progress) * 0.18 + 0.70 * 0.82))
    elif stat.winner == "right":
        left_width = int(330 * (_clamp(progress) * 0.18 + 0.70 * 0.82))
        right_width = int(330 * (_clamp(progress) * 0.18 + 1.0 * 0.82))
    else:
        left_width = right_width = int(330 * (_clamp(progress) * 0.18 + 0.90 * 0.82))

    draw.rounded_rectangle((150, bar_y, 480, bar_y + 18), radius=9, fill=(255, 255, 255, 20))
    draw.rounded_rectangle((600, bar_y, 930, bar_y + 18), radius=9, fill=(255, 255, 255, 20))
    draw.rounded_rectangle((150, bar_y, 150 + left_width, bar_y + 18), radius=9, fill=(*SGA["accent"], 235))
    draw.rounded_rectangle((600 + 330 - right_width, bar_y, 930, bar_y + 18), radius=9, fill=(*JOKIC["accent"], 235))

    score_text = _score_text(score)
    draw.text((540, 1566), score_text, font=_load_font(30, bold=True), fill="#f4f7fb", anchor="mm")
    note_font = _fit_font_size(draw, stat.note, 620, 36, 22, bold=True)
    note_color = (255, 242, 220) if stat.winner == "right" else (232, 242, 255)
    _draw_glow_text(frame, (540, 1624), stat.note, note_font, note_color, (255, 220, 160) if stat.winner == "right" else (170, 224, 255), anchor="mm", stroke_width=2)


def _draw_transition_text(frame: Image.Image, text: str, font_size: int, y: int, fill: tuple[int, int, int], glow: tuple[int, int, int]) -> None:
    font = _load_font(font_size, bold=True)
    _draw_glow_text(frame, (WIDTH // 2, y), text, font, fill, glow, anchor="mm", stroke_width=3)


def render_video(output_path: Path, audio_path: Path, duration: float, fps: int) -> Path:
    if duration > TOTAL_DURATION + 0.05:
        raise ValueError(f"Configured duration mismatch: expected at most {TOTAL_DURATION:.2f}s, got {duration:.2f}s")

    left_portrait = _load_player_portrait(SGA)
    right_portrait = _load_player_portrait(JOKIC)
    backgrounds = {
        "split": _make_background("split"),
        "clutch": _make_background("clutch"),
        "jokic": _make_background("jokic"),
    }
    hook_texts = {
        "flash": "SGA vs JOKIC",
        "hook": "POINTS PAR MATCH ?",
        "install": "8 STATS. 1 DECISION.",
        "stat_1": "SGA STARTS HOT",
        "stat_2": "JOKIC PLAYMAKES",
        "stat_3": "JOKIC DOMINATES THE GLASS",
        "stat_4": "FG% / TS%",
        "stat_5": "MATCHS JOUES",
        "stat_6": "WIN % BATTLE",
        "stat_7": "30+ POINT GAMES",
        "stat_8": "TRIPLE-DOUBLES",
        "final_board": "SCOREBOARD",
        "ending": "WHO YOU GOT ?",
    }

    def make_frame(t: float) -> np.ndarray:
        scene = _scene_for_time(t)
        bg = backgrounds["split"].copy()
        if scene.name in {"stat_3", "stat_4", "stat_8"}:
            bg = backgrounds["jokic"].copy()
        elif scene.name in {"stat_1", "stat_5", "stat_6", "stat_7"}:
            bg = backgrounds["clutch"].copy()
        _apply_motion(bg, scene, t)

        left_score, right_score = _score_at_time(t)
        draw = ImageDraw.Draw(bg, "RGBA")

        if scene.name == "flash":
            flash = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
            fd = ImageDraw.Draw(flash, "RGBA")
            alpha = int(240 * _ease_out(scene.progress))
            fd.rectangle((0, 0, WIDTH, HEIGHT), fill=(0, 0, 0, alpha))
            fd.rectangle((448, 848, 632, 1004), fill=(255, 255, 255, int(28 * scene.progress)))
            fd.text((540, 926), "SGA vs JOKIC", font=_fit_font_size(fd, "SGA vs JOKIC", 900, 86, 32, bold=True), fill=(255, 248, 236, 255), anchor="mm", stroke_width=3, stroke_fill=(0, 0, 0, 210))
            bg.alpha_composite(flash)
            return np.array(bg.convert("RGB"))

        hook_text = hook_texts.get(scene.name, "SGA vs JOKIC")
        subtitle = "8 STATS. 1 DECISION."
        if scene.name == "hook":
            scale = 0.55 + 0.45 * _ease_out(scene.progress)
        elif scene.name == "install":
            scale = 1.0
        else:
            scale = 0.0
        _draw_header(bg, scene, hook_text, subtitle, scale)
        _draw_duel_cards(bg, scene, left_portrait, right_portrait)
        _draw_score_chip(bg, (left_score, right_score), scene)

        if scene.name.startswith("stat_"):
            stat_index = int(scene.name.split("_")[1]) - 1
            _draw_stat_card(bg, STATS[stat_index], scene.progress, (left_score, right_score), scene)

        if scene.name == "hook":
            _draw_transition_text(bg, "WHO IS THE REAL MVP?", 44, 176, (255, 242, 220), (255, 190, 120))
        elif scene.name == "install":
            _draw_transition_text(bg, "LET'S SEE THE RECEIPTS.", 40, 790, (255, 248, 236), (255, 220, 160))
        elif scene.name == "stat_4":
            _draw_transition_text(bg, "FG% / TS% = CLEAN OR NOT", 38, 936, (234, 244, 252), (255, 214, 134))
        elif scene.name == "stat_8":
            _draw_transition_text(bg, "JOKIC HAS THE CRAZY LINE.", 40, 946, (241, 247, 252), (170, 224, 255))
        elif scene.name == "final_board":
            score = _score_text((left_score, right_score))
            _draw_transition_text(bg, score, 98, 1040, (255, 248, 236), (255, 206, 153))
            draw.text((540, 1122), "STAT BATTLE SCORE", font=_load_font(22, bold=True), fill="#dbe6f0", anchor="mm")
            final_box = (182, 1188, 898, 1518)
            draw.rounded_rectangle(final_box, radius=44, fill=(16, 20, 30, 184), outline=(255, 122, 122, 230), width=8)
            _draw_glow_text(bg, (540, 1328), "4 - 4", _fit_font_size(draw, "4 - 4", 600, 78, 30, bold=True), (255, 240, 240), (255, 220, 160), anchor="mm", stroke_width=3)
            draw.text((540, 1400), "YOU DECIDE", font=_load_font(24, bold=True), fill="#f4f7fb", anchor="mm")
            draw.text((540, 1450), "SGA OR JOKIC ?", font=_load_font(24, bold=True), fill="#dce7f2", anchor="mm")
        elif scene.name == "ending":
            end_box = (216, 1530, 864, 1710)
            draw.rounded_rectangle(end_box, radius=40, fill=(10, 18, 30, 220), outline=(255, 255, 255, 24), width=2)
            draw.text((540, 1594), "WHO YOU GOT ?", font=_load_font(30, bold=True), fill="#f4f7fb", anchor="mm")
            draw.text((540, 1642), "COMMENT SGA OR JOKIC", font=_load_font(22, bold=True), fill="#dce7f2", anchor="mm")

        return np.array(bg.convert("RGB"))

    clip = VideoClip(make_frame, duration=duration)
    audio_track = None
    keep_alive: list[object] = []
    if audio_path.exists():
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
        if audio_track is not None:
            audio_track.close()
        for item in keep_alive:
            try:
                item.close()
            except Exception:
                pass
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
    parser = argparse.ArgumentParser(description="Generate an SGA vs Jokic stats Shorts video.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--preview-output", type=Path, default=DEFAULT_PREVIEW_OUTPUT)
    parser.add_argument("--audio", type=Path, default=DEFAULT_AUDIO_PATH)
    parser.add_argument("--duration", type=float, default=TOTAL_DURATION)
    parser.add_argument("--fps", type=int, default=FPS)
    parser.add_argument("--preview", action="store_true", help="Render a shorter preview cut instead of the full 50s version.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.preview:
        output_path = args.preview_output
        duration = min(args.duration, PREVIEW_DURATION)
        fps = min(args.fps, 30)
    else:
        output_path = args.output
        duration = args.duration
        fps = args.fps
    output = render_video(output_path, args.audio, duration, fps)
    print(f"[video_generator] SGA vs Jokic stats Shorts generated -> {output}")


if __name__ == "__main__":
    main()
