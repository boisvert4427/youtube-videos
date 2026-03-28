from __future__ import annotations

import argparse
from pathlib import Path
from uuid import uuid4

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
ASSETS_DIR = PROJECT_ROOT / "data" / "raw" / "nba_goat_assets"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "basketball" / "lebron_vs_jordan_goat_split_shorts.mp4"

WIDTH = 1080
HEIGHT = 1920
FPS = 30
TOTAL_DURATION = 30.0

LEBRON_BLUE = (70, 146, 255)
JORDAN_RED = (225, 52, 68)
WHITE = (246, 248, 252)

STAT_ROWS = [
    ("TITLES", "4", "6"),
    ("MVP", "4", "5"),
    ("POINTS", "43,180", "32,292"),
    ("PTS/G", "26.8", "30.1"),
    ("REBOUNDS", "11,997", "6,672"),
    ("REB/G", "7.5", "6.2"),
    ("ASSISTS", "11,909", "5,633"),
    ("AST/G", "7.4", "5.3"),
    ("STEALS", "2,399", "2,514"),
    ("STL/G", "1.5", "2.3"),
    ("BLOCKS", "1,179", "893"),
    ("BLK/G", "0.7", "0.8"),
    ("ALL-STAR", "21", "14"),
    ("1ST TEAM", "13", "10"),
    ("DPOY", "0", "1"),
]
STAT_REVEAL_DURATION = 1.5


def _ease_out(value: float) -> float:
    value = min(max(value, 0.0), 1.0)
    return 1.0 - (1.0 - value) ** 3


def _make_background() -> Image.Image:
    xx = np.linspace(0, 1, WIDTH, dtype=np.float32)
    yy = np.linspace(0, 1, HEIGHT, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(xx, yy)
    black = np.array([4, 7, 14], dtype=np.float32)
    navy = np.array([8, 26, 58], dtype=np.float32)
    blue = np.array(LEBRON_BLUE, dtype=np.float32)
    red = np.array(JORDAN_RED, dtype=np.float32)
    white = np.array(WHITE, dtype=np.float32)

    mix = np.clip(0.56 * grid_y + 0.18 * grid_x, 0, 1)
    left_glow = np.exp(-(((grid_x - 0.18) / 0.18) ** 2 + ((grid_y - 0.45) / 0.22) ** 2))
    right_glow = np.exp(-(((grid_x - 0.82) / 0.18) ** 2 + ((grid_y - 0.45) / 0.22) ** 2))
    center_glow = np.exp(-(((grid_x - 0.5) / 0.30) ** 2 + ((grid_y - 0.52) / 0.20) ** 2))
    img = np.clip(
        black[None, None, :] * (1.0 - mix[..., None])
        + navy[None, None, :] * (0.82 * mix[..., None])
        + blue[None, None, :] * (0.16 * left_glow[..., None])
        + red[None, None, :] * (0.16 * right_glow[..., None])
        + white[None, None, :] * (0.04 * center_glow[..., None]),
        0,
        255,
    ).astype(np.uint8)
    frame = Image.fromarray(img, mode="RGB").convert("RGBA")
    overlay = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")
    draw.rounded_rectangle((36, 36, WIDTH - 36, HEIGHT - 36), radius=44, outline=(255, 255, 255, 16), width=2)
    draw.ellipse((170, 180, WIDTH - 170, 960), outline=(255, 255, 255, 10), width=3)
    overlay = overlay.filter(ImageFilter.GaussianBlur(radius=2))
    frame.alpha_composite(overlay)
    return frame


def _load_split_images() -> dict[str, dict[str, Image.Image]]:
    mapping = {
        "lebron": {
            "portrait": ASSETS_DIR / "lebron_portrait.jpg",
            "action": ASSETS_DIR / "lebron_action.jpg",
        },
        "jordan": {
            "portrait": ASSETS_DIR / "jordan_portrait.jpg",
            "action": ASSETS_DIR / "jordan_action.jpg",
        },
    }
    cache: dict[str, dict[str, Image.Image]] = {}
    for player, variants in mapping.items():
        cache[player] = {}
        for variant, path in variants.items():
            img = ImageOps.exif_transpose(Image.open(path)).convert("RGBA")
            cache[player][variant] = img
    return cache


def _make_top_portrait(source: Image.Image, ring_color: tuple[int, int, int]) -> Image.Image:
    size = 168
    portrait = ImageOps.fit(source, (size, size), method=Image.Resampling.LANCZOS, centering=(0.5, 0.22))
    portrait = ImageEnhance.Brightness(portrait).enhance(1.12)
    portrait = ImageEnhance.Contrast(portrait).enhance(1.08)
    mask = Image.new("L", (size, size), 0)
    ImageDraw.Draw(mask).ellipse((0, 0, size - 1, size - 1), fill=255)
    circle = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    circle.paste(portrait, (0, 0), mask)
    tile = Image.new("RGBA", (size + 20, size + 20), (0, 0, 0, 0))
    td = ImageDraw.Draw(tile, "RGBA")
    td.ellipse((0, 0, size + 19, size + 19), fill=(255, 255, 255, 230))
    td.ellipse((5, 5, size + 14, size + 14), fill=(*ring_color, 255))
    td.ellipse((10, 10, size + 9, size + 9), fill=(8, 18, 34, 255))
    tile.alpha_composite(circle, (10, 10))
    return tile


def _make_half_image(source: Image.Image, side: str, zoom: float = 1.0) -> Image.Image:
    target_w = WIDTH // 2
    target_h = HEIGHT
    scaled_w = max(target_w, int(source.width * zoom))
    scaled_h = max(target_h, int(source.height * zoom))
    scaled = ImageOps.contain(source, (scaled_w, scaled_h))
    centering = (0.42, 0.26) if side == "left" else (0.52, 0.24)
    if scaled.width < target_w or scaled.height < target_h:
        scaled = ImageOps.fit(source, (target_w, target_h), method=Image.Resampling.LANCZOS, centering=centering)
    else:
        scaled = ImageOps.fit(scaled, (target_w, target_h), method=Image.Resampling.LANCZOS, centering=centering)
    scaled = ImageEnhance.Brightness(scaled).enhance(1.18)
    scaled = ImageEnhance.Contrast(scaled).enhance(1.12)
    scaled = ImageEnhance.Color(scaled).enhance(1.06)
    canvas = Image.new("RGBA", (target_w, target_h), (0, 0, 0, 0))
    canvas.alpha_composite(scaled, (0, 0))
    tint = Image.new("RGBA", (target_w, target_h), (* (LEBRON_BLUE if side == "left" else JORDAN_RED), 34))
    canvas.alpha_composite(tint)
    vignette = Image.new("RGBA", (target_w, target_h), (0, 0, 0, 0))
    vd = ImageDraw.Draw(vignette, "RGBA")
    for i in range(220):
        alpha = int(92 * (i / 219) ** 1.5)
        vd.line((0, target_h - i - 1, target_w, target_h - i - 1), fill=(0, 0, 0, alpha))
    for i in range(120):
        alpha = int(52 * (i / 119) ** 1.7)
        vd.line((0, i, target_w, i), fill=(0, 0, 0, alpha))
    canvas.alpha_composite(vignette)
    return canvas


def _draw_glow_text(frame: Image.Image, pos: tuple[int, int], text: str, font: ImageFont.ImageFont, fill: tuple[int, int, int], glow: tuple[int, int, int], anchor: str = "ma") -> None:
    ImageDraw.Draw(frame, "RGBA").text(
        pos,
        text,
        font=font,
        fill=(*fill, 255),
        anchor=anchor,
        stroke_width=1,
        stroke_fill=(8, 18, 34, 220),
    )


def _apply_energy_overlay(frame: Image.Image, t: float) -> None:
    overlay = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")
    pulse = 0.5 + 0.5 * np.sin(t * 2.6)
    pulse2 = 0.5 + 0.5 * np.sin(t * 1.9 + 1.3)
    draw.ellipse((-120, 180, 380, 860), fill=(*LEBRON_BLUE, int(38 + 36 * pulse)))
    draw.ellipse((700, 160, 1200, 860), fill=(*JORDAN_RED, int(38 + 36 * pulse2)))
    draw.ellipse((260, 1020, 820, 1760), fill=(255, 255, 255, int(12 + 16 * pulse)))
    for idx in range(7):
        offset = int((t * 140 + idx * 150) % (HEIGHT + 400)) - 200
        alpha = 26 if idx % 2 == 0 else 18
        draw.rounded_rectangle((90 + idx * 18, offset, 160 + idx * 18, offset + 280), radius=22, fill=(255, 255, 255, alpha))
        draw.rounded_rectangle((WIDTH - 160 - idx * 18, HEIGHT - offset - 280, WIDTH - 90 - idx * 18, HEIGHT - offset), radius=22, fill=(255, 255, 255, alpha))
    frame.alpha_composite(overlay)


def _winner_side(left_value: str, right_value: str) -> str:
    left_score = float(left_value.replace(",", ""))
    right_score = float(right_value.replace(",", ""))
    return "left" if left_score > right_score else "right"


def _score_totals() -> tuple[int, int]:
    left_total = 0
    right_total = 0
    for _, left_value, right_value in STAT_ROWS:
        if _winner_side(left_value, right_value) == "left":
            left_total += 1
        else:
            right_total += 1
    return left_total, right_total


def _build_scenes(total_duration: float) -> list[dict]:
    reveal_total = len(STAT_ROWS) * STAT_REVEAL_DURATION
    final_duration = max(4.0, total_duration - reveal_total)
    return [
        *[
            {"kind": "stat", "duration": STAT_REVEAL_DURATION, "index": idx}
            for idx in range(len(STAT_ROWS))
        ],
        {"kind": "board_final", "duration": final_duration},
    ]


def _winning_player_name() -> str:
    left_total, right_total = _score_totals()
    return "LEBRON WINS" if left_total > right_total else "JORDAN WINS"


def _composite_split(frame: Image.Image, images: dict[str, dict[str, Image.Image]], left_variant: str, right_variant: str, zoom: float) -> None:
    left = _make_half_image(images["lebron"][left_variant], "left", zoom)
    right = _make_half_image(images["jordan"][right_variant], "right", zoom)
    frame.alpha_composite(left, (0, 0))
    frame.alpha_composite(right, (WIDTH // 2, 0))


def _draw_top_player_cards(
    frame: Image.Image,
    images: dict[str, dict[str, Image.Image]],
    title_font: ImageFont.ImageFont,
    sub_font: ImageFont.ImageFont,
) -> None:
    draw = ImageDraw.Draw(frame, "RGBA")
    left_box = (42, 104, 500, 422)
    right_box = (580, 104, 1038, 422)
    draw.rounded_rectangle(left_box, radius=42, fill=(9, 24, 48, 224), outline=(*LEBRON_BLUE, 168), width=2)
    draw.rounded_rectangle(right_box, radius=42, fill=(46, 10, 20, 224), outline=(*JORDAN_RED, 168), width=2)

    left_portrait = _make_top_portrait(images["lebron"]["portrait"], LEBRON_BLUE)
    right_portrait = _make_top_portrait(images["jordan"]["portrait"], JORDAN_RED)
    frame.alpha_composite(left_portrait, (74, 146))
    frame.alpha_composite(right_portrait, (612, 146))

    _draw_glow_text(frame, (354, 214), "LEBRON", title_font, WHITE, LEBRON_BLUE)
    _draw_glow_text(frame, (892, 214), "JORDAN", title_font, WHITE, JORDAN_RED)
    draw.text((354, 266), "JAMES", font=sub_font, fill="#dce8f5", anchor="ma")
    draw.text((892, 266), "MJ", font=sub_font, fill="#ffe0e4", anchor="ma")


def _draw_center_stat_scene(
    frame: Image.Image,
    small_font: ImageFont.ImageFont,
    list_font: ImageFont.ImageFont,
    value_font_map: dict[str, ImageFont.ImageFont],
    stat_index: int | None,
    phase: float,
) -> None:
    draw = ImageDraw.Draw(frame, "RGBA")
    panel = (34, 476, WIDTH - 34, 1598)
    draw.rounded_rectangle(panel, radius=48, fill=(7, 18, 36, 220), outline=(255, 255, 255, 20), width=2)
    draw.text((170, 540), "LEBRON", font=small_font, fill="#dce8f5", anchor="ma")
    draw.text((WIDTH // 2, 540), "STATS", font=small_font, fill="#f4f7fb", anchor="ma")
    draw.text((WIDTH - 170, 540), "JORDAN", font=small_font, fill="#ffe0e4", anchor="ma")

    left_target_x = 238
    center_x = WIDTH // 2
    right_target_x = WIDTH - 238
    left_start_x = 238
    right_start_x = WIDTH - 238
    final_board = stat_index is None

    row_h = 66
    for idx, (stat_name, left_value, right_value) in enumerate(STAT_ROWS):
        row_top = 592 + idx * row_h
        row_bottom = row_top + 62
        row_center_y = (row_top + row_bottom) // 2
        active = stat_index is not None and idx == stat_index
        revealed = final_board or (stat_index is not None and idx < stat_index)
        fill = (20, 50, 86, 255) if (active or revealed) else (255, 255, 255, 10)
        outline = (*WHITE, 32) if (active or revealed) else (255, 255, 255, 12)
        draw.rounded_rectangle((84, row_top, WIDTH - 84, row_bottom), radius=18, fill=fill, outline=outline, width=2)
        row_phase = 1.0 if revealed else max(0.0, min(1.0, phase / 0.72))
        row_visible = revealed or active

        if row_visible:
            stat_y = row_center_y - 13 + int((1.0 - row_phase) * 10)
            label_fill = WHITE if active else (220, 229, 239)
            _draw_glow_text(frame, (center_x, stat_y), stat_name, list_font, label_fill, WHITE)
            left_font = value_font_map[left_value]
            right_font = value_font_map[right_value]

            if active and not revealed:
                left_x = int(left_start_x + (left_target_x - left_start_x) * row_phase)
                right_x = int(right_start_x + (right_target_x - right_start_x) * row_phase)
            else:
                left_x = left_target_x
                right_x = right_target_x

            _draw_glow_text(frame, (left_x, stat_y), left_value, left_font, WHITE, LEBRON_BLUE)
            _draw_glow_text(frame, (right_x, stat_y), right_value, right_font, WHITE, JORDAN_RED)
            winner_side = _winner_side(left_value, right_value)

            check_phase = 1.0 if revealed else max(0.0, min(1.0, (phase - 0.56) / 0.34))
            if check_phase > 0:
                if winner_side == "left":
                    x0, y0 = 118, row_top + 29
                    x1, y1 = 128, row_top + 40
                    x2, y2 = 146, row_top + 18
                else:
                    x0, y0 = WIDTH - 146, row_top + 29
                    x1, y1 = WIDTH - 136, row_top + 40
                    x2, y2 = WIDTH - 118, row_top + 18

                if check_phase < 0.5:
                    mid = check_phase / 0.5
                    xe = int(x0 + (x1 - x0) * mid)
                    ye = int(y0 + (y1 - y0) * mid)
                    draw.line((x0, y0, xe, ye), fill=(72, 255, 142, 255), width=6)
                else:
                    draw.line((x0, y0, x1, y1), fill=(72, 255, 142, 255), width=6)
                    mid = (check_phase - 0.5) / 0.5
                    xe = int(x1 + (x2 - x1) * mid)
                    ye = int(y1 + (y2 - y1) * mid)
                    draw.line((x1, y1, xe, ye), fill=(72, 255, 142, 255), width=6)

    if final_board:
        left_total, right_total = _score_totals()
        score_phase = max(0.0, min(1.0, phase / 0.42))
        score_y = 1648
        draw.rounded_rectangle((74, score_y, WIDTH - 74, score_y + 154), radius=30, fill=(10, 28, 54, 220), outline=(255, 255, 255, 18), width=2)
        draw.text((208, score_y + 38), "POINTS", font=_load_font(24, bold=True), fill="#dce8f5", anchor="ma")
        draw.text((WIDTH - 208, score_y + 38), "POINTS", font=_load_font(24, bold=True), fill="#ffe0e4", anchor="ma")
        left_score = int(round(left_total * score_phase))
        right_score = int(round(right_total * score_phase))
        left_score_font = _fit_font_size(draw, str(left_total), 80, 52, 22, bold=True)
        right_score_font = _fit_font_size(draw, str(right_total), 80, 52, 22, bold=True)
        _draw_glow_text(frame, (208, score_y + 102), str(left_score), left_score_font, WHITE, LEBRON_BLUE)
        _draw_glow_text(frame, (WIDTH - 208, score_y + 102), str(right_score), right_score_font, WHITE, JORDAN_RED)
        draw.text((WIDTH // 2, score_y + 102), "SCORE", font=_load_font(30, bold=True), fill="#f4f7fb", anchor="ma")

        stamp_phase = max(0.0, min(1.0, (phase - 0.26) / 0.48))
        if stamp_phase > 0:
            stamp = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
            sd = ImageDraw.Draw(stamp, "RGBA")
            winner_text = _winning_player_name()
            stamp_box = (204, 1116, 878, 1396)
            stamp_red = (154, 18, 26)
            fill_red = (120, 8, 14, int(44 * stamp_phase))
            sd.rounded_rectangle(stamp_box, radius=38, fill=fill_red, outline=(*stamp_red, int(252 * stamp_phase)), width=14)
            sd.rounded_rectangle((224, 1136, 858, 1376), radius=32, outline=(255, 214, 214, int(84 * stamp_phase)), width=3)
            stamp_font = _fit_font_size(sd, winner_text, 560, 118, 40, bold=True)
            sd.text((541, 1256), winner_text, font=stamp_font, fill=(255, 232, 232, int(255 * stamp_phase)), anchor="ma", stroke_width=3, stroke_fill=(*stamp_red, int(255 * stamp_phase)))
            sd.text((541, 1256), winner_text, font=stamp_font, fill=(*stamp_red, int(255 * stamp_phase)), anchor="ma")
            stamp = stamp.rotate(-10, resample=Image.Resampling.BICUBIC, center=(541, 1256))
            frame.alpha_composite(stamp)


def _draw_vote_scene(
    frame: Image.Image,
    images: dict[str, dict[str, Image.Image]],
    title_font: ImageFont.ImageFont,
    name_font: ImageFont.ImageFont,
    phase: float,
) -> None:
    draw = ImageDraw.Draw(frame, "RGBA")
    left_card = (86, 670, 486, 1256)
    right_card = (594, 670, 994, 1256)
    draw.rounded_rectangle(left_card, radius=40, fill=(9, 24, 48, 204), outline=(*LEBRON_BLUE, 156), width=3)
    draw.rounded_rectangle(right_card, radius=40, fill=(46, 10, 20, 204), outline=(*JORDAN_RED, 156), width=3)

    left_image = _make_top_portrait(images["lebron"]["action"], LEBRON_BLUE).resize((210, 210), Image.Resampling.LANCZOS)
    right_image = _make_top_portrait(images["jordan"]["action"], JORDAN_RED).resize((210, 210), Image.Resampling.LANCZOS)
    frame.alpha_composite(left_image, (180, 748))
    frame.alpha_composite(right_image, (688, 748))
    _draw_glow_text(frame, (WIDTH // 2, 564), "PICK YOUR GOAT", title_font, WHITE, WHITE)
    _draw_glow_text(frame, (286, 1030), "LEBRON", name_font, WHITE, LEBRON_BLUE)
    _draw_glow_text(frame, (794, 1030), "JORDAN", name_font, WHITE, JORDAN_RED)

    hand_x = int(660 + 110 * np.sin(phase * np.pi))
    hand_y = int(1180 - 90 * np.sin(phase * np.pi * 0.9))
    points = [
        (hand_x, hand_y),
        (hand_x + 44, hand_y + 30),
        (hand_x + 20, hand_y + 38),
        (hand_x + 50, hand_y + 92),
        (hand_x + 24, hand_y + 104),
        (hand_x - 4, hand_y + 50),
    ]
    draw.polygon(points, fill=(252, 235, 212, 255), outline=(30, 20, 18, 190))

    check_progress = max(0.0, min(1.0, (phase - 0.28) / 0.52))
    if check_progress > 0:
        x0, y0 = 870, 1138
        x1, y1 = 904, 1172
        x2, y2 = 962, 1108
        if check_progress < 0.5:
            mid = check_progress / 0.5
            xe = int(x0 + (x1 - x0) * mid)
            ye = int(y0 + (y1 - y0) * mid)
            draw.line((x0, y0, xe, ye), fill=(72, 255, 142, 255), width=14)
        else:
            draw.line((x0, y0, x1, y1), fill=(72, 255, 142, 255), width=14)
            mid = (check_progress - 0.5) / 0.5
            xe = int(x1 + (x2 - x1) * mid)
            ye = int(y1 + (y2 - y1) * mid)
            draw.line((x1, y1, xe, ye), fill=(72, 255, 142, 255), width=14)

    stamp_phase = max(0.0, min(1.0, (phase - 0.52) / 0.38))
    if stamp_phase > 0:
        stamp = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
        sd = ImageDraw.Draw(stamp, "RGBA")
        stamp_box = (642, 878, 954, 1018)
        sd.rounded_rectangle(stamp_box, radius=26, outline=(255, 72, 72, int(220 * stamp_phase)), width=8)
        sd.rounded_rectangle((650, 886, 946, 1010), radius=22, outline=(255, 72, 72, int(92 * stamp_phase)), width=3)
        stamp_font = _fit_font_size(sd, "JORDAN WINS", 250, 54, 24, bold=True)
        sd.text((798, 949), "JORDAN WINS", font=stamp_font, fill=(255, 92, 92, int(240 * stamp_phase)), anchor="ma")
        stamp = stamp.rotate(-12, resample=Image.Resampling.BICUBIC, center=(798, 949))
        stamp = stamp.filter(ImageFilter.GaussianBlur(radius=max(0.0, 2.0 * (1.0 - stamp_phase))))
        frame.alpha_composite(stamp)
    draw.text((WIDTH // 2, 1360), "COMMENT YOUR PICK", font=_load_font(26, bold=True), fill="#dfe9f3", anchor="ma")


def render_video(output_path: Path, audio_path: Path, duration: float, fps: int) -> Path:
    scenes = _build_scenes(duration)
    background = _make_background()
    images = _load_split_images()
    temp_draw = ImageDraw.Draw(background.copy(), "RGBA")
    side_name_font = _load_font(28, bold=True)
    top_name_font = _load_font(30, bold=True)
    top_sub_font = _load_font(16, bold=True)
    stat_list_font = _load_font(22, bold=True)
    value_font_map = {
        value: _fit_font_size(temp_draw, value, 120, 40, 16, bold=True)
        for row in STAT_ROWS
        for value in (row[1], row[2])
    }

    def scene_at(t: float) -> tuple[dict, float]:
        elapsed = 0.0
        for scene in scenes:
            if t < elapsed + scene["duration"]:
                return scene, (t - elapsed) / scene["duration"]
            elapsed += scene["duration"]
        return scenes[-1], 1.0

    def make_frame(t: float) -> np.ndarray:
        scene, local = scene_at(t)
        phase = _ease_out(local)
        frame = background.copy()
        _apply_energy_overlay(frame, t)
        kind = scene["kind"]
        _draw_top_player_cards(frame, images, top_name_font, top_sub_font)
        glow = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
        gd = ImageDraw.Draw(glow, "RGBA")
        gd.ellipse((-80, 360, 480, 1160), fill=(*LEBRON_BLUE, 42))
        gd.ellipse((WIDTH - 480, 360, WIDTH + 80, 1160), fill=(*JORDAN_RED, 42))
        glow = glow.filter(ImageFilter.GaussianBlur(radius=34))
        frame.alpha_composite(glow)
        _draw_center_stat_scene(
            frame,
            side_name_font,
            stat_list_font,
            value_font_map,
            scene.get("index"),
            phase,
        )
        return np.array(frame.convert("RGB"))

    clip = VideoClip(make_frame, duration=duration)
    audio_clip, keep_alive = build_audio_track(audio_path, duration)
    clip = clip.with_audio(audio_clip)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    run_id = uuid4().hex[:8]
    temp_output = output_path.with_name(f"{output_path.stem}.{run_id}.render.mp4")
    temp_audio = output_path.with_name(f"{output_path.stem}.{run_id}.temp_audio.m4a")
    try:
        clip.write_videofile(
            str(temp_output),
            fps=fps,
            codec="libx264",
            audio_codec="aac",
            temp_audiofile=str(temp_audio),
            remove_temp=False,
        )
        final_path = output_path
        if output_path.exists():
            try:
                output_path.unlink()
            except PermissionError:
                final_path = output_path.with_name(f"{output_path.stem}_{run_id}.mp4")
        temp_output.replace(final_path)
        output_path = final_path
    finally:
        clip.close()
        audio_clip.close()
        for item in keep_alive:
            item.close()
        if temp_output.exists() and temp_output != output_path:
            try:
                temp_output.unlink()
            except OSError:
                pass
        if temp_audio.exists():
            try:
                temp_audio.unlink()
            except OSError:
                pass
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a split-screen LeBron vs Jordan GOAT Shorts video.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--audio", type=Path, default=DEFAULT_AUDIO)
    parser.add_argument("--duration", type=float, default=TOTAL_DURATION)
    parser.add_argument("--fps", type=int, default=FPS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = render_video(args.output, args.audio, args.duration, args.fps)
    print(f"[video_generator] LeBron vs Jordan split Shorts generated -> {output}")


if __name__ == "__main__":
    main()
