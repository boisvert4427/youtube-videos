from __future__ import annotations

import argparse
import json
import unicodedata
from pathlib import Path
from tempfile import NamedTemporaryFile
from uuid import uuid4

import numpy as np
from moviepy import AudioFileClip, CompositeAudioClip, VideoClip
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageOps

from video_generator.generate_ucl_barchart_race_moviepy import (
    DEFAULT_AUDIO,
    _fit_font_size,
    _load_font,
    build_audio_track,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PHOTOS_DIR = PROJECT_ROOT / "data" / "raw" / "player_photos"
DEFAULT_DATA = PROJECT_ROOT / "data" / "processed" / "tennis" / "federer_vs_nadal_stats_shorts_v3.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "tennis" / "federer_vs_nadal_stats_shorts_v3.mp4"

WIDTH = 1080
HEIGHT = 1920
FPS = 30
TOTAL_DURATION = 11.5
MUSIC_VOLUME = 0.42

FEDERER_BLUE = (110, 193, 255)
NADAL_RED = (255, 107, 107)
GOLD = (242, 197, 94)
WHITE = (246, 248, 252)

DATA_FALLBACK = {
    "hook": "NADAL DÉTRUIT FEDERER ICI",
    "left_name": "FEDERER",
    "right_name": "NADAL",
    "left_color": "#6EC1FF",
    "right_color": "#FF6B6B",
    "left_image": "roger_federer.jpg",
    "right_image": "rafael_nadal.jpg",
    "stats_sequence": [
        {"label": "GRAND SLAMS", "left_value": 20, "right_value": 22, "highlight": "close battle"},
        {"label": "HEAD-TO-HEAD", "left_value": 16, "right_value": 24, "highlight": "Nadal leads"},
        {"label": "CLAY TITLES", "left_value": 11, "right_value": 63, "highlight": "Nadal domination"},
        {"label": "GRASS TITLES", "left_value": 19, "right_value": 4, "highlight": "Federer advantage"},
    ],
    "mid_text": "CLAY MONSTER",
    "climax_text": "UNPLAYABLE",
    "ending_text": "REGARDE ENCORE",
}

STAT_WINDOWS = [
    (1.40, 2.20),
    (2.20, 3.00),
    (3.80, 5.10),
    (6.60, 7.80),
]


def _ease_out(value: float) -> float:
    value = min(max(value, 0.0), 1.0)
    return 1.0 - (1.0 - value) ** 3


def _slugify(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    return "_".join(part for part in "".join(ch if ch.isalnum() else "_" for ch in normalized.lower()).split("_") if part)


def _load_data(path: Path | None) -> dict:
    if path is not None and path.exists():
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    return DATA_FALLBACK.copy()


def _load_portrait(path: Path, fallback_name: str) -> Image.Image:
    if path.exists():
        return ImageOps.exif_transpose(Image.open(path)).convert("RGBA")
    img = Image.new("RGBA", (720, 720), (12, 18, 28, 255))
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle((20, 20, 700, 700), radius=60, fill=(22, 30, 44, 255), outline=(255, 255, 255, 70), width=4)
    font = _load_font(62, bold=True)
    draw.text((360, 340), fallback_name[:2].upper(), font=font, fill=WHITE, anchor="mm")
    return img


def _make_background(t: float) -> Image.Image:
    xx = np.linspace(0, 1, WIDTH, dtype=np.float32)
    yy = np.linspace(0, 1, HEIGHT, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(xx, yy)
    base = np.array([6, 10, 18], dtype=np.float32)
    navy = np.array([10, 28, 56], dtype=np.float32)
    left_glow = np.exp(-(((grid_x - 0.20) / 0.20) ** 2 + ((grid_y - 0.35) / 0.24) ** 2))
    right_glow = np.exp(-(((grid_x - 0.80) / 0.20) ** 2 + ((grid_y - 0.35) / 0.24) ** 2))
    center_glow = np.exp(-(((grid_x - 0.50) / 0.30) ** 2 + ((grid_y - 0.56) / 0.24) ** 2))
    pulse = 0.5 + 0.5 * np.sin(t * 1.5)
    img = np.clip(
        base[None, None, :] * (1.0 - 0.88 * grid_y[..., None])
        + navy[None, None, :] * (0.68 * grid_y[..., None])
        + np.array(FEDERER_BLUE, dtype=np.float32)[None, None, :] * (0.18 * left_glow[..., None] * (0.7 + 0.3 * pulse))
        + np.array(NADAL_RED, dtype=np.float32)[None, None, :] * (0.18 * right_glow[..., None] * (0.7 + 0.3 * (1.0 - pulse)))
        + np.array((255, 255, 255), dtype=np.float32)[None, None, :] * (0.03 * center_glow[..., None]),
        0,
        255,
    ).astype(np.uint8)
    frame = Image.fromarray(img, mode="RGB").convert("RGBA")
    overlay = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")
    draw.rounded_rectangle((40, 40, WIDTH - 40, HEIGHT - 40), radius=46, outline=(255, 255, 255, 18), width=2)
    draw.ellipse((60, 170, 460, 920), fill=(*FEDERER_BLUE, 22))
    draw.ellipse((620, 170, 1020, 920), fill=(*NADAL_RED, 20))
    draw.ellipse((260, 860, 820, 1700), fill=(255, 255, 255, 10))
    overlay = overlay.filter(ImageFilter.GaussianBlur(radius=10))
    frame.alpha_composite(overlay)
    return frame


def _make_padded_portrait(source: Image.Image, ring_color: tuple[int, int, int]) -> Image.Image:
    size = 168
    portrait = ImageOps.fit(source, (size, size), method=Image.Resampling.LANCZOS, centering=(0.5, 0.22))
    portrait = ImageEnhance.Brightness(portrait).enhance(1.12)
    portrait = ImageEnhance.Contrast(portrait).enhance(1.12)
    mask = Image.new("L", (size, size), 0)
    ImageDraw.Draw(mask).ellipse((0, 0, size - 1, size - 1), fill=255)
    circle = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    circle.paste(portrait, (0, 0), mask)
    tile = Image.new("RGBA", (size + 24, size + 24), (0, 0, 0, 0))
    td = ImageDraw.Draw(tile, "RGBA")
    td.ellipse((0, 0, size + 23, size + 23), fill=(255, 255, 255, 220))
    td.ellipse((5, 5, size + 18, size + 18), fill=(*ring_color, 255))
    td.ellipse((11, 11, size + 12, size + 12), fill=(8, 18, 34, 255))
    tile.alpha_composite(circle, (12, 12))
    return tile


def _make_half_image(source: Image.Image, side: str, zoom: float = 1.0) -> Image.Image:
    target_w = WIDTH // 2
    target_h = HEIGHT
    centering = (0.42, 0.26) if side == "left" else (0.52, 0.24)
    scaled_w = max(target_w, int(source.width * zoom))
    scaled_h = max(target_h, int(source.height * zoom))
    scaled = ImageOps.contain(source, (scaled_w, scaled_h))
    if scaled.width < target_w or scaled.height < target_h:
        scaled = ImageOps.fit(source, (target_w, target_h), method=Image.Resampling.LANCZOS, centering=centering)
    else:
        scaled = ImageOps.fit(scaled, (target_w, target_h), method=Image.Resampling.LANCZOS, centering=centering)
    scaled = ImageEnhance.Brightness(scaled).enhance(1.18)
    scaled = ImageEnhance.Contrast(scaled).enhance(1.12)
    scaled = ImageEnhance.Color(scaled).enhance(1.06)
    canvas = Image.new("RGBA", (target_w, target_h), (0, 0, 0, 0))
    canvas.alpha_composite(scaled, (0, 0))
    tint = Image.new("RGBA", (target_w, target_h), (*(FEDERER_BLUE if side == "left" else NADAL_RED), 34))
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


def _draw_text(frame: Image.Image, xy: tuple[int, int], text: str, font: ImageFont.ImageFont, fill: tuple[int, int, int]) -> None:
    draw = ImageDraw.Draw(frame, "RGBA")
    draw.text(xy, text, font=font, fill=(*fill, 255), anchor="mm", stroke_width=2, stroke_fill=(6, 10, 20, 220))


def _count_value(value: float | int, progress: float) -> str:
    target = float(value)
    current = target * progress
    if abs(target - round(target)) < 1e-9:
        return str(int(round(current)))
    return f"{current:.1f}" if progress < 0.98 else f"{target:.1f}".rstrip("0").rstrip(".")


def _winner_side(left_value: float, right_value: float) -> str:
    if abs(left_value - right_value) < 1e-9:
        return "tie"
    return "left" if left_value > right_value else "right"


def _score_totals(stats: list[dict]) -> tuple[int, int]:
    left = 0
    right = 0
    for stat in stats:
        winner = _winner_side(float(stat["left_value"]), float(stat["right_value"]))
        if winner == "left":
            left += 1
        elif winner == "right":
            right += 1
    return left, right


def _score_for_time(stats: list[dict], t: float) -> tuple[int, int]:
    left = 0
    right = 0
    for (start, end), stat in zip(STAT_WINDOWS, stats):
        if t >= end:
            winner = _winner_side(float(stat["left_value"]), float(stat["right_value"]))
            if winner == "left":
                left += 1
            elif winner == "right":
                right += 1
    return left, right


def _build_scene_bounds(total_duration: float, stat_count: int) -> list[tuple[float, float | None]]:
    return [
        (0.00, 0.35),  # hook flash
        (0.35, 0.90),  # hook principal
        (0.90, 1.40),  # duel install
        (1.40, 2.20),  # stat 1
        (2.20, 3.00),  # stat 2
        (3.00, 3.80),  # pause
        (3.80, 5.10),  # stat 3
        (5.10, 5.80),  # micro respiration
        (5.80, 6.60),  # renversement
        (6.60, 7.80),  # stat 4
        (7.80, 8.50),  # score final
        (8.50, 9.40),  # climax
        (9.40, 10.20), # twist
        (10.20, total_duration),  # ending loop
    ]


def _composite_split(
    frame: Image.Image,
    images: dict[str, Image.Image],
    zoom: float,
) -> None:
    left = _make_half_image(images["federer"], "left", zoom)
    right = _make_half_image(images["nadal"], "right", zoom)
    frame.alpha_composite(left, (0, 0))
    frame.alpha_composite(right, (WIDTH // 2, 0))


def _draw_top_cards(frame: Image.Image, portraits: dict[str, Image.Image], data: dict, t: float) -> None:
    draw = ImageDraw.Draw(frame, "RGBA")
    left_box = (42, 94, 500, 418)
    right_box = (580, 94, 1038, 418)
    draw.rounded_rectangle(left_box, radius=40, fill=(26, 34, 54, 225), outline=(*FEDERER_BLUE, 160), width=2)
    draw.rounded_rectangle(right_box, radius=40, fill=(52, 22, 24, 225), outline=(*NADAL_RED, 160), width=2)
    left_portrait = _make_padded_portrait(portraits["federer"], FEDERER_BLUE)
    right_portrait = _make_padded_portrait(portraits["nadal"], NADAL_RED)
    frame.alpha_composite(left_portrait, (74, 146))
    frame.alpha_composite(right_portrait, (612, 146))
    _draw_text(frame, (354, 212), data["left_name"], _load_font(30, bold=True), WHITE)
    _draw_text(frame, (892, 212), data["right_name"], _load_font(30, bold=True), WHITE)
    _draw_text(frame, (354, 264), "ROGER", _load_font(18, bold=True), (214, 226, 239))
    _draw_text(frame, (892, 264), "RAFAEL", _load_font(18, bold=True), (214, 226, 239))


def _draw_hook(frame: Image.Image, data: dict, t: float) -> None:
    phase = _ease_out(min(t / 1.15, 1.0))
    scale = 1.0 + 0.02 * np.sin(t * 2.2)
    title_font = _fit_font_size(ImageDraw.Draw(frame), data["hook"], 900, 60, 24, bold=True)
    subtitle_font = _load_font(20, bold=True)
    title_img = Image.new("RGBA", (WIDTH, 180), (0, 0, 0, 0))
    td = ImageDraw.Draw(title_img, "RGBA")
    td.text((WIDTH // 2, 58), data["hook"], font=title_font, fill=(255, 246, 226, int(255 * phase)), anchor="ma", stroke_width=3, stroke_fill=(0, 0, 0, int(190 * phase)))
    td.text((WIDTH // 2, 100), "FEDERER VS NADAL", font=subtitle_font, fill=(214, 226, 239, int(220 * phase)), anchor="ma")
    if scale != 1.0:
        title_img = title_img.resize((int(WIDTH * scale), int(180 * scale)), Image.Resampling.LANCZOS)
        x = (title_img.width - WIDTH) // 2
        y = (title_img.height - 180) // 2
        title_img = title_img.crop((x, y, x + WIDTH, y + 180))
    frame.alpha_composite(title_img, (0, 0))


def _draw_stat_scene(
    frame: Image.Image,
    stat: dict,
    stat_index: int,
    progress: float,
    data: dict,
    score: tuple[int, int],
    final_board: bool = False,
) -> None:
    draw = ImageDraw.Draw(frame, "RGBA")
    panel = (32, 470, WIDTH - 32, 1668)
    draw.rounded_rectangle(panel, radius=48, fill=(7, 18, 36, 220), outline=(255, 255, 255, 18), width=2)
    _draw_text(frame, (170, 540), data["left_name"], _load_font(24, bold=True), (234, 240, 247))
    _draw_text(frame, (WIDTH // 2, 540), "STATS", _load_font(24, bold=True), (234, 240, 247))
    _draw_text(frame, (WIDTH - 170, 540), data["right_name"], _load_font(24, bold=True), (234, 240, 247))

    score_x, score_y = WIDTH // 2, 458
    draw.rounded_rectangle((406, 424, 674, 512), radius=28, fill=(10, 24, 43, 235), outline=(255, 214, 140, 130), width=2)
    _draw_text(frame, (482, 467), str(score[0]), _load_font(34, bold=True), FEDERER_BLUE)
    _draw_text(frame, (538, 467), "-", _load_font(28, bold=True), WHITE)
    _draw_text(frame, (594, 467), str(score[1]), _load_font(34, bold=True), NADAL_RED)

    stat_name = stat["label"]
    left_value = float(stat["left_value"])
    right_value = float(stat["right_value"])
    winner = _winner_side(left_value, right_value)
    reveal = _ease_out(progress)
    row_top = 604
    row_gap = 102
    active_y = row_top + stat_index * row_gap
    for idx, s in enumerate(data["stats_sequence"]):
        y = row_top + idx * row_gap
        active = idx == stat_index
        passed = idx < stat_index
        base_fill = (20, 50, 86, 220) if (passed or active or final_board) else (255, 255, 255, 14)
        outline = (255, 255, 255, 36) if (passed or active or final_board) else (255, 255, 255, 12)
        draw.rounded_rectangle((88, y, WIDTH - 88, y + 72), radius=20, fill=base_fill, outline=outline, width=2)

        label_font = _load_font(22, bold=True)
        _draw_text(frame, (WIDTH // 2, y + 22), s["label"], label_font, (170, 188, 204) if not active else WHITE)

        if passed or active or final_board:
            l = float(s["left_value"])
            r = float(s["right_value"])
            total = max(l, r, 1.0)
            bar_len = int(310 * (0.2 + 0.8 * reveal))
            left_len = int(bar_len * (l / total))
            right_len = int(bar_len * (r / total))
            bar_y = y + 52
            draw.rounded_rectangle((266, bar_y - 6, 266 + 310, bar_y + 6), radius=6, fill=(255, 255, 255, 20))
            draw.rounded_rectangle((506, bar_y - 6, 506 + 310, bar_y + 6), radius=6, fill=(255, 255, 255, 20))
            draw.rounded_rectangle((266 + 310 - left_len, bar_y - 6, 266 + 310, bar_y + 6), radius=6, fill=(*FEDERER_BLUE, 230))
            draw.rounded_rectangle((506, bar_y - 6, 506 + right_len, bar_y + 6), radius=6, fill=(*NADAL_RED, 230))

            left_text = _count_value(l, 1.0 if passed or final_board else reveal)
            right_text = _count_value(r, 1.0 if passed or final_board else reveal)
            left_font = _fit_font_size(draw, left_text, 120, 38, 20, bold=True)
            right_font = _fit_font_size(draw, right_text, 120, 38, 20, bold=True)
            _draw_text(frame, (214, y + 36), left_text, left_font, WHITE)
            _draw_text(frame, (WIDTH - 214, y + 36), right_text, right_font, WHITE)

            if winner != "tie":
                if winner == "left":
                    draw.line((128, y + 40, 144, y + 54), fill=(88, 255, 152, 255), width=6)
                    draw.line((144, y + 54, 168, y + 28), fill=(88, 255, 152, 255), width=6)
                else:
                    draw.line((WIDTH - 168, y + 40, WIDTH - 152, y + 54), fill=(88, 255, 152, 255), width=6)
                    draw.line((WIDTH - 152, y + 54, WIDTH - 128, y + 28), fill=(88, 255, 152, 255), width=6)

    if final_board:
        draw.rounded_rectangle((156, 1208, 924, 1498), radius=42, fill=(120, 10, 14, 140), outline=(192, 50, 58, 240), width=14)
        stamp = "NADAL WINS" if score[1] > score[0] else "FEDERER WINS" if score[0] > score[1] else "YOU DECIDE"
        stamp_font = _fit_font_size(draw, stamp, 620, 110, 36, bold=True)
        _draw_text(frame, (540, 1360), stamp, stamp_font, (255, 236, 236))
        _draw_text(frame, (540, 1604), "REGARDE ENCORE", _load_font(28, bold=True), WHITE)


def _build_narration_text(data: dict) -> str:
    return (
        f"{data['hook']}. "
        "Federer contre Nadal. "
        "Grand slams, face à face, terre battue, gazon. "
        f"{data['climax_text']}. "
        f"{data['ending_text']}."
    )


def render_video(output_path: Path, audio_path: Path, data_path: Path | None, duration: float, fps: int) -> Path:
    data = _load_data(data_path)
    stats = data["stats_sequence"]
    background_frames = {i: _make_background(i * 0.25) for i in range(4)}
    portraits = {
        "federer": _load_portrait(PHOTOS_DIR / data["left_image"], "RF"),
        "nadal": _load_portrait(PHOTOS_DIR / data["right_image"], "RN"),
    }
    temp_draw = ImageDraw.Draw(Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0)))
    value_font_map = {}
    for stat in stats:
        for key in ("left_value", "right_value"):
            value = stat[key]
            if value not in value_font_map:
                txt = str(value) if isinstance(value, int) or float(value).is_integer() else f"{float(value):.1f}".rstrip("0").rstrip(".")
                value_font_map[value] = _fit_font_size(temp_draw, txt, 120, 40, 18, bold=True)

    def stat_state(t: float) -> tuple[int | None, float, bool]:
        if t < 1.40:
            return None, 0.0, False
        stat_windows = [
            (1.40, 2.20),
            (2.20, 3.00),
            (3.80, 5.10),
            (6.60, 7.80),
        ]
        for idx, (start, end) in enumerate(stat_windows):
            if start <= t < end:
                return idx, (t - start) / (end - start), False
        if t < 7.80:
            return 2, 1.0, False
        return 3, 1.0, t >= 10.20

    def make_frame(t: float) -> np.ndarray:
        stat_index, phase, final_board = stat_state(t)
        bg = background_frames[int((t * 2.0) % 4)].copy()
        frame = bg
        if t < 0.35:
            flash = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
            fd = ImageDraw.Draw(flash, "RGBA")
            fd.rectangle((0, 0, WIDTH, HEIGHT), fill=(0, 0, 0, 220))
            fd.text((WIDTH // 2, 930), "FEDERER vs NADAL", font=_fit_font_size(fd, "FEDERER vs NADAL", 940, 74, 28, bold=True), fill=(255, 248, 232, 255), anchor="mm", stroke_width=3, stroke_fill=(0, 0, 0, 200))
            frame.alpha_composite(flash)
            return np.array(frame.convert("RGB"))

        if t < 0.90:
            _draw_hook(frame, data, t)
            _draw_top_cards(frame, portraits, data, t)
        else:
            _draw_top_cards(frame, portraits, data, t)
            if t < 1.40:
                install = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
                idd = ImageDraw.Draw(install, "RGBA")
                idd.text((WIDTH // 2, 424), "VS", font=_load_font(36, bold=True), fill=GOLD, anchor="mm")
                idd.rounded_rectangle((332, 450, 748, 508), radius=24, fill=(10, 24, 43, 224), outline=(255, 255, 255, 26), width=2)
                idd.text((540, 478), "4 STATS. 1 VAINQUEUR.", font=_load_font(20, bold=True), fill=(230, 238, 246, 255), anchor="mm")
                if t > 0.90:
                    alpha = _ease_out((t - 0.90) / 0.50)
                    idd.text((540, 430), "0 - 0", font=_load_font(30, bold=True), fill=(255, 246, 226, int(255 * alpha)), anchor="mm")
                frame.alpha_composite(install)
            score = _score_for_time(stats, t)
            if stat_index is not None and stat_index < len(stats):
                current = stats[stat_index]
                _draw_stat_scene(frame, current, stat_index, phase, data, score, final_board=final_board)
            elif t >= 8.50:
                _draw_stat_scene(frame, stats[-1], len(stats) - 1, 1.0, data, score, final_board=final_board)
        if 3.00 <= t < 3.80:
            pause = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
            pd = ImageDraw.Draw(pause, "RGBA")
            pd.text((WIDTH // 2, 785), "ET SUR TERRE BATTUE...", font=_load_font(34, bold=True), fill=WHITE, anchor="mm")
            frame.alpha_composite(pause)
        if 5.80 <= t < 6.60:
            twist = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
            td = ImageDraw.Draw(twist, "RGBA")
            td.text((WIDTH // 2, 790), "MAIS SUR GAZON...", font=_load_font(34, bold=True), fill=(230, 238, 246, 255), anchor="mm")
            frame.alpha_composite(twist)
        if 8.50 <= t < 9.40:
            climax = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
            cd = ImageDraw.Draw(climax, "RGBA")
            cd.rounded_rectangle((228, 1224, 852, 1422), radius=36, fill=(150, 18, 26, 185), outline=(255, 122, 122, 210), width=10)
            cd.text((540, 1318), "NADAL WINS THIS ONE", font=_fit_font_size(cd, "NADAL WINS THIS ONE", 540, 74, 28, bold=True), fill=(255, 236, 236, 255), anchor="mm")
            frame.alpha_composite(climax)
        if 9.40 <= t < 10.20:
            twist2 = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
            tw = ImageDraw.Draw(twist2, "RGBA")
            tw.text((WIDTH // 2, 1508), "MAIS EST-CE ASSEZ POUR LE GOAT ?", font=_load_font(28, bold=True), fill=(243, 247, 252, 255), anchor="mm")
            frame.alpha_composite(twist2)
        if t >= 10.20:
            end = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
            ed = ImageDraw.Draw(end, "RGBA")
            ed.rounded_rectangle((244, 1560, 836, 1708), radius=32, fill=(10, 24, 43, 222), outline=(255, 255, 255, 24), width=2)
            ed.text((540, 1632), "TU CHOISIS QUI ?", font=_load_font(30, bold=True), fill=(255, 246, 226, 255), anchor="mm")
            frame.alpha_composite(end)
        return np.array(frame.convert("RGB"))

    clip = VideoClip(make_frame, duration=duration)
    audio_clip, keep_alive = build_audio_track(audio_path, duration)
    composite_audio = CompositeAudioClip([audio_clip.with_volume_scaled(MUSIC_VOLUME)]).with_duration(duration)
    clip = clip.with_audio(composite_audio)
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
            bitrate="12000k",
            preset="slow",
            temp_audiofile=str(temp_audio),
            remove_temp=False,
        )
        if output_path.exists():
            try:
                output_path.unlink()
            except PermissionError:
                output_path = output_path.with_name(f"{output_path.stem}_{run_id}.mp4")
        temp_output.replace(output_path)
    finally:
        clip.close()
        composite_audio.close()
        audio_clip.close()
        for item in keep_alive:
            item.close()
        if temp_audio.exists():
            try:
                temp_audio.unlink()
            except OSError:
                pass
        if temp_output.exists() and temp_output != output_path:
            try:
                temp_output.unlink()
            except OSError:
                pass
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a fresh Federer vs Nadal viral Shorts video.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--audio", type=Path, default=DEFAULT_AUDIO)
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument("--duration", type=float, default=TOTAL_DURATION)
    parser.add_argument("--fps", type=int, default=FPS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = render_video(args.output, args.audio, args.data, args.duration, args.fps)
    print(f"[video_generator] Federer vs Nadal fresh Shorts generated -> {output}")


if __name__ == "__main__":
    main()
