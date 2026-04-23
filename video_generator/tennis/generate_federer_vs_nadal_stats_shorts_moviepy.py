from __future__ import annotations

import argparse
import subprocess
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
PIPER_MODEL_DIR = PROJECT_ROOT / "data" / "raw" / "tts_models" / "piper" / "fr_FR_siwis_medium"
PIPER_MODEL = PIPER_MODEL_DIR / "fr_FR-siwis-medium.onnx"
PIPER_CONFIG = PIPER_MODEL_DIR / "fr_FR-siwis-medium.onnx.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "tennis" / "federer_vs_nadal_stats_shorts_v2.mp4"

WIDTH = 1080
HEIGHT = 1920
FPS = 30
TOTAL_DURATION = 11.5
VOICE_VOLUME = 1.05
MUSIC_VOLUME = 0.38

FEDERER_RED = (152, 43, 64)
NADAL_ORANGE = (238, 123, 41)
WHITE = (246, 248, 252)

STAT_ROWS = [
    ("GRAND SLAMS", "20", "22"),
    ("HEAD-TO-HEAD", "16", "24"),
    ("CLAY TITLES", "11", "63"),
    ("GRASS TITLES", "19", "4"),
]
STAT_REVEAL_DURATION = 0.95
NARRATION_RATE = -1
NARRATION_VOICE = "Microsoft Hortense Desktop"


def _slugify(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    return "_".join(part for part in "".join(ch if ch.isalnum() else "_" for ch in normalized.lower()).split("_") if part)


def _ease_out(value: float) -> float:
    value = min(max(value, 0.0), 1.0)
    return 1.0 - (1.0 - value) ** 3


def _make_background() -> Image.Image:
    xx = np.linspace(0, 1, WIDTH, dtype=np.float32)
    yy = np.linspace(0, 1, HEIGHT, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(xx, yy)
    black = np.array([5, 8, 16], dtype=np.float32)
    navy = np.array([10, 27, 56], dtype=np.float32)
    red = np.array(FEDERER_RED, dtype=np.float32)
    orange = np.array(NADAL_ORANGE, dtype=np.float32)
    white = np.array(WHITE, dtype=np.float32)

    mix = np.clip(0.56 * grid_y + 0.12 * grid_x, 0, 1)
    left_glow = np.exp(-(((grid_x - 0.18) / 0.18) ** 2 + ((grid_y - 0.45) / 0.22) ** 2))
    right_glow = np.exp(-(((grid_x - 0.82) / 0.18) ** 2 + ((grid_y - 0.45) / 0.22) ** 2))
    center_glow = np.exp(-(((grid_x - 0.5) / 0.28) ** 2 + ((grid_y - 0.52) / 0.20) ** 2))
    img = np.clip(
        black[None, None, :] * (1.0 - mix[..., None])
        + navy[None, None, :] * (0.82 * mix[..., None])
        + red[None, None, :] * (0.16 * left_glow[..., None])
        + orange[None, None, :] * (0.16 * right_glow[..., None])
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


def _load_portraits() -> dict[str, Image.Image]:
    mapping = {
        "federer": PHOTOS_DIR / "roger_federer.jpg",
        "nadal": PHOTOS_DIR / "rafael_nadal.jpg",
    }
    cache: dict[str, Image.Image] = {}
    for key, path in mapping.items():
        img = ImageOps.exif_transpose(Image.open(path)).convert("RGBA")
        cache[key] = img
    return cache


def _make_top_portrait(source: Image.Image, ring_color: tuple[int, int, int]) -> Image.Image:
    size = 168
    portrait = ImageOps.fit(source, (size, size), method=Image.Resampling.LANCZOS, centering=(0.5, 0.22))
    portrait = ImageEnhance.Brightness(portrait).enhance(1.1)
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


def _draw_glow_text(
    frame: Image.Image,
    pos: tuple[int, int],
    text: str,
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int],
    glow: tuple[int, int, int],
    anchor: str = "ma",
) -> None:
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
    pulse = 0.5 + 0.5 * np.sin(t * 2.4)
    pulse2 = 0.5 + 0.5 * np.sin(t * 1.9 + 1.0)
    draw.ellipse((-120, 180, 380, 860), fill=(*FEDERER_RED, int(36 + 34 * pulse)))
    draw.ellipse((700, 160, 1200, 860), fill=(*NADAL_ORANGE, int(36 + 34 * pulse2)))
    draw.ellipse((240, 980, 840, 1760), fill=(255, 255, 255, int(10 + 14 * pulse)))
    frame.alpha_composite(overlay)


def _winner_side(left_value: str, right_value: str) -> str:
    left_score = float(left_value.replace(",", ""))
    right_score = float(right_value.replace(",", ""))
    if abs(left_score - right_score) < 1e-9:
        return "tie"
    return "left" if left_score > right_score else "right"


def _score_totals() -> tuple[int, int]:
    left_total = 0
    right_total = 0
    for _, left_value, right_value in STAT_ROWS:
        winner = _winner_side(left_value, right_value)
        if winner == "left":
            left_total += 1
        elif winner == "right":
            right_total += 1
    return left_total, right_total


def _build_scenes(total_duration: float) -> list[dict]:
    reveal_total = len(STAT_ROWS) * STAT_REVEAL_DURATION
    final_duration = max(5.0, total_duration - reveal_total)
    return [
        *[
            {"kind": "stat", "duration": STAT_REVEAL_DURATION, "index": idx}
            for idx in range(len(STAT_ROWS))
        ],
        {"kind": "board_final", "duration": final_duration},
    ]


def _final_stamp_text() -> str:
    left_total, right_total = _score_totals()
    if left_total == right_total:
        return "YOU DECIDE"
    return "FEDERER WINS" if left_total > right_total else "NADAL WINS"


def _winner_word(left_value: str, right_value: str) -> str:
    winner = _winner_side(left_value, right_value)
    if winner == "left":
        return "Federer"
    if winner == "right":
        return "Nadal"
    return "Tie"


def _build_narration_text() -> str:
    final_text = _final_stamp_text()
    ending = (
        "Et au final, Federer gagne ce duel."
        if final_text == "FEDERER WINS"
        else "Et au final, Nadal gagne ce duel."
        if final_text == "NADAL WINS"
        else "Et au final, c'est à vous de décider."
    )
    return (
        "Federer contre Nadal. "
        "Federer domine à Wimbledon, à l'Open d'Australie, au Masters et en semaines numéro un. "
        "Nadal écrase Roland Garros, les Masters 1000, le face à face et le total de Grands Chelems. "
        "Le pourcentage de victoires est presque identique, et les saisons terminées numéro un sont à égalité. "
        f"{ending}"
    )


def _build_short_narration_text() -> str:
    final_text = _final_stamp_text()
    ending = (
        "Federer gagne."
        if final_text == "FEDERER WINS"
        else "Nadal gagne."
        if final_text == "NADAL WINS"
        else "À vous de voir."
    )
    return (
        "Nadal est injouable ici. "
        "Federer contre Nadal. "
        "Grand slams, face à face, terre battue, gazon. "
        "Regarde encore. "
        f"{ending}"
    )


def _synthesize_voiceover(text: str) -> Path | None:
    with NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        target_path = Path(tmp.name)

    if PIPER_MODEL.exists() and PIPER_CONFIG.exists():
        try:
            subprocess.run(
                [
                    "python",
                    "-m",
                    "piper",
                    "--model",
                    str(PIPER_MODEL),
                    "--config",
                    str(PIPER_CONFIG),
                    "--output_file",
                    str(target_path),
                    "--length_scale",
                    "1.08",
                    "--noise_scale",
                    "0.55",
                    "--noise_w_scale",
                    "0.75",
                    "--sentence_silence",
                    "0.12",
                ],
                input=text,
                text=True,
                check=True,
                capture_output=True,
            )
            return target_path
        except (subprocess.CalledProcessError, FileNotFoundError):
            try:
                target_path.unlink(missing_ok=True)
            except OSError:
                pass
            with NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                target_path = Path(tmp.name)

    escaped_text = text.replace("'", "''")
    escaped_output = str(target_path).replace("'", "''")
    command = (
        "Add-Type -AssemblyName System.Speech; "
        "$speak = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
        f"$voice = '{NARRATION_VOICE}'; "
        "if (($speak.GetInstalledVoices() | ForEach-Object { $_.VoiceInfo.Name }) -contains $voice) { $speak.SelectVoice($voice) }; "
        f"$speak.Rate = {NARRATION_RATE}; "
        "$speak.Volume = 100; "
        f"$text = '{escaped_text}'; "
        f"$out = '{escaped_output}'; "
        "$speak.SetOutputToWaveFile($out); "
        "$speak.Speak($text); "
        "$speak.Dispose();"
    )

    try:
        subprocess.run(["powershell", "-Command", command], check=True, capture_output=True, text=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        try:
            target_path.unlink(missing_ok=True)
        except OSError:
            pass
        return None
    return target_path


def _draw_top_player_cards(
    frame: Image.Image,
    portraits: dict[str, Image.Image],
    title_font: ImageFont.ImageFont,
    sub_font: ImageFont.ImageFont,
) -> None:
    draw = ImageDraw.Draw(frame, "RGBA")
    left_box = (42, 104, 500, 422)
    right_box = (580, 104, 1038, 422)
    draw.rounded_rectangle(left_box, radius=42, fill=(46, 10, 20, 222), outline=(*FEDERER_RED, 168), width=2)
    draw.rounded_rectangle(right_box, radius=42, fill=(64, 28, 10, 222), outline=(*NADAL_ORANGE, 168), width=2)

    left_portrait = _make_top_portrait(portraits["federer"], FEDERER_RED)
    right_portrait = _make_top_portrait(portraits["nadal"], NADAL_ORANGE)
    frame.alpha_composite(left_portrait, (74, 146))
    frame.alpha_composite(right_portrait, (612, 146))

    _draw_glow_text(frame, (354, 214), "FEDERER", title_font, WHITE, FEDERER_RED)
    _draw_glow_text(frame, (892, 214), "NADAL", title_font, WHITE, NADAL_ORANGE)
    draw.text((354, 266), "ROGER", font=sub_font, fill="#ffe5ea", anchor="ma")
    draw.text((892, 266), "RAFAEL", font=sub_font, fill="#ffe7d7", anchor="ma")


def _draw_presenter_overlay(
    frame: Image.Image,
    t: float,
    voice_active: bool,
    title_font: ImageFont.ImageFont,
    sub_font: ImageFont.ImageFont,
) -> None:
    box = (722, 1468, 1032, 1758)
    bubble = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    draw = ImageDraw.Draw(bubble, "RGBA")
    draw.rounded_rectangle(box, radius=34, fill=(10, 24, 43, 218), outline=(255, 255, 255, 30), width=2)
    draw.rounded_rectangle((758, 1714, 994, 1752), radius=16, fill=(24, 46, 76, 235))
    draw.text((876, 1734), "VOIX OFF", font=sub_font, fill="#dcecff", anchor="mm")

    face_center = (877, 1566)
    face_radius = 78
    pulse = 0.92 + 0.08 * np.sin(t * 6.5)
    glow_alpha = 46 + int(26 * max(0.0, np.sin(t * 5.0)))
    draw.ellipse(
        (
            face_center[0] - face_radius - 24,
            face_center[1] - face_radius - 24,
            face_center[0] + face_radius + 24,
            face_center[1] + face_radius + 24,
        ),
        fill=(82, 190, 255, glow_alpha),
    )
    head_box = (
        face_center[0] - face_radius,
        face_center[1] - face_radius,
        face_center[0] + face_radius,
        face_center[1] + face_radius,
    )
    head = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    hd = ImageDraw.Draw(head, "RGBA")
    hd.ellipse(head_box, fill=(236, 205, 178, 255), outline=(255, 255, 255, 210), width=3)
    hd.ellipse(
        (
            face_center[0] - 60,
            face_center[1] - 70,
            face_center[0] + 60,
            face_center[1] + 4,
        ),
        fill=(58, 39, 31, 255),
    )
    hd.pieslice(
        (
            face_center[0] - 70,
            face_center[1] - 74,
            face_center[0] + 70,
            face_center[1] + 22,
        ),
        start=180,
        end=360,
        fill=(54, 36, 29, 255),
    )
    hd.ellipse((face_center[0] - 44, face_center[1] - 6, face_center[0] - 24, face_center[1] + 12), fill=(34, 38, 44, 255))
    hd.ellipse((face_center[0] + 24, face_center[1] - 6, face_center[0] + 44, face_center[1] + 12), fill=(34, 38, 44, 255))
    hd.line((face_center[0] - 48, face_center[1] - 22, face_center[0] - 16, face_center[1] - 16), fill=(75, 56, 47, 255), width=5)
    hd.line((face_center[0] + 16, face_center[1] - 16, face_center[0] + 48, face_center[1] - 22), fill=(75, 56, 47, 255), width=5)
    hd.line((face_center[0], face_center[1] + 2, face_center[0] - 5, face_center[1] + 28), fill=(154, 118, 98, 255), width=3)
    hd.arc((face_center[0] - 26, face_center[1] + 24, face_center[0] + 26, face_center[1] + 58), start=10, end=170, fill=(138, 86, 76, 255), width=4)
    head = head.filter(ImageFilter.GaussianBlur(radius=0.3))
    frame.alpha_composite(head)

    mouth_width = 20 if not voice_active else int(26 + 12 * max(0.0, np.sin(t * 19.0)) * pulse)
    mouth_height = 7 if not voice_active else int(12 + 6 * max(0.0, np.sin(t * 17.0 + 1.1)))
    draw.rounded_rectangle(
        (
            face_center[0] - mouth_width,
            face_center[1] + 36,
            face_center[0] + mouth_width,
            face_center[1] + 36 + mouth_height,
        ),
        radius=8,
        fill=(145, 52, 62, 255),
    )
    draw.rectangle((829, 1642, 925, 1692), fill=(32, 70, 114, 255))
    draw.polygon([(925, 1642), (967, 1666), (925, 1692)], fill=(32, 70, 114, 255))
    draw.rectangle((848, 1694, 906, 1740), fill=(16, 28, 48, 255))
    draw.rectangle((860, 1588, 894, 1646), fill=(16, 28, 48, 255))
    draw.ellipse((846, 1558, 908, 1612), fill=(40, 47, 58, 255))
    draw.text((876, 1492), "PRESENTATEUR", font=title_font, fill="#f5fbff", anchor="mm")
    frame.alpha_composite(bubble)


def _draw_center_stat_scene(
    frame: Image.Image,
    small_font: ImageFont.ImageFont,
    list_font: ImageFont.ImageFont,
    value_font_map: dict[str, ImageFont.ImageFont],
    stat_index: int | None,
    phase: float,
) -> None:
    draw = ImageDraw.Draw(frame, "RGBA")
    panel = (34, 476, WIDTH - 34, 1658)
    draw.rounded_rectangle(panel, radius=48, fill=(7, 18, 36, 220), outline=(255, 255, 255, 20), width=2)
    draw.text((170, 540), "FEDERER", font=small_font, fill="#ffe3ea", anchor="ma")
    draw.text((WIDTH // 2, 540), "STATS", font=small_font, fill="#f4f7fb", anchor="ma")
    draw.text((WIDTH - 170, 540), "NADAL", font=small_font, fill="#ffe5d8", anchor="ma")

    left_seen = 0
    right_seen = 0
    for idx, (_, left_value, right_value) in enumerate(STAT_ROWS):
        if stat_index is None or idx < stat_index or (idx == stat_index and phase > 0.72):
            winner = _winner_side(left_value, right_value)
            if winner == "left":
                left_seen += 1
            elif winner == "right":
                right_seen += 1
    score_box = (386, 438, 694, 506)
    draw.rounded_rectangle(score_box, radius=28, fill=(10, 24, 43, 235), outline=(255, 215, 140, 128), width=2)
    draw.text((488, 472), str(left_seen), font=_load_font(34, bold=True), fill="#dfefff", anchor="mm")
    draw.text((540, 472), "-", font=_load_font(26, bold=True), fill="#ffe3ea", anchor="mm")
    draw.text((592, 472), str(right_seen), font=_load_font(34, bold=True), fill="#ffe8d7", anchor="mm")

    left_target_x = 238
    center_x = WIDTH // 2
    right_target_x = WIDTH - 238
    left_start_x = 238
    right_start_x = WIDTH - 238
    final_board = stat_index is None

    row_h = 76
    for idx, (stat_name, left_value, right_value) in enumerate(STAT_ROWS):
        row_top = 592 + idx * row_h
        row_bottom = row_top + 70
        row_center_y = (row_top + row_bottom) // 2
        active = stat_index is not None and idx == stat_index
        revealed = final_board or (stat_index is not None and idx < stat_index)
        fill = (20, 50, 86, 255) if (active or revealed) else (255, 255, 255, 16)
        outline = (*WHITE, 36) if (active or revealed) else (255, 255, 255, 12)
        draw.rounded_rectangle((84, row_top, WIDTH - 84, row_bottom), radius=18, fill=fill, outline=outline, width=2)
        row_phase = 1.0 if revealed else max(0.0, min(1.0, phase / 0.72))
        row_visible = True

        if row_visible:
            stat_y = row_center_y - 14 + int((1.0 - row_phase) * 12)
            label_fill = WHITE if active else (175, 188, 204)
            _draw_glow_text(frame, (center_x, stat_y), stat_name, list_font, label_fill, WHITE)
            left_font = value_font_map[left_value]
            right_font = value_font_map[right_value]

            if active and not revealed:
                left_x = int(left_start_x + (left_target_x - left_start_x) * row_phase)
                right_x = int(right_start_x + (right_target_x - right_start_x) * row_phase)
            else:
                left_x = left_target_x
                right_x = right_target_x

            _draw_glow_text(frame, (left_x, stat_y), left_value, left_font, WHITE, FEDERER_RED)
            _draw_glow_text(frame, (right_x, stat_y), right_value, right_font, WHITE, NADAL_ORANGE)
            winner_side = _winner_side(left_value, right_value)

            left_num = float(left_value.replace(",", ""))
            right_num = float(right_value.replace(",", ""))
            bar_base = 176
            bar_max = 276
            left_bar = int(bar_max * row_phase * (left_num / max(left_num, right_num, 1.0)))
            right_bar = int(bar_max * row_phase * (right_num / max(left_num, right_num, 1.0)))
            bar_y = row_center_y + 26
            draw.rounded_rectangle((center_x - bar_base - bar_max, bar_y - 6, center_x - bar_base, bar_y + 6), radius=6, fill=(255, 255, 255, 18))
            draw.rounded_rectangle((center_x + bar_base, bar_y - 6, center_x + bar_base + bar_max, bar_y + 6), radius=6, fill=(255, 255, 255, 18))
            draw.rounded_rectangle((center_x - bar_base - left_bar, bar_y - 6, center_x - bar_base, bar_y + 6), radius=6, fill=(*FEDERER_RED, 220))
            draw.rounded_rectangle((center_x + bar_base, bar_y - 6, center_x + bar_base + right_bar, bar_y + 6), radius=6, fill=(*NADAL_ORANGE, 220))
            if active:
                glow_color = (255, 225, 160, 90)
                if winner_side == "left":
                    draw.ellipse((left_x - 54, stat_y - 22, left_x + 54, stat_y + 34), fill=glow_color)
                elif winner_side == "right":
                    draw.ellipse((right_x - 54, stat_y - 22, right_x + 54, stat_y + 34), fill=glow_color)

            check_phase = 1.0 if revealed else max(0.0, min(1.0, (phase - 0.56) / 0.34))
            if check_phase > 0 and winner_side != "tie":
                if winner_side == "left":
                    x0, y0 = 118, row_top + 34
                    x1, y1 = 128, row_top + 46
                    x2, y2 = 148, row_top + 20
                else:
                    x0, y0 = WIDTH - 148, row_top + 34
                    x1, y1 = WIDTH - 138, row_top + 46
                    x2, y2 = WIDTH - 118, row_top + 20

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
        score_y = 1714
        draw.rounded_rectangle((74, score_y, WIDTH - 74, score_y + 154), radius=30, fill=(10, 28, 54, 220), outline=(255, 255, 255, 18), width=2)
        draw.text((208, score_y + 38), "POINTS", font=_load_font(24, bold=True), fill="#ffe3ea", anchor="ma")
        draw.text((WIDTH - 208, score_y + 38), "POINTS", font=_load_font(24, bold=True), fill="#ffe5d8", anchor="ma")
        left_score = int(round(left_total * score_phase))
        right_score = int(round(right_total * score_phase))
        left_score_font = _fit_font_size(draw, str(left_total), 80, 52, 22, bold=True)
        right_score_font = _fit_font_size(draw, str(right_total), 80, 52, 22, bold=True)
        _draw_glow_text(frame, (208, score_y + 102), str(left_score), left_score_font, WHITE, FEDERER_RED)
        _draw_glow_text(frame, (WIDTH - 208, score_y + 102), str(right_score), right_score_font, WHITE, NADAL_ORANGE)
        draw.text((WIDTH // 2, score_y + 102), "SCORE", font=_load_font(30, bold=True), fill="#f4f7fb", anchor="ma")

        stamp_phase = max(0.0, min(1.0, (phase - 0.22) / 0.50))
        if stamp_phase > 0:
            stamp = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
            sd = ImageDraw.Draw(stamp, "RGBA")
            stamp_text = _final_stamp_text()
            stamp_box = (168, 1226, 912, 1498)
            stamp_red = (155, 30, 42)
            fill_red = (120, 10, 14, int(44 * stamp_phase))
            sd.rounded_rectangle(stamp_box, radius=38, fill=fill_red, outline=(*stamp_red, int(252 * stamp_phase)), width=14)
            sd.rounded_rectangle((198, 1180, 882, 1402), radius=32, outline=(255, 214, 214, int(84 * stamp_phase)), width=3)
            stamp_font = _fit_font_size(sd, stamp_text, 600, 108, 36, bold=True)
            sd.text((540, 1362), stamp_text, font=stamp_font, fill=(255, 234, 234, int(255 * stamp_phase)), anchor="ma", stroke_width=3, stroke_fill=(*stamp_red, int(255 * stamp_phase)))
            sd.text((540, 1362), stamp_text, font=stamp_font, fill=(*stamp_red, int(255 * stamp_phase)), anchor="ma")
            stamp = stamp.rotate(-10, resample=Image.Resampling.BICUBIC, center=(540, 1362))
            frame.alpha_composite(stamp)


def render_video(output_path: Path, audio_path: Path, duration: float, fps: int) -> Path:
    scenes = _build_scenes(duration)
    background = _make_background()
    portraits = _load_portraits()
    temp_draw = ImageDraw.Draw(background.copy(), "RGBA")
    side_name_font = _load_font(28, bold=True)
    top_name_font = _load_font(30, bold=True)
    top_sub_font = _load_font(18, bold=True)
    stat_list_font = _load_font(20, bold=True)
    presenter_title_font = _load_font(22, bold=True)
    presenter_sub_font = _load_font(16, bold=True)
    value_font_map = {
        value: _fit_font_size(temp_draw, value, 120, 38, 16, bold=True)
        for row in STAT_ROWS
        for value in (row[1], row[2])
    }
    narration_temp = _synthesize_voiceover(_build_short_narration_text())

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
        _draw_top_player_cards(frame, portraits, top_name_font, top_sub_font)
        glow = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
        gd = ImageDraw.Draw(glow, "RGBA")
        gd.ellipse((-80, 360, 480, 1160), fill=(*FEDERER_RED, 42))
        gd.ellipse((WIDTH - 480, 360, WIDTH + 80, 1160), fill=(*NADAL_ORANGE, 42))
        glow = glow.filter(ImageFilter.GaussianBlur(radius=34))
        frame.alpha_composite(glow)
        if t < 1.15:
            hook_phase = _ease_out(t / 1.15)
            hook_layer = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
            hd = ImageDraw.Draw(hook_layer, "RGBA")
            hook_font = _fit_font_size(hd, "NADAL EST INJOUABLE ICI", 860, 56, 22, bold=True)
            hd.text((WIDTH // 2, 56), "NADAL EST INJOUABLE ICI", font=hook_font, fill=(255, 244, 220, int(255 * hook_phase)), anchor="ma", stroke_width=3, stroke_fill=(0, 0, 0, int(180 * hook_phase)))
            hd.text((WIDTH // 2, 88), "FEDERER VS NADAL", font=_load_font(20, bold=True), fill=(220, 229, 239, int(210 * hook_phase)), anchor="ma")
            frame.alpha_composite(hook_layer)
        _draw_center_stat_scene(frame, side_name_font, stat_list_font, value_font_map, scene.get("index"), phase)
        zoom = 1.0 + (0.014 * np.sin(t * 0.85)) + (0.018 if scene["kind"] == "board_final" else 0.0) * phase
        if zoom > 1.0005:
            zoom_w = int(WIDTH * zoom)
            zoom_h = int(HEIGHT * zoom)
            zoomed = frame.resize((zoom_w, zoom_h), Image.Resampling.LANCZOS)
            pan_x = int((zoom_w - WIDTH) * (0.5 + 0.03 * np.sin(t * 0.6)))
            pan_y = int((zoom_h - HEIGHT) * (0.5 + 0.025 * np.cos(t * 0.7)))
            pan_x = max(0, min(pan_x, zoom_w - WIDTH))
            pan_y = max(0, min(pan_y, zoom_h - HEIGHT))
            frame = zoomed.crop((pan_x, pan_y, pan_x + WIDTH, pan_y + HEIGHT))
        return np.array(frame.convert("RGB"))

    clip = VideoClip(make_frame, duration=duration)
    audio_clip, keep_alive = build_audio_track(audio_path, duration)
    audio_sources = [audio_clip.with_volume_scaled(MUSIC_VOLUME)]
    narration_clip = None
    if narration_temp is not None and narration_temp.exists():
        narration_clip = AudioFileClip(str(narration_temp))
        audio_sources.append(narration_clip.with_volume_scaled(VOICE_VOLUME))
    composite_audio = CompositeAudioClip(audio_sources).with_duration(duration)
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
        composite_audio.close()
        audio_clip.close()
        if narration_clip is not None:
            narration_clip.close()
        for item in keep_alive:
            item.close()
        if narration_temp is not None:
            try:
                narration_temp.unlink(missing_ok=True)
            except OSError:
                pass
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
    parser = argparse.ArgumentParser(description="Generate a Federer vs Nadal tennis stats Shorts video.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--audio", type=Path, default=DEFAULT_AUDIO)
    parser.add_argument("--duration", type=float, default=TOTAL_DURATION)
    parser.add_argument("--fps", type=int, default=FPS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = render_video(args.output, args.audio, args.duration, args.fps)
    print(f"[video_generator] Federer vs Nadal stats Shorts generated -> {output}")


if __name__ == "__main__":
    main()
