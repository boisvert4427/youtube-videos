from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
from moviepy import AudioFileClip, CompositeAudioClip, VideoClip
from moviepy.audio.fx import AudioFadeIn, AudioFadeOut
from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "football" / "psg_liverpool_double_confrontation_shorts.mp4"
DEFAULT_AUDIO = PROJECT_ROOT / "data" / "raw" / "audio" / "audio.mp3"

WIDTH = 1080
HEIGHT = 1920
FPS = 30
TOTAL_DURATION = 18.0
MUSIC_VOLUME = 0.22
FADE_OUT_AUDIO = 4.0
LOOP_CROSSFADE = 1.25

PSG_BLUE = (15, 38, 95)
PSG_RED = (198, 31, 63)
LIVERPOOL_RED = (208, 20, 48)
GOLD = (247, 199, 89)
ICE = (241, 246, 252)
MUTED = (179, 194, 214)
BLACK = (4, 8, 18)
PANEL = (10, 18, 34, 234)
PANEL_LIGHT = (17, 31, 57, 230)

MATCHES = [
    {
        "date": "8 avril 2026",
        "venue": "Parc des Princes",
        "score_left": 2,
        "score_right": 0,
        "subtitle": "Doue + Kvaratskhelia",
        "events": [("11'", "Doue"), ("65'", "Kvaratskhelia")],
    },
    {
        "date": "14 avril 2026",
        "venue": "Anfield",
        "score_left": 0,
        "score_right": 2,
        "subtitle": "Dembele x2",
        "events": [("72'", "Dembele"), ("90+1'", "Dembele")],
    },
]


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _ease_out(value: float) -> float:
    value = _clamp(value)
    return 1.0 - (1.0 - value) ** 3


def _load_font(size: int, bold: bool = False):
    candidates = [
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf",
    ]
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def _fit_font(text: str, max_width: int, start_size: int, min_size: int, bold: bool = True):
    probe = ImageDraw.Draw(Image.new("RGB", (10, 10), "black"))
    size = start_size
    while size >= min_size:
        font = _load_font(size, bold=bold)
        if probe.textbbox((0, 0), text, font=font)[2] <= max_width:
            return font
        size -= 1
    return _load_font(min_size, bold=bold)


def _draw_text(
    frame: Image.Image,
    position: tuple[int, int],
    text: str,
    font,
    fill: tuple[int, int, int],
    stroke_width: int = 2,
    stroke_fill: tuple[int, int, int] = BLACK,
    anchor: str = "mm",
) -> None:
    draw = ImageDraw.Draw(frame, "RGBA")
    draw.text(
        position,
        text,
        font=font,
        fill=(*fill, 255),
        anchor=anchor,
        stroke_width=stroke_width,
        stroke_fill=(*stroke_fill, 210),
    )


def _draw_pill(
    frame: Image.Image,
    box: tuple[int, int, int, int],
    text: str,
    fill: tuple[int, int, int, int],
    text_fill: tuple[int, int, int],
    font,
    radius: int = 28,
    outline: tuple[int, int, int, int] | None = None,
) -> None:
    draw = ImageDraw.Draw(frame, "RGBA")
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=2 if outline else 0)
    x0, y0, x1, y1 = box
    draw.text(((x0 + x1) // 2, (y0 + y1) // 2), text, font=font, fill=(*text_fill, 255), anchor="mm")


def _build_background_base() -> Image.Image:
    yy = np.linspace(0.0, 1.0, HEIGHT, dtype=np.float32)[:, None]
    top = np.array([8, 15, 31, 255], dtype=np.float32)
    bottom = np.array([3, 7, 16, 255], dtype=np.float32)
    arr = (top * (1.0 - yy) + bottom * yy).astype(np.uint8)
    arr = np.repeat(arr[:, None, :], WIDTH, axis=1)
    image = Image.fromarray(arr, "RGBA")

    draw = ImageDraw.Draw(image, "RGBA")
    for y in (220, 560, 1000, 1360, 1700):
        draw.line((0, y, WIDTH, y - 120), fill=(255, 255, 255, 15), width=3)
    draw.ellipse((-140, 180, 380, 700), outline=(*PSG_BLUE, 40), width=6)
    draw.ellipse((760, 320, 1240, 800), outline=(*LIVERPOOL_RED, 34), width=6)
    draw.rectangle((0, 260, WIDTH, 1660), outline=(255, 255, 255, 18), width=2)
    draw.rectangle((140, 420, WIDTH - 140, 1500), outline=(255, 255, 255, 12), width=2)
    return image


BACKGROUND_BASE = _build_background_base()


def _overlay_motion(frame: Image.Image, t: float) -> None:
    draw = ImageDraw.Draw(frame, "RGBA")

    left_x = int(150 + 70 * math.sin(t * 1.15))
    left_y = int(360 + 60 * math.cos(t * 0.85))
    right_x = int(920 + 80 * math.sin(t * 0.95 + 1.7))
    right_y = int(420 + 70 * math.cos(t * 0.72 + 0.8))
    center_y = int(970 + 52 * math.sin(t * 1.3))

    draw.ellipse((left_x - 280, left_y - 280, left_x + 280, left_y + 280), fill=(*PSG_BLUE, 48))
    draw.ellipse((right_x - 260, right_y - 260, right_x + 260, right_y + 260), fill=(*LIVERPOOL_RED, 46))
    draw.ellipse((390, center_y - 120, 690, center_y + 120), fill=(*GOLD, 28))

    sweep = int((math.sin(t * 1.1) * 0.5 + 0.5) * (WIDTH + 180)) - 180
    draw.rectangle((sweep, 0, sweep + 22, HEIGHT), fill=(255, 255, 255, 10))
    draw.rectangle((sweep + 44, 0, sweep + 58, HEIGHT), fill=(255, 255, 255, 6))

    for index in range(10):
        phase = t * 0.35 + index * 0.63
        x = int((index * 109 + phase * 160) % WIDTH)
        y = int((180 + index * 137 + phase * 120) % HEIGHT)
        radius = 2 + (index % 3)
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=(255, 255, 255, 110))


def _draw_header(frame: Image.Image, progress: float) -> None:
    title_font = _fit_font("PSG vs LIVERPOOL", 820, 72, 34, True)
    subtitle_font = _fit_font("DOUBLE CONFRONTATION 2026", 700, 30, 20, True)
    small_font = _fit_font("CHAMPIONS LEAGUE KNOCKOUT DRAMA", 640, 20, 16, False)

    slide = 1.0 - _ease_out(progress)
    title_y = int(160 + 40 * slide)
    subtitle_y = title_y + 90
    chip_y = title_y - 76

    _draw_pill(
        frame,
        (295, chip_y - 22, 785, chip_y + 36),
        "CHAMPIONS LEAGUE QUARTER-FINAL",
        PANEL_LIGHT,
        ICE,
        small_font,
        radius=22,
        outline=(255, 255, 255, 28),
    )

    psg_font = _fit_font("PSG", 220, 84, 44, True)
    lfc_font = _fit_font("LIVERPOOL", 260, 84, 42, True)
    left_x = int(280 - 48 * slide)
    right_x = int(800 + 52 * slide)
    vs_scale = 1.0 + 0.06 * math.sin(progress * math.pi)

    _draw_text(frame, (left_x, title_y), "PSG", psg_font, PSG_BLUE, stroke_width=3)
    _draw_text(frame, (540, title_y + 3), "VS", _fit_font("VS", 130, int(58 * vs_scale), 30, True), GOLD, stroke_width=3)
    _draw_text(frame, (right_x, title_y), "LIVERPOOL", lfc_font, LIVERPOOL_RED, stroke_width=3)
    _draw_text(frame, (540, subtitle_y), "DOUBLE CONFRONTATION 2026", subtitle_font, ICE, stroke_width=2)


def _draw_score_card(
    frame: Image.Image,
    match: dict[str, object],
    y: int,
    local_progress: float,
    reverse: bool = False,
) -> None:
    draw = ImageDraw.Draw(frame, "RGBA")
    card_w = 850
    card_h = 350
    x0 = 115
    x1 = x0 + card_w
    y0 = y
    y1 = y + card_h

    offset = int((1.0 - _ease_out(local_progress)) * 70)
    if reverse:
        offset = -offset

    draw.rounded_rectangle((x0 + offset, y0 + 14, x1 + offset, y1 + 14), radius=36, fill=(0, 0, 0, 70))
    draw.rounded_rectangle((x0 + offset, y0, x1 + offset, y1), radius=36, fill=PANEL, outline=(255, 255, 255, 28), width=2)
    draw.rounded_rectangle((x0 + offset + 16, y0 + 16, x1 + offset - 16, y1 - 16), radius=28, fill=(255, 255, 255, 8))

    venue_font = _fit_font(str(match["venue"]), 500, 28, 18, True)
    date_font = _fit_font(str(match["date"]), 260, 24, 16, True)
    score_font = _fit_font("0 - 0", 280, 110, 68, True)
    label_font = _fit_font("MATCH 1", 160, 20, 16, True)
    team_font = _fit_font("LIVERPOOL", 260, 42, 26, True)
    subtitle_font = _fit_font(str(match["subtitle"]), 360, 28, 18, True)
    event_font = _fit_font("90+1' Dembele", 280, 26, 18, True)

    _draw_pill(
        frame,
        (x0 + offset + 34, y0 + 28, x0 + offset + 228, y0 + 74),
        "MATCH " + ("1" if not reverse else "2"),
        PANEL_LIGHT,
        GOLD,
        label_font,
        radius=18,
        outline=(255, 255, 255, 24),
    )
    _draw_pill(
        frame,
        (x1 - 248 + offset, y0 + 28, x1 - 34 + offset, y0 + 74),
        str(match["date"]),
        (*PSG_BLUE, 255),
        ICE,
        date_font,
        radius=18,
    )

    _draw_text(frame, (540 + offset, y0 + 122), str(match["venue"]), venue_font, ICE, stroke_width=2)

    left_score = int(round(float(match["score_left"]) * _ease_out(local_progress)))
    right_score = int(round(float(match["score_right"]) * _ease_out(local_progress)))
    _draw_text(frame, (540 + offset, y0 + 204), f"{left_score} - {right_score}", score_font, ICE, stroke_width=4)

    left_team = "PSG" if not reverse else "LIVERPOOL"
    right_team = "LIVERPOOL" if not reverse else "PSG"
    left_color = PSG_BLUE if not reverse else LIVERPOOL_RED
    right_color = LIVERPOOL_RED if not reverse else PSG_BLUE
    left_team_font = _fit_font(left_team, 240, 42, 26, True)
    right_team_font = _fit_font(right_team, 240, 42, 26, True)

    _draw_pill(frame, (x0 + offset + 56, y0 + 246, x0 + offset + 240, y0 + 296), left_team, (*left_color, 255), ICE, left_team_font, radius=24)
    _draw_pill(frame, (x1 - 240 + offset, y0 + 246, x1 - 56 + offset, y0 + 296), right_team, (*right_color, 255), ICE, right_team_font, radius=24)

    _draw_text(frame, (540 + offset, y0 + 322), str(match["subtitle"]), subtitle_font, GOLD, stroke_width=2)

    if local_progress > 0.32:
        event_progress = _clamp((local_progress - 0.32) / 0.68)
        first_text = f"{match['events'][0][0]}  {match['events'][0][1]}"
        first_x = x0 + offset + 70
        draw.rounded_rectangle((first_x, y0 + 300, first_x + 300, y0 + 346), radius=18, fill=(255, 255, 255, 10))
        _draw_text(frame, (first_x + 150, y0 + 323), first_text, event_font, ICE, stroke_width=2)
        if event_progress > 0.46 and len(match["events"]) > 1:
            second_text = f"{match['events'][1][0]}  {match['events'][1][1]}"
            second_x = x1 - offset - 370
            draw.rounded_rectangle((second_x, y0 + 300, second_x + 300, y0 + 346), radius=18, fill=(255, 255, 255, 10))
            _draw_text(frame, (second_x + 150, y0 + 323), second_text, event_font, ICE, stroke_width=2)

    agg_font = _fit_font("AGG 2-0", 180, 24, 16, True)
    _draw_pill(frame, (x1 - offset - 190, y0 + 112, x1 - offset - 34, y0 + 154), f"AGG {'2-0' if not reverse else '0-2'}", (*GOLD, 255), BLACK, agg_font, radius=16)

    if reverse:
        spotlight = int(0.15 * 255 * _ease_out(local_progress))
        draw.polygon(
            [(x0 + offset + 10, y0 + 20), (x1 + offset - 10, y0 + 20), (x1 + offset - 70, y1 - 20), (x0 + offset + 70, y1 - 20)],
            fill=(255, 255, 255, spotlight),
        )


def _draw_outro(frame: Image.Image, progress: float) -> None:
    scale = 1.0 + 0.05 * math.sin(progress * math.pi * 2.0)
    alpha = int(255 * _ease_out(progress))
    agg_font = _fit_font("4 - 0", 360, int(128 * scale), 82, True)
    title_font = _fit_font("PSG A CASSE LE MATCH", 760, 52, 28, True)
    sub_font = _fit_font("ANFIELD N'A PAS CHANGE LE VERDICT", 820, 30, 18, True)
    final_font = _fit_font("MISSION ACCOMPLIE", 360, 44, 24, True)

    draw = ImageDraw.Draw(frame, "RGBA")
    draw.rounded_rectangle((100, 560, 980, 1120), radius=52, fill=(6, 12, 24, 168), outline=(255, 255, 255, 24), width=2)
    _draw_pill(frame, (310, 632, 770, 690), "FINAL AGGREGATE", PANEL_LIGHT, GOLD, _fit_font("FINAL AGGREGATE", 260, 22, 16, True), radius=22)
    _draw_text(frame, (540, 820), "4 - 0", agg_font, ICE, stroke_width=6)
    _draw_text(frame, (540, 1010), "PSG A CASSE LE MATCH", title_font, ICE, stroke_width=3)
    _draw_text(frame, (540, 1084), "ANFIELD N'A PAS CHANGE LE VERDICT", sub_font, MUTED, stroke_width=2)
    _draw_pill(frame, (360, 1188, 720, 1250), "PSG EN DEMI-FINALE", (*GOLD, alpha), BLACK, final_font, radius=26)


def _scene_name(t: float) -> str:
    if t < 3.2:
        return "intro"
    if t < 8.7:
        return "first"
    if t < 14.0:
        return "second"
    return "outro"


def _make_frame(t: float) -> np.ndarray:
    frame = BACKGROUND_BASE.copy()
    _overlay_motion(frame, t)

    scene = _scene_name(t)
    if scene == "intro":
        progress = t / 3.2
        _draw_header(frame, progress)
        draw = ImageDraw.Draw(frame, "RGBA")
        draw.rounded_rectangle((140, 1300, 940, 1540), radius=42, fill=(8, 14, 28, 160), outline=(255, 255, 255, 24), width=2)
        note_font = _fit_font("LE GROS MESSAGE", 540, 32, 20, True)
        _draw_text(frame, (540, 1380), "Le PSG n'a pas survecu au duel.", note_font, ICE, stroke_width=2)
        _draw_text(frame, (540, 1446), "Il l'a domine.", _fit_font("Il l'a domine.", 360, 56, 34, True), GOLD, stroke_width=3)
        return np.array(frame.convert("RGB"))

    if scene == "first":
        local = (t - 3.2) / 5.5
        _draw_header(frame, 1.0)
        draw = ImageDraw.Draw(frame, "RGBA")
        draw.rounded_rectangle((92, 500, 988, 1210), radius=54, fill=(7, 13, 26, 170), outline=(255, 255, 255, 16), width=2)
        _draw_pill(frame, (392, 430, 688, 486), "ALLER", (*PSG_BLUE, 255), ICE, _fit_font("ALLER", 220, 24, 18, True), radius=20)
        _draw_score_card(frame, MATCHES[0], 565, local, reverse=False)
        return np.array(frame.convert("RGB"))

    if scene == "second":
        local = (t - 8.7) / 5.3
        _draw_header(frame, 1.0)
        draw = ImageDraw.Draw(frame, "RGBA")
        draw.rounded_rectangle((92, 470, 988, 1210), radius=54, fill=(7, 13, 26, 170), outline=(255, 255, 255, 16), width=2)
        _draw_pill(frame, (378, 398, 702, 454), "RETOUR", (*LIVERPOOL_RED, 255), ICE, _fit_font("RETOUR", 220, 24, 18, True), radius=20)
        _draw_score_card(frame, MATCHES[1], 545, local, reverse=True)
        return np.array(frame.convert("RGB"))

    progress = (t - 14.0) / 4.0
    _draw_header(frame, 1.0)
    _draw_outro(frame, progress)
    return np.array(frame.convert("RGB"))


def build_audio_track(audio_path: Path, duration: float):
    if not audio_path.exists():
        return None, []

    base = AudioFileClip(str(audio_path))
    if base.duration >= duration:
        return base.subclipped(0, duration).with_effects([AudioFadeOut(min(FADE_OUT_AUDIO, duration))]), [base]

    clips = []
    keep_alive = [base]
    step = max(0.1, base.duration - LOOP_CROSSFADE)
    loops = int(math.ceil(max(0.0, duration - LOOP_CROSSFADE) / step))
    for index in range(loops):
        segment = base.with_start(index * step).with_effects([AudioFadeIn(LOOP_CROSSFADE), AudioFadeOut(LOOP_CROSSFADE)])
        clips.append(segment)
    mixed = CompositeAudioClip(clips).with_duration(duration).with_effects([AudioFadeOut(min(FADE_OUT_AUDIO, duration))])
    return mixed, keep_alive


def render_video(output_path: Path, duration: float, fps: int, audio_path: Path) -> Path:
    clip = VideoClip(_make_frame, duration=duration)
    audio_clip, keep_alive = build_audio_track(audio_path, duration)
    if audio_clip is not None:
        clip = clip.with_audio(audio_clip.with_volume_scaled(MUSIC_VOLUME))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    clip.write_videofile(
        str(output_path),
        fps=fps,
        codec="libx264",
        audio_codec="aac" if audio_clip is not None else None,
        threads=4,
    )
    del keep_alive
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a viral PSG vs Liverpool double-confrontation Shorts video.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output MP4 path.")
    parser.add_argument("--duration", type=float, default=TOTAL_DURATION, help="Video duration in seconds.")
    parser.add_argument("--fps", type=int, default=FPS, help="Frames per second.")
    parser.add_argument("--audio", type=Path, default=DEFAULT_AUDIO, help="Background music path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    render_video(args.output, args.duration, args.fps, args.audio)


if __name__ == "__main__":
    main()
