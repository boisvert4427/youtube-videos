from __future__ import annotations

import argparse
from dataclasses import dataclass
from functools import lru_cache
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
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "tennis" / "federer_vs_nadal_h2h_scores_shorts.mp4"

WIDTH = 1080
HEIGHT = 1920
FPS = 24
TOTAL_DURATION = 45.0
MUSIC_VOLUME = 0.40

PORTRAIT_TOP = 410
SCOREBOARD_TOP = 750
SCOREBOARD_LINE_Y = SCOREBOARD_TOP + 78
SCOREBOARD_BOTTOM = SCOREBOARD_TOP + 150
SCORE_TOUCH_Y = SCOREBOARD_BOTTOM + 6

TIMELINE_START_Y = 1840
ROW_SPACING = 428
TIMELINE_LEFT = 0
TIMELINE_RIGHT = WIDTH
TIMELINE_END_PADDING = 18

BADGE_W = 186
BADGE_H = 66
BADGE_GAP_X = 16

FEDERER = {
    "photo": "roger_federer.jpg",
    "accent": (116, 198, 255),
}
NADAL = {
    "photo": "rafael_nadal.jpg",
    "accent": (255, 118, 72),
}


@dataclass(frozen=True)
class MatchEntry:
    year: int
    month: int
    event: str
    winner: str
    scoreline: str


MATCHES: list[MatchEntry] = [
    MatchEntry(2004, 3, "MIAMI R3", "right", "6-3 6-3"),
    MatchEntry(2005, 4, "MIAMI F", "left", "2-6 6-7 7-6 6-3 6-1"),
    MatchEntry(2005, 6, "ROLAND GARROS SF", "right", "6-3 4-6 6-4 6-3"),
    MatchEntry(2006, 3, "DUBAI F", "right", "2-6 6-4 6-4"),
    MatchEntry(2006, 4, "MONTE CARLO F", "right", "6-2 6-7 6-3 7-6"),
    MatchEntry(2006, 5, "ROME F", "right", "6-7 7-6 6-4 2-6 7-6"),
    MatchEntry(2006, 6, "ROLAND GARROS F", "right", "1-6 6-1 6-4 7-6"),
    MatchEntry(2006, 7, "WIMBLEDON F", "left", "6-0 7-6 6-7 6-3"),
    MatchEntry(2006, 11, "MASTERS CUP SF", "left", "6-4 7-5"),
    MatchEntry(2007, 4, "MONTE CARLO F", "right", "6-4 6-4"),
    MatchEntry(2007, 5, "HAMBURG F", "left", "2-6 6-2 6-0"),
    MatchEntry(2007, 6, "ROLAND GARROS F", "right", "6-3 4-6 6-3 6-4"),
    MatchEntry(2007, 7, "WIMBLEDON F", "left", "7-6 4-6 7-6 2-6 6-2"),
    MatchEntry(2007, 11, "MASTERS CUP SF", "left", "6-4 6-1"),
    MatchEntry(2008, 4, "MONTE CARLO F", "right", "7-5 7-5"),
    MatchEntry(2008, 5, "HAMBURG F", "right", "7-5 6-7 6-3"),
    MatchEntry(2008, 6, "ROLAND GARROS F", "right", "6-1 6-3 6-0"),
    MatchEntry(2008, 7, "WIMBLEDON F", "right", "6-4 6-4 6-7 6-7 9-7"),
    MatchEntry(2009, 2, "AUSTRALIAN OPEN F", "right", "7-5 3-6 7-6 3-6 6-2"),
    MatchEntry(2009, 5, "MADRID F", "left", "6-4 6-4"),
    MatchEntry(2010, 5, "MADRID F", "right", "6-4 7-6"),
    MatchEntry(2010, 11, "ATP FINALS F", "left", "6-3 3-6 6-1"),
    MatchEntry(2011, 4, "MIAMI SF", "right", "6-3 6-2"),
    MatchEntry(2011, 5, "MADRID SF", "right", "5-7 6-1 6-3"),
    MatchEntry(2011, 6, "ROLAND GARROS F", "right", "7-5 7-6 5-7 6-1"),
    MatchEntry(2011, 11, "ATP FINALS RR", "left", "6-3 6-0"),
    MatchEntry(2012, 1, "AUSTRALIAN OPEN SF", "right", "6-7 6-2 7-6 6-4"),
    MatchEntry(2012, 3, "INDIAN WELLS SF", "left", "6-3 6-4"),
    MatchEntry(2013, 3, "INDIAN WELLS QF", "right", "6-4 6-2"),
    MatchEntry(2013, 5, "ROME F", "right", "6-1 6-3"),
    MatchEntry(2013, 8, "CINCINNATI QF", "right", "5-7 6-4 6-3"),
    MatchEntry(2013, 11, "ATP FINALS SF", "right", "7-5 6-3"),
    MatchEntry(2014, 1, "AUSTRALIAN OPEN SF", "right", "7-6 6-3 6-3"),
    MatchEntry(2015, 11, "BASEL F", "left", "6-3 5-7 6-3"),
    MatchEntry(2017, 1, "AUSTRALIAN OPEN F", "left", "6-4 3-6 6-1 3-6 6-3"),
    MatchEntry(2017, 3, "INDIAN WELLS R4", "left", "6-2 6-3"),
    MatchEntry(2017, 4, "MIAMI F", "left", "6-3 6-4"),
    MatchEntry(2017, 10, "SHANGHAI F", "left", "6-4 6-3"),
    MatchEntry(2019, 6, "ROLAND GARROS SF", "right", "6-3 6-4 6-2"),
    MatchEntry(2019, 7, "WIMBLEDON SF", "left", "7-6 1-6 6-3 6-4"),
]

YEARS = list(range(2004, 2020))
MONTH_LABELS = ("JAN", "FEV", "MAR", "AVR", "MAI", "JUN", "JUL", "AOU", "SEP", "OCT", "NOV", "DEC")


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return min(max(value, low), high)


def _ease_out(value: float) -> float:
    value = _clamp(value)
    return 1.0 - (1.0 - value) ** 3


def _load_portrait(path: Path, initials: str) -> Image.Image:
    if path.exists():
        return ImageOps.exif_transpose(Image.open(path)).convert("RGBA")
    img = Image.new("RGBA", (720, 720), (18, 24, 34, 255))
    draw = ImageDraw.Draw(img)
    draw.text((360, 360), initials, font=_load_font(80, bold=True), fill=(245, 248, 252, 255), anchor="mm")
    return img


def _make_background() -> Image.Image:
    xx = np.linspace(0, 1, WIDTH, dtype=np.float32)
    yy = np.linspace(0, 1, HEIGHT, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(xx, yy)
    left = np.array([166, 129, 166], dtype=np.float32)
    right = np.array([92, 118, 148], dtype=np.float32)
    shade = (1.0 - 0.22 * grid_y)[..., None]
    split = (grid_x < 0.5).astype(np.float32)[..., None]
    image = np.clip((left * split + right * (1.0 - split)) * shade, 0, 255).astype(np.uint8)
    return Image.fromarray(image, mode="RGB").convert("RGBA")


def _draw_glow_text(
    frame: Image.Image,
    position: tuple[int, int],
    text: str,
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int],
    stroke_width: int = 2,
    anchor: str = "mm",
) -> None:
    glow = Image.new("RGBA", frame.size, (0, 0, 0, 0))
    gd = ImageDraw.Draw(glow, "RGBA")
    gd.text(position, text, font=font, fill=(0, 0, 0, 120), anchor=anchor)
    frame.alpha_composite(glow.filter(ImageFilter.GaussianBlur(radius=5)))
    ImageDraw.Draw(frame, "RGBA").text(
        position,
        text,
        font=font,
        fill=(*fill, 255),
        anchor=anchor,
        stroke_width=stroke_width,
        stroke_fill=(0, 0, 0, 180),
    )


def _circle_portrait(source: Image.Image, accent: tuple[int, int, int]) -> Image.Image:
    size = 178
    portrait = ImageOps.fit(source, (size, size), method=Image.Resampling.LANCZOS, centering=(0.5, 0.22))
    portrait = ImageEnhance.Contrast(ImageEnhance.Brightness(portrait).enhance(1.06)).enhance(1.08)
    mask = Image.new("L", (size, size), 0)
    ImageDraw.Draw(mask).ellipse((0, 0, size - 1, size - 1), fill=255)

    tile = Image.new("RGBA", (size + 28, size + 28), (0, 0, 0, 0))
    draw = ImageDraw.Draw(tile, "RGBA")
    draw.ellipse((0, 0, size + 27, size + 27), fill=(245, 248, 252, 245))
    draw.ellipse((7, 7, size + 20, size + 20), fill=(*accent, 255))
    draw.ellipse((13, 13, size + 14, size + 14), fill=(20, 24, 30, 255))
    tile.paste(portrait, (14, 14), mask)
    return tile


def _draw_static_layer(portraits: dict[str, Image.Image]) -> Image.Image:
    frame = _make_background()
    draw = ImageDraw.Draw(frame, "RGBA")
    draw.line((WIDTH // 2, 0, WIDTH // 2, HEIGHT), fill=(255, 255, 255, 26), width=2)

    top_font = _load_font(40, bold=True)
    title_font = _fit_font_size(draw, "FEDERER vs NADAL", 900, 80, 40, bold=True)
    sub_font = _load_font(30, bold=True)
    draw.text((WIDTH // 2, 134), "Palmares", font=top_font, fill=(247, 250, 253, 255), anchor="mm")
    _draw_glow_text(frame, (WIDTH // 2, 232), "FEDERER vs NADAL", title_font, (247, 250, 253), stroke_width=2)
    draw.text((WIDTH // 2, 330), "LES SCORES DU DUEL DEPUIS 2004", font=sub_font, fill=(238, 244, 248, 255), anchor="mm")

    frame.alpha_composite(_circle_portrait(portraits["federer"], FEDERER["accent"]), (166, PORTRAIT_TOP))
    frame.alpha_composite(_circle_portrait(portraits["nadal"], NADAL["accent"]), (742, PORTRAIT_TOP))
    draw.text((WIDTH // 2, 1848), "@club.versus", font=_load_font(28, bold=True), fill=(255, 255, 255, 58), anchor="mm")
    return frame


def _short_score(scoreline: str) -> str:
    parts = scoreline.split()
    return " ".join(parts[:3]) if len(parts) > 3 else scoreline


def _short_event(event: str) -> str:
    replacements = {
        "AUSTRALIAN OPEN": "AO",
        "ROLAND GARROS": "RG",
        "INDIAN WELLS": "IW",
        "MONTE CARLO": "MC",
        "ATP FINALS": "FINALS",
        "MASTERS CUP": "MASTERS",
        "WIMBLEDON": "WIMB",
    }
    for source, short in replacements.items():
        event = event.replace(source, short)
    return event


@lru_cache(maxsize=None)
def _score_badge(scoreline: str, accent: tuple[int, int, int]) -> Image.Image:
    img = Image.new("RGBA", (BADGE_W, BADGE_H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img, "RGBA")
    shadow = Image.new("RGBA", (BADGE_W, BADGE_H), (0, 0, 0, 0))
    ImageDraw.Draw(shadow, "RGBA").rounded_rectangle((5, 7, BADGE_W - 5, BADGE_H - 3), radius=16, fill=(0, 0, 0, 92))
    img.alpha_composite(shadow.filter(ImageFilter.GaussianBlur(radius=4)))
    draw.rounded_rectangle((0, 0, BADGE_W - 1, BADGE_H - 1), radius=16, fill=(248, 250, 253, 246), outline=(*accent, 230), width=3)
    draw.rounded_rectangle((8, 8, BADGE_W - 9, BADGE_H - 9), radius=11, outline=(25, 30, 38, 110), width=1)
    text = _short_score(scoreline)
    font = _fit_font_size(draw, text, BADGE_W - 22, 24, 16, bold=True)
    draw.text((BADGE_W // 2, BADGE_H // 2), text, font=font, fill=(16, 18, 22, 255), anchor="mm")
    return img


def _time_offset(year: int, month: int = 1) -> float:
    return (year - YEARS[0]) + (month - 1) / 12.0


@lru_cache(maxsize=1)
def _match_layout() -> tuple[tuple[MatchEntry, int, int], ...]:
    slots: list[tuple[MatchEntry, int, int]] = []
    last_slot_y = {"left": [-10_000.0, -10_000.0], "right": [-10_000.0, -10_000.0]}
    min_gap = BADGE_H + 38
    for match in MATCHES:
        slot_y = _time_offset(match.year, match.month) * ROW_SPACING
        lanes = last_slot_y[match.winner]
        lane = next((idx for idx, previous_y in enumerate(lanes) if slot_y - previous_y >= min_gap), 0)
        if match.winner == "left":
            x = 142 + lane * (BADGE_W + BADGE_GAP_X)
        else:
            x = WIDTH - 142 - BADGE_W - lane * (BADGE_W + BADGE_GAP_X)
        lanes[lane] = slot_y
        slots.append((match, x, 28))
    return tuple(slots)


@lru_cache(maxsize=1)
def _timeline_end_scroll() -> float:
    last_match_y = max(
        _time_offset(match.year, match.month) * ROW_SPACING + y_offset
        for match, _x, y_offset in _match_layout()
    )
    return TIMELINE_START_Y + last_match_y - SCORE_TOUCH_Y + TIMELINE_END_PADDING


def _timeline_y(year: int, month: int, progress: float) -> float:
    return TIMELINE_START_Y + _time_offset(year, month) * ROW_SPACING - _timeline_end_scroll() * _ease_out(progress)


@lru_cache(maxsize=1)
def _timeline_layer() -> Image.Image:
    content_height = int(
        max(
            _time_offset(YEARS[-1], 12) * ROW_SPACING + 120,
            max(_time_offset(match.year, match.month) * ROW_SPACING + offset + BADGE_H + 80 for match, _x, offset in _match_layout()),
        )
    )
    layer = Image.new("RGBA", (WIDTH, content_height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer, "RGBA")
    year_font = _load_font(34, bold=True)
    month_font = _load_font(15, bold=True)

    for year in YEARS:
        for month in range(2, 13):
            y = _time_offset(year, month) * ROW_SPACING
            x = TIMELINE_LEFT
            while x < TIMELINE_RIGHT:
                draw.line((x, y, min(x + 7, TIMELINE_RIGHT), y), fill=(255, 255, 255, 24), width=1)
                x += 24
            if month in (3, 6, 9, 12):
                draw.text(
                    (92, y - 1),
                    f"{month:02d}",
                    font=month_font,
                    fill=(247, 250, 253, 112),
                    anchor="rm",
                )

        y = _time_offset(year, 1) * ROW_SPACING
        x = TIMELINE_LEFT
        while x < TIMELINE_RIGHT:
            draw.line((x, y, min(x + 19, TIMELINE_RIGHT), y), fill=(255, 255, 255, 54), width=2)
            x += 26
        draw.text((48, y - 2), str(year), font=year_font, fill=(247, 250, 253, 255), anchor="lm")

    return layer


def _draw_timeline(frame: Image.Image, progress: float) -> None:
    layer = _timeline_layer()
    scroll = _timeline_end_scroll() * _ease_out(progress)
    paste_y = int(round(TIMELINE_START_Y - scroll))
    visible_top = max(SCORE_TOUCH_Y, paste_y)
    visible_bottom = min(HEIGHT, paste_y + layer.height)
    if visible_bottom <= visible_top:
        return

    crop_top = visible_top - paste_y
    crop_bottom = crop_top + (visible_bottom - visible_top)
    crop = layer.crop((0, crop_top, WIDTH, crop_bottom))
    frame.alpha_composite(crop, (0, visible_top))

    draw = ImageDraw.Draw(frame, "RGBA")
    for match, x, offset in _match_layout():
        by = int(_timeline_y(match.year, match.month, progress) + offset)
        fade = _clamp((by - SCORE_TOUCH_Y) / 28.0)
        if fade <= 0 or by > HEIGHT + 190:
            continue

        accent = FEDERER["accent"] if match.winner == "left" else NADAL["accent"]
        badge = _score_badge(match.scoreline, accent)
        if fade < 1:
            badge = badge.copy()
            badge.putalpha(ImageEnhance.Brightness(badge.getchannel("A")).enhance(fade))

        label = f"{match.month:02d}/{match.year} - {_short_event(match.event)}"
        label_font = _fit_font_size(draw, label, BADGE_W + 18, 25, 16, bold=True)
        draw.text(
            (x + BADGE_W // 2, by - 20),
            label,
            font=label_font,
            fill=(247, 250, 253, int(238 * fade)),
            anchor="mm",
            stroke_width=1,
            stroke_fill=(0, 0, 0, int(95 * fade)),
        )
        frame.alpha_composite(badge, (x, by))


def _scores_for_progress(progress: float) -> tuple[int, int]:
    left = 0
    right = 0
    for match, _x, offset in _match_layout():
        if _timeline_y(match.year, match.month, progress) + offset > SCORE_TOUCH_Y:
            continue
        if match.winner == "left":
            left += 1
        else:
            right += 1
    return left, right


def _draw_scoreboard(frame: Image.Image, left_score: int, right_score: int) -> None:
    draw = ImageDraw.Draw(frame, "RGBA")
    left_box = (148, SCOREBOARD_TOP - 2, 394, SCOREBOARD_BOTTOM + 10)
    right_box = (686, SCOREBOARD_TOP - 2, 932, SCOREBOARD_BOTTOM + 10)
    draw.line((0, SCOREBOARD_LINE_Y, WIDTH, SCOREBOARD_LINE_Y), fill=(255, 255, 255, 180), width=4)

    small_font = _load_font(31, bold=True)
    draw.text((271, SCOREBOARD_TOP - 29), "Victoires", font=small_font, fill=(247, 249, 253, 255), anchor="mm")
    draw.text((809, SCOREBOARD_TOP - 29), "Victoires", font=small_font, fill=(247, 249, 253, 255), anchor="mm")
    draw.text((WIDTH // 2, SCOREBOARD_TOP - 24), "H2H", font=_load_font(24, bold=True), fill=(247, 249, 253, 255), anchor="mm")

    for box, accent in ((left_box, FEDERER["accent"]), (right_box, NADAL["accent"])):
        shadow = Image.new("RGBA", frame.size, (0, 0, 0, 0))
        sd = ImageDraw.Draw(shadow, "RGBA")
        sd.rounded_rectangle((box[0] + 8, box[1] + 10, box[2] + 8, box[3] + 10), radius=22, fill=(0, 0, 0, 105))
        frame.alpha_composite(shadow.filter(ImageFilter.GaussianBlur(radius=8)))
        draw.rounded_rectangle(box, radius=22, fill=(247, 249, 253, 255), outline=(*accent, 235), width=3)
        draw.rounded_rectangle(
            (box[0] + 8, box[1] + 8, box[2] - 8, box[3] - 8),
            radius=16,
            outline=(255, 255, 255, 160),
            width=2,
        )
        draw.rounded_rectangle((box[0] + 1, box[1] + 1, box[2] - 1, box[1] + 18), radius=18, fill=(*accent, 56))

    left_font = _fit_font_size(draw, str(left_score), 190, 102, 54, bold=True)
    right_font = _fit_font_size(draw, str(right_score), 190, 102, 54, bold=True)
    _draw_glow_text(frame, (271, SCOREBOARD_TOP + 87), str(left_score), left_font, (6, 7, 10), stroke_width=3)
    _draw_glow_text(frame, (809, SCOREBOARD_TOP + 87), str(right_score), right_font, (6, 7, 10), stroke_width=3)


def render_video(output_path: Path, audio_path: Path, duration: float, fps: int) -> Path:
    if duration > 50:
        raise ValueError("La video doit durer 50 secondes maximum.")
    portraits = {
        "federer": _load_portrait(PHOTOS_DIR / FEDERER["photo"], "RF"),
        "nadal": _load_portrait(PHOTOS_DIR / NADAL["photo"], "RN"),
    }
    static_layer = _draw_static_layer(portraits)

    def make_frame(t: float) -> np.ndarray:
        progress = _clamp(t / max(duration, 1e-6))
        frame = static_layer.copy()
        _draw_timeline(frame, progress)
        _draw_scoreboard(frame, *_scores_for_progress(progress))
        return np.asarray(frame.convert("RGB"))

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
            bitrate="10000k",
            preset="veryfast",
            threads=4,
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a Federer vs Nadal H2H score timeline Shorts video.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--audio", type=Path, default=DEFAULT_AUDIO)
    parser.add_argument("--duration", type=float, default=TOTAL_DURATION)
    parser.add_argument("--fps", type=int, default=FPS)
    args = parser.parse_args()
    render_video(args.output, args.audio, args.duration, args.fps)
    print(f"[video_generator] Federer vs Nadal H2H score timeline generated -> {args.output}")


if __name__ == "__main__":
    main()
