from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from moviepy import VideoClip
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageOps

from video_generator.tennis.generate_atp_shorts_timeline_moviepy import (
    DEFAULT_AUDIO,
    FPS,
    HEIGHT,
    HOLD_END,
    HOLD_START,
    WIDTH,
    _fit_font,
    _load_font,
    _resolve_player_image,
    _truncate_to_width,
    build_audio_track,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_INPUT = PROJECT_ROOT / "data" / "processed" / "tennis" / "roland_garros_titles_top12_cards.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "tennis" / "roland_garros_titles_cards_shorts.mp4"
DEFAULT_PHOTOS_DIR = PROJECT_ROOT / "data" / "raw" / "player_photos"
DEFAULT_LOGO = PROJECT_ROOT / "data" / "raw" / "tennis_logos" / "roland_garros.jpg"

VISIBLE_CARDS = 3
CARD_W = 304
CARD_H = 1368
CARD_GAP = 24
TOTAL_DURATION = 42.0


@dataclass(frozen=True)
class RolandGarrosEntry:
    rank: int
    player_name: str
    country_code: str
    titles: int
    years_won: str
    first_title: str
    last_title: str
    badge_label: str
    card_bg_color: str
    accent_color: str


def _hex_to_rgb(value: str) -> tuple[int, int, int]:
    cleaned = value.lstrip("#")
    return tuple(int(cleaned[i : i + 2], 16) for i in (0, 2, 4))


def _draw_text(draw: ImageDraw.ImageDraw, x: int, y: int, text: str, font, fill: str | tuple[int, int, int], anchor: str = "la") -> None:
    draw.text((x, y), text, font=font, fill=fill, anchor=anchor)


def _wrap_lines(draw: ImageDraw.ImageDraw, text: str, font, max_width: int, max_lines: int) -> list[str]:
    words = [word for word in text.split() if word]
    if not words:
        return []

    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        if draw.textbbox((0, 0), candidate, font=font)[2] <= max_width:
            current = candidate
            continue
        lines.append(current)
        current = word
        if len(lines) == max_lines - 1:
            break

    if len(lines) < max_lines:
        remaining = words[len(" ".join(lines + [current]).split()) :]
        tail = " ".join([current] + remaining).strip()
        lines.append(_truncate_to_width(draw, tail, font, max_width))
    return lines[:max_lines]


def load_entries(input_csv: Path) -> list[RolandGarrosEntry]:
    entries: list[RolandGarrosEntry] = []
    with input_csv.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            entries.append(
                RolandGarrosEntry(
                    rank=int(row["rank"]),
                    player_name=row["player_name"].strip(),
                    country_code=row["country_code"].strip().lower(),
                    titles=int(row["titles"]),
                    years_won=row["years_won"].strip(),
                    first_title=row["first_title"].strip(),
                    last_title=row["last_title"].strip(),
                    badge_label=row["badge_label"].strip(),
                    card_bg_color=row["card_bg_color"].strip(),
                    accent_color=row["accent_color"].strip(),
                )
            )
    return entries


def _background() -> Image.Image:
    xx = np.linspace(0, 1, WIDTH, dtype=np.float32)
    yy = np.linspace(0, 1, HEIGHT, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(xx, yy)
    clay = np.array([176, 92, 51], dtype=np.float32)
    sand = np.array([246, 220, 193], dtype=np.float32)
    deep = np.array([21, 39, 31], dtype=np.float32)
    glow = np.array([255, 198, 120], dtype=np.float32)
    mix = np.clip(0.62 * grid_y + 0.12 * (1.0 - grid_x), 0.0, 1.0)
    top_glow = np.exp(-(((grid_x - 0.82) / 0.28) ** 2 + ((grid_y - 0.14) / 0.24) ** 2))
    lower_glow = np.exp(-(((grid_x - 0.22) / 0.26) ** 2 + ((grid_y - 0.82) / 0.30) ** 2))
    bg = np.clip(
        sand[None, None, :] * (1.0 - mix[..., None])
        + clay[None, None, :] * (0.72 * mix[..., None])
        + deep[None, None, :] * (0.18 * (1.0 - grid_y[..., None]))
        + glow[None, None, :] * (0.07 * top_glow[..., None] + 0.08 * lower_glow[..., None]),
        0,
        255,
    ).astype(np.uint8)
    frame = Image.fromarray(np.dstack([bg, np.full((HEIGHT, WIDTH), 255, dtype=np.uint8)]), mode="RGBA")
    overlay = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay, "RGBA")
    od.rounded_rectangle((44, 44, WIDTH - 44, HEIGHT - 44), radius=44, outline=(255, 255, 255, 16), width=2)
    od.ellipse((130, 160, WIDTH - 130, 810), outline=(255, 228, 190, 14), width=3)
    od.line((122, 374, WIDTH - 122, 374), fill=(255, 255, 255, 10), width=2)
    od.line((122, 454, WIDTH - 122, 454), fill=(255, 255, 255, 8), width=1)
    od.line((122, 1560, WIDTH - 122, 1560), fill=(255, 255, 255, 8), width=1)
    frame.alpha_composite(overlay)
    return frame


def _render_logo(size: int) -> Image.Image | None:
    if not DEFAULT_LOGO.exists():
        return None
    try:
        logo = ImageOps.exif_transpose(Image.open(DEFAULT_LOGO)).convert("RGBA")
        logo = ImageOps.contain(logo, (size, size), method=Image.Resampling.LANCZOS)
        return logo
    except Exception:
        return None


def _resolve_flag(country_code: str, flags_dir: Path) -> Path | None:
    if not country_code:
        return None
    path = flags_dir / f"{country_code.lower()}.png"
    return path if path.exists() else None


def _make_placeholder(draw: ImageDraw.ImageDraw, rect: tuple[int, int, int, int], player_name: str) -> None:
    draw.rounded_rectangle(rect, radius=32, fill=(239, 225, 208, 255))
    initials = "".join(part[0] for part in player_name.split()[:2]).upper()
    font = _fit_font(draw, initials, rect[2] - rect[0] - 24, 110, 44, bold=True)
    draw.text(((rect[0] + rect[2]) // 2, (rect[1] + rect[3]) // 2), initials, font=font, fill="#6d432d", anchor="mm")


def _render_track_card(entry: RolandGarrosEntry, photos_dir: Path, flags_dir: Path) -> Image.Image:
    frame = Image.new("RGBA", (CARD_W, CARD_H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(frame, "RGBA")
    accent_rgb = _hex_to_rgb(entry.accent_color)

    draw.rounded_rectangle((8, 8, CARD_W - 8, CARD_H - 8), radius=36, fill=entry.card_bg_color, outline=(*accent_rgb, 255), width=4)
    draw.rounded_rectangle((24, 24, CARD_W - 24, CARD_H - 24), radius=30, outline=(255, 255, 255, 22), width=2)

    rank_box = (28, 28, 124, 124)
    draw.rounded_rectangle(rank_box, radius=28, fill=(9, 20, 25, 234), outline=(*accent_rgb, 120), width=2)
    rank_font = _fit_font(draw, str(entry.rank), 70, 54, 24, bold=True)
    draw.text(((rank_box[0] + rank_box[2]) // 2, 76), str(entry.rank), font=rank_font, fill="#f7fbff", anchor="mm")
    _draw_text(draw, 76, 102, "RANG", _load_font(15, bold=True), "#c6e8dd", anchor="mm")

    badge_box = (146, 34, CARD_W - 28, 114)
    draw.rounded_rectangle(badge_box, radius=20, fill=(9, 20, 25, 212), outline=(*accent_rgb, 95), width=2)
    badge_font = _fit_font(draw, entry.badge_label, badge_box[2] - badge_box[0] - 28, 28, 16, bold=True)
    _draw_text(draw, badge_box[0] + 18, 72, entry.badge_label, badge_font, "#d7fbf1", anchor="lm")

    photo_rect = (28, 138, CARD_W - 28, 834)
    photo_path = _resolve_player_image("", entry.player_name, photos_dir)
    if photo_path is not None:
        try:
            photo = ImageOps.exif_transpose(Image.open(photo_path)).convert("RGB")
            photo = ImageOps.fit(
                photo,
                (photo_rect[2] - photo_rect[0], photo_rect[3] - photo_rect[1]),
                method=Image.Resampling.LANCZOS,
                centering=(0.5, 0.16),
            )
            photo = ImageEnhance.Contrast(photo).enhance(1.08)
            photo = ImageEnhance.Brightness(photo).enhance(1.03)
            frame.alpha_composite(photo.convert("RGBA"), (photo_rect[0], photo_rect[1]))
        except Exception:
            _make_placeholder(draw, photo_rect, entry.player_name)
    else:
        _make_placeholder(draw, photo_rect, entry.player_name)

    fade = Image.new("RGBA", (CARD_W, CARD_H), (0, 0, 0, 0))
    fade_draw = ImageDraw.Draw(fade, "RGBA")
    fade_draw.rectangle((photo_rect[0], photo_rect[3] - 190, photo_rect[2], photo_rect[3]), fill=(7, 18, 24, 124))
    fade = fade.filter(ImageFilter.GaussianBlur(radius=18))
    frame.alpha_composite(fade)

    name_y = photo_rect[3] + 22
    name_font = _fit_font(draw, entry.player_name.upper(), CARD_W - 96, 42, 22, bold=True)
    name_lines = _wrap_lines(draw, entry.player_name.upper(), name_font, CARD_W - 96, 2)
    for idx, line in enumerate(name_lines):
        _draw_text(draw, 42, name_y + idx * 42, line, name_font, "#fff8ef")

    flag_path = _resolve_flag(entry.country_code, flags_dir)
    if flag_path is not None:
        try:
            flag = Image.open(flag_path).convert("RGBA").resize((62, 42), Image.Resampling.LANCZOS)
            frame.alpha_composite(flag, (42, name_y + 86))
        except Exception:
            pass

    meta_font = _load_font(23, bold=True)
    _draw_text(draw, 116, name_y + 108, f"{entry.titles} TITRES", meta_font, "#d7fbf1", anchor="lm")
    sub_font = _load_font(19, bold=False)
    _draw_text(draw, 42, name_y + 146, "SIMPLE MESSIEURS", sub_font, "#f2d7c8")

    years_panel = (28, 996, CARD_W - 28, 1188)
    draw.rounded_rectangle(years_panel, radius=24, fill=(10, 24, 21, 184), outline=(255, 255, 255, 16), width=2)
    years_title_font = _load_font(20, bold=True)
    years_text_font = _fit_font(draw, entry.years_won, CARD_W - 88, 22, 14, bold=True)
    _draw_text(draw, 42, years_panel[1] + 24, "ANNEES GAGNEES", years_title_font, "#f7cf94")
    years_lines = _wrap_lines(draw, entry.years_won, years_text_font, CARD_W - 88, 4)
    for idx, line in enumerate(years_lines):
        _draw_text(draw, 42, years_panel[1] + 64 + idx * 28, line, years_text_font, "#ffffff")

    summary_panel = (28, 1208, CARD_W - 28, 1336)
    draw.rounded_rectangle(summary_panel, radius=24, fill=(242, 232, 221, 244))
    left_box = (42, summary_panel[1] + 16, 170, summary_panel[3] - 16)
    right_box = (174, summary_panel[1] + 16, CARD_W - 42, summary_panel[3] - 16)
    _draw_text(draw, left_box[0], left_box[1], "PREMIER TITRE", _load_font(16, bold=True), "#8b6341")
    _draw_text(draw, left_box[0], left_box[1] + 34, entry.first_title, _load_font(28, bold=True), "#214451")
    _draw_text(draw, right_box[0], right_box[1], "DERNIER TITRE", _load_font(16, bold=True), "#8b6341")
    _draw_text(draw, right_box[0], right_box[1] + 34, entry.last_title, _load_font(28, bold=True), "#214451")

    return frame


def _make_header_bar(logo: Image.Image | None) -> Image.Image:
    bar = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    draw = ImageDraw.Draw(bar, "RGBA")
    draw.rounded_rectangle((36, 36, WIDTH - 36, 164), radius=34, fill=(8, 18, 20, 216), outline=(255, 255, 255, 20), width=2)

    title_font = _load_font(48, bold=True)
    subtitle_font = _load_font(22, bold=False)
    _draw_text(draw, 58, 68, "LES GEANTS DE ROLAND-GARROS", title_font, "#fff6ef")
    _draw_text(draw, 58, 120, "Top 12 messieurs de l'Open Era, classes par nombre de titres", subtitle_font, "#f0d9c7")

    logo_box = (WIDTH - 160, 46, WIDTH - 54, 152)
    draw.rounded_rectangle(logo_box, radius=28, fill=(255, 255, 255, 16), outline=(255, 255, 255, 30), width=2)
    if logo is not None:
        lx = logo_box[0] + (logo_box[2] - logo_box[0] - logo.width) // 2
        ly = logo_box[1] + (logo_box[3] - logo_box[1] - logo.height) // 2
        bar.alpha_composite(logo, (lx, ly))
    return bar


def render_video(
    input_csv: Path,
    output_path: Path,
    photos_dir: Path,
    flags_dir: Path,
    audio_path: Path,
    duration: float,
    fps: int,
) -> None:
    entries = load_entries(input_csv)
    if not entries:
        raise RuntimeError("No Roland-Garros entries to render.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    photos_dir.mkdir(parents=True, exist_ok=True)

    card_images = [_render_track_card(entry, photos_dir, flags_dir) for entry in entries]
    background = _background()
    header = _make_header_bar(_render_logo(84))

    track_w = len(card_images) * CARD_W + max(0, len(card_images) - 1) * CARD_GAP
    track = Image.new("RGBA", (track_w, CARD_H), (0, 0, 0, 0))
    x = 0
    for image in card_images:
        track.alpha_composite(image, (x, 0))
        x += CARD_W + CARD_GAP

    visible_w = VISIBLE_CARDS * CARD_W + (VISIBLE_CARDS - 1) * CARD_GAP
    rail_left = (WIDTH - visible_w) // 2
    rail_top = 194
    max_offset = max(0, track_w - visible_w)
    scroll_duration = max(0.1, duration - HOLD_START - HOLD_END)

    def make_frame(t: float) -> np.ndarray:
        frame = background.copy()
        if t <= HOLD_START:
            shift = 0.0
        elif t >= duration - HOLD_END:
            shift = float(max_offset)
        else:
            progress = (t - HOLD_START) / scroll_duration
            eased = 0.5 - 0.5 * math.cos(progress * math.pi)
            shift = max_offset * eased

        canvas = frame.copy()
        canvas.alpha_composite(header)
        viewport = track.crop((int(shift), 0, int(shift) + visible_w, CARD_H))
        canvas.alpha_composite(viewport, (rail_left, rail_top))
        return np.array(canvas.convert("RGB"))

    clip = VideoClip(make_frame, duration=duration)
    if audio_path.exists():
        audio_clip, keep_alive = build_audio_track(audio_path, duration)
        clip = clip.with_audio(audio_clip)
    else:
        audio_clip, keep_alive = None, []

    clip.write_videofile(
        str(output_path),
        fps=fps,
        codec="libx264",
        audio_codec="aac" if audio_clip else None,
    )

    clip.close()
    if audio_clip is not None:
        audio_clip.close()
    for item in keep_alive:
        item.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a Roland-Garros top winners Shorts cards video.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--photos-dir", type=Path, default=DEFAULT_PHOTOS_DIR)
    parser.add_argument("--flags-dir", type=Path, default=PROJECT_ROOT / "data" / "raw" / "flags")
    parser.add_argument("--audio", type=Path, default=DEFAULT_AUDIO)
    parser.add_argument("--duration", type=float, default=TOTAL_DURATION)
    parser.add_argument("--fps", type=int, default=FPS)
    args = parser.parse_args()

    render_video(
        input_csv=args.input,
        output_path=args.output,
        photos_dir=args.photos_dir,
        flags_dir=args.flags_dir,
        audio_path=args.audio,
        duration=args.duration,
        fps=args.fps,
    )
    print(f"[video_generator] Roland-Garros titles cards Shorts generated -> {args.output}")


if __name__ == "__main__":
    main()

