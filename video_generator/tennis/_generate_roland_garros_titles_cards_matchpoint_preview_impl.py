from __future__ import annotations

import argparse
import math
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
from moviepy import CompositeVideoClip, ImageClip, VideoFileClip, concatenate_videoclips
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageOps

from video_generator.tennis.generate_atp_shorts_timeline_moviepy import (
    _fit_font,
    _load_font,
    _resolve_player_image,
    _truncate_to_width,
)
from video_generator.tennis.generate_roland_garros_titles_cards_shorts_moviepy import (
    DEFAULT_INPUT,
    DEFAULT_PHOTOS_DIR,
    _background,
    _hex_to_rgb,
    _render_logo,
    _resolve_flag,
    load_entries,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "tennis" / "roland_garros_titles_cards_matchpoint_preview.mp4"
DEFAULT_CACHE_DIR = PROJECT_ROOT / "tmp" / "roland_garros_matchpoint_preview"
DEFAULT_URL = "https://www.youtube.com/watch?v=Fkv_NJLsvAU"
DEFAULT_START = "9:08"
DEFAULT_END = "9:25"
DEFAULT_FOCUS_PLAYER = "Nadal"

WIDTH = 1080
HEIGHT = 1920
FPS = 30
PREVIEW_DURATION = 18.0
SCENE_DURATION = 2.5

CARD_COUNT = 5
CARD_W = WIDTH
CARD_H = HEIGHT // 2
CARD_X = 0
CARD_Y = 0

PANEL_W = WIDTH
PANEL_H = HEIGHT - CARD_H
PANEL_X = 0
PANEL_Y = CARD_H
PANEL_INNER_PAD = 0


def _slugify(value: str) -> str:
    return re.sub(r"[^0-9A-Za-z]+", "_", value).strip("_").lower() or "clip"


def _extract_youtube_id(url: str) -> str:
    patterns = [
        r"[?&]v=([0-9A-Za-z_-]{11})",
        r"youtu\.be/([0-9A-Za-z_-]{11})",
        r"shorts/([0-9A-Za-z_-]{11})",
        r"embed/([0-9A-Za-z_-]{11})",
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return "youtube"


def _hex_to_rgb(value: str) -> tuple[int, int, int]:
    cleaned = value.lstrip("#")
    if len(cleaned) != 6:
        raise ValueError(f"Unsupported color value: {value}")
    return tuple(int(cleaned[index : index + 2], 16) for index in (0, 2, 4))


def _parse_time_to_seconds(value: str) -> float:
    parts = [float(part) for part in value.split(":")]
    if len(parts) == 2:
        return parts[0] * 60.0 + parts[1]
    if len(parts) == 3:
        return parts[0] * 3600.0 + parts[1] * 60.0 + parts[2]
    raise ValueError(f"Unsupported time format: {value}")


def _format_seconds(value: float) -> str:
    total_seconds = max(0, int(round(value)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _make_header_bar(logo: Image.Image | None) -> Image.Image:
    header = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    header_draw = ImageDraw.Draw(header, "RGBA")
    header_draw.rounded_rectangle((36, 30, WIDTH - 36, 122), radius=28, fill=(8, 18, 20, 152), outline=(255, 255, 255, 18), width=1)
    header_draw.text((58, 48), "LES GEANTS DE ROLAND-GARROS", font=_load_font(45, bold=True), fill="#fff6ef")
    if logo is not None:
        header.alpha_composite(logo, (WIDTH - 98, 47))
    return header


def _download_youtube_section(url: str, start: str, end: str, cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)

    start_seconds = _parse_time_to_seconds(start)
    end_seconds = _parse_time_to_seconds(end)
    if end_seconds <= start_seconds:
        raise ValueError(f"End time must be greater than start time: {start} -> {end}")

    start_tag = _format_seconds(start_seconds)
    end_tag = _format_seconds(end_seconds)
    video_id = _extract_youtube_id(url)
    safe_start = _slugify(start_tag)
    safe_end = _slugify(end_tag)
    target = cache_dir / f"{video_id}_{safe_start}_{safe_end}.mp4"
    if target.exists():
        return target

    output_template = str(cache_dir / f"{video_id}_{safe_start}_{safe_end}.%(ext)s")
    cmd = [
        sys.executable,
        "-m",
        "yt_dlp",
        "--no-playlist",
        "--quiet",
        "--no-warnings",
        "--force-keyframes-at-cuts",
        "--download-sections",
        f"*{start_tag}-{end_tag}",
        "-f",
        "bv*+ba/b",
        "--merge-output-format",
        "mp4",
        "-o",
        output_template,
        url,
    ]
    subprocess.run(cmd, check=True)

    if target.exists():
        return target

    candidates = [
        path
        for path in cache_dir.glob(f"{video_id}_{safe_start}_{safe_end}.*")
        if path.suffix.lower() not in {".part", ".ytdl", ".json"}
    ]
    if not candidates:
        raise RuntimeError("yt-dlp finished but no downloaded video file was found.")
    candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return candidates[0]


def _fit_clip_to_panel(clip: VideoFileClip, panel_w: int, panel_h: int) -> VideoFileClip:
    scale = max(panel_w / clip.w, panel_h / clip.h)
    resized = clip.resized((max(1, int(math.ceil(clip.w * scale))), max(1, int(math.ceil(clip.h * scale)))))
    x1 = max(0, int((resized.w - panel_w) / 2))
    y1 = max(0, int((resized.h - panel_h) / 2))
    x2 = x1 + panel_w
    y2 = y1 + panel_h
    return resized.cropped(x1=x1, y1=y1, x2=x2, y2=y2)


def _parse_years(years_won: str) -> list[int]:
    years: list[int] = []
    for chunk in years_won.split("/"):
        token = chunk.strip()
        if token.isdigit():
            years.append(int(token))
    return sorted(set(years))


def _draw_centered_lines(
    draw: ImageDraw.ImageDraw,
    center_x: int,
    top_y: int,
    lines: list[str],
    font,
    fill: str | tuple[int, int, int],
    line_gap: int = 6,
) -> None:
    if not lines:
        return
    y = top_y
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        draw.text((center_x, y), line, font=font, fill=fill, anchor="ma")
        y += (bbox[3] - bbox[1]) + line_gap


def _wrap_player_name(draw: ImageDraw.ImageDraw, text: str, max_width: int, bold: bool = True) -> tuple[Any, list[str]]:
    font = _fit_font(draw, text, max_width, 76, 40, bold=bold)
    if draw.textbbox((0, 0), text, font=font)[2] <= max_width:
        return font, [text]

    words = [word for word in text.split() if word]
    if len(words) <= 1:
        return font, [_truncate_to_width(draw, text, font, max_width)]

    first_line = " ".join(words[:-1])
    second_line = words[-1]
    while draw.textbbox((0, 0), first_line, font=font)[2] > max_width and len(first_line) > 1:
        parts = first_line.split()
        if len(parts) <= 1:
            break
        second_line = " ".join(parts[-1:] + [second_line])
        first_line = " ".join(parts[:-1])

    if draw.textbbox((0, 0), first_line, font=font)[2] > max_width:
        first_line = _truncate_to_width(draw, first_line, font, max_width)
    if draw.textbbox((0, 0), second_line, font=font)[2] > max_width:
        second_line = _truncate_to_width(draw, second_line, font, max_width)
    return font, [first_line, second_line]


def _make_placeholder_photo(draw: ImageDraw.ImageDraw, rect: tuple[int, int, int, int], player_name: str, accent_rgb: tuple[int, int, int]) -> None:
    draw.rounded_rectangle(rect, radius=34, fill=(23, 31, 43, 255), outline=(*accent_rgb, 160), width=2)
    initials = "".join(part[0] for part in player_name.split()[:2]).upper()
    font = _fit_font(draw, initials, rect[2] - rect[0] - 24, 132, 54, bold=True)
    draw.text(((rect[0] + rect[2]) // 2, (rect[1] + rect[3]) // 2), initials, font=font, fill="#f9f2ea", anchor="mm")


def _render_wide_card(entry: Any, photos_dir: Path, flags_dir: Path) -> Image.Image:
    frame = Image.new("RGBA", (CARD_W, CARD_H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(frame, "RGBA")
    accent_rgb = _hex_to_rgb(entry.accent_color)
    card_rgb = _hex_to_rgb(entry.card_bg_color)

    draw.rectangle((0, 0, CARD_W, CARD_H), fill=entry.card_bg_color)

    glow = Image.new("RGBA", frame.size, (0, 0, 0, 0))
    glow_draw = ImageDraw.Draw(glow, "RGBA")
    glow_draw.ellipse((-220, -180, 560, 520), fill=(*accent_rgb, 56))
    glow_draw.ellipse((CARD_W - 520, -140, CARD_W + 180, 420), fill=(*card_rgb, 36))
    glow_draw.ellipse((CARD_W - 380, CARD_H - 320, CARD_W + 140, CARD_H + 160), fill=(*accent_rgb, 24))
    glow = glow.filter(ImageFilter.GaussianBlur(radius=66))
    frame.alpha_composite(glow)

    draw.rectangle((0, 0, CARD_W - 1, CARD_H - 1), outline=(*accent_rgb, 155), width=4)
    draw.rectangle((14, 14, CARD_W - 15, CARD_H - 15), outline=(255, 255, 255, 10), width=1)

    photo_rect = (24, 24, 410, CARD_H - 24)
    draw.rounded_rectangle(photo_rect, radius=34, fill=(17, 23, 32, 255), outline=(255, 255, 255, 16), width=1)
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
            photo = ImageEnhance.Contrast(photo).enhance(1.05)
            photo = ImageEnhance.Brightness(photo).enhance(1.02)
            photo_layer = Image.new("RGBA", frame.size, (0, 0, 0, 0))
            photo_mask = Image.new("L", (photo_rect[2] - photo_rect[0], photo_rect[3] - photo_rect[1]), 0)
            ImageDraw.Draw(photo_mask).rounded_rectangle(
                (0, 0, photo_mask.width - 1, photo_mask.height - 1),
                radius=34,
                fill=255,
            )
            photo_layer.paste(photo.convert("RGBA"), (photo_rect[0], photo_rect[1]), photo_mask)
            frame.alpha_composite(photo_layer)
        except Exception:
            _make_placeholder_photo(draw, photo_rect, entry.player_name, accent_rgb)
    else:
        _make_placeholder_photo(draw, photo_rect, entry.player_name, accent_rgb)

    photo_fade = Image.new("RGBA", frame.size, (0, 0, 0, 0))
    photo_fade_draw = ImageDraw.Draw(photo_fade, "RGBA")
    photo_fade_draw.rectangle((photo_rect[0], photo_rect[3] - 220, photo_rect[2], photo_rect[3]), fill=(7, 15, 24, 138))
    photo_fade = photo_fade.filter(ImageFilter.GaussianBlur(radius=22))
    frame.alpha_composite(photo_fade)

    rank_box = (42, 42, 156, 156)
    draw.rounded_rectangle(rank_box, radius=28, fill=(12, 18, 25, 224), outline=(*accent_rgb, 120), width=2)
    rank_font = _fit_font(draw, str(entry.rank), rank_box[2] - rank_box[0] - 18, 60, 28, bold=True)
    draw.text(((rank_box[0] + rank_box[2]) // 2, rank_box[1] + 32), str(entry.rank), font=rank_font, fill="#f8fbff", anchor="ma")
    draw.text((rank_box[0] + 16, rank_box[3] - 20), "RANG", font=_load_font(14, bold=True), fill="#c9dfef", anchor="ls")

    right_x = photo_rect[2] + 28
    right_w = CARD_W - right_x - 24
    title_font, title_lines = _wrap_player_name(draw, entry.player_name.upper(), right_w, bold=True)
    title_top = 34
    _draw_centered_lines(draw, right_x + right_w // 2, title_top, title_lines[:2], title_font, "#fff8ef", line_gap=4)

    name_bottom = title_top
    for line in title_lines[:2]:
        bbox = draw.textbbox((0, 0), line, font=title_font)
        name_bottom += (bbox[3] - bbox[1]) + 4

    flag_path = _resolve_flag(entry.country_code, flags_dir)
    if flag_path is not None:
        try:
            flag = Image.open(flag_path).convert("RGBA").resize((72, 48), Image.Resampling.LANCZOS)
            frame.alpha_composite(flag, (CARD_W - 96, 42))
        except Exception:
            pass

    label_font = _load_font(18, bold=True)
    sub_font = _load_font(16, bold=True)
    count_box = (right_x, max(name_bottom + 22, 196), CARD_W - 24, max(name_bottom + 22, 196) + 198)
    draw.rounded_rectangle(count_box, radius=30, fill=(240, 225, 201, 236), outline=(*accent_rgb, 120), width=2)
    count_strip = (count_box[0] + 14, count_box[1] + 14, count_box[0] + 188, count_box[3] - 14)
    draw.rounded_rectangle(count_strip, radius=24, fill=(28, 48, 70, 248), outline=(255, 255, 255, 16), width=1)
    count_font = _fit_font(draw, f"{entry.titles}x", count_strip[2] - count_strip[0] - 10, 100, 54, bold=True)
    draw.text(((count_strip[0] + count_strip[2]) // 2, (count_strip[1] + count_strip[3]) // 2 - 2), f"{entry.titles}x", font=count_font, fill="#fffaf3", anchor="mm")
    draw.text((count_box[0] + 214, count_box[1] + 28), "ROLAND-GARROS", font=label_font, fill="#8a6340")
    draw.text((count_box[0] + 214, count_box[1] + 68), entry.badge_label, font=sub_font, fill="#2f485d")
    if entry.first_title != entry.last_title:
        draw.text((count_box[0] + 214, count_box[1] + 106), f"{entry.first_title} -> {entry.last_title}", font=sub_font, fill="#44606c")
    else:
        draw.text((count_box[0] + 214, count_box[1] + 106), f"Depuis {entry.first_title}", font=sub_font, fill="#44606c")

    years = _parse_years(entry.years_won)
    if not years:
        years = [int(entry.first_title)]

    years_box = (right_x, count_box[3] + 18, CARD_W - 24, CARD_H - 24)
    draw.rounded_rectangle(years_box, radius=30, fill=(20, 29, 40, 222), outline=(255, 255, 255, 12), width=1)
    draw.text((years_box[0] + 18, years_box[1] + 16), "ANNEES GAGNEES", font=_load_font(18, bold=True), fill="#c5d9e7")

    if len(years) > 1:
        draw.text((years_box[0] + 18, years_box[1] + 48), f"{years[0]} -> {years[-1]}", font=_load_font(14, bold=True), fill="#91a7b6")
    else:
        draw.text((years_box[0] + 18, years_box[1] + 48), str(years[0]), font=_load_font(14, bold=True), fill="#91a7b6")

    grid_top = years_box[1] + 88
    grid_left = years_box[0] + 18
    grid_w = years_box[2] - years_box[0] - 36
    pill_cols = 5
    pill_gap_x = 8
    pill_gap_y = 8
    pill_w = max(64, (grid_w - pill_gap_x * (pill_cols - 1)) // pill_cols)
    pill_h = 34
    for idx, year in enumerate(years):
        row = idx // pill_cols
        col = idx % pill_cols
        x0 = grid_left + col * (pill_w + pill_gap_x)
        y0 = grid_top + row * (pill_h + pill_gap_y)
        rect = (x0, y0, x0 + pill_w, y0 + pill_h)
        pill_fill = (*accent_rgb, 214) if idx % 2 == 0 else (239, 230, 217, 234)
        text_fill = "#12212f" if idx % 2 == 0 else "#203f56"
        draw.rounded_rectangle(rect, radius=16, fill=pill_fill, outline=(255, 255, 255, 14), width=1)
        year_font = _fit_font(draw, str(year), pill_w - 20, 28, 18, bold=True)
        draw.text(((rect[0] + rect[2]) // 2, (rect[1] + rect[3]) // 2), str(year), font=year_font, fill=text_fill, anchor="mm")

    return frame


def _render_panel_background(width: int, height: int, accent: tuple[int, int, int], active: bool) -> Image.Image:
    frame = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(frame, "RGBA")
    draw.rectangle((0, 0, width - 1, height - 1), fill=(4, 6, 10, 255))
    draw.rectangle((0, 0, width - 1, 5), fill=(*accent, 64 if active else 28))
    draw.rectangle((0, 0, width - 1, height - 1), outline=(*accent, 90 if active else 18), width=2 if active else 1)

    if active:
        glow = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        glow_draw = ImageDraw.Draw(glow, "RGBA")
        glow_draw.ellipse((width - 280, -140, width + 180, 280), fill=(*accent, 42))
        glow_draw.ellipse((-160, height - 260, 260, height + 120), fill=(255, 255, 255, 12))
        glow = glow.filter(ImageFilter.GaussianBlur(radius=44))
        frame.alpha_composite(glow)

    return frame


def _prepare_entries(input_csv: Path, top_n: int) -> list:
    entries = sorted(load_entries(input_csv), key=lambda entry: entry.rank)
    return entries[:top_n]


def _render_slide_base(entry: Any, photos_dir: Path, flags_dir: Path, active: bool) -> Image.Image:
    background = _background()
    base = background.copy()
    card = _render_wide_card(entry, photos_dir, flags_dir)
    base.alpha_composite(card, (CARD_X, CARD_Y))

    try:
        accent = _hex_to_rgb(entry.accent_color)
    except Exception:
        accent = (255, 198, 120)
    panel_bg = _render_panel_background(PANEL_W, PANEL_H, accent, active)
    base.alpha_composite(panel_bg, (PANEL_X, PANEL_Y))
    return base


def _find_focus_entry(entries: list, focus_player: str) -> tuple[int, Any]:
    target = focus_player.lower().strip()
    for index, entry in enumerate(entries):
        if target in entry.player_name.lower():
            return index, entry
    available = ", ".join(entry.player_name for entry in entries)
    raise RuntimeError(f"Could not find {focus_player!r} among the selected entries: {available}")


def render_video(
    input_csv: Path,
    output_path: Path,
    photos_dir: Path,
    flags_dir: Path,
    url: str,
    start: str,
    end: str,
    cache_dir: Path,
    duration: float,
    scene_duration: float,
    fps: int,
    top_n: int,
    focus_player: str,
) -> Path:
    entries = _prepare_entries(input_csv, top_n)
    if len(entries) < top_n:
        raise RuntimeError(f"Expected at least {top_n} Roland-Garros entries.")

    _, focus_entry = _find_focus_entry(entries, focus_player)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    photos_dir.mkdir(parents=True, exist_ok=True)
    flags_dir.mkdir(parents=True, exist_ok=True)

    clip_path = _download_youtube_section(url, start, end, cache_dir)
    match_clip = VideoFileClip(str(clip_path))
    usable_duration = min(duration, match_clip.duration)
    if usable_duration <= 0:
        raise RuntimeError("Downloaded clip has no usable duration.")
    match_clip = match_clip.subclipped(0, usable_duration)

    slides: list[CompositeVideoClip | ImageClip] = []
    focus_slide: CompositeVideoClip | None = None
    try:
        for entry in reversed(entries):
            active = entry == focus_entry
            slide_duration = usable_duration if active else scene_duration
            base = _render_slide_base(entry, photos_dir, flags_dir, active)
            base_clip = ImageClip(np.array(base.convert("RGB"))).with_duration(slide_duration)
            if active:
                active_x = PANEL_X + PANEL_INNER_PAD
                active_y = PANEL_Y + PANEL_INNER_PAD
                active_w = PANEL_W - PANEL_INNER_PAD * 2
                active_h = PANEL_H - PANEL_INNER_PAD * 2
                active_clip = _fit_clip_to_panel(match_clip, active_w, active_h).with_position((active_x, active_y))
                focus_slide = CompositeVideoClip([base_clip, active_clip], size=(WIDTH, HEIGHT)).with_duration(slide_duration)
                if active_clip.audio is not None:
                    focus_slide = focus_slide.with_audio(active_clip.audio)
                slides.append(focus_slide)
            else:
                slides.append(base_clip)

        final_clip = concatenate_videoclips(slides, method="compose")
        audio_codec = "aac" if final_clip.audio is not None else None
        final_clip.write_videofile(
            str(output_path),
            fps=fps,
            codec="libx264",
            audio_codec=audio_codec,
        )
        final_clip.close()
    finally:
        for clip in slides:
            try:
                clip.close()
            except Exception:
                pass
        try:
            if focus_slide is not None:
                focus_slide.close()
        except Exception:
            pass
        match_clip.close()
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a Roland-Garros 50/50 split preview video.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--photos-dir", type=Path, default=DEFAULT_PHOTOS_DIR)
    parser.add_argument("--flags-dir", type=Path, default=PROJECT_ROOT / "data" / "raw" / "flags")
    parser.add_argument("--url", type=str, default=DEFAULT_URL)
    parser.add_argument("--start", type=str, default=DEFAULT_START)
    parser.add_argument("--end", type=str, default=DEFAULT_END)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--duration", type=float, default=PREVIEW_DURATION)
    parser.add_argument("--scene-duration", type=float, default=SCENE_DURATION)
    parser.add_argument("--fps", type=int, default=FPS)
    parser.add_argument("--top-n", type=int, default=CARD_COUNT)
    parser.add_argument("--focus-player", type=str, default=DEFAULT_FOCUS_PLAYER)
    args = parser.parse_args()

    render_video(
        input_csv=args.input,
        output_path=args.output,
        photos_dir=args.photos_dir,
        flags_dir=args.flags_dir,
        url=args.url,
        start=args.start,
        end=args.end,
        cache_dir=args.cache_dir,
        duration=args.duration,
        scene_duration=args.scene_duration,
        fps=args.fps,
        top_n=args.top_n,
        focus_player=args.focus_player,
    )
    print(f"[video_generator] Roland-Garros 50/50 split preview generated -> {args.output}")


if __name__ == "__main__":
    main()
