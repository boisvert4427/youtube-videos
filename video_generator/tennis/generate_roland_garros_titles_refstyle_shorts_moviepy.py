from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
from moviepy import VideoClip
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageOps

from video_generator.tennis.generate_atp_shorts_timeline_moviepy import (
    DEFAULT_AUDIO,
    FPS,
    HEIGHT,
    WIDTH,
    _fit_font,
    _load_font,
    _resolve_player_image,
    _truncate_to_width,
    build_audio_track,
)
from video_generator.tennis.generate_roland_garros_titles_cards_shorts_moviepy import (
    DEFAULT_INPUT,
    DEFAULT_LOGO,
    DEFAULT_PHOTOS_DIR,
    _background,
    _hex_to_rgb,
    _render_logo,
    _resolve_flag,
    load_entries,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "tennis" / "roland_garros_titles_cards_refstyle_shorts.mp4"

TOTAL_DURATION = 40.0
HOLD_START_SECONDS = 5.0
HOLD_END_SECONDS = 5.0
CARD_W = 720
CARD_H = 1560
CARD_GAP = 0
VISIBLE_CARDS = 1.5
HEADER_H = 116
CARD_TOP = 246
CARD_RADIUS = 38
PHOTO_H = 900
NAME_H = 160
BOTTOM_H = CARD_H - PHOTO_H - NAME_H


def _title_word(count: int) -> str:
    return "TITRE" if count == 1 else "TITRES"


def _parse_years(years_won: str) -> list[int]:
    years: list[int] = []
    for chunk in years_won.split("/"):
        token = chunk.strip()
        if token.isdigit():
            years.append(int(token))
    return sorted(set(years))


def _make_placeholder(draw: ImageDraw.ImageDraw, rect: tuple[int, int, int, int], player_name: str) -> None:
    draw.rounded_rectangle(rect, radius=32, fill=(235, 222, 205, 255))
    initials = "".join(part[0] for part in player_name.split()[:2]).upper()
    font = _fit_font(draw, initials, rect[2] - rect[0] - 24, 112, 46, bold=True)
    draw.text(((rect[0] + rect[2]) // 2, (rect[1] + rect[3]) // 2), initials, font=font, fill="#6d432d", anchor="mm")


def _wrap_name(draw: ImageDraw.ImageDraw, text: str, max_width: int, bold: bool = True) -> tuple[ImageFont.ImageFont, list[str]]:
    font = _fit_font(draw, text, max_width, 70, 34, bold=bold)
    if draw.textbbox((0, 0), text, font=font)[2] <= max_width:
        return font, [text]

    parts = [part for part in text.split() if part]
    if len(parts) <= 1:
        return font, [_truncate_to_width(draw, text, font, max_width)]

    first_line = " ".join(parts[:-1])
    second_line = parts[-1]
    while draw.textbbox((0, 0), first_line, font=font)[2] > max_width and len(first_line) > 1:
        pieces = first_line.split()
        if len(pieces) <= 1:
            break
        second_line = " ".join(pieces[-1:] + [second_line])
        first_line = " ".join(pieces[:-1])

    if draw.textbbox((0, 0), first_line, font=font)[2] > max_width:
        first_line = _truncate_to_width(draw, first_line, font, max_width)
    if draw.textbbox((0, 0), second_line, font=font)[2] > max_width:
        second_line = _truncate_to_width(draw, second_line, font, max_width)
    return font, [first_line, second_line]


def _draw_centered_lines(
    draw: ImageDraw.ImageDraw,
    center_x: int,
    top_y: int,
    lines: list[str],
    font,
    fill: str | tuple[int, int, int],
    line_gap: int = 8,
) -> None:
    if not lines:
        return
    bboxes = [draw.textbbox((0, 0), line, font=font) for line in lines]
    heights = [bbox[3] - bbox[1] for bbox in bboxes]
    total_h = sum(heights) + line_gap * max(0, len(lines) - 1)
    y = top_y + (0 if len(lines) == 1 else 0)
    for idx, line in enumerate(lines):
        draw.text((center_x, y), line, font=font, fill=fill, anchor="ma")
        y += heights[idx] + line_gap


def _render_card(entry, photos_dir: Path, flags_dir: Path) -> Image.Image:
    frame = Image.new("RGBA", (CARD_W, CARD_H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(frame, "RGBA")
    accent_rgb = _hex_to_rgb(entry.accent_color)
    card_rgb = _hex_to_rgb(entry.card_bg_color)

    shadow = Image.new("RGBA", frame.size, (0, 0, 0, 0))
    shadow_draw = ImageDraw.Draw(shadow, "RGBA")
    shadow_draw.rounded_rectangle((12, 12, CARD_W - 12, CARD_H - 12), radius=CARD_RADIUS, fill=(0, 0, 0, 100))
    shadow = shadow.filter(ImageFilter.GaussianBlur(radius=18))
    frame.alpha_composite(shadow)

    draw.rounded_rectangle(
        (0, 0, CARD_W - 1, CARD_H - 1),
        radius=CARD_RADIUS,
        fill=entry.card_bg_color,
        outline=(*accent_rgb, 240),
        width=4,
    )
    draw.rounded_rectangle(
        (12, 12, CARD_W - 13, CARD_H - 13),
        radius=CARD_RADIUS - 6,
        outline=(255, 255, 255, 18),
        width=1,
    )

    # Subtle glow and texture to keep the card premium without looking noisy.
    glow = Image.new("RGBA", frame.size, (0, 0, 0, 0))
    glow_draw = ImageDraw.Draw(glow, "RGBA")
    glow_draw.ellipse((CARD_W - 320, -120, CARD_W + 90, 250), fill=(*accent_rgb, 46))
    glow_draw.ellipse((-120, 240, 260, 620), fill=(*card_rgb, 26))
    glow_draw.ellipse((CARD_W - 420, CARD_H - 360, CARD_W + 130, CARD_H + 110), fill=(*accent_rgb, 20))
    glow = glow.filter(ImageFilter.GaussianBlur(radius=44))
    frame.alpha_composite(glow)

    # Photo block.
    photo_rect = (18, 18, CARD_W - 18, PHOTO_H)
    photo_path = _resolve_player_image("", entry.player_name, photos_dir)
    if photo_path is not None:
        try:
            photo = ImageOps.exif_transpose(Image.open(photo_path)).convert("RGB")
            photo = ImageOps.fit(
                photo,
                (photo_rect[2] - photo_rect[0], photo_rect[3] - photo_rect[1]),
                method=Image.Resampling.LANCZOS,
                centering=(0.5, 0.14),
            )
            photo = ImageEnhance.Contrast(photo).enhance(1.06)
            photo = ImageEnhance.Brightness(photo).enhance(1.02)
            frame.alpha_composite(photo.convert("RGBA"), (photo_rect[0], photo_rect[1]))
        except Exception:
            _make_placeholder(draw, photo_rect, entry.player_name)
    else:
        _make_placeholder(draw, photo_rect, entry.player_name)

    photo_fade = Image.new("RGBA", frame.size, (0, 0, 0, 0))
    photo_fade_draw = ImageDraw.Draw(photo_fade, "RGBA")
    photo_fade_draw.rectangle((photo_rect[0], photo_rect[3] - 210, photo_rect[2], photo_rect[3]), fill=(7, 18, 24, 132))
    photo_fade = photo_fade.filter(ImageFilter.GaussianBlur(radius=24))
    frame.alpha_composite(photo_fade)

    # Small corner badges.
    rank_box = (30, 38, 126, 126)
    draw.rounded_rectangle(rank_box, radius=24, fill=(9, 20, 25, 232), outline=(*accent_rgb, 110), width=2)
    rank_font = _fit_font(draw, str(entry.rank), rank_box[2] - rank_box[0] - 16, 58, 28, bold=True)
    draw.text(((rank_box[0] + rank_box[2]) // 2, 76), str(entry.rank), font=rank_font, fill="#f7fbff", anchor="mm")
    draw.text((78, 112), "RANG", font=_load_font(14, bold=True), fill="#c6e8dd", anchor="mm")

    flag_path = _resolve_flag(entry.country_code, flags_dir)
    if flag_path is not None:
        try:
            flag = Image.open(flag_path).convert("RGBA").resize((62, 42), Image.Resampling.LANCZOS)
            flag_box = (CARD_W - 120, 34)
            frame.alpha_composite(flag, flag_box)
        except Exception:
            pass

    # Name band, close to the reference video's big player-name stripe.
    name_band = (18, PHOTO_H - 22, CARD_W - 18, PHOTO_H + NAME_H)
    name_band_draw = Image.new("RGBA", frame.size, (0, 0, 0, 0))
    nbd = ImageDraw.Draw(name_band_draw, "RGBA")
    nbd.rounded_rectangle(name_band, radius=28, fill=(18, 34, 56, 236), outline=(255, 255, 255, 12), width=1)
    nbd.rounded_rectangle((name_band[0] + 10, name_band[1] + 10, name_band[2] - 10, name_band[3] - 10), radius=22, outline=(255, 255, 255, 14), width=1)
    frame.alpha_composite(name_band_draw)

    name_font, lines = _wrap_name(draw, entry.player_name.upper(), CARD_W - 92, bold=True)
    name_y = PHOTO_H + 14
    _draw_centered_lines(draw, CARD_W // 2, name_y, lines[:2], name_font, "#fff8ef", line_gap=4)

    # Bottom band: big title count on the left, years grid on the right.
    bottom_top = PHOTO_H + NAME_H - 10
    bottom_rect = (18, bottom_top, CARD_W - 18, CARD_H - 18)
    bottom_layer = Image.new("RGBA", frame.size, (0, 0, 0, 0))
    bdraw = ImageDraw.Draw(bottom_layer, "RGBA")
    bdraw.rounded_rectangle(bottom_rect, radius=28, fill=(224, 187, 122, 246), outline=(*accent_rgb, 110), width=2)
    bdraw.rounded_rectangle((bottom_rect[0] + 10, bottom_rect[1] + 10, bottom_rect[2] - 10, bottom_rect[3] - 10), radius=22, outline=(255, 255, 255, 10), width=1)
    frame.alpha_composite(bottom_layer)

    years = _parse_years(entry.years_won)
    if not years:
        years = [int(entry.first_title)]
    first_year = years[0]
    last_year = years[-1]
    # Top summary block.
    summary_box = (46, bottom_top + 26, CARD_W - 46, bottom_top + 228)
    draw.rounded_rectangle(summary_box, radius=24, fill=(245, 234, 220, 232), outline=(255, 255, 255, 20), width=1)
    summary_title_font = _load_font(16, bold=True)
    count_font = _fit_font(draw, f"{entry.titles}x", 132, 88, 54, bold=True)
    sub_font = _load_font(18, bold=True)
    tiny_font = _load_font(14, bold=True)
    draw.text((summary_box[0] + 18, summary_box[1] + 18), "TITRES", font=summary_title_font, fill="#8b6341")
    count_box = (summary_box[0] + 18, summary_box[1] + 40, summary_box[0] + 150, summary_box[1] + 140)
    draw.rounded_rectangle(count_box, radius=20, fill=(37, 64, 95, 184), outline=(255, 255, 255, 18), width=1)
    draw.text(((count_box[0] + count_box[2]) // 2, (count_box[1] + count_box[3]) // 2 - 2), f"{entry.titles}x", font=count_font, fill="#fffaf3", anchor="mm")
    logo_small = _render_logo(74) if DEFAULT_LOGO.exists() else None
    if logo_small is not None:
        sx = summary_box[0] + 172
        sy = summary_box[1] + 44
        frame.alpha_composite(logo_small, (sx, sy))
        draw.text((sx + 86, sy + 10), "ROLAND-GARROS", font=sub_font, fill="#2a2219")
        draw.text((sx + 86, sy + 42), f"{entry.titles} {_title_word(entry.titles)}", font=tiny_font, fill="#6f5237")
    else:
        draw.text((summary_box[0] + 172, summary_box[1] + 54), "ROLAND-GARROS", font=sub_font, fill="#2a2219")
        draw.text((summary_box[0] + 172, summary_box[1] + 86), f"{entry.titles} {_title_word(entry.titles)}", font=tiny_font, fill="#6f5237")

    # Years block underneath.
    right_box = (46, summary_box[3] + 18, CARD_W - 46, CARD_H - 88)
    draw.rounded_rectangle(right_box, radius=24, fill=(236, 214, 170, 228), outline=(255, 255, 255, 14), width=1)
    draw.text((right_box[0] + 18, right_box[1] + 16), "ANNEES GAGNEES", font=_load_font(18, bold=True), fill="#8b6341")

    pills_top = right_box[1] + 50
    pill_gap_x = 8
    pill_gap_y = 8
    pill_cols = 5
    pill_pad_x = 18
    pill_w = (right_box[2] - right_box[0] - (pill_pad_x * 2) - pill_gap_x * (pill_cols - 1)) // pill_cols
    pill_h = 34
    for idx, year in enumerate(years):
        row = idx // pill_cols
        col = idx % pill_cols
        x0 = right_box[0] + pill_pad_x + col * (pill_w + pill_gap_x)
        y0 = pills_top + row * (pill_h + pill_gap_y)
        rect = (x0, y0, x0 + pill_w, y0 + pill_h)
        pill_fill = (*accent_rgb, 214) if idx % 2 == 0 else (244, 232, 219, 232)
        text_fill = "#111a22" if idx % 2 == 0 else "#214451"
        draw.rounded_rectangle(rect, radius=16, fill=pill_fill, outline=(255, 255, 255, 14), width=1)
        year_font = _fit_font(draw, str(year), pill_w - 24, 28, 18, bold=True)
        draw.text(((rect[0] + rect[2]) // 2, (rect[1] + rect[3]) // 2), str(year), font=year_font, fill=text_fill, anchor="mm")

    return frame


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

    card_images = [_render_card(entry, photos_dir, flags_dir) for entry in entries]
    background = _background()

    track_w = len(card_images) * CARD_W + max(0, len(card_images) - 1) * CARD_GAP
    track = Image.new("RGBA", (track_w, CARD_H), (0, 0, 0, 0))
    x = 0
    for image in card_images:
        track.alpha_composite(image, (x, 0))
        x += CARD_W + CARD_GAP

    viewport_src_w = int(round(VISIBLE_CARDS * CARD_W + max(0.0, VISIBLE_CARDS - 1.0) * CARD_GAP))
    max_offset = max(0, track_w - viewport_src_w)
    scroll_duration = max(0.1, duration - HOLD_START_SECONDS - HOLD_END_SECONDS)
    rail_top = CARD_TOP

    header = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    header_draw = ImageDraw.Draw(header, "RGBA")
    header_draw.rounded_rectangle((36, 30, WIDTH - 36, 122), radius=28, fill=(8, 18, 20, 152), outline=(255, 255, 255, 18), width=1)
    header_draw.text((58, 48), "LES GEANTS DE ROLAND-GARROS", font=_load_font(45, bold=True), fill="#fff6ef")
    header_logo = _render_logo(58) if DEFAULT_LOGO.exists() else None
    if header_logo is not None:
        header.alpha_composite(header_logo, (WIDTH - 98, 47))

    footer = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    footer_draw = ImageDraw.Draw(footer, "RGBA")
    footer_draw.ellipse((-120, HEIGHT - 520, 440, HEIGHT + 120), fill=(255, 200, 124, 24))
    footer_draw.ellipse((WIDTH - 520, HEIGHT - 420, WIDTH + 120, HEIGHT + 180), fill=(88, 70, 120, 20))
    footer = footer.filter(ImageFilter.GaussianBlur(radius=48))

    static_base = background.copy()
    static_base.alpha_composite(header)
    static_base.alpha_composite(footer)

    def make_frame(t: float) -> np.ndarray:
        frame = static_base.copy()
        if t <= HOLD_START_SECONDS:
            shift = 0.0
        elif t >= duration - HOLD_END_SECONDS:
            shift = float(max_offset)
        else:
            progress = (t - HOLD_START_SECONDS) / scroll_duration
            shift = max_offset * progress

        canvas = frame.copy()
        viewport = track.crop((int(shift), 0, int(shift) + viewport_src_w, CARD_H))
        canvas.alpha_composite(viewport, (0, rail_top))

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
    parser = argparse.ArgumentParser(description="Generate a Roland-Garros ref-style cards Shorts video.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--photos-dir", type=Path, default=DEFAULT_PHOTOS_DIR)
    parser.add_argument("--flags-dir", type=Path, default=PROJECT_ROOT / "data" / "raw" / "flags")
    parser.add_argument("--audio", type=Path, default=DEFAULT_AUDIO)
    parser.add_argument("--duration", type=float, default=TOTAL_DURATION)
    parser.add_argument("--fps", type=int, default=FPS)
    args = parser.parse_args()
    fps = min(args.fps, 60)

    render_video(
        input_csv=args.input,
        output_path=args.output,
        photos_dir=args.photos_dir,
        flags_dir=args.flags_dir,
        audio_path=args.audio,
        duration=args.duration,
        fps=fps,
    )
    print(f"[video_generator] Roland-Garros ref-style cards Shorts generated -> {args.output}")


if __name__ == "__main__":
    main()
