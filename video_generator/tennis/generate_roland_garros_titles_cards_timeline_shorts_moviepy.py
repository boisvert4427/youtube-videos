from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
from moviepy import VideoClip
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageEnhance, ImageOps

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
from video_generator.tennis.generate_roland_garros_titles_cards_shorts_moviepy import (
    DEFAULT_INPUT,
    DEFAULT_LOGO,
    DEFAULT_PHOTOS_DIR,
    _background,
    _draw_text,
    _hex_to_rgb,
    _render_logo,
    _resolve_flag,
    load_entries,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "tennis" / "roland_garros_titles_cards_timeline_shorts.mp4"

TOTAL_DURATION = 48.0
CARD_W = 720
CARD_H = 560
CARD_GAP = 0
VISIBLE_CARDS = 1.5
CARD_MARGIN = 28
CARD_RADIUS = 36
PANEL_MARGIN = 34
PANEL_GAP = 24


def _title_word(count: int) -> str:
    return "TITRE" if count == 1 else "TITRES"


def _parse_years(years_won: str) -> list[int]:
    years: list[int] = []
    for chunk in years_won.split("/"):
        token = chunk.strip()
        if token.isdigit():
            years.append(int(token))
    return sorted(set(years))


def _wrap_tokens(draw: ImageDraw.ImageDraw, tokens: list[str], font, max_width: int, max_lines: int) -> list[str]:
    lines: list[str] = []
    current: list[str] = []
    token_index = 0

    while token_index < len(tokens):
        token = tokens[token_index]
        candidate_tokens = current + [token]
        candidate = " / ".join(candidate_tokens)
        if not current or draw.textbbox((0, 0), candidate, font=font)[2] <= max_width:
            current = candidate_tokens
            token_index += 1
            continue

        lines.append(" / ".join(current))
        current = []
        if len(lines) == max_lines - 1:
            break

    remaining = tokens[token_index:]
    if current and len(lines) < max_lines:
        lines.append(" / ".join(current))
    elif remaining and len(lines) < max_lines:
        tail = " / ".join(remaining)
        if current:
            tail = " / ".join(current + remaining)
        lines.append(tail)

    if len(lines) > max_lines:
        lines = lines[:max_lines]
    return lines


def _make_placeholder(draw: ImageDraw.ImageDraw, rect: tuple[int, int, int, int], player_name: str) -> None:
    draw.rounded_rectangle(rect, radius=32, fill=(239, 225, 208, 255))
    initials = "".join(part[0] for part in player_name.split()[:2]).upper()
    font = _fit_font(draw, initials, rect[2] - rect[0] - 24, 110, 44, bold=True)
    draw.text(((rect[0] + rect[2]) // 2, (rect[1] + rect[3]) // 2), initials, font=font, fill="#6d432d", anchor="mm")


def _render_wide_track_card(entry, photos_dir: Path, flags_dir: Path) -> Image.Image:
    frame = Image.new("RGBA", (CARD_W, CARD_H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(frame, "RGBA")
    accent_rgb = _hex_to_rgb(entry.accent_color)
    card_rgb = _hex_to_rgb(entry.card_bg_color)

    shadow = Image.new("RGBA", frame.size, (0, 0, 0, 0))
    shadow_draw = ImageDraw.Draw(shadow, "RGBA")
    shadow_draw.rounded_rectangle((12, 12, CARD_W - 12, CARD_H - 12), radius=CARD_RADIUS, fill=(0, 0, 0, 82))
    shadow = shadow.filter(ImageFilter.GaussianBlur(radius=16))
    frame.alpha_composite(shadow)

    draw.rounded_rectangle(
        (0, 0, CARD_W - 1, CARD_H - 1),
        radius=CARD_RADIUS,
        fill=entry.card_bg_color,
        outline=(*accent_rgb, 255),
        width=4,
    )
    draw.rounded_rectangle(
        (12, 12, CARD_W - 13, CARD_H - 13),
        radius=CARD_RADIUS - 6,
        outline=(255, 255, 255, 18),
        width=1,
    )

    glow = Image.new("RGBA", frame.size, (0, 0, 0, 0))
    glow_draw = ImageDraw.Draw(glow, "RGBA")
    glow_draw.ellipse((CARD_W - 260, -120, CARD_W + 80, 220), fill=(*accent_rgb, 48))
    glow_draw.ellipse((-120, 240, 220, 560), fill=(*card_rgb, 30))
    glow = glow.filter(ImageFilter.GaussianBlur(radius=44))
    frame.alpha_composite(glow)

    rank_box = (CARD_MARGIN, 28, 120, 110)
    draw.rounded_rectangle(rank_box, radius=24, fill=(9, 20, 25, 234), outline=(*accent_rgb, 120), width=2)
    rank_font = _fit_font(draw, str(entry.rank), rank_box[2] - rank_box[0] - 14, 56, 28, bold=True)
    draw.text(((rank_box[0] + rank_box[2]) // 2, 67), str(entry.rank), font=rank_font, fill="#f7fbff", anchor="mm")
    draw.text((74, 96), "RANG", font=_load_font(15, bold=True), fill="#c6e8dd", anchor="mm")

    badge_box = (146, 30, CARD_W - CARD_MARGIN, 90)
    draw.rounded_rectangle(badge_box, radius=22, fill=(9, 20, 25, 212), outline=(*accent_rgb, 92), width=2)
    badge_font = _fit_font(draw, entry.badge_label, badge_box[2] - badge_box[0] - 28, 30, 15, bold=True)
    draw.text((badge_box[0] + 18, 60), entry.badge_label, font=badge_font, fill="#d7fbf1", anchor="lm")

    photo_rect = (CARD_MARGIN, 124, 284, CARD_H - 42)
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

    fade = Image.new("RGBA", frame.size, (0, 0, 0, 0))
    fade_draw = ImageDraw.Draw(fade, "RGBA")
    fade_draw.rectangle((photo_rect[0], photo_rect[3] - 140, photo_rect[2], photo_rect[3]), fill=(7, 18, 24, 110))
    fade = fade.filter(ImageFilter.GaussianBlur(radius=16))
    frame.alpha_composite(fade)

    right_x = 314
    right_w = CARD_W - right_x - CARD_MARGIN
    name_y = 124
    name_font = _fit_font(draw, entry.player_name.upper(), right_w, 44, 24, bold=True)
    player_name = _truncate_to_width(draw, entry.player_name.upper(), name_font, right_w)
    draw.text((right_x, name_y), player_name, font=name_font, fill="#fff8ef")

    flag_path = _resolve_flag(entry.country_code, flags_dir)
    flag_x = right_x
    titles_y = 190
    if flag_path is not None:
        try:
            flag = Image.open(flag_path).convert("RGBA").resize((62, 42), Image.Resampling.LANCZOS)
            frame.alpha_composite(flag, (flag_x, titles_y))
            flag_x += 78
        except Exception:
            pass

    title_pill = (flag_x, titles_y, CARD_W - CARD_MARGIN, titles_y + 44)
    draw.rounded_rectangle(title_pill, radius=18, fill=(*accent_rgb, 250))
    titles_font = _fit_font(draw, f"{entry.titles} {_title_word(entry.titles)}", title_pill[2] - title_pill[0] - 28, 30, 16, bold=True)
    draw.text(((title_pill[0] + title_pill[2]) // 2, titles_y + 22), f"{entry.titles} {_title_word(entry.titles)}", font=titles_font, fill="#111a22", anchor="mm")

    years = _parse_years(entry.years_won)
    if not years:
        years = [int(entry.first_title)]
    years_box = (right_x, 248, CARD_W - CARD_MARGIN, 398)
    draw.rounded_rectangle(years_box, radius=22, fill=(255, 255, 255, 10), outline=(255, 255, 255, 16), width=1)
    draw.text((right_x + 18, 270), "ANNEES GAGNEES", font=_load_font(16, bold=True), fill="#f7cf94")
    years_font = _fit_font(draw, " / ".join(str(year) for year in years[:4]), years_box[2] - years_box[0] - 40, 18, 13, bold=True)
    year_lines = _wrap_tokens(draw, [str(year) for year in years], years_font, years_box[2] - years_box[0] - 40, 4)
    for idx, line in enumerate(year_lines):
        draw.text((right_x + 18, 308 + idx * 26), line, font=years_font, fill="#ffffff")

    first_box = (right_x, 424, right_x + 186, CARD_H - 42)
    last_box = (right_x + 204, 424, CARD_W - CARD_MARGIN, CARD_H - 42)
    draw.rounded_rectangle(first_box, radius=18, fill=(242, 232, 221, 244))
    draw.rounded_rectangle(last_box, radius=18, fill=(242, 232, 221, 244))
    stat_title_font = _load_font(14, bold=True)
    stat_value_font = _fit_font(draw, entry.first_title, first_box[2] - first_box[0] - 24, 32, 20, bold=True)
    draw.text((first_box[0] + 16, first_box[1] + 14), "PREMIER TITRE", font=stat_title_font, fill="#8b6341")
    draw.text((first_box[0] + 16, first_box[1] + 44), entry.first_title, font=stat_value_font, fill="#214451")
    draw.text((last_box[0] + 16, last_box[1] + 14), "DERNIER TITRE", font=stat_title_font, fill="#8b6341")
    draw.text((last_box[0] + 16, last_box[1] + 44), entry.last_title, font=stat_value_font, fill="#214451")

    return frame


def _make_panel_base(accent_color: str, card_bg_color: str, panel_h: int) -> Image.Image:
    panel_w = WIDTH - PANEL_MARGIN * 2
    panel = Image.new("RGBA", (panel_w, panel_h), (0, 0, 0, 0))

    shadow = Image.new("RGBA", panel.size, (0, 0, 0, 0))
    shadow_draw = ImageDraw.Draw(shadow, "RGBA")
    shadow_draw.rounded_rectangle((10, 10, panel_w - 10, panel_h - 10), radius=30, fill=(0, 0, 0, 86))
    shadow = shadow.filter(ImageFilter.GaussianBlur(radius=16))
    panel.alpha_composite(shadow)

    base_draw = ImageDraw.Draw(panel, "RGBA")
    accent_rgb = _hex_to_rgb(accent_color)
    card_rgb = _hex_to_rgb(card_bg_color)

    base_draw.rounded_rectangle(
        (0, 0, panel_w - 1, panel_h - 1),
        radius=30,
        fill=(11, 15, 20, 232),
        outline=(*accent_rgb, 120),
        width=2,
    )
    base_draw.rounded_rectangle(
        (8, 8, panel_w - 9, panel_h - 9),
        radius=24,
        outline=(255, 255, 255, 14),
        width=1,
    )

    glow = Image.new("RGBA", panel.size, (0, 0, 0, 0))
    glow_draw = ImageDraw.Draw(glow, "RGBA")
    glow_draw.ellipse((panel_w - 340, -130, panel_w + 110, 220), fill=(*accent_rgb, 50))
    glow_draw.ellipse((-120, 120, 230, 420), fill=(*card_rgb, 30))
    glow_draw.ellipse((panel_w - 420, panel_h - 340, panel_w + 120, panel_h + 150), fill=(*accent_rgb, 26))
    glow = glow.filter(ImageFilter.GaussianBlur(radius=42))
    panel.alpha_composite(glow)

    return panel


def _render_timeline_panel(entry, flags_dir: Path, panel_h: int) -> Image.Image:
    panel_w = WIDTH - PANEL_MARGIN * 2
    panel = _make_panel_base(entry.accent_color, entry.card_bg_color, panel_h)
    draw = ImageDraw.Draw(panel, "RGBA")
    accent_rgb = _hex_to_rgb(entry.accent_color)
    years = _parse_years(entry.years_won)
    if not years:
        years = [int(entry.first_title)]
    first_year = years[0]
    last_year = years[-1]
    years_span = max(1, last_year - first_year)

    title_font = _load_font(20, bold=True)
    subtitle_font = _load_font(15, bold=False)
    label_font = _fit_font(draw, entry.player_name.upper(), 186, 30, 18, bold=True)
    rank_font = _fit_font(draw, f"#{entry.rank}", 110, 54, 28, bold=True)
    count_font = _fit_font(draw, f"{entry.titles} {_title_word(entry.titles)}", 130, 28, 16, bold=True)
    span_font = _load_font(15, bold=True)
    year_font = _load_font(15, bold=True)
    stat_title_font = _load_font(14, bold=True)
    stat_value_font = _fit_font(draw, str(first_year), 120, 32, 20, bold=True)
    section_title_font = _load_font(20, bold=True)
    big_title_font = _fit_font(draw, str(entry.titles), 180, 104, 68, bold=True)
    footer_label_font = _load_font(14, bold=True)
    footer_year_font = _fit_font(draw, str(first_year), 100, 28, 18, bold=True)

    draw.text((28, 22), "CHRONOLOGIE DES TITRES", font=title_font, fill="#f7e6c8")
    draw.text((28, 48), "OPEN ERA MEN SINGLES", font=subtitle_font, fill="#c3d3dd")

    left_box = (24, 78, 248, 392)
    draw.rounded_rectangle(left_box, radius=24, fill=(255, 255, 255, 10), outline=(255, 255, 255, 16), width=1)
    draw.text((42, 100), "RANG", font=_load_font(14, bold=True), fill="#cfd8de")
    draw.text((42, 126), f"#{entry.rank}", font=rank_font, fill="#fff6ef")

    player_name = _truncate_to_width(draw, entry.player_name.upper(), label_font, left_box[2] - left_box[0] - 36)
    draw.text((42, 190), player_name, font=label_font, fill="#fff6ef")

    title_pill = (42, 226, 196, 260)
    draw.rounded_rectangle(title_pill, radius=14, fill=(*accent_rgb, 255))
    draw.text((119, 243), f"{entry.titles} {_title_word(entry.titles)}", font=count_font, fill="#111a22", anchor="mm")

    span_text = f"{first_year} -> {last_year}"
    draw.text((42, 268), span_text, font=span_font, fill="#d8e6ea")

    axis_left = 272
    axis_right = panel_w - 24
    axis_y = 124
    axis_w = max(1, axis_right - axis_left)
    draw.text((axis_left, 94), "TIMELINE", font=_load_font(18, bold=True), fill="#f7cf94")
    draw.line((axis_left, axis_y, axis_right, axis_y), fill=(*accent_rgb, 146), width=4)

    dot_layer = Image.new("RGBA", panel.size, (0, 0, 0, 0))
    dot_draw = ImageDraw.Draw(dot_layer, "RGBA")
    label_positions: list[tuple[int, int, int]] = []
    for idx, year in enumerate(years):
        if len(years) == 1:
            x = int((axis_left + axis_right) / 2)
        else:
            x = int(axis_left + ((year - first_year) / years_span) * axis_w)
        radius = 11 if idx in (0, len(years) - 1) else 8
        dot_draw.ellipse((x - radius - 6, axis_y - radius - 6, x + radius + 6, axis_y + radius + 6), fill=(*accent_rgb, 64))
        dot_draw.ellipse((x - radius, axis_y - radius, x + radius, axis_y + radius), fill=(*accent_rgb, 255))
        label_positions.append((x, year, idx))

    dot_layer = dot_layer.filter(ImageFilter.GaussianBlur(radius=2))
    panel.alpha_composite(dot_layer)
    draw = ImageDraw.Draw(panel, "RGBA")

    for x, year, idx in label_positions:
        year_text = str(year)
        year_width = draw.textbbox((0, 0), year_text, font=year_font)[2]
        label_x = max(axis_left, min(axis_right - year_width, x - year_width // 2))
        draw.text((label_x, axis_y + 16), year_text, font=year_font, fill="#f8f3ec")
        if idx == 0:
            draw.text((label_x, axis_y - 26), "FIRST", font=_load_font(12, bold=True), fill="#cfd8de")
        elif idx == len(label_positions) - 1:
            draw.text((label_x, axis_y - 26), "LAST", font=_load_font(12, bold=True), fill="#cfd8de")

    stat_box_w = 132
    stat_box_h = 72
    stat_y = 184
    first_box = (axis_left, stat_y, axis_left + stat_box_w, stat_y + stat_box_h)
    last_box = (axis_left + stat_box_w + 18, stat_y, axis_left + 2 * stat_box_w + 18, stat_y + stat_box_h)
    draw.rounded_rectangle(first_box, radius=18, fill=(255, 255, 255, 12), outline=(255, 255, 255, 18), width=1)
    draw.rounded_rectangle(last_box, radius=18, fill=(255, 255, 255, 12), outline=(255, 255, 255, 18), width=1)
    draw.text((first_box[0] + 16, first_box[1] + 12), "PREMIER TITRE", font=stat_title_font, fill="#f7cf94")
    draw.text((first_box[0] + 16, first_box[1] + 36), str(first_year), font=stat_value_font, fill="#f4f7fb")
    draw.text((last_box[0] + 16, last_box[1] + 12), "DERNIER TITRE", font=stat_title_font, fill="#f7cf94")
    draw.text((last_box[0] + 16, last_box[1] + 36), str(last_year), font=stat_value_font, fill="#f4f7fb")

    separator_y = 438
    draw.line((24, separator_y, panel_w - 24, separator_y), fill=(255, 255, 255, 18), width=1)
    draw.text((28, 460), "BILAN COMPLET", font=section_title_font, fill="#f7cf94")

    lower_top = 500
    lower_bottom = panel_h - 28
    summary_box = (24, lower_top, 262, lower_bottom - 122)
    grid_box = (286, lower_top, panel_w - 24, lower_bottom - 122)
    footer_box = (24, lower_bottom - 100, panel_w - 24, lower_bottom)

    draw.rounded_rectangle(summary_box, radius=24, fill=(255, 255, 255, 10), outline=(255, 255, 255, 16), width=1)
    draw.text((44, lower_top + 22), "TITRES", font=_load_font(16, bold=True), fill="#cfd8de")
    draw.text((42, lower_top + 38), str(entry.titles), font=big_title_font, fill="#fff6ef")
    draw.text((44, lower_top + 156), entry.player_name.upper(), font=_fit_font(draw, entry.player_name.upper(), 188, 30, 18, bold=True), fill="#f7cf94")
    draw.text((44, lower_top + 196), f"{first_year} -> {last_year}", font=_load_font(18, bold=True), fill="#d8e6ea")
    draw.rounded_rectangle((44, lower_top + 236, 102, lower_top + 278), radius=16, fill=(*accent_rgb, 230))
    draw.text((73, lower_top + 257), str(first_year), font=footer_year_font, fill="#111a22", anchor="mm")
    draw.rounded_rectangle((114, lower_top + 236, 232, lower_top + 278), radius=16, fill=(242, 232, 221, 240))
    draw.text((173, lower_top + 257), str(last_year), font=footer_year_font, fill="#214451", anchor="mm")

    draw.text((grid_box[0] + 4, lower_top + 12), "TOUTES LES VICTOIRES", font=section_title_font, fill="#f7cf94")
    draw.text((grid_box[0] + 4, lower_top + 40), "SIMPLE MESSIEURS", font=_load_font(14, bold=False), fill="#c3d3dd")

    pill_top = lower_top + 84
    pill_gap_x = 14
    pill_gap_y = 14
    pill_cols = 3
    pill_w = max(140, (grid_box[2] - grid_box[0] - pill_gap_x * (pill_cols - 1)) // pill_cols)
    pill_h = 58
    for idx, year in enumerate(years):
        row = idx // pill_cols
        col = idx % pill_cols
        x0 = grid_box[0] + col * (pill_w + pill_gap_x)
        y0 = pill_top + row * (pill_h + pill_gap_y)
        rect = (x0, y0, x0 + pill_w, y0 + pill_h)
        draw.rounded_rectangle(rect, radius=18, fill=(*accent_rgb, 220), outline=(255, 255, 255, 16), width=1)
        year_label_font = _fit_font(draw, str(year), pill_w - 24, 28, 18, bold=True)
        draw.text(((rect[0] + rect[2]) // 2, (rect[1] + rect[3]) // 2), str(year), font=year_label_font, fill="#111a22", anchor="mm")

    draw.rounded_rectangle(footer_box, radius=22, fill=(255, 255, 255, 9), outline=(255, 255, 255, 16), width=1)
    footer_third = (footer_box[2] - footer_box[0] - 40) // 3
    footer_items = [
        ("PREMIER TITRE", str(first_year)),
        ("DERNIER TITRE", str(last_year)),
        ("SPAN", f"{last_year - first_year + 1} ANS"),
    ]
    for idx, (label, value) in enumerate(footer_items):
        x0 = footer_box[0] + 20 + idx * footer_third
        draw.text((x0, footer_box[1] + 14), label, font=footer_label_font, fill="#cfd8de")
        value_font = _fit_font(draw, value, footer_third - 24, 28, 18, bold=True)
        draw.text((x0, footer_box[1] + 40), value, font=value_font, fill="#fff6ef")

    return panel


def _active_entry_index(shift: float, track_visible_w: int, card_pitch: int, count: int) -> tuple[int, int, float]:
    center_track_x = shift + track_visible_w / 2.0
    float_index = (center_track_x - CARD_W / 2.0) / card_pitch
    base_idx = max(0, min(count - 1, int(math.floor(float_index))))
    next_idx = max(0, min(count - 1, base_idx + 1))
    alpha = max(0.0, min(1.0, float_index - base_idx))
    return base_idx, next_idx, alpha


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

    card_images = [_render_wide_track_card(entry, photos_dir, flags_dir) for entry in entries]
    background = _background()
    logo = _render_logo(84) if DEFAULT_LOGO.exists() else None

    track_w = len(card_images) * CARD_W + max(0, len(card_images) - 1) * CARD_GAP
    track = Image.new("RGBA", (track_w, CARD_H), (0, 0, 0, 0))
    x = 0
    for image in card_images:
        track.alpha_composite(image, (x, 0))
        x += CARD_W + CARD_GAP

    viewport_src_w = int(round(VISIBLE_CARDS * CARD_W + max(0.0, VISIBLE_CARDS - 1.0) * CARD_GAP))
    rail_left = 0
    rail_top = 194
    max_offset = max(0, track_w - viewport_src_w)
    scroll_duration = max(0.1, duration - HOLD_START - HOLD_END)
    panel_top = rail_top + CARD_H + PANEL_GAP
    panel_h = HEIGHT - panel_top - 32
    panel_left = PANEL_MARGIN

    header = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    header_draw = ImageDraw.Draw(header, "RGBA")
    header_draw.rounded_rectangle((36, 36, WIDTH - 36, 164), radius=34, fill=(8, 18, 20, 216), outline=(255, 255, 255, 20), width=2)
    title_font = _load_font(48, bold=True)
    subtitle_font = _load_font(22, bold=False)
    _draw_text(header_draw, 58, 68, "LES GEANTS DE ROLAND-GARROS", title_font, "#fff6ef")
    _draw_text(header_draw, 58, 120, "Top 12 Open Era winners, cards + title timeline", subtitle_font, "#f0d9c7")

    logo_box = (WIDTH - 160, 46, WIDTH - 54, 152)
    header_draw.rounded_rectangle(logo_box, radius=28, fill=(255, 255, 255, 16), outline=(255, 255, 255, 30), width=2)
    if logo is not None:
        lx = logo_box[0] + (logo_box[2] - logo_box[0] - logo.width) // 2
        ly = logo_box[1] + (logo_box[3] - logo_box[1] - logo.height) // 2
        header.alpha_composite(logo, (lx, ly))

    timeline_cache: dict[int, Image.Image] = {}

    def timeline_for(idx: int) -> Image.Image:
        if idx not in timeline_cache:
            timeline_cache[idx] = _render_timeline_panel(entries[idx], flags_dir, panel_h)
        return timeline_cache[idx]

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
        viewport = track.crop((int(shift), 0, int(shift) + viewport_src_w, CARD_H))
        canvas.alpha_composite(viewport, (rail_left, rail_top))

        base_idx, _, _ = _active_entry_index(shift, viewport_src_w, CARD_W + CARD_GAP, len(entries))
        panel = timeline_for(base_idx)
        canvas.alpha_composite(panel, (panel_left, panel_top))
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
    parser = argparse.ArgumentParser(description="Generate a Roland-Garros cards + timeline Shorts video.")
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
    print(f"[video_generator] Roland-Garros cards + timeline Shorts generated -> {args.output}")


if __name__ == "__main__":
    main()
