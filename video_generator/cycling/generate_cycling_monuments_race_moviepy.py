from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
from moviepy import VideoClip
from PIL import Image, ImageDraw, ImageFilter

from video_generator.generate_ucl_barchart_race_moviepy import (
    DEFAULT_AUDIO,
    DEFAULT_FLAGS_DIR,
    _fit_font_size,
    _load_font,
    build_audio_track,
)
from video_generator.tennis.generate_grand_slam_titles_shorts_moviepy import (
    _build_flag_cache,
    _build_photo_cache,
    _continuous_rank_position,
    _mix_rgb,
    _text_on,
    _truncate_text_to_width,
    load_snapshots,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_INPUT = PROJECT_ROOT / "data" / "processed" / "cycling" / "cycling_monuments_timeseries_1892_2025.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "cycling" / "cycling_monuments_race_landscape_1892_2025_4min.mp4"
DEFAULT_PHOTOS_DIR = PROJECT_ROOT / "data" / "raw" / "player_photos"

WIDTH = 1920
HEIGHT = 1080
TOP_N = 12
FPS = 30
TOTAL_DURATION = 240.0
FINAL_HOLD_DURATION = 15.0

TITLE = "CYCLING MONUMENTS RACE"
SUBTITLE = "Cumulative wins across the five Monuments"

RIDER_COLORS = {
    "Eddy Merckx": "#29c5f6",
    "Roger De Vlaeminck": "#ffb027",
    "Fausto Coppi": "#86e67c",
    "Sean Kelly": "#f36c6c",
    "Rik Van Looy": "#c48cff",
    "Tom Boonen": "#ff8f3f",
    "Johan Museeuw": "#53d7c2",
    "Costante Girardengo": "#ffd84d",
    "Alfredo Binda": "#58a6ff",
    "Francesco Moser": "#e774d6",
    "Tadej Pogacar": "#7ee081",
    "Mathieu van der Poel": "#ff6e66",
}
DEFAULT_BAR_COLORS = [
    "#29c5f6",
    "#ffb027",
    "#86e67c",
    "#f36c6c",
    "#c48cff",
    "#ff8f3f",
    "#53d7c2",
    "#ffd84d",
    "#58a6ff",
    "#e774d6",
    "#7ee081",
    "#ff6e66",
    "#73d2ff",
    "#b5ef53",
    "#ff9f68",
    "#9a7cff",
]
COUNTRY_ALPHA2_OVERRIDES = {
    "BEL": "be",
    "LUX": "lu",
    "SUI": "ch",
    "IRL": "ie",
    "MDA": "md",
    "LTU": "lt",
    "KAZ": "kz",
    "COL": "co",
    "SVK": "sk",
    "SLO": "si",
    "POL": "pl",
    "FRG": "de",
}


def _parse_season_summary(summary: str) -> list[str]:
    parts = [part.strip() for part in summary.split("|") if part.strip()]
    return parts[:5]


def _ease_in_out(value: float) -> float:
    value = min(max(value, 0.0), 1.0)
    return value * value * (3.0 - 2.0 * value)


def _phase_delay(value: float, delay: float, span: float) -> float:
    if span <= 0.0:
        return 1.0 if value >= delay else 0.0
    return min(max((value - delay) / span, 0.0), 1.0)


def _interp_values(prev, nxt, alpha: float):
    prev_map = {state.player_name: state for state in prev.states}
    next_map = {state.player_name: state for state in nxt.states}
    names = sorted(set(prev_map) | set(next_map))
    states = []
    for name in names:
        a = prev_map.get(name) or next_map[name]
        b = next_map.get(name) or prev_map[name]
        titles = a.titles + (b.titles - a.titles) * alpha
        states.append(type(a)(player_name=name, country_code=b.country_code or a.country_code, titles=titles))
    return states


def _rank_with_priority(states, top_n: int, priority_order: dict[str, int] | None = None) -> dict[str, int]:
    priority_order = priority_order or {}
    ranked = sorted(
        (state for state in states if state.titles > 0),
        key=lambda item: (-item.titles, priority_order.get(item.player_name, 10_000), item.player_name),
    )
    return {state.player_name: idx for idx, state in enumerate(ranked[:top_n])}


def _build_stable_snapshot_priorities(snapshots) -> list[dict[str, int]]:
    priorities: list[dict[str, int]] = []
    prev_priority: dict[str, int] | None = None
    for snapshot in snapshots:
        ranked = sorted(
            (state for state in snapshot.states if state.titles > 0),
            key=lambda item: (-item.titles, (prev_priority or {}).get(item.player_name, 10_000), item.player_name),
        )
        current_priority = {state.player_name: idx for idx, state in enumerate(ranked)}
        priorities.append(current_priority)
        prev_priority = current_priority
    return priorities


def _build_cycling_flag_cache(states, flags_dir: Path) -> dict[str, Image.Image]:
    cache = _build_flag_cache(states, flags_dir)
    countries = {state.country_code.strip().upper() for state in states if state.country_code.strip()}
    for country_code in countries:
        if country_code in cache:
            continue
        alpha2 = COUNTRY_ALPHA2_OVERRIDES.get(country_code)
        if not alpha2:
            continue
        path = flags_dir / f"{alpha2}.png"
        if not path.exists():
            continue
        try:
            img = Image.open(path).convert("RGBA")
            img.thumbnail((38, 26), Image.Resampling.LANCZOS)
            cache[country_code] = img
        except Exception:
            continue
    return cache


def _make_background() -> Image.Image:
    xx = np.linspace(0, 1, WIDTH, dtype=np.float32)
    yy = np.linspace(0, 1, HEIGHT, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(xx, yy)
    night = np.array([8, 18, 34], dtype=np.float32)
    road = np.array([24, 46, 73], dtype=np.float32)
    gold = np.array([242, 199, 92], dtype=np.float32)
    sky = np.array([80, 176, 255], dtype=np.float32)
    green = np.array([112, 214, 132], dtype=np.float32)

    mix = np.clip(0.50 * grid_x + 0.42 * grid_y, 0, 1)
    top_glow = np.exp(-(((grid_x - 0.65) / 0.32) ** 2 + ((grid_y - 0.10) / 0.15) ** 2))
    side_glow = np.exp(-(((grid_x - 0.12) / 0.15) ** 2 + ((grid_y - 0.70) / 0.32) ** 2))
    side_glow += np.exp(-(((grid_x - 0.88) / 0.15) ** 2 + ((grid_y - 0.72) / 0.30) ** 2))

    img = np.clip(
        night[None, None, :] * (1.0 - mix[..., None])
        + road[None, None, :] * (0.84 * mix[..., None])
        + sky[None, None, :] * (0.15 * top_glow[..., None])
        + gold[None, None, :] * (0.10 * side_glow[..., None])
        + green[None, None, :] * (0.05 * side_glow[..., None]),
        0,
        255,
    ).astype(np.uint8)
    frame = Image.fromarray(img, mode="RGB").convert("RGBA")

    overlay = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")
    draw.rounded_rectangle((34, 30, WIDTH - 34, HEIGHT - 34), radius=44, outline=(255, 255, 255, 18), width=2)
    draw.line((100, 535, WIDTH - 100, 535), fill=(255, 255, 255, 10), width=2)
    draw.ellipse((710, 45, 1210, 545), outline=(242, 199, 92, 18), width=3)
    draw.line((960, 80, 960, 510), fill=(255, 255, 255, 9), width=2)
    draw.line((180, 890, 800, 890), fill=(255, 255, 255, 8), width=3)
    draw.line((1120, 890, 1740, 890), fill=(255, 255, 255, 8), width=3)
    overlay = overlay.filter(ImageFilter.GaussianBlur(radius=3))
    frame.alpha_composite(overlay)
    return frame


def render_video(
    input_csv: Path,
    output_path: Path,
    flags_dir: Path,
    photos_dir: Path,
    audio_path: Path,
    duration: float,
    final_hold_duration: float,
    fps: int,
    top_n: int,
) -> Path:
    snapshots = load_snapshots(input_csv)
    if len(snapshots) < 2:
        raise RuntimeError("Not enough cycling snapshots to render.")
    first_snapshot = snapshots[0]
    intro_snapshot = first_snapshot.__class__(
        ranking_date=f"{first_snapshot.year - 1}-12-31",
        year=first_snapshot.year,
        season_summary="",
        states=[],
    )
    snapshots = [intro_snapshot, *snapshots]

    all_states = [state for snapshot in snapshots for state in snapshot.states]
    flag_cache = _build_cycling_flag_cache(all_states, flags_dir)
    photo_cache = _build_photo_cache(all_states, photos_dir, 58)
    priorities = _build_stable_snapshot_priorities(snapshots)

    periods = len(snapshots) - 1
    transition_duration = max(0.1, duration - max(0.0, final_hold_duration))
    seconds_per_period = transition_duration / periods
    axis_caps = [
        float(max((state.titles for state in snapshot.states[:top_n]), default=1.0) + 1)
        for snapshot in snapshots
    ]

    background = _make_background()
    title_font = _load_font(58, bold=True)
    subtitle_font = _load_font(24, bold=False)
    year_font = _load_font(94, bold=True)
    summary_font_cache = {}
    name_font = _load_font(30, bold=True)
    value_font = _load_font(30, bold=True)
    rank_font = _load_font(28, bold=True)
    tick_font = _load_font(20, bold=True)

    bar_left = 156
    bar_right = 1780
    bar_max_w = bar_right - bar_left
    base_y = 228
    row_h = 58
    row_gap = 8
    avatar_size = 58
    rank_left = 86

    def make_frame(t: float) -> np.ndarray:
        frame = background.copy()
        draw = ImageDraw.Draw(frame, "RGBA")

        if t >= transition_duration:
            period_index = periods - 1
            alpha = 1.0
        else:
            period_index = min(int(t / seconds_per_period), periods - 1)
            local_t = (t - period_index * seconds_per_period) / seconds_per_period
            alpha = _ease_in_out(local_t)

        prev = snapshots[period_index]
        nxt = snapshots[period_index + 1]
        axis_cap = axis_caps[period_index] + (axis_caps[period_index + 1] - axis_caps[period_index]) * alpha
        interp = _interp_values(prev, nxt, alpha)
        interp_map = {state.player_name: state for state in interp}
        priority = priorities[period_index]
        prev_rank = _rank_with_priority(prev.states, top_n, priority)
        next_rank = _rank_with_priority(nxt.states, top_n, priority)
        visible_names = sorted(set(prev_rank) | set(next_rank))
        top_states = [interp_map[name] for name in visible_names if name in interp_map]
        max_titles = max(1, int(math.ceil(axis_cap)))

        draw.text((72, 54), TITLE, font=title_font, fill="#f4f7fb")
        draw.text((74, 116), SUBTITLE, font=subtitle_font, fill="#bfd8ec")

        ranking_bottom = base_y + top_n * row_h + (top_n - 1) * row_gap
        ranking_center_y = (base_y + ranking_bottom) // 2
        header_height = 384
        header_top = ranking_center_y - header_height // 2 + 102
        header_box = (1194, header_top, WIDTH - 54, header_top + header_height)

        for tick in range(1, max_titles + 1):
            x = bar_left + int((tick / axis_cap) * bar_max_w)
            draw.line((x, base_y - 30, x, HEIGHT - 92), fill=(0, 0, 0, 64), width=2)
            draw.text((x - 6, base_y - 60), str(tick), font=tick_font, fill=(0, 0, 0, 108))

        draw.rounded_rectangle(header_box, radius=28, fill=(8, 20, 38, 225), outline=(242, 199, 92, 84), width=2)
        year_text = str(nxt.year)
        year_bbox = draw.textbbox((0, 0), year_text, font=year_font)
        year_x = header_box[0] + (header_box[2] - header_box[0] - (year_bbox[2] - year_bbox[0])) // 2
        draw.text((year_x, header_box[1] + 22), year_text, font=year_font, fill="#f6d46b")

        summary_lines = _parse_season_summary(nxt.season_summary)
        summary_key = "\n".join(summary_lines)
        summary_font = summary_font_cache.get(summary_key)
        if summary_font is None:
            longest = max(summary_lines, key=len) if summary_lines else ""
            summary_font = _fit_font_size(draw, longest, 600, 28, 17, bold=True)
            summary_font_cache[summary_key] = summary_font
        line_gap = 8
        total_summary_h = len(summary_lines) * summary_font.size + max(0, len(summary_lines) - 1) * line_gap
        summary_top = header_box[1] + 138
        summary_bottom = header_box[3] - 30
        line_y = summary_top + max(0, (summary_bottom - summary_top - total_summary_h) // 2)
        for line in summary_lines:
            fitted = _truncate_text_to_width(draw, line, summary_font, 606)
            line_bbox = draw.textbbox((0, 0), fitted, font=summary_font)
            line_x = header_box[0] + (header_box[2] - header_box[0] - (line_bbox[2] - line_bbox[0])) // 2
            draw.text((line_x, line_y), fitted, font=summary_font, fill="#eef8ff")
            line_y += summary_font.size + line_gap

        for rank_idx in range(top_n):
            y0 = base_y + rank_idx * (row_h + row_gap)
            y1 = y0 + row_h
            draw.rounded_rectangle((rank_left, y0, rank_left + 56, y1), radius=18, fill=(244, 205, 98, 255))
            rank_text = str(rank_idx + 1)
            bbox = draw.textbbox((0, 0), rank_text, font=rank_font)
            draw.text(
                (rank_left + 28 - (bbox[2] - bbox[0]) // 2, y0 + (row_h - (bbox[3] - bbox[1])) // 2 - 1),
                rank_text,
                font=rank_font,
                fill="#132742",
            )

        items: list[tuple[int, float, object, int, int]] = []
        for state in top_states:
            prev_idx = prev_rank.get(state.player_name, top_n + 1)
            next_idx = next_rank.get(state.player_name, top_n + 1)
            entering = prev_idx > top_n and next_idx <= top_n
            effective_prev_idx = float(top_n + 2.4) if entering else float(prev_idx)
            effective_move_alpha = alpha
            places_moved = abs(float(next_idx) - effective_prev_idx)
            if places_moved > 0.0:
                extra_delay = min(0.08, max(0.0, (places_moved - 1.0) * 0.015))
                active_span = max(0.82, 0.98 - extra_delay)
                effective_move_alpha = _ease_in_out(_phase_delay(effective_move_alpha, extra_delay, active_span))
            if entering:
                effective_move_alpha = _ease_in_out(_phase_delay(effective_move_alpha, 0.02, 0.96))
            y_idx = _continuous_rank_position(effective_prev_idx, float(next_idx), effective_move_alpha)
            y = base_y + y_idx * (row_h + row_gap)
            bar_w = max(108, int((state.titles / axis_cap) * bar_max_w))
            moving_up = 1 if next_idx < prev_idx else 0
            color_rank = next_idx if next_idx <= top_n else prev_idx
            items.append((moving_up, y, state, bar_w, int(color_rank)))
        items.sort(key=lambda item: (item[0], item[1]))

        for _, y, state, bar_w, color_rank in items:
            y0 = int(y)
            y1 = y0 + row_h
            color = RIDER_COLORS.get(state.player_name, DEFAULT_BAR_COLORS[color_rank % len(DEFAULT_BAR_COLORS)])
            text_color = _text_on(color)
            outline = _mix_rgb(color, (255, 255, 255), 0.18)
            highlight = _mix_rgb(color, (255, 255, 255), 0.30)
            shadow = _mix_rgb(color, (0, 0, 0), 0.22)

            draw.rounded_rectangle((bar_left + 6, y0 + 6, bar_left + bar_w + 6, y1 + 6), radius=24, fill=(0, 0, 0, 84))
            draw.rounded_rectangle((bar_left, y0, bar_left + bar_w, y1), radius=24, fill=color, outline=outline, width=2)
            draw.rounded_rectangle(
                (bar_left + 10, y0 + 8, bar_left + max(90, int(bar_w * 0.56)), y0 + 18),
                radius=8,
                fill=(*highlight, 52),
            )
            draw.line((bar_left + 22, y1 - 8, bar_left + max(44, int(bar_w * 0.68)), y1 - 8), fill=(*shadow, 92), width=3)

            avatar_x = bar_left + 8
            avatar_y = y0 + (row_h - avatar_size) // 2
            draw.ellipse((avatar_x - 2, avatar_y - 2, avatar_x + avatar_size + 2, avatar_y + avatar_size + 2), fill=(255, 255, 255, 228))
            photo = photo_cache.get(state.player_name)
            if photo is not None:
                frame.alpha_composite(photo, (avatar_x, avatar_y))
            else:
                draw.ellipse((avatar_x, avatar_y, avatar_x + avatar_size, avatar_y + avatar_size), fill=(8, 20, 40, 186))

            label_x = bar_left + avatar_size + 20
            flag = flag_cache.get(state.country_code)
            if flag is not None:
                fx = label_x
                fy = y0 + (row_h - flag.height) // 2
                draw.rounded_rectangle((fx - 4, fy - 4, fx + flag.width + 4, fy + flag.height + 4), radius=8, fill=(255, 255, 255, 228))
                frame.alpha_composite(flag, (fx, fy))
                label_x = fx + flag.width + 16

            label_max_width = max(0, bar_w - (label_x - bar_left) - 24)
            rider_name = _truncate_text_to_width(draw, state.player_name, name_font, label_max_width)
            draw.text((label_x, y0 + (row_h - 28) // 2 - 1), rider_name, font=name_font, fill=text_color)

            value_text = str(int(round(state.titles)))
            vbox = draw.textbbox((0, 0), value_text, font=value_font)
            value_x = min(bar_left + bar_w + 18, WIDTH - 64 - (vbox[2] - vbox[0]))
            draw.text((value_x, y0 + (row_h - 28) // 2 - 1), value_text, font=value_font, fill="#f4f7fb")

        return np.array(frame.convert("RGB"))

    clip = VideoClip(make_frame, duration=duration)
    audio_clip, keep_alive = build_audio_track(audio_path, duration)
    clip = clip.with_audio(audio_clip)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    clip.write_videofile(str(output_path), fps=fps, codec="libx264", audio_codec="aac")
    clip.close()
    audio_clip.close()
    for item in keep_alive:
        item.close()
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a landscape cycling Monuments bar chart race video.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--flags-dir", type=Path, default=DEFAULT_FLAGS_DIR)
    parser.add_argument("--photos-dir", type=Path, default=DEFAULT_PHOTOS_DIR)
    parser.add_argument("--audio", type=Path, default=DEFAULT_AUDIO)
    parser.add_argument("--duration", type=float, default=TOTAL_DURATION)
    parser.add_argument("--final-hold", type=float, default=FINAL_HOLD_DURATION)
    parser.add_argument("--fps", type=int, default=FPS)
    parser.add_argument("--top-n", type=int, default=TOP_N)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = render_video(
        input_csv=args.input,
        output_path=args.output,
        flags_dir=args.flags_dir,
        photos_dir=args.photos_dir,
        audio_path=args.audio,
        duration=args.duration,
        final_hold_duration=args.final_hold,
        fps=args.fps,
        top_n=args.top_n,
    )
    print(f"[video_generator] Cycling Monuments landscape race generated -> {output}")


if __name__ == "__main__":
    main()
