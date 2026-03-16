from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
from moviepy import VideoClip
from PIL import Image, ImageDraw, ImageFilter, ImageOps

from video_generator.generate_ucl_barchart_race_moviepy import (
    DEFAULT_AUDIO,
    DEFAULT_FLAGS_DIR,
    FINAL_AUDIO_FADE_OUT,
    LOOP_CROSSFADE,
    _fit_font_size,
    _load_font,
    build_audio_track,
)
from video_generator.tennis.generate_grand_slam_titles_shorts_moviepy import (
    DEFAULT_INPUT,
    DEFAULT_PHOTOS_DIR,
    PLAYER_COLORS,
    _build_flag_cache,
    _build_photo_cache,
    _continuous_rank_position,
    _mix_rgb,
    _parse_season_summary,
    _text_on,
    _truncate_text_to_width,
    load_snapshots,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "tennis" / "grand_slam_titles_race_landscape_2000_2025_4min.mp4"

WIDTH = 1920
HEIGHT = 1080
TOP_N = 12
FPS = 30
TOTAL_DURATION = 240.0

TITLE = "GRAND SLAM TITLES RACE"
SUBTITLE = "ATP men - cumulative majors since start of Open Era"


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


def _make_background() -> Image.Image:
    xx = np.linspace(0, 1, WIDTH, dtype=np.float32)
    yy = np.linspace(0, 1, HEIGHT, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(xx, yy)
    navy = np.array([6, 16, 36], dtype=np.float32)
    court = np.array([14, 66, 132], dtype=np.float32)
    sky = np.array([120, 205, 255], dtype=np.float32)
    ball = np.array([204, 255, 92], dtype=np.float32)

    mix = np.clip(0.56 * grid_x + 0.34 * grid_y, 0, 1)
    center_glow = np.exp(-(((grid_x - 0.55) / 0.28) ** 2 + ((grid_y - 0.18) / 0.18) ** 2))
    side_glow = np.exp(-(((grid_x - 0.08) / 0.12) ** 2 + ((grid_y - 0.55) / 0.36) ** 2))
    side_glow += np.exp(-(((grid_x - 0.92) / 0.12) ** 2 + ((grid_y - 0.55) / 0.36) ** 2))

    img = np.clip(
        navy[None, None, :] * (1.0 - mix[..., None])
        + court[None, None, :] * (0.78 * mix[..., None])
        + sky[None, None, :] * (0.16 * center_glow[..., None])
        + ball[None, None, :] * (0.08 * side_glow[..., None]),
        0,
        255,
    ).astype(np.uint8)
    frame = Image.fromarray(img, mode="RGB").convert("RGBA")

    overlay = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")
    draw.rounded_rectangle((38, 34, WIDTH - 38, HEIGHT - 40), radius=42, outline=(255, 255, 255, 22), width=2)
    draw.line((112, 530, WIDTH - 112, 530), fill=(255, 255, 255, 12), width=2)
    draw.ellipse((720, 38, 1200, 514), outline=(190, 255, 110, 22), width=3)
    draw.line((960, 70, 960, 486), fill=(255, 255, 255, 10), width=2)
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
    fps: int,
    top_n: int,
) -> Path:
    snapshots = load_snapshots(input_csv)
    if len(snapshots) < 2:
        raise RuntimeError("Not enough Grand Slam snapshots to render.")
    first_snapshot = snapshots[0]
    intro_snapshot = first_snapshot.__class__(
        ranking_date=f"{first_snapshot.year - 1}-12-31",
        year=first_snapshot.year,
        season_summary="",
        states=[],
    )
    snapshots = [intro_snapshot, *snapshots]

    all_states = [state for snapshot in snapshots for state in snapshot.states]
    flag_cache = _build_flag_cache(all_states, flags_dir)
    photo_size = 60
    photo_cache = _build_photo_cache(all_states, photos_dir, photo_size)
    priorities = _build_stable_snapshot_priorities(snapshots)

    periods = len(snapshots) - 1
    seconds_per_period = duration / periods
    axis_caps = [
        float(max((state.titles for state in snapshot.states[:top_n]), default=1.0) + 1)
        for snapshot in snapshots
    ]

    background = _make_background()
    title_font = _load_font(58, bold=True)
    subtitle_font = _load_font(24, bold=False)
    year_font = _load_font(96, bold=True)
    summary_font_cache = {}
    name_font = _load_font(30, bold=True)
    value_font = _load_font(30, bold=True)
    rank_font = _load_font(28, bold=True)
    initials_font = _load_font(22, bold=True)
    tick_font = _load_font(20, bold=True)

    bar_left = 162
    bar_right = 1782
    bar_max_w = bar_right - bar_left
    base_y = 228
    row_h = 58
    row_gap = 6
    photo_box = 60
    rank_left = 90

    def make_frame(t: float) -> np.ndarray:
        frame = background.copy()
        draw = ImageDraw.Draw(frame, "RGBA")

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
        draw.text((74, 116), SUBTITLE, font=subtitle_font, fill="#b6d6ee")

        ranking_bottom = base_y + top_n * row_h + (top_n - 1) * row_gap
        ranking_center_y = (base_y + ranking_bottom) // 2
        header_height = 340
        header_top = ranking_center_y - header_height // 2 + 122
        header_box = (1188, header_top, WIDTH - 52, header_top + header_height)

        for tick in range(1, max_titles + 1):
            x = bar_left + int((tick / axis_cap) * bar_max_w)
            draw.line((x, base_y - 30, x, HEIGHT - 88), fill=(0, 0, 0, 70), width=2)
            draw.text((x - 6, base_y - 60), str(tick), font=tick_font, fill=(0, 0, 0, 110))

        draw.rounded_rectangle(header_box, radius=28, fill=(6, 18, 38, 220), outline=(198, 255, 92, 84), width=2)
        year_text = str(nxt.year)
        year_bbox = draw.textbbox((0, 0), year_text, font=year_font)
        year_x = header_box[0] + (header_box[2] - header_box[0] - (year_bbox[2] - year_bbox[0])) // 2
        draw.text((year_x, header_box[1] + 22), year_text, font=year_font, fill="#d4ff5d")

        summary_lines = _parse_season_summary(nxt.season_summary)
        summary_key = "\n".join(summary_lines)
        summary_font = summary_font_cache.get(summary_key)
        if summary_font is None:
            longest = max(summary_lines, key=len) if summary_lines else ""
            summary_font = _fit_font_size(draw, longest, 600, 26, 17, bold=True)
            summary_font_cache[summary_key] = summary_font
        line_gap = 10
        total_summary_h = len(summary_lines) * summary_font.size + max(0, len(summary_lines) - 1) * line_gap
        summary_top = header_box[1] + 140
        summary_bottom = header_box[3] - 30
        line_y = summary_top + max(0, (summary_bottom - summary_top - total_summary_h) // 2)
        for line in summary_lines:
            fitted = _truncate_text_to_width(draw, line, summary_font, 600)
            line_bbox = draw.textbbox((0, 0), fitted, font=summary_font)
            line_x = header_box[0] + (header_box[2] - header_box[0] - (line_bbox[2] - line_bbox[0])) // 2
            draw.text((line_x, line_y), fitted, font=summary_font, fill="#eef8ff")
            line_y += summary_font.size + line_gap

        for rank_idx in range(top_n):
            y0 = base_y + rank_idx * (row_h + row_gap)
            y1 = y0 + row_h
            draw.rounded_rectangle((rank_left, y0, rank_left + 56, y1), radius=18, fill=(203, 255, 94, 255))
            rank_text = str(rank_idx + 1)
            bbox = draw.textbbox((0, 0), rank_text, font=rank_font)
            draw.text(
                (rank_left + 28 - (bbox[2] - bbox[0]) // 2, y0 + (row_h - (bbox[3] - bbox[1])) // 2 - 1),
                rank_text,
                font=rank_font,
                fill="#112642",
            )

        items: list[tuple[int, float, object, int]] = []
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
            bar_w = max(120, int((state.titles / axis_cap) * bar_max_w))
            moving_up = 1 if next_idx < prev_idx else 0
            items.append((moving_up, y, state, bar_w))
        items.sort(key=lambda item: (item[0], item[1]))

        for _, y, state, bar_w in items:
            y0 = int(y)
            y1 = y0 + row_h
            color = PLAYER_COLORS.get(state.player_name, "#3dbdf5")
            text_color = _text_on(color)
            outline = _mix_rgb(color, (255, 255, 255), 0.18)
            highlight = _mix_rgb(color, (255, 255, 255), 0.32)
            shadow = _mix_rgb(color, (0, 0, 0), 0.20)

            draw.rounded_rectangle((bar_left + 6, y0 + 6, bar_left + bar_w + 6, y1 + 6), radius=24, fill=(0, 0, 0, 84))
            draw.rounded_rectangle((bar_left, y0, bar_left + bar_w, y1), radius=24, fill=color, outline=outline, width=2)
            draw.rounded_rectangle(
                (bar_left + 10, y0 + 8, bar_left + max(90, int(bar_w * 0.62)), y0 + 18),
                radius=8,
                fill=(*highlight, 52),
            )
            draw.line((bar_left + 20, y1 - 8, bar_left + max(40, int(bar_w * 0.72)), y1 - 8), fill=(*shadow, 90), width=3)

            photo = photo_cache.get(state.player_name)
            avatar_x = bar_left + 8
            avatar_y = y0 + (row_h - photo_box) // 2
            if photo is not None:
                draw.ellipse((avatar_x - 3, avatar_y - 3, avatar_x + photo_box + 3, avatar_y + photo_box + 3), fill=(255, 255, 255, 235))
                frame.alpha_composite(photo, (avatar_x, avatar_y))
            else:
                draw.rounded_rectangle((avatar_x, avatar_y, avatar_x + photo_box, avatar_y + photo_box), radius=20, fill=(255, 255, 255, 220))
                initials = "".join(part[0] for part in state.player_name.split()[:2]).upper()[:2]
                ibox = draw.textbbox((0, 0), initials, font=initials_font)
                draw.text(
                    (avatar_x + photo_box // 2 - (ibox[2] - ibox[0]) // 2, avatar_y + photo_box // 2 - (ibox[3] - ibox[1]) // 2 - 1),
                    initials,
                    font=initials_font,
                    fill="#10233f",
                )

            label_x = bar_left + photo_box + 20
            flag = flag_cache.get(state.country_code)
            if flag is not None:
                fx = label_x
                fy = y0 + (row_h - flag.height) // 2
                draw.rounded_rectangle((fx - 4, fy - 4, fx + flag.width + 4, fy + flag.height + 4), radius=8, fill=(255, 255, 255, 228))
                frame.alpha_composite(flag, (fx, fy))
                label_x = fx + flag.width + 16

            label_max_width = max(0, bar_w - (label_x - bar_left) - 24)
            player_name = _truncate_text_to_width(draw, state.player_name, name_font, label_max_width)
            draw.text((label_x, y0 + (row_h - 28) // 2 - 1), player_name, font=name_font, fill=text_color)

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
    parser = argparse.ArgumentParser(description="Generate a landscape Grand Slam bar chart race video.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--flags-dir", type=Path, default=DEFAULT_FLAGS_DIR)
    parser.add_argument("--photos-dir", type=Path, default=DEFAULT_PHOTOS_DIR)
    parser.add_argument("--audio", type=Path, default=DEFAULT_AUDIO)
    parser.add_argument("--duration", type=float, default=TOTAL_DURATION)
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
        fps=args.fps,
        top_n=args.top_n,
    )
    print(f"[video_generator] Grand Slam landscape race generated -> {output}")


if __name__ == "__main__":
    main()
