from __future__ import annotations

import argparse
from pathlib import Path

from video_generator.tennis.generate_grand_slam_titles_shorts_moviepy import (
    DEFAULT_AUDIO,
    DEFAULT_FPS,
    DEFAULT_FLAGS_DIR,
    DEFAULT_PHOTOS_DIR,
    FPS,
    HEIGHT,
    Image,
    ImageDraw,
    ImageFilter,
    ImageFont,
    PLAYER_COLORS,
    PlayerState,
    Snapshot,
    TOTAL_DURATION,
    TOP_N,
    VideoClip,
    WIDTH,
    _build_flag_cache,
    _build_photo_cache,
    _continuous_rank_position,
    _draw_glow_text,
    _ease_in_out,
    _fit_font_size,
    _interp_values,
    _load_font,
    _make_background,
    _mix_rgb,
    _parse_season_summary,
    _player_initials,
    _rank_with_tie_priority,
    _smoothstep,
    _text_on,
    _truncate_text_to_width,
    build_audio_track,
    load_snapshots,
    np,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_INPUT = PROJECT_ROOT / "data" / "processed" / "tennis" / "indian_wells_titles_timeseries_1976_2025.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "tennis" / "indian_wells_titles_shorts_1976_2025_45s.mp4"

TITLE = "INDIAN WELLS TITLES"
SUBTITLE = "ATP men · cumulative champions race"


def _indian_wells_background() -> Image.Image:
    frame = _make_background()
    overlay = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")
    draw.ellipse((160, 180, WIDTH - 160, 760), outline=(232, 201, 120, 20), width=3)
    draw.line((120, 1490, WIDTH - 120, 1490), fill=(232, 201, 120, 18), width=3)
    overlay = overlay.filter(ImageFilter.GaussianBlur(radius=2))
    frame.alpha_composite(overlay)
    return frame


def _indian_wells_color(name: str) -> str:
    special = {
        "Carlos Alcaraz": "#e8c05f",
        "Novak Djokovic": "#35b4ff",
        "Rafael Nadal": "#ff9b3f",
        "Roger Federer": "#b77b4a",
        "Taylor Fritz": "#8dd65f",
        "Andre Agassi": "#d8f15f",
        "Pete Sampras": "#8b72ff",
        "Michael Chang": "#45c0ff",
        "Boris Becker": "#2ec4b6",
        "Jimmy Connors": "#ff5d8f",
    }
    return special.get(name, PLAYER_COLORS.get(name, "#39c0ff"))


def _build_stable_snapshot_priorities(snapshots: list[Snapshot]) -> list[dict[str, int]]:
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
        raise RuntimeError("Not enough Indian Wells snapshots to render.")

    all_states = [state for snapshot in snapshots for state in snapshot.states]
    flag_cache = _build_flag_cache(all_states, flags_dir)
    photo_size = 54
    photo_cache = _build_photo_cache(all_states, photos_dir, photo_size)
    priorities = _build_stable_snapshot_priorities(snapshots)
    periods = len(snapshots) - 1
    seconds_per_period = duration / periods
    axis_caps = [float(max(state.titles for state in snapshot.states[:top_n]) + 1) for snapshot in snapshots]

    background = _indian_wells_background()
    title_font = _load_font(60, bold=True)
    subtitle_font = _load_font(24, bold=False)
    year_font = _load_font(90, bold=True)
    summary_font_cache: dict[str, ImageFont.ImageFont] = {}
    name_font = _load_font(28, bold=True)
    value_font = _load_font(32, bold=True)
    rank_font = _load_font(26, bold=True)
    initials_font = _load_font(18, bold=True)

    bar_left = 104
    bar_right = 1054
    bar_max_w = bar_right - bar_left
    base_y = 394
    rank_x = 18
    row_gap = 0
    row_h = 96
    bar_height = 88
    photo_box = 54

    def make_frame(t: float) -> np.ndarray:
        frame = background.copy()
        draw = ImageDraw.Draw(frame, "RGBA")

        period_index = min(int(t / seconds_per_period), periods - 1)
        local_t = (t - period_index * seconds_per_period) / seconds_per_period
        alpha = _ease_in_out(local_t)
        axis_cap = axis_caps[period_index] + (axis_caps[period_index + 1] - axis_caps[period_index]) * alpha

        prev = snapshots[period_index]
        nxt = snapshots[period_index + 1]
        interp = _interp_values(prev, nxt, alpha)
        interp_map = {state.player_name: state for state in interp}
        priority = priorities[period_index]
        prev_rank = _rank_with_tie_priority(prev.states, top_n, priority)
        next_rank = _rank_with_tie_priority(nxt.states, top_n, priority)
        visible_names = sorted(set(prev_rank) | set(next_rank))
        top_states = [interp_map[name] for name in visible_names if name in interp_map]

        header = (46, 44, WIDTH - 46, 364)
        draw.rounded_rectangle(header, radius=42, fill=(10, 30, 52, 214), outline=(255, 255, 255, 18), width=1)
        draw.text((76, 74), TITLE, font=title_font, fill="#f6f4ec")
        draw.text((78, 146), SUBTITLE, font=subtitle_font, fill="#d4e0ea")
        draw.rounded_rectangle((72, 196, 260, 294), radius=30, fill=(224, 196, 110, 255))
        year_bbox = draw.textbbox((0, 0), str(nxt.year), font=year_font)
        draw.text((166 - (year_bbox[2] - year_bbox[0]) // 2, 204), str(nxt.year), font=year_font, fill="#0d223e")

        summary_lines = _parse_season_summary(nxt.season_summary.replace(" def. ", " ").replace(" | ", " "))
        if not summary_lines:
            summary_lines = [nxt.season_summary]
        summary_key = "\n".join(summary_lines)
        summary_font = summary_font_cache.get(summary_key)
        if summary_font is None:
            longest = max(summary_lines, key=len) if summary_lines else ""
            summary_font = _fit_font_size(draw, longest, 650, 34, 17, bold=True)
            summary_font_cache[summary_key] = summary_font
        summary_rect = (278, 172, WIDTH - 58, 342)
        draw.rounded_rectangle(summary_rect, radius=28, fill=(14, 54, 84, 222), outline=(225, 191, 96, 85), width=2)
        line_height = max(18, int(summary_font.size * 0.95))
        total_text_height = len(summary_lines) * line_height + max(0, len(summary_lines) - 1) * 4
        line_y = summary_rect[1] + max(10, ((summary_rect[3] - summary_rect[1]) - total_text_height) // 2)
        for line in summary_lines[:2]:
            fitted_line = _truncate_text_to_width(draw, line, summary_font, summary_rect[2] - summary_rect[0] - 36)
            draw.text((summary_rect[0] + 18, line_y), fitted_line, font=summary_font, fill="#eef7ff")
            line_y += line_height + 4

        for lane_index in range(top_n):
            lane_y = base_y + lane_index * (row_h + row_gap)
            rank_top = lane_y + (row_h - bar_height) // 2
            rank_bottom = rank_top + bar_height
            draw.rounded_rectangle((rank_x, rank_top, rank_x + 54, rank_bottom), radius=18, fill=(224, 196, 110, 255))
            rank_text = str(lane_index + 1)
            rank_bbox = draw.textbbox((0, 0), rank_text, font=rank_font)
            draw.text(
                (rank_x + 27 - (rank_bbox[2] - rank_bbox[0]) // 2, rank_top + (bar_height - (rank_bbox[3] - rank_bbox[1])) // 2 - 1),
                rank_text,
                font=rank_font,
                fill="#10233f",
            )

        items: list[tuple[int, float, PlayerState, int]] = []
        for state in top_states:
            prev_idx = prev_rank.get(state.player_name, top_n + 1)
            next_idx = next_rank.get(state.player_name, top_n + 1)
            y_idx = _continuous_rank_position(float(prev_idx), float(next_idx), alpha)
            y = base_y + y_idx * (row_h + row_gap)
            bar_w = max(110, int((state.titles / axis_cap) * bar_max_w))
            moving_up = 1 if next_idx < prev_idx else 0
            items.append((moving_up, y, state, bar_w))
        items.sort(key=lambda item: (item[0], item[1]))

        for _, y, state, bar_w in items:
            y0 = int(y)
            color = _indian_wells_color(state.player_name)
            text_color = _text_on(color)
            bar_top = y0 + (row_h - bar_height) // 2
            bar_bottom = bar_top + bar_height
            radius = max(18, bar_height // 2)

            draw.rounded_rectangle((bar_left + 8, bar_top + 8, bar_left + bar_w + 8, bar_bottom + 8), radius=radius, fill=(0, 0, 0, 82))
            outline_color = _mix_rgb(color, (255, 255, 255), 0.15)
            highlight_color = _mix_rgb(color, (255, 255, 255), 0.28)
            inner_shadow = _mix_rgb(color, (0, 0, 0), 0.16)
            draw.rounded_rectangle((bar_left, bar_top, bar_left + bar_w, bar_bottom), radius=radius, fill=color, outline=outline_color, width=2)
            draw.rounded_rectangle(
                (bar_left + 8, bar_top + 8, bar_left + max(84, int(bar_w * 0.58)), bar_top + 20),
                radius=8,
                fill=(*highlight_color, 48),
            )
            draw.line((bar_left + 18, bar_bottom - 7, bar_left + max(34, int(bar_w * 0.72)), bar_bottom - 7), fill=(*inner_shadow, 82), width=3)

            avatar_x = bar_left + 8
            avatar_y = bar_top + (bar_height - photo_box) // 2
            photo = photo_cache.get(state.player_name)
            if photo is not None:
                draw.ellipse((avatar_x - 3, avatar_y - 3, avatar_x + photo_box + 3, avatar_y + photo_box + 3), fill=(255, 255, 255, 232))
                frame.alpha_composite(photo, (avatar_x, avatar_y))
            else:
                initials_box = (avatar_x, avatar_y, avatar_x + photo_box, avatar_y + photo_box)
                draw.rounded_rectangle(initials_box, radius=16, fill=(255, 255, 255, 220))
                initials = _player_initials(state.player_name)
                initials_bbox = draw.textbbox((0, 0), initials, font=initials_font)
                draw.text(
                    (
                        (initials_box[0] + initials_box[2]) // 2 - (initials_bbox[2] - initials_bbox[0]) // 2,
                        (initials_box[1] + initials_box[3]) // 2 - (initials_bbox[3] - initials_bbox[1]) // 2 - 1,
                    ),
                    initials,
                    font=initials_font,
                    fill="#10233f",
                )

            flag = flag_cache.get(state.country_code)
            label_x = bar_left + photo_box + 18
            if flag is not None:
                fx = label_x
                fy = bar_top + (bar_height - flag.height) // 2
                draw.rounded_rectangle((fx - 4, fy - 4, fx + flag.width + 4, fy + flag.height + 4), radius=8, fill=(255, 255, 255, 228))
                frame.alpha_composite(flag, (fx, fy))
                label_x = fx + flag.width + 16

            label_max_width = max(0, bar_w - (label_x - bar_left) - 18)
            player_name = _truncate_text_to_width(draw, state.player_name, name_font, label_max_width)
            draw.text((label_x, bar_top + (bar_height - 31) // 2), player_name, font=name_font, fill=text_color)

            value_text = str(int(round(state.titles)))
            value_bbox = draw.textbbox((0, 0), value_text, font=value_font)
            value_x = min(bar_left + bar_w + 18, WIDTH - 56 - (value_bbox[2] - value_bbox[0]))
            _draw_glow_text(frame, (value_x, bar_top + (bar_height - 30) // 2), value_text, value_font, (246, 244, 236), (225, 191, 96))

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
    parser = argparse.ArgumentParser(description="Generate an Indian Wells Shorts bar chart race.")
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
    print(f"[video_generator] Indian Wells Shorts race generated -> {output}")


if __name__ == "__main__":
    main()
