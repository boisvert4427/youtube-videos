from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
from moviepy import VideoClip
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from video_generator.cycling.generate_tour_de_france_stage_wins_race_moviepy import (
    DEFAULT_AUDIO,
    DEFAULT_FLAGS_DIR,
    DEFAULT_INPUT,
    DEFAULT_PHOTOS_DIR,
    PlayerState,
    Snapshot,
    _build_flag_cache,
    _build_photo_cache,
    _build_rider_color_map,
    _build_stable_snapshot_priorities,
    _continuous_rank_position,
    _ease_in_out,
    _fit_font_size,
    _interp_values,
    _load_font,
    _mix_rgb,
    _parse_season_summary,
    _phase_delay,
    _player_initials,
    _rank_with_tie_priority,
    _text_on,
    _truncate_text_to_width,
    build_audio_track,
    load_snapshots,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "cycling"
    / "tour_de_france"
    / "tour_de_france_stage_wins_postwar_1947_2025_shorts.mp4"
)

WIDTH = 1080
HEIGHT = 1920
TOP_N = 12
FPS = 60
TOTAL_DURATION = 60.0
FINAL_HOLD_DURATION = 5.0

TITLE = "TOUR DE FRANCE"
SUBTITLE = "MOST STAGE WINS | 1947-2025"


def _make_background() -> Image.Image:
    xx = np.linspace(0, 1, WIDTH, dtype=np.float32)
    yy = np.linspace(0, 1, HEIGHT, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(xx, yy)
    night = np.array([5, 16, 35], dtype=np.float32)
    blue = np.array([17, 55, 105], dtype=np.float32)
    sky = np.array([68, 151, 224], dtype=np.float32)
    yellow = np.array([246, 208, 91], dtype=np.float32)

    mix = np.clip(0.24 * grid_x + 0.72 * grid_y, 0, 1)
    top_glow = np.exp(-(((grid_x - 0.76) / 0.34) ** 2 + ((grid_y - 0.06) / 0.15) ** 2))
    side_glow = np.exp(-(((grid_x - 0.10) / 0.24) ** 2 + ((grid_y - 0.72) / 0.34) ** 2))
    rgb = np.clip(
        night[None, None, :] * (1.0 - mix[..., None])
        + blue[None, None, :] * (0.82 * mix[..., None])
        + sky[None, None, :] * (0.13 * top_glow[..., None])
        + yellow[None, None, :] * (0.08 * side_glow[..., None]),
        0,
        255,
    ).astype(np.uint8)
    frame = Image.fromarray(rgb, mode="RGB").convert("RGBA")

    overlay = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")
    draw.rounded_rectangle((24, 24, WIDTH - 24, HEIGHT - 24), radius=42, outline=(255, 255, 255, 18), width=2)
    draw.ellipse((660, 10, 1120, 470), outline=(246, 208, 91, 18), width=3)
    draw.ellipse((760, 110, 1020, 370), outline=(246, 208, 91, 10), width=2)
    draw.line((110, 1820, 990, 1710), fill=(255, 255, 255, 8), width=2)
    overlay = overlay.filter(ImageFilter.GaussianBlur(radius=2))
    frame.alpha_composite(overlay)
    return frame


def _center_text(
    draw: ImageDraw.ImageDraw,
    rect: tuple[int, int, int, int],
    text: str,
    font: ImageFont.ImageFont,
    fill: str,
) -> None:
    bbox = draw.textbbox((0, 0), text, font=font)
    x = (rect[0] + rect[2] - (bbox[2] - bbox[0])) // 2 - bbox[0]
    y = (rect[1] + rect[3] - (bbox[3] - bbox[1])) // 2 - bbox[1]
    draw.text((x, y), text, font=font, fill=fill)


def _compact_player_name(name: str) -> str:
    tokens = [token for token in name.split() if token]
    if len(tokens) <= 1:
        return name
    particles = {"de", "del", "da", "di", "du", "la", "le", "van", "von"}
    surname = [tokens[-1]]
    index = len(tokens) - 2
    while index >= 1 and tokens[index].lower() in particles:
        surname.insert(0, tokens[index])
        index -= 1
    return " ".join(surname)


def _fit_player_name(
    draw: ImageDraw.ImageDraw,
    name: str,
    font: ImageFont.ImageFont,
    max_width: int,
) -> str:
    if draw.textbbox((0, 0), name, font=font)[2] <= max_width:
        return name
    compact = _compact_player_name(name)
    if draw.textbbox((0, 0), compact, font=font)[2] <= max_width:
        return compact
    return _truncate_text_to_width(draw, compact, font, max_width)


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
        raise RuntimeError("Not enough Tour de France snapshots to render.")

    first_snapshot = snapshots[0]
    snapshots = [
        Snapshot(
            ranking_date=f"{first_snapshot.year - 1}-12-31",
            year=first_snapshot.year,
            season_summary="Post-war race|Cumulative stage wins|Since 1947",
            states=[],
        ),
        *snapshots,
    ]

    all_states = [state for snapshot in snapshots for state in snapshot.states]
    flag_cache = _build_flag_cache(all_states, flags_dir)
    photo_cache = _build_photo_cache(all_states, photos_dir, 72)
    rider_color_map = _build_rider_color_map(snapshots)
    priorities = _build_stable_snapshot_priorities(snapshots)

    periods = len(snapshots) - 1
    transition_duration = max(0.1, duration - max(0.0, final_hold_duration))
    seconds_per_period = transition_duration / periods
    axis_caps = [
        float(max((state.titles for state in snapshot.states[:top_n]), default=1.0) + 1)
        for snapshot in snapshots
    ]

    background = _make_background()
    title_font = _load_font(68, bold=True)
    subtitle_font = _load_font(30, bold=True)
    year_font = _load_font(96, bold=True)
    summary_label_font = _load_font(25, bold=True)
    summary_font_cache: dict[str, ImageFont.ImageFont] = {}
    name_font = _load_font(28, bold=True)
    value_font = _load_font(34, bold=True)
    rank_font = _load_font(28, bold=True)
    tick_font = _load_font(19, bold=True)
    initials_font = _load_font(22, bold=True)

    header_box = (36, 40, WIDTH - 36, 492)
    year_badge = (350, 198, 730, 326)
    summary_box = (76, 344, WIDTH - 76, 468)

    bar_left = 130
    bar_right = 990
    bar_max_w = bar_right - bar_left
    base_y = 554
    pitch = 105
    row_h = 86
    avatar_size = 72
    rank_left = 34
    ranking_bottom = base_y + (top_n - 1) * pitch + row_h

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
        prev_rank = _rank_with_tie_priority(prev.states, top_n, priority)
        next_rank = _rank_with_tie_priority(nxt.states, top_n, priority)
        visible_names = sorted(set(prev_rank) | set(next_rank))
        top_states = [interp_map[name] for name in visible_names if name in interp_map]
        max_titles = max(1, int(math.ceil(axis_cap)))

        draw.rounded_rectangle(header_box, radius=40, fill=(7, 20, 40, 220), outline=(255, 255, 255, 24), width=2)
        title_bbox = draw.textbbox((0, 0), TITLE, font=title_font)
        draw.text(((WIDTH - (title_bbox[2] - title_bbox[0])) // 2, 67), TITLE, font=title_font, fill="#f4f7fb")
        subtitle_bbox = draw.textbbox((0, 0), SUBTITLE, font=subtitle_font)
        draw.text(
            ((WIDTH - (subtitle_bbox[2] - subtitle_bbox[0])) // 2, 146),
            SUBTITLE,
            font=subtitle_font,
            fill="#bfd8ec",
        )

        draw.rounded_rectangle(year_badge, radius=34, fill=(246, 208, 91, 255))
        _center_text(draw, year_badge, str(nxt.year), year_font, "#10233f")

        draw.rounded_rectangle(summary_box, radius=24, fill=(14, 42, 77, 226), outline=(180, 255, 120, 48), width=2)
        summary_lines = _parse_season_summary(nxt.season_summary)
        summary_label = summary_lines[0].upper() if summary_lines else "TOP STAGE WINNERS"
        _center_text(draw, (summary_box[0], summary_box[1] + 4, summary_box[2], summary_box[1] + 44), summary_label, summary_label_font, "#f6d05b")
        detail = "  |  ".join(summary_lines[1:4]) if len(summary_lines) > 1 else "Cumulative stage wins since 1947"
        summary_font = summary_font_cache.get(detail)
        if summary_font is None:
            summary_font = _fit_font_size(draw, detail, summary_box[2] - summary_box[0] - 36, 26, 17, bold=True)
            summary_font_cache[detail] = summary_font
        fitted_detail = _truncate_text_to_width(draw, detail, summary_font, summary_box[2] - summary_box[0] - 36)
        _center_text(
            draw,
            (summary_box[0] + 18, summary_box[1] + 48, summary_box[2] - 18, summary_box[3] - 6),
            fitted_detail,
            summary_font,
            "#eef7ff",
        )

        tick_step = 1 if max_titles <= 12 else 2 if max_titles <= 24 else 5
        for tick in range(1, max_titles + 1):
            x = bar_left + int((tick / axis_cap) * bar_max_w)
            major = tick % tick_step == 0
            draw.line(
                (x, base_y - 42, x, ranking_bottom + 18),
                fill=(0, 0, 0, 66 if major else 32),
                width=2 if major else 1,
            )
            if major:
                label = str(tick)
                bbox = draw.textbbox((0, 0), label, font=tick_font)
                draw.text((x - (bbox[2] - bbox[0]) // 2, base_y - 38), label, font=tick_font, fill=(0, 0, 0, 132))

        for rank_idx in range(top_n):
            y0 = base_y + rank_idx * pitch
            y1 = y0 + row_h
            draw.rounded_rectangle((rank_left, y0, rank_left + 76, y1), radius=24, fill=(246, 208, 91, 255))
            _center_text(draw, (rank_left, y0, rank_left + 76, y1), str(rank_idx + 1), rank_font, "#132742")

        items: list[tuple[int, float, PlayerState, int]] = []
        for state in top_states:
            prev_idx = prev_rank.get(state.player_name, top_n + 1)
            next_idx = next_rank.get(state.player_name, top_n + 1)
            entering = prev_idx > top_n and next_idx <= top_n
            effective_prev_idx = float(top_n + 2.0) if entering else float(prev_idx)
            move_alpha = alpha
            places_moved = abs(float(next_idx) - effective_prev_idx)
            if places_moved > 0.0:
                delay = min(0.08, max(0.0, (places_moved - 1.0) * 0.015))
                move_alpha = _ease_in_out(_phase_delay(move_alpha, delay, max(0.82, 0.98 - delay)))
            if entering:
                move_alpha = _ease_in_out(_phase_delay(move_alpha, 0.02, 0.96))
            y_idx = _continuous_rank_position(effective_prev_idx, float(next_idx), move_alpha)
            y = base_y + y_idx * pitch
            bar_w = max(150, int((state.titles / axis_cap) * bar_max_w))
            moving_up = 1 if next_idx < prev_idx else 0
            items.append((moving_up, y, state, bar_w))
        items.sort(key=lambda item: (item[0], item[1]))

        for _, y, state, bar_w in items:
            y0 = int(y)
            y1 = y0 + row_h
            color = rider_color_map[state.player_name]
            text_color = _text_on(color)
            outline = _mix_rgb(color, (255, 255, 255), 0.20)
            highlight = _mix_rgb(color, (255, 255, 255), 0.32)
            shadow = _mix_rgb(color, (0, 0, 0), 0.24)

            draw.rounded_rectangle((bar_left + 7, y0 + 8, bar_left + bar_w + 7, y1 + 8), radius=34, fill=(0, 0, 0, 86))
            draw.rounded_rectangle((bar_left, y0, bar_left + bar_w, y1), radius=34, fill=color, outline=outline, width=2)
            draw.rounded_rectangle(
                (bar_left + 12, y0 + 10, bar_left + max(112, int(bar_w * 0.58)), y0 + 24),
                radius=9,
                fill=(*highlight, 54),
            )
            draw.line(
                (bar_left + 30, y1 - 11, bar_left + max(62, int(bar_w * 0.68)), y1 - 11),
                fill=(*shadow, 96),
                width=4,
            )

            avatar_x = bar_left + 8
            avatar_y = y0 + (row_h - avatar_size) // 2
            draw.ellipse(
                (avatar_x - 3, avatar_y - 3, avatar_x + avatar_size + 3, avatar_y + avatar_size + 3),
                fill=(255, 255, 255, 235),
            )
            photo = photo_cache.get(state.player_name)
            if photo is not None:
                frame.alpha_composite(photo, (avatar_x, avatar_y))
            else:
                draw.ellipse((avatar_x, avatar_y, avatar_x + avatar_size, avatar_y + avatar_size), fill=(8, 20, 40, 190))
                _center_text(
                    draw,
                    (avatar_x, avatar_y, avatar_x + avatar_size, avatar_y + avatar_size),
                    _player_initials(state.player_name),
                    initials_font,
                    "#f5f7fb",
                )

            label_x = bar_left + avatar_size + 24
            flag = flag_cache.get(state.country_code)
            if flag is not None:
                fx = label_x
                fy = y0 + (row_h - flag.height) // 2
                draw.rounded_rectangle(
                    (fx - 4, fy - 4, fx + flag.width + 4, fy + flag.height + 4),
                    radius=8,
                    fill=(255, 255, 255, 230),
                )
                frame.alpha_composite(flag, (fx, fy))
                label_x = fx + flag.width + 15

            label_max_width = max(0, bar_w - (label_x - bar_left) - 22)
            rider_name = _fit_player_name(draw, state.player_name, name_font, label_max_width)
            name_bbox = draw.textbbox((0, 0), rider_name, font=name_font)
            name_y = y0 + (row_h - (name_bbox[3] - name_bbox[1])) // 2 - name_bbox[1]
            draw.text((label_x, name_y), rider_name, font=name_font, fill=text_color)

            value_text = str(int(round(state.titles)))
            value_bbox = draw.textbbox((0, 0), value_text, font=value_font)
            value_x = min(bar_left + bar_w + 16, WIDTH - 30 - (value_bbox[2] - value_bbox[0]))
            value_y = y0 + (row_h - (value_bbox[3] - value_bbox[1])) // 2 - value_bbox[1]
            draw.text((value_x, value_y), value_text, font=value_font, fill="#f4f7fb")

        footer = "CUMULATIVE INDIVIDUAL STAGE WINS"
        footer_bbox = draw.textbbox((0, 0), footer, font=summary_label_font)
        draw.text(
            ((WIDTH - (footer_bbox[2] - footer_bbox[0])) // 2, 1850),
            footer,
            font=summary_label_font,
            fill=(191, 216, 236, 190),
        )

        return np.array(frame.convert("RGB"))

    clip = VideoClip(make_frame, duration=duration)
    audio_clip = None
    keep_alive: list[object] = []
    if audio_path.exists():
        audio_clip, keep_alive = build_audio_track(audio_path, duration)
        clip = clip.with_audio(audio_clip)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_kwargs = {
        "fps": fps,
        "codec": "libx264",
        "ffmpeg_params": ["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
    }
    if audio_clip is not None:
        write_kwargs["audio_codec"] = "aac"
    else:
        write_kwargs["audio"] = False
    clip.write_videofile(str(output_path), **write_kwargs)

    clip.close()
    if audio_clip is not None:
        audio_clip.close()
    for item in keep_alive:
        close = getattr(item, "close", None)
        if callable(close):
            close()
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a vertical Tour de France stage wins bar chart race Short.")
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
    print(f"[video_generator] Tour de France stage wins Short generated -> {output}")


if __name__ == "__main__":
    main()
