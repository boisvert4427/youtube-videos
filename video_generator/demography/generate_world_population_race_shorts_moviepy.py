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

from video_generator.demography.generate_world_population_race_moviepy import (
    DEFAULT_AUDIO,
    DEFAULT_FLAGS_DIR,
    DEFAULT_INPUT,
    CountryState,
    Snapshot,
    _axis_scale,
    _build_color_map,
    _build_flag_cache,
    _build_priorities,
    _center_text,
    _continuous_rank_position,
    _ease_in_out,
    _fit_font_size,
    _format_population,
    _interpolate,
    _load_font,
    _mix_rgb,
    _phase_delay,
    _rank,
    _truncate,
    build_audio_track,
    load_snapshots,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "demography"
    / "world_population"
    / "world_population_race_1960_2024_shorts.mp4"
)

WIDTH = 1080
HEIGHT = 1920
TOP_N = 12
FPS = 60
TOTAL_DURATION = 60.0
FINAL_HOLD_DURATION = 5.0

TITLE = "WORLD POPULATION"
SUBTITLE = "TOP 12 COUNTRIES | 1960-2024"
LEFT_HEADER_LABEL = "COUNTRY"
RIGHT_HEADER_LABEL = "POPULATION"
FOOTER = "SOURCE: WORLD BANK | INDICATOR SP.POP.TOTL"


def YEAR_LABEL(snapshot: Snapshot) -> str:
    return str(snapshot.year)


def _make_background() -> Image.Image:
    x_values = np.linspace(0, 1, WIDTH, dtype=np.float32)
    y_values = np.linspace(0, 1, HEIGHT, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(x_values, y_values)
    deep = np.array([3, 15, 28], dtype=np.float32)
    ocean = np.array([7, 52, 72], dtype=np.float32)
    teal = np.array([28, 136, 132], dtype=np.float32)
    sand = np.array([244, 194, 107], dtype=np.float32)

    mix = np.clip(0.18 * grid_x + 0.78 * grid_y, 0, 1)
    top_glow = np.exp(-(((grid_x - 0.80) / 0.36) ** 2 + ((grid_y - 0.06) / 0.12) ** 2))
    lower_glow = np.exp(-(((grid_x - 0.06) / 0.32) ** 2 + ((grid_y - 0.92) / 0.20) ** 2))
    pixels = np.clip(
        deep[None, None, :] * (1.0 - mix[..., None])
        + ocean[None, None, :] * (0.84 * mix[..., None])
        + teal[None, None, :] * (0.20 * top_glow[..., None])
        + sand[None, None, :] * (0.06 * lower_glow[..., None]),
        0,
        255,
    ).astype(np.uint8)
    frame = Image.fromarray(pixels, mode="RGB").convert("RGBA")

    overlay = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")
    draw.rounded_rectangle((22, 22, WIDTH - 22, HEIGHT - 22), radius=42, outline=(255, 255, 255, 20), width=2)
    draw.ellipse((640, -80, 1160, 440), outline=(102, 224, 210, 22), width=3)
    draw.ellipse((735, 15, 1065, 345), outline=(102, 224, 210, 13), width=2)
    for offset in (-75, 0, 75):
        draw.arc((640 + offset, -80, 1160 - offset, 440), 80, 280, fill=(102, 224, 210, 12), width=2)
    draw.line((80, 1830, 980, 1710), fill=(244, 194, 107, 10), width=2)
    overlay = overlay.filter(ImageFilter.GaussianBlur(radius=2))
    frame.alpha_composite(overlay)
    return frame


def render_video(
    input_csv: Path,
    output_path: Path,
    flags_dir: Path,
    audio_path: Path,
    duration: float,
    final_hold_duration: float,
    fps: int,
    top_n: int,
) -> Path:
    snapshots = load_snapshots(input_csv)
    if len(snapshots) < 2:
        raise RuntimeError("Not enough World Population snapshots to render.")

    first = snapshots[0]
    snapshots = [
        Snapshot(
            ranking_date=f"{first.year - 1}-12-31",
            year=first.year,
            season_summary=first.season_summary,
            states=[],
        ),
        *snapshots,
    ]

    all_states = [state for snapshot in snapshots for state in snapshot.states]
    flags = _build_flag_cache(all_states, flags_dir)
    colors = _build_color_map(snapshots)
    priorities = _build_priorities(snapshots)

    periods = len(snapshots) - 1
    transition_duration = max(0.1, duration - max(0.0, final_hold_duration))
    seconds_per_period = transition_duration / periods
    axis_scales = [
        _axis_scale(max((state.population for state in snapshot.states[:top_n]), default=1.0))
        for snapshot in snapshots
    ]
    axis_scales[0] = axis_scales[1]

    background = _make_background()
    title_font = _load_font(65, bold=True)
    subtitle_font = _load_font(28, bold=True)
    year_font = _load_font(92, bold=True)
    insight_font_cache: dict[str, ImageFont.ImageFont] = {}
    name_font = _load_font(30, bold=True)
    value_font = _load_font(29, bold=True)
    rank_font = _load_font(27, bold=True)
    tick_font = _load_font(18, bold=True)
    footer_font = _load_font(20, bold=True)
    measure_canvas = Image.new("RGB", (1, 1))
    measure_draw = ImageDraw.Draw(measure_canvas)
    name_font_cache: dict[str, ImageFont.ImageFont] = {}

    header_box = (34, 36, WIDTH - 34, 430)
    year_box = (344, 174, 736, 300)
    insight_box = (66, 324, WIDTH - 66, 406)

    rank_left = 24
    label_left = 104
    flag_left = 106
    name_left = 170
    bar_left = 104
    bar_right = 1024
    bar_max_width = bar_right - bar_left
    base_y = 502
    pitch = 108
    bar_height = 44
    label_height = 39
    ranking_bottom = base_y + (top_n - 1) * pitch + label_height + bar_height

    def _name_font_for(text: str, max_width: int) -> ImageFont.ImageFont:
        font = name_font_cache.get(text)
        if font is None or measure_draw.textbbox((0, 0), text, font=font)[2] > max_width:
            font = _fit_font_size(measure_draw, text, max_width, 30, 18, bold=True)
            name_font_cache[text] = font
        return font

    def make_frame(t: float) -> np.ndarray:
        frame = background.copy()
        draw = ImageDraw.Draw(frame, "RGBA")

        if t >= transition_duration:
            period_index = periods - 1
            value_alpha = 1.0
            rank_alpha = 1.0
        else:
            period_index = min(int(t / seconds_per_period), periods - 1)
            local_time = (t - period_index * seconds_per_period) / seconds_per_period
            value_alpha = min(max(local_time, 0.0), 1.0)
            rank_alpha = _ease_in_out(value_alpha)

        prev = snapshots[period_index]
        nxt = snapshots[period_index + 1]
        axis_cap = axis_scales[period_index][0] + (
            axis_scales[period_index + 1][0] - axis_scales[period_index][0]
        ) * value_alpha
        tick_step = axis_scales[period_index][1] + (
            axis_scales[period_index + 1][1] - axis_scales[period_index][1]
        ) * value_alpha

        interpolated = _interpolate(prev, nxt, value_alpha)
        states_by_iso3 = {state.country_iso3: state for state in interpolated}
        priority = priorities[period_index]
        previous_rank = _rank(prev.states, top_n, priority)
        target_rank = _rank(nxt.states, top_n, priority)
        visible_iso3 = sorted(set(previous_rank) | set(target_rank))

        draw.rounded_rectangle(header_box, radius=38, fill=(3, 16, 30, 224), outline=(255, 255, 255, 26), width=2)
        title_bbox = draw.textbbox((0, 0), TITLE, font=title_font)
        draw.text(
            ((WIDTH - (title_bbox[2] - title_bbox[0])) // 2, 62),
            TITLE,
            font=title_font,
            fill="#f5f8fb",
        )
        subtitle_bbox = draw.textbbox((0, 0), SUBTITLE, font=subtitle_font)
        draw.text(
            ((WIDTH - (subtitle_bbox[2] - subtitle_bbox[0])) // 2, 133),
            SUBTITLE,
            font=subtitle_font,
            fill="#83d9d0",
        )

        draw.rounded_rectangle(year_box, radius=34, fill=(244, 194, 107, 255), outline=(255, 235, 184, 180), width=2)
        _center_text(draw, year_box, YEAR_LABEL(nxt), year_font, "#10273a")

        draw.rounded_rectangle(insight_box, radius=24, fill=(9, 43, 58, 232), outline=(102, 224, 210, 54), width=2)
        insight_lines = [part.strip() for part in nxt.season_summary.split("|") if part.strip()][:2]
        insight_text = "  |  ".join(insight_lines)
        insight_font = insight_font_cache.get(insight_text)
        if insight_font is None:
            insight_font = _fit_font_size(draw, insight_text, insight_box[2] - insight_box[0] - 36, 25, 16, bold=True)
            insight_font_cache[insight_text] = insight_font
        _center_text(
            draw,
            (insight_box[0] + 18, insight_box[1], insight_box[2] - 18, insight_box[3]),
            insight_text,
            insight_font,
            "#eef9f7",
        )

        tick_count = max(1, int(math.floor(axis_cap / max(tick_step, 1.0))))
        for tick_index in range(tick_count + 1):
            value = min(axis_cap, tick_index * tick_step)
            x = bar_left + int((value / axis_cap) * bar_max_width)
            draw.line(
                (x, base_y - 35, x, ranking_bottom + 10),
                fill=(1, 9, 18, 76),
                width=2 if tick_index else 3,
            )
            if tick_index:
                tick_text = _format_population(value)
                bbox = draw.textbbox((0, 0), tick_text, font=tick_font)
                draw.text(
                    (x - (bbox[2] - bbox[0]) // 2, base_y - 34),
                    tick_text,
                    font=tick_font,
                    fill=(2, 13, 24, 155),
                )

        for rank_index in range(top_n):
            row_y = base_y + rank_index * pitch
            fill = (244, 194, 107, 255) if rank_index == 0 else (20, 73, 87, 245)
            text_fill = "#10273a" if rank_index == 0 else "#edf8f7"
            rank_box = (rank_left, row_y + 4, rank_left + 66, row_y + 70)
            draw.rounded_rectangle(rank_box, radius=21, fill=fill)
            _center_text(draw, rank_box, str(rank_index + 1), rank_font, text_fill)

        render_items: list[tuple[int, float, CountryState, int]] = []
        for iso3 in visible_iso3:
            state = states_by_iso3.get(iso3)
            if state is None:
                continue
            previous_index = previous_rank.get(iso3, top_n + 1)
            target_index = target_rank.get(iso3, top_n + 1)
            entering = previous_index > top_n and target_index <= top_n
            effective_previous = float(top_n + 1.8) if entering else float(previous_index)
            movement_alpha = rank_alpha
            places_moved = abs(float(target_index) - effective_previous)
            if places_moved > 0:
                delay = min(0.08, max(0.0, (places_moved - 1.0) * 0.015))
                movement_alpha = _ease_in_out(
                    _phase_delay(movement_alpha, delay, max(0.82, 0.98 - delay))
                )
            if entering:
                movement_alpha = _ease_in_out(_phase_delay(movement_alpha, 0.02, 0.96))
            y_index = _continuous_rank_position(effective_previous, float(target_index), movement_alpha)
            row_y = base_y + y_index * pitch
            bar_width = max(8, int((state.population / axis_cap) * bar_max_width))
            moving_up = 1 if target_index < previous_index else 0
            render_items.append((moving_up, row_y, state, bar_width))

        render_items.sort(key=lambda item: (item[0], item[1]))
        for _, row_y, state, bar_width in render_items:
            label_y = int(row_y)
            bar_y = label_y + label_height
            if bar_y + bar_height < base_y - pitch or label_y > ranking_bottom + pitch:
                continue

            color = colors[state.country_iso3]
            highlight = _mix_rgb(color, (255, 255, 255), 0.28)
            shadow = _mix_rgb(color, (0, 0, 0), 0.24)

            flag = flags.get(state.country_code)
            if flag is not None:
                flag_y = label_y + (label_height - flag.height) // 2
                draw.rounded_rectangle(
                    (flag_left - 4, flag_y - 4, flag_left + flag.width + 4, flag_y + flag.height + 4),
                    radius=7,
                    fill=(247, 250, 252, 238),
                )
                frame.alpha_composite(flag, (flag_left, flag_y))

            value_text = _format_population(state.population)
            value_bbox = draw.textbbox((0, 0), value_text, font=value_font)
            value_x = WIDTH - 28 - (value_bbox[2] - value_bbox[0])
            value_y = label_y + (label_height - (value_bbox[3] - value_bbox[1])) // 2 - value_bbox[1]
            draw.text((value_x, value_y), value_text, font=value_font, fill="#f7fbfc")

            max_name_width = value_x - name_left - 18
            country_name = state.country_name
            name_font_state = _name_font_for(country_name, max_name_width)
            name_bbox = draw.textbbox((0, 0), country_name, font=name_font_state)
            name_y = label_y + (label_height - (name_bbox[3] - name_bbox[1])) // 2 - name_bbox[1]
            draw.text((name_left, name_y), country_name, font=name_font_state, fill="#f1f7f8")

            draw.rounded_rectangle(
                (bar_left + 6, bar_y + 6, bar_left + bar_width + 6, bar_y + bar_height + 6),
                radius=18,
                fill=(0, 0, 0, 80),
            )
            draw.rounded_rectangle(
                (bar_left, bar_y, bar_left + bar_width, bar_y + bar_height),
                radius=18,
                fill=color,
                outline=highlight,
                width=2,
            )
            if bar_width > 40:
                draw.rounded_rectangle(
                    (bar_left + 9, bar_y + 7, bar_left + max(28, int(bar_width * 0.64)), bar_y + 16),
                    radius=6,
                    fill=(*highlight, 58),
                )
                draw.line(
                    (
                        bar_left + 18,
                        bar_y + bar_height - 7,
                        bar_left + max(28, int(bar_width * 0.72)),
                        bar_y + bar_height - 7,
                    ),
                    fill=(*shadow, 92),
                    width=3,
                )

        footer = FOOTER
        footer_bbox = draw.textbbox((0, 0), footer, font=footer_font)
        draw.text(
            ((WIDTH - (footer_bbox[2] - footer_bbox[0])) // 2, 1852),
            footer,
            font=footer_font,
            fill=(174, 207, 214, 185),
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
    parser = argparse.ArgumentParser(description="Generate a vertical World Population bar chart race Short.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--flags-dir", type=Path, default=DEFAULT_FLAGS_DIR)
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
        audio_path=args.audio,
        duration=args.duration,
        final_hold_duration=args.final_hold,
        fps=args.fps,
        top_n=args.top_n,
    )
    print(f"[video_generator] World Population Short generated -> {output}")


if __name__ == "__main__":
    main()
