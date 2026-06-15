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

from video_generator.demography.generate_france_female_first_names_race_moviepy import (
    DEFAULT_AUDIO,
    NameState,
    Snapshot,
    _axis_scale,
    _build_color_map,
    _build_priorities,
    _center_text,
    _continuous_rank_position,
    _ease_in_out,
    _fit_font_size,
    _format_births,
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
DEFAULT_INPUT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "demography"
    / "france_female_first_names"
    / "france_female_first_names_1900_2024.csv"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "demography"
    / "france_female_first_names"
    / "france_female_first_names_race_1900_2024_shorts.mp4"
)

WIDTH = 1080
HEIGHT = 1920
TOP_N = 12
FPS = 60
TOTAL_DURATION = 100.0
FINAL_HOLD_DURATION = 8.0
INTRO_HOLD_DURATION = 4.0

TITLE = "PRÉNOMS FÉMININS"
SUBTITLE = "FRANCE | 1900-2024"
FOOTER = "PRÉNOMS FÉMININS | FRANCE | 1900-2024"


def _make_background() -> Image.Image:
    x_values = np.linspace(0, 1, WIDTH, dtype=np.float32)
    y_values = np.linspace(0, 1, HEIGHT, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(x_values, y_values)

    deep = np.array([4, 12, 26], dtype=np.float32)
    navy = np.array([11, 34, 63], dtype=np.float32)
    coral = np.array([229, 91, 114], dtype=np.float32)
    cream = np.array([246, 221, 177], dtype=np.float32)
    teal = np.array([74, 170, 165], dtype=np.float32)

    mix = np.clip(0.18 * grid_x + 0.80 * grid_y, 0, 1)
    upper_glow = np.exp(-(((grid_x - 0.84) / 0.33) ** 2 + ((grid_y - 0.08) / 0.12) ** 2))
    lower_glow = np.exp(-(((grid_x - 0.10) / 0.30) ** 2 + ((grid_y - 0.90) / 0.20) ** 2))
    pixels = np.clip(
        deep[None, None, :] * (1.0 - mix[..., None])
        + navy[None, None, :] * (0.82 * mix[..., None])
        + coral[None, None, :] * (0.15 * lower_glow[..., None])
        + cream[None, None, :] * (0.07 * upper_glow[..., None])
        + teal[None, None, :] * (0.04 * upper_glow[..., None]),
        0,
        255,
    ).astype(np.uint8)

    frame = Image.fromarray(pixels, mode="RGB").convert("RGBA")
    overlay = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")

    draw.rounded_rectangle((22, 22, WIDTH - 22, HEIGHT - 22), radius=42, outline=(255, 255, 255, 18), width=2)
    draw.arc((610, -150, 1210, 450), 70, 280, fill=(246, 221, 177, 18), width=4)
    draw.arc((700, -55, 1120, 365), 70, 280, fill=(229, 91, 114, 12), width=3)
    draw.line((72, 1760, 990, 1635), fill=(74, 170, 165, 10), width=3)
    draw.line((100, 1785, 930, 1670), fill=(246, 221, 177, 7), width=2)
    for x, y in ((120, 980), (250, 1030), (880, 760), (940, 860)):
        draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=(246, 221, 177, 66))
        draw.ellipse((x - 22, y - 22, x + 22, y + 22), outline=(246, 221, 177, 16), width=2)

    overlay = overlay.filter(ImageFilter.GaussianBlur(radius=2))
    frame.alpha_composite(overlay)
    return frame


def render_video(
    input_csv: Path,
    output_path: Path,
    audio_path: Path,
    duration: float,
    final_hold_duration: float,
    intro_hold_duration: float,
    fps: int,
    top_n: int,
) -> Path:
    snapshots = load_snapshots(input_csv)
    if len(snapshots) < 2:
        raise RuntimeError("Not enough female first-name snapshots to render.")

    colors = _build_color_map(snapshots)
    priorities = _build_priorities(snapshots)

    periods = len(snapshots) - 1
    transition_duration = max(0.1, duration - max(0.0, final_hold_duration) - max(0.0, intro_hold_duration))
    seconds_per_period = transition_duration / periods
    axis_scales = [
        _axis_scale(max((state.births for state in snapshot.states[:top_n]), default=1.0))
        for snapshot in snapshots
    ]
    axis_scales[0] = axis_scales[1]

    background = _make_background()
    title_font = _load_font(60, bold=True)
    subtitle_font = _load_font(24, bold=True)
    year_font = _load_font(74, bold=True)
    insight_font_cache: dict[str, ImageFont.ImageFont] = {}
    label_font = _load_font(18, bold=True)
    name_font = _load_font(29, bold=True)
    value_font = _load_font(27, bold=True)
    rank_font = _load_font(24, bold=True)
    tick_font = _load_font(18, bold=True)
    initial_font = _load_font(23, bold=True)
    footer_font = _load_font(18, bold=True)

    header_box = (34, 36, WIDTH - 34, 430)
    insight_box = (66, 324, WIDTH - 66, 406)
    year_box = (344, 174, 736, 300)

    rank_left = 24
    initial_left = 104
    name_left = 170
    bar_left = 450
    bar_right = 1024
    bar_max_width = bar_right - bar_left
    base_y = 502
    pitch = 108
    row_height = 64
    bar_height = 44
    ranking_bottom = base_y + (top_n - 1) * pitch + row_height

    def make_frame(t: float) -> np.ndarray:
        frame = background.copy()
        draw = ImageDraw.Draw(frame, "RGBA")

        if t < intro_hold_duration:
            period_time = 0.0
        else:
            period_time = min(max(t - intro_hold_duration, 0.0), transition_duration)

        if period_time >= transition_duration:
            period_index = periods - 1
            value_alpha = 1.0
            rank_alpha = 1.0
        else:
            period_index = min(int(period_time / seconds_per_period), periods - 1)
            local_time = (period_time - period_index * seconds_per_period) / seconds_per_period
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
        states_by_name = {state.first_name: state for state in interpolated}
        priority = priorities[period_index]
        previous_rank = _rank(prev.states, top_n, priority)
        target_rank = _rank(nxt.states, top_n, priority)
        visible_names = sorted(set(previous_rank) | set(target_rank))

        draw.rounded_rectangle(
            header_box,
            radius=35,
            fill=(7, 17, 34, 232),
            outline=(246, 221, 177, 42),
            width=2,
        )
        draw.text((70, 54), TITLE, font=title_font, fill="#FAF7F2")
        draw.text((73, 124), SUBTITLE, font=subtitle_font, fill="#F0A9B5")

        draw.rounded_rectangle(
            insight_box,
            radius=27,
            fill=(19, 48, 72, 238),
            outline=(240, 169, 181, 72),
            width=2,
        )
        visible_leader = max(
            interpolated,
            key=lambda state: state.births,
            default=NameState("N/A", 0.0),
        )
        display_year = prev.year if value_alpha < 0.5 and prev.states else nxt.year
        insight = f"N°1 : {visible_leader.first_name}  |  {_format_births(visible_leader.births)} naissances"
        insight_font = insight_font_cache.get(insight)
        if insight_font is None:
            insight_font = _fit_font_size(
                draw,
                insight,
                insight_box[2] - insight_box[0] - 36,
                25,
                17,
                bold=True,
            )
            insight_font_cache[insight] = insight_font
        _center_text(
            draw,
            (insight_box[0] + 16, insight_box[1], insight_box[2] - 16, insight_box[3]),
            insight,
            insight_font,
            "#FFF7F3",
        )

        draw.rounded_rectangle(
            year_box,
            radius=29,
            fill=(246, 221, 177, 255),
            outline=(255, 244, 223, 220),
            width=2,
        )
        _center_text(draw, year_box, str(display_year), year_font, "#19283B")

        column_label_y = 438
        tick_label_y = 456
        axis_line_start_y = base_y - 8

        draw.text((name_left, column_label_y), "PRÉNOM", font=label_font, fill=(202, 218, 224, 210))
        draw.text((bar_left + 18, column_label_y), "NAISSANCES", font=label_font, fill=(202, 218, 224, 210))

        tick_count = max(1, int(math.floor(axis_cap / max(tick_step, 1.0))))
        for tick_index in range(tick_count + 1):
            value = min(axis_cap, tick_index * tick_step)
            x = bar_left + int((value / axis_cap) * bar_max_width)
            draw.line(
                (x, axis_line_start_y, x, ranking_bottom + 14),
                fill=(2, 9, 20, 98),
                width=3 if tick_index == 0 else 2,
            )
            tick_text = _format_births(value) if tick_index else "0"
            bbox = draw.textbbox((0, 0), tick_text, font=tick_font)
            draw.text(
                (x - (bbox[2] - bbox[0]) // 2, tick_label_y),
                tick_text,
                font=tick_font,
                fill=(7, 20, 34, 172),
            )

        for rank_index in range(top_n):
            row_y = base_y + rank_index * pitch
            fill = (246, 221, 177, 255) if rank_index == 0 else (26, 78, 98, 245)
            text_fill = "#19283B" if rank_index == 0 else "#F1F8F8"
            rank_box = (
                rank_left,
                row_y + (row_height - 52) // 2,
                rank_left + 52,
                row_y + (row_height - 52) // 2 + 52,
            )
            draw.rounded_rectangle(rank_box, radius=17, fill=fill)
            _center_text(draw, rank_box, str(rank_index + 1), rank_font, text_fill)

        render_items: list[tuple[int, float, NameState, int]] = []
        for name in visible_names:
            state = states_by_name.get(name)
            if state is None:
                continue
            previous_index = previous_rank.get(name, top_n + 1)
            target_index = target_rank.get(name, top_n + 1)
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
            y_index = _continuous_rank_position(
                effective_previous,
                float(target_index),
                movement_alpha,
            )
            row_y = base_y + y_index * pitch
            bar_width = max(8, int((state.births / axis_cap) * bar_max_width))
            moving_up = 1 if target_index < previous_index else 0
            render_items.append((moving_up, row_y, state, bar_width))

        render_items.sort(key=lambda item: (item[0], item[1]))
        for _, row_y, state, bar_width in render_items:
            row_top = int(row_y)
            row_bottom = row_top + row_height
            bar_y = row_top + (row_height - bar_height) // 2
            if row_bottom < base_y - pitch or row_top > ranking_bottom + pitch:
                continue

            color = colors[state.first_name]
            highlight = _mix_rgb(color, (255, 255, 255), 0.28)
            shadow = _mix_rgb(color, (0, 0, 0), 0.24)

            draw.rounded_rectangle(
                (108, row_top - 3, WIDTH - 55, row_bottom + 3),
                radius=19,
                fill=(4, 14, 27, 58),
            )

            initial_box = (
                initial_left,
                row_top + (row_height - 54) // 2,
                initial_left + 54,
                row_top + (row_height - 54) // 2 + 54,
            )
            draw.rounded_rectangle(
                initial_box,
                radius=15,
                fill=(250, 247, 241, 245),
                outline=(*highlight, 210),
                width=2,
            )
            _center_text(
                draw,
                initial_box,
                state.first_name[:1].upper(),
                initial_font,
                color,
            )

            display_name = _truncate(
                draw,
                state.first_name,
                name_font,
                bar_left - name_left - 22,
            )
            name_bbox = draw.textbbox((0, 0), display_name, font=name_font)
            name_y = row_top + (row_height - (name_bbox[3] - name_bbox[1])) // 2 - name_bbox[1]
            draw.text((name_left, name_y), display_name, font=name_font, fill="#F8F5F1")

            if bar_width > 0:
                draw.rounded_rectangle(
                    (bar_left + 6, bar_y + 6, bar_left + bar_width + 6, bar_y + bar_height + 6),
                    radius=20,
                    fill=(0, 0, 0, 86),
                )
                draw.rounded_rectangle(
                    (bar_left, bar_y, bar_left + max(7, bar_width), bar_y + bar_height),
                    radius=20,
                    fill=color,
                    outline=highlight,
                    width=2,
                )
                if bar_width > 42:
                    draw.rounded_rectangle(
                        (bar_left + 9, bar_y + 8, bar_left + max(30, int(bar_width * 0.64)), bar_y + 17),
                        radius=6,
                        fill=(*highlight, 64),
                    )
                    draw.line(
                        (
                            bar_left + 18,
                            bar_y + bar_height - 8,
                            bar_left + max(30, int(bar_width * 0.72)),
                            bar_y + bar_height - 8,
                        ),
                        fill=(*shadow, 92),
                        width=3,
                    )

            value_text = _format_births(state.births)
            value_bbox = draw.textbbox((0, 0), value_text, font=value_font)
            value_width = value_bbox[2] - value_bbox[0]
            value_x = min(bar_left + max(7, bar_width) + 15, WIDTH - 42 - value_width)
            value_y = row_top + (row_height - (value_bbox[3] - value_bbox[1])) // 2 - value_bbox[1]
            draw.text((value_x, value_y), value_text, font=value_font, fill="#FFF9F5")

        footer = FOOTER
        footer_bbox = draw.textbbox((0, 0), footer, font=footer_font)
        draw.text(
            ((WIDTH - (footer_bbox[2] - footer_bbox[0])) // 2, 1852),
            footer,
            font=footer_font,
            fill=(203, 220, 224, 185),
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
    parser = argparse.ArgumentParser(
        description="Generate a vertical French female first-name bar chart race Short."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--audio", type=Path, default=DEFAULT_AUDIO)
    parser.add_argument("--duration", type=float, default=TOTAL_DURATION)
    parser.add_argument("--final-hold", type=float, default=FINAL_HOLD_DURATION)
    parser.add_argument("--intro-hold", type=float, default=INTRO_HOLD_DURATION)
    parser.add_argument("--fps", type=int, default=FPS)
    parser.add_argument("--top-n", type=int, default=TOP_N)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = render_video(
        input_csv=args.input,
        output_path=args.output,
        audio_path=args.audio,
        duration=args.duration,
        final_hold_duration=args.final_hold,
        intro_hold_duration=args.intro_hold,
        fps=args.fps,
        top_n=args.top_n,
    )
    print(f"[video_generator] France female first-name short generated -> {output}")


if __name__ == "__main__":
    main()
