from __future__ import annotations

import argparse
import colorsys
import csv
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
from moviepy import VideoClip
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from video_generator.generate_ucl_barchart_race_moviepy import (
    DEFAULT_AUDIO,
    _fit_font_size,
    _load_font,
    build_audio_track,
)
from video_generator.technology.generate_browser_market_share_race_shorts_moviepy import (
    _center_text,
    _continuous_rank_position,
    _ease_in_out,
    _mix_rgb,
    _phase_delay,
    _truncate,
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
    / "france_female_first_names_race_1900_2024_3min.mp4"
)

WIDTH = 1920
HEIGHT = 1080
TOP_N = 12
FPS = 60
TOTAL_DURATION = 180.0
FINAL_HOLD_DURATION = 10.0

TITLE = "PRÉNOMS FÉMININS"
SUBTITLE = "LES PLUS DONNÉS EN FRANCE CHAQUE ANNÉE | 1900-2024"
FOOTER = "PRÉNOMS FÉMININS | FRANCE | 1900-2024"

FEATURED_COLORS = {
    "Marie": "#E55B72",
    "Jeanne": "#EAA85D",
    "Louise": "#4FB7B3",
    "Marguerite": "#D48BDD",
    "Madeleine": "#F07C68",
    "Jacqueline": "#699DE0",
    "Monique": "#C9856A",
    "Martine": "#EB6F9C",
    "Catherine": "#8A78D1",
    "Sylvie": "#54B58A",
    "Nathalie": "#E95E76",
    "Valérie": "#B477D5",
    "Sandrine": "#E49153",
    "Céline": "#4AAFC2",
    "Emilie": "#728ED6",
    "Aurélie": "#D871B2",
    "Elodie": "#DFA552",
    "Laura": "#5AA8DA",
    "Julie": "#EF7A86",
    "Léa": "#4DBDA7",
    "Manon": "#E36B9A",
    "Camille": "#7B8FE4",
    "Chloé": "#E79C48",
    "Emma": "#E65E69",
    "Jade": "#48A985",
    "Alice": "#6A9FD8",
    "Ambre": "#D79842",
    "Alba": "#D8708E",
    "Alma": "#8A7ACB",
    "Romy": "#54AAB8",
    "Rose": "#E36A7B",
}


@dataclass(frozen=True)
class NameState:
    first_name: str
    births: float


@dataclass(frozen=True)
class Snapshot:
    ranking_date: str
    year: int
    season_summary: str
    states: list[NameState]


def load_snapshots(input_csv: Path) -> list[Snapshot]:
    grouped: dict[str, list[NameState]] = defaultdict(list)
    summaries: dict[str, str] = {}
    with input_csv.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            ranking_date = row["ranking_date"].strip()
            grouped[ranking_date].append(
                NameState(
                    first_name=row["first_name"].strip(),
                    births=float(row["births"]),
                )
            )
            summaries[ranking_date] = row.get("season_summary", "").strip()

    snapshots: list[Snapshot] = []
    for ranking_date in sorted(grouped):
        snapshots.append(
            Snapshot(
                ranking_date=ranking_date,
                year=int(ranking_date[:4]),
                season_summary=summaries.get(ranking_date, ""),
                states=sorted(
                    grouped[ranking_date],
                    key=lambda state: (-state.births, state.first_name),
                ),
            )
        )
    return snapshots


def _interpolate(prev: Snapshot, nxt: Snapshot, alpha: float) -> list[NameState]:
    previous = {state.first_name: state for state in prev.states}
    target = {state.first_name: state for state in nxt.states}
    states: list[NameState] = []
    for name in sorted(set(previous) | set(target)):
        before = previous.get(name)
        after = target.get(name)
        before_value = before.births if before else 0.0
        after_value = after.births if after else 0.0
        states.append(
            NameState(
                first_name=name,
                births=before_value + (after_value - before_value) * alpha,
            )
        )
    return states


def _rank(
    states: list[NameState],
    top_n: int,
    priority: dict[str, int] | None = None,
) -> dict[str, int]:
    priority = priority or {}
    ranked = sorted(
        (state for state in states if state.births > 0),
        key=lambda state: (
            -state.births,
            priority.get(state.first_name, 10_000),
            state.first_name,
        ),
    )
    return {state.first_name: index for index, state in enumerate(ranked[:top_n])}


def _build_priorities(snapshots: list[Snapshot]) -> list[dict[str, int]]:
    priorities: list[dict[str, int]] = []
    previous: dict[str, int] = {}
    for snapshot in snapshots:
        ranked = sorted(
            snapshot.states,
            key=lambda state: (
                -state.births,
                previous.get(state.first_name, 10_000),
                state.first_name,
            ),
        )
        current = {state.first_name: index for index, state in enumerate(ranked)}
        priorities.append(current)
        previous = current
    return priorities


def _name_color(name: str) -> str:
    explicit = FEATURED_COLORS.get(name)
    if explicit:
        return explicit
    seed = sum((index + 1) * ord(char) for index, char in enumerate(name))
    hue = ((seed * 0.61803398875) % 1.0)
    saturation = 0.50 + ((seed % 13) / 100.0)
    lightness = 0.54 + ((seed % 9) / 100.0)
    red, green, blue = colorsys.hls_to_rgb(hue, lightness, saturation)
    return f"#{round(red * 255):02X}{round(green * 255):02X}{round(blue * 255):02X}"


def _build_color_map(snapshots: list[Snapshot]) -> dict[str, str]:
    return {
        state.first_name: _name_color(state.first_name)
        for snapshot in snapshots
        for state in snapshot.states
    }


def _nice_number(value: float) -> float:
    if value <= 0:
        return 1.0
    exponent = math.floor(math.log10(value))
    fraction = value / (10**exponent)
    if fraction <= 1:
        nice_fraction = 1
    elif fraction <= 2:
        nice_fraction = 2
    elif fraction <= 2.5:
        nice_fraction = 2.5
    elif fraction <= 5:
        nice_fraction = 5
    else:
        nice_fraction = 10
    return nice_fraction * (10**exponent)


def _axis_scale(maximum: float) -> tuple[float, float]:
    step = _nice_number(maximum * 1.10 / 5.0)
    cap = math.ceil(maximum * 1.10 / step) * step
    return max(step, cap), step


def _format_births(value: float) -> str:
    return f"{int(round(value)):,}".replace(",", " ")


def _make_background() -> Image.Image:
    x_values = np.linspace(0, 1, WIDTH, dtype=np.float32)
    y_values = np.linspace(0, 1, HEIGHT, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(x_values, y_values)
    ink = np.array([12, 20, 38], dtype=np.float32)
    blue = np.array([24, 67, 102], dtype=np.float32)
    coral = np.array([229, 91, 114], dtype=np.float32)
    cream = np.array([246, 221, 177], dtype=np.float32)

    mix = np.clip(0.45 * grid_x + 0.58 * grid_y, 0, 1)
    coral_glow = np.exp(-(((grid_x - 0.08) / 0.27) ** 2 + ((grid_y - 0.88) / 0.24) ** 2))
    cream_glow = np.exp(-(((grid_x - 0.88) / 0.28) ** 2 + ((grid_y - 0.07) / 0.18) ** 2))
    pixels = np.clip(
        ink[None, None, :] * (1.0 - mix[..., None])
        + blue[None, None, :] * (0.78 * mix[..., None])
        + coral[None, None, :] * (0.08 * coral_glow[..., None])
        + cream[None, None, :] * (0.07 * cream_glow[..., None]),
        0,
        255,
    ).astype(np.uint8)
    frame = Image.fromarray(pixels, mode="RGB").convert("RGBA")

    overlay = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")
    draw.rounded_rectangle(
        (28, 24, WIDTH - 28, HEIGHT - 28),
        radius=42,
        outline=(255, 255, 255, 18),
        width=2,
    )
    draw.arc((1450, -190, 2050, 410), 70, 280, fill=(246, 221, 177, 26), width=4)
    draw.arc((1535, -105, 1965, 325), 70, 280, fill=(246, 221, 177, 14), width=2)
    draw.line((65, 1000, 760, 900), fill=(229, 91, 114, 15), width=3)
    draw.line((110, 1025, 830, 925), fill=(229, 91, 114, 8), width=2)
    for x, y in ((110, 940), (245, 985), (1590, 740), (1810, 890)):
        draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=(246, 221, 177, 72))
        draw.ellipse((x - 22, y - 22, x + 22, y + 22), outline=(246, 221, 177, 18), width=2)
    overlay = overlay.filter(ImageFilter.GaussianBlur(radius=1.8))
    frame.alpha_composite(overlay)
    return frame


def render_video(
    input_csv: Path,
    output_path: Path,
    audio_path: Path,
    duration: float,
    final_hold_duration: float,
    fps: int,
    top_n: int,
) -> Path:
    snapshots = load_snapshots(input_csv)
    if len(snapshots) < 2:
        raise RuntimeError("Not enough female first-name snapshots to render.")

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

    colors = _build_color_map(snapshots)
    priorities = _build_priorities(snapshots)
    periods = len(snapshots) - 1
    transition_duration = max(0.1, duration - max(0.0, final_hold_duration))
    seconds_per_period = transition_duration / periods
    axis_scales = [
        _axis_scale(max((state.births for state in snapshot.states[:top_n]), default=1.0))
        for snapshot in snapshots
    ]
    axis_scales[0] = axis_scales[1]

    background = _make_background()
    title_font = _load_font(56, bold=True)
    subtitle_font = _load_font(23, bold=True)
    year_font = _load_font(70, bold=True)
    insight_font_cache: dict[str, ImageFont.ImageFont] = {}
    label_font = _load_font(18, bold=True)
    name_font = _load_font(29, bold=True)
    value_font = _load_font(27, bold=True)
    rank_font = _load_font(24, bold=True)
    tick_font = _load_font(18, bold=True)
    initial_font = _load_font(23, bold=True)
    footer_font = _load_font(18, bold=True)

    header_box = (38, 34, WIDTH - 38, 188)
    insight_box = (850, 58, 1450, 163)
    year_box = (1510, 52, 1844, 168)
    rank_left = 48
    initial_left = 116
    name_left = 186
    bar_left = 450
    bar_right = 1778
    bar_max_width = bar_right - bar_left
    base_y = 256
    pitch = 64
    row_height = 48
    ranking_bottom = base_y + (top_n - 1) * pitch + row_height

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
        interpolated = _interpolate(prev, nxt, value_alpha)
        states_by_name = {state.first_name: state for state in interpolated}
        priority = priorities[period_index]
        previous_rank = _rank(prev.states, top_n, priority)
        target_rank = _rank(nxt.states, top_n, priority)
        visible_names = sorted(set(previous_rank) | set(target_rank))
        axis_cap = axis_scales[period_index][0] + (
            axis_scales[period_index + 1][0] - axis_scales[period_index][0]
        ) * value_alpha
        tick_step = axis_scales[period_index][1] + (
            axis_scales[period_index + 1][1] - axis_scales[period_index][1]
        ) * value_alpha

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

        draw.text((name_left, 210), "PRÉNOM", font=label_font, fill=(202, 218, 224, 210))
        draw.text((bar_left + 18, 210), "NAISSANCES", font=label_font, fill=(202, 218, 224, 210))

        tick_count = max(1, int(math.floor(axis_cap / max(tick_step, 1.0))))
        for tick_index in range(tick_count + 1):
            value = min(axis_cap, tick_index * tick_step)
            x = bar_left + int((value / axis_cap) * bar_max_width)
            draw.line(
                (x, 239, x, ranking_bottom + 14),
                fill=(2, 9, 20, 98),
                width=3 if tick_index == 0 else 2,
            )
            tick_text = _format_births(value) if tick_index else "0"
            bbox = draw.textbbox((0, 0), tick_text, font=tick_font)
            draw.text(
                (x - (bbox[2] - bbox[0]) // 2, 208),
                tick_text,
                font=tick_font,
                fill=(7, 20, 34, 172),
            )

        for rank_index in range(top_n):
            row_y = base_y + rank_index * pitch
            fill = (246, 221, 177, 255) if rank_index == 0 else (26, 78, 98, 245)
            text_fill = "#19283B" if rank_index == 0 else "#F1F8F8"
            rank_box = (rank_left, row_y, rank_left + 52, row_y + row_height)
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
            bar_width = max(0, int((state.births / axis_cap) * bar_max_width))
            moving_up = 1 if target_index < previous_index else 0
            render_items.append((moving_up, row_y, state, bar_width))

        render_items.sort(key=lambda item: (item[0], item[1]))
        for _, row_y, state, bar_width in render_items:
            y0 = int(row_y)
            y1 = y0 + row_height
            if y1 < base_y - pitch or y0 > ranking_bottom + pitch:
                continue

            color = colors[state.first_name]
            highlight = _mix_rgb(color, (255, 255, 255), 0.30)
            shadow = _mix_rgb(color, (0, 0, 0), 0.25)
            draw.rounded_rectangle(
                (108, y0 - 3, WIDTH - 55, y1 + 3),
                radius=19,
                fill=(4, 14, 27, 58),
            )

            initial_box = (initial_left, y0 - 2, initial_left + 54, y1 + 2)
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
            name_y = y0 + (row_height - (name_bbox[3] - name_bbox[1])) // 2 - name_bbox[1]
            draw.text((name_left, name_y), display_name, font=name_font, fill="#F8F5F1")

            if bar_width > 0:
                draw.rounded_rectangle(
                    (bar_left + 6, y0 + 6, bar_left + bar_width + 6, y1 + 6),
                    radius=20,
                    fill=(0, 0, 0, 86),
                )
                draw.rounded_rectangle(
                    (bar_left, y0, bar_left + max(7, bar_width), y1),
                    radius=20,
                    fill=color,
                    outline=highlight,
                    width=2,
                )
                if bar_width > 42:
                    draw.rounded_rectangle(
                        (bar_left + 9, y0 + 8, bar_left + max(30, int(bar_width * 0.64)), y0 + 17),
                        radius=6,
                        fill=(*highlight, 64),
                    )
                    draw.line(
                        (
                            bar_left + 18,
                            y1 - 8,
                            bar_left + max(30, int(bar_width * 0.72)),
                            y1 - 8,
                        ),
                        fill=(*shadow, 102),
                        width=3,
                    )

            value_text = _format_births(state.births)
            value_bbox = draw.textbbox((0, 0), value_text, font=value_font)
            value_width = value_bbox[2] - value_bbox[0]
            value_x = min(
                bar_left + max(7, bar_width) + 15,
                WIDTH - 42 - value_width,
            )
            value_y = y0 + (row_height - (value_bbox[3] - value_bbox[1])) // 2 - value_bbox[1]
            draw.text((value_x, value_y), value_text, font=value_font, fill="#FFF9F5")

        footer = FOOTER
        footer_bbox = draw.textbbox((0, 0), footer, font=footer_font)
        draw.text(
            ((WIDTH - (footer_bbox[2] - footer_bbox[0])) // 2, 1037),
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
        description="Generate a landscape French female first-name bar chart race."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
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
        audio_path=args.audio,
        duration=args.duration,
        final_hold_duration=args.final_hold,
        fps=args.fps,
        top_n=args.top_n,
    )
    print(f"[video_generator] France female first-name race generated -> {output}")


if __name__ == "__main__":
    main()
