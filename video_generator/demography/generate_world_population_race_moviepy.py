from __future__ import annotations

import argparse
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
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps

from video_generator.generate_ucl_barchart_race_moviepy import (
    DEFAULT_AUDIO,
    DEFAULT_FLAGS_DIR,
    _fit_font_size,
    _load_font,
    build_audio_track,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_INPUT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "demography"
    / "world_population"
    / "world_population_1960_2024.csv"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "demography"
    / "world_population"
    / "world_population_race_1960_2024_3min.mp4"
)

WIDTH = 1920
HEIGHT = 1080
TOP_N = 12
FPS = 60
TOTAL_DURATION = 180.0
FINAL_HOLD_DURATION = 10.0

TITLE = "WORLD POPULATION RACE"
SUBTITLE = "TOP 12 COUNTRIES | 1960-2024"
LEFT_HEADER_LABEL = "COUNTRY"
RIGHT_HEADER_LABEL = "POPULATION"
FOOTER = "SOURCE: WORLD BANK | INDICATOR SP.POP.TOTL"
SNAP_TO_CURRENT_RANKS = False
SHOW_INSIGHT_BOX = True
SHOW_ROW_BACKPLATE = True
NAME_IN_BAR = False


def YEAR_LABEL(snapshot: Snapshot) -> str:
    return str(snapshot.year)

DISPLAY_NAME_ALIASES = {
    "Russian Federation": "Russia",
}

COUNTRY_COLORS = {
    "CHN": "#ef5b5b",
    "IND": "#ffad49",
    "USA": "#5aa9ff",
    "IDN": "#42c6b6",
    "PAK": "#5dd08a",
    "BRA": "#8ac84a",
    "NGA": "#c1d84c",
    "BGD": "#ef7c9b",
    "RUS": "#8c9cf3",
    "JPN": "#ff8f78",
    "MEX": "#40cece",
    "PHL": "#4d82e8",
    "ETH": "#d8a846",
    "DEU": "#f2cf5b",
    "GBR": "#75a6e8",
    "ITA": "#66c99a",
    "FRA": "#76b7f0",
    "VNM": "#db725e",
    "TUR": "#df6271",
    "THA": "#b989e8",
}

FALLBACK_COLORS = [
    "#ef5b5b",
    "#ffad49",
    "#5aa9ff",
    "#42c6b6",
    "#5dd08a",
    "#8ac84a",
    "#c1d84c",
    "#ef7c9b",
    "#8c9cf3",
    "#ff8f78",
    "#40cece",
    "#4d82e8",
    "#d8a846",
    "#f2cf5b",
]


@dataclass(frozen=True)
class CountryState:
    country_name: str
    country_code: str
    country_iso3: str
    population: float


@dataclass(frozen=True)
class Snapshot:
    ranking_date: str
    year: int
    season_summary: str
    states: list[CountryState]


def _ease_in_out(value: float) -> float:
    value = min(max(value, 0.0), 1.0)
    return value * value * (3.0 - 2.0 * value)


def _phase_delay(value: float, delay: float, span: float) -> float:
    if span <= 0.0:
        return 1.0 if value >= delay else 0.0
    return min(max((value - delay) / span, 0.0), 1.0)


def _continuous_rank_position(previous: float, target: float, alpha: float) -> float:
    alpha = min(max(alpha, 0.0), 1.0)
    if math.isclose(previous, target):
        return target
    distance = abs(target - previous)
    steps = max(1, int(math.ceil(distance)))
    direction = 1.0 if target > previous else -1.0
    gap = 1.0 / steps
    span = min(0.9, gap * 1.35)
    travelled = 0.0
    end_travel = 0.0
    for step in range(steps):
        start = step * gap
        segment = min(1.0, max(0.0, distance - step))
        local = _ease_in_out(min(max((alpha - start) / span, 0.0), 1.0))
        end_local = _ease_in_out(min(max((1.0 - start) / span, 0.0), 1.0))
        travelled += local * segment
        end_travel += end_local * segment
    if end_travel > 1e-9:
        travelled *= distance / end_travel
    return previous + direction * min(distance, travelled)


def _hex_to_rgb(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    return tuple(int(value[index : index + 2], 16) for index in (0, 2, 4))


def _mix_rgb(color: str, target: tuple[int, int, int], amount: float) -> tuple[int, int, int]:
    red, green, blue = _hex_to_rgb(color)
    amount = min(max(amount, 0.0), 1.0)
    return (
        int(red + (target[0] - red) * amount),
        int(green + (target[1] - green) * amount),
        int(blue + (target[2] - blue) * amount),
    )


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


def _truncate(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.ImageFont,
    max_width: int,
) -> str:
    if draw.textbbox((0, 0), text, font=font)[2] <= max_width:
        return text
    suffix = "..."
    low, high = 0, len(text)
    best = suffix
    while low <= high:
        middle = (low + high) // 2
        candidate = text[:middle].rstrip() + suffix
        if draw.textbbox((0, 0), candidate, font=font)[2] <= max_width:
            best = candidate
            low = middle + 1
        else:
            high = middle - 1
    return best


def _format_population(value: float) -> str:
    return f"{value / 1_000_000:,.2f}M"


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
    step = _nice_number(maximum * 1.08 / 6.0)
    cap = math.ceil(maximum * 1.08 / step) * step
    return max(step, cap), step


def load_snapshots(input_csv: Path) -> list[Snapshot]:
    grouped: dict[str, list[CountryState]] = defaultdict(list)
    summaries: dict[str, str] = {}
    with input_csv.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            ranking_date = row["ranking_date"].strip()
            grouped[ranking_date].append(
                CountryState(
                    country_name=DISPLAY_NAME_ALIASES.get(
                        row["country_name"].strip(),
                        row["country_name"].strip(),
                    ),
                    country_code=row["country_code"].strip().upper(),
                    country_iso3=row["country_iso3"].strip().upper(),
                    population=float(row["population"]),
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
                states=sorted(grouped[ranking_date], key=lambda state: (-state.population, state.country_name)),
            )
        )
    return snapshots


def _interpolate(prev: Snapshot, nxt: Snapshot, alpha: float) -> list[CountryState]:
    previous = {state.country_iso3: state for state in prev.states}
    target = {state.country_iso3: state for state in nxt.states}
    states: list[CountryState] = []
    for iso3 in sorted(set(previous) | set(target)):
        before = previous.get(iso3) or target[iso3]
        after = target.get(iso3) or previous[iso3]
        states.append(
            CountryState(
                country_name=after.country_name or before.country_name,
                country_code=after.country_code or before.country_code,
                country_iso3=iso3,
                population=before.population + (after.population - before.population) * alpha,
            )
        )
    return states


def _rank(
    states: list[CountryState],
    top_n: int,
    priority: dict[str, int] | None = None,
) -> dict[str, int]:
    priority = priority or {}
    ranked = sorted(
        (state for state in states if state.population > 0),
        key=lambda state: (-state.population, priority.get(state.country_iso3, 10_000), state.country_name),
    )
    return {state.country_iso3: index for index, state in enumerate(ranked[:top_n])}


def _build_priorities(snapshots: list[Snapshot]) -> list[dict[str, int]]:
    priorities: list[dict[str, int]] = []
    previous: dict[str, int] = {}
    for snapshot in snapshots:
        ranked = sorted(
            snapshot.states,
            key=lambda state: (-state.population, previous.get(state.country_iso3, 10_000), state.country_name),
        )
        current = {state.country_iso3: index for index, state in enumerate(ranked)}
        priorities.append(current)
        previous = current
    return priorities


def _build_flag_cache(states: list[CountryState], flags_dir: Path) -> dict[str, Image.Image]:
    cache: dict[str, Image.Image] = {}
    for alpha2 in sorted({state.country_code.lower() for state in states if state.country_code}):
        path = flags_dir / f"{alpha2}.png"
        if not path.exists():
            continue
        try:
            image = Image.open(path).convert("RGBA")
            cache[alpha2.upper()] = ImageOps.fit(image, (46, 30), method=Image.Resampling.LANCZOS)
        except Exception:
            continue
    return cache


def _build_color_map(snapshots: list[Snapshot]) -> dict[str, str]:
    colors: dict[str, str] = {}
    fallback_index = 0
    for snapshot in snapshots:
        for state in snapshot.states:
            if state.country_iso3 in colors:
                continue
            color = COUNTRY_COLORS.get(state.country_iso3)
            if color is None:
                color = FALLBACK_COLORS[fallback_index % len(FALLBACK_COLORS)]
                fallback_index += 1
            colors[state.country_iso3] = color
    return colors


def _make_background() -> Image.Image:
    x_values = np.linspace(0, 1, WIDTH, dtype=np.float32)
    y_values = np.linspace(0, 1, HEIGHT, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(x_values, y_values)
    deep = np.array([4, 17, 31], dtype=np.float32)
    ocean = np.array([8, 55, 75], dtype=np.float32)
    teal = np.array([30, 132, 132], dtype=np.float32)
    sand = np.array([244, 194, 107], dtype=np.float32)

    mix = np.clip(0.35 * grid_x + 0.72 * grid_y, 0, 1)
    globe_glow = np.exp(-(((grid_x - 0.86) / 0.24) ** 2 + ((grid_y - 0.13) / 0.18) ** 2))
    lower_glow = np.exp(-(((grid_x - 0.08) / 0.28) ** 2 + ((grid_y - 0.94) / 0.20) ** 2))
    pixels = np.clip(
        deep[None, None, :] * (1.0 - mix[..., None])
        + ocean[None, None, :] * (0.82 * mix[..., None])
        + teal[None, None, :] * (0.22 * globe_glow[..., None])
        + sand[None, None, :] * (0.06 * lower_glow[..., None]),
        0,
        255,
    ).astype(np.uint8)
    frame = Image.fromarray(pixels, mode="RGB").convert("RGBA")

    overlay = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")
    draw.rounded_rectangle((28, 24, WIDTH - 28, HEIGHT - 28), radius=42, outline=(255, 255, 255, 20), width=2)
    draw.ellipse((1450, -145, 2020, 425), outline=(102, 224, 210, 26), width=3)
    draw.ellipse((1518, -77, 1952, 357), outline=(102, 224, 210, 16), width=2)
    for offset in (-120, -55, 25, 105):
        draw.arc((1450 + offset, -145, 2020 - offset, 425), 80, 280, fill=(102, 224, 210, 14), width=2)
    draw.line((54, 1040, 820, 930), fill=(244, 194, 107, 12), width=2)
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
    axis_scales = []
    for snapshot in snapshots:
        maximum = max((state.population for state in snapshot.states[:top_n]), default=1.0)
        if NAME_IN_BAR:
            axis_scales.append((maximum * 1.04, _nice_number(maximum / 5.0)))
        else:
            axis_scales.append(_axis_scale(maximum))
    axis_scales[0] = axis_scales[1]

    background = _make_background()
    title_font = _load_font(54, bold=True)
    subtitle_font = _load_font(23, bold=True)
    insight_fonts: dict[str, ImageFont.ImageFont] = {}
    year_font = _load_font(76, bold=True)
    label_font = _load_font(18, bold=True)
    name_font = _load_font(27, bold=True)
    value_font = _load_font(27, bold=True)
    rank_font = _load_font(24, bold=True)
    tick_font = _load_font(18, bold=True)
    measure_canvas = Image.new("RGB", (1, 1))
    measure_draw = ImageDraw.Draw(measure_canvas)
    name_font_cache: dict[str, ImageFont.ImageFont] = {}

    header_box = (38, 34, WIDTH - 38, 170)
    insight_box = (830, 54, 1448, 150)
    year_box = (1518, 50, 1848, 154)
    if NAME_IN_BAR:
        year_box = (1490, 828, 1820, 932)

    rank_left = 48
    flag_left = 118
    name_left = 182
    bar_left = 120 if NAME_IN_BAR else 430
    bar_right = 1818 if NAME_IN_BAR else 1762
    bar_max_width = bar_right - bar_left
    name_max_width = 360 if NAME_IN_BAR else bar_left - name_left - 12
    base_y = 246
    pitch = 64
    row_height = 48
    ranking_bottom = base_y + (top_n - 1) * pitch + row_height

    def _name_font_for(text: str) -> ImageFont.ImageFont:
        font = name_font_cache.get(text)
        if font is None:
            font = _fit_font_size(measure_draw, text, name_max_width, 27, 15, bold=True)
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
        if SNAP_TO_CURRENT_RANKS:
            current_ranked = sorted(
                (state for state in interpolated if state.population > 0),
                key=lambda state: (-state.population, priority.get(state.country_iso3, 10_000), state.country_name),
            )[:top_n]
            visible_iso3 = [state.country_iso3 for state in current_ranked]
            current_rank = {state.country_iso3: index for index, state in enumerate(current_ranked)}
        else:
            visible_iso3 = sorted(set(previous_rank) | set(target_rank))
            current_rank = {}

        draw.rounded_rectangle(header_box, radius=34, fill=(3, 16, 30, 218), outline=(255, 255, 255, 24), width=2)
        draw.text((68, 55), TITLE, font=title_font, fill="#f5f8fb")
        draw.text((70, 119), SUBTITLE, font=subtitle_font, fill="#83d9d0")

        if SHOW_INSIGHT_BOX:
            draw.rounded_rectangle(insight_box, radius=26, fill=(9, 43, 58, 230), outline=(102, 224, 210, 54), width=2)
            insight_lines = [part.strip() for part in nxt.season_summary.split("|") if part.strip()][:2]
            for line_index, line in enumerate(insight_lines):
                font = insight_fonts.get(line)
                if font is None:
                    font = _fit_font_size(draw, line, insight_box[2] - insight_box[0] - 40, 24, 17, bold=True)
                    insight_fonts[line] = font
                line_rect = (
                    insight_box[0] + 18,
                    insight_box[1] + 8 + line_index * 41,
                    insight_box[2] - 18,
                    insight_box[1] + 48 + line_index * 41,
                )
                _center_text(draw, line_rect, line, font, "#f4d39a" if line_index == 0 else "#eaf7f5")

        if LEFT_HEADER_LABEL and not NAME_IN_BAR:
            draw.text((name_left, 195), LEFT_HEADER_LABEL, font=label_font, fill=(177, 210, 219, 205))
        draw.text((bar_left + 18, 195), RIGHT_HEADER_LABEL, font=label_font, fill=(196, 225, 230, 220))

        tick_count = max(1, int(math.floor(axis_cap / max(tick_step, 1.0))))
        for tick_index in range(tick_count + 1):
            value = min(axis_cap, tick_index * tick_step)
            x = bar_left + int((value / axis_cap) * bar_max_width)
            draw.line((x, 224, x, ranking_bottom + 12), fill=(1, 9, 18, 82), width=2 if tick_index else 3)
            tick_text = _format_population(value) if tick_index else "0"
            if not (NAME_IN_BAR and tick_index == 0):
                bbox = draw.textbbox((0, 0), tick_text, font=tick_font)
                draw.text(
                    (x - (bbox[2] - bbox[0]) // 2, 194),
                    tick_text,
                    font=tick_font,
                    fill=(196, 225, 230, 190) if NAME_IN_BAR else (3, 15, 27, 165),
                )

        draw.rounded_rectangle(year_box, radius=30, fill=(244, 194, 107, 255), outline=(255, 235, 184, 180), width=2)
        _center_text(draw, year_box, YEAR_LABEL(nxt), year_font, "#10273a")

        for rank_index in range(top_n):
            y0 = base_y + rank_index * pitch
            fill = (244, 194, 107, 255) if rank_index == 0 else (20, 73, 87, 245)
            text_fill = "#10273a" if rank_index == 0 else "#edf8f7"
            draw.rounded_rectangle((rank_left, y0, rank_left + 52, y0 + row_height), radius=17, fill=fill)
            _center_text(draw, (rank_left, y0, rank_left + 52, y0 + row_height), str(rank_index + 1), rank_font, text_fill)

        render_items: list[tuple[int, float, CountryState, int]] = []
        for iso3 in visible_iso3:
            state = states_by_iso3.get(iso3)
            if state is None:
                continue
            previous_index = previous_rank.get(iso3, top_n + 1)
            target_index = target_rank.get(iso3, top_n + 1)
            if SNAP_TO_CURRENT_RANKS:
                y_index = float(current_rank[iso3])
                bar_width = max(8, int((state.population / axis_cap) * bar_max_width))
                render_items.append((0, base_y + y_index * pitch, state, bar_width))
                continue
            entering = previous_index > top_n and target_index <= top_n
            exiting = previous_index <= top_n and target_index > top_n
            entering_iso3 = sorted(
                (code for code, rank in target_rank.items() if rank <= top_n and previous_rank.get(code, top_n + 1) > top_n),
                key=lambda code: target_rank[code],
            )
            exiting_iso3 = sorted(
                (code for code, rank in previous_rank.items() if rank <= top_n and target_rank.get(code, top_n + 1) > top_n),
                key=lambda code: previous_rank[code],
            )
            entry_lane = entering_iso3.index(iso3) if entering and iso3 in entering_iso3 else 0
            exit_lane = exiting_iso3.index(iso3) if exiting and iso3 in exiting_iso3 else 0
            entry_index = float(top_n + 1.55 + entry_lane * 0.7)
            exit_index = float(top_n + 1.55 + exit_lane * 0.7)
            effective_previous = entry_index if entering else float(previous_index)
            effective_target = exit_index if exiting else float(target_index)
            movement_alpha = rank_alpha
            places_moved = abs(effective_target - effective_previous)
            if places_moved > 0:
                delay = min(0.08, max(0.0, (places_moved - 1.0) * 0.015))
                movement_alpha = _ease_in_out(
                    _phase_delay(movement_alpha, delay, max(0.82, 0.98 - delay))
                )
            if entering:
                movement_alpha = _ease_in_out(_phase_delay(movement_alpha, 0.46 + entry_lane * 0.035, 0.50))
                if movement_alpha <= 0.001:
                    continue
            if exiting:
                movement_alpha = _ease_in_out(_phase_delay(movement_alpha, exit_lane * 0.035, 0.46))
                if movement_alpha >= 0.999:
                    continue
            y_index = _continuous_rank_position(effective_previous, effective_target, movement_alpha)
            y = base_y + y_index * pitch
            bar_width = max(8, int((state.population / axis_cap) * bar_max_width))
            moving_up = 1 if target_index < previous_index else 0
            render_items.append((moving_up, y, state, bar_width))

        render_items.sort(key=lambda item: (item[0], item[1]))
        for _, y, state, bar_width in render_items:
            y0 = int(y)
            y1 = y0 + row_height
            if y1 < base_y - pitch or y0 > ranking_bottom + pitch:
                continue

            color = colors[state.country_iso3]
            highlight = _mix_rgb(color, (255, 255, 255), 0.28)
            shadow = _mix_rgb(color, (0, 0, 0), 0.24)
            value_min_x = bar_left + bar_width + 14

            if SHOW_ROW_BACKPLATE:
                draw.rounded_rectangle((108, y0 - 3, WIDTH - 64, y1 + 3), radius=19, fill=(3, 15, 27, 62))

            flag = flags.get(state.country_code)
            if flag is not None and not NAME_IN_BAR:
                flag_y = y0 + (row_height - flag.height) // 2
                draw.rounded_rectangle(
                    (flag_left - 4, flag_y - 4, flag_left + flag.width + 4, flag_y + flag.height + 4),
                    radius=7,
                    fill=(247, 250, 252, 238),
                )
                frame.alpha_composite(flag, (flag_left, flag_y))

            country_name = state.country_name
            if not NAME_IN_BAR:
                name_font_state = _name_font_for(country_name)
                name_bbox = draw.textbbox((0, 0), country_name, font=name_font_state)
                name_y = y0 + (row_height - (name_bbox[3] - name_bbox[1])) // 2 - name_bbox[1]
                draw.text((name_left, name_y), country_name, font=name_font_state, fill="#f1f7f8")

            draw.rounded_rectangle(
                (bar_left + 6, y0 + 6, bar_left + bar_width + 6, y1 + 6),
                radius=20,
                fill=(0, 0, 0, 76),
            )
            draw.rounded_rectangle(
                (bar_left, y0, bar_left + bar_width, y1),
                radius=20,
                fill=color,
                outline=highlight,
                width=2,
            )
            if bar_width > 42:
                draw.rounded_rectangle(
                    (bar_left + 9, y0 + 8, bar_left + max(30, int(bar_width * 0.64)), y0 + 17),
                    radius=6,
                    fill=(*highlight, 58),
                )
                draw.line(
                    (bar_left + 18, y1 - 8, bar_left + max(28, int(bar_width * 0.72)), y1 - 8),
                    fill=(*shadow, 92),
                    width=3,
                )

            if NAME_IN_BAR:
                name_x = bar_left + 14
                if flag is not None:
                    flag_target = (46, 30)
                    flag_img = flag if flag.size == flag_target else flag.resize(flag_target, Image.Resampling.LANCZOS)
                    flag_y = y0 + (row_height - flag_img.height) // 2
                    flag_x = bar_left + 10
                    draw.rounded_rectangle(
                        (flag_x - 3, flag_y - 3, flag_x + flag_img.width + 3, flag_y + flag_img.height + 3),
                        radius=7,
                        fill=(247, 250, 252, 238),
                    )
                    frame.alpha_composite(flag_img, (flag_x, flag_y))
                    name_x = flag_x + flag_img.width + 8
                available_name_width = min(360, bar_left + bar_width - name_x - 10)
                if available_name_width >= 12:
                    name_font_state = _load_font(24, bold=True)
                    name_text = _truncate(measure_draw, country_name, name_font_state, available_name_width)
                    name_bbox = draw.textbbox((0, 0), name_text, font=name_font_state)
                    name_y = y0 + (row_height - (name_bbox[3] - name_bbox[1])) // 2 - name_bbox[1]
                    draw.text(
                        (name_x, name_y),
                        name_text,
                        font=name_font_state,
                        fill="#ffffff",
                        stroke_width=2,
                        stroke_fill=(0, 0, 0, 145),
                    )

            value_text = _format_population(state.population)
            value_bbox = draw.textbbox((0, 0), value_text, font=value_font)
            value_x = min(value_min_x, WIDTH - 38 - (value_bbox[2] - value_bbox[0]))
            value_y = y0 + (row_height - (value_bbox[3] - value_bbox[1])) // 2 - value_bbox[1]
            draw.text((value_x, value_y), value_text, font=value_font, fill="#f7fbfc")

        footer = FOOTER
        footer_font = _fit_font_size(draw, footer, WIDTH - 120, 19, 15, bold=True)
        footer_bbox = draw.textbbox((0, 0), footer, font=footer_font)
        draw.text(
            ((WIDTH - (footer_bbox[2] - footer_bbox[0])) // 2, 1041),
            footer,
            font=footer_font,
            fill=(174, 207, 214, 175),
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
    parser = argparse.ArgumentParser(description="Generate a landscape World Population bar chart race.")
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
    print(f"[video_generator] World Population race generated -> {output}")


if __name__ == "__main__":
    main()
