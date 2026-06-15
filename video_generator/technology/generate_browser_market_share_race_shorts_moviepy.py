from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
from moviepy import VideoClip
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps

from video_generator.generate_ucl_barchart_race_moviepy import (
    DEFAULT_AUDIO,
    _fit_font_size,
    _load_font,
    build_audio_track,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_INPUT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "technology"
    / "browser_market_share"
    / "browser_market_share_1995_2026.csv"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "technology"
    / "browser_market_share"
    / "browser_market_share_race_1995_2026_shorts.mp4"
)
DEFAULT_LOGOS_DIR = (
    PROJECT_ROOT / "data" / "raw" / "technology" / "browser_market_share" / "logos"
)

WIDTH = 1080
HEIGHT = 1920
TOP_N = 8
FPS = 60
TOTAL_DURATION = 60.0
FINAL_HOLD_DURATION = 5.0
AXIS_CAP = 100.0

TITLE = "BROWSER WARS"
SUBTITLE = "MARKET SHARE HISTORY | 1995-2026"
FOOTER = "BROWSER MARKET SHARE | 1995-2026"


def ERA_LABEL(ranking_date: str) -> str:
    if ranking_date < "2001-01-01":
        return "Netscape vs Internet Explorer"
    if ranking_date < "2009-01-01":
        return "Internet Explorer era"
    if ranking_date < "2015-01-01":
        return "Chrome enters the race"
    return "The modern browser era"

BROWSER_COLORS = {
    "chrome": "#4F8CFF",
    "internet_explorer": "#23B7E5",
    "firefox": "#FF7139",
    "safari": "#32B7F0",
    "opera": "#FF365F",
    "android": "#47D17C",
    "uc_browser": "#FF8A3D",
    "samsung_internet": "#7A72FF",
    "edge": "#18C4B8",
    "edge_legacy": "#168CE7",
    "nokia": "#4B71D9",
    "blackberry": "#9BA9B7",
    "brave": "#FB542B",
    "yandex_browser": "#FF4A4A",
    "qq_browser": "#45B8FF",
    "maxthon": "#3C88E8",
    "aol": "#B8C3CF",
    "mozilla": "#E76F51",
    "netscape": "#49B7A5",
    "mosaic": "#C09A57",
    "other": "#708090",
}

FALLBACK_COLORS = [
    "#5CE1E6",
    "#FFB84D",
    "#A98BFF",
    "#FF6B8A",
    "#66D19E",
    "#6CA7FF",
    "#E4CB58",
    "#E481D8",
]


@dataclass(frozen=True)
class BrowserState:
    browser_name: str
    browser_key: str
    market_share: float


@dataclass(frozen=True)
class Snapshot:
    ranking_date: str
    season_summary: str
    states: list[BrowserState]


def _ease_in_out(value: float) -> float:
    value = min(max(value, 0.0), 1.0)
    return value * value * (3.0 - 2.0 * value)


def _phase_delay(value: float, delay: float, span: float) -> float:
    if span <= 0:
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


def load_snapshots(input_csv: Path) -> list[Snapshot]:
    grouped: dict[str, list[BrowserState]] = defaultdict(list)
    summaries: dict[str, str] = {}
    with input_csv.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            ranking_date = row["ranking_date"].strip()
            grouped[ranking_date].append(
                BrowserState(
                    browser_name=row["browser_name"].strip(),
                    browser_key=row["browser_key"].strip(),
                    market_share=float(row["market_share"]),
                )
            )
            summaries[ranking_date] = row.get("season_summary", "").strip()

    snapshots: list[Snapshot] = []
    for ranking_date in sorted(grouped):
        snapshots.append(
            Snapshot(
                ranking_date=ranking_date,
                season_summary=summaries.get(ranking_date, ""),
                states=sorted(
                    grouped[ranking_date],
                    key=lambda state: (-state.market_share, state.browser_name),
                ),
            )
        )
    return snapshots


def _interpolate(prev: Snapshot, nxt: Snapshot, alpha: float) -> list[BrowserState]:
    previous = {state.browser_key: state for state in prev.states}
    target = {state.browser_key: state for state in nxt.states}
    states: list[BrowserState] = []
    for key in sorted(set(previous) | set(target)):
        before = previous.get(key)
        after = target.get(key)
        meta = after or before
        if meta is None:
            continue
        before_value = before.market_share if before else 0.0
        after_value = after.market_share if after else 0.0
        states.append(
            BrowserState(
                browser_name=meta.browser_name,
                browser_key=key,
                market_share=before_value + (after_value - before_value) * alpha,
            )
        )
    return states


def _rank(
    states: list[BrowserState],
    top_n: int,
    priority: dict[str, int] | None = None,
) -> dict[str, int]:
    priority = priority or {}
    ranked = sorted(
        (state for state in states if state.market_share > 0),
        key=lambda state: (
            -state.market_share,
            priority.get(state.browser_key, 10_000),
            state.browser_name,
        ),
    )
    return {state.browser_key: index for index, state in enumerate(ranked[:top_n])}


def _build_priorities(snapshots: list[Snapshot]) -> list[dict[str, int]]:
    priorities: list[dict[str, int]] = []
    previous: dict[str, int] = {}
    for snapshot in snapshots:
        ranked = sorted(
            snapshot.states,
            key=lambda state: (
                -state.market_share,
                previous.get(state.browser_key, 10_000),
                state.browser_name,
            ),
        )
        current = {state.browser_key: index for index, state in enumerate(ranked)}
        priorities.append(current)
        previous = current
    return priorities


def _build_color_map(snapshots: list[Snapshot]) -> dict[str, str]:
    colors: dict[str, str] = {}
    fallback_index = 0
    for snapshot in snapshots:
        for state in snapshot.states:
            if state.browser_key in colors:
                continue
            color = BROWSER_COLORS.get(state.browser_key)
            if color is None:
                color = FALLBACK_COLORS[fallback_index % len(FALLBACK_COLORS)]
                fallback_index += 1
            colors[state.browser_key] = color
    return colors


def _build_logo_cache(logos_dir: Path) -> dict[str, Image.Image]:
    cache: dict[str, Image.Image] = {}
    for path in logos_dir.glob("*.png"):
        try:
            image = Image.open(path).convert("RGBA")
            cache[path.stem.lower()] = ImageOps.contain(
                image,
                (58, 58),
                method=Image.Resampling.LANCZOS,
            )
        except Exception:
            continue
    return cache


def _make_background() -> Image.Image:
    x_values = np.linspace(0, 1, WIDTH, dtype=np.float32)
    y_values = np.linspace(0, 1, HEIGHT, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(x_values, y_values)
    deep = np.array([4, 8, 18], dtype=np.float32)
    blue = np.array([8, 38, 67], dtype=np.float32)
    cyan = np.array([0, 202, 214], dtype=np.float32)
    orange = np.array([255, 143, 66], dtype=np.float32)

    mix = np.clip(0.3 * grid_x + 0.72 * grid_y, 0, 1)
    cyan_glow = np.exp(-(((grid_x - 0.88) / 0.34) ** 2 + ((grid_y - 0.08) / 0.16) ** 2))
    orange_glow = np.exp(-(((grid_x - 0.08) / 0.34) ** 2 + ((grid_y - 0.90) / 0.20) ** 2))
    pixels = np.clip(
        deep[None, None, :] * (1.0 - mix[..., None])
        + blue[None, None, :] * (0.92 * mix[..., None])
        + cyan[None, None, :] * (0.13 * cyan_glow[..., None])
        + orange[None, None, :] * (0.06 * orange_glow[..., None]),
        0,
        255,
    ).astype(np.uint8)
    frame = Image.fromarray(pixels, mode="RGB").convert("RGBA")

    overlay = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")
    for x in range(30, WIDTH, 90):
        draw.line((x, 430, x, HEIGHT - 70), fill=(52, 211, 222, 9), width=1)
    for y in range(460, HEIGHT - 60, 90):
        draw.line((30, y, WIDTH - 30, y), fill=(52, 211, 222, 8), width=1)
    for x, y in ((85, 1780), (225, 1700), (820, 155), (955, 260), (885, 1760)):
        draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=(93, 230, 237, 80))
        draw.ellipse((x - 24, y - 24, x + 24, y + 24), outline=(93, 230, 237, 24), width=2)
    draw.rounded_rectangle((22, 22, WIDTH - 22, HEIGHT - 22), radius=42, outline=(255, 255, 255, 18), width=2)
    overlay = overlay.filter(ImageFilter.GaussianBlur(radius=1.4))
    frame.alpha_composite(overlay)
    return frame


def _draw_logo(
    frame: Image.Image,
    draw: ImageDraw.ImageDraw,
    logo: Image.Image | None,
    state: BrowserState,
    x: int,
    y: int,
    color: str,
    monogram_font: ImageFont.ImageFont,
) -> None:
    box = (x, y, x + 66, y + 66)
    draw.rounded_rectangle(
        box,
        radius=17,
        fill=(241, 247, 251, 245),
        outline=(*_mix_rgb(color, (255, 255, 255), 0.38), 210),
        width=2,
    )
    if logo is not None:
        logo_x = x + (66 - logo.width) // 2
        logo_y = y + (66 - logo.height) // 2
        frame.alpha_composite(logo, (logo_x, logo_y))
        return
    monograms = {
        "internet_explorer": "e",
        "ie_mobile": "e",
        "netscape": "N",
        "mosaic": "M",
        "other": "+",
    }
    initials = monograms.get(
        state.browser_key,
        "".join(part[0] for part in state.browser_name.split()[:2]).upper(),
    )
    _center_text(draw, box, initials or "?", monogram_font, color)


def render_video(
    input_csv: Path,
    output_path: Path,
    logos_dir: Path,
    audio_path: Path,
    duration: float,
    final_hold_duration: float,
    fps: int,
    top_n: int,
    start_date: str | None = None,
    excluded_keys: set[str] | None = None,
    axis_cap: float = AXIS_CAP,
) -> Path:
    snapshots = load_snapshots(input_csv)
    if start_date is not None:
        snapshots = [
            snapshot for snapshot in snapshots if snapshot.ranking_date >= start_date
        ]
    if excluded_keys:
        snapshots = [
            Snapshot(
                ranking_date=snapshot.ranking_date,
                season_summary=snapshot.season_summary,
                states=[
                    state
                    for state in snapshot.states
                    if state.browser_key not in excluded_keys
                ],
            )
            for snapshot in snapshots
        ]
    if len(snapshots) < 2:
        raise RuntimeError("Not enough browser market share snapshots to render.")

    first_date = datetime.strptime(snapshots[0].ranking_date, "%Y-%m-%d")
    snapshots = [
        Snapshot(
            ranking_date=(first_date - timedelta(days=1)).strftime("%Y-%m-%d"),
            season_summary=snapshots[0].season_summary,
            states=[],
        ),
        *snapshots,
    ]

    logos = _build_logo_cache(logos_dir)
    colors = _build_color_map(snapshots)
    priorities = _build_priorities(snapshots)
    periods = len(snapshots) - 1
    transition_duration = max(0.1, duration - max(0.0, final_hold_duration))
    seconds_per_period = transition_duration / periods
    axis_cap = max(1.0, axis_cap)

    background = _make_background()
    title_font = _load_font(73, bold=True)
    subtitle_font = _load_font(27, bold=True)
    date_font = _load_font(58, bold=True)
    insight_font_cache: dict[str, ImageFont.ImageFont] = {}
    name_font = _load_font(33, bold=True)
    value_font = _load_font(31, bold=True)
    rank_font = _load_font(29, bold=True)
    tick_font = _load_font(20, bold=True)
    footer_font = _load_font(20, bold=True)
    monogram_font = _load_font(24, bold=True)

    header_box = (34, 34, WIDTH - 34, 430)
    date_box = (290, 180, 790, 304)
    insight_box = (64, 330, WIDTH - 64, 407)
    rank_left = 22
    logo_left = 102
    name_left = 184
    bar_left = 102
    bar_right = 690
    bar_max_width = bar_right - bar_left
    base_y = 488 if top_n > 8 else 512
    pitch = 132 if top_n > 8 else 151
    label_height = 66 if top_n > 8 else 72
    bar_height = 48 if top_n > 8 else 54
    ranking_bottom = base_y + (top_n - 1) * pitch + label_height + bar_height

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
        states_by_key = {state.browser_key: state for state in interpolated}
        priority = priorities[period_index]
        previous_rank = _rank(prev.states, top_n, priority)
        target_rank = _rank(nxt.states, top_n, priority)
        visible_keys = sorted(set(previous_rank) | set(target_rank))

        draw.rounded_rectangle(
            header_box,
            radius=38,
            fill=(3, 13, 27, 232),
            outline=(90, 226, 234, 45),
            width=2,
        )
        title_bbox = draw.textbbox((0, 0), TITLE, font=title_font)
        draw.text(
            ((WIDTH - (title_bbox[2] - title_bbox[0])) // 2, 53),
            TITLE,
            font=title_font,
            fill="#F5FAFF",
        )
        subtitle_bbox = draw.textbbox((0, 0), SUBTITLE, font=subtitle_font)
        draw.text(
            ((WIDTH - (subtitle_bbox[2] - subtitle_bbox[0])) // 2, 137),
            SUBTITLE,
            font=subtitle_font,
            fill="#62DDE6",
        )

        draw.rounded_rectangle(
            date_box,
            radius=30,
            fill=(255, 143, 66, 255),
            outline=(255, 212, 171, 220),
            width=2,
        )
        display_snapshot = prev if value_alpha < 0.5 and prev.states else nxt
        date_label = datetime.strptime(
            display_snapshot.ranking_date,
            "%Y-%m-%d",
        ).strftime("%b %Y").upper()
        _center_text(draw, date_box, date_label, date_font, "#102033")

        draw.rounded_rectangle(
            insight_box,
            radius=23,
            fill=(8, 38, 59, 238),
            outline=(90, 226, 234, 70),
            width=2,
        )
        visible_leader = max(
            interpolated,
            key=lambda state: state.market_share,
            default=BrowserState("N/A", "n_a", 0.0),
        )
        era_label = ERA_LABEL(display_snapshot.ranking_date)
        insight = (
            f"Leader: {visible_leader.browser_name} {visible_leader.market_share:.1f}%"
            f"  |  {era_label}"
        )
        insight_font = insight_font_cache.get(insight)
        if insight_font is None:
            insight_font = _fit_font_size(
                draw,
                insight,
                insight_box[2] - insight_box[0] - 34,
                24,
                16,
                bold=True,
            )
            insight_font_cache[insight] = insight_font
        _center_text(
            draw,
            (insight_box[0] + 16, insight_box[1], insight_box[2] - 16, insight_box[3]),
            insight,
            insight_font,
            "#F2FAFC",
        )

        tick_step = 5.0 if axis_cap <= 50.0 else 20.0
        tick_values: list[float] = []
        tick = 0.0
        while tick < axis_cap - 1e-6:
            tick_values.append(tick)
            tick += tick_step
        if not tick_values or not math.isclose(tick_values[-1], axis_cap, abs_tol=1e-6):
            tick_values.append(axis_cap)
        for tick in tick_values:
            x = bar_left + int((tick / axis_cap) * bar_max_width)
            draw.line(
                (x, base_y - 40, x, ranking_bottom + 14),
                fill=(1, 8, 17, 106),
                width=3 if tick == 0 else 2,
            )
            if tick:
                tick_text = f"{tick:.0f}%"
                bbox = draw.textbbox((0, 0), tick_text, font=tick_font)
                draw.text(
                    (x - (bbox[2] - bbox[0]) // 2, base_y - 39),
                    tick_text,
                    font=tick_font,
                    fill=(6, 21, 34, 170),
                )

        for rank_index in range(top_n):
            row_y = base_y + rank_index * pitch
            fill = (255, 143, 66, 255) if rank_index == 0 else (15, 73, 92, 245)
            text_fill = "#102033" if rank_index == 0 else "#EDF9FB"
            rank_box = (rank_left, row_y + 8, rank_left + 67, row_y + 76)
            draw.rounded_rectangle(rank_box, radius=21, fill=fill)
            _center_text(draw, rank_box, str(rank_index + 1), rank_font, text_fill)

        render_items: list[tuple[int, float, BrowserState, int]] = []
        for key in visible_keys:
            state = states_by_key.get(key)
            if state is None:
                continue
            previous_index = previous_rank.get(key, top_n + 1)
            target_index = target_rank.get(key, top_n + 1)
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
            bar_width = max(0, int((state.market_share / axis_cap) * bar_max_width))
            moving_up = 1 if target_index < previous_index else 0
            render_items.append((moving_up, row_y, state, bar_width))

        render_items.sort(key=lambda item: (item[0], item[1]))
        for _, row_y, state, bar_width in render_items:
            label_y = int(row_y)
            bar_y = label_y + label_height
            if bar_y + bar_height < base_y - pitch or label_y > ranking_bottom + pitch:
                continue

            color = colors[state.browser_key]
            highlight = _mix_rgb(color, (255, 255, 255), 0.30)
            shadow = _mix_rgb(color, (0, 0, 0), 0.25)
            logo = logos.get(state.browser_key)
            _draw_logo(
                frame,
                draw,
                logo,
                state,
                logo_left,
                label_y + 2,
                color,
                monogram_font,
            )

            value_text = f"{state.market_share:.1f}%"
            value_bbox = draw.textbbox((0, 0), value_text, font=value_font)
            value_x = WIDTH - 28 - (value_bbox[2] - value_bbox[0])
            value_y = label_y + (label_height - (value_bbox[3] - value_bbox[1])) // 2 - value_bbox[1]
            draw.text((value_x, value_y), value_text, font=value_font, fill="#F7FBFC")

            max_name_width = value_x - name_left - 18
            browser_name = _truncate(draw, state.browser_name, name_font, max_name_width)
            name_bbox = draw.textbbox((0, 0), browser_name, font=name_font)
            name_y = label_y + (label_height - (name_bbox[3] - name_bbox[1])) // 2 - name_bbox[1]
            draw.text((name_left, name_y), browser_name, font=name_font, fill="#F3F8FA")

            if bar_width > 0:
                draw.rounded_rectangle(
                    (bar_left + 6, bar_y + 7, bar_left + bar_width + 6, bar_y + bar_height + 7),
                    radius=20,
                    fill=(0, 0, 0, 92),
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
                        (bar_left + 9, bar_y + 8, bar_left + max(30, int(bar_width * 0.64)), bar_y + 18),
                        radius=6,
                        fill=(*highlight, 66),
                    )
                    draw.line(
                        (
                            bar_left + 18,
                            bar_y + bar_height - 8,
                            bar_left + max(30, int(bar_width * 0.72)),
                            bar_y + bar_height - 8,
                        ),
                        fill=(*shadow, 105),
                        width=3,
                    )

        footer = FOOTER
        footer_bbox = draw.textbbox((0, 0), footer, font=footer_font)
        draw.text(
            ((WIDTH - (footer_bbox[2] - footer_bbox[0])) // 2, 1850),
            footer,
            font=footer_font,
            fill=(172, 211, 220, 190),
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
        description="Generate a vertical worldwide browser market share bar chart race Short."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--logos-dir", type=Path, default=DEFAULT_LOGOS_DIR)
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
        logos_dir=args.logos_dir,
        audio_path=args.audio,
        duration=args.duration,
        final_hold_duration=args.final_hold,
        fps=args.fps,
        top_n=args.top_n,
    )
    print(f"[video_generator] Browser market share Short generated -> {output}")


if __name__ == "__main__":
    main()
