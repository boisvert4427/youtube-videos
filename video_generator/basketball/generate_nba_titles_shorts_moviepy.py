from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from moviepy import VideoClip
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from video_generator.generate_ucl_barchart_race_moviepy import (
    DEFAULT_AUDIO,
    FINAL_AUDIO_FADE_OUT,
    FPS as DEFAULT_FPS,
    LOOP_CROSSFADE,
    _fit_font_size,
    _load_font,
    build_audio_track,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_INPUT = PROJECT_ROOT / "data" / "processed" / "basketball" / "nba_titles_timeseries_1947_2025.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "basketball" / "nba_titles_shorts_1947_2025_45s.mp4"

WIDTH = 1080
HEIGHT = 1920
TOP_N = 12
FPS = 30
TOTAL_DURATION = 45.0

TITLE = "NBA TITLES RACE"
SUBTITLE = "Franchise championships"

TEAM_COLORS = {
    "Atlanta Hawks": ("#c8102e", "#fdb927"),
    "Baltimore Bullets": ("#6c7a89", "#d9e2ec"),
    "Boston Celtics": ("#007a33", "#ba9653"),
    "Brooklyn Nets": ("#111111", "#f4f4f4"),
    "Chicago Bulls": ("#ce1141", "#111111"),
    "Cleveland Cavaliers": ("#6f263d", "#ffb81c"),
    "Dallas Mavericks": ("#00538c", "#b8c4ca"),
    "Denver Nuggets": ("#0e2240", "#fec524"),
    "Detroit Pistons": ("#1d42ba", "#c8102e"),
    "Golden State Warriors": ("#1d428a", "#ffc72c"),
    "Houston Rockets": ("#ce1141", "#c4ced4"),
    "Los Angeles Lakers": ("#552583", "#fdb927"),
    "Miami Heat": ("#98002e", "#f9a01b"),
    "Milwaukee Bucks": ("#00471b", "#eee1c6"),
    "New York Knicks": ("#006bb6", "#f58426"),
    "Oklahoma City Thunder": ("#007ac1", "#ef3b24"),
    "Orlando Magic": ("#0077c0", "#c4ced4"),
    "Philadelphia 76ers": ("#006bb6", "#ed174c"),
    "Phoenix Suns": ("#1d1160", "#e56020"),
    "Portland Trail Blazers": ("#e03a3e", "#111111"),
    "Sacramento Kings": ("#5a2d81", "#c4ced4"),
    "San Antonio Spurs": ("#111111", "#c4ced4"),
    "Toronto Raptors": ("#ce1141", "#111111"),
    "Utah Jazz": ("#002b5c", "#f9a01b"),
    "Washington Capitols": ("#193a6b", "#f4f4f4"),
    "Washington Wizards": ("#002b5c", "#e31837"),
}


@dataclass(frozen=True)
class TeamState:
    team_name: str
    team_abbr: str
    titles: float


@dataclass(frozen=True)
class Snapshot:
    ranking_date: str
    year: int
    season_summary: str
    states: list[TeamState]


def _hex_to_rgb(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    return tuple(int(value[i : i + 2], 16) for i in (0, 2, 4))


def _mix_rgb(color: str, target: tuple[int, int, int], amount: float) -> tuple[int, int, int]:
    r, g, b = _hex_to_rgb(color)
    amount = min(max(amount, 0.0), 1.0)
    return (
        int(r + (target[0] - r) * amount),
        int(g + (target[1] - g) * amount),
        int(b + (target[2] - b) * amount),
    )


def _text_on(color: str) -> str:
    r, g, b = _hex_to_rgb(color)
    luminance = (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255.0
    return "#10233f" if luminance > 0.67 else "#f4f7fb"


def _ease_in_out(value: float) -> float:
    value = min(max(value, 0.0), 1.0)
    return value * value * (3.0 - 2.0 * value)


def _smoothstep(value: float) -> float:
    value = min(max(value, 0.0), 1.0)
    return value * value * (3.0 - 2.0 * value)


def _continuous_rank_position(prev_idx: float, next_idx: float, alpha: float) -> float:
    alpha = min(max(alpha, 0.0), 1.0)
    if math.isclose(prev_idx, next_idx):
        return float(next_idx)
    total_distance = abs(next_idx - prev_idx)
    steps = max(1, int(math.ceil(total_distance)))
    direction = 1.0 if next_idx > prev_idx else -1.0
    gap = 1.0 / steps
    span = min(0.9, gap * 1.35)
    travelled = 0.0
    end_travel = 0.0
    for step in range(steps):
        start = step * gap
        segment_distance = min(1.0, max(0.0, total_distance - step))
        local = min(max((alpha - start) / span, 0.0), 1.0)
        end_local = min(max((1.0 - start) / span, 0.0), 1.0)
        travelled += _smoothstep(local) * segment_distance
        end_travel += _smoothstep(end_local) * segment_distance
    if end_travel > 1e-9:
        travelled *= total_distance / end_travel
    travelled = min(total_distance, travelled)
    return float(prev_idx + direction * travelled)


def _truncate_text_to_width(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int) -> str:
    if max_width <= 0:
        return ""
    if draw.textbbox((0, 0), text, font=font)[2] <= max_width:
        return text
    suffix = "..."
    lo, hi = 0, len(text)
    best = suffix
    while lo <= hi:
        mid = (lo + hi) // 2
        candidate = text[:mid].rstrip() + suffix
        if draw.textbbox((0, 0), candidate, font=font)[2] <= max_width:
            best = candidate
            lo = mid + 1
        else:
            hi = mid - 1
    return best


def load_snapshots(input_csv: Path) -> list[Snapshot]:
    grouped: dict[str, list[TeamState]] = {}
    summaries: dict[str, str] = {}
    with input_csv.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            ranking_date = row["ranking_date"].strip()
            grouped.setdefault(ranking_date, []).append(
                TeamState(
                    team_name=row["team_name"].strip(),
                    team_abbr=row["team_abbr"].strip(),
                    titles=float(row["titles"]),
                )
            )
            summaries[ranking_date] = row.get("season_summary", "").strip()

    snapshots: list[Snapshot] = []
    for ranking_date in sorted(grouped.keys()):
        snapshots.append(
            Snapshot(
                ranking_date=ranking_date,
                year=int(ranking_date[:4]),
                season_summary=summaries.get(ranking_date, ""),
                states=sorted(grouped[ranking_date], key=lambda item: (-item.titles, item.team_name)),
            )
        )
    return snapshots


def _make_background() -> Image.Image:
    xx = np.linspace(0, 1, WIDTH, dtype=np.float32)
    yy = np.linspace(0, 1, HEIGHT, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(xx, yy)
    navy = np.array([6, 14, 34], dtype=np.float32)
    arena = np.array([9, 44, 92], dtype=np.float32)
    glow = np.array([231, 72, 92], dtype=np.float32)
    sky = np.array([84, 192, 255], dtype=np.float32)

    mix = np.clip(0.70 * grid_y + 0.20 * grid_x, 0, 1)
    top_glow = np.exp(-(((grid_x - 0.50) / 0.34) ** 2 + ((grid_y - 0.10) / 0.12) ** 2))
    side_glow = np.exp(-(((grid_x - 0.10) / 0.16) ** 2 + ((grid_y - 0.48) / 0.28) ** 2))
    side_glow += np.exp(-(((grid_x - 0.90) / 0.16) ** 2 + ((grid_y - 0.48) / 0.28) ** 2))

    img = np.clip(
        navy[None, None, :] * (1.0 - mix[..., None])
        + arena[None, None, :] * (0.80 * mix[..., None])
        + sky[None, None, :] * (0.12 * top_glow[..., None])
        + glow[None, None, :] * (0.08 * side_glow[..., None]),
        0,
        255,
    ).astype(np.uint8)
    frame = Image.fromarray(img, mode="RGB").convert("RGBA")

    overlay = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")
    draw.rounded_rectangle((66, 98, WIDTH - 66, HEIGHT - 110), radius=52, outline=(255, 255, 255, 18), width=2)
    draw.line((160, 1470, WIDTH - 160, 1470), fill=(255, 255, 255, 14), width=3)
    draw.line((WIDTH // 2, 320, WIDTH // 2, HEIGHT - 240), fill=(255, 255, 255, 10), width=2)
    draw.ellipse((214, 226, WIDTH - 214, 760), outline=(255, 255, 255, 16), width=3)
    overlay = overlay.filter(ImageFilter.GaussianBlur(radius=2))
    frame.alpha_composite(overlay)
    return frame


def _draw_glow_text(
    frame: Image.Image,
    position: tuple[int, int],
    text: str,
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int],
    glow: tuple[int, int, int],
    anchor: str = "la",
) -> None:
    blur_layer = Image.new("RGBA", frame.size, (0, 0, 0, 0))
    blur_draw = ImageDraw.Draw(blur_layer, "RGBA")
    blur_draw.text(position, text, font=font, fill=(*glow, 120), anchor=anchor)
    blur_layer = blur_layer.filter(ImageFilter.GaussianBlur(radius=10))
    frame.alpha_composite(blur_layer)
    draw = ImageDraw.Draw(frame, "RGBA")
    draw.text(position, text, font=font, fill=(*fill, 255), anchor=anchor)


def _make_team_badge(size: int, team_name: str, team_abbr: str) -> Image.Image:
    badge = Image.new("RGBA", (size + 6, size + 6), (0, 0, 0, 0))
    primary, secondary = TEAM_COLORS.get(team_name, ("#39c0ff", "#f4f7fb"))
    primary_rgb = _hex_to_rgb(primary)
    secondary_rgb = _hex_to_rgb(secondary)
    layer = ImageDraw.Draw(badge, "RGBA")
    layer.ellipse((0, 0, size + 5, size + 5), fill=(255, 255, 255, 230))
    layer.ellipse((3, 3, size + 2, size + 2), fill=primary_rgb + (255,))
    layer.arc((9, 9, size - 3, size - 3), start=205, end=335, fill=secondary_rgb + (255,), width=6)
    inner_font = _fit_font_size(layer, team_abbr, size - 14, 24, 12, bold=True)
    bbox = layer.textbbox((0, 0), team_abbr, font=inner_font)
    text_fill = "#10233f" if _text_on(primary) == "#10233f" else "#f4f7fb"
    layer.text(
        (3 + size // 2 - (bbox[2] - bbox[0]) // 2, 3 + size // 2 - (bbox[3] - bbox[1]) // 2 - 1),
        team_abbr,
        font=inner_font,
        fill=text_fill,
    )
    return badge


def _draw_team_badge(frame: Image.Image, x: int, y: int, badge: Image.Image) -> None:
    frame.alpha_composite(badge, (x - 3, y - 3))


def _build_badge_cache(states: list[TeamState], badge_size: int) -> dict[str, Image.Image]:
    cache: dict[str, Image.Image] = {}
    for state in states:
        if state.team_name in cache:
            continue
        cache[state.team_name] = _make_team_badge(badge_size, state.team_name, state.team_abbr)
    return cache


def _rank_with_tie_priority(states: list[TeamState], top_n: int, priority_order: dict[str, int] | None = None) -> dict[str, int]:
    priority_order = priority_order or {}
    ranked = sorted(
        (state for state in states if state.titles > 0),
        key=lambda item: (-item.titles, priority_order.get(item.team_name, 10_000), item.team_name),
    )
    return {state.team_name: idx for idx, state in enumerate(ranked[:top_n])}


def _build_stable_snapshot_priorities(snapshots: list[Snapshot]) -> list[dict[str, int]]:
    priorities: list[dict[str, int]] = []
    prev_priority: dict[str, int] | None = None
    for snapshot in snapshots:
        ranked = sorted(
            (state for state in snapshot.states if state.titles > 0),
            key=lambda item: (-item.titles, (prev_priority or {}).get(item.team_name, 10_000), item.team_name),
        )
        current_priority = {state.team_name: idx for idx, state in enumerate(ranked)}
        priorities.append(current_priority)
        prev_priority = current_priority
    return priorities


def _interp_values(prev: Snapshot, nxt: Snapshot, alpha: float) -> list[TeamState]:
    prev_map = {state.team_name: state for state in prev.states}
    next_map = {state.team_name: state for state in nxt.states}
    names = sorted(set(prev_map) | set(next_map))
    states: list[TeamState] = []
    for name in names:
        a = prev_map.get(name) or next_map[name]
        b = next_map.get(name) or prev_map[name]
        titles = a.titles + (b.titles - a.titles) * alpha
        states.append(TeamState(team_name=name, team_abbr=b.team_abbr or a.team_abbr, titles=titles))
    return states


def render_video(
    input_csv: Path,
    output_path: Path,
    audio_path: Path,
    duration: float,
    fps: int,
    top_n: int,
) -> Path:
    snapshots = load_snapshots(input_csv)
    if len(snapshots) < 2:
        raise RuntimeError("Not enough NBA snapshots to render.")

    first_snapshot = snapshots[0]
    intro_snapshot = Snapshot(
        ranking_date=f"{first_snapshot.year - 1}-06-30",
        year=first_snapshot.year,
        season_summary="",
        states=[],
    )
    snapshots = [intro_snapshot, *snapshots]

    all_states = [state for snapshot in snapshots for state in snapshot.states]
    badge_cache = _build_badge_cache(all_states, 50)
    priorities = _build_stable_snapshot_priorities(snapshots)
    periods = len(snapshots) - 1
    seconds_per_period = duration / periods
    axis_caps = [
        float(max((state.titles for state in snapshot.states[:top_n]), default=1.0) + 1)
        for snapshot in snapshots
    ]

    background = _make_background()
    title_font = _load_font(64, bold=True)
    subtitle_font = _load_font(24, bold=False)
    year_font = _load_font(84, bold=True)
    summary_font_cache: dict[str, ImageFont.ImageFont] = {}
    name_font = _load_font(26, bold=True)
    value_font = _load_font(30, bold=True)
    rank_font = _load_font(26, bold=True)

    bar_left = 98
    bar_right = 1058
    bar_max_w = bar_right - bar_left
    base_y = 394
    rank_x = 16
    row_gap = -2
    row_h = 98
    bar_height = 92
    badge_size = 50

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
        interp_map = {state.team_name: state for state in interp}
        priority = priorities[period_index]
        prev_rank = _rank_with_tie_priority(prev.states, top_n, priority)
        next_rank = _rank_with_tie_priority(nxt.states, top_n, priority)
        visible_names = sorted(set(prev_rank) | set(next_rank))
        top_states = [interp_map[name] for name in visible_names if name in interp_map]

        header = (46, 44, WIDTH - 46, 370)
        draw.rounded_rectangle(header, radius=42, fill=(5, 16, 34, 214), outline=(255, 255, 255, 22), width=1)
        draw.text((78, 76), TITLE, font=title_font, fill="#f3f7fc")
        draw.text((80, 148), SUBTITLE, font=subtitle_font, fill="#b8d2ee")
        draw.rounded_rectangle((74, 198, 262, 292), radius=30, fill=(255, 204, 82, 255))
        year_bbox = draw.textbbox((0, 0), str(nxt.year), font=year_font)
        draw.text((168 - (year_bbox[2] - year_bbox[0]) // 2, 206), str(nxt.year), font=year_font, fill="#132847")

        summary = nxt.season_summary or ""
        summary_font = summary_font_cache.get(summary)
        if summary_font is None:
            summary_font = _fit_font_size(draw, summary or " ", 650, 30, 16, bold=True)
            summary_font_cache[summary] = summary_font
        summary_rect = (280, 172, WIDTH - 58, 348)
        draw.rounded_rectangle(summary_rect, radius=28, fill=(14, 42, 77, 224), outline=(255, 255, 255, 26), width=2)
        summary_line = _truncate_text_to_width(draw, summary, summary_font, summary_rect[2] - summary_rect[0] - 36)
        summary_bbox = draw.textbbox((0, 0), summary_line, font=summary_font)
        draw.text(
            (
                summary_rect[0] + (summary_rect[2] - summary_rect[0] - (summary_bbox[2] - summary_bbox[0])) // 2,
                summary_rect[1] + (summary_rect[3] - summary_rect[1] - (summary_bbox[3] - summary_bbox[1])) // 2 - 1,
            ),
            summary_line,
            font=summary_font,
            fill="#eef7ff",
        )

        for lane_index in range(top_n):
            lane_y = base_y + lane_index * (row_h + row_gap)
            rank_top = lane_y + max(1, (row_h - bar_height) // 2)
            rank_bottom = rank_top + bar_height
            draw.rounded_rectangle((rank_x, rank_top, rank_x + 54, rank_bottom), radius=18, fill=(255, 204, 82, 255))
            rank_text = str(lane_index + 1)
            rank_bbox = draw.textbbox((0, 0), rank_text, font=rank_font)
            draw.text(
                (rank_x + 27 - (rank_bbox[2] - rank_bbox[0]) // 2, rank_top + max(10, (bar_height - (rank_bbox[3] - rank_bbox[1])) // 2 - 1)),
                rank_text,
                font=rank_font,
                fill="#10233f",
            )

        items: list[tuple[int, float, TeamState, int]] = []
        for state in top_states:
            prev_idx = prev_rank.get(state.team_name, top_n + 1)
            next_idx = next_rank.get(state.team_name, top_n + 1)
            y_idx = _continuous_rank_position(float(prev_idx), float(next_idx), alpha)
            y = base_y + y_idx * (row_h + row_gap)
            bar_w = max(112, int((state.titles / axis_cap) * bar_max_w))
            moving_up = 1 if next_idx < prev_idx else 0
            items.append((moving_up, y, state, bar_w))
        items.sort(key=lambda item: (item[0], item[1]))

        for _, y, state, bar_w in items:
            y0 = int(y)
            primary, secondary = TEAM_COLORS.get(state.team_name, ("#39c0ff", "#f4f7fb"))
            text_color = _text_on(primary)
            bar_top = y0 + max(1, (row_h - bar_height) // 2)
            bar_bottom = bar_top + bar_height
            radius = max(18, bar_height // 2)

            shadow_rect = (bar_left + 8, bar_top + 8, bar_left + bar_w + 8, bar_bottom + 8)
            draw.rounded_rectangle(shadow_rect, radius=radius, fill=(0, 0, 0, 78))
            outline_color = _mix_rgb(primary, (255, 255, 255), 0.18)
            highlight_color = _mix_rgb(primary, (255, 255, 255), 0.34)
            inner_shadow = _mix_rgb(primary, (0, 0, 0), 0.18)
            draw.rounded_rectangle((bar_left, bar_top, bar_left + bar_w, bar_bottom), radius=radius, fill=primary, outline=outline_color, width=2)
            draw.rounded_rectangle(
                (bar_left + 10, bar_top + 8, bar_left + max(90, int(bar_w * 0.62)), bar_top + 18),
                radius=8,
                fill=(*highlight_color, 56),
            )
            draw.line(
                (bar_left + 18, bar_bottom - 7, bar_left + max(32, int(bar_w * 0.72)), bar_bottom - 7),
                fill=(*inner_shadow, 88),
                width=3,
            )

            badge_x = bar_left + 8
            badge_y = bar_top + (bar_height - badge_size) // 2
            badge = badge_cache.get(state.team_name)
            if badge is not None:
                _draw_team_badge(frame, badge_x, badge_y, badge)

            label_x = bar_left + badge_size + 18
            label_max_width = max(0, bar_w - (label_x - bar_left) - 18)
            team_name = _truncate_text_to_width(draw, state.team_name, name_font, label_max_width)
            draw.text((label_x, bar_top + max(8, (bar_height - 31) // 2)), team_name, font=name_font, fill=text_color)

            value_text = str(int(round(state.titles)))
            value_bbox = draw.textbbox((0, 0), value_text, font=value_font)
            value_x = min(bar_left + bar_w + 18, WIDTH - 56 - (value_bbox[2] - value_bbox[0]))
            _draw_glow_text(frame, (value_x, bar_top + max(10, (bar_height - 30) // 2)), value_text, value_font, (244, 247, 251), _hex_to_rgb(secondary))

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
    parser = argparse.ArgumentParser(description="Generate an NBA franchise titles Shorts bar chart race.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
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
        audio_path=args.audio,
        duration=args.duration,
        fps=args.fps,
        top_n=args.top_n,
    )
    print(f"[video_generator] NBA Shorts race generated -> {output}")


if __name__ == "__main__":
    main()
