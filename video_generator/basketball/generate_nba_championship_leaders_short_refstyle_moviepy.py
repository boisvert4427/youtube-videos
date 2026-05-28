from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from moviepy import VideoClip
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps

from video_generator.generate_ucl_barchart_race_moviepy import _fit_font_size, _load_font, build_audio_track


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_INPUT = PROJECT_ROOT / "data" / "processed" / "basketball" / "nba_championships_timeseries_1947_2025.csv"
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "data" / "processed" / "basketball" / "nba_championship_leaders_short_refstyle_1947_2025_80s.mp4"
)
DEFAULT_AUDIO = PROJECT_ROOT / "data" / "raw" / "audio" / "Midnight_Grip_20260402_0828.mp3"
NBA_LOGO = PROJECT_ROOT / "data" / "raw" / "nba_logo.png"
TROPHY = PROJECT_ROOT / "data" / "raw" / "nba_trophy_photo_alt.png"
LOGO_DIR = PROJECT_ROOT / "data" / "raw" / "nba_team_logos"

WIDTH = 1080
HEIGHT = 1920
FPS = 30
TOTAL_DURATION = 80.0
BASE_INTRO_DURATION = 5.5
BASE_OUTRO_DURATION = 10.0
TOP_N = 10

TITLE = "NBA CHAMPIONSHIP LEADERS"
SUBTITLE = "Official NBA history | 1947-2025"

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


def hex_to_rgb(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    return tuple(int(value[i : i + 2], 16) for i in (0, 2, 4))


def mix_rgb(color: str, target: tuple[int, int, int], amount: float) -> tuple[int, int, int]:
    r, g, b = hex_to_rgb(color)
    amount = min(max(amount, 0.0), 1.0)
    return (
        int(r + (target[0] - r) * amount),
        int(g + (target[1] - g) * amount),
        int(b + (target[2] - b) * amount),
    )


def text_on(color: str) -> str:
    r, g, b = hex_to_rgb(color)
    luminance = (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255.0
    return "#10233f" if luminance > 0.67 else "#f4f7fb"


def ease(value: float) -> float:
    value = min(max(value, 0.0), 1.0)
    return value * value * (3.0 - 2.0 * value)


def smoothstep(value: float) -> float:
    value = min(max(value, 0.0), 1.0)
    return value * value * (3.0 - 2.0 * value)


def rank_position(prev_idx: float, next_idx: float, alpha: float) -> float:
    alpha = min(max(alpha, 0.0), 1.0)
    if math.isclose(prev_idx, next_idx):
        return float(next_idx)
    total_distance = abs(next_idx - prev_idx)
    steps = max(1, int(math.ceil(total_distance)))
    direction = 1.0 if next_idx > prev_idx else -1.0
    gap = 1.0 / steps
    span = min(0.9, gap * 1.25)
    moved = 0.0
    end_moved = 0.0
    for step in range(steps):
        start = step * gap
        segment_distance = min(1.0, max(0.0, total_distance - step))
        local = min(max((alpha - start) / span, 0.0), 1.0)
        end_local = min(max((1.0 - start) / span, 0.0), 1.0)
        moved += smoothstep(local) * segment_distance
        end_moved += smoothstep(end_local) * segment_distance
    if end_moved > 1e-9:
        moved *= total_distance / end_moved
    return float(prev_idx + direction * min(total_distance, moved))


def truncate_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int) -> str:
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


def wrap_years(years: list[int], per_line: int = 6) -> str:
    if not years:
        return ""
    return "\n".join(" \u00b7 ".join(str(year) for year in years[i : i + per_line]) for i in range(0, len(years), per_line))


def slug(team_name: str) -> str:
    value = team_name.lower().replace(".", "").replace("&", "and").replace("-", "_").replace(" ", "_")
    while "__" in value:
        value = value.replace("__", "_")
    return value


def logo_path(team_name: str) -> Path:
    return LOGO_DIR / f"{slug(team_name)}.png"


def segment_durations(total_duration: float) -> tuple[float, float, float]:
    if total_duration >= 30.0:
        intro = BASE_INTRO_DURATION
        outro = BASE_OUTRO_DURATION
    else:
        intro = max(2.0, min(BASE_INTRO_DURATION, total_duration * 0.22))
        outro = max(2.0, min(BASE_OUTRO_DURATION, total_duration * 0.20))
    race = total_duration - intro - outro
    if race < 1.0:
        intro = max(1.5, total_duration * 0.25)
        outro = max(1.5, total_duration * 0.25)
        race = max(1.0, total_duration - intro - outro)
    return intro, outro, race


def make_background() -> Image.Image:
    xx = np.linspace(0, 1, WIDTH, dtype=np.float32)
    yy = np.linspace(0, 1, HEIGHT, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(xx, yy)

    navy = np.array([6, 14, 34], dtype=np.float32)
    blue = np.array([11, 38, 72], dtype=np.float32)
    gold = np.array([239, 194, 84], dtype=np.float32)
    steel = np.array([18, 27, 45], dtype=np.float32)

    mix = np.clip(0.66 * grid_y + 0.14 * grid_x, 0, 1)
    top_glow = np.exp(-(((grid_x - 0.50) / 0.34) ** 2 + ((grid_y - 0.10) / 0.12) ** 2))
    side_glow = np.exp(-(((grid_x - 0.12) / 0.14) ** 2 + ((grid_y - 0.50) / 0.24) ** 2))
    side_glow += np.exp(-(((grid_x - 0.88) / 0.14) ** 2 + ((grid_y - 0.50) / 0.24) ** 2))
    lower_glow = np.exp(-(((grid_x - 0.50) / 0.40) ** 2 + ((grid_y - 0.78) / 0.13) ** 2))

    img = np.clip(
        navy[None, None, :] * (1.0 - mix[..., None])
        + blue[None, None, :] * (0.80 * mix[..., None])
        + steel[None, None, :] * (0.22 * lower_glow[..., None])
        + gold[None, None, :] * (0.08 * top_glow[..., None] + 0.06 * side_glow[..., None]),
        0,
        255,
    ).astype(np.uint8)
    frame = Image.fromarray(img, mode="RGB").convert("RGBA")

    overlay = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")
    draw.rounded_rectangle((48, 78, WIDTH - 48, HEIGHT - 84), radius=52, outline=(255, 255, 255, 16), width=2)
    draw.line((150, 1488, WIDTH - 150, 1488), fill=(255, 255, 255, 10), width=2)
    draw.line((WIDTH // 2, 320, WIDTH // 2, HEIGHT - 220), fill=(255, 255, 255, 8), width=2)
    draw.ellipse((196, 238, WIDTH - 196, 748), outline=(255, 255, 255, 12), width=3)
    draw.ellipse((282, 1110, WIDTH - 282, 1704), outline=(239, 194, 84, 10), width=2)
    overlay = overlay.filter(ImageFilter.GaussianBlur(radius=1.6))
    frame.alpha_composite(overlay)
    return frame


def load_snapshots(input_csv: Path) -> list[Snapshot]:
    grouped: dict[str, list[TeamState]] = defaultdict(list)
    summaries: dict[str, str] = {}
    with input_csv.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            date = row["ranking_date"].strip()
            grouped[date].append(
                TeamState(
                    team_name=row["team_name"].strip(),
                    team_abbr=row["team_abbr"].strip(),
                    titles=float(row["titles"]),
                )
            )
            summaries[date] = row.get("season_summary", "").strip()

    snapshots: list[Snapshot] = []
    for ranking_date in sorted(grouped):
        snapshots.append(
            Snapshot(
                ranking_date=ranking_date,
                year=int(ranking_date[:4]),
                season_summary=summaries.get(ranking_date, ""),
                states=sorted(grouped[ranking_date], key=lambda item: (-item.titles, item.team_name)),
            )
        )
    return snapshots


def title_years(snapshots: list[Snapshot]) -> dict[str, list[int]]:
    years: dict[str, list[int]] = defaultdict(list)
    previous: dict[str, int] = defaultdict(int)
    for snapshot in snapshots:
        current = {state.team_name: int(round(state.titles)) for state in snapshot.states}
        for team_name, count in current.items():
            if count > previous.get(team_name, 0):
                years[team_name].append(snapshot.year)
        previous = current
    return years


def build_snapshot_priorities(snapshots: list[Snapshot]) -> list[dict[str, int]]:
    priorities: list[dict[str, int]] = []
    previous_priority: dict[str, int] | None = None
    for snapshot in snapshots:
        ranked = sorted(
            (state for state in snapshot.states if state.titles > 0),
            key=lambda state: (-state.titles, (previous_priority or {}).get(state.team_name, 10_000), state.team_name),
        )
        current_priority = {state.team_name: idx for idx, state in enumerate(ranked)}
        priorities.append(current_priority)
        previous_priority = current_priority
    return priorities


def rank_with_priority(states: list[TeamState], top_n: int, priority: dict[str, int]) -> dict[str, int]:
    ranked = sorted(
        (state for state in states if state.titles > 0),
        key=lambda state: (-state.titles, priority.get(state.team_name, 10_000), state.team_name),
    )
    return {state.team_name: idx for idx, state in enumerate(ranked[:top_n])}


def interpolate_states(prev: Snapshot, nxt: Snapshot, alpha: float) -> list[TeamState]:
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


def build_badges(states: list[TeamState], size: int = 68) -> dict[str, Image.Image]:
    cache: dict[str, Image.Image] = {}
    for state in states:
        if state.team_name in cache:
            continue
        badge = Image.new("RGBA", (size + 10, size + 10), (0, 0, 0, 0))
        draw = ImageDraw.Draw(badge, "RGBA")
        primary, secondary = TEAM_COLORS.get(state.team_name, ("#39c0ff", "#f4f7fb"))
        outer = mix_rgb(primary, (0, 0, 0), 0.28)
        draw.rounded_rectangle((2, 2, size + 8, size + 8), radius=18, fill=(10, 20, 42, 238), outline=outer + (255,), width=2)
        draw.rounded_rectangle((6, 6, size + 4, size + 4), radius=14, fill=(255, 255, 255, 18), outline=mix_rgb(secondary, (255, 255, 255), 0.08) + (80,), width=1)
        path = logo_path(state.team_name)
        if path.exists():
            logo = Image.open(path).convert("RGBA")
            logo = ImageOps.contain(logo, (size - 12, size - 12), method=Image.Resampling.LANCZOS)
            badge.alpha_composite(logo, ((badge.width - logo.width) // 2, (badge.height - logo.height) // 2))
        else:
            font = _fit_font_size(draw, state.team_abbr, size - 14, 22, 12, bold=True)
            bbox = draw.textbbox((0, 0), state.team_abbr, font=font)
            fill = text_on(primary)
            draw.text(
                (
                    (badge.width - (bbox[2] - bbox[0])) // 2,
                    (badge.height - (bbox[3] - bbox[1])) // 2 - 2,
                ),
                state.team_abbr,
                font=font,
                fill=fill,
            )
        cache[state.team_name] = badge
    return cache


def fit_multiline_font(draw: ImageDraw.ImageDraw, text: str, max_width: int, start_size: int, min_size: int, bold: bool = False):
    lines = [line for line in text.splitlines() if line]
    longest = max(lines, key=len) if lines else text
    return _fit_font_size(draw, longest, max_width, start_size, min_size, bold=bold)


def draw_header(
    frame: Image.Image,
    draw: ImageDraw.ImageDraw,
    year: int,
    summary: str,
    title_font: ImageFont.ImageFont,
    subtitle_font: ImageFont.ImageFont,
    year_font: ImageFont.ImageFont,
    summary_font: ImageFont.ImageFont,
    trophy_img: Image.Image,
    nba_logo: Image.Image,
) -> None:
    draw.rounded_rectangle((48, 70, WIDTH - 48, 386), radius=44, fill=(7, 16, 34, 238), outline=(255, 255, 255, 18), width=2)
    draw.text((80, 118), TITLE, font=title_font, fill="#f4f7fb")
    draw.text((82, 188), SUBTITLE, font=subtitle_font, fill="#b8d2ee")
    draw.rounded_rectangle((80, 240, 202, 326), radius=26, fill=(255, 204, 82, 255))
    year_bbox = draw.textbbox((0, 0), str(year), font=year_font)
    draw.text((141 - (year_bbox[2] - year_bbox[0]) // 2, 250), str(year), font=year_font, fill="#12223d")

    summary_rect = (220, 236, WIDTH - 266, 326)
    draw.rounded_rectangle(summary_rect, radius=26, fill=(12, 34, 62, 228), outline=(255, 255, 255, 16), width=2)
    summary_line = truncate_text(draw, summary, summary_font, summary_rect[2] - summary_rect[0] - 36)
    draw.text((242, 258), summary_line, font=summary_font, fill="#eef7ff")

    trophy_panel = (WIDTH - 224, 104, WIDTH - 82, 322)
    draw.rounded_rectangle(trophy_panel, radius=28, fill=(255, 255, 255, 10), outline=(255, 255, 255, 18), width=2)
    trophy = ImageOps.contain(trophy_img.copy(), (120, 190), method=Image.Resampling.LANCZOS)
    trophy.putalpha(220)
    frame.alpha_composite(trophy, (WIDTH - 195, 112))
    draw.text((WIDTH - 150, 296), "NBA", font=_load_font(24, bold=True), fill="#f2d27a", anchor="mm")

    watermark = ImageOps.contain(nba_logo.copy(), (160, 320), method=Image.Resampling.LANCZOS)
    watermark.putalpha(28)
    frame.alpha_composite(watermark, (58, 610))
    draw.line((76, 368, WIDTH - 76, 368), fill=(255, 255, 255, 10), width=2)


def draw_team_card(
    frame: Image.Image,
    draw: ImageDraw.ImageDraw,
    rect: tuple[int, int, int, int],
    state: TeamState,
    rank_label: str,
    badge: Image.Image,
    *,
    emphasize: bool = False,
) -> None:
    x0, y0, x1, y1 = rect
    primary, secondary = TEAM_COLORS.get(state.team_name, ("#39c0ff", "#f4f7fb"))
    fill = mix_rgb(primary, (10, 18, 34), 0.70)
    outline = mix_rgb(primary, (255, 255, 255), 0.18)
    draw.rounded_rectangle(rect, radius=28, fill=(*fill, 236), outline=outline + (255,), width=2)
    if emphasize:
        draw.rounded_rectangle((x0 - 2, y0 - 2, x1 + 2, y1 + 2), radius=30, outline=(255, 204, 82, 120), width=4)
    frame.alpha_composite(badge, (x0 + 14, y0 + 16))
    draw.rounded_rectangle((x0 + 98, y0 + 18, x0 + 148, y0 + 60), radius=16, fill=(255, 204, 82, 255))
    rank_bbox = draw.textbbox((0, 0), rank_label, font=_load_font(24, bold=True))
    draw.text((123 - (rank_bbox[2] - rank_bbox[0]) // 2, y0 + 27), rank_label, font=_load_font(24, bold=True), fill="#12223d")
    team_font = _fit_font_size(draw, state.team_name, x1 - x0 - 174, 28, 20, bold=True)
    draw.text((x0 + 98, y0 + 66), state.team_name, font=team_font, fill="#f4f7fb")
    draw.text((x0 + 98, y0 + 102), f"{int(round(state.titles))} titles", font=_load_font(18, bold=True), fill="#d7e7f7")
    draw.text((x1 - 18, y0 + 32), str(int(round(state.titles))), font=_load_font(30, bold=True), fill="#f2d27a", anchor="rm")


def draw_years_panel(
    frame: Image.Image,
    draw: ImageDraw.ImageDraw,
    rect: tuple[int, int, int, int],
    leader: TeamState,
    years_text: str,
    badge: Image.Image,
    *,
    large: bool,
) -> None:
    x0, y0, x1, y1 = rect
    primary, secondary = TEAM_COLORS.get(leader.team_name, ("#39c0ff", "#f4f7fb"))
    draw.rounded_rectangle(rect, radius=36, fill=(8, 18, 36, 236), outline=(255, 255, 255, 18), width=2)
    draw.rounded_rectangle((x0 + 24, y0 + 24, x0 + 176, y0 + 72), radius=16, fill=(255, 204, 82, 255))
    draw.text((x0 + 100, y0 + 36), "TITLE YEARS", font=_load_font(22, bold=True), fill="#12223d", anchor="mm")
    frame.alpha_composite(badge, (x0 + 24, y0 + 92))
    draw.text((x0 + 112, y0 + 98), leader.team_name, font=_load_font(30, bold=True), fill="#f4f7fb")
    draw.text((x0 + 112, y0 + 138), f"{int(round(leader.titles))} championships", font=_load_font(18, bold=True), fill="#d7e7f7")
    draw.rounded_rectangle((x1 - 160, y0 + 88, x1 - 28, y0 + 182), radius=24, fill=(*hex_to_rgb(primary), 220), outline=mix_rgb(primary, (255, 255, 255), 0.18) + (255,), width=2)
    draw.text((x1 - 94, y0 + 130), str(int(round(leader.titles))), font=_load_font(40, bold=True), fill=text_on(primary), anchor="mm")
    years_font = fit_multiline_font(draw, years_text, x1 - x0 - 64, 24 if large else 20, 14, bold=False)
    draw.multiline_text((x0 + 26, y0 + 214), years_text, font=years_font, fill="#edf4ff", spacing=4)
    draw.text((x0 + 26, y1 - 36), "Official NBA history data from NBA.com", font=_load_font(16, bold=False), fill="#9eb7d9")


def draw_race_frame(
    frame: Image.Image,
    draw: ImageDraw.ImageDraw,
    snapshots: list[Snapshot],
    priorities: list[dict[str, int]],
    badges: dict[str, Image.Image],
    title_years_map: dict[str, list[int]],
    *,
    t: float,
    total_duration: float,
    intro_duration: float,
    outro_duration: float,
    title_font: ImageFont.ImageFont,
    subtitle_font: ImageFont.ImageFont,
    year_font: ImageFont.ImageFont,
    summary_font: ImageFont.ImageFont,
    row_font: ImageFont.ImageFont,
    sub_font: ImageFont.ImageFont,
    value_font: ImageFont.ImageFont,
    rank_font: ImageFont.ImageFont,
    trophy_img: Image.Image,
    nba_logo: Image.Image,
) -> None:
    race_t = t - intro_duration
    race_duration = max(1e-6, total_duration - intro_duration - outro_duration)
    periods = len(snapshots) - 1
    seconds_per_period = race_duration / periods
    period_index = min(int(race_t / seconds_per_period), periods - 1)
    local = (race_t - period_index * seconds_per_period) / seconds_per_period
    alpha = ease(local)

    prev = snapshots[period_index]
    nxt = snapshots[period_index + 1]
    current = interpolate_states(prev, nxt, alpha)
    state_map = {state.team_name: state for state in current}

    priority = priorities[period_index]
    prev_rank = rank_with_priority(prev.states, TOP_N, priority)
    next_rank = rank_with_priority(nxt.states, TOP_N, priority)
    visible_names = [name for name in sorted(set(prev_rank) | set(next_rank)) if name in state_map]

    draw_header(frame, draw, nxt.year, nxt.season_summary, title_font, subtitle_font, year_font, summary_font, trophy_img, nba_logo)

    draw.text((72, 420), "LIVE TITLE RACE", font=_load_font(22, bold=True), fill="#f2d27a")

    axis_cap = float(int(math.ceil(max(state.titles for snapshot in snapshots for state in snapshot.states))) + 1)
    row_h = 88
    row_gap = 10
    base_y = 456
    row_x0 = 54
    row_x1 = WIDTH - 54
    bar_left = 210
    bar_max = 660
    bar_height = 16

    for lane in range(TOP_N):
        y = base_y + lane * (row_h + row_gap)
        draw.rounded_rectangle((row_x0, y, row_x1, y + row_h), radius=28, fill=(9, 18, 34, 190), outline=(255, 255, 255, 10), width=1)
        draw.rounded_rectangle((66, y + 12, 122, y + 68), radius=18, fill=(255, 204, 82, 255))
        number = str(lane + 1)
        bbox = draw.textbbox((0, 0), number, font=rank_font)
        draw.text((94 - (bbox[2] - bbox[0]) // 2, y + 27), number, font=rank_font, fill="#12223d")

    items: list[tuple[int, float, TeamState, int]] = []
    for state in visible_names:
        prev_idx = prev_rank.get(state, TOP_N + 1)
        next_idx = next_rank.get(state, TOP_N + 1)
        y_idx = rank_position(float(prev_idx), float(next_idx), alpha)
        y = base_y + y_idx * (row_h + row_gap)
        bar_w = max(92, int((state_map[state].titles / axis_cap) * bar_max))
        items.append((1 if next_idx < prev_idx else 0, y, state_map[state], bar_w))
    items.sort(key=lambda item: (item[0], item[1]))

    ranked_current = sorted(
        current,
        key=lambda state: (-state.titles, priority.get(state.team_name, 10_000), state.team_name),
    )
    leader = ranked_current[0]

    for _, y, state, bar_w in items:
        y0 = int(y)
        primary, secondary = TEAM_COLORS.get(state.team_name, ("#39c0ff", "#f4f7fb"))
        frame.alpha_composite(badges[state.team_name], (136, y0 + 10))
        draw.text((214, y0 + 16), truncate_text(draw, state.team_name, row_font, 380), font=row_font, fill="#f4f7fb")
        draw.text((214, y0 + 48), f"{int(round(state.titles))} championships", font=sub_font, fill="#d7e7f7")
        bar_rect = (bar_left, y0 + 62, bar_left + bar_w, y0 + 62 + bar_height)
        draw.rounded_rectangle((bar_left + 6, y0 + 67, bar_left + bar_w + 6, y0 + 67 + bar_height), radius=bar_height // 2, fill=(0, 0, 0, 55))
        draw.rounded_rectangle(bar_rect, radius=bar_height // 2, fill=(*hex_to_rgb(primary), 240), outline=mix_rgb(primary, (255, 255, 255), 0.18) + (255,), width=2)
        draw.rounded_rectangle((WIDTH - 142, y0 + 20, WIDTH - 72, y0 + 68), radius=18, fill=(255, 204, 82, 255))
        draw.text((WIDTH - 107, y0 + 42), str(int(round(state.titles))), font=value_font, fill="#12223d", anchor="mm")
        if state.team_name == leader.team_name:
            draw.rounded_rectangle((row_x0 - 2, y0 - 2, row_x1 + 2, y0 + row_h + 2), radius=30, outline=(255, 204, 82, 120), width=4)
            draw.text((WIDTH - 172, y0 + 20), "★", font=_load_font(26, bold=True), fill="#f2d27a")

    ranked_current = sorted(
        current,
        key=lambda state: (-state.titles, priority.get(state.team_name, 10_000), state.team_name),
    )
    leader = ranked_current[0]
    leader_years = wrap_years(title_years_map[leader.team_name], 6)
    years_font = fit_multiline_font(draw, leader_years, WIDTH - 120, 22, 14, bold=False)
    draw.rounded_rectangle((58, 1508, WIDTH - 58, 1846), radius=36, fill=(8, 18, 36, 236), outline=(255, 255, 255, 16), width=2)
    draw.rounded_rectangle((78, 1530, 210, 1576), radius=16, fill=(255, 204, 82, 255))
    draw.text((144, 1552), "TITLE YEARS", font=_load_font(20, bold=True), fill="#12223d", anchor="mm")
    frame.alpha_composite(badges[leader.team_name], (78, 1588))
    draw.text((180, 1594), leader.team_name, font=_load_font(28, bold=True), fill="#f4f7fb")
    draw.text((180, 1636), f"{int(round(leader.titles))} titles", font=_load_font(18, bold=True), fill="#d7e7f7")
    draw.text((WIDTH - 112, 1602), str(int(round(leader.titles))), font=_load_font(54, bold=True), fill="#f2d27a", anchor="mm")
    draw.multiline_text((78, 1678), leader_years, font=years_font, fill="#edf4ff", spacing=4)
    draw.text((78, 1810), "All-time NBA championship counts as of 2025", font=_load_font(16, bold=False), fill="#9eb7d9")


def draw_intro_frame(
    frame: Image.Image,
    draw: ImageDraw.ImageDraw,
    latest: Snapshot,
    badges: dict[str, Image.Image],
    title_years_map: dict[str, list[int]],
    *,
    progress: float,
    title_font: ImageFont.ImageFont,
    subtitle_font: ImageFont.ImageFont,
    year_font: ImageFont.ImageFont,
    summary_font: ImageFont.ImageFont,
    trophy_img: Image.Image,
    nba_logo: Image.Image,
) -> None:
    draw_header(frame, draw, latest.year, latest.season_summary, title_font, subtitle_font, year_font, summary_font, trophy_img, nba_logo)
    title_priority = sorted(latest.states, key=lambda state: (-state.titles, state.team_name))
    top3 = title_priority[:3]
    slide = int((1.0 - progress) * 20)

    leader = top3[0]
    leader_years = wrap_years(title_years_map[leader.team_name], 6)
    leader_panel = (58, 420 + slide, 560, 1780 + slide)
    draw_years_panel(frame, draw, leader_panel, leader, leader_years, badges[leader.team_name], large=True)

    card_rects = [
        (586, 420 + slide, 1022, 554 + slide),
        (586, 574 + slide, 1022, 708 + slide),
        (586, 728 + slide, 1022, 862 + slide),
    ]
    for index, (rect, state) in enumerate(zip(card_rects, top3, strict=True), start=1):
        draw_team_card(frame, draw, rect, state, f"#{index}", badges[state.team_name], emphasize=index == 1)

    trophy_card = (586, 896 + slide, 1022, 1268 + slide)
    draw.rounded_rectangle(trophy_card, radius=34, fill=(10, 20, 42, 230), outline=(255, 255, 255, 16), width=2)
    trophy = ImageOps.contain(trophy_img.copy(), (300, 250), method=Image.Resampling.LANCZOS)
    trophy.putalpha(220)
    frame.alpha_composite(trophy, (trophy_card[0] + 68, trophy_card[1] + 28))
    draw.text((trophy_card[0] + 218, trophy_card[1] + 280), "NBA", font=_load_font(36, bold=True), fill="#f2d27a", anchor="mm")
    draw.text(
        (trophy_card[0] + 218, trophy_card[1] + 324),
        "Every title reshapes the board",
        font=_load_font(18, bold=True),
        fill="#d7e7f7",
        anchor="mm",
    )

    draw.text((594, 1294 + slide), "Championship leaders, franchise by franchise.", font=_load_font(20, bold=True), fill="#b8d2ee")
    draw.text((594, 1330 + slide), "Official NBA data from NBA.com", font=_load_font(16, bold=False), fill="#9eb7d9")


def draw_outro_frame(
    frame: Image.Image,
    draw: ImageDraw.ImageDraw,
    latest: Snapshot,
    badges: dict[str, Image.Image],
    title_years_map: dict[str, list[int]],
    *,
    progress: float,
    title_font: ImageFont.ImageFont,
    subtitle_font: ImageFont.ImageFont,
    year_font: ImageFont.ImageFont,
    summary_font: ImageFont.ImageFont,
    row_font: ImageFont.ImageFont,
    sub_font: ImageFont.ImageFont,
    value_font: ImageFont.ImageFont,
    rank_font: ImageFont.ImageFont,
    trophy_img: Image.Image,
    nba_logo: Image.Image,
) -> None:
    draw_header(frame, draw, latest.year, latest.season_summary, title_font, subtitle_font, year_font, summary_font, trophy_img, nba_logo)
    draw.text((72, 420), "FINAL LEADERBOARD", font=_load_font(22, bold=True), fill="#f2d27a")

    top_states = sorted(latest.states, key=lambda state: (-state.titles, state.team_name))[:TOP_N]
    row_h = 88
    row_gap = 10
    base_y = 456
    row_x0 = 54
    row_x1 = WIDTH - 54
    bar_left = 210
    bar_max = 660
    bar_height = 16
    max_titles = max(state.titles for state in top_states)

    for lane in range(TOP_N):
        y = base_y + lane * (row_h + row_gap)
        draw.rounded_rectangle((row_x0, y, row_x1, y + row_h), radius=28, fill=(9, 18, 34, 190), outline=(255, 255, 255, 10), width=1)
        draw.rounded_rectangle((66, y + 12, 122, y + 68), radius=18, fill=(255, 204, 82, 255))
        number = str(lane + 1)
        bbox = draw.textbbox((0, 0), number, font=rank_font)
        draw.text((94 - (bbox[2] - bbox[0]) // 2, y + 27), number, font=rank_font, fill="#12223d")

    for index, state in enumerate(top_states):
        y0 = base_y + index * (row_h + row_gap)
        primary, secondary = TEAM_COLORS.get(state.team_name, ("#39c0ff", "#f4f7fb"))
        frame.alpha_composite(badges[state.team_name], (136, y0 + 10))
        draw.text((214, y0 + 16), truncate_text(draw, state.team_name, row_font, 380), font=row_font, fill="#f4f7fb")
        draw.text((214, y0 + 48), f"{int(round(state.titles))} championships", font=sub_font, fill="#d7e7f7")
        bar_w = max(92, int((state.titles / max_titles) * bar_max))
        draw.rounded_rectangle((bar_left + 6, y0 + 67, bar_left + bar_w + 6, y0 + 67 + bar_height), radius=bar_height // 2, fill=(0, 0, 0, 55))
        draw.rounded_rectangle(
            (bar_left, y0 + 62, bar_left + bar_w, y0 + 62 + bar_height),
            radius=bar_height // 2,
            fill=(*hex_to_rgb(primary), 240),
            outline=mix_rgb(primary, (255, 255, 255), 0.18) + (255,),
            width=2,
        )
        draw.rounded_rectangle((WIDTH - 142, y0 + 20, WIDTH - 72, y0 + 68), radius=18, fill=(255, 204, 82, 255))
        draw.text((WIDTH - 107, y0 + 42), str(int(round(state.titles))), font=value_font, fill="#12223d", anchor="mm")

    leader = top_states[0]
    leader_years = wrap_years(title_years_map[leader.team_name], 6)
    years_font = fit_multiline_font(draw, leader_years, WIDTH - 120, 22, 14, bold=False)
    panel_progress = int((1.0 - progress) * 10)
    panel = (58, 1508 - panel_progress, WIDTH - 58, 1846 - panel_progress)
    draw_years_panel(frame, draw, panel, leader, leader_years, badges[leader.team_name], large=False)
    draw.text((72, 1468), "The final all-time picture", font=_load_font(18, bold=True), fill="#b8d2ee")


def render_video(
    input_csv: Path,
    output_path: Path,
    audio_path: Path,
    duration: float,
    fps: int,
) -> Path:
    snapshots = load_snapshots(input_csv)
    if len(snapshots) < 2:
        raise RuntimeError("Not enough NBA snapshots to render.")

    intro_duration, outro_duration, race_duration = segment_durations(duration)
    if race_duration <= 0:
        raise RuntimeError("Duration is too short for an intro, race, and outro.")

    priorities = build_snapshot_priorities(snapshots)
    title_years_map = title_years(snapshots)
    badges = build_badges([state for snapshot in snapshots for state in snapshot.states])

    background = make_background()
    trophy_img = Image.open(TROPHY).convert("RGBA")
    nba_logo = Image.open(NBA_LOGO).convert("RGBA")

    title_font = _load_font(52, bold=True)
    subtitle_font = _load_font(24, bold=False)
    year_font = _load_font(78, bold=True)
    summary_font = _load_font(24, bold=True)
    row_font = _load_font(28, bold=True)
    sub_font = _load_font(18, bold=True)
    value_font = _load_font(30, bold=True)
    rank_font = _load_font(24, bold=True)

    latest = snapshots[-1]

    def make_frame(t: float) -> np.ndarray:
        frame = background.copy().convert("RGBA")
        draw = ImageDraw.Draw(frame, "RGBA")
        if t < intro_duration:
            progress = ease(t / max(1e-6, intro_duration))
            draw_intro_frame(
                frame,
                draw,
                latest,
                badges,
                title_years_map,
                progress=progress,
                title_font=title_font,
                subtitle_font=subtitle_font,
                year_font=year_font,
                summary_font=summary_font,
                trophy_img=trophy_img,
                nba_logo=nba_logo,
            )
        elif t >= duration - outro_duration:
            progress = ease((t - (duration - outro_duration)) / max(1e-6, outro_duration))
            draw_outro_frame(
                frame,
                draw,
                latest,
                badges,
                title_years_map,
                progress=progress,
                title_font=title_font,
                subtitle_font=subtitle_font,
                year_font=year_font,
                summary_font=summary_font,
                row_font=row_font,
                sub_font=sub_font,
                value_font=value_font,
                rank_font=rank_font,
                trophy_img=trophy_img,
                nba_logo=nba_logo,
            )
        else:
            draw_race_frame(
                frame,
                draw,
                snapshots,
                priorities,
                badges,
                title_years_map,
                t=t,
                total_duration=duration,
                intro_duration=intro_duration,
                outro_duration=outro_duration,
                title_font=title_font,
                subtitle_font=subtitle_font,
                year_font=year_font,
                summary_font=summary_font,
                row_font=row_font,
                sub_font=sub_font,
                value_font=value_font,
                rank_font=rank_font,
                trophy_img=trophy_img,
                nba_logo=nba_logo,
            )
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
    parser = argparse.ArgumentParser(description="Generate a vertical NBA championship leaders short inspired by the reference video.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--audio", type=Path, default=DEFAULT_AUDIO)
    parser.add_argument("--duration", type=float, default=TOTAL_DURATION)
    parser.add_argument("--fps", type=int, default=FPS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = render_video(
        input_csv=args.input,
        output_path=args.output,
        audio_path=args.audio,
        duration=args.duration,
        fps=args.fps,
    )
    print(f"[video_generator] NBA championship leaders short generated -> {output}")


if __name__ == "__main__":
    main()
