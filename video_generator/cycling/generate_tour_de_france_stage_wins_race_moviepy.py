from __future__ import annotations

import argparse
import csv
import math
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
import sys


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
    / "cycling"
    / "tour_de_france"
    / "tour_de_france_stage_wins_postwar_1947_2025.csv"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "cycling"
    / "tour_de_france"
    / "tour_de_france_stage_wins_postwar_1947_2025.mp4"
)
DEFAULT_PHOTOS_DIR = PROJECT_ROOT / "data" / "raw" / "player_photos"

WIDTH = 1920
HEIGHT = 1080
TOP_N = 12
FPS = 30
TOTAL_DURATION = 240.0
FINAL_HOLD_DURATION = 15.0

TITLE = "TOUR DE FRANCE STAGE WINS"
SUBTITLE = "Cumulative victories since 1947"

ALPHA3_TO_ALPHA2 = {
    "ARG": "ar",
    "AUS": "au",
    "AUT": "at",
    "BEL": "be",
    "BGR": "bg",
    "CAN": "ca",
    "COL": "co",
    "CRO": "hr",
    "CZE": "cz",
    "DEN": "dk",
    "ECU": "ec",
    "ERI": "er",
    "ESP": "es",
    "FRA": "fr",
    "GBR": "gb",
    "GER": "de",
    "IRL": "ie",
    "ITA": "it",
    "KAZ": "kz",
    "LTU": "lt",
    "LUX": "lu",
    "NED": "nl",
    "NOR": "no",
    "NZL": "nz",
    "POL": "pl",
    "POR": "pt",
    "ROU": "ro",
    "RUS": "ru",
    "SLO": "si",
    "SRB": "rs",
    "SUI": "ch",
    "SVK": "sk",
    "SWE": "se",
    "USA": "us",
    "ZAF": "za",
}

RIDER_COLORS = {
    "eddy merckx": "#f6d365",
    "bernard hinault": "#4fc3f7",
    "andre darrigade": "#ff8a80",
    "tadej pogacar": "#7ee081",
    "mark cavendish": "#7bdff2",
    "jacques anquetil": "#ffb74d",
    "freddy maertens": "#ff6f61",
    "marcel kittel": "#b39ddb",
    "erik zabel": "#64b5f6",
    "mario cipollini": "#f06292",
    "peter sagan": "#aed581",
    "andre greipel": "#ffcc80",
    "louison bobet": "#81d4fa",
    "charly gaul": "#ffd54f",
    "jasper philipsen": "#80deea",
    "miguel indurain": "#ffab91",
    "walter godefroot": "#ce93d8",
    "wout van aert": "#a5d6a7",
    "lance armstrong": "#ffb300",
    "alessandro petacchi": "#dcedc8",
}

DEFAULT_BAR_COLORS = [
    "#f6d365",
    "#7bdff2",
    "#ff8a80",
    "#aed581",
    "#b39ddb",
    "#ffb74d",
    "#80deea",
    "#ffcc80",
    "#64b5f6",
    "#f06292",
    "#81d4fa",
    "#ce93d8",
    "#a5d6a7",
    "#ffd54f",
    "#ffab91",
    "#4fc3f7",
]


@dataclass(frozen=True)
class PlayerState:
    player_name: str
    country_code: str
    titles: float


@dataclass(frozen=True)
class Snapshot:
    ranking_date: str
    year: int
    season_summary: str
    states: list[PlayerState]


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
    return "#10233f" if luminance > 0.66 else "#f4f7fb"


def _ascii_key(value: str) -> str:
    return unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii").strip().lower()


def _slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", _ascii_key(value)).strip("_")


def _parse_season_summary(summary: str) -> list[str]:
    return [part.strip() for part in summary.split("|") if part.strip()][:4]


def _ease_in_out(value: float) -> float:
    value = min(max(value, 0.0), 1.0)
    return value * value * (3.0 - 2.0 * value)


def _smoothstep(value: float) -> float:
    value = min(max(value, 0.0), 1.0)
    return value * value * (3.0 - 2.0 * value)


def _phase_delay(value: float, delay: float, span: float) -> float:
    if span <= 0.0:
        return 1.0 if value >= delay else 0.0
    return min(max((value - delay) / span, 0.0), 1.0)


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


def _player_initials(name: str) -> str:
    parts = [part for part in _ascii_key(name).replace("-", " ").split() if part]
    if len(parts) == 1:
        return parts[0][:2].upper()
    return (parts[0][0] + parts[-1][0]).upper()


def _to_alpha2(country_code: str) -> str:
    code = country_code.strip().upper()
    if len(code) == 2 and code.isalpha():
        return code.lower()
    if len(code) == 3 and code.isalpha():
        return ALPHA3_TO_ALPHA2.get(code, "")
    return ""


def _strip_tags(text: str) -> str:
    text = re.sub(r"<sup[^>]*>.*?</sup>", "", text, flags=re.S)
    text = re.sub(r"<span[^>]*class=\"mw-ref\"[^>]*>.*?</span>", "", text, flags=re.S)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\[[^\]]*\]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _parse_csv_rows(input_csv: Path) -> list[Snapshot]:
    grouped: dict[str, list[PlayerState]] = {}
    summaries: dict[str, str] = {}
    with input_csv.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            ranking_date = row["ranking_date"].strip()
            grouped.setdefault(ranking_date, []).append(
                PlayerState(
                    player_name=row["player_name"].strip(),
                    country_code=row.get("country_code", "").strip().upper(),
                    titles=float(row["points"]),
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
                states=sorted(grouped[ranking_date], key=lambda item: (-item.titles, item.player_name)),
            )
        )
    return snapshots


def load_snapshots(input_csv: Path) -> list[Snapshot]:
    return _parse_csv_rows(input_csv)


def _build_flag_cache(states: list[PlayerState], flags_dir: Path) -> dict[str, Image.Image]:
    cache: dict[str, Image.Image] = {}
    countries = {state.country_code.strip().upper() for state in states if state.country_code.strip()}
    for country_code in countries:
        alpha2 = _to_alpha2(country_code)
        if not alpha2:
            continue
        path = flags_dir / f"{alpha2.lower()}.png"
        if not path.exists():
            continue
        try:
            img = Image.open(path).convert("RGBA")
            img.thumbnail((38, 26), Image.Resampling.LANCZOS)
            cache[country_code] = img
        except Exception:
            continue
    return cache


def _build_photo_cache(states: list[PlayerState], photos_dir: Path, photo_size: int) -> dict[str, Image.Image]:
    cache: dict[str, Image.Image] = {}
    if not photos_dir.exists():
        return cache
    seen = {state.player_name for state in states}
    for player_name in seen:
        slug = _slugify(player_name)
        candidates = [
            photos_dir / f"{slug}.jpg",
            photos_dir / f"{slug}.jpeg",
            photos_dir / f"{slug}.png",
            photos_dir / f"{slug}.webp",
        ]
        path = next((candidate for candidate in candidates if candidate.exists()), None)
        if path is None:
            continue
        try:
            img = Image.open(path).convert("RGBA")
            img = ImageOps.fit(img, (photo_size, photo_size), method=Image.Resampling.LANCZOS, centering=(0.5, 0.3))
            mask = Image.new("L", (photo_size, photo_size), 0)
            mask_draw = ImageDraw.Draw(mask)
            mask_draw.ellipse((0, 0, photo_size - 1, photo_size - 1), fill=255)
            avatar = Image.new("RGBA", (photo_size, photo_size), (0, 0, 0, 0))
            avatar.paste(img, (0, 0), mask)
            cache[player_name] = avatar
        except Exception:
            continue
    return cache


def _rank_with_tie_priority(
    states: list[PlayerState],
    top_n: int,
    priority_order: dict[str, int] | None = None,
) -> dict[str, int]:
    priority_order = priority_order or {}
    ranked = sorted(
        (state for state in states if state.titles > 0),
        key=lambda item: (-item.titles, priority_order.get(item.player_name, 10_000), item.player_name),
    )
    return {state.player_name: idx for idx, state in enumerate(ranked[:top_n])}


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


def _interp_values(prev: Snapshot, nxt: Snapshot, alpha: float) -> list[PlayerState]:
    prev_map = {state.player_name: state for state in prev.states}
    next_map = {state.player_name: state for state in nxt.states}
    names = sorted(set(prev_map) | set(next_map))
    states: list[PlayerState] = []
    for name in names:
        a = prev_map.get(name) or next_map[name]
        b = next_map.get(name) or prev_map[name]
        titles = a.titles + (b.titles - a.titles) * alpha
        states.append(PlayerState(player_name=name, country_code=b.country_code or a.country_code, titles=titles))
    return states


def _make_background() -> Image.Image:
    xx = np.linspace(0, 1, WIDTH, dtype=np.float32)
    yy = np.linspace(0, 1, HEIGHT, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(xx, yy)
    night = np.array([7, 18, 38], dtype=np.float32)
    blue = np.array([23, 64, 118], dtype=np.float32)
    sky = np.array([95, 176, 255], dtype=np.float32)
    yellow = np.array([246, 208, 91], dtype=np.float32)

    mix = np.clip(0.52 * grid_x + 0.42 * grid_y, 0, 1)
    top_glow = np.exp(-(((grid_x - 0.72) / 0.33) ** 2 + ((grid_y - 0.12) / 0.16) ** 2))
    side_glow = np.exp(-(((grid_x - 0.15) / 0.16) ** 2 + ((grid_y - 0.72) / 0.30) ** 2))
    side_glow += np.exp(-(((grid_x - 0.90) / 0.16) ** 2 + ((grid_y - 0.68) / 0.28) ** 2))

    img = np.clip(
        night[None, None, :] * (1.0 - mix[..., None])
        + blue[None, None, :] * (0.82 * mix[..., None])
        + sky[None, None, :] * (0.16 * top_glow[..., None])
        + yellow[None, None, :] * (0.10 * side_glow[..., None]),
        0,
        255,
    ).astype(np.uint8)
    frame = Image.fromarray(img, mode="RGB").convert("RGBA")

    overlay = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")
    draw.rounded_rectangle((34, 30, WIDTH - 34, HEIGHT - 34), radius=44, outline=(255, 255, 255, 18), width=2)
    draw.line((90, 950, WIDTH - 90, 950), fill=(255, 255, 255, 9), width=2)
    draw.line((130, 930, WIDTH - 130, 930), fill=(255, 255, 255, 6), width=1)
    draw.ellipse((1320, 58, 1880, 618), outline=(246, 208, 91, 20), width=4)
    draw.ellipse((1440, 178, 1760, 498), outline=(246, 208, 91, 12), width=2)
    draw.line((960, 80, 960, 510), fill=(255, 255, 255, 8), width=2)
    draw.line((1060, 110, 1760, 260), fill=(255, 255, 255, 8), width=2)
    draw.line((1120, 1000, 1770, 880), fill=(255, 255, 255, 7), width=2)
    overlay = overlay.filter(ImageFilter.GaussianBlur(radius=3))
    frame.alpha_composite(overlay)
    return frame


def _stage_winner_label(state: PlayerState, fill: str) -> str:
    color_key = _ascii_key(state.player_name)
    if color_key in RIDER_COLORS:
        return RIDER_COLORS[color_key]
    return fill


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
    intro_snapshot = Snapshot(
        ranking_date=f"{first_snapshot.year - 1}-12-31",
        year=first_snapshot.year,
        season_summary="Post-war race|Cumulative stage wins|Since 1947",
        states=[],
    )
    snapshots = [intro_snapshot, *snapshots]

    all_states = [state for snapshot in snapshots for state in snapshot.states]
    flag_cache = _build_flag_cache(all_states, flags_dir)
    photo_cache = _build_photo_cache(all_states, photos_dir, 58)
    priorities = _build_stable_snapshot_priorities(snapshots)

    periods = len(snapshots) - 1
    transition_duration = max(0.1, duration - max(0.0, final_hold_duration))
    seconds_per_period = transition_duration / periods
    axis_caps = [
        float(max((state.titles for state in snapshot.states[:top_n]), default=1.0) + 1)
        for snapshot in snapshots
    ]

    background = _make_background()
    title_font = _load_font(58, bold=True)
    subtitle_font = _load_font(24, bold=False)
    year_font = _load_font(92, bold=True)
    summary_font_cache: dict[str, ImageFont.ImageFont] = {}
    name_font = _load_font(30, bold=True)
    value_font = _load_font(30, bold=True)
    rank_font = _load_font(28, bold=True)
    tick_font = _load_font(20, bold=True)
    initials_font = _load_font(20, bold=True)

    bar_left = 156
    bar_right = 1780
    bar_max_w = bar_right - bar_left
    base_y = 228
    row_h = 58
    row_gap = 8
    avatar_size = 58
    rank_left = 86

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

        draw.text((72, 54), TITLE, font=title_font, fill="#f4f7fb")
        draw.text((74, 116), SUBTITLE, font=subtitle_font, fill="#bfd8ec")

        ranking_bottom = base_y + top_n * row_h + (top_n - 1) * row_gap
        ranking_center_y = (base_y + ranking_bottom) // 2
        header_height = 392
        header_top = ranking_center_y - header_height // 2 + 102
        header_box = (1194, header_top, WIDTH - 54, header_top + header_height)

        for tick in range(1, max_titles + 1):
            x = bar_left + int((tick / axis_cap) * bar_max_w)
            draw.line((x, base_y - 30, x, HEIGHT - 92), fill=(0, 0, 0, 58), width=2)
            draw.text((x - 6, base_y - 60), str(tick), font=tick_font, fill=(0, 0, 0, 104))

        draw.rounded_rectangle(header_box, radius=28, fill=(8, 20, 38, 236), outline=(246, 208, 91, 82), width=2)

        year_badge_width = 228
        year_badge_left = header_box[0] + (header_box[2] - header_box[0] - year_badge_width) // 2
        year_badge = (year_badge_left, header_box[1] + 20, year_badge_left + year_badge_width, header_box[1] + 116)
        draw.rounded_rectangle(year_badge, radius=28, fill=(246, 208, 91, 255))
        year_text = str(nxt.year)
        year_bbox = draw.textbbox((0, 0), year_text, font=year_font)
        year_x = year_badge[0] + (year_badge[2] - year_badge[0] - (year_bbox[2] - year_bbox[0])) // 2
        year_y = year_badge[1] + (year_badge[3] - year_badge[1] - (year_bbox[3] - year_bbox[1])) // 2 - 3
        draw.text((year_x, year_y), year_text, font=year_font, fill="#10233f")

        summary_lines = _parse_season_summary(nxt.season_summary)
        summary_key = "\n".join(summary_lines)
        summary_font = summary_font_cache.get(summary_key)
        if summary_font is None:
            longest = max(summary_lines, key=len) if summary_lines else ""
            summary_font = _fit_font_size(draw, longest, 620, 28, 17, bold=True)
            summary_font_cache[summary_key] = summary_font

        summary_rect = (header_box[0] + 18, header_box[1] + 140, header_box[2] - 18, header_box[3] - 24)
        draw.rounded_rectangle(summary_rect, radius=22, fill=(14, 42, 77, 220), outline=(180, 255, 120, 42), width=2)
        line_gap = 8
        line_height = max(18, int(summary_font.size * 0.92))
        total_summary_h = len(summary_lines) * line_height + max(0, len(summary_lines) - 1) * line_gap
        line_y = summary_rect[1] + max(8, ((summary_rect[3] - summary_rect[1]) - total_summary_h) // 2 - 1)
        for line in summary_lines:
            fitted = _truncate_text_to_width(draw, line, summary_font, summary_rect[2] - summary_rect[0] - 36)
            line_bbox = draw.textbbox((0, 0), fitted, font=summary_font)
            line_x = summary_rect[0] + (summary_rect[2] - summary_rect[0] - (line_bbox[2] - line_bbox[0])) // 2
            draw.text((line_x, line_y), fitted, font=summary_font, fill="#eef7ff")
            line_y += line_height + line_gap

        for rank_idx in range(top_n):
            y0 = base_y + rank_idx * (row_h + row_gap)
            y1 = y0 + row_h
            draw.rounded_rectangle((rank_left, y0, rank_left + 56, y1), radius=18, fill=(244, 205, 98, 255))
            rank_text = str(rank_idx + 1)
            bbox = draw.textbbox((0, 0), rank_text, font=rank_font)
            draw.text(
                (rank_left + 28 - (bbox[2] - bbox[0]) // 2, y0 + (row_h - (bbox[3] - bbox[1])) // 2 - 1),
                rank_text,
                font=rank_font,
                fill="#132742",
            )

        items: list[tuple[int, float, PlayerState, int, int]] = []
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
            bar_w = max(108, int((state.titles / axis_cap) * bar_max_w))
            moving_up = 1 if next_idx < prev_idx else 0
            color_rank = next_idx if next_idx <= top_n else prev_idx
            items.append((moving_up, y, state, bar_w, int(color_rank)))
        items.sort(key=lambda item: (item[0], item[1]))

        for _, y, state, bar_w, color_rank in items:
            y0 = int(y)
            y1 = y0 + row_h
            color = _stage_winner_label(state, DEFAULT_BAR_COLORS[color_rank % len(DEFAULT_BAR_COLORS)])
            text_color = _text_on(color)
            outline = _mix_rgb(color, (255, 255, 255), 0.18)
            highlight = _mix_rgb(color, (255, 255, 255), 0.30)
            shadow = _mix_rgb(color, (0, 0, 0), 0.22)

            draw.rounded_rectangle((bar_left + 6, y0 + 6, bar_left + bar_w + 6, y1 + 6), radius=24, fill=(0, 0, 0, 84))
            draw.rounded_rectangle((bar_left, y0, bar_left + bar_w, y1), radius=24, fill=color, outline=outline, width=2)
            draw.rounded_rectangle(
                (bar_left + 10, y0 + 8, bar_left + max(90, int(bar_w * 0.56)), y0 + 18),
                radius=8,
                fill=(*highlight, 52),
            )
            draw.line((bar_left + 22, y1 - 8, bar_left + max(44, int(bar_w * 0.68)), y1 - 8), fill=(*shadow, 92), width=3)

            avatar_x = bar_left + 8
            avatar_y = y0 + (row_h - avatar_size) // 2
            draw.ellipse((avatar_x - 2, avatar_y - 2, avatar_x + avatar_size + 2, avatar_y + avatar_size + 2), fill=(255, 255, 255, 228))
            photo = photo_cache.get(state.player_name)
            if photo is not None:
                frame.alpha_composite(photo, (avatar_x, avatar_y))
            else:
                draw.ellipse((avatar_x, avatar_y, avatar_x + avatar_size, avatar_y + avatar_size), fill=(8, 20, 40, 186))
                initials = _player_initials(state.player_name)
                bbox = draw.textbbox((0, 0), initials, font=initials_font)
                draw.text(
                    (
                        avatar_x + avatar_size // 2 - (bbox[2] - bbox[0]) // 2,
                        avatar_y + avatar_size // 2 - (bbox[3] - bbox[1]) // 2 - 1,
                    ),
                    initials,
                    font=initials_font,
                    fill="#f5f7fb",
                )

            label_x = bar_left + avatar_size + 20
            flag = flag_cache.get(state.country_code)
            if flag is not None:
                fx = label_x
                fy = y0 + (row_h - flag.height) // 2
                draw.rounded_rectangle((fx - 4, fy - 4, fx + flag.width + 4, fy + flag.height + 4), radius=8, fill=(255, 255, 255, 228))
                frame.alpha_composite(flag, (fx, fy))
                label_x = fx + flag.width + 16

            label_max_width = max(0, bar_w - (label_x - bar_left) - 24)
            rider_name = _truncate_text_to_width(draw, state.player_name, name_font, label_max_width)
            draw.text((label_x, y0 + (row_h - 28) // 2 - 1), rider_name, font=name_font, fill=text_color)

            value_text = str(int(round(state.titles)))
            vbox = draw.textbbox((0, 0), value_text, font=value_font)
            value_x = min(bar_left + bar_w + 18, WIDTH - 64 - (vbox[2] - vbox[0]))
            draw.text((value_x, y0 + (row_h - 28) // 2 - 1), value_text, font=value_font, fill="#f4f7fb")

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
    parser = argparse.ArgumentParser(description="Generate a landscape Tour de France stage wins race video.")
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
    print(f"[video_generator] Tour de France stage wins landscape race generated -> {output}")


if __name__ == "__main__":
    main()
