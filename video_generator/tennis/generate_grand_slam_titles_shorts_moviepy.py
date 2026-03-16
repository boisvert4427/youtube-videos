from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import re

import numpy as np
from moviepy import VideoClip
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps

from video_generator.generate_atp_barchart_race import _slugify_player_name, to_alpha2
from video_generator.generate_ucl_barchart_race_moviepy import (
    DEFAULT_AUDIO,
    DEFAULT_FLAGS_DIR,
    FINAL_AUDIO_FADE_OUT,
    FPS as DEFAULT_FPS,
    LOOP_CROSSFADE,
    _fit_font_size,
    _load_font,
    build_audio_track,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_INPUT = PROJECT_ROOT / "data" / "processed" / "tennis" / "grand_slam_titles_timeseries_2000_2025.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "tennis" / "grand_slam_titles_shorts_2000_2025_45s.mp4"
DEFAULT_PHOTOS_DIR = PROJECT_ROOT / "data" / "raw" / "player_photos"

WIDTH = 1080
HEIGHT = 1920
TOP_N = 15
FPS = 60
TOTAL_DURATION = 45.0

TITLE = "GRAND SLAM RACE"
SUBTITLE = "ATP men • cumulative majors since 2000"

PLAYER_COLORS = {
    "Andre Agassi": "#d8f15f",
    "Andy Murray": "#3c82ff",
    "Andy Roddick": "#ff7a45",
    "Carlos Alcaraz": "#ffb000",
    "Daniil Medvedev": "#22c4ff",
    "Dominic Thiem": "#f05a5a",
    "Gaston Gaudio": "#b46a43",
    "Goran Ivanisevic": "#7f9aff",
    "Gustavo Kuerten": "#2dd27a",
    "Jannik Sinner": "#ff6b4a",
    "Juan Carlos Ferrero": "#d49b2e",
    "Juan Martin del Potro": "#76b4ff",
    "Lleyton Hewitt": "#fff07c",
    "Marat Safin": "#f55086",
    "Marin Cilic": "#cc5146",
    "Novak Djokovic": "#21b36b",
    "Pete Sampras": "#8b72ff",
    "Rafael Nadal": "#ff8b24",
    "Roger Federer": "#9f3148",
    "Stan Wawrinka": "#f0d8d1",
    "Thomas Johansson": "#b8d6ff",
    "Albert Costa": "#d5a36b",
    "Arthur Ashe": "#4ed0c8",
    "Bjorn Borg": "#35a8e8",
    "Guillermo Vilas": "#8d56ff",
    "Ivan Lendl": "#26c0f6",
    "Jim Courier": "#f3be32",
    "Jimmy Connors": "#36d174",
    "John McEnroe": "#ff5d8f",
    "John Newcombe": "#4cc9f0",
    "Ken Rosewall": "#ff9f1c",
    "Mats Wilander": "#2ec4b6",
    "Rod Laver": "#c77dff",
    "Stefan Edberg": "#52b788",
}


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


def _parse_season_summary(summary: str) -> list[str]:
    parts = [part.strip() for part in summary.split("|") if part.strip()]
    return parts[:4]


def load_snapshots(input_csv: Path) -> list[Snapshot]:
    grouped: dict[str, list[PlayerState]] = {}
    summaries: dict[str, str] = {}
    with input_csv.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            ranking_date = row["ranking_date"].strip()
            grouped.setdefault(ranking_date, []).append(
                PlayerState(
                    player_name=row["player_name"].strip(),
                    country_code=row["country_code"].strip(),
                    titles=float(row["points"]),
                )
            )
            summaries[ranking_date] = row.get("season_summary", "").strip()

    snapshots: list[Snapshot] = []
    for ranking_date in sorted(grouped.keys()):
        snapshots.append(
            Snapshot(
                ranking_date=ranking_date,
                year=datetime.strptime(ranking_date, "%Y-%m-%d").year,
                season_summary=summaries.get(ranking_date, ""),
                states=sorted(grouped[ranking_date], key=lambda item: (-item.titles, item.player_name)),
            )
        )
    return snapshots


def _build_flag_cache(states: list[PlayerState], flags_dir: Path) -> dict[str, Image.Image]:
    cache: dict[str, Image.Image] = {}
    countries = {state.country_code.strip().upper() for state in states if state.country_code.strip()}
    for country_code in countries:
        alpha2 = to_alpha2(country_code)
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
        slug = _slugify_player_name(player_name)
        candidates = [
            photos_dir / f"{slug}.jpg",
            photos_dir / f"{slug}.jpeg",
            photos_dir / f"{slug}.png",
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


def _make_background() -> Image.Image:
    xx = np.linspace(0, 1, WIDTH, dtype=np.float32)
    yy = np.linspace(0, 1, HEIGHT, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(xx, yy)
    navy = np.array([5, 16, 34], dtype=np.float32)
    court = np.array([12, 56, 110], dtype=np.float32)
    neon = np.array([196, 255, 75], dtype=np.float32)
    sky = np.array([111, 202, 255], dtype=np.float32)

    mix = np.clip(0.68 * grid_y + 0.20 * (1.0 - grid_x), 0, 1)
    top_glow = np.exp(-(((grid_x - 0.52) / 0.34) ** 2 + ((grid_y - 0.11) / 0.12) ** 2))
    side_glow = np.exp(-(((grid_x - 0.12) / 0.16) ** 2 + ((grid_y - 0.50) / 0.30) ** 2))
    side_glow += np.exp(-(((grid_x - 0.88) / 0.16) ** 2 + ((grid_y - 0.50) / 0.30) ** 2))

    img = np.clip(
        navy[None, None, :] * (1.0 - mix[..., None])
        + court[None, None, :] * (0.78 * mix[..., None])
        + sky[None, None, :] * (0.16 * top_glow[..., None])
        + neon[None, None, :] * (0.06 * side_glow[..., None]),
        0,
        255,
    ).astype(np.uint8)
    frame = Image.fromarray(img, mode="RGB").convert("RGBA")

    overlay = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")
    draw.rounded_rectangle((70, 110, WIDTH - 70, HEIGHT - 120), radius=52, outline=(255, 255, 255, 18), width=2)
    draw.line((160, 1490, WIDTH - 160, 1490), fill=(255, 255, 255, 15), width=3)
    draw.line((WIDTH // 2, 330, WIDTH // 2, HEIGHT - 220), fill=(255, 255, 255, 11), width=2)
    draw.ellipse((230, 220, WIDTH - 230, 780), outline=(180, 255, 120, 20), width=3)
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
    bd = ImageDraw.Draw(blur_layer, "RGBA")
    bd.text(position, text, font=font, fill=(*glow, 120), anchor=anchor)
    blur_layer = blur_layer.filter(ImageFilter.GaussianBlur(radius=10))
    frame.alpha_composite(blur_layer)
    draw = ImageDraw.Draw(frame, "RGBA")
    draw.text(position, text, font=font, fill=(*fill, 255), anchor=anchor)


def _player_initials(name: str) -> str:
    parts = [part for part in name.replace("-", " ").split() if part]
    if len(parts) == 1:
        return parts[0][:2].upper()
    return (parts[0][0] + parts[-1][0]).upper()


def _rank_with_tie_priority(states: list[PlayerState], top_n: int, priority_order: dict[str, int] | None = None) -> dict[str, int]:
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
        raise RuntimeError("Not enough Grand Slam snapshots to render.")

    all_states = [state for snapshot in snapshots for state in snapshot.states]
    flag_cache = _build_flag_cache(all_states, flags_dir)
    photo_size = 50
    photo_cache = _build_photo_cache(all_states, photos_dir, photo_size)
    priorities = _build_stable_snapshot_priorities(snapshots)
    periods = len(snapshots) - 1
    seconds_per_period = duration / periods
    axis_caps = [float(max(state.titles for state in snapshot.states[:top_n]) + 1) for snapshot in snapshots]

    background = _make_background()
    title_font = _load_font(64, bold=True)
    subtitle_font = _load_font(24, bold=False)
    year_font = _load_font(82, bold=True)
    summary_font_cache: dict[str, ImageFont.ImageFont] = {}
    name_font = _load_font(26, bold=True)
    value_font = _load_font(30, bold=True)
    rank_font = _load_font(26, bold=True)
    initials_font = _load_font(18, bold=True)

    bar_left = 100
    bar_right = 1058
    bar_max_w = bar_right - bar_left
    base_y = 398
    rank_x = 16
    row_gap = -2
    row_h = 98
    bar_height = 93
    photo_box = 50

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

        header = (46, 44, WIDTH - 46, 370)
        draw.rounded_rectangle(header, radius=42, fill=(5, 16, 34, 212), outline=(255, 255, 255, 22), width=1)
        draw.text((78, 76), TITLE, font=title_font, fill="#f3f7fc")
        draw.text((80, 148), SUBTITLE, font=subtitle_font, fill="#b5d7ef")
        draw.rounded_rectangle((74, 198, 262, 292), radius=30, fill=(198, 255, 79, 255))
        year_bbox = draw.textbbox((0, 0), str(nxt.year), font=year_font)
        draw.text((168 - (year_bbox[2] - year_bbox[0]) // 2, 206), str(nxt.year), font=year_font, fill="#0d223e")

        summary_lines = _parse_season_summary(nxt.season_summary)
        summary_key = "\n".join(summary_lines)
        summary_font = summary_font_cache.get(summary_key)
        if summary_font is None:
            longest = max(summary_lines, key=len) if summary_lines else ""
            summary_font = _fit_font_size(draw, longest, 650, 32, 16, bold=True)
            summary_font_cache[summary_key] = summary_font
        summary_rect = (280, 172, WIDTH - 58, 348)
        draw.rounded_rectangle(summary_rect, radius=28, fill=(14, 42, 77, 224), outline=(180, 255, 120, 60), width=2)
        line_height = max(18, int(summary_font.size * 0.92))
        total_text_height = len(summary_lines) * line_height + max(0, len(summary_lines) - 1) * 5
        line_y = summary_rect[1] + max(8, ((summary_rect[3] - summary_rect[1]) - total_text_height) // 2 - 1)
        for line in summary_lines:
            fitted_line = _truncate_text_to_width(draw, line, summary_font, summary_rect[2] - summary_rect[0] - 36)
            draw.text((summary_rect[0] + 18, line_y), fitted_line, font=summary_font, fill="#eef7ff")
            line_y += line_height + 5

        for lane_index in range(top_n):
            lane_y = base_y + lane_index * (row_h + row_gap)
            rank_top = lane_y + max(1, (row_h - bar_height) // 2)
            rank_bottom = rank_top + bar_height
            draw.rounded_rectangle((rank_x, rank_top, rank_x + 54, rank_bottom), radius=18, fill=(198, 255, 79, 255))
            rank_text = str(lane_index + 1)
            rank_bbox = draw.textbbox((0, 0), rank_text, font=rank_font)
            draw.text((rank_x + 27 - (rank_bbox[2] - rank_bbox[0]) // 2, rank_top + max(10, (bar_height - (rank_bbox[3] - rank_bbox[1])) // 2 - 1)), rank_text, font=rank_font, fill="#10233f")

        items: list[tuple[int, float, PlayerState, int]] = []
        for state in top_states:
            prev_idx = prev_rank.get(state.player_name, top_n + 1)
            next_idx = next_rank.get(state.player_name, top_n + 1)
            y_idx = _continuous_rank_position(float(prev_idx), float(next_idx), alpha)
            y = base_y + y_idx * (row_h + row_gap)
            bar_w = max(112, int((state.titles / axis_cap) * bar_max_w))
            moving_up = 1 if next_idx < prev_idx else 0
            items.append((moving_up, y, state, bar_w))
        items.sort(key=lambda item: (item[0], item[1]))

        for _, y, state, bar_w in items:
            y0 = int(y)
            color = PLAYER_COLORS.get(state.player_name, "#39c0ff")
            text_color = _text_on(color)
            bar_top = y0 + max(1, (row_h - bar_height) // 2)
            bar_bottom = bar_top + bar_height
            radius = max(18, bar_height // 2)

            shadow_rect = (bar_left + 8, bar_top + 8, bar_left + bar_w + 8, bar_bottom + 8)
            draw.rounded_rectangle(shadow_rect, radius=radius, fill=(0, 0, 0, 78))
            outline_color = _mix_rgb(color, (255, 255, 255), 0.18)
            highlight_color = _mix_rgb(color, (255, 255, 255), 0.34)
            inner_shadow = _mix_rgb(color, (0, 0, 0), 0.18)
            bar_rect = (bar_left, bar_top, bar_left + bar_w, bar_bottom)
            draw.rounded_rectangle(bar_rect, radius=radius, fill=color, outline=outline_color, width=2)
            sheen_top = bar_top + 8
            sheen_bottom = min(bar_bottom - 8, bar_top + max(14, int(bar_height * 0.24)))
            sheen_right = bar_left + max(82, int(bar_w * 0.62))
            if sheen_bottom > sheen_top and sheen_right > bar_left + 18:
                draw.rounded_rectangle(
                    (bar_left + 8, sheen_top, sheen_right, sheen_bottom),
                    radius=max(10, (sheen_bottom - sheen_top) // 2),
                    fill=(*highlight_color, 52),
                )
            draw.line(
                (bar_left + 18, bar_bottom - 7, bar_left + max(32, int(bar_w * 0.72)), bar_bottom - 7),
                fill=(*inner_shadow, 88),
                width=3,
            )

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
                fy = bar_top + max(1, (bar_height - flag.height) // 2)
                draw.rounded_rectangle((fx - 4, fy - 4, fx + flag.width + 4, fy + flag.height + 4), radius=8, fill=(255, 255, 255, 228))
                frame.alpha_composite(flag, (fx, fy))
                label_x = fx + flag.width + 16

            label_max_width = max(0, bar_w - (label_x - bar_left) - 18)
            player_name = _truncate_text_to_width(draw, state.player_name, name_font, label_max_width)
            draw.text((label_x, bar_top + max(8, (bar_height - 31) // 2)), player_name, font=name_font, fill=text_color)

            value_text = str(int(round(state.titles)))
            value_bbox = draw.textbbox((0, 0), value_text, font=value_font)
            value_x = min(bar_left + bar_w + 18, WIDTH - 56 - (value_bbox[2] - value_bbox[0]))
            _draw_glow_text(frame, (value_x, bar_top + max(10, (bar_height - 30) // 2)), value_text, value_font, (244, 247, 251), (198, 255, 79))

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
    parser = argparse.ArgumentParser(description="Generate a tennis Grand Slam Shorts bar chart race.")
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
    print(f"[video_generator] Grand Slam Shorts race generated -> {output}")


if __name__ == "__main__":
    main()
