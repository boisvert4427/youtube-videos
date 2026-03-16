from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
from moviepy import VideoClip
from PIL import Image, ImageDraw, ImageFilter, ImageOps

from video_generator.generate_ucl_barchart_race_moviepy import (
    DEFAULT_AUDIO,
    _fit_font_size,
    _load_font,
    build_audio_track,
)
from video_generator.basketball.generate_nba_titles_shorts_moviepy import (
    DEFAULT_INPUT,
    TEAM_COLORS,
    Snapshot,
    TeamState,
    _build_badge_cache,
    _continuous_rank_position,
    _draw_team_badge,
    _ease_in_out,
    _mix_rgb,
    _text_on,
    _truncate_text_to_width,
    load_snapshots,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "basketball" / "nba_titles_race_landscape_1947_2025_4min.mp4"
LOGO_DIR = PROJECT_ROOT / "data" / "raw" / "nba_team_logos"

WIDTH = 1920
HEIGHT = 1080
TOP_N = 12
FPS = 30
TOTAL_DURATION = 240.0

TITLE = "NBA TITLES RACE"
SUBTITLE = "Franchise championships"

TEAM_LOGO_FILES = {
    "Atlanta Hawks": "atlanta_hawks.png",
    "Baltimore Bullets": "baltimore_bullets.png",
    "Boston Celtics": "boston_celtics.png",
    "Chicago Bulls": "chicago_bulls.png",
    "Dallas Mavericks": "dallas_mavericks.png",
    "Detroit Pistons": "detroit_pistons.png",
    "Golden State Warriors": "golden_state_warriors.png",
    "Houston Rockets": "houston_rockets.png",
    "Los Angeles Lakers": "los_angeles_lakers.png",
    "Miami Heat": "miami_heat.png",
    "Milwaukee Bucks": "milwaukee_bucks.png",
    "New York Knicks": "new_york_knicks.png",
    "Oklahoma City Thunder": "oklahoma_city_thunder.png",
    "Philadelphia 76ers": "philadelphia_76ers.png",
    "Portland Trail Blazers": "portland_trail_blazers.png",
    "Sacramento Kings": "sacramento_kings.png",
    "San Antonio Spurs": "san_antonio_spurs.png",
    "Washington Wizards": "washington_wizards.png",
}


def _phase_delay(value: float, delay: float, span: float) -> float:
    if span <= 0.0:
        return 1.0 if value >= delay else 0.0
    return min(max((value - delay) / span, 0.0), 1.0)


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


def _rank_with_priority(states: list[TeamState], top_n: int, priority_order: dict[str, int] | None = None) -> dict[str, int]:
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


def _make_background() -> Image.Image:
    xx = np.linspace(0, 1, WIDTH, dtype=np.float32)
    yy = np.linspace(0, 1, HEIGHT, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(xx, yy)
    navy = np.array([6, 14, 34], dtype=np.float32)
    arena = np.array([9, 44, 92], dtype=np.float32)
    glow = np.array([231, 72, 92], dtype=np.float32)
    sky = np.array([84, 192, 255], dtype=np.float32)

    mix = np.clip(0.56 * grid_x + 0.34 * grid_y, 0, 1)
    center_glow = np.exp(-(((grid_x - 0.55) / 0.28) ** 2 + ((grid_y - 0.18) / 0.18) ** 2))
    side_glow = np.exp(-(((grid_x - 0.08) / 0.12) ** 2 + ((grid_y - 0.55) / 0.36) ** 2))
    side_glow += np.exp(-(((grid_x - 0.92) / 0.12) ** 2 + ((grid_y - 0.55) / 0.36) ** 2))

    img = np.clip(
        navy[None, None, :] * (1.0 - mix[..., None])
        + arena[None, None, :] * (0.78 * mix[..., None])
        + sky[None, None, :] * (0.14 * center_glow[..., None])
        + glow[None, None, :] * (0.08 * side_glow[..., None]),
        0,
        255,
    ).astype(np.uint8)
    frame = Image.fromarray(img, mode="RGB").convert("RGBA")

    overlay = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")
    draw.rounded_rectangle((38, 34, WIDTH - 38, HEIGHT - 40), radius=42, outline=(255, 255, 255, 22), width=2)
    draw.line((112, 530, WIDTH - 112, 530), fill=(255, 255, 255, 12), width=2)
    draw.ellipse((720, 38, 1200, 514), outline=(255, 255, 255, 18), width=3)
    draw.line((960, 70, 960, 486), fill=(255, 255, 255, 10), width=2)
    overlay = overlay.filter(ImageFilter.GaussianBlur(radius=3))
    frame.alpha_composite(overlay)
    return frame


def _make_logo_tile(team_name: str, size: int) -> Image.Image | None:
    logo_name = TEAM_LOGO_FILES.get(team_name)
    if logo_name is None:
        return None
    logo_path = LOGO_DIR / logo_name
    if not logo_path.exists():
        return None

    tile = Image.new("RGBA", (size + 8, size + 8), (0, 0, 0, 0))
    layer = ImageDraw.Draw(tile, "RGBA")
    layer.rounded_rectangle((0, 0, size + 7, size + 7), radius=20, fill=(255, 255, 255, 242))
    layer.rounded_rectangle((3, 3, size + 4, size + 4), radius=17, fill=(11, 26, 52, 252))
    layer.rounded_rectangle((6, 6, size + 1, size + 1), radius=15, fill=(255, 255, 255, 246))

    logo = Image.open(logo_path).convert("RGBA")
    logo = ImageOps.contain(logo, (size - 10, size - 10))
    logo_x = 4 + (size - logo.width) // 2
    logo_y = 4 + (size - logo.height) // 2
    tile.alpha_composite(logo, (logo_x, logo_y))
    return tile


def _build_logo_cache(states: list[TeamState], size: int) -> dict[str, Image.Image]:
    cache: dict[str, Image.Image] = {}
    for state in states:
        if state.team_name in cache:
            continue
        logo = _make_logo_tile(state.team_name, size)
        if logo is not None:
            cache[state.team_name] = logo
    return cache


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
    logo_cache = _build_logo_cache(all_states, 60)
    badge_cache = _build_badge_cache(all_states, 60)
    priorities = _build_stable_snapshot_priorities(snapshots)

    periods = len(snapshots) - 1
    seconds_per_period = duration / periods
    axis_caps = [
        float(max((state.titles for state in snapshot.states[:top_n]), default=1.0) + 1)
        for snapshot in snapshots
    ]

    background = _make_background()
    title_font = _load_font(58, bold=True)
    subtitle_font = _load_font(24, bold=False)
    year_font = _load_font(96, bold=True)
    summary_font_cache: dict[str, object] = {}
    name_font = _load_font(30, bold=True)
    value_font = _load_font(30, bold=True)
    rank_font = _load_font(28, bold=True)
    tick_font = _load_font(20, bold=True)

    bar_left = 162
    bar_right = 1782
    bar_max_w = bar_right - bar_left
    base_y = 228
    row_h = 58
    row_gap = 6
    logo_size = 60
    rank_left = 90

    def make_frame(t: float) -> np.ndarray:
        frame = background.copy()
        draw = ImageDraw.Draw(frame, "RGBA")

        period_index = min(int(t / seconds_per_period), periods - 1)
        local_t = (t - period_index * seconds_per_period) / seconds_per_period
        alpha = _ease_in_out(local_t)

        prev = snapshots[period_index]
        nxt = snapshots[period_index + 1]
        axis_cap = axis_caps[period_index] + (axis_caps[period_index + 1] - axis_caps[period_index]) * alpha
        interp = _interp_values(prev, nxt, alpha)
        interp_map = {state.team_name: state for state in interp}
        priority = priorities[period_index]
        prev_rank = _rank_with_priority(prev.states, top_n, priority)
        next_rank = _rank_with_priority(nxt.states, top_n, priority)
        visible_names = sorted(set(prev_rank) | set(next_rank))
        top_states = [interp_map[name] for name in visible_names if name in interp_map]
        max_titles = max(1, int(math.ceil(axis_cap)))

        draw.text((72, 54), TITLE, font=title_font, fill="#f4f7fb")
        draw.text((74, 116), SUBTITLE, font=subtitle_font, fill="#b6d6ee")

        ranking_bottom = base_y + top_n * row_h + (top_n - 1) * row_gap
        ranking_center_y = (base_y + ranking_bottom) // 2
        header_height = 340
        header_top = ranking_center_y - header_height // 2 + 122
        header_box = (1188, header_top, WIDTH - 52, header_top + header_height)

        for tick in range(1, max_titles + 1):
            x = bar_left + int((tick / axis_cap) * bar_max_w)
            draw.line((x, base_y - 30, x, HEIGHT - 88), fill=(0, 0, 0, 70), width=2)
            draw.text((x - 6, base_y - 60), str(tick), font=tick_font, fill=(0, 0, 0, 110))

        draw.rounded_rectangle(header_box, radius=28, fill=(6, 18, 38, 220), outline=(255, 255, 255, 30), width=2)
        year_text = str(nxt.year)
        year_bbox = draw.textbbox((0, 0), year_text, font=year_font)
        year_x = header_box[0] + (header_box[2] - header_box[0] - (year_bbox[2] - year_bbox[0])) // 2
        draw.text((year_x, header_box[1] + 22), year_text, font=year_font, fill="#ffcc52")

        summary_text = nxt.season_summary or ""
        summary_font = summary_font_cache.get(summary_text)
        if summary_font is None:
            summary_font = _fit_font_size(draw, summary_text or " ", 600, 26, 17, bold=True)
            summary_font_cache[summary_text] = summary_font
        fitted = _truncate_text_to_width(draw, summary_text, summary_font, 600)
        line_bbox = draw.textbbox((0, 0), fitted, font=summary_font)
        line_x = header_box[0] + (header_box[2] - header_box[0] - (line_bbox[2] - line_bbox[0])) // 2
        line_y = header_box[1] + 170
        draw.text((line_x, line_y), fitted, font=summary_font, fill="#eef8ff")

        for rank_idx in range(top_n):
            y0 = base_y + rank_idx * (row_h + row_gap)
            y1 = y0 + row_h
            draw.rounded_rectangle((rank_left, y0, rank_left + 56, y1), radius=18, fill=(255, 204, 82, 255))
            rank_text = str(rank_idx + 1)
            bbox = draw.textbbox((0, 0), rank_text, font=rank_font)
            draw.text(
                (rank_left + 28 - (bbox[2] - bbox[0]) // 2, y0 + (row_h - (bbox[3] - bbox[1])) // 2 - 1),
                rank_text,
                font=rank_font,
                fill="#112642",
            )

        items: list[tuple[int, float, TeamState, int]] = []
        for state in top_states:
            prev_idx = prev_rank.get(state.team_name, top_n + 1)
            next_idx = next_rank.get(state.team_name, top_n + 1)
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
            bar_w = max(120, int((state.titles / axis_cap) * bar_max_w))
            moving_up = 1 if next_idx < prev_idx else 0
            items.append((moving_up, y, state, bar_w))
        items.sort(key=lambda item: (item[0], item[1]))

        for _, y, state, bar_w in items:
            y0 = int(y)
            y1 = y0 + row_h
            primary, secondary = TEAM_COLORS.get(state.team_name, ("#3dbdf5", "#f4f7fb"))
            text_color = _text_on(primary)
            outline = _mix_rgb(primary, (255, 255, 255), 0.18)
            highlight = _mix_rgb(primary, (255, 255, 255), 0.32)
            shadow = _mix_rgb(primary, (0, 0, 0), 0.20)

            draw.rounded_rectangle((bar_left + 6, y0 + 6, bar_left + bar_w + 6, y1 + 6), radius=24, fill=(0, 0, 0, 84))
            draw.rounded_rectangle((bar_left, y0, bar_left + bar_w, y1), radius=24, fill=primary, outline=outline, width=2)
            draw.rounded_rectangle(
                (bar_left + 10, y0 + 8, bar_left + max(90, int(bar_w * 0.62)), y0 + 18),
                radius=8,
                fill=(*highlight, 52),
            )
            draw.line((bar_left + 20, y1 - 8, bar_left + max(40, int(bar_w * 0.72)), y1 - 8), fill=(*shadow, 90), width=3)

            logo = logo_cache.get(state.team_name)
            logo_x = bar_left + 8
            logo_y = y0 + (row_h - logo_size) // 2
            if logo is not None:
                frame.alpha_composite(logo, (logo_x - 4, logo_y - 4))
            else:
                badge = badge_cache.get(state.team_name)
                if badge is not None:
                    _draw_team_badge(frame, logo_x, logo_y, badge)

            label_x = bar_left + logo_size + 20
            label_max_width = max(0, bar_w - (label_x - bar_left) - 24)
            team_name = _truncate_text_to_width(draw, state.team_name, name_font, label_max_width)
            draw.text((label_x, y0 + (row_h - 28) // 2 - 1), team_name, font=name_font, fill=text_color)

            value_text = str(int(round(state.titles)))
            vbox = draw.textbbox((0, 0), value_text, font=value_font)
            value_x = min(bar_left + bar_w + 18, WIDTH - 64 - (vbox[2] - vbox[0]))
            draw.text((value_x, y0 + (row_h - 28) // 2 - 1), value_text, font=value_font, fill="#f4f7fb")

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
    parser = argparse.ArgumentParser(description="Generate a landscape NBA franchise titles race video.")
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
    print(f"[video_generator] NBA landscape race generated -> {output}")


if __name__ == "__main__":
    main()
