from __future__ import annotations

import argparse
import math
import sys
from datetime import datetime, timedelta
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
    AXIS_CAP,
    DEFAULT_INPUT,
    DEFAULT_LOGOS_DIR,
    BrowserState,
    Snapshot,
    _build_color_map,
    _build_logo_cache,
    _build_priorities,
    _center_text,
    _continuous_rank_position,
    _ease_in_out,
    _interpolate,
    _mix_rgb,
    _phase_delay,
    _rank,
    _truncate,
    load_snapshots,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "technology"
    / "browser_market_share"
    / "browser_market_share_race_1995_2026_3min.mp4"
)

WIDTH = 1920
HEIGHT = 1080
TOP_N = 8
FPS = 60
TOTAL_DURATION = 180.0
FINAL_HOLD_DURATION = 10.0

TITLE = "BROWSER WARS"
SUBTITLE = "WORLDWIDE MARKET SHARE HISTORY | 1995-2026"
TITLE_FONT_SIZE = 64
SUBTITLE_FONT_SIZE = 24
LEFT_HEADER_LABEL = "BROWSER"
RIGHT_HEADER_LABEL = "MARKET SHARE"
FOOTER = "BROWSER MARKET SHARE | 1995-2026"


def ERA_LABEL(ranking_date: str) -> str:
    if ranking_date < "2001-01-01":
        return "Netscape vs Internet Explorer"
    if ranking_date < "2009-01-01":
        return "Internet Explorer era"
    if ranking_date < "2015-01-01":
        return "Chrome enters the race"
    return "The modern browser era"


def _make_background() -> Image.Image:
    x_values = np.linspace(0, 1, WIDTH, dtype=np.float32)
    y_values = np.linspace(0, 1, HEIGHT, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(x_values, y_values)
    deep = np.array([4, 8, 18], dtype=np.float32)
    blue = np.array([8, 38, 67], dtype=np.float32)
    cyan = np.array([0, 202, 214], dtype=np.float32)
    orange = np.array([255, 143, 66], dtype=np.float32)

    mix = np.clip(0.48 * grid_x + 0.54 * grid_y, 0, 1)
    cyan_glow = np.exp(-(((grid_x - 0.90) / 0.24) ** 2 + ((grid_y - 0.09) / 0.17) ** 2))
    orange_glow = np.exp(-(((grid_x - 0.06) / 0.27) ** 2 + ((grid_y - 0.90) / 0.22) ** 2))
    pixels = np.clip(
        deep[None, None, :] * (1.0 - mix[..., None])
        + blue[None, None, :] * (0.92 * mix[..., None])
        + cyan[None, None, :] * (0.14 * cyan_glow[..., None])
        + orange[None, None, :] * (0.055 * orange_glow[..., None]),
        0,
        255,
    ).astype(np.uint8)
    frame = Image.fromarray(pixels, mode="RGB").convert("RGBA")

    overlay = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")
    for x in range(34, WIDTH, 105):
        draw.line((x, 206, x, HEIGHT - 48), fill=(52, 211, 222, 8), width=1)
    for y in range(225, HEIGHT - 45, 84):
        draw.line((34, y, WIDTH - 34, y), fill=(52, 211, 222, 7), width=1)
    draw.ellipse((1445, -205, 2065, 415), outline=(93, 230, 237, 23), width=3)
    draw.ellipse((1545, -105, 1965, 315), outline=(93, 230, 237, 13), width=2)
    for x, y in ((96, 930), (250, 990), (1650, 765), (1810, 900), (1470, 130)):
        draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=(93, 230, 237, 72))
        draw.ellipse((x - 22, y - 22, x + 22, y + 22), outline=(93, 230, 237, 19), width=2)
    draw.rounded_rectangle(
        (28, 24, WIDTH - 28, HEIGHT - 28),
        radius=42,
        outline=(255, 255, 255, 18),
        width=2,
    )
    overlay = overlay.filter(ImageFilter.GaussianBlur(radius=1.6))
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
    box = (x, y, x + 58, y + 58)
    draw.rounded_rectangle(
        box,
        radius=15,
        fill=(241, 247, 251, 245),
        outline=(*_mix_rgb(color, (255, 255, 255), 0.38), 210),
        width=2,
    )
    if logo is not None:
        fitted = logo.copy()
        fitted.thumbnail((50, 50), Image.Resampling.LANCZOS)
        frame.alpha_composite(
            fitted,
            (x + (58 - fitted.width) // 2, y + (58 - fitted.height) // 2),
        )
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

    background = _make_background()
    title_font = _load_font(TITLE_FONT_SIZE, bold=True)
    subtitle_font = _load_font(SUBTITLE_FONT_SIZE, bold=True)
    date_font = _load_font(48, bold=True)
    insight_font_cache: dict[str, ImageFont.ImageFont] = {}
    label_font = _load_font(18, bold=True)
    name_font = _load_font(29, bold=True)
    value_font = _load_font(28, bold=True)
    rank_font = _load_font(25, bold=True)
    tick_font = _load_font(18, bold=True)
    footer_font = _load_font(18, bold=True)
    monogram_font = _load_font(22, bold=True)

    header_box = (38, 34, WIDTH - 38, 188)
    insight_box = (735, 58, 1392, 163)
    date_box = (1460, 52, 1844, 168)

    rank_left = 48
    logo_left = 118
    name_left = 192
    bar_left = 500
    bar_right = 1778
    bar_max_width = bar_right - bar_left
    base_y = 274
    pitch = 74 if top_n > 8 else 88
    row_height = 58 if top_n > 8 else 62
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
        states_by_key = {state.browser_key: state for state in interpolated}
        priority = priorities[period_index]
        previous_rank = _rank(prev.states, top_n, priority)
        target_rank = _rank(nxt.states, top_n, priority)
        visible_keys = sorted(set(previous_rank) | set(target_rank))

        draw.rounded_rectangle(
            header_box,
            radius=35,
            fill=(3, 13, 27, 230),
            outline=(90, 226, 234, 43),
            width=2,
        )
        draw.text((72, 53), TITLE, font=title_font, fill="#F5FAFF")
        draw.text((76, 128), SUBTITLE, font=subtitle_font, fill="#62DDE6")

        draw.rounded_rectangle(
            insight_box,
            radius=27,
            fill=(8, 38, 59, 238),
            outline=(90, 226, 234, 67),
            width=2,
        )
        display_snapshot = prev if value_alpha < 0.5 and prev.states else nxt
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
                insight_box[2] - insight_box[0] - 38,
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
            "#F2FAFC",
        )

        draw.rounded_rectangle(
            date_box,
            radius=29,
            fill=(255, 143, 66, 255),
            outline=(255, 212, 171, 220),
            width=2,
        )
        date_label = datetime.strptime(
            display_snapshot.ranking_date,
            "%Y-%m-%d",
        ).strftime("%b %Y").upper()
        _center_text(draw, date_box, date_label, date_font, "#102033")

        draw.text((name_left, 219), LEFT_HEADER_LABEL, font=label_font, fill=(177, 215, 222, 210))
        draw.text((bar_left + 18, 219), RIGHT_HEADER_LABEL, font=label_font, fill=(177, 215, 222, 210))

        for tick in range(0, 101, 20):
            x = bar_left + int((tick / AXIS_CAP) * bar_max_width)
            draw.line(
                (x, 252, x, ranking_bottom + 16),
                fill=(1, 8, 17, 105),
                width=3 if tick == 0 else 2,
            )
            tick_text = f"{tick}%"
            bbox = draw.textbbox((0, 0), tick_text, font=tick_font)
            draw.text(
                (x - (bbox[2] - bbox[0]) // 2, 218),
                tick_text,
                font=tick_font,
                fill=(4, 17, 30, 175),
            )

        for rank_index in range(top_n):
            row_y = base_y + rank_index * pitch
            fill = (255, 143, 66, 255) if rank_index == 0 else (15, 73, 92, 245)
            text_fill = "#102033" if rank_index == 0 else "#EDF9FB"
            rank_box = (rank_left, row_y + 2, rank_left + 58, row_y + 60)
            draw.rounded_rectangle(rank_box, radius=19, fill=fill)
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
            bar_width = max(0, int((state.market_share / AXIS_CAP) * bar_max_width))
            moving_up = 1 if target_index < previous_index else 0
            render_items.append((moving_up, row_y, state, bar_width))

        render_items.sort(key=lambda item: (item[0], item[1]))
        for _, row_y, state, bar_width in render_items:
            y0 = int(row_y)
            y1 = y0 + row_height
            if y1 < base_y - pitch or y0 > ranking_bottom + pitch:
                continue

            color = colors[state.browser_key]
            highlight = _mix_rgb(color, (255, 255, 255), 0.30)
            shadow = _mix_rgb(color, (0, 0, 0), 0.25)
            draw.rounded_rectangle(
                (108, y0 - 5, WIDTH - 54, y1 + 5),
                radius=22,
                fill=(3, 15, 27, 56),
            )
            _draw_logo(
                frame,
                draw,
                logos.get(state.browser_key),
                state,
                logo_left,
                y0 + 2,
                color,
                monogram_font,
            )

            browser_name = _truncate(
                draw,
                state.browser_name,
                name_font,
                bar_left - name_left - 28,
            )
            name_bbox = draw.textbbox((0, 0), browser_name, font=name_font)
            name_y = y0 + (row_height - (name_bbox[3] - name_bbox[1])) // 2 - name_bbox[1]
            draw.text((name_left, name_y), browser_name, font=name_font, fill="#F3F8FA")

            if bar_width > 0:
                draw.rounded_rectangle(
                    (bar_left + 7, y0 + 7, bar_left + bar_width + 7, y1 + 7),
                    radius=22,
                    fill=(0, 0, 0, 92),
                )
                draw.rounded_rectangle(
                    (bar_left, y0, bar_left + max(7, bar_width), y1),
                    radius=22,
                    fill=color,
                    outline=highlight,
                    width=2,
                )
                if bar_width > 48:
                    draw.rounded_rectangle(
                        (bar_left + 10, y0 + 9, bar_left + max(34, int(bar_width * 0.64)), y0 + 20),
                        radius=7,
                        fill=(*highlight, 66),
                    )
                    draw.line(
                        (
                            bar_left + 20,
                            y1 - 9,
                            bar_left + max(34, int(bar_width * 0.72)),
                            y1 - 9,
                        ),
                        fill=(*shadow, 105),
                        width=3,
                    )

            value_text = f"{state.market_share:.1f}%"
            value_bbox = draw.textbbox((0, 0), value_text, font=value_font)
            value_width = value_bbox[2] - value_bbox[0]
            value_x = min(
                bar_left + max(7, bar_width) + 16,
                WIDTH - 42 - value_width,
            )
            value_y = y0 + (row_height - (value_bbox[3] - value_bbox[1])) // 2 - value_bbox[1]
            draw.text((value_x, value_y), value_text, font=value_font, fill="#F7FBFC")

        footer = FOOTER
        footer_bbox = draw.textbbox((0, 0), footer, font=footer_font)
        draw.text(
            ((WIDTH - (footer_bbox[2] - footer_bbox[0])) // 2, 1037),
            footer,
            font=footer_font,
            fill=(172, 211, 220, 185),
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
        description="Generate a landscape worldwide browser market share bar chart race."
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
    print(f"[video_generator] Browser market share landscape race generated -> {output}")


if __name__ == "__main__":
    main()
