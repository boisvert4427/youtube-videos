from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from moviepy import VideoClip
from PIL import Image, ImageDraw, ImageFont, ImageOps

from video_generator.generate_ucl_barchart_race_moviepy import (
    CLUB_COLORS,
    DEFAULT_AUDIO,
    DEFAULT_INPUT,
    DEFAULT_LOGOS_DIR,
    FINAL_AUDIO_FADE_OUT,
    FPS as DEFAULT_FPS,
    LOOP_CROSSFADE,
    _build_logo_cache,
    _build_stable_snapshot_priorities,
    _continuous_rank_position,
    _filter_snapshots,
    _fit_font_size,
    _interp_values,
    _load_font,
    _rank_with_tie_priority,
    _slugify,
    _text_on,
    build_audio_track,
    load_snapshots,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "ucl_titles_race_shorts_premium.mp4"

WIDTH = 1080
HEIGHT = 1920
TOP_N = 15
FPS = DEFAULT_FPS
TOTAL_DURATION = 24.0
TITLE = "MOST CHAMPIONS LEAGUE TITLES"


def _make_background() -> Image.Image:
    xx = np.linspace(0, 1, WIDTH, dtype=np.float32)
    yy = np.linspace(0, 1, HEIGHT, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(xx, yy)
    navy = np.array([5, 17, 39], dtype=np.float32)
    blue = np.array([12, 42, 87], dtype=np.float32)
    cyan = np.array([41, 122, 188], dtype=np.float32)
    gold = np.array([216, 169, 73], dtype=np.float32)
    mix = np.clip(0.62 * grid_y + 0.22 * (1.0 - grid_x), 0, 1)
    glow = np.exp(-(((grid_x - 0.82) / 0.28) ** 2 + ((grid_y - 0.08) / 0.22) ** 2))
    img = np.clip(
        navy[None, None, :] * (1.0 - mix[..., None])
        + blue[None, None, :] * (0.66 * mix[..., None])
        + cyan[None, None, :] * (0.18 * (1.0 - grid_y[..., None]))
        + gold[None, None, :] * (0.14 * glow[..., None]),
        0,
        255,
    ).astype(np.uint8)
    return Image.fromarray(img, mode="RGB")


def _paste_logo(frame: Image.Image, logo_img: Image.Image | None, x: int, y: int, box_size: int) -> None:
    if logo_img is None:
        return
    plate = Image.new("RGBA", (box_size, box_size), (0, 0, 0, 0))
    plate_draw = ImageDraw.Draw(plate, "RGBA")
    plate_draw.rounded_rectangle((0, 0, box_size - 1, box_size - 1), radius=16, fill=(255, 255, 255, 232))
    fitted = ImageOps.contain(logo_img, (box_size - 12, box_size - 12), method=Image.Resampling.LANCZOS)
    plate.alpha_composite(fitted, ((box_size - fitted.width) // 2, (box_size - fitted.height) // 2))
    frame.alpha_composite(plate, (x, y))


def _truncate_text_to_width(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.ImageFont,
    max_width: int,
) -> str:
    if max_width <= 0:
        return ""
    if draw.textbbox((0, 0), text, font=font)[2] <= max_width:
        return text
    ellipsis = "..."
    low = 0
    high = len(text)
    best = ellipsis
    while low <= high:
        mid = (low + high) // 2
        candidate = text[:mid].rstrip() + ellipsis
        if draw.textbbox((0, 0), candidate, font=font)[2] <= max_width:
            best = candidate
            low = mid + 1
        else:
            high = mid - 1
    return best


def render_video(
    input_csv: Path,
    output_path: Path,
    logos_dir: Path,
    audio_path: Path,
    duration: float,
    fps: int,
    top_n: int,
    start_year: int | None = None,
    end_year: int | None = None,
) -> Path:
    snapshots = load_snapshots(input_csv)
    snapshots = _filter_snapshots(snapshots, start_year, end_year)
    if len(snapshots) < 2:
        raise RuntimeError("Not enough UCL snapshots to render.")

    snapshot_priorities = _build_stable_snapshot_priorities(snapshots)
    logo_cache_raw = _build_logo_cache(logos_dir)
    logo_cache = {
        key: Image.fromarray(value).convert("RGBA")
        for key, value in logo_cache_raw.items()
    }

    periods = len(snapshots) - 1
    seconds_per_period = duration / periods

    background = _make_background()
    title_font = _load_font(50, bold=True)
    subtitle_font = _load_font(24, bold=False)
    year_font = _load_font(68, bold=True)
    score_font_cache: dict[str, ImageFont.ImageFont] = {}
    name_font = _load_font(28, bold=True)
    value_font = _load_font(30, bold=True)
    rank_font = _load_font(24, bold=True)

    bar_left = 140
    bar_right = 984
    bar_max_w = bar_right - bar_left
    base_y = 410
    rank_x = 42
    logo_box = 58

    global_peak = 1.0
    for snapshot in snapshots:
        for state in snapshot.states:
            global_peak = max(global_peak, state.titles)
    axis_cap = float(int(np.ceil(global_peak)) + 1)
    available_height = HEIGHT - base_y - 120
    pitch = max(74, int(available_height / max(1, top_n)))
    row_gap = max(4, min(8, int(pitch * 0.10)))
    row_h = max(54, pitch - row_gap)
    bar_height = max(44, row_h - 16)
    logo_box = min(52, max(38, bar_height - 10))

    def _ease_in_out(value: float) -> float:
        value = min(max(value, 0.0), 1.0)
        return 3.0 * value * value - 2.0 * value * value * value

    def make_frame(t: float) -> np.ndarray:
        frame = background.copy().convert("RGBA")
        draw = ImageDraw.Draw(frame, "RGBA")

        period_index = min(int(t / seconds_per_period), periods - 1)
        local_t = (t - period_index * seconds_per_period) / seconds_per_period
        alpha = _ease_in_out(local_t)

        prev = snapshots[period_index]
        nxt = snapshots[period_index + 1]
        interp = _interp_values(prev, nxt, alpha)
        interp_map = {state.club_name: state for state in interp}
        priority = snapshot_priorities[period_index]
        prev_rank = _rank_with_tie_priority(prev.states, top_n, priority)
        next_rank = _rank_with_tie_priority(nxt.states, top_n, priority)
        top_states = [interp_map[name] for name in sorted(set(prev_rank) | set(next_rank)) if name in interp_map]

        header = (44, 48, WIDTH - 44, 324)
        draw.rounded_rectangle(header, radius=38, fill=(8, 20, 43, 184), outline=(255, 255, 255, 22), width=1)
        draw.text((74, 76), TITLE, font=title_font, fill="#f4f7fb")
        draw.text((76, 144), "European Cup + Champions League", font=subtitle_font, fill="#aac1e0")
        draw.rounded_rectangle((74, 188, 266, 274), radius=26, fill=(243, 196, 79, 255))
        year_bbox = draw.textbbox((0, 0), str(nxt.year), font=year_font)
        draw.text(
            (170 - (year_bbox[2] - year_bbox[0]) // 2, 200),
            str(nxt.year),
            font=year_font,
            fill="#0d223e",
        )
        if nxt.final_runner_up:
            draw.text((296, 214), f"Final vs {nxt.final_runner_up}", font=subtitle_font, fill="#edf3ff")
        if nxt.final_score_line:
            score_font = score_font_cache.get(nxt.final_score_line)
            if score_font is None:
                score_font = _fit_font_size(draw, nxt.final_score_line, 330, 28, 16, bold=True)
                score_font_cache[nxt.final_score_line] = score_font
            score_rect = (WIDTH - 404, 196, WIDTH - 72, 274)
            draw.rounded_rectangle(score_rect, radius=22, fill=(243, 196, 79, 255))
            score_bbox = draw.textbbox((0, 0), nxt.final_score_line, font=score_font)
            draw.text(
                (
                    score_rect[0] + (score_rect[2] - score_rect[0] - (score_bbox[2] - score_bbox[0])) // 2,
                    score_rect[1] + 22,
                ),
                nxt.final_score_line,
                font=score_font,
                fill="#0d223e",
            )

        for lane_index in range(top_n):
            lane_y = base_y + lane_index * (row_h + row_gap)
            rank_top = lane_y + max(2, (row_h - bar_height) // 2)
            rank_bottom = rank_top + bar_height
            draw.rounded_rectangle((rank_x, rank_top, rank_x + 62, rank_bottom), radius=18, fill=(243, 196, 79, 255))
            rank_text = str(lane_index + 1)
            rank_bbox = draw.textbbox((0, 0), rank_text, font=rank_font)
            draw.text(
                (rank_x + 31 - (rank_bbox[2] - rank_bbox[0]) // 2, rank_top + max(8, (bar_height - (rank_bbox[3] - rank_bbox[1])) // 2 - 1)),
                rank_text,
                font=rank_font,
                fill="#10233f",
            )

        items: list[tuple[int, float, object, int]] = []
        for state in top_states:
            prev_idx = prev_rank.get(state.club_name, top_n + 1)
            next_idx = next_rank.get(state.club_name, top_n + 1)
            y_idx = _continuous_rank_position(float(prev_idx), float(next_idx), alpha)
            y = base_y + y_idx * (row_h + row_gap)
            bar_w = max(90, int((state.titles / axis_cap) * bar_max_w))
            moving_up = 1 if next_idx < prev_idx else 0
            items.append((moving_up, y, state, bar_w))
        items.sort(key=lambda item: (item[0], item[1]))

        for _, y, state, bar_w in items:
            y0 = int(y)
            color = CLUB_COLORS.get(state.club_name, "#39c0ff")
            text_color = _text_on(color)

            bar_top = y0 + max(2, (row_h - bar_height) // 2)
            bar_bottom = bar_top + bar_height
            bar_rect = (bar_left, bar_top, bar_left + bar_w, bar_bottom)
            shadow_rect = (bar_left + 6, bar_top + 7, bar_left + bar_w + 6, bar_bottom + 7)
            radius = max(16, bar_height // 2)
            draw.rounded_rectangle(shadow_rect, radius=radius, fill=(0, 0, 0, 70))
            draw.rounded_rectangle(bar_rect, radius=radius, fill=color)

            logo = logo_cache.get(_slugify(state.club_name))
            _paste_logo(frame, logo, bar_left + 8, bar_top + max(2, (bar_height - logo_box) // 2), logo_box)

            label_x = bar_left + logo_box + 26
            label_max_width = max(0, bar_w - (label_x - bar_left) - 24)
            club_name = _truncate_text_to_width(draw, state.club_name, name_font, label_max_width)
            draw.text((label_x, bar_top + max(8, (bar_height - 28) // 2)), club_name, font=name_font, fill=text_color)

            value_text = str(int(round(state.titles)))
            value_bbox = draw.textbbox((0, 0), value_text, font=value_font)
            value_x = min(bar_left + bar_w + 18, WIDTH - 58 - (value_bbox[2] - value_bbox[0]))
            draw.text((value_x, bar_top + max(8, (bar_height - 30) // 2)), value_text, font=value_font, fill="#f4f7fb")

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
    parser = argparse.ArgumentParser(description="Generate a premium MoviePy Shorts Champions League bar chart race.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--logos-dir", type=Path, default=DEFAULT_LOGOS_DIR)
    parser.add_argument("--audio", type=Path, default=DEFAULT_AUDIO)
    parser.add_argument("--duration", type=float, default=TOTAL_DURATION)
    parser.add_argument("--fps", type=int, default=FPS)
    parser.add_argument("--top-n", type=int, default=TOP_N)
    parser.add_argument("--start-year", type=int, default=None)
    parser.add_argument("--end-year", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = render_video(
        input_csv=args.input,
        output_path=args.output,
        logos_dir=args.logos_dir,
        audio_path=args.audio,
        duration=args.duration,
        fps=args.fps,
        top_n=args.top_n,
        start_year=args.start_year,
        end_year=args.end_year,
    )
    print(f"[video_generator] premium UCL Shorts race generated -> {output}")


if __name__ == "__main__":
    main()
