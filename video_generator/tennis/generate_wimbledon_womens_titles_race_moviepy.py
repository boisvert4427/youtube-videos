from __future__ import annotations

import argparse
import math
from pathlib import Path

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
from video_generator.cycling.generate_tour_de_france_stage_wins_race_moviepy import (
    DEFAULT_BAR_COLORS,
    PlayerState,
    Snapshot,
    _ascii_key,
    _build_flag_cache,
    _build_photo_cache,
    _continuous_rank_position,
    _ease_in_out,
    _interp_values,
    _mix_rgb,
    _parse_csv_rows,
    _phase_delay,
    _rank_with_tie_priority,
    _slugify,
    _text_on,
    _truncate_text_to_width,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_INPUT = PROJECT_ROOT / "data" / "processed" / "tennis" / "wimbledon_womens_titles_timeseries_1968_2025.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "tennis" / "wimbledon_womens_titles_race_landscape_1968_2025_4min_60fps.mp4"
DEFAULT_PREVIEW = PROJECT_ROOT / "data" / "processed" / "tennis" / "wimbledon_womens_titles_race_preview.png"
DEFAULT_BACKGROUND_IMAGE = Path(r"C:\Users\leona\Downloads\ChatGPT Image 3 juil. 2026, 21_25_24.png")
DEFAULT_WIMBLEDON_LOGO = Path(r"C:\Users\leona\Downloads\Logo_Wimbledon.svg.webp")
DEFAULT_PHOTOS_DIR = PROJECT_ROOT / "data" / "raw" / "player_photos"

WIDTH = 1920
HEIGHT = 1080
OUTPUT_WIDTH = 3840
OUTPUT_HEIGHT = 2160
TOP_N = 12
FPS = 60
TOTAL_DURATION = 240.0

TITLE = "WIMBLEDON WOMEN'S TITLES RACE"
SUBTITLE = "LADIES' SINGLES - CUMULATIVE WIMBLEDON TITLES SINCE THE OPEN ERA"

WIMBLEDON_COLORS = {
    "Martina Navratilova": "#0a5a37",
    "Serena Williams": "#5c2f84",
    "Steffi Graf": "#f0eadf",
    "Venus Williams": "#0d6a42",
    "Billie Jean King": "#d4af5b",
    "Chris Evert": "#efe8d8",
    "Evonne Goolagong Cawley": "#0c5637",
    "Petra Kvitova": "#6a3b98",
    "Ann Jones": "#e1d2a6",
    "Margaret Court": "#d8c48c",
    "Virginia Wade": "#ebe3d3",
    "Conchita Martinez": "#3f8f4f",
    "Martina Hingis": "#efe2c7",
    "Jana Novotna": "#c7b88a",
    "Lindsay Davenport": "#7044a8",
    "Maria Sharapova": "#eadbb5",
    "Amelie Mauresmo": "#0f6b43",
    "Marion Bartoli": "#5f3a88",
    "Garbine Muguruza": "#d9c27c",
    "Angelique Kerber": "#f4ead8",
    "Simona Halep": "#0b6642",
    "Ashleigh Barty": "#d2aa53",
    "Elena Rybakina": "#663c96",
    "Marketa Vondrousova": "#e9dfc6",
    "Barbora Krejcikova": "#0d7047",
    "Iga Swiatek": "#cda348",
}


def _load_wimbledon_logo(logo_path: Path | None, size: int) -> Image.Image | None:
    if logo_path is None or not logo_path.exists():
        return None
    try:
        img = Image.open(logo_path).convert("RGBA")
        return ImageOps.fit(img, (size, size), method=Image.Resampling.LANCZOS, centering=(0.5, 0.5))
    except Exception:
        return None


def _draw_wimbledon_badge(frame: Image.Image, x: int, y: int, size: int, logo: Image.Image | None = None) -> None:
    draw = ImageDraw.Draw(frame, "RGBA")
    if logo is not None:
        frame.alpha_composite(logo, (x, y))
        return
    draw.ellipse((x, y, x + size, y + size), fill=(80, 38, 128, 240), outline=(245, 241, 232, 230), width=4)
    draw.ellipse((x + 7, y + 7, x + size - 7, y + size - 7), outline=(48, 148, 85, 220), width=6)
    draw.line((x + size * 0.36, y + size * 0.30, x + size * 0.52, y + size * 0.66), fill=(245, 241, 232, 210), width=4)
    draw.line((x + size * 0.64, y + size * 0.30, x + size * 0.48, y + size * 0.66), fill=(245, 241, 232, 210), width=4)
    draw.ellipse((x + size * 0.27, y + size * 0.26, x + size * 0.47, y + size * 0.46), outline=(245, 241, 232, 180), width=3)
    draw.ellipse((x + size * 0.53, y + size * 0.26, x + size * 0.73, y + size * 0.46), outline=(245, 241, 232, 180), width=3)


def _make_background(background_image: Path | None = None) -> Image.Image:
    if background_image is not None and background_image.exists():
        try:
            base = Image.open(background_image).convert("RGBA")
            base = ImageOps.fit(base, (WIDTH, HEIGHT), method=Image.Resampling.LANCZOS, centering=(0.5, 0.5))
            base = base.filter(ImageFilter.GaussianBlur(radius=0.55))
            veil = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
            vdraw = ImageDraw.Draw(veil, "RGBA")
            vdraw.rectangle((0, 0, WIDTH, HEIGHT), fill=(0, 0, 0, 44))
            vdraw.rectangle((1110, 0, WIDTH, HEIGHT), fill=(0, 0, 0, 88))
            vdraw.ellipse((820, -150, 1940, 850), fill=(18, 70, 38, 24))
            vdraw.ellipse((1230, 110, 1920, 860), fill=(0, 0, 0, 58))
            vdraw.ellipse((-240, 640, 620, 1220), fill=(0, 0, 0, 42))
            return Image.alpha_composite(base, veil)
        except Exception:
            pass

    xx = np.linspace(0, 1, WIDTH, dtype=np.float32)
    yy = np.linspace(0, 1, HEIGHT, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(xx, yy)
    dark = np.array([4, 16, 9], dtype=np.float32)
    green = np.array([12, 64, 36], dtype=np.float32)
    gold = np.array([214, 178, 94], dtype=np.float32)
    mix = np.clip(0.62 * grid_y + 0.16 * (1.0 - grid_x), 0, 1)
    glow = np.exp(-(((grid_x - 0.58) / 0.30) ** 2 + ((grid_y - 0.18) / 0.16) ** 2))
    pixels = np.clip(
        dark[None, None, :] * (1.0 - mix[..., None])
        + green[None, None, :] * (0.82 * mix[..., None])
        + gold[None, None, :] * (0.08 * glow[..., None]),
        0,
        255,
    ).astype(np.uint8)
    frame = Image.fromarray(pixels, mode="RGB").convert("RGBA")
    overlay = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")
    draw.rounded_rectangle((28, 20, WIDTH - 28, HEIGHT - 24), radius=38, outline=(214, 178, 94, 110), width=2)
    draw.line((0, HEIGHT - 30, WIDTH, HEIGHT - 30), fill=(243, 241, 232, 84), width=3)
    overlay = overlay.filter(ImageFilter.GaussianBlur(radius=2))
    frame.alpha_composite(overlay)
    return frame


def _build_stable_snapshot_priorities(snapshots) -> list[dict[str, int]]:
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


def _build_rider_color_map(snapshots) -> dict[str, str]:
    color_map: dict[str, str] = {}
    fallback_index = 0
    for snapshot in snapshots:
        for state in snapshot.states:
            if state.player_name in color_map:
                continue
            color_map[state.player_name] = WIMBLEDON_COLORS.get(
                state.player_name,
                DEFAULT_BAR_COLORS[fallback_index % len(DEFAULT_BAR_COLORS)],
            )
            fallback_index += 1
    return color_map


def render_video(
    input_csv: Path,
    output_path: Path,
    preview_path: Path | None,
    preview_time: float,
    flags_dir: Path,
    photos_dir: Path,
    background_image: Path | None,
    wimbledon_logo: Path | None,
    audio_path: Path,
    duration: float,
    fps: int,
    top_n: int,
) -> Path:
    snapshots = _parse_csv_rows(input_csv)
    if len(snapshots) < 2:
        raise RuntimeError("Not enough Wimbledon snapshots to render.")

    first_snapshot = snapshots[0]
    intro_snapshot = first_snapshot.__class__(
        ranking_date=f"{first_snapshot.year - 1}-12-31",
        year=first_snapshot.year,
        season_summary="Wimbledon titles|Open Era cumulative race|Premium broadcast template",
        states=[],
    )
    snapshots = [intro_snapshot, *snapshots]

    all_states = [state for snapshot in snapshots for state in snapshot.states]
    flag_cache = _build_flag_cache(all_states, flags_dir)
    photo_cache = _build_photo_cache(all_states, photos_dir, 58)
    logo_image = _load_wimbledon_logo(wimbledon_logo, 72)
    priorities = _build_stable_snapshot_priorities(snapshots)
    color_map = _build_rider_color_map(snapshots)
    background = _make_background(background_image)
    title_font = _load_font(56, bold=True)
    subtitle_font = _load_font(22, bold=False)
    name_font = _load_font(28, bold=True)
    value_font = _load_font(29, bold=True)
    rank_font = _load_font(23, bold=True)
    initials_font = _load_font(17, bold=True)
    year_font = _load_font(64, bold=True)
    label_font = _load_font(18, bold=True)
    summary_font_cache: dict[str, ImageFont.ImageFont] = {}

    periods = len(snapshots) - 1
    seconds_per_period = duration / periods
    axis_caps = [
        float(max((state.titles for state in snapshot.states[:top_n]), default=1.0))
        for snapshot in snapshots
    ]

    bar_left = 106
    bar_right = 1750
    bar_max_w = bar_right - bar_left
    base_y = 228
    row_h = 58
    row_gap = 6
    bar_h = 48
    rank_left = 36
    photo_box = 52
    flag_box = (44, 28)
    header_box = (34, 28, WIDTH - 34, 170)

    def render_frame(t: float) -> Image.Image:
        frame = background.copy()
        draw = ImageDraw.Draw(frame, "RGBA")

        if t >= duration:
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

        draw.rounded_rectangle(header_box, radius=36, fill=(3, 16, 30, 196), outline=(255, 255, 255, 20), width=2)
        logo_x = 40
        logo_y = header_box[1] + (header_box[3] - header_box[1] - 72) // 2
        _draw_wimbledon_badge(frame, logo_x, logo_y, 72, logo=logo_image)
        title_bbox = draw.textbbox((0, 0), TITLE, font=title_font)
        subtitle_bbox = draw.textbbox((0, 0), SUBTITLE, font=subtitle_font)
        title_x = logo_x + 72 + 26
        title_y = header_box[1] + (header_box[3] - header_box[1] - (title_bbox[3] - title_bbox[1])) // 2 - 8
        subtitle_x = title_x + 2
        subtitle_y = title_y + (title_bbox[3] - title_bbox[1]) + 8
        draw.text((title_x, title_y), TITLE, font=title_font, fill="#f5f8fb")
        draw.text((subtitle_x, subtitle_y), SUBTITLE, font=subtitle_font, fill="#c9d8cf")
        year_text = str(nxt.year)
        year_bbox = draw.textbbox((0, 0), year_text, font=year_font)
        year_pad = 30
        year_box_w = (year_bbox[2] - year_bbox[0]) + year_pad * 2
        year_box_h = (year_bbox[3] - year_bbox[1]) + year_pad * 2
        year_box_right = WIDTH - 100
        year_box_bottom = HEIGHT - 46
        year_box = (
            year_box_right - year_box_w,
            year_box_bottom - year_box_h,
            year_box_right,
            year_box_bottom,
        )
        draw.rounded_rectangle(year_box, radius=24, fill=(243, 196, 79, 255), outline=(255, 235, 184, 170), width=2)
        year_text_x = year_box[0] + (year_box_w - (year_bbox[2] - year_bbox[0])) // 2 - year_bbox[0]
        year_text_y = year_box[1] + (year_box_h - (year_bbox[3] - year_bbox[1])) // 2 - year_bbox[1]
        draw.text(
            (year_text_x, year_text_y),
            year_text,
            font=year_font,
            fill="#10273a",
        )

        for tick in range(1, max_titles + 1):
            x = bar_left + int((tick / axis_cap) * bar_max_w)
            if x >= year_box[0] - 24:
                continue
            draw.line((x, header_box[3] + 20, x, HEIGHT - 88), fill=(0, 0, 0, 68), width=2)
            draw.text((x - 6, header_box[3] + 8), str(tick), font=label_font, fill=(214, 178, 94, 200))

        for rank_idx in range(top_n):
            y0 = base_y + rank_idx * (row_h + row_gap)
            y1 = y0 + bar_h
            draw.rounded_rectangle((rank_left + 2, y0 + 4, rank_left + 52, y1 + 4), radius=14, fill=(0, 0, 0, 72))
            draw.rounded_rectangle((rank_left, y0 + 2, rank_left + 50, y1 + 2), radius=14, fill=(9, 44, 26, 255), outline=(214, 178, 94, 220), width=2)
            rank_text = str(rank_idx + 1)
            bbox = draw.textbbox((0, 0), rank_text, font=rank_font)
            draw.text((rank_left + 25 - (bbox[2] - bbox[0]) // 2, y0 + 9), rank_text, font=rank_font, fill="#f3d27a")

        items: list[tuple[int, float, object, int]] = []
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
            bar_w = max(120, int((state.titles / axis_cap) * bar_max_w))
            moving_up = 1 if next_idx < prev_idx else 0
            items.append((moving_up, y, state, bar_w))
        items.sort(key=lambda item: (item[0], item[1]))

        for _, y, state, bar_w in items:
            y0 = int(y)
            y1 = y0 + row_h
            color = color_map.get(state.player_name, "#3dbdf5")
            text_color = _text_on(color)
            outline = _mix_rgb(color, (255, 255, 255), 0.18)
            highlight = _mix_rgb(color, (255, 255, 255), 0.22)
            shadow = _mix_rgb(color, (0, 0, 0), 0.20)

            draw.rounded_rectangle((bar_left + 4, y0 + 6, bar_left + bar_w + 4, y1 + 6), radius=24, fill=(0, 0, 0, 84))
            draw.rounded_rectangle((bar_left, y0, bar_left + bar_w, y1), radius=24, fill=color, outline=outline, width=2)
            draw.rounded_rectangle((bar_left + 10, y0 + 8, bar_left + max(90, int(bar_w * 0.62)), y0 + 18), radius=8, fill=(*highlight, 52))
            draw.line((bar_left + 20, y1 - 8, bar_left + max(40, int(bar_w * 0.72)), y1 - 8), fill=(*shadow, 90), width=3)

            photo = photo_cache.get(state.player_name)
            avatar_x = bar_left + 6
            avatar_y = y0 + (row_h - photo_box) // 2
            if photo is not None:
                draw.ellipse((avatar_x - 3, avatar_y - 3, avatar_x + photo_box + 3, avatar_y + photo_box + 3), fill=(255, 255, 255, 235))
                frame.alpha_composite(photo, (avatar_x, avatar_y))
            else:
                draw.rounded_rectangle((avatar_x, avatar_y, avatar_x + photo_box, avatar_y + photo_box), radius=20, fill=(255, 255, 255, 220))
                initials = "".join(part[0] for part in state.player_name.split()[:2]).upper()[:2]
                ibox = draw.textbbox((0, 0), initials, font=initials_font)
                draw.text(
                    (avatar_x + photo_box // 2 - (ibox[2] - ibox[0]) // 2, avatar_y + photo_box // 2 - (ibox[3] - ibox[1]) // 2 - 1),
                    initials,
                    font=initials_font,
                    fill="#10233f",
                )

            label_x = bar_left + photo_box + 30
            flag = flag_cache.get(state.country_code)
            if flag is not None:
                fx = label_x
                fy = y0 + (row_h - flag.height) // 2
                draw.rounded_rectangle((fx - 4, fy - 4, fx + flag.width + 4, fy + flag.height + 4), radius=8, fill=(255, 255, 255, 228))
                frame.alpha_composite(flag, (fx, fy))
                label_x = fx + flag.width + 30

            label_max_width = max(0, bar_w - (label_x - bar_left) - 24)
            player_name = _truncate_text_to_width(draw, state.player_name, name_font, label_max_width)
            draw.text((label_x, y0 + (row_h - 28) // 2 - 1), player_name, font=name_font, fill=text_color)

            value_text = str(int(round(state.titles)))
            vbox = draw.textbbox((0, 0), value_text, font=value_font)
            value_x = min(bar_left + bar_w + 18, WIDTH - 64 - (vbox[2] - vbox[0]))
            draw.text((value_x, y0 + (row_h - 28) // 2 - 1), value_text, font=value_font, fill="#f4f7fb")

        return np.array(frame.convert("RGB"))

    if preview_path is not None:
        preview_path.parent.mkdir(parents=True, exist_ok=True)
        preview = Image.fromarray(render_frame(preview_time))
        if preview.width < OUTPUT_WIDTH or preview.height < OUTPUT_HEIGHT:
            preview = preview.resize((OUTPUT_WIDTH, OUTPUT_HEIGHT), Image.Resampling.LANCZOS)
        preview.save(preview_path)
        return preview_path

    clip = VideoClip(render_frame, duration=duration)
    audio_clip, keep_alive = build_audio_track(audio_path, duration)
    clip = clip.resized((OUTPUT_WIDTH, OUTPUT_HEIGHT)).with_audio(audio_clip)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    clip.write_videofile(
        str(output_path),
        fps=fps,
        codec="libx264",
        audio_codec="aac",
        ffmpeg_params=["-crf", "16", "-preset", "slow", "-pix_fmt", "yuv420p"],
    )
    clip.close()
    audio_clip.close()
    for item in keep_alive:
        close = getattr(item, "close", None)
        if callable(close):
            close()
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a Wimbledon titles race video.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--preview", type=Path, default=None)
    parser.add_argument("--preview-time", type=float, default=120.0)
    parser.add_argument("--flags-dir", type=Path, default=DEFAULT_FLAGS_DIR)
    parser.add_argument("--photos-dir", type=Path, default=DEFAULT_PHOTOS_DIR)
    parser.add_argument("--background-image", type=Path, default=DEFAULT_BACKGROUND_IMAGE)
    parser.add_argument("--wimbledon-logo", type=Path, default=DEFAULT_WIMBLEDON_LOGO)
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
        preview_path=args.preview,
        preview_time=args.preview_time,
        flags_dir=args.flags_dir,
        photos_dir=args.photos_dir,
        background_image=args.background_image,
        wimbledon_logo=args.wimbledon_logo,
        audio_path=args.audio,
        duration=args.duration,
        fps=args.fps,
        top_n=args.top_n,
    )
    if args.preview is not None:
        print(f"[video_generator] Wimbledon titles preview generated -> {output}")
    else:
        print(f"[video_generator] Wimbledon titles race generated -> {output}")


if __name__ == "__main__":
    main()
