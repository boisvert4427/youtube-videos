from __future__ import annotations

import argparse
import math
import textwrap
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from moviepy import VideoClip
from PIL import Image, ImageDraw, ImageFilter, ImageOps

from video_generator.generate_ucl_barchart_race_moviepy import (
    CLUB_COLORS,
    DEFAULT_AUDIO,
    DEFAULT_INPUT,
    DEFAULT_LOGOS_DIR,
    DEFAULT_TROPHY_PATH,
    FPS as DEFAULT_FPS,
    _build_logo_cache,
    _fit_font_size,
    _load_font,
    _slugify,
    build_audio_track,
    load_snapshots,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "champions_league_winners_shorts_2000_2024_45s.mp4"

WIDTH = 1080
HEIGHT = 1920
FPS = 30
TOTAL_DURATION = 45.0
START_YEAR = 2000
END_YEAR = 2024

HOOK_TEXT = "Can you name every Champions League winner since 2000?"
OUTRO_TEXT = "Which winner surprised you the most?"
SUBTITLE_TEXT = "Football history in 45 seconds"


@dataclass(frozen=True)
class WinnerEntry:
    year: int
    club_name: str
    final_runner_up: str
    final_score_line: str


def _hex_to_rgb(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    return tuple(int(value[i : i + 2], 16) for i in (0, 2, 4))


def _ease_out_cubic(value: float) -> float:
    value = min(max(value, 0.0), 1.0)
    return 1.0 - (1.0 - value) ** 3


def _ease_in_out(value: float) -> float:
    value = min(max(value, 0.0), 1.0)
    return value * value * (3.0 - 2.0 * value)


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return min(max(value, lo), hi)


def _wrap_text_to_width(draw: ImageDraw.ImageDraw, text: str, font, max_width: int, max_lines: int) -> list[str]:
    words = text.split()
    if not words:
        return []
    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        if draw.textbbox((0, 0), candidate, font=font)[2] <= max_width:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    if len(lines) <= max_lines:
        return lines

    wrapped = textwrap.wrap(text, width=max(10, len(text) // max_lines))
    lines = []
    for chunk in wrapped:
        if draw.textbbox((0, 0), chunk, font=font)[2] <= max_width:
            lines.append(chunk)
        else:
            words = chunk.split()
            current = words[0]
            for word in words[1:]:
                candidate = f"{current} {word}"
                if draw.textbbox((0, 0), candidate, font=font)[2] <= max_width:
                    current = candidate
                else:
                    lines.append(current)
                    current = word
            lines.append(current)
    return lines[:max_lines]


def _make_background(trophy_path: Path) -> Image.Image:
    xx = np.linspace(0, 1, WIDTH, dtype=np.float32)
    yy = np.linspace(0, 1, HEIGHT, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(xx, yy)
    deep_navy = np.array([5, 12, 28], dtype=np.float32)
    stadium_blue = np.array([9, 36, 84], dtype=np.float32)
    electric = np.array([45, 138, 255], dtype=np.float32)
    white_glow = np.array([220, 241, 255], dtype=np.float32)

    mix = np.clip(0.72 * grid_y + 0.18 * (1.0 - grid_x), 0, 1)
    top_glow = np.exp(-(((grid_x - 0.5) / 0.28) ** 2 + ((grid_y - 0.18) / 0.14) ** 2))
    side_glow = np.exp(-(((grid_x - 0.1) / 0.18) ** 2 + ((grid_y - 0.52) / 0.28) ** 2))
    side_glow += np.exp(-(((grid_x - 0.9) / 0.18) ** 2 + ((grid_y - 0.52) / 0.28) ** 2))

    img = np.clip(
        deep_navy[None, None, :] * (1.0 - mix[..., None])
        + stadium_blue[None, None, :] * (0.78 * mix[..., None])
        + electric[None, None, :] * (0.25 * top_glow[..., None])
        + white_glow[None, None, :] * (0.08 * side_glow[..., None]),
        0,
        255,
    ).astype(np.uint8)
    frame = Image.fromarray(img, mode="RGB").convert("RGBA")

    overlay = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    o = ImageDraw.Draw(overlay, "RGBA")
    for i in range(7):
        alpha = max(0, 90 - i * 12)
        o.ellipse((140 - i * 22, 70 - i * 20, WIDTH - 140 + i * 22, 550 + i * 30), fill=(110, 200, 255, alpha))
    for y in range(1080, 1600, 34):
        o.line((60, y, WIDTH - 60, y), fill=(255, 255, 255, 8), width=2)
    for x in range(80, WIDTH, 72):
        o.ellipse((x, 430, x + 6, 436), fill=(180, 220, 255, 60))
        o.ellipse((x, 1470, x + 5, 1475), fill=(180, 220, 255, 36))
    overlay = overlay.filter(ImageFilter.GaussianBlur(radius=8))
    frame.alpha_composite(overlay)

    if trophy_path.exists():
        try:
            trophy = Image.open(trophy_path).convert("RGBA")
            trophy = ImageOps.contain(trophy, (220, 220), method=Image.Resampling.LANCZOS)
            trophy_glow = Image.new("RGBA", trophy.size, (0, 0, 0, 0))
            glow_draw = ImageDraw.Draw(trophy_glow, "RGBA")
            glow_draw.ellipse((-30, -20, trophy.width + 30, trophy.height + 40), fill=(80, 170, 255, 105))
            glow_draw.ellipse((40, 20, trophy.width - 40, trophy.height), fill=(255, 255, 255, 42))
            trophy_glow = trophy_glow.filter(ImageFilter.GaussianBlur(radius=26))
            tx = (WIDTH - trophy.width) // 2
            ty = 106
            frame.alpha_composite(trophy_glow, (tx, ty))
            trophy.putalpha(150)
            frame.alpha_composite(trophy, (tx, ty))
        except Exception:
            pass

    vignette = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    v = ImageDraw.Draw(vignette, "RGBA")
    v.rounded_rectangle((-40, -40, WIDTH + 40, HEIGHT + 40), radius=120, outline=(0, 0, 0, 180), width=120)
    vignette = vignette.filter(ImageFilter.GaussianBlur(radius=28))
    frame.alpha_composite(vignette)
    return frame


def _build_winners(input_csv: Path, start_year: int, end_year: int) -> list[WinnerEntry]:
    winners: list[WinnerEntry] = []
    for snapshot in load_snapshots(input_csv):
        if snapshot.year < start_year or snapshot.year > end_year:
            continue
        winner = next((state for state in snapshot.states if state.won_this_year), None)
        if winner is None:
            continue
        winners.append(
            WinnerEntry(
                year=snapshot.year,
                club_name=winner.club_name,
                final_runner_up=snapshot.final_runner_up,
                final_score_line=snapshot.final_score_line,
            )
        )
    if not winners:
        raise RuntimeError("No Champions League winners found for the requested range.")
    return winners


def _build_logo_tiles(entries: list[WinnerEntry], logos_dir: Path) -> dict[str, Image.Image]:
    cache_raw = _build_logo_cache(logos_dir)
    tiles: dict[str, Image.Image] = {}
    for entry in entries:
        slug = _slugify(entry.club_name)
        if slug in tiles:
            continue
        logo_arr = cache_raw.get(slug)
        tile = Image.new("RGBA", (220, 220), (0, 0, 0, 0))
        draw = ImageDraw.Draw(tile, "RGBA")
        accent = CLUB_COLORS.get(entry.club_name, "#2d7fff")
        draw.rounded_rectangle((12, 12, 208, 208), radius=48, fill=(7, 19, 42, 214), outline=(160, 220, 255, 90), width=2)
        draw.rounded_rectangle((26, 26, 194, 194), radius=38, fill=(255, 255, 255, 18))
        draw.ellipse((48, 48, 172, 172), fill=(*_hex_to_rgb(accent), 48))
        if logo_arr is not None:
            logo_img = Image.fromarray(logo_arr).convert("RGBA")
            logo_img = ImageOps.contain(logo_img, (112, 112), method=Image.Resampling.LANCZOS)
            plate = Image.new("RGBA", (134, 134), (0, 0, 0, 0))
            plate_draw = ImageDraw.Draw(plate, "RGBA")
            plate_draw.rounded_rectangle((0, 0, 133, 133), radius=32, fill=(255, 255, 255, 235))
            plate.alpha_composite(logo_img, ((134 - logo_img.width) // 2, (134 - logo_img.height) // 2))
            tile.alpha_composite(plate, (43, 43))
        tiles[slug] = tile
    return tiles


def _draw_glow_text(
    frame: Image.Image,
    position: tuple[int, int],
    text: str,
    font,
    fill: tuple[int, int, int],
    glow_color: tuple[int, int, int],
    anchor: str = "la",
) -> None:
    glow = Image.new("RGBA", frame.size, (0, 0, 0, 0))
    gd = ImageDraw.Draw(glow, "RGBA")
    for radius, alpha in ((0, 120), (2, 90), (6, 55), (12, 24)):
        gd.text(position, text, font=font, fill=(*glow_color, alpha), anchor=anchor, stroke_width=radius // 3)
    glow = glow.filter(ImageFilter.GaussianBlur(radius=8))
    frame.alpha_composite(glow)
    draw = ImageDraw.Draw(frame, "RGBA")
    draw.text(position, text, font=font, fill=(*fill, 255), anchor=anchor)


def render_video(
    input_csv: Path,
    output_path: Path,
    logos_dir: Path,
    trophy_path: Path,
    audio_path: Path,
    duration: float,
    fps: int,
    start_year: int,
    end_year: int,
) -> Path:
    entries = _build_winners(input_csv, start_year, end_year)
    logo_tiles = _build_logo_tiles(entries, logos_dir)
    background = _make_background(trophy_path)

    hook_duration = 4.0
    outro_duration = 3.8
    timeline_duration = max(6.0, duration - hook_duration - outro_duration)
    per_entry = timeline_duration / len(entries)

    title_font = _load_font(60, bold=True)
    bg_draw = ImageDraw.Draw(background.copy())
    hook_font = _fit_font_size(bg_draw, HOOK_TEXT, 760, 60, 30, bold=True)
    subtitle_font = _load_font(24, bold=False)
    year_font = _load_font(96, bold=True)
    club_font = _fit_font_size(ImageDraw.Draw(background.copy()), "Borussia Dortmund", 700, 52, 28, bold=True)
    score_font = _fit_font_size(ImageDraw.Draw(background.copy()), "Bayern Munich 1-1 (5-4 pens.) Valencia", 820, 31, 18, bold=True)
    outro_font = _fit_font_size(ImageDraw.Draw(background.copy()), OUTRO_TEXT, 900, 72, 36, bold=True)
    small_year_font = _load_font(26, bold=True)

    def make_frame(t: float) -> np.ndarray:
        frame = background.copy()
        draw = ImageDraw.Draw(frame, "RGBA")

        # Persistent title
        draw.text((48, 40), "CHAMPIONS LEAGUE", font=title_font, fill=(245, 248, 252, 255))
        draw.text((50, 102), "Football history", font=subtitle_font, fill=(153, 191, 232, 255))

        if t < hook_duration:
            phase = _ease_out_cubic(t / hook_duration)
            panel_y = 560 - int((1.0 - phase) * 110)
            panel = (66, panel_y, WIDTH - 66, panel_y + 500)
            draw.rounded_rectangle(panel, radius=46, fill=(6, 18, 42, 210), outline=(130, 205, 255, 90), width=2)
            draw.rounded_rectangle((panel[0] + 18, panel[1] + 18, panel[2] - 18, panel[3] - 18), radius=34, outline=(255, 255, 255, 22), width=1)
            _draw_glow_text(frame, (WIDTH // 2, panel_y + 140), "QUIZ", _load_font(36, bold=True), (128, 208, 255), (68, 155, 255), anchor="ma")
            hook_lines = _wrap_text_to_width(draw, HOOK_TEXT, hook_font, panel[2] - panel[0] - 120, 3)
            line_height = int(hook_font.size * 1.18)
            total_height = len(hook_lines) * line_height
            line_y = panel_y + 224 - total_height // 2
            for line in hook_lines:
                _draw_glow_text(frame, (WIDTH // 2, line_y), line, hook_font, (245, 249, 255), (67, 161, 255), anchor="ma")
                line_y += line_height
            draw.text((WIDTH // 2, panel_y + 402), SUBTITLE_TEXT, font=subtitle_font, fill=(188, 219, 249, int(255 * phase)), anchor="ma")
            return np.array(frame.convert("RGB"))

        if t >= duration - outro_duration:
            local = _clamp((t - (duration - outro_duration)) / outro_duration)
            phase = _ease_out_cubic(local)
            panel_top = 640
            draw.rounded_rectangle((72, panel_top, WIDTH - 72, panel_top + 520), radius=52, fill=(5, 18, 43, 220), outline=(130, 205, 255, 90), width=2)
            _draw_glow_text(frame, (WIDTH // 2, panel_top + 170), "END OF THE ERA", _load_font(34, bold=True), (130, 208, 255), (70, 158, 255), anchor="ma")
            _draw_glow_text(frame, (WIDTH // 2, panel_top + 292), OUTRO_TEXT, outro_font, (247, 250, 255), (74, 168, 255), anchor="mm")
            draw.text((WIDTH // 2, panel_top + 426), "Comment your answer", font=subtitle_font, fill=(194, 224, 248, int(255 * phase)), anchor="ma")
            return np.array(frame.convert("RGB"))

        timeline_t = t - hook_duration
        active_index = min(int(timeline_t / per_entry), len(entries) - 1)
        local_t = (timeline_t - active_index * per_entry) / per_entry
        reveal = _ease_out_cubic(_clamp(local_t / 0.42))
        pulse = 0.5 + 0.5 * math.sin(local_t * math.pi)

        entry = entries[active_index]
        accent_rgb = _hex_to_rgb(CLUB_COLORS.get(entry.club_name, "#2d7fff"))
        timeline_left = 92
        timeline_right = WIDTH - 92
        timeline_y = 1818

        draw.line((timeline_left, timeline_y, timeline_right, timeline_y), fill=(110, 175, 230, 90), width=4)
        spacing = (timeline_right - timeline_left) / max(1, len(entries) - 1)

        for idx, item in enumerate(entries):
            x = int(timeline_left + idx * spacing)
            shown = idx < active_index
            active = idx == active_index
            if shown:
                r = 14
                draw.ellipse((x - r, timeline_y - r, x + r, timeline_y + r), fill=(80, 178, 255, 255))
                draw.ellipse((x - r - 8, timeline_y - r - 8, x + r + 8, timeline_y + r + 8), outline=(130, 212, 255, 80), width=3)
            elif active:
                r = int(16 + 10 * pulse)
                draw.ellipse((x - r, timeline_y - r, x + r, timeline_y + r), fill=(255, 255, 255, 255))
                draw.ellipse((x - r - 16, timeline_y - r - 16, x + r + 16, timeline_y + r + 16), outline=(*accent_rgb, 120), width=5)
            else:
                r = 10
                draw.ellipse((x - r, timeline_y - r, x + r, timeline_y + r), fill=(14, 42, 77, 255), outline=(120, 170, 215, 80), width=2)

            if idx % 2 == 0 or active:
                year_alpha = 255 if idx <= active_index else 105
                draw.text((x, timeline_y + 24), str(item.year), font=small_year_font, fill=(210, 228, 244, year_alpha), anchor="ma")

        year_card = (48, 190, 290, 318)
        draw.rounded_rectangle(year_card, radius=30, fill=(6, 18, 42, 220), outline=(*accent_rgb, 160), width=2)
        _draw_glow_text(frame, ((year_card[0] + year_card[2]) // 2, 253), str(entry.year), year_font, (245, 249, 255), accent_rgb, anchor="mm")

        hero_scale = 0.62 + 0.12 * reveal
        base_tile = logo_tiles.get(_slugify(entry.club_name))
        if base_tile is not None:
            tile_w = int(base_tile.width * hero_scale)
            tile_h = int(base_tile.height * hero_scale)
            hero = base_tile.resize((tile_w, tile_h), Image.Resampling.LANCZOS)
            hero_glow = Image.new("RGBA", (tile_w + 80, tile_h + 80), (0, 0, 0, 0))
            hg = ImageDraw.Draw(hero_glow, "RGBA")
            hg.rounded_rectangle((28, 28, tile_w + 52, tile_h + 52), radius=56, fill=(*accent_rgb, int(70 + 55 * pulse)))
            hero_glow = hero_glow.filter(ImageFilter.GaussianBlur(radius=22))
            hx = (WIDTH - tile_w) // 2
            hy = 360
            frame.alpha_composite(hero_glow, (hx - 40, hy - 40))
            frame.alpha_composite(hero, (hx, hy))

        club_card = (84, 590, WIDTH - 84, 730)
        draw.rounded_rectangle(club_card, radius=36, fill=(7, 19, 42, 224), outline=(180, 225, 255, 54), width=2)
        _draw_glow_text(frame, (WIDTH // 2, 640), entry.club_name.upper(), club_font, (247, 250, 255), accent_rgb, anchor="ma")
        if entry.final_score_line:
            draw.text((WIDTH // 2, 684), entry.final_score_line, font=score_font, fill=(198, 224, 248, 255), anchor="ma")

        # Revealed winner wall
        grid_top = 770
        cols = 5
        mini_size = 100
        gap_x = 18
        gap_y = 10
        start_x = (WIDTH - (cols * mini_size + (cols - 1) * gap_x)) // 2
        for idx in range(active_index + 1):
            item = entries[idx]
            tile = logo_tiles.get(_slugify(item.club_name))
            if tile is None:
                continue
            row = idx // cols
            col = idx % cols
            tx = start_x + col * (mini_size + gap_x)
            ty = grid_top + row * (mini_size + 42 + gap_y)
            scale = 1.0
            alpha = 255
            if idx == active_index:
                scale = 0.78 + 0.22 * reveal
                alpha = int(170 + 85 * reveal)
            mini = tile.resize((int(mini_size * scale), int(mini_size * scale)), Image.Resampling.LANCZOS)
            px = tx + (mini_size - mini.width) // 2
            py = ty + (mini_size - mini.height) // 2
            if idx == active_index:
                glow = Image.new("RGBA", (mini.width + 36, mini.height + 36), (0, 0, 0, 0))
                gd = ImageDraw.Draw(glow, "RGBA")
                gd.rounded_rectangle((10, 10, glow.width - 10, glow.height - 10), radius=28, fill=(*accent_rgb, 95))
                glow = glow.filter(ImageFilter.GaussianBlur(radius=14))
                frame.alpha_composite(glow, (px - 18, py - 18))
            if alpha < 255:
                mini = mini.copy()
                mini.putalpha(alpha)
            frame.alpha_composite(mini, (px, py))
            draw.text((tx + mini_size // 2, ty + mini_size + 16), str(item.year), font=small_year_font, fill=(222, 235, 247, 255), anchor="ma")

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
    parser = argparse.ArgumentParser(description="Generate a vertical Champions League winners history Shorts video.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--logos-dir", type=Path, default=DEFAULT_LOGOS_DIR)
    parser.add_argument("--trophy-path", type=Path, default=DEFAULT_TROPHY_PATH)
    parser.add_argument("--audio", type=Path, default=DEFAULT_AUDIO)
    parser.add_argument("--duration", type=float, default=TOTAL_DURATION)
    parser.add_argument("--fps", type=int, default=FPS)
    parser.add_argument("--start-year", type=int, default=START_YEAR)
    parser.add_argument("--end-year", type=int, default=END_YEAR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = render_video(
        input_csv=args.input,
        output_path=args.output,
        logos_dir=args.logos_dir,
        trophy_path=args.trophy_path,
        audio_path=args.audio,
        duration=args.duration,
        fps=args.fps,
        start_year=args.start_year,
        end_year=args.end_year,
    )
    print(f"[video_generator] Champions League winners Shorts generated -> {output}")


if __name__ == "__main__":
    main()
