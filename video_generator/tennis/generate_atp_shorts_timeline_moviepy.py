from __future__ import annotations

import argparse
import math
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from moviepy import AudioFileClip, CompositeAudioClip, VideoClip
from moviepy.audio.fx import AudioFadeIn, AudioFadeOut
from PIL import Image, ImageDraw, ImageFont, ImageOps

from video_generator.generate_atp_vertical_timeline_moviepy import load_entries


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_INPUT = PROJECT_ROOT / "data" / "processed" / "atp_vertical_timeline_indian_wells_winners_1976_2025.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "atp_shorts_timeline_indian_wells_1976_2025.mp4"
DEFAULT_PHOTOS_DIR = PROJECT_ROOT / "data" / "raw" / "player_photos"
DEFAULT_AUDIO = PROJECT_ROOT / "data" / "raw" / "audio" / "audio.mp3"

WIDTH = 1080
HEIGHT = 1920
TOTAL_DURATION = 75.0
HOLD_START = 2.5
HOLD_END = 4.0
FPS = 60
FINAL_AUDIO_FADE_OUT = 6.0
LOOP_CROSSFADE = 4.0


@dataclass(frozen=True)
class Layout:
    outer_pad: int = 42
    top_pad: int = 44
    year_h: int = 88
    photo_h: int = 980
    info_h: int = 308
    block_gap: int = 22
    inner_pad: int = 22


def _load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf",
    ]
    for p in candidates:
        path = Path(p)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def _fit_font(draw: ImageDraw.ImageDraw, text: str, max_width: int, max_size: int, min_size: int, bold: bool = True):
    size = max_size
    while size >= min_size:
        font = _load_font(size=size, bold=bold)
        bbox = draw.textbbox((0, 0), text, font=font)
        if bbox[2] - bbox[0] <= max_width:
            return font
        size -= 1
    return _load_font(size=min_size, bold=bold)


def _draw_center_text(draw: ImageDraw.ImageDraw, rect: tuple[int, int, int, int], text: str, font, fill: str) -> None:
    x0, y0, x1, y1 = rect
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    tx = x0 + (x1 - x0 - tw) // 2
    ty = y0 + (y1 - y0 - th) // 2
    draw.text((tx, ty), text, font=font, fill=fill)


def _draw_text(draw: ImageDraw.ImageDraw, x: int, y: int, text: str, font, fill: str, anchor: str = "la") -> None:
    draw.text((x, y), text, font=font, fill=fill, anchor=anchor)


def _truncate_to_width(draw: ImageDraw.ImageDraw, text: str, font, max_width: int) -> str:
    if draw.textbbox((0, 0), text, font=font)[2] <= max_width:
        return text
    ell = "..."
    low = 0
    high = len(text)
    best = ell
    while low <= high:
        mid = (low + high) // 2
        candidate = text[:mid].rstrip() + ell
        w = draw.textbbox((0, 0), candidate, font=font)[2]
        if w <= max_width:
            best = candidate
            low = mid + 1
        else:
            high = mid - 1
    return best


def _wrap_text(draw: ImageDraw.ImageDraw, text: str, font, max_width: int, max_lines: int) -> list[str]:
    words = [word for word in text.split() if word]
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
            if len(lines) == max_lines - 1:
                break
    if len(lines) < max_lines:
        remaining_words = words[len(" ".join(lines + [current]).split()):]
        if remaining_words:
            tail = " ".join([current] + remaining_words)
            current = _truncate_to_width(draw, tail, font, max_width)
        lines.append(current)
    return lines[:max_lines]


def _draw_vignette(image: Image.Image, rect: tuple[int, int, int, int], strength: float = 0.55) -> None:
    x0, y0, x1, y1 = rect
    w = max(1, x1 - x0)
    h = max(1, y1 - y0)
    xx = np.linspace(-1.0, 1.0, w, dtype=np.float32)
    yy = np.linspace(-1.0, 1.0, h, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(xx, yy)
    radius = np.sqrt(grid_x**2 + grid_y**2)
    alpha = np.clip((radius - 0.18) / 0.88, 0, 1) ** 1.8
    overlay = np.zeros((h, w, 4), dtype=np.uint8)
    overlay[..., 3] = np.clip(alpha * 255 * strength, 0, 255).astype(np.uint8)
    image.alpha_composite(Image.fromarray(overlay, mode="RGBA"), (x0, y0))


def _slugify(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", normalized.strip().lower())
    return cleaned.strip("_")


def _resolve_player_image(image_path: str, player_name: str, photos_dir: Path) -> Path | None:
    if image_path:
        direct = PROJECT_ROOT / image_path
        if direct.exists():
            return direct
    slug = _slugify(player_name)
    for ext in (".jpg", ".jpeg", ".png", ".webp"):
        candidate = photos_dir / f"{slug}{ext}"
        if candidate.exists():
            return candidate
    return None


def _extract_result_parts(line: str) -> tuple[str, str, str]:
    text = line.strip()
    if not text:
        return ("", "", "")
    tokens = text.split()
    rnd = tokens[0] if tokens else ""
    valid_rounds = {"R128", "R64", "R32", "R16", "QF", "SF", "F"}
    if rnd not in valid_rounds:
        return ("", text, "")
    rest = " ".join(tokens[1:]).strip()
    if not rest:
        return (rnd, "-", "")

    parts = rest.split()
    score_parts: list[str] = []
    while parts:
        token = parts[-1]
        if any(ch.isdigit() for ch in token) or "-" in token:
            score_parts.insert(0, parts.pop())
        else:
            break
    opponent = " ".join(parts).strip() or "-"
    score = " ".join(score_parts).strip()
    return (rnd, opponent, score)


def _normalize_results(rows: list[str]) -> list[tuple[str, str, str]]:
    rounds = ["R128", "R64", "R32", "R16", "QF", "SF", "F"]
    by_round: dict[str, tuple[str, str]] = {}
    for row in rows:
        rnd, opp, score = _extract_result_parts(row)
        if rnd:
            by_round[rnd] = (opp, score)

    return [(rnd, *by_round.get(rnd, ("-", ""))) for rnd in rounds]


def _short_player_name(full_name: str) -> str:
    tokens = [t for t in full_name.strip().split() if t]
    if len(tokens) <= 1:
        return full_name.strip()
    particles = {"de", "del", "da", "di", "du", "la", "le", "van", "von", "st", "saint"}
    surname = [tokens[-1]]
    i = len(tokens) - 2
    while i >= 1 and tokens[i].lower() in particles:
        surname.insert(0, tokens[i])
        i -= 1
    return f"{tokens[0][0]}. {' '.join(surname)}"


def build_audio_track(audio_path: Path, duration: float):
    base = AudioFileClip(str(audio_path))
    if base.duration >= duration:
        return base.subclipped(0, duration).with_effects([AudioFadeOut(FINAL_AUDIO_FADE_OUT)]), [base]

    clips = []
    keep_alive = [base]
    step = max(0.1, base.duration - LOOP_CROSSFADE)
    loops = int(math.ceil(max(0.0, duration - LOOP_CROSSFADE) / step))
    for index in range(loops):
        segment = (
            base.with_start(index * step)
            .with_effects([AudioFadeIn(LOOP_CROSSFADE), AudioFadeOut(LOOP_CROSSFADE)])
        )
        clips.append(segment)
    mixed = CompositeAudioClip(clips).with_duration(duration).with_effects([AudioFadeOut(FINAL_AUDIO_FADE_OUT)])
    return mixed, keep_alive


def render_card(entry, photos_dir: Path) -> np.ndarray:
    layout = Layout()
    card = Image.new("RGBA", (WIDTH, HEIGHT), color="#d7a65f")

    xx = np.linspace(0, 1, WIDTH, dtype=np.float32)
    yy = np.linspace(0, 1, HEIGHT, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(xx, yy)
    cream = np.array([243, 236, 222], dtype=np.float32)
    sand = np.array([216, 181, 122], dtype=np.float32)
    ink = np.array([18, 45, 52], dtype=np.float32)
    mix = np.clip(0.58 * grid_y + 0.34 * grid_x, 0, 1)
    bg = np.clip(
        cream[None, None, :] * (1.0 - mix[..., None])
        + sand[None, None, :] * (0.82 * mix[..., None])
        + ink[None, None, :] * (0.20 * (1.0 - grid_y[..., None] * 0.5)),
        0,
        255,
    ).astype(np.uint8)
    card = Image.fromarray(np.dstack([bg, np.full((HEIGHT, WIDTH), 255, dtype=np.uint8)]), mode="RGBA")
    draw = ImageDraw.Draw(card, "RGBA")

    left = layout.outer_pad
    right = WIDTH - layout.outer_pad
    top = layout.top_pad
    bottom = HEIGHT - layout.outer_pad
    panel_shadow = (left + 8, top + 12, right + 8, bottom + 12)
    draw.rounded_rectangle(panel_shadow, radius=46, fill=(40, 34, 25, 52))
    draw.rounded_rectangle((left, top, right, bottom), radius=46, fill=(245, 241, 232, 224), outline="#f5e7c6", width=2)

    label_font = _load_font(24, bold=True)
    year_font = _fit_font(draw, str(entry.year), max_width=200, max_size=78, min_size=40)
    _draw_text(draw, left + 26, top + 26, "INDIAN WELLS", label_font, "#43616b")
    _draw_text(draw, right - 24, top + 34, str(entry.year), year_font, "#132e35", anchor="ra")

    y = top + 104
    photo_rect = (left + layout.inner_pad, y, right - layout.inner_pad, y + layout.photo_h)
    photo_shadow = (photo_rect[0], photo_rect[1] + 12, photo_rect[2], photo_rect[3] + 12)
    draw.rounded_rectangle(photo_shadow, radius=34, fill=(20, 19, 18, 55))
    source_path = _resolve_player_image(entry.image_path, entry.player_name, photos_dir)
    if source_path:
        try:
            photo = ImageOps.exif_transpose(Image.open(source_path)).convert("RGB")
            photo = ImageOps.fit(
                photo,
                (photo_rect[2] - photo_rect[0], photo_rect[3] - photo_rect[1]),
                method=Image.Resampling.LANCZOS,
                centering=(0.5, 0.16),
            )
            photo_rgba = photo.convert("RGBA")
            mask = Image.new("L", photo_rgba.size, 0)
            ImageDraw.Draw(mask).rounded_rectangle((0, 0, photo_rgba.width, photo_rgba.height), radius=34, fill=255)
            photo_rgba.putalpha(mask)
            card.alpha_composite(photo_rgba, (photo_rect[0], photo_rect[1]))
        except Exception:
            draw.rounded_rectangle(photo_rect, radius=34, fill="#d9d9d9")
    else:
        draw.rounded_rectangle(photo_rect, radius=34, fill="#d9d9d9")
    _draw_vignette(card, photo_rect, strength=0.44)
    photo_glow_h = 240
    glow_overlay = Image.new("RGBA", (photo_rect[2] - photo_rect[0], photo_glow_h), (0, 0, 0, 0))
    glow_draw = ImageDraw.Draw(glow_overlay, "RGBA")
    for i in range(photo_glow_h):
        alpha = int(165 * (i / max(1, photo_glow_h - 1)) ** 1.7)
        glow_draw.line((0, i, glow_overlay.width, i), fill=(8, 20, 25, alpha))
    card.alpha_composite(glow_overlay, (photo_rect[0], photo_rect[3] - photo_glow_h))

    info_top = photo_rect[3] - 168
    info_rect = (left + 18, info_top, right - 18, info_top + layout.info_h)
    info_shadow = (info_rect[0], info_rect[1] + 10, info_rect[2], info_rect[3] + 10)
    draw.rounded_rectangle(info_shadow, radius=36, fill=(17, 24, 28, 55))
    draw.rounded_rectangle(info_rect, radius=36, fill=(10, 24, 29, 206), outline=(255, 255, 255, 26), width=1)

    name_font = _fit_font(draw, entry.player_name, max_width=info_rect[2] - info_rect[0] - 72, max_size=82, min_size=42)
    _draw_text(draw, info_rect[0] + 34, info_rect[1] + 54, entry.player_name, name_font, "#f6f2e9")

    badge_rect = (info_rect[0] + 34, info_rect[1] + 132, info_rect[0] + 360, info_rect[1] + 196)
    draw.rounded_rectangle(badge_rect, radius=18, fill=(233, 193, 106, 255))
    rank_font = _fit_font(draw, entry.rank_label, max_width=badge_rect[2] - badge_rect[0] - 32, max_size=36, min_size=20)
    _draw_center_text(draw, badge_rect, entry.rank_label, rank_font, "#13252b")

    subtitle = entry.subtitle.strip()
    if subtitle:
        subtitle_font = _fit_font(draw, subtitle, max_width=info_rect[2] - info_rect[0] - 72, max_size=30, min_size=18, bold=False)
        subtitle_lines = _wrap_text(draw, subtitle, subtitle_font, info_rect[2] - info_rect[0] - 72, max_lines=2)
        for idx, line in enumerate(subtitle_lines):
            _draw_text(draw, info_rect[0] + 36, info_rect[1] + 228 + idx * 30, line, subtitle_font, "#d5ddd6")

    table_top = info_rect[3] + layout.block_gap
    table_rect = (left + 18, table_top, right - 18, bottom - 18)
    draw.rounded_rectangle(table_rect, radius=34, fill=(248, 244, 235, 236), outline=(18, 46, 53, 18), width=1)
    header_rect = (table_rect[0] + 20, table_rect[1] + 18, table_rect[2] - 20, table_rect[1] + 86)
    draw.rounded_rectangle(header_rect, radius=20, fill=(227, 216, 190, 255))
    header_font = _load_font(size=24, bold=True)
    _draw_text(draw, header_rect[0] + 24, header_rect[1] + 43, "MATCH PATH", header_font, "#29424a", anchor="lm")

    rows = _normalize_results(entry.results)
    row_h = (table_rect[3] - header_rect[3] - 20) // 7
    round_font = _load_font(size=20, bold=True)
    opp_font = _load_font(size=32, bold=True)
    score_font = _load_font(size=28, bold=True)
    round_x = table_rect[0] + 40
    opp_x = table_rect[0] + 144
    score_right = table_rect[2] - 30
    score_max_w = 186
    opp_max_w = max(60, score_right - score_max_w - 20 - opp_x)

    for idx, (rnd, opp, score) in enumerate(rows):
        row_top = header_rect[3] + idx * row_h
        row_bottom = row_top + row_h
        if idx > 0:
            draw.line((table_rect[0] + 24, row_top, table_rect[2] - 24, row_top), fill=(31, 64, 72, 38), width=1)
        if idx % 2 == 0:
            draw.rounded_rectangle((table_rect[0] + 12, row_top + 4, table_rect[2] - 12, row_bottom - 4), radius=16, fill=(246, 239, 226, 255))
        opp_display = _truncate_to_width(draw, _short_player_name(opp), opp_font, opp_max_w)
        score_display = score or "-"
        draw.text((round_x, row_top + 18), rnd, font=round_font, fill="#a5762e")
        draw.text((opp_x, row_top + 10), opp_display, font=opp_font, fill="#132f36")
        score_bbox = draw.textbbox((0, 0), score_display, font=score_font)
        score_w = score_bbox[2] - score_bbox[0]
        draw.text((score_right - score_w, row_top + 12), score_display, font=score_font, fill="#2f8d61")

    return np.array(card.convert("RGB"))


def render_video(entries: list, output_path: Path, photos_dir: Path, audio_path: Path, fps: int, duration: float) -> Path:
    if not entries:
        raise RuntimeError("No entries to render.")
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    photos_dir.mkdir(parents=True, exist_ok=True)
    card_arrays = [render_card(entry, photos_dir=photos_dir) for entry in entries]

    scroll_duration = duration - HOLD_START - HOLD_END
    if scroll_duration <= 0:
        raise RuntimeError("Invalid Shorts timing configuration.")
    total_shift = max(0.0, len(entries) - 1)

    def make_frame(t: float) -> np.ndarray:
        if t <= HOLD_START:
            progress = 0.0
        elif t >= duration - HOLD_END:
            progress = total_shift
        else:
            progress = total_shift * ((t - HOLD_START) / scroll_duration)

        base_index = min(int(progress), max(0, len(card_arrays) - 1))
        next_index = min(base_index + 1, len(card_arrays) - 1)
        alpha = progress - base_index

        frame = card_arrays[base_index].astype(np.float32)
        if next_index != base_index and alpha > 0:
            frame = frame * (1.0 - alpha) + card_arrays[next_index].astype(np.float32) * alpha
        return np.clip(frame, 0, 255).astype(np.uint8)

    clip = VideoClip(make_frame, duration=duration)
    audio_clip, keep_alive = build_audio_track(audio_path, duration)
    clip = clip.with_audio(audio_clip)
    output_path = output_path.with_suffix(".mp4")
    clip.write_videofile(str(output_path), fps=fps, codec="libx264", audio_codec="aac")
    clip.close()
    audio_clip.close()
    for item in keep_alive:
        item.close()
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate an ATP Shorts timeline video (MoviePy + Pillow)")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="CSV path")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output .mp4 path")
    parser.add_argument("--photos-dir", type=Path, default=DEFAULT_PHOTOS_DIR, help="Player photos folder")
    parser.add_argument("--audio", type=Path, default=DEFAULT_AUDIO, help="Background music path")
    parser.add_argument("--fps", type=int, default=FPS, help="Video fps")
    parser.add_argument("--duration", type=float, default=TOTAL_DURATION, help="Video duration in seconds")
    parser.add_argument("--start-year", type=int, default=None, help="First year to include")
    parser.add_argument("--end-year", type=int, default=None, help="Last year to include")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    entries = load_entries(args.input)
    if args.start_year is not None:
        entries = [entry for entry in entries if entry.year >= args.start_year]
    if args.end_year is not None:
        entries = [entry for entry in entries if entry.year <= args.end_year]
    output = render_video(
        entries=entries,
        output_path=args.output,
        photos_dir=args.photos_dir,
        audio_path=args.audio,
        fps=args.fps,
        duration=args.duration,
    )
    print(f"[video_generator] ATP Shorts timeline generated -> {output}")


if __name__ == "__main__":
    main()
