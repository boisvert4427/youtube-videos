from __future__ import annotations

import argparse
import csv
import math
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from moviepy import VideoClip
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageOps

from video_generator.generate_ucl_barchart_race_moviepy import DEFAULT_AUDIO, build_audio_track


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_INPUT = PROJECT_ROOT / "data" / "processed" / "cycling" / "paris_roubaix_titles_top10_cards.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "cycling" / "paris_roubaix_titles_top10_shorts.mp4"
DEFAULT_PHOTOS_DIR = PROJECT_ROOT / "data" / "raw" / "player_photos"
DEFAULT_FLAGS_DIR = PROJECT_ROOT / "data" / "raw" / "flags"

WIDTH = 1080
HEIGHT = 1920
FPS = 60
TOTAL_DURATION = 40.0
HOLD_START = 0.0
HOLD_END = 4.0


@dataclass(frozen=True)
class RoubaixEntry:
    rank: int
    player_name: str
    country_code: str
    titles: int
    years_won: str
    first_title: str
    last_title: str
    badge_label: str
    card_bg_color: str
    accent_color: str


def _load_font(size: int, bold: bool = False):
    candidates = [
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf",
    ]
    for font_path in candidates:
        path = Path(font_path)
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


def _truncate(draw: ImageDraw.ImageDraw, text: str, font, max_width: int) -> str:
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


def _slugify(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", normalized.strip().lower())
    return cleaned.strip("_")


def _resolve_photo(player_name: str, photos_dir: Path) -> Path | None:
    slug = _slugify(player_name)
    for ext in (".jpg", ".jpeg", ".png", ".webp"):
        candidate = photos_dir / f"{slug}{ext}"
        if candidate.exists():
            return candidate
    return None


def load_entries(input_csv: Path) -> list[RoubaixEntry]:
    entries: list[RoubaixEntry] = []
    with input_csv.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            entries.append(
                RoubaixEntry(
                    rank=int(row["rank"]),
                    player_name=row["player_name"].strip(),
                    country_code=row["country_code"].strip().lower(),
                    titles=int(row["titles"]),
                    years_won=row["years_won"].strip(),
                    first_title=row["first_title"].strip(),
                    last_title=row["last_title"].strip(),
                    badge_label=row["badge_label"].strip(),
                    card_bg_color=row["card_bg_color"].strip(),
                    accent_color=row["accent_color"].strip(),
                )
            )
    return entries


def _background() -> np.ndarray:
    xx = np.linspace(0, 1, WIDTH, dtype=np.float32)
    yy = np.linspace(0, 1, HEIGHT, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(xx, yy)
    deep = np.array([15, 16, 20], dtype=np.float32)
    dust = np.array([92, 68, 38], dtype=np.float32)
    steel = np.array([120, 126, 136], dtype=np.float32)
    mix = np.clip(0.6 * grid_y + 0.1 * np.abs(grid_x - 0.5), 0.0, 1.0)
    glow_left = np.exp(-(((grid_x - 0.16) / 0.24) ** 2 + ((grid_y - 0.78) / 0.26) ** 2))
    glow_right = np.exp(-(((grid_x - 0.84) / 0.22) ** 2 + ((grid_y - 0.22) / 0.22) ** 2))
    bg = np.clip(
        deep[None, None, :] * (1.0 - mix[..., None])
        + dust[None, None, :] * (0.54 * mix[..., None])
        + steel[None, None, :] * (0.10 * glow_right[..., None] + 0.08 * glow_left[..., None]),
        0,
        255,
    ).astype(np.uint8)
    return bg


def render_card(entry: RoubaixEntry, photos_dir: Path, flags_dir: Path) -> np.ndarray:
    base = Image.new("RGBA", (WIDTH, HEIGHT - 220), (0, 0, 0, 0))
    draw = ImageDraw.Draw(base, "RGBA")
    accent_rgb = tuple(int(entry.accent_color[i : i + 2], 16) for i in (1, 3, 5))
    card_w, card_h = base.size

    draw.rounded_rectangle((18, 18, card_w - 18, card_h - 18), radius=42, fill=entry.card_bg_color, outline=(*accent_rgb, 255), width=5)
    draw.rounded_rectangle((34, 34, card_w - 34, card_h - 34), radius=34, outline=(255, 255, 255, 24), width=2)

    rank_box = (34, 34, 160, 128)
    draw.rounded_rectangle(rank_box, radius=24, fill=(10, 15, 22, 230))
    rank_font = _fit_font(draw, str(entry.rank), 88, 60, 28, bold=True)
    draw.text(((rank_box[0] + rank_box[2]) // 2, 82), str(entry.rank), font=rank_font, fill="#f7fbff", anchor="mm")

    badge_box = (182, 40, card_w - 34, 122)
    draw.rounded_rectangle(badge_box, radius=22, fill=(10, 16, 24, 214), outline=(*accent_rgb, 84), width=2)
    badge_font = _fit_font(draw, entry.badge_label, badge_box[2] - badge_box[0] - 30, 38, 20, bold=True)
    draw.text((badge_box[0] + 22, 81), entry.badge_label, font=badge_font, fill="#f6df96", anchor="lm")

    photo_rect = (34, 150, card_w - 34, 1010)
    photo_path = _resolve_photo(entry.player_name, photos_dir)
    if photo_path is not None:
        try:
            photo = ImageOps.exif_transpose(Image.open(photo_path)).convert("RGB")
            photo = ImageOps.fit(photo, (photo_rect[2] - photo_rect[0], photo_rect[3] - photo_rect[1]), method=Image.Resampling.LANCZOS, centering=(0.5, 0.16))
            photo = ImageEnhance.Contrast(photo).enhance(1.05)
            photo = ImageEnhance.Brightness(photo).enhance(1.02)
            base.alpha_composite(photo.convert("RGBA"), (photo_rect[0], photo_rect[1]))
        except Exception:
            draw.rounded_rectangle(photo_rect, radius=30, fill=(214, 218, 224, 255))
    else:
        draw.rounded_rectangle(photo_rect, radius=30, fill=(214, 218, 224, 255))
        initials = "".join(part[0] for part in entry.player_name.split()[:2]).upper()
        initials_font = _fit_font(draw, initials, 260, 150, 64, bold=True)
        draw.text(((photo_rect[0] + photo_rect[2]) // 2, (photo_rect[1] + photo_rect[3]) // 2), initials, font=initials_font, fill="#273647", anchor="mm")

    fade = Image.new("RGBA", base.size, (0, 0, 0, 0))
    fd = ImageDraw.Draw(fade, "RGBA")
    fd.rectangle((34, photo_rect[3] - 210, card_w - 34, photo_rect[3]), fill=(4, 8, 14, 126))
    fade = fade.filter(ImageFilter.GaussianBlur(radius=18))
    base.alpha_composite(fade)

    name_y = photo_rect[3] + 36
    name_font = _fit_font(draw, entry.player_name, card_w - 120, 74, 34, bold=True)
    draw.text((52, name_y), _truncate(draw, entry.player_name, name_font, card_w - 120), font=name_font, fill="#ffffff")

    flag_path = flags_dir / f"{entry.country_code}.png"
    if flag_path.exists():
        try:
            flag = Image.open(flag_path).convert("RGBA").resize((78, 52), Image.Resampling.LANCZOS)
            base.alpha_composite(flag, (52, name_y + 92))
        except Exception:
            pass

    meta_font = _load_font(28, bold=True)
    draw.text((148, name_y + 118), f"{entry.titles} WINS", font=meta_font, fill="#f5e3b0", anchor="lm")
    era_font = _load_font(26, bold=False)
    draw.text((card_w - 52, name_y + 118), f"{entry.first_title} - {entry.last_title}", font=era_font, fill="#d7e7f6", anchor="rm")

    years_panel = (34, name_y + 170, card_w - 34, card_h - 34)
    draw.rounded_rectangle(years_panel, radius=28, fill=(10, 17, 29, 190), outline=(255, 255, 255, 22), width=2)
    title_font = _load_font(34, bold=True)
    years_text_font = _load_font(52, bold=True)
    draw.text((52, years_panel[1] + 34), "WINNING YEARS", font=title_font, fill="#ffd76a")

    year_chunks = [chunk.strip() for chunk in entry.years_won.split("/") if chunk.strip()]
    year_lines: list[str] = []
    current = ""
    max_width = card_w - 104
    for chunk in year_chunks:
        candidate = chunk if not current else f"{current} / {chunk}"
        if draw.textbbox((0, 0), candidate, font=years_text_font)[2] <= max_width:
            current = candidate
        else:
            if current:
                year_lines.append(current)
            current = chunk
    if current:
        year_lines.append(current)

    for idx, line in enumerate(year_lines[:5]):
        draw.text((52, years_panel[1] + 104 + idx * 70), line, font=years_text_font, fill="#ffffff")

    return np.array(base.convert("RGB"))


def render_video(
    input_csv: Path,
    output_path: Path,
    photos_dir: Path,
    flags_dir: Path,
    audio_path: Path,
    duration: float,
    fps: int,
) -> None:
    entries = load_entries(input_csv)
    if not entries:
        raise RuntimeError("No Paris-Roubaix entries to render.")

    gap = 40
    side_padding = 0
    card_w = WIDTH
    pitch = card_w + gap
    total_shift = max(0.0, (len(entries) - 1) * pitch)
    cards = [render_card(entry, photos_dir, flags_dir) for entry in entries]
    bg = _background()
    scroll_duration = max(0.1, duration - HOLD_START - HOLD_END)

    title_bar = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    td = ImageDraw.Draw(title_bar, "RGBA")
    td.rounded_rectangle((36, 36, WIDTH - 36, 154), radius=30, fill=(10, 17, 29, 215), outline=(255, 255, 255, 20), width=2)
    title_font = _load_font(44, bold=True)
    sub_font = _load_font(22, bold=False)
    td.text((54, 60), "PARIS-ROUBAIX KINGS", font=title_font, fill="#ffffff")
    td.text((54, 112), f"Top {len(entries)} winners / full-card Shorts timeline", font=sub_font, fill="#efd99a")

    def make_frame(t: float) -> np.ndarray:
        frame = bg.copy()
        if t <= HOLD_START:
            shift = 0.0
        elif t >= duration - HOLD_END:
            shift = total_shift
        else:
            progress = (t - HOLD_START) / scroll_duration
            eased = 0.5 - 0.5 * math.cos(progress * math.pi)
            shift = total_shift * eased

        canvas = Image.fromarray(frame).convert("RGBA")
        canvas.alpha_composite(title_bar)
        for idx, card in enumerate(cards):
            x = int(side_padding + idx * pitch - shift)
            y = 178
            if x >= WIDTH or x + card_w <= 0:
                continue
            canvas.alpha_composite(Image.fromarray(card).convert("RGBA"), (x, y))
        return np.array(canvas.convert("RGB"))

    clip = VideoClip(make_frame, duration=duration)
    if audio_path.exists():
        audio_clip, keep_alive = build_audio_track(audio_path, duration)
        clip = clip.with_audio(audio_clip)
    else:
        audio_clip, keep_alive = None, []

    output_path.parent.mkdir(parents=True, exist_ok=True)
    clip.write_videofile(str(output_path), fps=fps, codec="libx264", audio_codec="aac" if audio_clip else None)

    clip.close()
    if audio_clip is not None:
        audio_clip.close()
    for item in keep_alive:
        item.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a Paris-Roubaix top winners Shorts cards video.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--photos-dir", type=Path, default=DEFAULT_PHOTOS_DIR)
    parser.add_argument("--flags-dir", type=Path, default=DEFAULT_FLAGS_DIR)
    parser.add_argument("--audio", type=Path, default=DEFAULT_AUDIO)
    parser.add_argument("--duration", type=float, default=TOTAL_DURATION)
    parser.add_argument("--fps", type=int, default=FPS)
    args = parser.parse_args()

    render_video(
        input_csv=args.input,
        output_path=args.output,
        photos_dir=args.photos_dir,
        flags_dir=args.flags_dir,
        audio_path=args.audio,
        duration=args.duration,
        fps=args.fps,
    )
    print(f"[video_generator] Paris-Roubaix titles cards Shorts generated -> {args.output}")


if __name__ == "__main__":
    main()
