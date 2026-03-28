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
DEFAULT_INPUT = PROJECT_ROOT / "data" / "processed" / "tennis" / "miami_titles_top20_cards.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "tennis" / "miami_titles_cards_shorts.mp4"
DEFAULT_PHOTOS_DIR = PROJECT_ROOT / "data" / "raw" / "player_photos"
DEFAULT_FLAGS_DIR = PROJECT_ROOT / "data" / "raw" / "flags"

WIDTH = 1080
HEIGHT = 1920
FPS = 60
TOTAL_DURATION = 40.0
HOLD_START = 2.0
HOLD_END = 4.0


@dataclass(frozen=True)
class MiamiEntry:
    rank: int
    player_name: str
    country_code: str
    titles: int
    years_won: str
    first_title: str
    last_title: str
    badge_label: str
    final_results: str
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


def load_entries(input_csv: Path) -> list[MiamiEntry]:
    entries: list[MiamiEntry] = []
    with input_csv.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            entries.append(
                MiamiEntry(
                    rank=int(row["rank"]),
                    player_name=row["player_name"].strip(),
                    country_code=row["country_code"].strip().lower(),
                    titles=int(row["titles"]),
                    years_won=row["years_won"].strip(),
                    first_title=row["first_title"].strip(),
                    last_title=row["last_title"].strip(),
                    badge_label=row["badge_label"].strip(),
                    final_results=row.get("final_results", "").strip(),
                    card_bg_color=row["card_bg_color"].strip(),
                    accent_color=row["accent_color"].strip(),
                )
            )
    return entries


def _background() -> np.ndarray:
    xx = np.linspace(0, 1, WIDTH, dtype=np.float32)
    yy = np.linspace(0, 1, HEIGHT, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(xx, yy)
    deep = np.array([5, 18, 42], dtype=np.float32)
    teal = np.array([7, 93, 121], dtype=np.float32)
    aqua = np.array([39, 198, 211], dtype=np.float32)
    mix = np.clip(0.66 * grid_y + 0.20 * (1.0 - grid_x), 0.0, 1.0)
    left_glow = np.exp(-(((grid_x - 0.1) / 0.25) ** 2 + ((grid_y - 0.75) / 0.32) ** 2))
    right_glow = np.exp(-(((grid_x - 0.88) / 0.22) ** 2 + ((grid_y - 0.20) / 0.22) ** 2))
    bg = np.clip(
        deep[None, None, :] * (1.0 - mix[..., None])
        + teal[None, None, :] * (0.76 * mix[..., None])
        + aqua[None, None, :] * (0.10 * left_glow[..., None] + 0.12 * right_glow[..., None]),
        0,
        255,
    ).astype(np.uint8)
    return bg


def render_card(entry: MiamiEntry, card_w: int, card_h: int, photos_dir: Path, flags_dir: Path) -> np.ndarray:
    base = Image.new("RGBA", (card_w, card_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(base, "RGBA")
    accent_rgb = tuple(int(entry.accent_color[i : i + 2], 16) for i in (1, 3, 5))
    draw.rounded_rectangle((8, 8, card_w - 8, card_h - 8), radius=36, fill=entry.card_bg_color, outline=(*accent_rgb, 255), width=4)
    draw.rounded_rectangle((24, 24, card_w - 24, card_h - 24), radius=30, outline=(255, 255, 255, 26), width=2)

    rank_box = (28, 28, 138, 114)
    draw.rounded_rectangle(rank_box, radius=20, fill=(7, 16, 34, 235))
    rank_font = _fit_font(draw, str(entry.rank), 74, 54, 24, bold=True)
    draw.text(((rank_box[0] + rank_box[2]) // 2, 69), str(entry.rank), font=rank_font, fill="#f6fbff", anchor="mm")

    badge_box = (156, 34, card_w - 28, 106)
    draw.rounded_rectangle(badge_box, radius=18, fill=(7, 16, 34, 210), outline=(*accent_rgb, 90), width=2)
    badge_font = _fit_font(draw, entry.badge_label, badge_box[2] - badge_box[0] - 24, 34, 18, bold=True)
    draw.text((badge_box[0] + 20, 70), entry.badge_label, font=badge_font, fill="#d9fbff", anchor="lm")

    photo_height = int(card_h * 0.50)
    photo_rect = (28, 134, card_w - 28, 134 + photo_height)
    photo_path = _resolve_photo(entry.player_name, photos_dir)
    if photo_path is not None:
        try:
            photo = ImageOps.exif_transpose(Image.open(photo_path)).convert("RGB")
            photo = ImageOps.fit(
                photo,
                (photo_rect[2] - photo_rect[0], photo_rect[3] - photo_rect[1]),
                method=Image.Resampling.LANCZOS,
                centering=(0.5, 0.20),
            )
            photo = ImageEnhance.Contrast(photo).enhance(1.05)
            photo = ImageEnhance.Brightness(photo).enhance(1.02)
            base.alpha_composite(photo.convert("RGBA"), (photo_rect[0], photo_rect[1]))
        except Exception:
            draw.rounded_rectangle(photo_rect, radius=28, fill=(217, 226, 232, 255))
    else:
        draw.rounded_rectangle(photo_rect, radius=28, fill=(217, 226, 232, 255))
        initials = "".join(part[0] for part in entry.player_name.split()[:2]).upper()
        initials_font = _fit_font(draw, initials, 220, 122, 48, bold=True)
        draw.text(
            ((photo_rect[0] + photo_rect[2]) // 2, (photo_rect[1] + photo_rect[3]) // 2),
            initials,
            font=initials_font,
            fill="#223746",
            anchor="mm",
        )

    fade = Image.new("RGBA", (card_w, card_h), (0, 0, 0, 0))
    fade_draw = ImageDraw.Draw(fade, "RGBA")
    fade_draw.rectangle((28, photo_rect[3] - 180, card_w - 28, photo_rect[3]), fill=(5, 13, 25, 110))
    fade = fade.filter(ImageFilter.GaussianBlur(radius=18))
    base.alpha_composite(fade)

    name_y = photo_rect[3] + 26
    name_font = _fit_font(draw, entry.player_name, card_w - 96, 52, 26, bold=True)
    draw.text((42, name_y), _truncate(draw, entry.player_name, name_font, card_w - 96), font=name_font, fill="#ffffff")

    flag_path = flags_dir / f"{entry.country_code}.png"
    if flag_path.exists():
        try:
            flag = Image.open(flag_path).convert("RGBA").resize((62, 42), Image.Resampling.LANCZOS)
            base.alpha_composite(flag, (42, name_y + 66))
        except Exception:
            pass
    meta_font = _load_font(23, bold=True)
    draw.text((118, name_y + 87), f"{entry.titles} TITLES", font=meta_font, fill="#d9fbff", anchor="lm")

    era_font = _load_font(20, bold=False)
    draw.text((42, name_y + 130), f"{entry.first_title} - {entry.last_title}", font=era_font, fill="#a7dce5")

    years_panel = (28, name_y + 168, card_w - 28, name_y + 320)
    draw.rounded_rectangle(years_panel, radius=24, fill=(9, 20, 38, 186), outline=(255, 255, 255, 18), width=2)
    years_title_font = _load_font(22, bold=True)
    years_text_font = _load_font(24, bold=True)
    draw.text((42, years_panel[1] + 26), "YEARS WON", font=years_title_font, fill="#7ee7ff")

    words = re.split(r"\s+[•/]\s+|/", entry.years_won)
    words = [word.strip() for word in words if word.strip()]
    lines: list[str] = []
    current = ""
    max_width = card_w - 84
    for word in words:
        candidate = word if not current else f"{current} / {word}"
        if draw.textbbox((0, 0), candidate, font=years_text_font)[2] <= max_width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    for idx, line in enumerate(lines[:2]):
        draw.text((42, years_panel[1] + 68 + idx * 42), line, font=years_text_font, fill="#ffffff")

    finals_panel = (28, years_panel[3] + 18, card_w - 28, card_h - 34)
    draw.rounded_rectangle(finals_panel, radius=24, fill=(9, 20, 38, 186), outline=(255, 255, 255, 18), width=2)
    final_title_font = _load_font(32, bold=True)
    final_text_font = _load_font(40, bold=True)
    draw.text((42, finals_panel[1] + 26), "FINAL RESULTS", font=final_title_font, fill="#7ee7ff")

    final_chunks = [chunk.strip() for chunk in entry.final_results.split("/") if chunk.strip()]
    final_lines: list[str] = []
    current = ""
    final_width = card_w - 84
    for chunk in final_chunks:
        candidate = chunk if not current else f"{current} / {chunk}"
        if draw.textbbox((0, 0), candidate, font=final_text_font)[2] <= final_width:
            current = candidate
        else:
            if current:
                final_lines.append(current)
            current = chunk
    if current:
        final_lines.append(current)
    for idx, line in enumerate(final_lines[:4]):
        draw.text((42, finals_panel[1] + 78 + idx * 56), line, font=final_text_font, fill="#ffffff")

    return np.array(base.convert("RGB"))


def render_video(
    input_csv: Path,
    output_path: Path,
    photos_dir: Path,
    flags_dir: Path,
    audio_path: Path,
    duration: float,
    fps: int,
    cards_visible: float,
) -> None:
    entries = load_entries(input_csv)
    if not entries:
        raise RuntimeError("No Miami entries to render.")

    gap = 24
    side_padding = 36
    card_w = int((WIDTH - 2 * side_padding - gap * max(cards_visible - 1.0, 0.0)) / cards_visible)
    card_h = HEIGHT - 220
    pitch = card_w + gap
    total_shift = max(0.0, (len(entries) - cards_visible) * pitch)
    cards = [render_card(entry, card_w, card_h, photos_dir, flags_dir) for entry in entries]
    bg = _background()

    scroll_duration = max(0.1, duration - HOLD_START - HOLD_END)

    title_bar = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    td = ImageDraw.Draw(title_bar, "RGBA")
    td.rounded_rectangle((36, 36, WIDTH - 36, 154), radius=30, fill=(4, 15, 31, 215), outline=(255, 255, 255, 20), width=2)
    title_font = _load_font(46, bold=True)
    sub_font = _load_font(22, bold=False)
    td.text((54, 62), "MIAMI OPEN KINGS", font=title_font, fill="#ffffff")
    td.text((54, 114), f"Top {len(entries)} champions / cards scroll format", font=sub_font, fill="#b8ebf3")

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
            card_img = Image.fromarray(card).convert("RGBA")
            canvas.alpha_composite(card_img, (x, y))
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
    parser = argparse.ArgumentParser(description="Generate a Miami Open top champions Shorts cards video.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--photos-dir", type=Path, default=DEFAULT_PHOTOS_DIR)
    parser.add_argument("--flags-dir", type=Path, default=DEFAULT_FLAGS_DIR)
    parser.add_argument("--audio", type=Path, default=DEFAULT_AUDIO)
    parser.add_argument("--duration", type=float, default=TOTAL_DURATION)
    parser.add_argument("--fps", type=int, default=FPS)
    parser.add_argument("--cards-visible", type=float, default=1.5)
    args = parser.parse_args()

    render_video(
        input_csv=args.input,
        output_path=args.output,
        photos_dir=args.photos_dir,
        flags_dir=args.flags_dir,
        audio_path=args.audio,
        duration=args.duration,
        fps=args.fps,
        cards_visible=args.cards_visible,
    )
    print(f"[video_generator] Miami titles cards Shorts generated -> {args.output}")


if __name__ == "__main__":
    main()
