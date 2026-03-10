from __future__ import annotations

import argparse
import csv
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from moviepy import VideoClip
from PIL import Image, ImageDraw, ImageFont, ImageOps


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_INPUT = PROJECT_ROOT / "data" / "processed" / "cycling" / "paris_nice" / "paris_nice_timeline_postwar_template.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "cycling" / "paris_nice" / "paris_nice_timeline_preview_10s.mp4"
DEFAULT_PHOTOS_DIR = PROJECT_ROOT / "data" / "raw" / "player_photos"
WIDTH = 1920
HEIGHT = 1080


@dataclass(frozen=True)
class ParisNiceEntry:
    year: int
    winner_name: str
    winner_team: str
    winner_country: str
    image_path: str
    card_bg_color: str
    accent_color: str
    badge_label: str
    top5: list[tuple[str, str, str, str]]
    points_name: str
    mountains_name: str


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


def _fit_font(draw: ImageDraw.ImageDraw, text: str, max_width: int, max_size: int, min_size: int):
    size = max_size
    while size >= min_size:
        font = _load_font(size=size, bold=True)
        bbox = draw.textbbox((0, 0), text, font=font)
        if bbox[2] - bbox[0] <= max_width:
            return font
        size -= 1
    return _load_font(size=min_size, bold=True)


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


def _short_rider_name(full_name: str) -> str:
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


def _slugify(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", normalized.strip().lower())
    return cleaned.strip("_")


def _resolve_photo(entry: ParisNiceEntry, photos_dir: Path) -> Path | None:
    if entry.image_path:
        direct = PROJECT_ROOT / entry.image_path
        if direct.exists():
            return direct
    slug = _slugify(entry.winner_name)
    for ext in (".jpg", ".jpeg", ".png", ".webp"):
        candidate = photos_dir / f"{slug}{ext}"
        if candidate.exists():
            return candidate
    return None


def load_entries(input_csv: Path, start_year: int | None, end_year: int | None) -> list[ParisNiceEntry]:
    entries: list[ParisNiceEntry] = []
    title_counts: Counter[str] = Counter()
    with input_csv.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            year = int(row["year"])
            winner_name = row.get("winner_name", "").strip() or "-"
            if winner_name != "-":
                title_counts[winner_name] += 1
            if start_year is not None and year < start_year:
                continue
            if end_year is not None and year > end_year:
                continue
            top5: list[tuple[str, str, str, str]] = []
            for idx in range(1, 6):
                top5.append(
                    (
                        row.get(f"gc{idx}_name", "").strip() or "-",
                        row.get(f"gc{idx}_team", "").strip(),
                        row.get(f"gc{idx}_country", "").strip(),
                        row.get(f"gc{idx}_gap", "").strip(),
                    )
                )
            entries.append(
                ParisNiceEntry(
                    year=year,
                    winner_name=winner_name,
                    winner_team=row.get("winner_team", "").strip(),
                    winner_country=row.get("winner_country", "").strip(),
                    image_path=row.get("image_path", "").strip(),
                    card_bg_color=row.get("card_bg_color", "#0092cf").strip() or "#0092cf",
                    accent_color=row.get("accent_color", "#f2c230").strip() or "#f2c230",
                    badge_label=f"PN #{title_counts[winner_name]}" if winner_name != "-" else "PN #",
                    top5=top5,
                    points_name=row.get("points_name", "").strip() or "-",
                    mountains_name=row.get("mountains_name", "").strip() or "-",
                )
            )
    entries.sort(key=lambda item: item.year)
    return entries


def render_card(entry: ParisNiceEntry, card_w: int, card_h: int, photos_dir: Path) -> np.ndarray:
    card_bg = "#0092cf"
    inner_panel = "#f8efcf"
    table_bg = "#003f63"
    img = Image.new("RGB", (card_w, card_h), color=card_bg)
    draw = ImageDraw.Draw(img)

    border = entry.accent_color
    draw.rounded_rectangle((4, 4, card_w - 5, card_h - 5), radius=26, outline=border, width=6, fill=card_bg)

    year_rect = (18, 18, 200, 98)
    draw.rounded_rectangle(year_rect, radius=18, fill="#0c1521")
    year_font = _fit_font(draw, str(entry.year), 150, 48, 24)
    draw.text((year_rect[0] + 34, year_rect[1] + 16), str(entry.year), font=year_font, fill="#f5d54f")

    photo_rect = (18, 120, card_w - 18, 500)
    source_path = _resolve_photo(entry, photos_dir)
    if source_path:
        try:
            photo = ImageOps.exif_transpose(Image.open(source_path)).convert("RGB")
            photo = ImageOps.fit(
                photo,
                (photo_rect[2] - photo_rect[0], photo_rect[3] - photo_rect[1]),
                method=Image.Resampling.LANCZOS,
                centering=(0.5, 0.18),
            )
            img.paste(photo, box=(photo_rect[0], photo_rect[1]))
        except Exception:
            draw.rounded_rectangle(photo_rect, radius=20, fill="#d8d0bc")
    else:
        draw.rounded_rectangle(photo_rect, radius=20, fill="#d8d0bc")
        fallback_font = _fit_font(draw, entry.winner_name, card_w - 60, 42, 20)
        bbox = draw.textbbox((0, 0), entry.winner_name, font=fallback_font)
        draw.text(
            ((card_w - (bbox[2] - bbox[0])) // 2, photo_rect[1] + 160),
            entry.winner_name,
            font=fallback_font,
            fill="#223746",
        )

    title_rect = (12, 506, card_w - 12, 600)
    draw.rectangle(title_rect, fill=inner_panel)
    winner_font = _load_font(28, bold=True)
    team_font = _load_font(18, bold=False)
    name_x = title_rect[0] + 18
    name_max_w = title_rect[2] - title_rect[0] - 36
    draw.text((name_x, 517), _truncate(draw, entry.winner_name, winner_font, name_max_w), font=winner_font, fill="#0b1b28")
    winner_sub = "Paris-Nice winner"
    if entry.winner_country or entry.winner_team:
        winner_sub = " | ".join([part for part in [entry.winner_country, entry.winner_team] if part])
    draw.text((name_x, 547), _truncate(draw, winner_sub, team_font, name_max_w), font=team_font, fill="#234154")

    badge_rect = (54, 606, card_w - 54, 670)
    draw.rounded_rectangle(badge_rect, radius=14, fill="#0c1521", outline=border, width=3)
    badge_font = _fit_font(draw, entry.badge_label, card_w - 160, 38, 18)
    bbox = draw.textbbox((0, 0), entry.badge_label, font=badge_font)
    draw.text(((card_w - (bbox[2] - bbox[0])) // 2, 621), entry.badge_label, font=badge_font, fill="#f5d54f")

    table_rect = (14, 682, card_w - 14, 930)
    draw.rounded_rectangle(table_rect, radius=18, fill=table_bg, outline=border, width=3)
    header_font = _load_font(21, bold=True)
    gc_x = table_rect[0] + 10
    rider_x = table_rect[0] + 58
    gap_col_w = 126
    gap_header_x = table_rect[2] - gap_col_w + 20
    draw.text((gc_x, 698), "GC", font=header_font, fill="#f5d54f")
    draw.text((rider_x, 698), "Rider", font=header_font, fill="#f5d54f")
    draw.text((gap_header_x, 698), "Gap", font=header_font, fill="#f5d54f")

    pos_font = _load_font(20, bold=True)
    rider_font = _load_font(20, bold=True)
    gap_font = _load_font(19, bold=True)
    score_right = table_rect[2] - 16
    rider_max_w = gap_header_x - rider_x - 24
    for idx, (name, team, country, gap) in enumerate(entry.top5, start=1):
        y = 730 + (idx - 1) * 36
        if idx > 1:
            draw.line((table_rect[0] + 10, y - 7, table_rect[2] - 10, y - 7), fill="#205469", width=1)
        draw.text((gc_x, y), str(idx), font=pos_font, fill="#f5d54f")
        rider_line = _short_rider_name(name)
        if country:
            rider_line = f"{rider_line} ({country})"
        rider_line = _truncate(draw, rider_line, rider_font, rider_max_w)
        draw.text((rider_x, y), rider_line, font=rider_font, fill="#edf3f4")
        gap_text = gap or "-"
        gap_bbox = draw.textbbox((0, 0), gap_text, font=gap_font)
        draw.text((score_right - (gap_bbox[2] - gap_bbox[0]), y), gap_text, font=gap_font, fill="#8ce37b")

    jersey_rect = (18, 938, card_w - 18, card_h - 18)
    draw.rounded_rectangle(jersey_rect, radius=18, fill=inner_panel, outline=border, width=3)
    jersey_font = _load_font(19, bold=True)
    value_font = _load_font(18, bold=False)
    draw.text((28, 950), "Green jersey", font=jersey_font, fill="#0f6e3a")
    draw.text((28, 976), _truncate(draw, entry.points_name, value_font, card_w - 58), font=value_font, fill="#0b1b28")
    draw.text((28, 1002), "Polka dots", font=jersey_font, fill="#9d1e2d")
    draw.text((28, 1028), _truncate(draw, entry.mountains_name, value_font, card_w - 58), font=value_font, fill="#0b1b28")

    return np.array(img)


def render_video(entries: list[ParisNiceEntry], output_path: Path, duration: float, fps: int, cards_visible: int, photos_dir: Path) -> None:
    if not entries:
        raise RuntimeError("No Paris-Nice entries to render.")

    gap = 10
    card_w = (WIDTH - gap * (cards_visible - 1)) // cards_visible
    pitch = card_w + gap
    total_shift = max(0.0, (len(entries) - cards_visible) * pitch)
    card_arrays = [render_card(entry, card_w, HEIGHT, photos_dir) for entry in entries]

    xx = np.linspace(0, 1, WIDTH, dtype=np.float32)
    yy = np.linspace(0, 1, HEIGHT, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(xx, yy)
    sea = np.array([11, 96, 143], dtype=np.float32)
    sky = np.array([75, 165, 213], dtype=np.float32)
    sun = np.array([244, 194, 48], dtype=np.float32)
    mix = np.clip(0.58 * grid_y + 0.28 * (1.0 - grid_x), 0, 1)
    bg = np.clip(
        sea[None, None, :] * (1 - mix[..., None])
        + sky[None, None, :] * (0.55 * mix[..., None])
        + sun[None, None, :] * (0.45 * mix[..., None]),
        0,
        255,
    ).astype(np.uint8)

    def make_frame(t: float) -> np.ndarray:
        frame = bg.copy()
        progress = 0.0 if duration <= 0 else min(max(t / duration, 0.0), 1.0)
        shift = total_shift * progress
        for idx, card in enumerate(card_arrays):
            x = int(idx * pitch - shift)
            if x >= WIDTH or x + card_w <= 0:
                continue
            src_x0 = 0 if x >= 0 else -x
            dst_x0 = 0 if x < 0 else x
            visible_w = min(card_w - src_x0, WIDTH - dst_x0)
            frame[:, dst_x0 : dst_x0 + visible_w] = card[:, src_x0 : src_x0 + visible_w]
        return frame

    clip = VideoClip(make_frame, duration=duration)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    clip.write_videofile(str(output_path), fps=fps, codec="libx264", audio=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a 10s preview for Paris-Nice timeline cards.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--cards-visible", type=int, default=4)
    parser.add_argument("--start-year", type=int, default=2014)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--photos-dir", type=Path, default=DEFAULT_PHOTOS_DIR)
    args = parser.parse_args()

    entries = load_entries(args.input, args.start_year, args.end_year)
    render_video(entries, args.output, args.duration, args.fps, args.cards_visible, args.photos_dir)
    print(f"[video_generator] Paris-Nice preview generated -> {args.output}")


if __name__ == "__main__":
    main()
