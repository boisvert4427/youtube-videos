from __future__ import annotations

import argparse
import csv
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from moviepy import AudioFileClip, VideoClip
from moviepy.audio.fx import AudioFadeOut, AudioLoop
from PIL import Image, ImageDraw, ImageFont, ImageOps


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = PROJECT_ROOT / "data" / "processed" / "atp_vertical_timeline_indian_wells_winners_1976_2025.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "atp_vertical_timeline_indian_wells_1976_2025_moviepy.mp4"
DEFAULT_PHOTOS_DIR = PROJECT_ROOT / "data" / "raw" / "player_photos"
DEFAULT_AUDIO = PROJECT_ROOT / "data" / "raw" / "audio" / "audio.mp3"

WIDTH = 1920
HEIGHT = 1080
ROUNDS = ["R128", "R64", "R32", "R16", "QF", "SF", "F"]
TOTAL_DURATION = 90.0
HOLD_START = 5.0
HOLD_END = 15.0
FADE_OUT_AUDIO = 10.0


@dataclass(frozen=True)
class TimelineEntry:
    year: int
    player_name: str
    subtitle: str
    image_path: str
    name_bg_color: str
    card_bg_color: str
    rank_label: str
    results: list[str]


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


def _slugify(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", normalized.strip().lower())
    return cleaned.strip("_")


def _parse_results(value: str) -> list[str]:
    if not value.strip():
        return []
    return [p.strip() for p in value.split("|") if p.strip()]


def _extract_result_parts(line: str) -> tuple[str, str, str]:
    text = line.strip()
    if not text:
        return ("", "", "")
    tokens = text.split()
    rnd = tokens[0] if tokens else ""
    if rnd not in set(ROUNDS):
        return ("", text, "")
    rest = " ".join(tokens[1:]).strip()
    if not rest:
        return (rnd, "-", "")

    parts = rest.split()
    score_parts: list[str] = []
    while parts:
        t = parts[-1]
        if any(ch.isdigit() for ch in t) or "-" in t:
            score_parts.insert(0, parts.pop())
        else:
            break
    opponent = " ".join(parts).strip() or "-"
    score = " ".join(score_parts).strip()
    return (rnd, opponent, score)


def _normalize_results(rows: list[str]) -> list[tuple[str, str, str]]:
    by_round: dict[str, tuple[str, str]] = {}
    for row in rows:
        rnd, opp, score = _extract_result_parts(row)
        if rnd:
            by_round[rnd] = (opp, score)

    result: list[tuple[str, str, str]] = []
    for rnd in ROUNDS:
        opp, score = by_round.get(rnd, ("-", ""))
        result.append((rnd, opp, score))
    return result


def load_entries(input_csv: Path) -> list[TimelineEntry]:
    entries: list[TimelineEntry] = []
    with input_csv.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            entries.append(
                TimelineEntry(
                    year=int(row["year"]),
                    player_name=row["player_name"].strip(),
                    subtitle=row.get("subtitle", "").strip(),
                    image_path=row.get("image_path", "").strip(),
                    name_bg_color=row.get("name_bg_color", "#f4df26").strip() or "#f4df26",
                    card_bg_color=row.get("card_bg_color", "#5f3518").strip() or "#5f3518",
                    rank_label=row.get("rank_label", "IW #1").strip() or "IW #1",
                    results=_parse_results(row.get("results", "")),
                )
            )
    entries.sort(key=lambda e: e.year)
    return entries


def _resolve_player_image(entry: TimelineEntry, photos_dir: Path) -> Path | None:
    if entry.image_path:
        direct = PROJECT_ROOT / entry.image_path
        if direct.exists():
            return direct
    slug = _slugify(entry.player_name)
    for ext in (".jpg", ".jpeg", ".png", ".webp"):
        candidate = photos_dir / f"{slug}{ext}"
        if candidate.exists():
            return candidate
    return None


def _draw_center_text(draw: ImageDraw.ImageDraw, rect: tuple[int, int, int, int], text: str, font, fill: str) -> None:
    x0, y0, x1, y1 = rect
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    tx = x0 + (x1 - x0 - tw) // 2
    ty = y0 + (y1 - y0 - th) // 2
    draw.text((tx, ty), text, font=font, fill=fill)


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


def _fit_font_to_width(draw: ImageDraw.ImageDraw, text: str, max_width: int, start_size: int, min_size: int):
    size = start_size
    while size >= min_size:
        font = _load_font(size=size, bold=True)
        w = draw.textbbox((0, 0), text, font=font)[2]
        if w <= max_width:
            return font
        size -= 1
    return _load_font(size=min_size, bold=True)


def render_card(entry: TimelineEntry, card_w: int, card_h: int, photos_dir: Path) -> np.ndarray:
    img = Image.new("RGB", (card_w, card_h), color=entry.card_bg_color)
    draw = ImageDraw.Draw(img)

    border_color = "#a86b38"
    draw.rounded_rectangle((1, 1, card_w - 2, card_h - 2), radius=20, outline=border_color, width=4, fill=entry.card_bg_color)

    year_badge = (16, 16, 186, 96)
    draw.rounded_rectangle(year_badge, radius=16, fill="black", outline="black", width=2)
    year_font = _fit_font(draw, str(entry.year), max_width=140, max_size=48, min_size=24)
    _draw_center_text(draw, year_badge, str(entry.year), year_font, "#f6de33")

    photo_rect = (10, 120, card_w - 10, 560)
    source_path = _resolve_player_image(entry, photos_dir)
    if source_path:
        try:
            p = Image.open(source_path).convert("RGB")
            # Keep the top of portraits visible (head), crop lower part first.
            p = ImageOps.fit(
                p,
                (photo_rect[2] - photo_rect[0], photo_rect[3] - photo_rect[1]),
                method=Image.Resampling.LANCZOS,
                centering=(0.5, 0.18),
            )
            img.paste(p, box=(photo_rect[0], photo_rect[1]))
        except Exception:
            draw.rectangle(photo_rect, fill="#cfcfcf")
    else:
        draw.rectangle(photo_rect, fill="#cfcfcf")
        fallback_font = _fit_font(draw, entry.player_name, max_width=card_w - 36, max_size=34, min_size=18)
        _draw_center_text(draw, photo_rect, entry.player_name, fallback_font, "#1a1a1a")

    name_rect = (10, 560, card_w - 10, 690)
    draw.rectangle(name_rect, fill=entry.name_bg_color)
    name_font = _fit_font(draw, entry.player_name, max_width=card_w - 34, max_size=54, min_size=24)
    _draw_center_text(draw, name_rect, entry.player_name, name_font, "#111111")

    badge_w = int(card_w * 0.72)
    badge_h = 74
    badge_x = (card_w - badge_w) // 2
    badge_y = 710
    badge_rect = (badge_x, badge_y, badge_x + badge_w, badge_y + badge_h)
    draw.rounded_rectangle(badge_rect, radius=14, fill="black", outline="#f1d233", width=3)
    rank_font = _fit_font(draw, entry.rank_label, max_width=badge_w - 28, max_size=46, min_size=20)
    _draw_center_text(draw, badge_rect, entry.rank_label, rank_font, "#f7dd2d")

    table_rect = (8, 800, card_w - 8, card_h - 8)
    draw.rounded_rectangle(table_rect, radius=10, fill="#2b1509", outline="#b06d30", width=2)
    rows = _normalize_results(entry.results)
    inner_top = table_rect[1] + 8
    row_h = int((table_rect[3] - table_rect[1] - 14) / 7)
    round_font = _load_font(size=24, bold=True)
    # Smaller fixed sizes to avoid collisions between opponent and score columns.
    opp_font = _load_font(size=23, bold=True)
    score_font_base_size = 22
    round_x = table_rect[0] + 16
    opp_x = table_rect[0] + 88
    score_right = table_rect[2] - 14
    score_max_w = int(card_w * 0.24)
    col_gap = 14
    opp_max_w = max(40, score_right - score_max_w - col_gap - opp_x)

    for i, (rnd, opp, score) in enumerate(rows):
        y0 = inner_top + i * row_h
        y1 = y0 + row_h
        if i > 0:
            draw.line((table_rect[0] + 8, y0, table_rect[2] - 8, y0), fill="#8b5124", width=1)

        opp_display = _truncate_to_width(draw, _short_player_name(opp), opp_font, opp_max_w)
        score_display = score or "-"
        score_font = _fit_font_to_width(draw, score_display, score_max_w, start_size=score_font_base_size, min_size=14)

        draw.text((round_x, y0 + 6), rnd, font=round_font, fill="#f0c93a")
        draw.text((opp_x, y0 + 6), opp_display, font=opp_font, fill="#e8e8e8")
        score_bbox = draw.textbbox((0, 0), score_display, font=score_font)
        score_w = score_bbox[2] - score_bbox[0]
        draw.text((score_right - score_w, y0 + 6), score_display, font=score_font, fill="#95dc77")

    return np.array(img)


def render_video(
    entries: list[TimelineEntry],
    output_path: Path,
    photos_dir: Path,
    audio_path: Path,
    fps: int,
    cards_visible: int,
) -> Path:
    if not entries:
        raise RuntimeError("No entries to render.")
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    cards_visible = max(3, cards_visible)
    gap = 8
    card_w = (WIDTH - (cards_visible - 1) * gap) // cards_visible
    pitch = card_w + gap

    output_path.parent.mkdir(parents=True, exist_ok=True)
    photos_dir.mkdir(parents=True, exist_ok=True)

    card_arrays = [render_card(entry, card_w=card_w, card_h=HEIGHT, photos_dir=photos_dir) for entry in entries]

    duration = TOTAL_DURATION
    scroll_duration = duration - HOLD_START - HOLD_END
    if scroll_duration <= 0:
        raise RuntimeError("Invalid fixed timing configuration.")
    total_shift = max(0.0, (len(entries) - 1) * pitch)

    # Indian Wells-like palette: desert sand + court blue gradient.
    xx = np.linspace(0, 1, WIDTH, dtype=np.float32)
    yy = np.linspace(0, 1, HEIGHT, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(xx, yy)
    mix = np.clip(0.62 * grid_y + 0.38 * (1.0 - grid_x), 0, 1)
    sand = np.array([176, 118, 62], dtype=np.float32)
    blue = np.array([37, 79, 102], dtype=np.float32)
    bg_template = np.clip(sand[None, None, :] * mix[..., None] + blue[None, None, :] * (1.0 - mix[..., None]), 0, 255).astype(np.uint8)

    def make_frame(t: float) -> np.ndarray:
        frame = bg_template.copy()
        if t <= HOLD_START:
            shift = 0.0
        elif t >= duration - HOLD_END:
            shift = total_shift
        else:
            progress = (t - HOLD_START) / scroll_duration
            shift = total_shift * progress
        for i, card in enumerate(card_arrays):
            x = int(i * pitch - shift)
            if x >= WIDTH or x + card_w <= 0:
                continue
            src_x0 = 0 if x >= 0 else -x
            dst_x0 = 0 if x < 0 else x
            visible_w = min(card_w - src_x0, WIDTH - dst_x0)
            if visible_w <= 0:
                continue
            frame[:, dst_x0 : dst_x0 + visible_w] = card[:, src_x0 : src_x0 + visible_w]
        return frame

    final = VideoClip(make_frame, duration=duration)
    audio_clip = AudioFileClip(str(audio_path)).with_effects(
        [
            AudioLoop(duration=duration),
            AudioFadeOut(FADE_OUT_AUDIO),
        ]
    )
    final = final.with_audio(audio_clip)
    output_path = output_path.with_suffix(".mp4")
    final.write_videofile(str(output_path), fps=fps, codec="libx264", audio_codec="aac")
    audio_clip.close()
    final.close()
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate ATP vertical timeline video (MoviePy + Pillow)")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="CSV path")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output .mp4 path")
    parser.add_argument("--photos-dir", type=Path, default=DEFAULT_PHOTOS_DIR, help="Player photos folder")
    parser.add_argument("--audio", type=Path, default=DEFAULT_AUDIO, help="Background music path")
    parser.add_argument("--fps", type=int, default=30, help="Video fps")
    parser.add_argument("--cards-visible", type=int, default=4, help="Cards visible at once")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    entries = load_entries(args.input)
    output = render_video(
        entries=entries,
        output_path=args.output,
        photos_dir=args.photos_dir,
        audio_path=args.audio,
        fps=args.fps,
        cards_visible=args.cards_visible,
    )
    print(f"[video_generator] moviepy timeline generated -> {output}")


if __name__ == "__main__":
    main()
