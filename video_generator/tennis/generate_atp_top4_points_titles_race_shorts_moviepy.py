from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from moviepy import VideoClip
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont, ImageOps


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from video_generator.generate_ucl_barchart_race_moviepy import build_audio_track


DEFAULT_RANKINGS = PROJECT_ROOT / "data" / "processed" / "tennis" / "atp_top4_points_since_2024.csv"
DEFAULT_TITLES = PROJECT_ROOT / "data" / "processed" / "tennis" / "atp_top4_titles_since_2024.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "tennis" / "atp_top4_points_titles_race_since_2024_shorts.mp4"
DEFAULT_PREVIEW = PROJECT_ROOT / "tmp_frames" / "atp_top4_points_titles_race_since_2024_preview.png"
DEFAULT_PHOTOS_DIR = PROJECT_ROOT / "data" / "raw" / "player_photos"
DEFAULT_AUDIO = PROJECT_ROOT / "data" / "raw" / "audio" / "Midnight_Grip_20260402_0828.mp3"

WIDTH = 1080
HEIGHT = 1920
FPS = 60
DURATION = 40.0
TOP_N = 4
INTRO_HOLD = 1.15
OUTRO_HOLD = 1.5

TITLE = "ATP TOP 4 RACE"
SUBTITLE = "2025 Ranking Points"

CARD_X = 68
CARD_SIZE = 150
CHART_LEFT = 305
ROW_TOP = 570
ROW_GAP = 255
BAR_HEIGHT = 150
MIN_BAR_W = 92
MAX_BAR_W = 430
TICK_ORIGIN_X = 810
PX_PER_DAY = 20.0
LOGO_RADIUS = 76

PLAYER_COLORS = {
    "Jannik Sinner": "#F4C542",
    "Carlos Alcaraz": "#F01832",
    "Alexander Zverev": "#F4F1E8",
    "Novak Djokovic": "#2D8CFF",
    "Daniil Medvedev": "#5A2A82",
    "Taylor Fritz": "#21A67A",
    "Jack Draper": "#45C3D5",
}


@dataclass(frozen=True)
class RankingSnapshot:
    date: datetime
    points: dict[str, int]
    ranks: dict[str, int]
    countries: dict[str, str]


@dataclass(frozen=True)
class TitleEvent:
    date: datetime
    player: str
    tournament: str
    logo_path: Path


def _hex_to_rgb(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    return tuple(int(value[i : i + 2], 16) for i in (0, 2, 4))


def _mix(a: tuple[int, int, int], b: tuple[int, int, int], amount: float) -> tuple[int, int, int]:
    amount = max(0.0, min(1.0, amount))
    return tuple(int(a[i] + (b[i] - a[i]) * amount) for i in range(3))


def _darken(color: tuple[int, int, int], amount: float) -> tuple[int, int, int]:
    return tuple(int(channel * (1.0 - amount)) for channel in color)


def _smoothstep(value: float) -> float:
    value = max(0.0, min(1.0, value))
    return value * value * (3.0 - 2.0 * value)


def _load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/calibrib.ttf" if bold else "C:/Windows/Fonts/calibri.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    box = draw.textbbox((0, 0), text, font=font)
    return box[2] - box[0], box[3] - box[1]


def _draw_centered(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int] | tuple[int, int, int, int],
    stroke_fill: tuple[int, int, int] | tuple[int, int, int, int] = (0, 0, 0, 190),
    stroke_width: int = 2,
) -> None:
    draw.text(xy, text, font=font, fill=fill, anchor="mm", stroke_fill=stroke_fill, stroke_width=stroke_width)


def _slugify_player_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")


def _player_photo_path(player: str, photos_dir: Path) -> Path | None:
    slug = _slugify_player_name(player)
    for suffix in [".jpg", ".jpeg", ".png"]:
        path = photos_dir / f"{slug}{suffix}"
        if path.exists():
            return path
    return None


def _make_background() -> Image.Image:
    y = np.linspace(0.0, 1.0, HEIGHT)[:, None]
    x = np.linspace(0.0, 1.0, WIDTH)[None, :]
    left = np.array([206, 74, 235], dtype=np.float32)
    right = np.array([45, 204, 232], dtype=np.float32)
    bottom = np.array([119, 88, 246], dtype=np.float32)
    base = left * (1 - x[..., None]) + right * x[..., None]
    base = base * (1 - y[..., None] * 0.42) + bottom * (y[..., None] * 0.42)
    glow = np.exp(-(((x - 0.62) ** 2) / 0.10 + ((y - 0.20) ** 2) / 0.08))[..., None]
    base += glow * np.array([74, 42, 128])
    noise = np.random.default_rng(42).normal(0, 1.6, (HEIGHT, WIDTH, 1)).astype(np.float32)
    img = Image.fromarray(np.clip(base + noise, 0, 255).astype(np.uint8), "RGB").convert("RGBA")
    overlay = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")
    draw.rounded_rectangle((28, 24, WIDTH - 28, HEIGHT - 24), radius=38, outline=(255, 255, 255, 22), width=2)
    draw.rounded_rectangle((42, 38, WIDTH - 42, HEIGHT - 38), radius=34, outline=(255, 255, 255, 10), width=1)
    return Image.alpha_composite(img, overlay).filter(ImageFilter.GaussianBlur(0.2))


def _load_player_card(player: str, photos_dir: Path) -> Image.Image:
    color = _hex_to_rgb(PLAYER_COLORS.get(player, "#F4F1E8"))
    photo_path = _player_photo_path(player, photos_dir)
    if photo_path:
        raw = ImageOps.exif_transpose(Image.open(photo_path)).convert("RGB")
        raw = ImageOps.fit(raw, (CARD_SIZE, CARD_SIZE), method=Image.Resampling.LANCZOS, centering=(0.5, 0.18))
    else:
        raw = Image.new("RGB", (CARD_SIZE, CARD_SIZE), _mix(color, (18, 25, 42), 0.35))
        draw = ImageDraw.Draw(raw)
        initials = "".join(part[0] for part in player.split()[:2]).upper()
        _draw_centered(draw, (CARD_SIZE // 2, CARD_SIZE // 2), initials, _load_font(48, True), (255, 255, 255), stroke_width=2)

    mask = Image.new("L", (CARD_SIZE, CARD_SIZE), 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, CARD_SIZE - 1, CARD_SIZE - 1), radius=24, fill=255)
    out = Image.new("RGBA", (CARD_SIZE + 18, CARD_SIZE + 18), (0, 0, 0, 0))
    shadow = Image.new("RGBA", (CARD_SIZE, CARD_SIZE), (0, 0, 0, 150))
    out.alpha_composite(shadow.filter(ImageFilter.GaussianBlur(8)), (10, 12))
    photo = raw.convert("RGBA")
    photo.putalpha(mask)
    out.alpha_composite(photo, (4, 4))
    return out


def _logo_tile(path: Path, size: int) -> Image.Image:
    try:
        logo = ImageOps.exif_transpose(Image.open(path)).convert("RGBA")
    except Exception:
        logo = Image.new("RGBA", (size, size), (35, 68, 140, 255))
        draw = ImageDraw.Draw(logo)
        _draw_centered(draw, (size // 2, size // 2), "ATP", _load_font(size // 3, True), (255, 255, 255), stroke_width=2)
    tile = Image.new("RGBA", (size, size), (255, 255, 255, 255))
    fitted = ImageOps.contain(logo, (size - 12, size - 12), method=Image.Resampling.LANCZOS)
    tile.alpha_composite(fitted, ((size - fitted.width) // 2, (size - fitted.height) // 2))
    return tile


def load_rankings(path: Path) -> list[RankingSnapshot]:
    grouped: dict[str, list[dict[str, str]]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            grouped.setdefault(row["ranking_date"], []).append(row)

    snapshots: list[RankingSnapshot] = []
    for date_str, rows in sorted(grouped.items()):
        points = {row["player_name"]: int(row["points"]) for row in rows}
        ranks = {row["player_name"]: int(row.get("rank") or index + 1) for index, row in enumerate(rows)}
        countries = {row["player_name"]: row.get("country_code", "") for row in rows}
        snapshots.append(RankingSnapshot(datetime.strptime(date_str, "%Y-%m-%d"), points, ranks, countries))
    return snapshots


def load_title_events(path: Path) -> list[TitleEvent]:
    events: list[TitleEvent] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            logo_path = Path(row["logo_file"])
            if not logo_path.is_absolute():
                logo_path = PROJECT_ROOT / logo_path
            events.append(
                TitleEvent(
                    date=datetime.strptime(row["event_date"], "%Y-%m-%d"),
                    player=row["player_name"],
                    tournament=row["tournament"],
                    logo_path=logo_path,
                )
            )
    return sorted(events, key=lambda item: item.date)


def _rank_positions(points: dict[str, float]) -> dict[str, int]:
    return {player: index for index, (player, _value) in enumerate(sorted(points.items(), key=lambda item: (-item[1], item[0])))}


def _state_for_progress(
    snapshots: list[RankingSnapshot],
    progress: float,
) -> tuple[datetime, list[tuple[str, float, float]]]:
    clamped_progress = max(0.0, min(1.0, progress))
    first_date = snapshots[0].date
    last_date = snapshots[-1].date
    target_date = first_date + timedelta(days=(last_date - first_date).days * clamped_progress)

    left_idx = 0
    for index in range(len(snapshots) - 1):
        if snapshots[index].date <= target_date <= snapshots[index + 1].date:
            left_idx = index
            break
    else:
        left_idx = max(0, len(snapshots) - 2)

    right_idx = min(left_idx + 1, len(snapshots) - 1)
    left = snapshots[left_idx]
    right = snapshots[right_idx]
    span_seconds = max(1.0, float((right.date - left.date).total_seconds()))
    alpha = max(0.0, min(1.0, (target_date - left.date).total_seconds() / span_seconds))
    players = set(left.points) | set(right.points)
    left_points = {player: float(left.points.get(player, 0)) for player in players}
    right_points = {player: float(right.points.get(player, 0)) for player in players}
    left_positions = _rank_positions(left_points)
    right_positions = _rank_positions(right_points)

    states: list[tuple[str, float, float]] = []
    for player in players:
        old_rank = float(left_positions.get(player, TOP_N + 2))
        new_rank = float(right_positions.get(player, TOP_N + 2))
        y_rank = old_rank + (new_rank - old_rank) * alpha
        points = left_points[player] + (right_points[player] - left_points[player]) * alpha
        if y_rank <= TOP_N - 0.05 and points > 1:
            states.append((player, y_rank, points))
    states.sort(key=lambda item: (item[1], -item[2], item[0]))
    return target_date, states[:TOP_N]


def _draw_header(frame: Image.Image, current_date: datetime) -> None:
    draw = ImageDraw.Draw(frame, "RGBA")
    title_font = _load_font(54, True)
    sub_font = _load_font(25, True)
    _draw_centered(draw, (WIDTH // 2, 176), TITLE, title_font, (246, 250, 255), (18, 26, 48), 4)
    pill = (305, 230, 775, 292)
    draw.rounded_rectangle(pill, radius=31, fill=(168, 92, 232, 185), outline=(255, 255, 255, 130), width=2)
    _draw_centered(draw, (WIDTH // 2, 261), SUBTITLE, sub_font, (255, 255, 255), (0, 0, 0), 2)


def _draw_date_grid(frame: Image.Image, current_date: datetime, grid_top: int, grid_bottom: int) -> None:
    draw = ImageDraw.Draw(frame, "RGBA")
    label_font = _load_font(24, True)
    start_month = datetime(current_date.year, current_date.month, 1)
    for offset_month in range(-1, 6):
        month = start_month.month + offset_month
        year = start_month.year + (month - 1) // 12
        month = (month - 1) % 12 + 1
        tick_date = datetime(year, month, 1)
        x = int(round(TICK_ORIGIN_X + (tick_date - current_date).days * PX_PER_DAY))
        if CHART_LEFT - 70 <= x <= WIDTH + 80:
            draw.line((x, grid_top, x, grid_bottom), fill=(255, 255, 255, 86), width=2)
            _draw_centered(draw, (x, grid_top - 32), tick_date.strftime("%b %Y"), label_font, (255, 255, 255), (18, 25, 45), 2)
        for day in [8, 15, 22]:
            minor_date = datetime(year, month, min(day, 28))
            mx = int(round(TICK_ORIGIN_X + (minor_date - current_date).days * PX_PER_DAY))
            if CHART_LEFT - 40 <= mx <= WIDTH + 40:
                draw.line((mx, grid_top + 96, mx, grid_bottom - 24), fill=(255, 255, 255, 24), width=1)


def _draw_clipped_logo_circle(
    frame: Image.Image,
    logo: Image.Image,
    circle_x: int,
    row_center: int,
    bar_right: int,
    outline: tuple[int, int, int],
) -> None:
    if circle_x <= bar_right - LOGO_RADIUS or circle_x > WIDTH + LOGO_RADIUS:
        return
    layer = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer, "RGBA")
    bbox = (circle_x - LOGO_RADIUS, row_center - LOGO_RADIUS, circle_x + LOGO_RADIUS, row_center + LOGO_RADIUS)
    draw.ellipse(bbox, fill=(250, 250, 253, 255), outline=(*outline, 255), width=3)
    mask = Image.new("L", (LOGO_RADIUS * 2 - 14, LOGO_RADIUS * 2 - 14), 0)
    ImageDraw.Draw(mask).ellipse((0, 0, mask.width - 1, mask.height - 1), fill=255)
    logo_tile = ImageOps.fit(logo, (mask.width, mask.height), method=Image.Resampling.LANCZOS)
    logo_tile.putalpha(ImageChops.multiply(logo_tile.getchannel("A"), mask))
    layer.alpha_composite(logo_tile, (circle_x - mask.width // 2, row_center - mask.height // 2))
    clip_mask = Image.new("L", (WIDTH, HEIGHT), 0)
    ImageDraw.Draw(clip_mask).rectangle((bar_right + 1, 0, WIDTH, HEIGHT), fill=255)
    layer.putalpha(ImageChops.multiply(layer.getchannel("A"), clip_mask))
    frame.alpha_composite(layer)


def render_video(
    rankings_path: Path,
    titles_path: Path,
    output_path: Path,
    photos_dir: Path,
    audio_path: Path,
    duration: float,
    fps: int,
    preview_image: Path | None,
    preview_progress: float,
) -> Path:
    snapshots = load_rankings(rankings_path)
    events = load_title_events(titles_path)
    if len(snapshots) < 2:
        raise RuntimeError("Not enough ATP ranking snapshots to render.")

    players = {player for snapshot in snapshots for player in snapshot.points}
    player_cards = {player: _load_player_card(player, photos_dir) for player in players}
    logo_cache = {event.logo_path: _logo_tile(event.logo_path, LOGO_RADIUS * 2 - 14) for event in events}
    background = _make_background()
    value_scale_max = max(max(snapshot.points.values()) for snapshot in snapshots) * 1.04
    row_centers = [ROW_TOP + i * ROW_GAP for i in range(TOP_N)]
    grid_top = ROW_TOP - 190
    grid_bottom = ROW_TOP + (TOP_N - 1) * ROW_GAP + 190
    value_font = _load_font(56, True)

    def make_frame(t: float) -> np.ndarray:
        if t <= INTRO_HOLD:
            progress = 0.0
        elif t >= duration - OUTRO_HOLD:
            progress = 1.0
        else:
            progress = (t - INTRO_HOLD) / max(0.01, duration - INTRO_HOLD - OUTRO_HOLD)

        current_date, states = _state_for_progress(snapshots, progress)
        frame = background.copy()
        _draw_header(frame, current_date)
        _draw_date_grid(frame, current_date, grid_top, grid_bottom)
        draw = ImageDraw.Draw(frame, "RGBA")
        draw.line((CHART_LEFT - 8, grid_top, CHART_LEFT - 8, grid_bottom), fill=(255, 255, 255, 255), width=4)

        render_rows = sorted(states, key=lambda item: item[1], reverse=True)
        row_by_player: dict[str, tuple[int, int, tuple[int, int, int]]] = {}
        for player, y_rank, points in render_rows:
            row_center = int(round(row_centers[0] + y_rank * ROW_GAP))
            if row_center > HEIGHT + 200 or row_center < 300:
                continue
            bar_w = int(MIN_BAR_W + (points / value_scale_max) * (MAX_BAR_W - MIN_BAR_W))
            bar_w = max(MIN_BAR_W, min(MAX_BAR_W, bar_w))
            bar_right = CHART_LEFT + bar_w
            draw.line((bar_right + 8, row_center, WIDTH, row_center), fill=(255, 255, 255, 58), width=2)
            row_by_player[player] = (row_center, bar_right, _hex_to_rgb(PLAYER_COLORS.get(player, "#F4F1E8")))

        for player, y_rank, points in render_rows:
            row_center = int(round(row_centers[0] + y_rank * ROW_GAP))
            if row_center > HEIGHT + 200 or row_center < 300:
                continue
            card = player_cards[player]
            frame.alpha_composite(card, (CARD_X, int(row_center - card.height / 2)))
            color = _hex_to_rgb(PLAYER_COLORS.get(player, "#F4F1E8"))
            text_color = (15, 22, 34) if sum(color) / 3 > 170 else (247, 248, 252)
            bar_w = int(MIN_BAR_W + (points / value_scale_max) * (MAX_BAR_W - MIN_BAR_W))
            bar_w = max(MIN_BAR_W, min(MAX_BAR_W, bar_w))
            bar_top = int(row_center - BAR_HEIGHT / 2)
            bar_bottom = bar_top + BAR_HEIGHT
            shadow = (CHART_LEFT + 8, bar_top + 9, CHART_LEFT + bar_w + 8, bar_bottom + 9)
            draw.rounded_rectangle(shadow, radius=18, fill=(0, 0, 0, 86))
            draw.rounded_rectangle(
                (CHART_LEFT, bar_top, CHART_LEFT + bar_w, bar_bottom),
                radius=18,
                fill=(*color, 255),
                outline=(*_mix(color, (255, 255, 255), 0.18), 255),
                width=2,
            )
            draw.rounded_rectangle(
                (CHART_LEFT + 8, bar_top + 8, CHART_LEFT + max(76, int(bar_w * 0.58)), bar_top + 24),
                radius=10,
                fill=(255, 255, 255, 62),
            )
            draw.line(
                (CHART_LEFT + 18, bar_bottom - 8, CHART_LEFT + max(34, int(bar_w * 0.8)), bar_bottom - 8),
                fill=(*_darken(color, 0.18), 92),
                width=3,
            )
            value_text = str(int(round(points)))
            value_w, _ = _text_size(draw, value_text, value_font)
            value_x = min(CHART_LEFT + bar_w - value_w // 2 - 22, CHART_LEFT + max(96, int(bar_w * 0.42)))
            value_x = max(CHART_LEFT + 94, value_x)
            _draw_centered(draw, (value_x, row_center + 2), value_text, value_font, text_color, (0, 0, 0) if text_color[0] > 100 else (255, 255, 255), 2)

        for event in events:
            row_data = row_by_player.get(event.player)
            if row_data is None:
                continue
            circle_x = int(round(TICK_ORIGIN_X + (event.date - current_date).days * PX_PER_DAY))
            if circle_x < CHART_LEFT - LOGO_RADIUS or circle_x > WIDTH + LOGO_RADIUS:
                continue
            row_center, bar_right, color = row_data
            _draw_clipped_logo_circle(frame, logo_cache[event.logo_path], circle_x, row_center, bar_right, color)

        return np.array(frame.convert("RGB"))

    if preview_image:
        preview_image.parent.mkdir(parents=True, exist_ok=True)
        active_duration = duration - INTRO_HOLD - OUTRO_HOLD
        preview_t = INTRO_HOLD + active_duration * preview_progress if active_duration > 0 else duration * preview_progress
        Image.fromarray(make_frame(preview_t)).save(preview_image)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    clip = VideoClip(make_frame, duration=duration)
    audio_clip, keep_alive = build_audio_track(audio_path, duration)
    clip = clip.with_audio(audio_clip)
    clip.write_videofile(str(output_path), fps=fps, codec="libx264", audio_codec="aac")
    clip.close()
    audio_clip.close()
    for item in keep_alive:
        item.close()
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate ATP top 4 ranking points race with title logos.")
    parser.add_argument("--rankings", type=Path, default=DEFAULT_RANKINGS)
    parser.add_argument("--titles", type=Path, default=DEFAULT_TITLES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--photos-dir", type=Path, default=DEFAULT_PHOTOS_DIR)
    parser.add_argument("--audio", type=Path, default=DEFAULT_AUDIO)
    parser.add_argument("--duration", type=float, default=DURATION)
    parser.add_argument("--fps", type=int, default=FPS)
    parser.add_argument("--preview-image", type=Path, default=DEFAULT_PREVIEW)
    parser.add_argument("--preview-progress", type=float, default=0.55)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = render_video(
        rankings_path=args.rankings,
        titles_path=args.titles,
        output_path=args.output,
        photos_dir=args.photos_dir,
        audio_path=args.audio,
        duration=args.duration,
        fps=args.fps,
        preview_image=args.preview_image,
        preview_progress=args.preview_progress,
    )
    print(f"[video_generator] ATP top 4 title-logo race generated -> {output}")


if __name__ == "__main__":
    main()
