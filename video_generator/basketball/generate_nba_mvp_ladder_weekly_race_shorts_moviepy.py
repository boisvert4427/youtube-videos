from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from moviepy import VideoClip
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from video_generator.generate_ucl_barchart_race_moviepy import build_audio_track

DEFAULT_INPUT = PROJECT_ROOT / "data" / "processed" / "basketball" / "nba_kia_mvp_ladder_2025_26_weekly.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "basketball" / "nba_mvp_ladder_cumulative_race_shorts.mp4"
DEFAULT_PREVIEW = PROJECT_ROOT / "tmp_frames" / "nba_mvp_ladder_cumulative_race_preview.png"
DEFAULT_ASSETS_DIR = PROJECT_ROOT / "data" / "raw" / "mvp_race_assets"
DEFAULT_AUDIO = PROJECT_ROOT / "data" / "raw" / "audio" / "Midnight_Grip_20260402_0828.mp3"

WIDTH = 1080
HEIGHT = 1920
FPS = 60
DURATION = 40.0
TOP_N = 5

TITLE = "NBA MVP LADDER RACE"
SUBTITLE = "Cumulative Weekly Points"
FOOTER = "Source: NBA.com Kia MVP Ladder"

BAR_LEFT = 302
BAR_MAX_WIDTH = 648
BAR_MIN_WIDTH = 92
ROW_TOP = 450
ROW_GAP = 238
ROW_HEIGHT = 132
PHOTO_SIZE = 124

RANK_SCORE_MAX = 1600.0
INTRO_HOLD = 1.2
OUTRO_HOLD = 2.0

PLAYER_META = {
    "Shai Gilgeous-Alexander": {"short": "SGA", "team": "OKC", "color": "#007AC1", "accent": "#EF3B24", "image": "sga.png"},
    "Nikola Jokic": {"short": "JOKIC", "team": "DEN", "color": "#FDB927", "accent": "#0E2240", "image": "jokic.png"},
    "Victor Wembanyama": {"short": "WEMBY", "team": "SAS", "color": "#C4CED4", "accent": "#111111", "image": "wemby.png"},
    "Luka Doncic": {"short": "LUKA", "team": "LAL", "color": "#552583", "accent": "#FDB927", "image": "luka.png"},
    "Jaylen Brown": {"short": "BROWN", "team": "BOS", "color": "#007A33", "accent": "#BA9653", "image": "jaylen_brown.png"},
    "Cade Cunningham": {"short": "CADE", "team": "DET", "color": "#1D42BA", "accent": "#C8102E", "image": "cade_cunningham.png"},
    "Giannis Antetokounmpo": {"short": "GIANNIS", "team": "MIL", "color": "#00471B", "accent": "#EEE1C6", "image": "giannis.png"},
    "Jalen Brunson": {"short": "BRUNSON", "team": "NYK", "color": "#006BB6", "accent": "#F58426", "image": "jalen_brunson.png"},
    "Tyrese Maxey": {"short": "MAXEY", "team": "PHI", "color": "#006BB6", "accent": "#ED174C", "image": "tyrese_maxey.png"},
}


@dataclass(frozen=True)
class PlayerFrame:
    player: str
    rank: int
    score: float


@dataclass(frozen=True)
class WeekSnapshot:
    week: int
    rows: list[PlayerFrame]


def _hex_to_rgb(value: str) -> tuple[int, int, int]:
    value = value.strip().lstrip("#")
    return tuple(int(value[i : i + 2], 16) for i in (0, 2, 4))


def _mix(a: tuple[int, int, int], b: tuple[int, int, int], amount: float) -> tuple[int, int, int]:
    amount = max(0.0, min(1.0, amount))
    return tuple(int(a[i] + (b[i] - a[i]) * amount) for i in range(3))


def _ease(value: float) -> float:
    value = max(0.0, min(1.0, value))
    return value * value * (3.0 - 2.0 * value)


def _load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
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
    box = draw.textbbox((0, 0), text, font=font, stroke_width=0)
    return box[2] - box[0], box[3] - box[1]


def _draw_centered(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int] | tuple[int, int, int, int],
    stroke_fill: tuple[int, int, int] | tuple[int, int, int, int] = (0, 0, 0, 180),
    stroke_width: int = 2,
) -> None:
    draw.text(xy, text, font=font, fill=fill, anchor="mm", stroke_fill=stroke_fill, stroke_width=stroke_width)


def _make_background() -> Image.Image:
    y = np.linspace(0.0, 1.0, HEIGHT)[:, None]
    x = np.linspace(0.0, 1.0, WIDTH)[None, :]
    left = np.array([205, 76, 236], dtype=np.float32)
    right = np.array([49, 202, 232], dtype=np.float32)
    bottom = np.array([122, 91, 245], dtype=np.float32)
    base = left * (1 - x[..., None]) + right * x[..., None]
    base = base * (1 - y[..., None] * 0.45) + bottom * (y[..., None] * 0.45)

    glow = np.exp(-(((x - 0.62) ** 2) / 0.10 + ((y - 0.24) ** 2) / 0.08))[..., None]
    magenta = np.exp(-(((x - 0.22) ** 2) / 0.12 + ((y - 0.72) ** 2) / 0.20))[..., None]
    base += glow * np.array([75, 36, 120]) + magenta * np.array([62, 0, 88])
    noise = (np.random.default_rng(31).normal(0, 1.8, (HEIGHT, WIDTH, 1))).astype(np.float32)
    base = np.clip(base + noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(base, "RGB").convert("RGBA")

    overlay = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay, "RGBA")
    for i in range(12):
        x0 = BAR_LEFT + i * 58
        od.line((x0, 335, x0, 1525), fill=(255, 255, 255, 34), width=1)
    od.ellipse((-180, 1070, 360, 1680), outline=(255, 255, 255, 18), width=3)
    od.ellipse((710, 100, 1320, 710), outline=(255, 255, 255, 14), width=3)
    return Image.alpha_composite(img, overlay).filter(ImageFilter.GaussianBlur(0.25))


def _crop_photo(path: Path, size: int, color: tuple[int, int, int], initials: str) -> Image.Image:
    if path.exists():
        raw = Image.open(path).convert("RGB")
        raw = ImageOps.fit(raw, (size, size), method=Image.Resampling.LANCZOS, centering=(0.5, 0.18))
    else:
        raw = Image.new("RGB", (size, size), _mix(color, (8, 10, 22), 0.25))
        d = ImageDraw.Draw(raw)
        for yy in range(size):
            shade = int(yy / size * 65)
            d.line((0, yy, size, yy), fill=_mix(color, (shade, shade, shade), 0.32))
        _draw_centered(d, (size // 2, size // 2), initials, _load_font(44, True), (255, 255, 255), stroke_width=2)

    mask = Image.new("L", (size, size), 0)
    md = ImageDraw.Draw(mask)
    md.rounded_rectangle((0, 0, size - 1, size - 1), radius=26, fill=255)

    out = Image.new("RGBA", (size + 16, size + 16), (0, 0, 0, 0))
    shadow = Image.new("RGBA", (size, size), (0, 0, 0, 150))
    out.alpha_composite(shadow.filter(ImageFilter.GaussianBlur(8)), (10, 12))
    photo = raw.convert("RGBA")
    photo.putalpha(mask)
    out.alpha_composite(photo, (4, 4))
    border = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    bd = ImageDraw.Draw(border)
    bd.rounded_rectangle((2, 2, size - 3, size - 3), radius=24, outline=(255, 255, 255, 235), width=4)
    bd.rounded_rectangle((8, 8, size - 9, size - 9), radius=20, outline=(*color, 180), width=2)
    out.alpha_composite(border, (4, 4))
    return out


def load_snapshots(path: Path) -> list[WeekSnapshot]:
    grouped: dict[int, list[PlayerFrame]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            week = int(row["week"])
            grouped.setdefault(week, []).append(
                PlayerFrame(player=row["player"], rank=int(row["rank"]), score=float(row["rank_score"]))
            )
    cumulative_scores: dict[str, float] = {}
    snapshots: list[WeekSnapshot] = []
    for week, rows in sorted(grouped.items()):
        for row in rows:
            cumulative_scores[row.player] = cumulative_scores.get(row.player, 0.0) + row.score

        ranked_rows = [
            PlayerFrame(player=player, rank=index + 1, score=score)
            for index, (player, score) in enumerate(
                sorted(cumulative_scores.items(), key=lambda item: (-item[1], item[0]))[:TOP_N]
            )
        ]
        snapshots.append(WeekSnapshot(week=week, rows=ranked_rows))
    return snapshots


def _rank_map(snapshot: WeekSnapshot) -> dict[str, int]:
    return {row.player: index for index, row in enumerate(snapshot.rows)}


def _score_map(snapshot: WeekSnapshot) -> dict[str, float]:
    return {row.player: row.score for row in snapshot.rows}


def _interpolate_state(snapshots: list[WeekSnapshot], progress: float) -> tuple[float, list[tuple[str, float, float, int]]]:
    if len(snapshots) == 1:
        snap = snapshots[0]
        return snap.week, [(row.player, float(i), row.score, row.rank) for i, row in enumerate(snap.rows)]

    scaled = max(0.0, min(1.0, progress)) * (len(snapshots) - 1)
    left_idx = min(int(math.floor(scaled)), len(snapshots) - 2)
    right_idx = left_idx + 1
    alpha = _ease(scaled - left_idx)
    left = snapshots[left_idx]
    right = snapshots[right_idx]
    left_ranks = _rank_map(left)
    right_ranks = _rank_map(right)
    left_scores = _score_map(left)
    right_scores = _score_map(right)
    players = set(left_ranks) | set(right_ranks)
    states: list[tuple[str, float, float, int]] = []
    for player in players:
        old_rank = float(left_ranks.get(player, TOP_N + 1))
        new_rank = float(right_ranks.get(player, TOP_N + 1))
        y_rank = old_rank + (new_rank - old_rank) * alpha
        old_score = left_scores.get(player, 0.0)
        new_score = right_scores.get(player, 0.0)
        score = old_score + (new_score - old_score) * alpha
        display_rank = max(1, min(TOP_N, int(round(y_rank)) + 1))
        if y_rank <= TOP_N - 0.15 or score > 1:
            states.append((player, y_rank, score, display_rank))
    week_value = left.week + (right.week - left.week) * alpha
    states.sort(key=lambda item: (item[1], -item[2], item[0]))
    return week_value, states[: TOP_N + 2]


def _build_cards(players: set[str], assets_dir: Path) -> dict[str, Image.Image]:
    cards = {}
    for player in players:
        meta = PLAYER_META.get(player, {})
        color = _hex_to_rgb(meta.get("color", "#4A6BFF"))
        short = meta.get("short", "".join(part[0] for part in player.split()[:2]).upper())
        image_name = meta.get("image", f"{player.lower().replace(' ', '_')}.png")
        cards[player] = _crop_photo(assets_dir / image_name, PHOTO_SIZE, color, short[:3])
    return cards


def _draw_header(frame: Image.Image, week_value: float, max_week: int) -> None:
    draw = ImageDraw.Draw(frame, "RGBA")
    title_font = _load_font(50, True)
    sub_font = _load_font(24, True)
    week_font = _load_font(28, True)
    small_font = _load_font(20, True)

    _draw_centered(draw, (WIDTH // 2, 98), TITLE, title_font, (255, 255, 255), (8, 12, 30), 4)
    pill = (308, 148, 772, 206)
    draw.rounded_rectangle(pill, radius=29, fill=(156, 83, 220, 145), outline=(255, 255, 255, 135), width=2)
    _draw_centered(draw, (WIDTH // 2, 177), SUBTITLE, sub_font, (255, 255, 255), (0, 0, 0), 2)

    week_int = int(round(week_value))
    _draw_centered(draw, (BAR_LEFT + 26, 258), f"Week {week_int}", week_font, (255, 255, 255), (20, 16, 48), 2)

    track_x0, track_x1 = BAR_LEFT + 118, WIDTH - 96
    track_y = 260
    draw.line((track_x0, track_y, track_x1, track_y), fill=(255, 255, 255, 60), width=6)
    pct = max(0.0, min(1.0, (week_value - 3) / max(1, max_week - 3)))
    draw.line((track_x0, track_y, int(track_x0 + (track_x1 - track_x0) * pct), track_y), fill=(255, 208, 98, 230), width=6)
    draw.ellipse((track_x0 + (track_x1 - track_x0) * pct - 10, track_y - 10, track_x0 + (track_x1 - track_x0) * pct + 10, track_y + 10), fill=(255, 255, 255, 245))
    _draw_centered(draw, (track_x0, track_y + 30), "W3", small_font, (255, 255, 255, 210), stroke_width=1)
    _draw_centered(draw, (track_x1, track_y + 30), f"W{max_week}", small_font, (255, 255, 255, 210), stroke_width=1)


def _draw_row(
    frame: Image.Image,
    player: str,
    y_rank: float,
    score: float,
    display_rank: int,
    card: Image.Image,
) -> None:
    draw = ImageDraw.Draw(frame, "RGBA")
    meta = PLAYER_META.get(player, {})
    color = _hex_to_rgb(meta.get("color", "#4A6BFF"))
    accent = _hex_to_rgb(meta.get("accent", "#FFFFFF"))
    y = int(ROW_TOP + y_rank * ROW_GAP)
    alpha = int(255 * max(0.0, min(1.0, 1.0 - max(0.0, y_rank - (TOP_N - 1)) * 0.75)))
    if alpha <= 4:
        return

    photo = card.copy()
    if alpha < 255:
        a = photo.getchannel("A").point(lambda v: int(v * alpha / 255))
        photo.putalpha(a)
    frame.alpha_composite(photo, (82, y - PHOTO_SIZE // 2 - 8))

    bar_w = int(BAR_MIN_WIDTH + (score / RANK_SCORE_MAX) * (BAR_MAX_WIDTH - BAR_MIN_WIDTH))
    bar_w = max(BAR_MIN_WIDTH, min(BAR_MAX_WIDTH, bar_w))
    bar_h = ROW_HEIGHT
    x0 = BAR_LEFT
    y0 = y - bar_h // 2
    x1 = x0 + bar_w
    y1 = y0 + bar_h

    shadow = (x0 + 8, y0 + 9, x1 + 8, y1 + 9)
    draw.rounded_rectangle(shadow, radius=16, fill=(0, 0, 0, int(112 * alpha / 255)))
    draw.rounded_rectangle((x0, y0, x1, y1), radius=14, fill=(*color, alpha), outline=(*_mix(color, (255, 255, 255), 0.30), alpha), width=2)
    draw.rounded_rectangle((x0 + 12, y0 + 12, x0 + max(72, int(bar_w * 0.42)), y0 + 28), radius=8, fill=(255, 255, 255, int(220 * alpha / 255)))
    draw.line((x0 + 22, y1 - 13, x0 + max(74, int(bar_w * 0.76)), y1 - 13), fill=(*_mix(color, (0, 0, 0), 0.30), int(120 * alpha / 255)), width=3)

    score_font = _load_font(56, True)
    rank_font = _load_font(19, True)

    draw.rounded_rectangle((x0 - 58, y - 22, x0 - 14, y + 22), radius=16, fill=(255, 255, 255, int(210 * alpha / 255)), outline=(*accent, alpha), width=2)
    _draw_centered(draw, (x0 - 36, y + 1), str(display_rank), rank_font, (*color, alpha), (255, 255, 255, alpha), 1)

    value_text = f"{int(round(score))}"
    value_w, _ = _text_size(draw, value_text, score_font)
    value_x = min(x1 - value_w // 2 - 24, x0 + bar_w - 50)
    value_x = max(x0 + 88, value_x)
    _draw_centered(draw, (value_x, y + 4), value_text, score_font, (255, 255, 255, alpha), (0, 0, 0, alpha), 3)


def _draw_footer(frame: Image.Image) -> None:
    draw = ImageDraw.Draw(frame, "RGBA")
    footer_font = _load_font(22, True)
    draw.rounded_rectangle((322, HEIGHT - 152, 758, HEIGHT - 96), radius=28, fill=(156, 83, 220, 95), outline=(255, 255, 255, 45), width=1)
    _draw_centered(draw, (WIDTH // 2, HEIGHT - 124), FOOTER, footer_font, (255, 255, 255, 190), stroke_width=1)


def render_video(
    input_csv: Path,
    output_path: Path,
    assets_dir: Path,
    audio_path: Path,
    duration: float,
    fps: int,
    preview_image: Path | None = None,
) -> Path:
    snapshots = load_snapshots(input_csv)
    global RANK_SCORE_MAX
    RANK_SCORE_MAX = max(RANK_SCORE_MAX, max(row.score for snapshot in snapshots for row in snapshot.rows))
    players = {row.player for snapshot in snapshots for row in snapshot.rows}
    cards = _build_cards(players, assets_dir)
    background = _make_background()
    max_week = max(snapshot.week for snapshot in snapshots)

    def make_frame(t: float) -> np.ndarray:
        if t <= INTRO_HOLD:
            progress = 0.0
        elif t >= duration - OUTRO_HOLD:
            progress = 1.0
        else:
            progress = (t - INTRO_HOLD) / max(0.01, duration - INTRO_HOLD - OUTRO_HOLD)
        week_value, states = _interpolate_state(snapshots, progress)
        frame = background.copy()
        _draw_header(frame, week_value, max_week)
        draw = ImageDraw.Draw(frame, "RGBA")
        draw.line((BAR_LEFT - 8, ROW_TOP - 130, BAR_LEFT - 8, ROW_TOP + (TOP_N - 1) * ROW_GAP + 134), fill=(255, 255, 255, 235), width=4)
        draw.line((BAR_LEFT - 16, ROW_TOP - 130, BAR_LEFT - 16, ROW_TOP + (TOP_N - 1) * ROW_GAP + 134), fill=(255, 255, 255, 100), width=2)

        # Draw lower rows first so the player moving upward visibly passes over.
        for player, y_rank, score, display_rank in sorted(states, key=lambda item: item[1], reverse=True):
            _draw_row(frame, player, y_rank, score, display_rank, cards[player])
        _draw_footer(frame)
        return np.array(frame.convert("RGB"))

    if preview_image:
        preview_image.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(make_frame(duration * 0.62)).save(preview_image)

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
    parser = argparse.ArgumentParser(description="Generate a vertical NBA Kia MVP Ladder weekly race Short.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--assets-dir", type=Path, default=DEFAULT_ASSETS_DIR)
    parser.add_argument("--audio", type=Path, default=DEFAULT_AUDIO)
    parser.add_argument("--duration", type=float, default=DURATION)
    parser.add_argument("--fps", type=int, default=FPS)
    parser.add_argument("--preview-image", type=Path, default=DEFAULT_PREVIEW)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = render_video(
        input_csv=args.input,
        output_path=args.output,
        assets_dir=args.assets_dir,
        audio_path=args.audio,
        duration=args.duration,
        fps=args.fps,
        preview_image=args.preview_image,
    )
    print(f"[video_generator] NBA MVP Ladder weekly race generated -> {output}")


if __name__ == "__main__":
    main()
