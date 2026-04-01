from __future__ import annotations

import argparse
import math
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from moviepy import CompositeVideoClip, ImageClip, VideoClip, concatenate_videoclips
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageOps


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "football" / "messi_vs_ronaldo_stats_shorts.mp4"
DEFAULT_PHOTOS_DIR = PROJECT_ROOT / "data" / "raw" / "player_photos"

WIDTH = 1080
HEIGHT = 1920
FPS = 30
STAT_DURATION = 2.5
INTRO_DURATION = 2.0
OUTRO_DURATION = 2.0

MESSI_BLUE = (52, 150, 255)
RONALDO_RED = (236, 61, 71)
WHITE = (246, 249, 252)
PANEL = (9, 16, 30, 228)


@dataclass(frozen=True)
class StatRow:
    label: str
    left: float
    right: float
    inverse: bool = False


STATS = [
    StatRow("GOALS", 168, 186, False),
    StatRow("ASSISTS", 99, 34, False),
    StatRow("G/A", 267, 220, False),
    StatRow("FREE KICKS", 14, 7, False),
    StatRow("PENALTIES", 23, 45, False),
    StatRow("MIN/GOAL", 127.4, 113, True),
    StatRow("MIN/G/A", 80.1, 95.6, True),
]


def _ease_out(value: float) -> float:
    value = min(max(value, 0.0), 1.0)
    return 1.0 - (1.0 - value) ** 3


def _ease_in_out(value: float) -> float:
    value = min(max(value, 0.0), 1.0)
    return 0.5 - 0.5 * math.cos(math.pi * value)


def _load_font(size: int, bold: bool = False):
    candidates = [
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def _fit_font(text: str, max_width: int, start_size: int, min_size: int, bold: bool = True):
    probe = ImageDraw.Draw(Image.new("RGB", (10, 10), "black"))
    size = start_size
    while size >= min_size:
        font = _load_font(size, bold=bold)
        if probe.textbbox((0, 0), text, font=font)[2] <= max_width:
            return font
        size -= 1
    return _load_font(min_size, bold=bold)


def _slugify(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    return "".join(ch if ch.isalnum() else "_" for ch in normalized.lower()).strip("_")


def _resolve_player_photo(player_name: str, photos_dir: Path) -> Path | None:
    custom_candidates = {
        "messi": ["lionel_messi", "leo_messi", "messi"],
        "ronaldo": ["cristiano_ronaldo", "ronaldo", "cr7"],
    }
    names = custom_candidates["messi" if "messi" in player_name.lower() else "ronaldo"]
    for stem in names:
        for ext in (".jpg", ".jpeg", ".png", ".webp"):
            candidate = photos_dir / f"{stem}{ext}"
            if candidate.exists():
                return candidate
    slug = _slugify(player_name)
    for ext in (".jpg", ".jpeg", ".png", ".webp"):
        candidate = photos_dir / f"{slug}{ext}"
        if candidate.exists():
            return candidate
    return None


def _make_background() -> np.ndarray:
    xx = np.linspace(0, 1, WIDTH, dtype=np.float32)
    yy = np.linspace(0, 1, HEIGHT, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(xx, yy)

    black = np.array([4, 6, 12], dtype=np.float32)
    navy = np.array([8, 18, 38], dtype=np.float32)
    blue = np.array(MESSI_BLUE, dtype=np.float32)
    red = np.array(RONALDO_RED, dtype=np.float32)
    white = np.array(WHITE, dtype=np.float32)

    left_glow = np.exp(-(((grid_x - 0.18) / 0.22) ** 2 + ((grid_y - 0.34) / 0.20) ** 2))
    right_glow = np.exp(-(((grid_x - 0.82) / 0.22) ** 2 + ((grid_y - 0.34) / 0.20) ** 2))
    center_glow = np.exp(-(((grid_x - 0.50) / 0.24) ** 2 + ((grid_y - 0.62) / 0.22) ** 2))
    mix = np.clip(0.65 * grid_y + 0.18 * np.abs(grid_x - 0.5), 0.0, 1.0)

    frame = np.clip(
        black[None, None, :] * (1.0 - mix[..., None])
        + navy[None, None, :] * (0.90 * mix[..., None])
        + blue[None, None, :] * (0.18 * left_glow[..., None])
        + red[None, None, :] * (0.18 * right_glow[..., None])
        + white[None, None, :] * (0.06 * center_glow[..., None]),
        0,
        255,
    ).astype(np.uint8)
    return frame


def _player_tile(player_name: str, photo_path: Path | None, accent: tuple[int, int, int]) -> Image.Image:
    tile = Image.new("RGBA", (420, 300), (0, 0, 0, 0))
    draw = ImageDraw.Draw(tile, "RGBA")
    draw.rounded_rectangle((2, 2, 418, 298), radius=36, fill=PANEL, outline=(*accent, 190), width=3)
    glow = Image.new("RGBA", tile.size, (0, 0, 0, 0))
    glow_draw = ImageDraw.Draw(glow, "RGBA")
    glow_draw.ellipse((36, 78, 210, 252), fill=(*accent, 80))
    glow = glow.filter(ImageFilter.GaussianBlur(radius=24))
    tile.alpha_composite(glow)

    portrait_size = 126
    if photo_path and photo_path.exists():
        source = ImageOps.exif_transpose(Image.open(photo_path)).convert("RGB")
        source = ImageOps.fit(source, (portrait_size, portrait_size), method=Image.Resampling.LANCZOS, centering=(0.5, 0.18))
        source = ImageEnhance.Contrast(source).enhance(1.08)
        source = ImageEnhance.Brightness(source).enhance(1.04)
        source_rgba = source.convert("RGBA")
    else:
        source_rgba = Image.new("RGBA", (portrait_size, portrait_size), (230, 234, 240, 255))
        temp_draw = ImageDraw.Draw(source_rgba)
        initials = "".join(part[0] for part in player_name.split()[:2]).upper()
        initials_font = _fit_font(initials, portrait_size - 24, 52, 24, True)
        temp_draw.text((portrait_size // 2, portrait_size // 2), initials, font=initials_font, fill="#1b2a38", anchor="mm")

    mask = Image.new("L", (portrait_size, portrait_size), 0)
    ImageDraw.Draw(mask).ellipse((0, 0, portrait_size - 1, portrait_size - 1), fill=255)
    ring = Image.new("RGBA", (portrait_size + 16, portrait_size + 16), (0, 0, 0, 0))
    ring_draw = ImageDraw.Draw(ring, "RGBA")
    ring_draw.ellipse((0, 0, portrait_size + 15, portrait_size + 15), fill=(255, 255, 255, 236))
    ring_draw.ellipse((4, 4, portrait_size + 11, portrait_size + 11), fill=(*accent, 255))
    ring_draw.ellipse((8, 8, portrait_size + 7, portrait_size + 7), fill=(8, 16, 28, 255))
    portrait_circle = Image.new("RGBA", (portrait_size, portrait_size), (0, 0, 0, 0))
    portrait_circle.paste(source_rgba, (0, 0), mask)
    ring.alpha_composite(portrait_circle, (8, 8))
    tile.alpha_composite(ring, (24, 80))

    name_font = _fit_font(player_name.upper(), 210, 44, 24, True)
    sub_font = _load_font(22, bold=False)
    draw.text((190, 138), player_name.upper(), font=name_font, fill="#ffffff", anchor="lm")
    draw.text((190, 182), "LAST 250 APPEARANCES", font=sub_font, fill="#dbe6f2", anchor="lm")
    return tile


def _text_image(
    text: str,
    width: int,
    height: int,
    font_size: int,
    color: tuple[int, int, int] | str = WHITE,
    bold: bool = True,
    stroke: int = 0,
    stroke_color: tuple[int, int, int] | str = "#08111d",
    align: str = "center",
) -> Image.Image:
    image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image, "RGBA")
    font = _fit_font(text, width - 12, font_size, max(12, int(font_size * 0.5)), bold)
    anchor = {"left": "lm", "center": "mm", "right": "rm"}[align]
    pos = {"left": (6, height // 2), "center": (width // 2, height // 2), "right": (width - 6, height // 2)}[align]
    draw.text(pos, text, font=font, fill=color, anchor=anchor, stroke_width=stroke, stroke_fill=stroke_color)
    return image


def _format_value(value: float) -> str:
    if float(value).is_integer():
        return f"{int(value)}"
    return f"{value:.1f}"


def _winner(left_value: float, right_value: float, inverse: bool) -> str:
    if abs(left_value - right_value) < 1e-9:
        return "tie"
    if inverse:
        return "left" if left_value < right_value else "right"
    return "left" if left_value > right_value else "right"


def _score_totals(stats: list[StatRow]) -> tuple[int, int]:
    left_score = 0
    right_score = 0
    for stat in stats:
        side = _winner(stat.left, stat.right, stat.inverse)
        if side == "left":
            left_score += 1
        elif side == "right":
            right_score += 1
    return left_score, right_score


def _dynamic_number_clip(
    start_value: float,
    end_value: float,
    duration: float,
    width: int,
    height: int,
    color: tuple[int, int, int],
    align: str,
) -> VideoClip:
    def make_frame(t: float) -> np.ndarray:
        progress = _ease_out(t / duration if duration else 1.0)
        current = start_value + (end_value - start_value) * progress
        text = _format_value(current)
        font_boost = 1.0 + 0.06 * math.sin(progress * math.pi)
        img = _text_image(text, width, height, int(68 * font_boost), color=color, bold=True, stroke=2, align=align)
        return np.array(img.convert("RGB"))

    return VideoClip(make_frame, duration=duration)


def _bar_clip(
    final_width: int,
    height: int,
    color: tuple[int, int, int],
    duration: float,
    winner: bool = False,
    side: str = "left",
) -> VideoClip:
    def make_frame(t: float) -> np.ndarray:
        progress = _ease_in_out(t / duration if duration else 1.0)
        width = max(4, int(final_width * progress))
        img = Image.new("RGBA", (max(final_width, 4), height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img, "RGBA")
        alpha = 255 if winner else 205
        rect = (0, 0, width, height) if side == "right" else (final_width - width, 0, final_width, height)
        draw.rounded_rectangle(rect, radius=height // 2, fill=(*color, alpha))
        if winner:
            sheen = Image.new("RGBA", img.size, (0, 0, 0, 0))
            sheen_draw = ImageDraw.Draw(sheen, "RGBA")
            if side == "right":
                sheen_draw.rounded_rectangle((0, 0, max(10, width // 2), max(8, height // 3)), radius=height // 3, fill=(255, 255, 255, 54))
            else:
                x0 = final_width - width
                x1 = min(final_width, x0 + max(10, width // 2))
                sheen_draw.rounded_rectangle((x0, 0, x1, max(8, height // 3)), radius=height // 3, fill=(255, 255, 255, 54))
            img.alpha_composite(sheen)
        return np.array(img.convert("RGB"))

    return VideoClip(make_frame, duration=duration)


def _background_clip(duration: float) -> VideoClip:
    base = Image.fromarray(_make_background()).convert("RGBA")

    def make_frame(t: float) -> np.ndarray:
        frame = base.copy()
        pulse = 0.5 + 0.5 * math.sin(t * 2.4)
        pulse2 = 0.5 + 0.5 * math.sin(t * 2.0 + 1.2)
        overlay = Image.new("RGBA", frame.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay, "RGBA")
        draw.ellipse((-120, 120, 430, 920), fill=(*MESSI_BLUE, int(26 + 34 * pulse)))
        draw.ellipse((650, 120, 1210, 920), fill=(*RONALDO_RED, int(26 + 34 * pulse2)))
        draw.rounded_rectangle((40, 40, WIDTH - 40, HEIGHT - 40), radius=42, outline=(255, 255, 255, 16), width=2)
        for idx in range(8):
            offset = int((t * 140 + idx * 170) % (HEIGHT + 320)) - 160
            alpha = 14 if idx % 2 == 0 else 10
            draw.rounded_rectangle((70 + idx * 20, offset, 118 + idx * 20, offset + 320), radius=24, fill=(255, 255, 255, alpha))
            draw.rounded_rectangle((WIDTH - 118 - idx * 20, HEIGHT - offset - 320, WIDTH - 70 - idx * 20, HEIGHT - offset), radius=24, fill=(255, 255, 255, alpha))
        overlay = overlay.filter(ImageFilter.GaussianBlur(radius=18))
        frame.alpha_composite(overlay)
        return np.array(frame.convert("RGB"))

    return VideoClip(make_frame, duration=duration)


def _score_badge(text: str, accent: tuple[int, int, int]) -> Image.Image:
    badge = Image.new("RGBA", (160, 82), (0, 0, 0, 0))
    draw = ImageDraw.Draw(badge, "RGBA")
    draw.rounded_rectangle((0, 0, 160, 82), radius=28, fill=(10, 18, 32, 228), outline=(*accent, 180), width=2)
    draw.text((80, 41), text, font=_fit_font(text, 132, 42, 20, True), fill="#ffffff", anchor="mm")
    return badge


def _intro_scene(left_card: np.ndarray, right_card: np.ndarray, duration: float) -> CompositeVideoClip:
    bg = _background_clip(duration)
    left = ImageClip(left_card).with_duration(duration).with_position((58, 170))
    right = ImageClip(right_card).with_duration(duration).with_position((WIDTH - 478, 170))
    left = left.resized(lambda t: 1.0 + 0.018 * (t / duration))
    right = right.resized(lambda t: 1.0 + 0.018 * (t / duration))

    title = ImageClip(np.array(_text_image("MESSI vs RONALDO", 840, 132, 82, WHITE, True, 3))).with_duration(duration)
    title = title.with_position(("center", 720))
    subtitle = ImageClip(np.array(_text_image("LAST 250 APPEARANCES", 620, 74, 34, "#dbe6f2", True, 1))).with_duration(duration)
    subtitle = subtitle.with_position(("center", 842))
    pulse = ImageClip(np.array(_score_badge("STAT BATTLE", (255, 210, 84)))).with_duration(duration).with_position(("center", 930))

    return CompositeVideoClip([bg, left, right, title, subtitle, pulse], size=(WIDTH, HEIGHT)).with_duration(duration)


def _stat_scene(
    stat: StatRow,
    index: int,
    left_card: np.ndarray,
    right_card: np.ndarray,
    duration: float,
    left_score: int,
    right_score: int,
    total_stats: int,
) -> CompositeVideoClip:
    bg = _background_clip(duration)

    left = ImageClip(left_card).with_duration(duration).with_position((58, 110))
    right = ImageClip(right_card).with_duration(duration).with_position((WIDTH - 478, 110))
    left = left.resized(lambda t: 1.0 + 0.014 * _ease_in_out(t / duration))
    right = right.resized(lambda t: 1.0 + 0.014 * _ease_in_out(t / duration))

    panel_img = Image.new("RGBA", (980, 920), (0, 0, 0, 0))
    draw = ImageDraw.Draw(panel_img, "RGBA")
    draw.rounded_rectangle((0, 0, 980, 920), radius=48, fill=PANEL, outline=(255, 255, 255, 22), width=2)
    draw.rounded_rectangle((40, 104, 940, 190), radius=28, fill=(19, 39, 72, 255))
    draw.text((490, 147), stat.label, font=_fit_font(stat.label, 760, 46, 24, True), fill="#ffffff", anchor="mm")
    draw.text((490, 50), f"STAT {index + 1}/{total_stats}", font=_load_font(24, bold=True), fill="#d2dff1", anchor="mm")
    panel_clip = ImageClip(np.array(panel_img.convert("RGB"))).with_duration(duration).with_position((50, 520))

    left_value = stat.left
    right_value = stat.right
    max_value = max(left_value, right_value)
    scale = 340.0 / max_value if max_value else 0.0
    left_bar_final = max(12, int(left_value * scale))
    right_bar_final = max(12, int(right_value * scale))
    winner = _winner(left_value, right_value, stat.inverse)

    bar_center_x = WIDTH // 2
    bar_y = 844
    base_left = ImageClip(np.full((38, 346, 3), 28, dtype=np.uint8)).with_duration(duration).with_position((bar_center_x - 370, bar_y))
    base_right = ImageClip(np.full((38, 346, 3), 28, dtype=np.uint8)).with_duration(duration).with_position((bar_center_x + 24, bar_y))

    left_bar = _bar_clip(left_bar_final, 38, MESSI_BLUE, duration * 0.78, winner == "left", "left").with_duration(duration)
    left_bar = left_bar.with_position((bar_center_x - 370 + 346 - left_bar_final, bar_y))
    right_bar = _bar_clip(right_bar_final, 38, RONALDO_RED, duration * 0.78, winner == "right", "right").with_duration(duration)
    right_bar = right_bar.with_position((bar_center_x + 24, bar_y))

    left_num = _dynamic_number_clip(0.0, left_value, duration * 0.78, 240, 88, WHITE, "right").with_duration(duration)
    left_num = left_num.with_position((bar_center_x - 440, 742))
    right_num = _dynamic_number_clip(0.0, right_value, duration * 0.78, 240, 88, WHITE, "left").with_duration(duration)
    right_num = right_num.with_position((bar_center_x + 200, 742))

    inverse_text = "LOWER IS BETTER" if stat.inverse else "HIGHER IS BETTER"
    note = ImageClip(np.array(_text_image(inverse_text, 280, 48, 22, "#bcd4ea", True, 1))).with_duration(duration)
    note = note.with_position(("center", 906))

    left_score_clip = ImageClip(np.array(_score_badge(f"{left_score}", MESSI_BLUE))).with_duration(duration).with_position((120, 1536))
    right_score_clip = ImageClip(np.array(_score_badge(f"{right_score}", RONALDO_RED))).with_duration(duration).with_position((WIDTH - 280, 1536))
    score_label = ImageClip(np.array(_text_image("STATS WON", 260, 64, 30, "#dbe6f2", True, 1))).with_duration(duration).with_position(("center", 1546))

    winner_glow = None
    if winner != "tie":
        accent = MESSI_BLUE if winner == "left" else RONALDO_RED
        winner_box = Image.new("RGBA", (240, 70), (0, 0, 0, 0))
        wd = ImageDraw.Draw(winner_box, "RGBA")
        wd.rounded_rectangle((0, 0, 240, 70), radius=26, fill=(*accent, 42), outline=(*accent, 180), width=2)
        wd.text((120, 35), "EDGE", font=_fit_font("EDGE", 180, 34, 18, True), fill="#ffffff", anchor="mm")
        winner_glow = ImageClip(np.array(winner_box.convert("RGB"))).with_duration(duration)
        winner_glow = winner_glow.with_position((120 if winner == "left" else WIDTH - 360, 654))

    clips = [
        bg,
        left,
        right,
        panel_clip,
        base_left,
        base_right,
        left_bar,
        right_bar,
        left_num,
        right_num,
        note,
        left_score_clip,
        right_score_clip,
        score_label,
    ]
    if winner_glow is not None:
        clips.append(winner_glow)
    return CompositeVideoClip(clips, size=(WIDTH, HEIGHT)).with_duration(duration)


def _outro_scene(left_card: np.ndarray, right_card: np.ndarray, duration: float, left_total: int, right_total: int) -> CompositeVideoClip:
    bg = _background_clip(duration)
    left = ImageClip(left_card).with_duration(duration).with_position((58, 170))
    right = ImageClip(right_card).with_duration(duration).with_position((WIDTH - 478, 170))
    left = left.resized(lambda t: 1.0 + 0.02 * _ease_in_out(t / duration))
    right = right.resized(lambda t: 1.0 + 0.02 * _ease_in_out(t / duration))

    main = ImageClip(np.array(_text_image("WHO IS THE GOAT?", 920, 160, 88, WHITE, True, 3))).with_duration(duration)
    main = main.with_position(("center", 760))
    left_score_clip = ImageClip(np.array(_score_badge(f"MESSI {left_total}", MESSI_BLUE))).with_duration(duration).with_position((120, 980))
    right_score_clip = ImageClip(np.array(_score_badge(f"RONALDO {right_total}", RONALDO_RED))).with_duration(duration).with_position((WIDTH - 360, 980))
    react = ImageClip(np.array(_text_image("COMMENT YOUR PICK", 760, 86, 46, "#ffe9a8", True, 2))).with_duration(duration)
    react = react.with_position(("center", 1142))

    return CompositeVideoClip([bg, left, right, main, left_score_clip, right_score_clip, react], size=(WIDTH, HEIGHT)).with_duration(duration)


def render_video(output_path: Path, messi_image: Path | None, ronaldo_image: Path | None, photos_dir: Path, fps: int) -> Path:
    total_duration = INTRO_DURATION + STAT_DURATION * len(STATS) + OUTRO_DURATION
    left_card = np.array(_player_tile("Messi", messi_image or _resolve_player_photo("Messi", photos_dir), MESSI_BLUE).convert("RGB"))
    right_card = np.array(_player_tile("Ronaldo", ronaldo_image or _resolve_player_photo("Ronaldo", photos_dir), RONALDO_RED).convert("RGB"))

    left_running = 0
    right_running = 0
    stat_scenes = []
    for idx, stat in enumerate(STATS):
        side = _winner(stat.left, stat.right, stat.inverse)
        if side == "left":
            left_running += 1
        elif side == "right":
            right_running += 1
        stat_scenes.append(
            _stat_scene(stat, idx, left_card, right_card, STAT_DURATION, left_running, right_running, len(STATS))
        )

    left_total, right_total = _score_totals(STATS)
    final_clip = concatenate_videoclips(
        [_intro_scene(left_card, right_card, INTRO_DURATION), *stat_scenes, _outro_scene(left_card, right_card, OUTRO_DURATION, left_total, right_total)],
        method="compose",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_clip.write_videofile(
        str(output_path),
        fps=fps,
        codec="libx264",
        audio=False,
        preset="medium",
        ffmpeg_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
    )
    final_clip.close()
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a Messi vs Ronaldo vertical Shorts stats battle video.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--messi-image", type=Path, default=None)
    parser.add_argument("--ronaldo-image", type=Path, default=None)
    parser.add_argument("--photos-dir", type=Path, default=DEFAULT_PHOTOS_DIR)
    parser.add_argument("--fps", type=int, default=FPS)
    args = parser.parse_args()

    render_video(
        output_path=args.output,
        messi_image=args.messi_image,
        ronaldo_image=args.ronaldo_image,
        photos_dir=args.photos_dir,
        fps=args.fps,
    )
    print(f"[video_generator] Messi vs Ronaldo Shorts generated -> {args.output}")


if __name__ == "__main__":
    main()
