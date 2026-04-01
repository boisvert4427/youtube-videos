from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from moviepy import VideoClip
from PIL import Image, ImageChops, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageOps

from video_generator.generate_ucl_barchart_race_moviepy import DEFAULT_AUDIO, build_audio_track


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "football" / "messi_vs_ronaldo_career_shorts.mp4"
DEFAULT_PHOTOS_DIR = PROJECT_ROOT / "data" / "raw" / "player_photos"

WIDTH = 1080
HEIGHT = 1920
FPS = 30
DURATION = 40.0
TITLE_HOLD = 1.2
OUTRO_HOLD = 2.6

MESSI_BLUE = (70, 196, 255)
RONALDO_RED = (255, 90, 52)
GOLD = (255, 214, 134)
WHITE = (244, 242, 236)
TRACK = (110, 116, 124, 110)


@dataclass(frozen=True)
class StatRow:
    label: str
    left: float
    right: float
    inverse: bool = False


STATS = [
    StatRow("BUTS", 900, 965, False),
    StatRow("PASSES DECISIVES", 407, 290, False),
    StatRow("CHAMPIONS LEAGUE", 4, 5, False),
    StatRow("CHAMPIONNATS", 12, 7, False),
    StatRow("COUPE DU MONDE", 1, 0, False),
    StatRow("EURO / COPA", 2, 1, False),
    StatRow("BALLON D'OR", 8, 5, False),
    StatRow("TOTAL TROPHIES", 47, 34, False),
]


def _load_font(size: int, bold: bool = False):
    candidates = [
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf",
    ]
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def _fit_font(draw: ImageDraw.ImageDraw, text: str, max_width: int, start_size: int, min_size: int, bold: bool = True):
    size = start_size
    while size >= min_size:
        font = _load_font(size, bold=bold)
        bbox = draw.textbbox((0, 0), text, font=font)
        if bbox[2] - bbox[0] <= max_width:
            return font
        size -= 1
    return _load_font(min_size, bold=bold)


def _ease_out(value: float) -> float:
    value = min(max(value, 0.0), 1.0)
    return 1.0 - (1.0 - value) ** 3


def _ease_in_out(value: float) -> float:
    value = min(max(value, 0.0), 1.0)
    return 0.5 - 0.5 * math.cos(math.pi * value)


def _resolve_player_photo(player_name: str, photos_dir: Path) -> Path | None:
    stems = {
        "messi": ["lionel_messi", "leo_messi", "messi"],
        "ronaldo": ["cristiano_ronaldo", "ronaldo", "cr7"],
    }["messi" if "messi" in player_name.lower() else "ronaldo"]
    for stem in stems:
        for ext in (".jpg", ".jpeg", ".png", ".webp"):
            candidate = photos_dir / f"{stem}{ext}"
            if candidate.exists():
                return candidate
    return None


def _winner(left: float, right: float, inverse: bool) -> str:
    if abs(left - right) < 1e-9:
        return "tie"
    if inverse:
        return "left" if left < right else "right"
    return "left" if left > right else "right"


def _score_until(row_index: int) -> tuple[int, int]:
    left_score = 0
    right_score = 0
    for idx, stat in enumerate(STATS):
        if idx > row_index:
            break
        side = _winner(stat.left, stat.right, stat.inverse)
        if side == "left":
            left_score += 1
        elif side == "right":
            right_score += 1
    return left_score, right_score


def _make_background() -> Image.Image:
    xx = np.linspace(0, 1, WIDTH, dtype=np.float32)
    yy = np.linspace(0, 1, HEIGHT, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(xx, yy)
    black = np.array([8, 8, 12], dtype=np.float32)
    steel = np.array([30, 28, 35], dtype=np.float32)
    blue = np.array(MESSI_BLUE, dtype=np.float32)
    red = np.array(RONALDO_RED, dtype=np.float32)
    mix = np.clip(0.70 * grid_y + 0.08 * np.abs(grid_x - 0.5), 0, 1)
    left_glow = np.exp(-(((grid_x - 0.13) / 0.16) ** 2 + ((grid_y - 0.44) / 0.18) ** 2))
    right_glow = np.exp(-(((grid_x - 0.87) / 0.16) ** 2 + ((grid_y - 0.44) / 0.18) ** 2))
    bottom_fog = np.exp(-(((grid_y - 0.92) / 0.18) ** 2))

    frame = np.clip(
        black[None, None, :] * (1.0 - mix[..., None])
        + steel[None, None, :] * (0.86 * mix[..., None])
        + blue[None, None, :] * (0.25 * left_glow[..., None])
        + red[None, None, :] * (0.25 * right_glow[..., None]),
        0,
        255,
    ).astype(np.uint8)
    image = Image.fromarray(frame, "RGB").convert("RGBA")

    fog = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    fd = ImageDraw.Draw(fog, "RGBA")
    for i in range(14):
        alpha = int(24 + i * 4)
        y0 = HEIGHT - 380 + i * 18
        fd.ellipse((-80 - i * 20, y0, WIDTH + 80 + i * 20, y0 + 250), fill=(255, 255, 255, alpha))
    fog = fog.filter(ImageFilter.GaussianBlur(radius=40))
    fog_np = np.array(fog, dtype=np.float32)
    fog_np[..., :3] *= bottom_fog[..., None]
    fog_np[..., 3] *= bottom_fog
    image.alpha_composite(Image.fromarray(np.clip(fog_np, 0, 255).astype(np.uint8), "RGBA"))
    return image


def _load_player_cutout(player_name: str, photos_dir: Path, accent: tuple[int, int, int], side: str) -> Image.Image:
    path = _resolve_player_photo(player_name, photos_dir)
    if path and path.exists():
        source = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
        target = ImageOps.fit(source, (430, 620), method=Image.Resampling.LANCZOS, centering=(0.5, 0.18))
        target = ImageEnhance.Contrast(target).enhance(1.12)
        target = ImageEnhance.Brightness(target).enhance(1.05)
        img = target.convert("RGBA")
    else:
        img = Image.new("RGBA", (430, 620), (230, 235, 242, 255))
        d = ImageDraw.Draw(img)
        font = _load_font(78, bold=True)
        initials = "LM" if side == "left" else "CR"
        d.text((215, 310), initials, font=font, fill="#182736", anchor="mm")

    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay, "RGBA")
    od.rectangle((0, 0, img.size[0], img.size[1]), fill=(*accent, 26))
    overlay = overlay.filter(ImageFilter.GaussianBlur(radius=14))
    img.alpha_composite(overlay)

    fade = Image.new("L", img.size, 255)
    fd = ImageDraw.Draw(fade)
    for i in range(160):
        alpha = max(0, 255 - int(i * 255 / 160))
        fd.line((0, img.size[1] - i - 1, img.size[0], img.size[1] - i - 1), fill=alpha)
    img.putalpha(ImageChops.multiply(img.getchannel("A"), fade))
    return img


def _draw_glow_text(
    canvas: Image.Image,
    pos: tuple[int, int],
    text: str,
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int],
    glow: tuple[int, int, int],
    anchor: str = "mm",
    stroke_width: int = 2,
) -> None:
    glow_layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    gd = ImageDraw.Draw(glow_layer, "RGBA")
    gd.text(pos, text, font=font, fill=(*glow, 170), anchor=anchor)
    glow_layer = glow_layer.filter(ImageFilter.GaussianBlur(radius=18))
    canvas.alpha_composite(glow_layer)
    d = ImageDraw.Draw(canvas, "RGBA")
    d.text(pos, text, font=font, fill=(*fill, 255), anchor=anchor, stroke_width=stroke_width, stroke_fill=(10, 12, 18, 220))


def _format_value(value: float, final: bool = True) -> str:
    return str(int(round(value)))


def _make_frame_factory(photos_dir: Path):
    background = _make_background()
    messi = _load_player_cutout("Messi", photos_dir, MESSI_BLUE, "left")
    ronaldo = _load_player_cutout("Ronaldo", photos_dir, RONALDO_RED, "right")

    title_font = _load_font(86, bold=True)
    vs_font = _load_font(54, bold=True)
    sub_font = _load_font(32, bold=True)
    label_font = _load_font(34, bold=True)
    number_font = _load_font(58, bold=True)
    small_font = _load_font(26, bold=True)
    outro_font = _load_font(78, bold=True)

    board_x0 = 24
    board_x1 = WIDTH - 24
    bar_left_x = 112
    center_x = WIDTH // 2
    bar_right_x = 658
    bar_max_w = 300
    row_start_y = 688
    row_gap = 110
    row_h = 88

    intro_end = TITLE_HOLD
    reveal_duration = DURATION - TITLE_HOLD - OUTRO_HOLD
    row_window = reveal_duration / len(STATS)

    def make_frame(t: float) -> np.ndarray:
        frame = background.copy()
        pulse_left = 0.5 + 0.5 * math.sin(t * 2.4)
        pulse_right = 0.5 + 0.5 * math.sin(t * 2.4 + 1.2)

        energy = Image.new("RGBA", frame.size, (0, 0, 0, 0))
        ed = ImageDraw.Draw(energy, "RGBA")
        ed.ellipse((-60, 250, 390, 1020), fill=(*MESSI_BLUE, int(28 + 42 * pulse_left)))
        ed.ellipse((690, 250, 1140, 1020), fill=(*RONALDO_RED, int(28 + 42 * pulse_right)))
        for idx in range(24):
            x = (idx * 71 + int(t * 34)) % (WIDTH + 180) - 90
            alpha = 18 if idx % 2 == 0 else 10
            ed.line((x, HEIGHT - 260, x + 32, HEIGHT - 200), fill=(255, 220, 180, alpha), width=2)
        energy = energy.filter(ImageFilter.GaussianBlur(radius=14))
        frame.alpha_composite(energy)

        messi_zoom = 1.0 + 0.018 * math.sin(t * 1.4)
        ronaldo_zoom = 1.0 + 0.018 * math.sin(t * 1.4 + 0.9)
        messi_img = ImageOps.fit(messi, (int(430 * messi_zoom), int(620 * messi_zoom)), method=Image.Resampling.LANCZOS, centering=(0.5, 0.18))
        ronaldo_img = ImageOps.fit(ronaldo, (int(430 * ronaldo_zoom), int(620 * ronaldo_zoom)), method=Image.Resampling.LANCZOS, centering=(0.5, 0.18))
        frame.alpha_composite(messi_img, (18, 250))
        frame.alpha_composite(ronaldo_img, (WIDTH - ronaldo_img.size[0] - 18, 250))

        _draw_glow_text(frame, (286, 118), "MESSI", title_font, WHITE, MESSI_BLUE)
        _draw_glow_text(frame, (544, 122), "vs", vs_font, GOLD, GOLD, stroke_width=1)
        _draw_glow_text(frame, (804, 118), "RONALDO", title_font, WHITE, RONALDO_RED)

        d = ImageDraw.Draw(frame, "RGBA")
        d.line((160, 176, 324, 176), fill=(255, 214, 134, 130), width=2)
        d.line((756, 176, 920, 176), fill=(255, 214, 134, 130), width=2)
        d.text((540, 168), "CAREER TROPHIES", font=sub_font, fill=(*WHITE, 245), anchor="mm")

        current_row = min(len(STATS) - 1, max(-1, int((t - intro_end) / row_window)))
        left_score, right_score = _score_until(current_row)
        score_text = f"{left_score} - {right_score}"
        score_box = (430, 514, 650, 602)
        d.rounded_rectangle(score_box, radius=20, fill=(24, 20, 16, 220), outline=(255, 214, 134, 120), width=2)
        score_font = _fit_font(d, score_text, 180, 48, 28, True)
        _draw_glow_text(frame, ((score_box[0] + score_box[2]) // 2, 558), score_text, score_font, WHITE, GOLD)

        for idx, stat in enumerate(STATS):
            local_t = (t - intro_end - idx * row_window) / row_window
            reveal = _ease_out(local_t / 0.72) if local_t > 0 else 0.0
            row_alpha = int(70 + 185 * min(max(local_t * 1.4, 0.0), 1.0))
            y = row_start_y + idx * row_gap
            winner = _winner(stat.left, stat.right, stat.inverse)
            if reveal <= 0:
                continue

            d.rounded_rectangle((board_x0, y + 22, board_x1, y + 24 + row_h), radius=20, fill=(255, 255, 255, 10), outline=(255, 255, 255, 14), width=1)
            d.rounded_rectangle((bar_left_x, y + 48, bar_left_x + bar_max_w, y + 80), radius=15, fill=TRACK)
            d.rounded_rectangle((bar_right_x, y + 48, bar_right_x + bar_max_w, y + 80), radius=15, fill=TRACK)

            left_ratio = stat.left / max(stat.left, stat.right) if max(stat.left, stat.right) else 0
            right_ratio = stat.right / max(stat.left, stat.right) if max(stat.left, stat.right) else 0
            left_w = max(8, int(bar_max_w * left_ratio * reveal))
            right_w = max(8, int(bar_max_w * right_ratio * reveal))

            left_bar = Image.new("RGBA", (bar_max_w, 40), (0, 0, 0, 0))
            lbd = ImageDraw.Draw(left_bar, "RGBA")
            lbd.rounded_rectangle((0, 4, left_w, 36), radius=16, fill=(*MESSI_BLUE, 245 if winner == "left" else 205))
            lbd.rounded_rectangle((0, 4, max(10, left_w // 2), 14), radius=8, fill=(255, 255, 255, 70))
            left_bar = left_bar.filter(ImageFilter.GaussianBlur(radius=1))
            frame.alpha_composite(left_bar, (bar_left_x + bar_max_w - left_w, y + 44))

            right_bar = Image.new("RGBA", (bar_max_w, 40), (0, 0, 0, 0))
            rbd = ImageDraw.Draw(right_bar, "RGBA")
            rbd.rounded_rectangle((0, 4, right_w, 36), radius=16, fill=(*RONALDO_RED, 245 if winner == "right" else 205))
            rbd.rounded_rectangle((0, 4, max(10, right_w // 2), 14), radius=8, fill=(255, 255, 255, 70))
            right_bar = right_bar.filter(ImageFilter.GaussianBlur(radius=1))
            frame.alpha_composite(right_bar, (bar_right_x, y + 44))

            left_current = stat.left * reveal
            right_current = stat.right * reveal
            left_text = _format_value(left_current if reveal >= 0.999 else left_current, final=reveal >= 0.999)
            right_text = _format_value(right_current if reveal >= 0.999 else right_current, final=reveal >= 0.999)

            value_font = _fit_font(d, left_text, 94, 56, 24, True)
            _draw_glow_text(frame, (76, y + 60), left_text, value_font, WHITE, MESSI_BLUE)
            value_font_r = _fit_font(d, right_text, 94, 56, 24, True)
            _draw_glow_text(frame, (1004, y + 60), right_text, value_font_r, WHITE, RONALDO_RED)

            label = stat.label
            label_box = (408, y + 30, 672, y + 94)
            d.rounded_rectangle(label_box, radius=18, fill=(248, 232, 205, min(235, row_alpha)))
            label_font_fit = _fit_font(d, label, 236, 30, 14, True)
            d.text((540, y + 61), label, font=label_font_fit, fill=(18, 18, 18, min(255, row_alpha + 20)), anchor="mm")

            if reveal > 0.88 and winner != "tie":
                marker_x = 90 if winner == "left" else 990
                marker_color = MESSI_BLUE if winner == "left" else RONALDO_RED
                d.ellipse((marker_x - 16, y + 8, marker_x + 16, y + 40), fill=(*marker_color, 220))
                d.text((marker_x, y + 24), "▲" if winner == "left" else "▲", font=small_font, fill="#ffffff", anchor="mm")

        if t >= DURATION - OUTRO_HOLD:
            outro_progress = _ease_in_out((t - (DURATION - OUTRO_HOLD)) / OUTRO_HOLD)
            y = int(1630 - 36 * (1 - outro_progress))
            line_alpha = int(90 + 110 * outro_progress)
            d.line((140, y - 28, 430, y - 28), fill=(255, 214, 134, line_alpha), width=2)
            d.line((650, y - 28, 940, y - 28), fill=(255, 214, 134, line_alpha), width=2)
            _draw_glow_text(frame, (540, y + 58), "WHO IS THE GOAT ?", outro_font, GOLD, GOLD)

        return np.array(frame.convert("RGB"))

    return make_frame


def render_video(output_path: Path, photos_dir: Path, audio_path: Path, fps: int, duration: float) -> Path:
    make_frame = _make_frame_factory(photos_dir)
    clip = VideoClip(make_frame, duration=duration)
    if audio_path.exists():
        audio_clip, keep_alive = build_audio_track(audio_path, duration)
        clip = clip.with_audio(audio_clip)
    else:
        audio_clip, keep_alive = None, []

    output_path.parent.mkdir(parents=True, exist_ok=True)
    clip.write_videofile(
        str(output_path),
        fps=fps,
        codec="libx264",
        audio_codec="aac" if audio_clip else None,
        ffmpeg_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
    )

    clip.close()
    if audio_clip is not None:
        audio_clip.close()
    for item in keep_alive:
        item.close()
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a Messi vs Ronaldo career trophies Shorts video.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--photos-dir", type=Path, default=DEFAULT_PHOTOS_DIR)
    parser.add_argument("--audio", type=Path, default=DEFAULT_AUDIO)
    parser.add_argument("--fps", type=int, default=FPS)
    parser.add_argument("--duration", type=float, default=DURATION)
    args = parser.parse_args()

    render_video(
        output_path=args.output,
        photos_dir=args.photos_dir,
        audio_path=args.audio,
        fps=args.fps,
        duration=args.duration,
    )
    print(f"[video_generator] Messi vs Ronaldo career Shorts generated -> {args.output}")


if __name__ == "__main__":
    main()
