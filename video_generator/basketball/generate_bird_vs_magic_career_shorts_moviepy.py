from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from moviepy import VideoClip
from PIL import Image, ImageChops, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageOps

from moviepy import AudioFileClip, CompositeAudioClip
from moviepy.audio.fx import AudioFadeIn, AudioFadeOut

from video_generator.generate_ucl_barchart_race_moviepy import DEFAULT_AUDIO


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "basketball" / "bird_vs_magic_career_shorts.mp4"
DEFAULT_ASSETS_DIR = PROJECT_ROOT / "data" / "raw" / "nba_goat_assets"

WIDTH = 1080
HEIGHT = 1920
FPS = 30
DURATION = 40.0
TITLE_HOLD = 1.2
OUTRO_HOLD = 2.8
FINAL_AUDIO_FADE_OUT = 10.0
LOOP_CROSSFADE = 5.0

CURRY_BLUE = (80, 180, 120)
MAGIC_GOLD = (255, 196, 64)
GOLD = (255, 214, 134)
WHITE = (244, 242, 236)
TRACK = (78, 84, 94, 92)


@dataclass(frozen=True)
class StatRow:
    label: str
    left: float
    right: float
    inverse: bool = False


STATS = [
    StatRow("TITLES", 3, 5, False),
    StatRow("MVP", 3, 3, False),
    StatRow("POINTS", 21791, 17707, False),
    StatRow("PTS/G", 24.3, 19.5, False),
    StatRow("REBOUNDS", 8974, 6559, False),
    StatRow("REB/G", 10.0, 7.2, False),
    StatRow("ASSISTS", 5695, 10141, False),
    StatRow("AST/G", 6.3, 11.2, False),
    StatRow("STEALS", 1556, 1724, False),
    StatRow("STL/G", 1.7, 1.9, False),
    StatRow("3PM", 649, 325, False),
    StatRow("3P%", 37.6, 30.3, False),
    StatRow("ALL-STAR", 12, 12, False),
    StatRow("1ST TEAM", 9, 9, False),
    StatRow("FINALS MVP", 2, 3, False),
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


DECIMAL_LABELS = {"PTS/G", "REB/G", "AST/G", "STL/G", "3P%"}


def _format_value(label: str, value: float, final: bool = True) -> str:
    if label in DECIMAL_LABELS:
        if not final and abs(value) < 0.1:
            return "0"
        return f"{value:.1f}"
    return str(int(round(value)))


def build_audio_track(audio_path: Path, duration: float, fade_out: float):
    base = AudioFileClip(str(audio_path))
    effective_fade = max(0.0, min(fade_out, duration))
    if base.duration >= duration:
        clip = base.subclipped(0, duration)
        if effective_fade > 0:
            clip = clip.with_effects([AudioFadeOut(effective_fade)])
        return clip, [base]

    clips = []
    keep_alive = [base]
    step = max(0.1, base.duration - LOOP_CROSSFADE)
    loops = int(math.ceil(max(0.0, duration - LOOP_CROSSFADE) / step))
    for index in range(loops):
        segment = base.with_start(index * step)
        if LOOP_CROSSFADE > 0:
            segment = segment.with_effects([AudioFadeIn(LOOP_CROSSFADE), AudioFadeOut(LOOP_CROSSFADE)])
        clips.append(segment)
    mixed = CompositeAudioClip(clips).with_duration(duration)
    if effective_fade > 0:
        mixed = mixed.with_effects([AudioFadeOut(effective_fade)])
    return mixed, keep_alive


def _make_background() -> Image.Image:
    xx = np.linspace(0, 1, WIDTH, dtype=np.float32)
    yy = np.linspace(0, 1, HEIGHT, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(xx, yy)
    black = np.array([7, 9, 14], dtype=np.float32)
    steel = np.array([24, 28, 38], dtype=np.float32)
    blue = np.array(CURRY_BLUE, dtype=np.float32)
    red = np.array(MAGIC_GOLD, dtype=np.float32)
    mix = np.clip(0.66 * grid_y + 0.10 * np.abs(grid_x - 0.5), 0, 1)
    left_glow = np.exp(-(((grid_x - 0.16) / 0.20) ** 2 + ((grid_y - 0.35) / 0.16) ** 2))
    right_glow = np.exp(-(((grid_x - 0.84) / 0.20) ** 2 + ((grid_y - 0.35) / 0.16) ** 2))
    smoke = np.exp(-(((grid_y - 0.86) / 0.22) ** 2))

    frame = np.clip(
        black[None, None, :] * (1.0 - mix[..., None])
        + steel[None, None, :] * (0.88 * mix[..., None])
        + blue[None, None, :] * (0.24 * left_glow[..., None])
        + red[None, None, :] * (0.24 * right_glow[..., None]),
        0,
        255,
    ).astype(np.uint8)
    image = Image.fromarray(frame, "RGB").convert("RGBA")

    fog = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    fd = ImageDraw.Draw(fog, "RGBA")
    for i in range(14):
        alpha = int(18 + i * 4)
        y0 = HEIGHT - 360 + i * 18
        fd.ellipse((-90 - i * 16, y0, WIDTH + 90 + i * 16, y0 + 250), fill=(255, 255, 255, alpha))
    fog = fog.filter(ImageFilter.GaussianBlur(radius=42))
    fog_np = np.array(fog, dtype=np.float32)
    fog_np[..., :3] *= smoke[..., None]
    fog_np[..., 3] *= smoke
    image.alpha_composite(Image.fromarray(np.clip(fog_np, 0, 255).astype(np.uint8), "RGBA"))
    return image


def _load_asset(path: Path, size: tuple[int, int], centering: tuple[float, float]) -> Image.Image:
    source = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    target = ImageOps.fit(source, size, method=Image.Resampling.LANCZOS, centering=centering)
    target = ImageEnhance.Contrast(target).enhance(1.12)
    target = ImageEnhance.Brightness(target).enhance(1.06)
    target = ImageEnhance.Color(target).enhance(1.04)
    return target.convert("RGBA")


def _load_player_cutout(path: Path, accent: tuple[int, int, int]) -> Image.Image:
    img = _load_asset(path, (430, 560), (0.5, 0.18))
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay, "RGBA")
    od.rectangle((0, 0, img.size[0], img.size[1]), fill=(*accent, 28))
    overlay = overlay.filter(ImageFilter.GaussianBlur(radius=12))
    img.alpha_composite(overlay)

    fade = Image.new("L", img.size, 255)
    fd = ImageDraw.Draw(fade)
    for i in range(140):
        alpha = max(0, 255 - int(i * 255 / 140))
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
    gd.text(pos, text, font=font, fill=(*glow, 160), anchor=anchor)
    glow_layer = glow_layer.filter(ImageFilter.GaussianBlur(radius=16))
    canvas.alpha_composite(glow_layer)
    d = ImageDraw.Draw(canvas, "RGBA")
    d.text(pos, text, font=font, fill=(*fill, 255), anchor=anchor, stroke_width=stroke_width, stroke_fill=(10, 12, 18, 220))


def _draw_stroke_text(
    draw: ImageDraw.ImageDraw,
    pos: tuple[int, int],
    text: str,
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int],
    anchor: str = "mm",
    stroke_width: int = 2,
) -> None:
    draw.text(pos, text, font=font, fill=(*fill, 255), anchor=anchor, stroke_width=stroke_width, stroke_fill=(10, 12, 18, 220))


def _make_frame_factory(assets_dir: Path, duration: float):
    background = _make_background()
    curry_source = assets_dir / "bird_wiki.jpg"
    if not curry_source.exists():
        curry_source = assets_dir / "lebron_action.jpg"
    curry = _load_player_cutout(curry_source, CURRY_BLUE)
    magic_source = assets_dir / "magic_wiki.jpg"
    if not magic_source.exists():
        magic_source = assets_dir / "jordan_action.jpg"
    magic = _load_player_cutout(magic_source, MAGIC_GOLD)

    title_font = _load_font(72, bold=True)
    vs_font = _load_font(48, bold=True)
    sub_font = _load_font(24, bold=True)
    small_font = _load_font(22, bold=True)
    outro_font = _load_font(60, bold=True)

    board_x0 = 22
    board_x1 = WIDTH - 22
    bar_left_x = 122
    bar_right_x = 640
    bar_max_w = 318
    row_start_y = 606
    row_gap = 72
    row_h = 50

    intro_end = TITLE_HOLD
    reveal_duration = max(0.1, duration - TITLE_HOLD - OUTRO_HOLD)
    row_window = reveal_duration / len(STATS)

    title_layer = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    td = ImageDraw.Draw(title_layer, "RGBA")
    _draw_glow_text(title_layer, (292, 92), "BIRD", title_font, WHITE, CURRY_BLUE)
    _draw_glow_text(title_layer, (540, 98), "vs", vs_font, GOLD, GOLD, stroke_width=1)
    _draw_glow_text(title_layer, (790, 92), "MAGIC", title_font, WHITE, MAGIC_GOLD)
    td.line((160, 142, 330, 142), fill=(255, 214, 134, 110), width=2)
    td.line((750, 142, 920, 142), fill=(255, 214, 134, 110), width=2)
    td.text((540, 137), "CAREER BATTLE", font=sub_font, fill=(*WHITE, 240), anchor="mm")

    board_layer = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    bd = ImageDraw.Draw(board_layer, "RGBA")
    row_y_positions: list[int] = []
    for idx, stat in enumerate(STATS):
        y = row_start_y + idx * row_gap
        row_y_positions.append(y)
        bd.rounded_rectangle((board_x0, y + 10, board_x1, y + 10 + row_h), radius=16, fill=(18, 22, 30, 148), outline=(255, 255, 255, 6), width=1)
        bd.rounded_rectangle((bar_left_x, y + 22, bar_left_x + bar_max_w, y + 40), radius=10, fill=TRACK)
        bd.rounded_rectangle((bar_right_x, y + 22, bar_right_x + bar_max_w, y + 40), radius=10, fill=TRACK)
        label_box = (448, y + 8, 632, y + 52)
        bd.rounded_rectangle(label_box, radius=14, fill=(242, 224, 192, 178))
        label_font_fit = _fit_font(bd, stat.label, 164, 22, 12, True)
        bd.text((540, y + 29), stat.label, font=label_font_fit, fill=(18, 18, 18, 255), anchor="mm")

    score_layers: dict[str, Image.Image] = {}
    for left_score in range(len(STATS) + 1):
        for right_score in range(len(STATS) + 1):
            score_text = f"{left_score}-{right_score}"
            score_img = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
            sd = ImageDraw.Draw(score_img, "RGBA")
            score_box = (450, 450, 630, 522)
            sd.rounded_rectangle(score_box, radius=18, fill=(24, 20, 16, 214), outline=(255, 214, 134, 120), width=2)
            score_font = _fit_font(sd, score_text, 150, 42, 26, True)
            _draw_glow_text(score_img, ((score_box[0] + score_box[2]) // 2, 486), score_text, score_font, WHITE, GOLD)
            score_layers[score_text] = score_img

    outro_layer = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    od = ImageDraw.Draw(outro_layer, "RGBA")
    od.line((160, HEIGHT - 184, 420, HEIGHT - 184), fill=(255, 214, 134, 120), width=2)
    od.line((660, HEIGHT - 184, 920, HEIGHT - 184), fill=(255, 214, 134, 120), width=2)
    _draw_glow_text(outro_layer, (540, HEIGHT - 136), "WHO IS THE GOAT ?", outro_font, GOLD, GOLD)

    @lru_cache(maxsize=192)
    def cached_energy(step: int) -> Image.Image:
        tt = step / 12.0
        pulse_left = 0.5 + 0.5 * math.sin(tt * 2.2)
        pulse_right = 0.5 + 0.5 * math.sin(tt * 2.2 + 0.9)
        energy = Image.new("RGBA", background.size, (0, 0, 0, 0))
        ed = ImageDraw.Draw(energy, "RGBA")
        ed.ellipse((-80, 200, 390, 860), fill=(*CURRY_BLUE, int(26 + 36 * pulse_left)))
        ed.ellipse((690, 200, 1160, 860), fill=(*MAGIC_GOLD, int(26 + 36 * pulse_right)))
        for idx in range(20):
            x = (idx * 83 + int(tt * 38)) % (WIDTH + 180) - 90
            alpha = 14 if idx % 2 == 0 else 8
            ed.line((x, HEIGHT - 230, x + 28, HEIGHT - 180), fill=(255, 220, 180, alpha), width=2)
        return energy.filter(ImageFilter.GaussianBlur(radius=14))

    @lru_cache(maxsize=160)
    def cached_player(image_key: str, step: int) -> Image.Image:
        tt = step / 16.0
        if image_key == "left":
            source = curry
            zoom = 1.0 + 0.018 * math.sin(tt * 1.3)
            centering = (0.5, 0.18)
        else:
            source = magic
            zoom = 1.0 + 0.018 * math.sin(tt * 1.3 + 0.8)
            centering = (0.5, 0.10)
        return ImageOps.fit(source, (int(430 * zoom), int(560 * zoom)), method=Image.Resampling.LANCZOS, centering=centering)

    @lru_cache(maxsize=256)
    def cached_value_font(text: str) -> ImageFont.ImageFont:
        probe = Image.new("RGBA", (32, 32), (0, 0, 0, 0))
        pd = ImageDraw.Draw(probe, "RGBA")
        return _fit_font(pd, text, 88, 38, 18, True)

    def make_frame(t: float) -> np.ndarray:
        frame = background.copy()
        frame.alpha_composite(cached_energy(int(t * 12)))

        left_img = cached_player("left", int(t * 16))
        right_img = cached_player("right", int(t * 16))
        frame.alpha_composite(left_img, (10, 170))
        frame.alpha_composite(right_img, (WIDTH - right_img.size[0] - 10, 170))
        frame.alpha_composite(title_layer)
        frame.alpha_composite(board_layer)

        d = ImageDraw.Draw(frame, "RGBA")
        current_row = min(len(STATS) - 1, max(-1, int((t - intro_end) / row_window)))
        left_score, right_score = _score_until(current_row)
        frame.alpha_composite(score_layers[f"{left_score}-{right_score}"])

        for idx, stat in enumerate(STATS):
            row_time = t - intro_end - idx * row_window
            local_t = min(max(row_time / (row_window * 0.78), 0.0), 1.0)
            reveal = _ease_out(local_t)
            y = row_y_positions[idx]
            winner = _winner(stat.left, stat.right, stat.inverse)

            max_value = max(stat.left, stat.right) if max(stat.left, stat.right) else 1.0
            left_ratio = stat.left / max_value
            right_ratio = stat.right / max_value
            left_w = max(6, int(bar_max_w * left_ratio * reveal))
            right_w = max(6, int(bar_max_w * right_ratio * reveal))

            left_x0 = bar_left_x + bar_max_w - left_w
            left_x1 = bar_left_x + bar_max_w
            right_x0 = bar_right_x
            right_x1 = bar_right_x + right_w
            d.rounded_rectangle((left_x0, y + 22, left_x1, y + 40), radius=9, fill=(*CURRY_BLUE, 245 if winner == "left" else 205))
            d.rounded_rectangle((right_x0, y + 22, right_x1, y + 40), radius=9, fill=(*MAGIC_GOLD, 245 if winner == "right" else 205))
            d.rounded_rectangle((left_x0, y + 22, min(left_x1, left_x0 + max(16, left_w // 2)), y + 28), radius=5, fill=(255, 255, 255, 56))
            d.rounded_rectangle((right_x0, y + 22, min(right_x1, right_x0 + max(16, right_w // 2)), y + 28), radius=5, fill=(255, 255, 255, 56))

            left_current = stat.left * reveal
            right_current = stat.right * reveal
            left_text = _format_value(stat.label, left_current if reveal < 0.999 else stat.left, final=reveal >= 0.999)
            right_text = _format_value(stat.label, right_current if reveal < 0.999 else stat.right, final=reveal >= 0.999)

            _draw_stroke_text(d, (72, y + 29), left_text, cached_value_font(left_text), WHITE)
            _draw_stroke_text(d, (1008, y + 29), right_text, cached_value_font(right_text), WHITE)

            if winner in {"left", "right"} and local_t > 0.6:
                marker_x = 88 if winner == "left" else 992
                marker_color = CURRY_BLUE if winner == "left" else MAGIC_GOLD
                d.ellipse((marker_x - 12, y - 2, marker_x + 12, y + 22), fill=(*marker_color, 228))
                d.text((marker_x, y + 10), "▲", font=small_font, fill="#ffffff", anchor="mm")

        if t >= duration - OUTRO_HOLD:
            fade = _ease_out((t - (duration - OUTRO_HOLD)) / OUTRO_HOLD)
            if fade > 0:
                outro = outro_layer.copy()
                alpha = outro.getchannel("A").point(lambda px: int(px * fade))
                outro.putalpha(alpha)
                frame.alpha_composite(outro)

        return np.array(frame.convert("RGB"))

    return make_frame


def render_video(output_path: Path, assets_dir: Path, audio_path: Path, fps: int, duration: float, audio_fade_out: float) -> Path:
    make_frame = _make_frame_factory(assets_dir, duration)
    clip = VideoClip(make_frame, duration=duration)
    if audio_path.exists():
        audio_clip, keep_alive = build_audio_track(audio_path, duration, audio_fade_out)
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
    for item in keep_alive:
        if not isinstance(item, (str, Path)):
            continue
        try:
            Path(item).unlink(missing_ok=True)
        except OSError:
            pass
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Bird vs Magic career Shorts with the cinematic board template.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--assets-dir", type=Path, default=DEFAULT_ASSETS_DIR)
    parser.add_argument("--audio", type=Path, default=DEFAULT_AUDIO)
    parser.add_argument("--fps", type=int, default=FPS)
    parser.add_argument("--duration", type=float, default=DURATION)
    parser.add_argument("--audio-fade-out", type=float, default=FINAL_AUDIO_FADE_OUT)
    args = parser.parse_args()

    output = render_video(
        output_path=args.output,
        assets_dir=args.assets_dir,
        audio_path=args.audio,
        fps=args.fps,
        duration=args.duration,
        audio_fade_out=args.audio_fade_out,
    )
    try:
        rel = output.relative_to(PROJECT_ROOT)
    except ValueError:
        rel = output
    print(f"[video_generator] Bird vs Magic career Shorts generated -> {rel}")


if __name__ == "__main__":
    main()
