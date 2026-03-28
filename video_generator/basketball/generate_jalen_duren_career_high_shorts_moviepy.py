from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from moviepy import VideoClip
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from video_generator.generate_ucl_barchart_race_moviepy import (
    DEFAULT_AUDIO,
    _fit_font_size,
    _load_font,
    build_audio_track,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "basketball" / "jalen_duren_career_high_shorts.mp4"

WIDTH = 1080
HEIGHT = 1920
FPS = 30
TOTAL_DURATION = 10.0

HOOK = "36?! FROM DUREN?!"
SCENES = [
    "CAREER HIGH NIGHT",
    "36 POINTS",
    "12 REBOUNDS",
    "PAINT DOMINATION",
    "TOO BIG TOO STRONG",
    "DETROIT NEEDED THIS",
    "DUREN TOOK OVER",
    "REMEMBER THIS GAME",
]
FINAL_SCENE = "SUPERSTAR NEXT?"


def _make_background() -> Image.Image:
    xx = np.linspace(0, 1, WIDTH, dtype=np.float32)
    yy = np.linspace(0, 1, HEIGHT, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(xx, yy)
    navy = np.array([7, 15, 32], dtype=np.float32)
    blue = np.array([20, 74, 156], dtype=np.float32)
    red = np.array([210, 44, 62], dtype=np.float32)
    white = np.array([244, 248, 252], dtype=np.float32)

    mix = np.clip(0.55 * grid_y + 0.2 * grid_x, 0, 1)
    center_glow = np.exp(-(((grid_x - 0.5) / 0.22) ** 2 + ((grid_y - 0.26) / 0.14) ** 2))
    lower_glow = np.exp(-(((grid_x - 0.52) / 0.28) ** 2 + ((grid_y - 0.75) / 0.24) ** 2))
    side_red = np.exp(-(((grid_x - 0.05) / 0.08) ** 2 + ((grid_y - 0.5) / 0.35) ** 2))
    side_red += np.exp(-(((grid_x - 0.95) / 0.08) ** 2 + ((grid_y - 0.5) / 0.35) ** 2))

    img = np.clip(
        navy[None, None, :] * (1.0 - mix[..., None])
        + blue[None, None, :] * (0.78 * mix[..., None])
        + white[None, None, :] * (0.08 * center_glow[..., None])
        + blue[None, None, :] * (0.16 * lower_glow[..., None])
        + red[None, None, :] * (0.16 * side_red[..., None]),
        0,
        255,
    ).astype(np.uint8)
    frame = Image.fromarray(img, mode="RGB").convert("RGBA")

    overlay = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")
    draw.rounded_rectangle((42, 42, WIDTH - 42, HEIGHT - 42), radius=48, outline=(255, 255, 255, 18), width=2)
    draw.ellipse((168, 190, WIDTH - 168, 860), outline=(255, 255, 255, 14), width=3)
    draw.line((140, 1110, WIDTH - 140, 1110), fill=(255, 255, 255, 18), width=2)
    draw.line((WIDTH // 2, 200, WIDTH // 2, HEIGHT - 180), fill=(255, 255, 255, 12), width=2)
    overlay = overlay.filter(ImageFilter.GaussianBlur(radius=2))
    frame.alpha_composite(overlay)
    return frame


def _draw_glow_text(
    frame: Image.Image,
    position: tuple[int, int],
    text: str,
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int],
    glow: tuple[int, int, int],
    anchor: str = "ma",
) -> None:
    blur_layer = Image.new("RGBA", frame.size, (0, 0, 0, 0))
    blur_draw = ImageDraw.Draw(blur_layer, "RGBA")
    blur_draw.text(position, text, font=font, fill=(*glow, 130), anchor=anchor)
    blur_layer = blur_layer.filter(ImageFilter.GaussianBlur(radius=12))
    frame.alpha_composite(blur_layer)
    draw = ImageDraw.Draw(frame, "RGBA")
    draw.text(position, text, font=font, fill=(*fill, 255), anchor=anchor)


def _wrap_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int) -> list[str]:
    words = text.split()
    if not words:
        return []
    lines = [words[0]]
    for word in words[1:]:
        candidate = f"{lines[-1]} {word}"
        if draw.textbbox((0, 0), candidate, font=font)[2] <= max_width:
            lines[-1] = candidate
        else:
            lines.append(word)
    return lines


def _ease_out(value: float) -> float:
    value = min(max(value, 0.0), 1.0)
    return 1.0 - (1.0 - value) ** 3


def render_video(output_path: Path, audio_path: Path, duration: float, fps: int) -> Path:
    background = _make_background()
    temp_draw = ImageDraw.Draw(background.copy(), "RGBA")
    hook_font = _fit_font_size(temp_draw, HOOK, 880, 96, 40, bold=True)
    scene_font = _fit_font_size(temp_draw, "TOO BIG TOO STRONG", 880, 88, 36, bold=True)
    final_font = _fit_font_size(temp_draw, FINAL_SCENE, 860, 100, 42, bold=True)
    label_font = _load_font(26, bold=True)
    big_number_font = _load_font(180, bold=True)

    hook_duration = 1.0
    final_duration = 1.6
    scene_duration = (duration - hook_duration - final_duration) / len(SCENES)

    def make_frame(t: float) -> np.ndarray:
        frame = background.copy()
        draw = ImageDraw.Draw(frame, "RGBA")
        draw.text((WIDTH // 2, 120), "LAST NIGHT", font=label_font, fill="#dbe6f2", anchor="ma")

        if t < hook_duration:
            phase = _ease_out(t / hook_duration)
            panel = (80, 620, WIDTH - 80, 1180)
            draw.rounded_rectangle(panel, radius=42, fill=(5, 13, 28, 218), outline=(255, 255, 255, 24), width=2)
            lines = _wrap_text(draw, HOOK, hook_font, panel[2] - panel[0] - 120)
            line_height = int(hook_font.size * 1.14)
            total_h = len(lines) * line_height
            start_y = (panel[1] + panel[3] - total_h) // 2
            for idx, line in enumerate(lines):
                y = start_y + idx * line_height + int((1.0 - phase) * 70)
                _draw_glow_text(frame, (WIDTH // 2, y), line, hook_font, (245, 249, 255), (80, 170, 255))
            return np.array(frame.convert("RGB"))

        if t >= duration - final_duration:
            phase = _ease_out((t - (duration - final_duration)) / final_duration)
            draw.rounded_rectangle((92, 702, WIDTH - 92, 1200), radius=38, fill=(13, 34, 58, 220), outline=(255, 255, 255, 18), width=2)
            _draw_glow_text(
                frame,
                (WIDTH // 2, 880 + int((1.0 - phase) * 50)),
                FINAL_SCENE,
                final_font,
                (248, 250, 253),
                (210, 44, 62),
            )
            draw.text((WIDTH // 2, 1048), "COMMENT BELOW", font=label_font, fill="#ffd3d8", anchor="ma")
            return np.array(frame.convert("RGB"))

        index = min(int((t - hook_duration) / scene_duration), len(SCENES) - 1)
        scene_t = ((t - hook_duration) - index * scene_duration) / scene_duration
        phase = _ease_out(scene_t)
        text = SCENES[index]

        draw.rounded_rectangle((76, 560, WIDTH - 76, 1280), radius=48, fill=(8, 20, 40, 180))
        if text in {"36 POINTS", "12 REBOUNDS"}:
            number, word = text.split(" ", 1)
            _draw_glow_text(frame, (WIDTH // 2, 760 + int((1.0 - phase) * 40)), number, big_number_font, (248, 250, 253), (80, 170, 255))
            _draw_glow_text(frame, (WIDTH // 2, 980), word, scene_font, (248, 250, 253), (210, 44, 62))
        else:
            lines = _wrap_text(draw, text, scene_font, 860)
            line_height = int(scene_font.size * 1.12)
            total_h = len(lines) * line_height
            start_y = 860 - total_h // 2
            for idx, line in enumerate(lines):
                y = start_y + idx * line_height + int((1.0 - phase) * 36)
                _draw_glow_text(frame, (WIDTH // 2, y), line, scene_font, (248, 250, 253), (80, 170, 255))

        return np.array(frame.convert("RGB"))

    clip = VideoClip(make_frame, duration=duration)
    audio_clip, keep_alive = build_audio_track(audio_path, duration)
    clip = clip.with_audio(audio_clip)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    clip.write_videofile(str(output_path), fps=fps, codec="libx264", audio_codec="aac")
    clip.close()
    audio_clip.close()
    for item in keep_alive:
        item.close()
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a Jalen Duren career-high NBA Shorts video.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--audio", type=Path, default=DEFAULT_AUDIO)
    parser.add_argument("--duration", type=float, default=TOTAL_DURATION)
    parser.add_argument("--fps", type=int, default=FPS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = render_video(args.output, args.audio, args.duration, args.fps)
    print(f"[video_generator] Jalen Duren Shorts generated -> {output}")


if __name__ == "__main__":
    main()
