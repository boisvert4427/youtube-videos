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
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "basketball" / "lebron_vs_jordan_goat_shorts.mp4"

WIDTH = 1080
HEIGHT = 1920
FPS = 30
TOTAL_DURATION = 11.0

HOOK = "ONE NAME ONLY"
BUILD_SCENES = [
    ("LEBRON", (64, 132, 255)),
    ("JORDAN", (218, 44, 62)),
    ("LEBRON", (64, 132, 255)),
    ("JORDAN", (218, 44, 62)),
]
BUILD_WORDS = [
    "RISES",
    "WAITS",
    "ENDURES",
    "STRIKES",
]
CONTRAST_LEFT = "LONGEVITY"
CONTRAST_RIGHT = "PERFECTION"
FINAL_SCENE = "WHO YOU GOT?"


def _make_background() -> Image.Image:
    xx = np.linspace(0, 1, WIDTH, dtype=np.float32)
    yy = np.linspace(0, 1, HEIGHT, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(xx, yy)
    black = np.array([5, 8, 16], dtype=np.float32)
    navy = np.array([10, 28, 58], dtype=np.float32)
    blue = np.array([64, 132, 255], dtype=np.float32)
    red = np.array([218, 44, 62], dtype=np.float32)
    white = np.array([246, 248, 252], dtype=np.float32)

    mix = np.clip(0.58 * grid_y + 0.14 * grid_x, 0, 1)
    center_glow = np.exp(-(((grid_x - 0.5) / 0.22) ** 2 + ((grid_y - 0.45) / 0.18) ** 2))
    left_glow = np.exp(-(((grid_x - 0.18) / 0.18) ** 2 + ((grid_y - 0.48) / 0.24) ** 2))
    right_glow = np.exp(-(((grid_x - 0.82) / 0.18) ** 2 + ((grid_y - 0.48) / 0.24) ** 2))

    img = np.clip(
        black[None, None, :] * (1.0 - mix[..., None])
        + navy[None, None, :] * (0.82 * mix[..., None])
        + white[None, None, :] * (0.04 * center_glow[..., None])
        + blue[None, None, :] * (0.18 * left_glow[..., None])
        + red[None, None, :] * (0.18 * right_glow[..., None]),
        0,
        255,
    ).astype(np.uint8)
    frame = Image.fromarray(img, mode="RGB").convert("RGBA")

    overlay = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")
    draw.rounded_rectangle((42, 42, WIDTH - 42, HEIGHT - 42), radius=48, outline=(255, 255, 255, 14), width=2)
    draw.line((WIDTH // 2, 180, WIDTH // 2, HEIGHT - 180), fill=(255, 255, 255, 16), width=2)
    draw.ellipse((240, 300, WIDTH - 240, 980), outline=(255, 255, 255, 10), width=3)
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


def _ease_out(value: float) -> float:
    value = min(max(value, 0.0), 1.0)
    return 1.0 - (1.0 - value) ** 3


def render_video(output_path: Path, audio_path: Path, duration: float, fps: int) -> Path:
    background = _make_background()
    temp_draw = ImageDraw.Draw(background.copy(), "RGBA")
    hook_font = _fit_font_size(temp_draw, HOOK, 860, 104, 42, bold=True)
    name_font = _fit_font_size(temp_draw, "LONGEVITY", 420, 82, 34, bold=True)
    word_font = _fit_font_size(temp_draw, "PERFECTION", 420, 92, 36, bold=True)
    final_font = _fit_font_size(temp_draw, FINAL_SCENE, 860, 112, 46, bold=True)
    label_font = _load_font(24, bold=True)

    hook_duration = 1.0
    contrast_duration = 1.8
    final_duration = 1.6
    build_duration = duration - hook_duration - contrast_duration - final_duration
    scene_duration = build_duration / len(BUILD_SCENES)

    def make_frame(t: float) -> np.ndarray:
        frame = background.copy()
        draw = ImageDraw.Draw(frame, "RGBA")
        draw.text((WIDTH // 2, 112), "GOAT DEBATE", font=label_font, fill="#dfe8f3", anchor="ma")

        if t < hook_duration:
            phase = _ease_out(t / hook_duration)
            panel = (84, 652, WIDTH - 84, 1140)
            draw.rounded_rectangle(panel, radius=42, fill=(7, 16, 30, 220), outline=(255, 255, 255, 18), width=2)
            _draw_glow_text(frame, (WIDTH // 2, 890 + int((1.0 - phase) * 60)), HOOK, hook_font, (246, 248, 252), (255, 255, 255))
            return np.array(frame.convert("RGB"))

        build_end = hook_duration + build_duration
        if t < build_end:
            local = t - hook_duration
            index = min(int(local / scene_duration), len(BUILD_SCENES) - 1)
            scene_t = (local - index * scene_duration) / scene_duration
            phase = _ease_out(scene_t)
            name, glow = BUILD_SCENES[index]
            word = BUILD_WORDS[index]

            draw.rounded_rectangle((94, 540, WIDTH - 94, 1260), radius=44, fill=(8, 18, 34, 188))
            _draw_glow_text(frame, (WIDTH // 2, 760 + int((1.0 - phase) * 30)), name, name_font, (246, 248, 252), glow)
            _draw_glow_text(frame, (WIDTH // 2, 950), word, word_font, (246, 248, 252), glow)
            return np.array(frame.convert("RGB"))

        contrast_end = build_end + contrast_duration
        if t < contrast_end:
            phase = _ease_out((t - build_end) / contrast_duration)
            draw.rounded_rectangle((70, 520, WIDTH - 70, 1280), radius=44, fill=(8, 18, 34, 200))
            _draw_glow_text(frame, (WIDTH // 2 - 220, 792 + int((1.0 - phase) * 28)), CONTRAST_LEFT, name_font, (246, 248, 252), (64, 132, 255))
            _draw_glow_text(frame, (WIDTH // 2, 948), "VS", _load_font(48, bold=True), (255, 233, 183), (255, 233, 183))
            _draw_glow_text(frame, (WIDTH // 2 + 220, 1104 - int((1.0 - phase) * 28)), CONTRAST_RIGHT, name_font, (246, 248, 252), (218, 44, 62))
            return np.array(frame.convert("RGB"))

        phase = _ease_out((t - contrast_end) / final_duration)
        draw.rounded_rectangle((88, 676, WIDTH - 88, 1188), radius=40, fill=(10, 24, 44, 220), outline=(255, 255, 255, 16), width=2)
        _draw_glow_text(frame, (WIDTH // 2, 884 + int((1.0 - phase) * 48)), FINAL_SCENE, final_font, (248, 250, 253), (255, 255, 255))
        draw.text((WIDTH // 2, 1046), "COMMENT NOW", font=label_font, fill="#ffd7da", anchor="ma")
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
    parser = argparse.ArgumentParser(description="Generate a LeBron vs Jordan GOAT debate Shorts video.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--audio", type=Path, default=DEFAULT_AUDIO)
    parser.add_argument("--duration", type=float, default=TOTAL_DURATION)
    parser.add_argument("--fps", type=int, default=FPS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = render_video(args.output, args.audio, args.duration, args.fps)
    print(f"[video_generator] LeBron vs Jordan Shorts generated -> {output}")


if __name__ == "__main__":
    main()
