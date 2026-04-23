from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from uuid import uuid4

import numpy as np
from moviepy import AudioFileClip, CompositeAudioClip, VideoClip
from moviepy.audio.fx import AudioFadeIn, AudioFadeOut
from PIL import Image, ImageChops, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageOps

from video_generator.generate_ucl_barchart_race_moviepy import DEFAULT_AUDIO, _fit_font_size, _load_font


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "tennis" / "federer_vs_nadal_birdstyle_shorts.mp4"
PHOTOS_DIR = PROJECT_ROOT / "data" / "raw" / "player_photos"
MIDNIGHT_GRIP_AUDIO = PROJECT_ROOT / "data" / "raw" / "audio" / "Midnight_Grip_20260402_0828.mp3"

WIDTH = 1080
HEIGHT = 1920
FPS = 30
DURATION = 30.0
FINAL_AUDIO_FADE_OUT = 5.0
MUSIC_VOLUME = 0.42
LOOP_CROSSFADE = 5.0

FEDERER_BLUE = (110, 193, 255)
NADAL_RED = (255, 107, 107)
GOLD = (242, 197, 94)
WHITE = (246, 248, 252)
TRACK = (78, 84, 94, 92)


@dataclass(frozen=True)
class StatRow:
    label: str
    left: float
    right: float
    inverse: bool = False


STATS = [
    StatRow("GRAND SLAMS", 20, 22, False),
    StatRow("ROLAND GARROS", 1, 14, False),
    StatRow("WIMBLEDON", 8, 2, False),
]

def _build_row_windows(total_duration: float, stat_count: int) -> list[tuple[float, float]]:
    intro_end = 2.0
    outro_start = min(max(intro_end + 0.5, total_duration - 10.0), total_duration - 5.0)
    step = (outro_start - intro_end) / max(1, stat_count)
    return [(intro_end + idx * step, intro_end + (idx + 1) * step) for idx in range(stat_count)]

DATA_FALLBACK = {
    "hook": "WHO DOMINATES?",
    "left_name": "FEDERER",
    "right_name": "NADAL",
    "left_image": "roger_federer.jpg",
    "right_image": "rafael_nadal.jpg",
    "mid_text": "",
    "climax_text": "",
    "ending_text": "",
}


def _ease_out(value: float) -> float:
    value = min(max(value, 0.0), 1.0)
    return 1.0 - (1.0 - value) ** 3


def _load_data(path: Path | None) -> dict:
    return DATA_FALLBACK.copy()


def _load_portrait(path: Path, fallback_name: str) -> Image.Image:
    if path.exists():
        return ImageOps.exif_transpose(Image.open(path)).convert("RGBA")
    img = Image.new("RGBA", (720, 720), (12, 18, 28, 255))
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle((20, 20, 700, 700), radius=60, fill=(22, 30, 44, 255), outline=(255, 255, 255, 70), width=4)
    font = _load_font(62, bold=True)
    draw.text((360, 340), fallback_name[:2].upper(), font=font, fill=WHITE, anchor="mm")
    return img


def _make_background(t: float) -> Image.Image:
    xx = np.linspace(0, 1, WIDTH, dtype=np.float32)
    yy = np.linspace(0, 1, HEIGHT, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(xx, yy)
    base = np.array([6, 10, 18], dtype=np.float32)
    navy = np.array([10, 28, 56], dtype=np.float32)
    left_glow = np.exp(-(((grid_x - 0.20) / 0.20) ** 2 + ((grid_y - 0.35) / 0.24) ** 2))
    right_glow = np.exp(-(((grid_x - 0.80) / 0.20) ** 2 + ((grid_y - 0.35) / 0.24) ** 2))
    center_glow = np.exp(-(((grid_x - 0.50) / 0.30) ** 2 + ((grid_y - 0.56) / 0.24) ** 2))
    pulse = 0.5 + 0.5 * np.sin(t * 1.5)
    img = np.clip(
        base[None, None, :] * (1.0 - 0.88 * grid_y[..., None])
        + navy[None, None, :] * (0.68 * grid_y[..., None])
        + np.array(FEDERER_BLUE, dtype=np.float32)[None, None, :] * (0.18 * left_glow[..., None] * (0.7 + 0.3 * pulse))
        + np.array(NADAL_RED, dtype=np.float32)[None, None, :] * (0.18 * right_glow[..., None] * (0.7 + 0.3 * (1.0 - pulse)))
        + np.array((255, 255, 255), dtype=np.float32)[None, None, :] * (0.03 * center_glow[..., None]),
        0,
        255,
    ).astype(np.uint8)
    frame = Image.fromarray(img, mode="RGB").convert("RGBA")
    overlay = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")
    draw.rounded_rectangle((40, 40, WIDTH - 40, HEIGHT - 40), radius=46, outline=(255, 255, 255, 18), width=2)
    draw.ellipse((60, 170, 460, 920), fill=(*FEDERER_BLUE, 22))
    draw.ellipse((620, 170, 1020, 920), fill=(*NADAL_RED, 20))
    draw.ellipse((260, 860, 820, 1700), fill=(255, 255, 255, 10))
    overlay = overlay.filter(ImageFilter.GaussianBlur(radius=10))
    frame.alpha_composite(overlay)
    return frame


def _make_player_cutout(source: Image.Image, accent: tuple[int, int, int]) -> Image.Image:
    img = ImageOps.fit(source, (430, 560), method=Image.Resampling.LANCZOS, centering=(0.5, 0.18))
    img = ImageEnhance.Brightness(img).enhance(1.12)
    img = ImageEnhance.Contrast(img).enhance(1.12)
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay, "RGBA")
    od.rectangle((0, 0, img.size[0], img.size[1]), fill=(*accent, 28))
    overlay = overlay.filter(ImageFilter.GaussianBlur(radius=12))
    img = img.convert("RGBA")
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


def _draw_text(draw: ImageDraw.ImageDraw, pos: tuple[int, int], text: str, font: ImageFont.ImageFont, fill: tuple[int, int, int], anchor: str = "mm", stroke_width: int = 2) -> None:
    draw.text(pos, text, font=font, fill=(*fill, 255), anchor=anchor, stroke_width=stroke_width, stroke_fill=(10, 12, 18, 220))


def _winner(left: float, right: float, inverse: bool) -> str:
    if abs(left - right) < 1e-9:
        return "tie"
    if inverse:
        return "left" if left < right else "right"
    return "left" if left > right else "right"


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


def _score_for_time(stats: list[StatRow], t: float) -> tuple[int, int]:
    left = 0
    right = 0
    for (start, end), stat in zip(_build_row_windows(DURATION, len(stats)), stats):
        if t >= end:
            side = _winner(stat.left, stat.right, stat.inverse)
            if side == "left":
                left += 1
            elif side == "right":
                right += 1
    return left, right


def _count_value(value: float, progress: float) -> str:
    current = value * progress
    return str(int(round(current if progress < 0.999 else value)))


def _build_background_cache() -> dict[int, Image.Image]:
    return {i: _make_background(i * 0.25) for i in range(4)}


def render_video(output_path: Path, audio_path: Path, duration: float, fps: int) -> Path:
    data = _load_data(None)
    stats = STATS
    bg_frames = _build_background_cache()
    portraits = {
        "federer": _make_player_cutout(_load_portrait(PHOTOS_DIR / data["left_image"], "RF"), FEDERER_BLUE),
        "nadal": _make_player_cutout(_load_portrait(PHOTOS_DIR / data["right_image"], "RN"), NADAL_RED),
    }

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
    intro_end = 1.40
    outro_start = max(intro_end + 0.5, duration - 4.0)
    row_windows = _build_row_windows(duration, len(stats))
    row_window = row_windows[0][1] - row_windows[0][0]

    title_layer = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    td = ImageDraw.Draw(title_layer, "RGBA")
    _draw_glow_text(title_layer, (292, 92), "FEDERER", title_font, WHITE, FEDERER_BLUE)
    _draw_glow_text(title_layer, (540, 98), "vs", vs_font, GOLD, GOLD, stroke_width=1)
    _draw_glow_text(title_layer, (790, 92), "NADAL", title_font, WHITE, NADAL_RED)
    td.line((160, 142, 330, 142), fill=(255, 214, 134, 110), width=2)
    td.line((750, 142, 920, 142), fill=(255, 214, 134, 110), width=2)
    td.text((540, 137), "4 STATS. 1 WINNER.", font=sub_font, fill=(*WHITE, 240), anchor="mm")

    board_layer = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    bd = ImageDraw.Draw(board_layer, "RGBA")
    row_y_positions: list[int] = []
    for idx, stat in enumerate(stats):
        y = row_start_y + idx * row_gap
        row_y_positions.append(y)
        bd.rounded_rectangle((board_x0, y + 10, board_x1, y + 10 + row_h), radius=16, fill=(18, 22, 30, 148), outline=(255, 255, 255, 6), width=1)
        bd.rounded_rectangle((bar_left_x, y + 22, bar_left_x + bar_max_w, y + 40), radius=10, fill=TRACK)
        bd.rounded_rectangle((bar_right_x, y + 22, bar_right_x + bar_max_w, y + 40), radius=10, fill=TRACK)
        label_box = (448, y + 8, 632, y + 52)
        bd.rounded_rectangle(label_box, radius=14, fill=(242, 224, 192, 178))
        label_font_fit = _fit_font_size(bd, stat.label, 164, 22, 12, True)
        bd.text((540, y + 29), stat.label, font=label_font_fit, fill=(18, 18, 18, 255), anchor="mm")

    score_layers: dict[str, Image.Image] = {}
    for left_score in range(len(stats) + 1):
        for right_score in range(len(stats) + 1):
            score_text = f"{left_score}-{right_score}"
            score_img = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
            sd = ImageDraw.Draw(score_img, "RGBA")
            score_box = (450, 450, 630, 522)
            sd.rounded_rectangle(score_box, radius=18, fill=(24, 20, 16, 214), outline=(255, 214, 134, 120), width=2)
            score_font = _fit_font_size(sd, score_text, 150, 42, 26, True)
            _draw_glow_text(score_img, ((score_box[0] + score_box[2]) // 2, 486), score_text, score_font, WHITE, GOLD)
            score_layers[score_text] = score_img

    outro_layer = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    od = ImageDraw.Draw(outro_layer, "RGBA")
    od.line((160, HEIGHT - 184, 420, HEIGHT - 184), fill=(255, 214, 134, 120), width=2)
    od.line((660, HEIGHT - 184, 920, HEIGHT - 184), fill=(255, 214, 134, 120), width=2)
    _draw_glow_text(outro_layer, (540, HEIGHT - 136), "WHO IS THE GOAT ?", outro_font, GOLD, GOLD)

    score_reveal_layer = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    srd = ImageDraw.Draw(score_reveal_layer, "RGBA")
    srd.rounded_rectangle((278, 1142, 802, 1378), radius=36, fill=(9, 12, 18, 216), outline=(255, 214, 134, 140), width=3)
    _draw_glow_text(score_reveal_layer, (540, 1220), "22 - 20", _load_font(84, bold=True), WHITE, GOLD)
    _draw_glow_text(score_reveal_layer, (540, 1320), "WHO DOMINATES?", _load_font(28, bold=True), GOLD, GOLD)

    @lru_cache(maxsize=192)
    def cached_energy(step: int) -> Image.Image:
        tt = step / 12.0
        pulse_left = 0.5 + 0.5 * math.sin(tt * 2.2)
        pulse_right = 0.5 + 0.5 * math.sin(tt * 2.2 + 0.9)
        energy = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
        ed = ImageDraw.Draw(energy, "RGBA")
        ed.ellipse((-80, 200, 390, 860), fill=(*FEDERER_BLUE, int(26 + 36 * pulse_left)))
        ed.ellipse((690, 200, 1160, 860), fill=(*NADAL_RED, int(26 + 36 * pulse_right)))
        for idx in range(20):
            x = (idx * 83 + int(tt * 38)) % (WIDTH + 180) - 90
            alpha = 14 if idx % 2 == 0 else 8
            ed.line((x, HEIGHT - 230, x + 28, HEIGHT - 180), fill=(255, 220, 180, alpha), width=2)
        return energy.filter(ImageFilter.GaussianBlur(radius=14))

    @lru_cache(maxsize=160)
    def cached_player(image_key: str, step: int) -> Image.Image:
        tt = step / 16.0
        if image_key == "left":
            source = portraits["federer"]
            zoom = 1.0 + 0.018 * math.sin(tt * 1.3)
            centering = (0.5, 0.18)
        else:
            source = portraits["nadal"]
            zoom = 1.0 + 0.018 * math.sin(tt * 1.3 + 0.8)
            centering = (0.5, 0.10)
        return ImageOps.fit(source, (int(430 * zoom), int(560 * zoom)), method=Image.Resampling.LANCZOS, centering=centering)

    @lru_cache(maxsize=256)
    def cached_value_font(text: str) -> ImageFont.ImageFont:
        probe = Image.new("RGBA", (32, 32), (0, 0, 0, 0))
        pd = ImageDraw.Draw(probe, "RGBA")
        return _fit_font_size(pd, text, 88, 38, 18, True)

    def make_frame(t: float) -> np.ndarray:
        if t < 0.35:
            black = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 255))
            flash = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
            fd = ImageDraw.Draw(flash, "RGBA")
            fd.ellipse((210, 360, 870, 1020), fill=(255, 255, 255, 18))
            fd.ellipse((260, 420, 820, 960), fill=(255, 214, 134, 16))
            black.alpha_composite(flash.filter(ImageFilter.GaussianBlur(radius=38)))
            dd = ImageDraw.Draw(black, "RGBA")
            _draw_glow_text(black, (540, 905), "FEDERER vs NADAL", _load_font(58, bold=True), WHITE, GOLD)
            return np.array(black.convert("RGB"))

        if t < 0.9:
            black = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 255))
            dd = ImageDraw.Draw(black, "RGBA")
            glow = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
            gd = ImageDraw.Draw(glow, "RGBA")
            gd.ellipse((120, 300, 500, 920), fill=(*FEDERER_BLUE, 45))
            gd.ellipse((580, 300, 960, 920), fill=(*NADAL_RED, 45))
            black.alpha_composite(glow.filter(ImageFilter.GaussianBlur(radius=32)))
            _draw_glow_text(black, (540, 860), "FEDERER vs NADAL", _load_font(54, bold=True), WHITE, GOLD)
            _draw_glow_text(black, (540, 1030), "WHO DOMINATES?", _load_font(34, bold=True), GOLD, GOLD)
            return np.array(black.convert("RGB"))

        frame = bg_frames[int((t * 2.0) % 4)].copy()
        frame.alpha_composite(cached_energy(int(t * 12)))
        left_img = cached_player("left", int(t * 16))
        right_img = cached_player("right", int(t * 16))
        if t >= 0.9:
            frame.alpha_composite(left_img, (10, 170))
            frame.alpha_composite(right_img, (WIDTH - right_img.size[0] - 10, 170))
            frame.alpha_composite(title_layer)
            frame.alpha_composite(board_layer)

        d = ImageDraw.Draw(frame, "RGBA")
        current_row = min(len(stats) - 1, max(-1, int((t - intro_end) / row_window)))
        left_score, right_score = _score_for_time(stats, t)
        if t >= 0.9:
            frame.alpha_composite(score_layers[f"{left_score}-{right_score}"])

        for idx, stat in enumerate(stats):
            start, end = row_windows[idx]
            row_time = t - start
            local_t = min(max(row_time / max(0.001, (end - start) * 0.78), 0.0), 1.0)
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
            d.rounded_rectangle((left_x0, y + 22, left_x1, y + 40), radius=9, fill=(*FEDERER_BLUE, 245 if winner == "left" else 205))
            d.rounded_rectangle((right_x0, y + 22, right_x1, y + 40), radius=9, fill=(*NADAL_RED, 245 if winner == "right" else 205))
            d.rounded_rectangle((left_x0, y + 22, min(left_x1, left_x0 + max(16, left_w // 2)), y + 28), radius=5, fill=(255, 255, 255, 56))
            d.rounded_rectangle((right_x0, y + 22, min(right_x1, right_x0 + max(16, right_w // 2)), y + 28), radius=5, fill=(255, 255, 255, 56))

            left_current = stat.left * reveal
            right_current = stat.right * reveal
            left_text = _count_value(stat.left if reveal >= 0.999 else left_current, 1.0)
            right_text = _count_value(stat.right if reveal >= 0.999 else right_current, 1.0)

            _draw_text(d, (72, y + 29), left_text, cached_value_font(left_text), WHITE)
            _draw_text(d, (1008, y + 29), right_text, cached_value_font(right_text), WHITE)

            if winner in {"left", "right"} and local_t > 0.6:
                marker_x = 88 if winner == "left" else 992
                marker_color = FEDERER_BLUE if winner == "left" else NADAL_RED
                d.ellipse((marker_x - 12, y - 2, marker_x + 12, y + 22), fill=(*marker_color, 228))
                d.text((marker_x, y + 10), "▲", font=small_font, fill="#ffffff", anchor="mm")

        if 20.0 <= t < 25.0:
            reveal = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
            reveal.alpha_composite(score_reveal_layer)
            frame.alpha_composite(reveal)

        if t >= duration - 4.0:
            end = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
            ed = ImageDraw.Draw(end, "RGBA")
            ed.rounded_rectangle((244, 1560, 836, 1708), radius=32, fill=(10, 24, 43, 222), outline=(255, 255, 255, 24), width=2)
            _draw_glow_text(end, (540, 1632), "WHO IS THE GOAT ?", _load_font(30, bold=True), GOLD, GOLD)
            frame.alpha_composite(end)

        return np.array(frame.convert("RGB"))

    clip = VideoClip(make_frame, duration=duration)
    audio_clip, keep_alive = build_audio_track(audio_path, duration, FINAL_AUDIO_FADE_OUT)
    composite_audio = CompositeAudioClip([audio_clip.with_volume_scaled(MUSIC_VOLUME)]).with_duration(duration)
    clip = clip.with_audio(composite_audio)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    run_id = uuid4().hex[:8]
    temp_output = output_path.with_name(f"{output_path.stem}.{run_id}.render.mp4")
    temp_audio = output_path.with_name(f"{output_path.stem}.{run_id}.temp_audio.m4a")
    try:
        clip.write_videofile(
            str(temp_output),
            fps=fps,
            codec="libx264",
            audio_codec="aac",
            bitrate="12000k",
            preset="slow",
            temp_audiofile=str(temp_audio),
            remove_temp=False,
        )
        if output_path.exists():
            try:
                output_path.unlink()
            except PermissionError:
                output_path = output_path.with_name(f"{output_path.stem}_{run_id}.mp4")
        temp_output.replace(output_path)
    finally:
        clip.close()
        composite_audio.close()
        audio_clip.close()
        for item in keep_alive:
            item.close()
        if temp_audio.exists():
            try:
                temp_audio.unlink()
            except OSError:
                pass
        if temp_output.exists() and temp_output != output_path:
            try:
                temp_output.unlink()
            except OSError:
                pass
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a Federer vs Nadal Shorts video with the Bird vs Magic style.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--audio", type=Path, default=MIDNIGHT_GRIP_AUDIO if MIDNIGHT_GRIP_AUDIO.exists() else DEFAULT_AUDIO)
    parser.add_argument("--fps", type=int, default=FPS)
    parser.add_argument("--duration", type=float, default=DURATION)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = render_video(args.output, args.audio, args.duration, args.fps)
    print(f"[video_generator] Federer vs Nadal Bird-style Shorts generated -> {output}")


if __name__ == "__main__":
    main()
