from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from moviepy import AudioFileClip, CompositeAudioClip, VideoClip
from moviepy.audio.fx import AudioFadeIn, AudioFadeOut
from PIL import Image, ImageChops, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageOps

from video_generator.generate_ucl_barchart_race_moviepy import DEFAULT_AUDIO


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "basketball" / "sga_vs_jokic_vs_wembanyama_mvp_short.mp4"
DEFAULT_PREVIEW_OUTPUT = PROJECT_ROOT / "data" / "processed" / "basketball" / "sga_vs_jokic_vs_wembanyama_preview.mp4"
DEFAULT_ASSETS_DIR = PROJECT_ROOT / "data" / "raw" / "mvp_race_assets"
MIDNIGHT_GRIP_AUDIO = PROJECT_ROOT / "data" / "raw" / "audio" / "Midnight_Grip_20260402_0828.mp3"

WIDTH = 1080
HEIGHT = 1920
FPS = 30
DURATION = 50.0
PREVIEW_DURATION = 15.0
TITLE_HOLD = 1.2
OUTRO_HOLD = 3.0
FINAL_AUDIO_FADE_OUT = 10.0
LOOP_CROSSFADE = 5.0

SGA_BLUE = (0, 122, 193)
JOKIC_GOLD = (253, 185, 39)
WEMBY_TEAL = (88, 214, 193)
GOLD = (255, 214, 134)
WHITE = (244, 242, 236)
TRACK = (78, 84, 94, 92)


@dataclass(frozen=True)
class Player:
    key: str
    label: str
    accent: tuple[int, int, int]
    initials: str


@dataclass(frozen=True)
class StatRow:
    label: str
    values: tuple[float, float, float]
    inverse: bool = False


PLAYERS = (
    Player("sga", "SGA", SGA_BLUE, "SGA"),
    Player("jokic", "JOKIC", JOKIC_GOLD, "JOKIC"),
    Player("wemby", "WEMBY", WEMBY_TEAL, "WEMBY"),
)

STATS = [
    StatRow("POINTS / GAME", (31.1, 27.7, 25.0)),
    StatRow("ASSISTS / GAME", (6.6, 10.7, 3.1)),
    StatRow("REBOUNDS / GAME", (4.3, 12.9, 11.5)),
    StatRow("FG%", (55.3, 56.9, 51.2)),
    StatRow("TS%", (66.5, 67.0, 62.6)),
    StatRow("FT%", (87.9, 83.1, 82.7)),
    StatRow("BLOCKS / GAME", (0.8, 0.8, 3.1)),
    StatRow("STEALS / GAME", (1.8, 1.3, 1.3)),
    StatRow("AST/TOV", (2.4, 3.7, 1.8)),
    StatRow("PER", (30.5, 31.9, 27.1)),
    StatRow("GAMES PLAYED", (68, 65, 64)),
    StatRow("WIN %", (78.0, 65.9, 75.6)),
    StatRow("30+ POINT GAMES", (43, 25, 18)),
    StatRow("TRIPLE-DOUBLES", (0, 34, 4)),
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


def _winner(values: tuple[float, float, float], inverse: bool = False) -> int | None:
    best = min(values) if inverse else max(values)
    winners = [idx for idx, value in enumerate(values) if abs(value - best) < 1e-9]
    if len(winners) != 1:
        return None
    return winners[0]


def _score_until(row_index: int) -> tuple[int, int, int]:
    scores = [0, 0, 0]
    for idx, stat in enumerate(STATS):
        if idx > row_index:
            break
        winner = _winner(stat.values, stat.inverse)
        if winner is not None:
            scores[winner] += 1
    return scores[0], scores[1], scores[2]


DECIMAL_LABELS = {
    "POINTS / GAME",
    "ASSISTS / GAME",
    "REBOUNDS / GAME",
    "FG%",
    "TS%",
    "FT%",
    "BLOCKS / GAME",
    "WIN %",
}


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
    blue = np.array(SGA_BLUE, dtype=np.float32)
    gold = np.array(JOKIC_GOLD, dtype=np.float32)
    teal = np.array(WEMBY_TEAL, dtype=np.float32)
    mix = np.clip(0.66 * grid_y + 0.10 * np.abs(grid_x - 0.5), 0, 1)
    left_glow = np.exp(-(((grid_x - 0.16) / 0.20) ** 2 + ((grid_y - 0.30) / 0.16) ** 2))
    center_glow = np.exp(-(((grid_x - 0.50) / 0.20) ** 2 + ((grid_y - 0.28) / 0.18) ** 2))
    right_glow = np.exp(-(((grid_x - 0.84) / 0.20) ** 2 + ((grid_y - 0.30) / 0.16) ** 2))
    smoke = np.exp(-(((grid_y - 0.86) / 0.22) ** 2))

    frame = np.clip(
        black[None, None, :] * (1.0 - mix[..., None])
        + steel[None, None, :] * (0.88 * mix[..., None])
        + blue[None, None, :] * (0.24 * left_glow[..., None])
        + gold[None, None, :] * (0.22 * center_glow[..., None])
        + teal[None, None, :] * (0.24 * right_glow[..., None]),
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
    img = ImageOps.exif_transpose(Image.open(path)).convert("RGBA")
    bbox = img.getbbox()
    if bbox is not None:
        img = img.crop(bbox)
    img = ImageOps.contain(img, (430, 560), method=Image.Resampling.LANCZOS)
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


def _make_placeholder_cutout(initials: str, accent: tuple[int, int, int]) -> Image.Image:
    size = (430, 560)
    image = Image.new("RGBA", size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(image, "RGBA")
    for y in range(size[1]):
        blend = y / max(1, size[1] - 1)
        color = tuple(int(12 + (accent[i] - 12) * blend) for i in range(3))
        draw.line((0, y, size[0], y), fill=(*color, 255))
    draw.ellipse((-40, -20, 380, 280), fill=(*accent, 50))
    draw.rounded_rectangle((18, 18, size[0] - 18, size[1] - 18), radius=28, outline=(255, 255, 255, 90), width=2)
    draw.rounded_rectangle((40, 46, size[0] - 40, size[1] - 74), radius=22, fill=(8, 10, 16, 72))
    font = _load_font(96, bold=True)
    draw.text((size[0] / 2, size[1] * 0.44), initials, font=font, fill=(255, 255, 255, 235), anchor="mm")
    return image


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

    sources: dict[str, Path | None] = {}
    candidates = {
        "sga": ("sga.png", "shai.png", "shai_gilgeous_alexander.png", "shai_gilgeous_alexander.jpg", "shai_gilgeous_alexander.webp"),
        "jokic": ("jokic.png", "nikola_jokic.png", "nikola_jokic.jpg", "nikola_jokic.webp"),
        "wemby": ("wemby.png", "victor_wembanyama.png", "victor_wembanyama.jpg", "victor_wembanyama.webp"),
    }
    for key, names in candidates.items():
        found = None
        for candidate in names:
            path = assets_dir / candidate
            if path.exists():
                found = path
                break
        sources[key] = found

    player_images = {
        player.key: _load_player_cutout(sources[player.key], player.accent) if sources[player.key] is not None else _make_placeholder_cutout(player.initials, player.accent)
        for player in PLAYERS
    }

    title_font = _load_font(46, bold=True)
    vs_font = _load_font(30, bold=True)
    sub_font = _load_font(24, bold=True)
    small_font = _load_font(20, bold=True)
    outro_font = _load_font(60, bold=True)

    title_layer = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    td = ImageDraw.Draw(title_layer, "RGBA")
    title_y = 42
    _draw_glow_text(title_layer, (406, title_y), "SGA", title_font, WHITE, SGA_BLUE)
    _draw_glow_text(title_layer, (528, title_y), "vs", vs_font, GOLD, GOLD, stroke_width=1)
    _draw_glow_text(title_layer, (650, title_y), "JOKIC", title_font, WHITE, JOKIC_GOLD)
    _draw_glow_text(title_layer, (786, title_y), "vs", vs_font, GOLD, GOLD, stroke_width=1)
    _draw_glow_text(title_layer, (922, title_y), "WEMBY", title_font, WHITE, WEMBY_TEAL)
    board_layer = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    bd = ImageDraw.Draw(board_layer, "RGBA")

    row_start_y = 392
    row_gap = 86
    row_h = 74
    label_box = (4, 0, 308, 0)
    card_boxes = ((286, 0, 526, 0), (530, 0, 770, 0), (774, 0, 1070, 0))
    card_colors = (SGA_BLUE, JOKIC_GOLD, WEMBY_TEAL)

    row_y_positions: list[int] = []
    for idx, stat in enumerate(STATS):
        y = row_start_y + idx * row_gap
        row_y_positions.append(y)
        bd.rounded_rectangle((0, y + 8, WIDTH, y + 8 + row_h), radius=18, fill=(18, 22, 30, 156), outline=(255, 255, 255, 6), width=1)
        bd.rounded_rectangle((6, y + 16, 304, y + 56), radius=16, fill=(242, 224, 192, 182))
        label_font_fit = _fit_font(bd, stat.label, 276, 24, 12, True)
        bd.text((155, y + 36), stat.label, font=label_font_fit, fill=(18, 18, 18, 255), anchor="mm")

        for col, card in enumerate(card_boxes):
            x0, _, x1, _ = card
            bd.rounded_rectangle((x0, y + 16, x1, y + 56), radius=14, fill=(26, 30, 42, 194), outline=(*card_colors[col], 80), width=1)

    outro_layer = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    od = ImageDraw.Draw(outro_layer, "RGBA")
    od.line((160, HEIGHT - 184, 420, HEIGHT - 184), fill=(255, 214, 134, 120), width=2)
    od.line((660, HEIGHT - 184, 920, HEIGHT - 184), fill=(255, 214, 134, 120), width=2)
    _draw_glow_text(outro_layer, (540, HEIGHT - 150), "WHO IS THE MVP?", outro_font, GOLD, GOLD)
    _draw_glow_text(outro_layer, (540, HEIGHT - 88), "DROP YOUR PICK IN THE COMMENTS", sub_font, WHITE, GOLD, stroke_width=1)

    @lru_cache(maxsize=192)
    def cached_energy(step: int) -> Image.Image:
        tt = step / 12.0
        pulse_left = 0.5 + 0.5 * math.sin(tt * 2.2)
        pulse_center = 0.5 + 0.5 * math.sin(tt * 2.2 + 0.8)
        pulse_right = 0.5 + 0.5 * math.sin(tt * 2.2 + 1.6)
        energy = Image.new("RGBA", background.size, (0, 0, 0, 0))
        ed = ImageDraw.Draw(energy, "RGBA")
        ed.ellipse((-80, 200, 390, 860), fill=(*SGA_BLUE, int(24 + 34 * pulse_left)))
        ed.ellipse((305, 190, 775, 850), fill=(*JOKIC_GOLD, int(22 + 32 * pulse_center)))
        ed.ellipse((690, 200, 1160, 860), fill=(*WEMBY_TEAL, int(24 + 34 * pulse_right)))
        for idx in range(20):
            x = (idx * 83 + int(tt * 38)) % (WIDTH + 180) - 90
            alpha = 14 if idx % 2 == 0 else 8
            ed.line((x, HEIGHT - 230, x + 28, HEIGHT - 180), fill=(255, 220, 180, alpha), width=2)
        return energy.filter(ImageFilter.GaussianBlur(radius=14))

    @lru_cache(maxsize=160)
    def cached_player(image_key: str, step: int) -> Image.Image:
        tt = step / 16.0
        source = player_images[image_key]
        if image_key == "sga":
            size = (260, 176)
        elif image_key == "jokic":
            size = (270, 176)
        else:
            size = (258, 176)
        pulse = 1.0 + 0.004 * math.sin(tt * 1.1 + (0.0 if image_key == "sga" else 0.8 if image_key == "jokic" else 1.6))
        return ImageOps.contain(source, (max(1, int(size[0] * pulse)), max(1, int(size[1] * pulse))), method=Image.Resampling.LANCZOS)

    @lru_cache(maxsize=256)
    def cached_value_font(text: str) -> ImageFont.ImageFont:
        probe = Image.new("RGBA", (32, 32), (0, 0, 0, 0))
        pd = ImageDraw.Draw(probe, "RGBA")
        return _fit_font(pd, text, 88, 36, 18, True)

    def make_frame(t: float) -> np.ndarray:
        frame = background.copy()
        frame.alpha_composite(cached_energy(int(t * 12)))

        left_img = cached_player("sga", int(t * 16))
        center_img = cached_player("jokic", int(t * 16))
        right_img = cached_player("wemby", int(t * 16))

        frame.alpha_composite(left_img, (406 - left_img.size[0] // 2, 92))
        frame.alpha_composite(center_img, (650 - center_img.size[0] // 2, 90))
        frame.alpha_composite(right_img, (922 - right_img.size[0] // 2, 92))
        frame.alpha_composite(title_layer)
        frame.alpha_composite(board_layer)

        d = ImageDraw.Draw(frame, "RGBA")
        current_row = min(len(STATS) - 1, max(-1, int((t - TITLE_HOLD) / max(0.1, (duration - TITLE_HOLD - OUTRO_HOLD) / len(STATS)))))
        scores = _score_until(current_row)
        sga_score_font = _fit_font(d, str(scores[0]), 100, 58, 30, True)
        jokic_score_font = _fit_font(d, str(scores[1]), 100, 58, 30, True)
        wemby_score_font = _fit_font(d, str(scores[2]), 100, 58, 30, True)
        score_positions = ((406, 290), (650, 290), (922, 290))
        for (sx, sy), score, font, accent, name in zip(
            score_positions,
            scores,
            (sga_score_font, jokic_score_font, wemby_score_font),
            (SGA_BLUE, JOKIC_GOLD, WEMBY_TEAL),
            ("SGA", "JOKIC", "WEMBY"),
        ):
            d.rounded_rectangle((sx - 84, sy - 40, sx + 84, sy + 38), radius=18, fill=(24, 20, 16, 210), outline=(*accent, 130), width=3)
            _draw_stroke_text(d, (sx, sy), str(score), font, WHITE)

        name_y = 262
        name_font = _load_font(28, bold=True)
        for x, name, accent in ((406, "SGA", SGA_BLUE), (650, "JOKIC", JOKIC_GOLD), (922, "WEMBY", WEMBY_TEAL)):
            d.text((x, name_y), name, font=name_font, fill=(*accent, 255), anchor="mm", stroke_width=2, stroke_fill=(10, 12, 18, 220))

        row_window = max(0.1, (duration - TITLE_HOLD - OUTRO_HOLD) / len(STATS))
        for idx, stat in enumerate(STATS):
            row_time = t - TITLE_HOLD - idx * row_window
            local_t = min(max(row_time / (row_window * 0.78), 0.0), 1.0)
            reveal = _ease_out(local_t)
            y = row_y_positions[idx]
            winner = _winner(stat.values, stat.inverse)

            max_value = max(stat.values) if max(stat.values) else 1.0
            for col, value in enumerate(stat.values):
                x0, _, x1, _ = card_boxes[col]
                card_w = x1 - x0
                bar_w = max(6, int((card_w - 22) * (value / max_value) * reveal))
                fill_x0 = x0 + 11
                fill_x1 = x0 + 11 + bar_w
                fill_color = card_colors[col]
                if winner == col:
                    fill_alpha = 245
                    outline_alpha = 180
                elif winner is None:
                    fill_alpha = 210
                    outline_alpha = 90
                else:
                    fill_alpha = 200
                    outline_alpha = 70
                d.rounded_rectangle((fill_x0, y + 24, fill_x1, y + 42), radius=9, fill=(*fill_color, fill_alpha))
                d.rounded_rectangle((x0 + 11, y + 24, x0 + 11 + max(16, bar_w // 2), y + 30), radius=5, fill=(255, 255, 255, 48))
                d.rounded_rectangle((x0, y + 14, x1, y + 50), radius=12, outline=(*fill_color, outline_alpha), width=1)

                text = _format_value(stat.label, value * reveal, final=reveal >= 0.999)
                value_font = cached_value_font(text)
                _draw_stroke_text(d, ((x0 + x1) // 2, y + 32), text, value_font, WHITE)

                if winner == col and local_t > 0.6:
                    marker_x = (x0 + x1) // 2
                    d.ellipse((marker_x - 12, y + 2, marker_x + 12, y + 26), fill=(*fill_color, 228))
                    d.text((marker_x, y + 14), "▲", font=small_font, fill="#ffffff", anchor="mm")

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
    parser = argparse.ArgumentParser(description="Generate SGA vs Jokic vs Wemby MVP Shorts with the cinematic board template.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--preview-output", type=Path, default=DEFAULT_PREVIEW_OUTPUT)
    parser.add_argument("--assets-dir", type=Path, default=DEFAULT_ASSETS_DIR)
    parser.add_argument("--audio", type=Path, default=MIDNIGHT_GRIP_AUDIO if MIDNIGHT_GRIP_AUDIO.exists() else DEFAULT_AUDIO)
    parser.add_argument("--fps", type=int, default=FPS)
    parser.add_argument("--duration", type=float, default=DURATION)
    parser.add_argument("--audio-fade-out", type=float, default=FINAL_AUDIO_FADE_OUT)
    parser.add_argument("--preview", action="store_true", help="Render a short preview cut instead of the full 50s version.")
    args = parser.parse_args()

    output_path = args.preview_output if args.preview else args.output
    duration = min(args.duration, PREVIEW_DURATION) if args.preview else args.duration
    fps = min(args.fps, 30) if args.preview else args.fps

    output = render_video(
        output_path=output_path,
        assets_dir=args.assets_dir,
        audio_path=args.audio,
        fps=fps,
        duration=duration,
        audio_fade_out=args.audio_fade_out,
    )
    try:
        rel = output.relative_to(PROJECT_ROOT)
    except ValueError:
        rel = output
    print(f"[video_generator] SGA vs Jokic vs Wemby MVP Shorts generated -> {rel}")


if __name__ == "__main__":
    main()
