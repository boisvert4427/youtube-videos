from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from moviepy import VideoClip
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageOps

from video_generator.generate_ucl_barchart_race_moviepy import DEFAULT_AUDIO, _fit_font_size, _load_font, build_audio_track


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PHOTOS_DIR = PROJECT_ROOT / "data" / "raw" / "player_photos"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "tennis" / "federer_nadal_djokovic_grand_slam_shorts.mp4"

WIDTH = 1080
HEIGHT = 1920
FPS = 30
TOTAL_DURATION = 9.0
MUSIC_VOLUME = 0.42

TITLE = "FEDERER vs NADAL vs DJOKOVIC"
SUBTITLE = "Grand Slam Titles"
FOOTER = "Career Grand Slam totals"


@dataclass(frozen=True)
class PlayerCard:
    surname: str
    full_name: str
    photo_name: str
    count: int
    bar_color: str
    ring_color: str
    country_code: str


PLAYERS = [
    PlayerCard(
        surname="FEDERER",
        full_name="Roger Federer",
        photo_name="roger_federer.jpg",
        count=20,
        bar_color="#F3C64B",
        ring_color="#76C8FF",
        country_code="SUI",
    ),
    PlayerCard(
        surname="NADAL",
        full_name="Rafael Nadal",
        photo_name="rafael_nadal.jpg",
        count=22,
        bar_color="#E92A33",
        ring_color="#FF8A57",
        country_code="ESP",
    ),
    PlayerCard(
        surname="DJOKOVIC",
        full_name="Novak Djokovic",
        photo_name="novak_djokovic.jpg",
        count=24,
        bar_color="#F4F6FA",
        ring_color="#79C9FF",
        country_code="SRB",
    ),
]


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _ease_out(value: float) -> float:
    value = _clamp(value)
    return 1.0 - (1.0 - value) ** 3


def _ease_in_out(value: float) -> float:
    value = _clamp(value)
    return value * value * (3.0 - 2.0 * value)


def _hex_to_rgb(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    return tuple(int(value[i : i + 2], 16) for i in (0, 2, 4))


def _mix_rgb(color: str, target: tuple[int, int, int], amount: float) -> tuple[int, int, int]:
    amount = _clamp(amount)
    r, g, b = _hex_to_rgb(color)
    return (
        int(r + (target[0] - r) * amount),
        int(g + (target[1] - g) * amount),
        int(b + (target[2] - b) * amount),
    )


def _text_on(color: str) -> str:
    r, g, b = _hex_to_rgb(color)
    luminance = (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255.0
    return "#08131F" if luminance > 0.66 else "#F4F7FB"


def _load_player_photo(photo_name: str, initials: str) -> Image.Image:
    path = PHOTOS_DIR / photo_name
    if path.exists():
        return ImageOps.exif_transpose(Image.open(path)).convert("RGBA")

    placeholder = Image.new("RGBA", (720, 720), (11, 18, 32, 255))
    draw = ImageDraw.Draw(placeholder)
    draw.rounded_rectangle((20, 20, 700, 700), radius=60, fill=(19, 28, 46, 255), outline=(255, 255, 255, 70), width=4)
    draw.text((360, 348), initials, font=_load_font(68, bold=True), fill=(244, 247, 251, 255), anchor="mm")
    return placeholder


def _make_avatar(source: Image.Image, ring_color: tuple[int, int, int], size: int = 150) -> Image.Image:
    portrait = ImageOps.fit(source, (size, size), method=Image.Resampling.LANCZOS, centering=(0.5, 0.24))
    portrait = ImageEnhance.Brightness(portrait).enhance(1.12)
    portrait = ImageEnhance.Contrast(portrait).enhance(1.08)
    portrait = ImageEnhance.Color(portrait).enhance(1.03)
    mask = Image.new("L", (size, size), 0)
    ImageDraw.Draw(mask).ellipse((0, 0, size - 1, size - 1), fill=255)
    circle = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    circle.paste(portrait, (0, 0), mask)

    tile = Image.new("RGBA", (size + 24, size + 24), (0, 0, 0, 0))
    td = ImageDraw.Draw(tile, "RGBA")
    td.ellipse((0, 0, size + 23, size + 23), fill=(0, 0, 0, 72))
    td.ellipse((4, 4, size + 19, size + 19), fill=(*ring_color, 255))
    td.ellipse((10, 10, size + 13, size + 13), fill=(10, 18, 30, 255))
    tile.alpha_composite(circle, (12, 12))
    return tile


def _draw_glow_text(
    frame: Image.Image,
    position: tuple[int, int],
    text: str,
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int],
    glow: tuple[int, int, int],
    anchor: str = "ma",
    stroke_width: int = 2,
) -> None:
    glow_layer = Image.new("RGBA", frame.size, (0, 0, 0, 0))
    glow_draw = ImageDraw.Draw(glow_layer, "RGBA")
    glow_draw.text(position, text, font=font, fill=(*glow, 130), anchor=anchor)
    glow_layer = glow_layer.filter(ImageFilter.GaussianBlur(radius=10))
    frame.alpha_composite(glow_layer)
    draw = ImageDraw.Draw(frame, "RGBA")
    draw.text(
        position,
        text,
        font=font,
        fill=(*fill, 255),
        anchor=anchor,
        stroke_width=stroke_width,
        stroke_fill=(0, 0, 0, 200),
    )


def _make_background() -> Image.Image:
    xx = np.linspace(0, 1, WIDTH, dtype=np.float32)
    yy = np.linspace(0, 1, HEIGHT, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(xx, yy)

    blue = np.array([13, 32, 82], dtype=np.float32)
    purple = np.array([132, 79, 255], dtype=np.float32)
    pink = np.array([214, 143, 255], dtype=np.float32)
    navy = np.array([7, 12, 24], dtype=np.float32)

    mix = np.clip(0.55 * grid_x + 0.42 * grid_y, 0.0, 1.0)
    top_glow = np.exp(-(((grid_x - 0.52) / 0.28) ** 2 + ((grid_y - 0.11) / 0.11) ** 2))
    left_glow = np.exp(-(((grid_x - 0.14) / 0.18) ** 2 + ((grid_y - 0.44) / 0.24) ** 2))
    right_glow = np.exp(-(((grid_x - 0.88) / 0.18) ** 2 + ((grid_y - 0.50) / 0.24) ** 2))
    bottom_glow = np.exp(-(((grid_x - 0.50) / 0.36) ** 2 + ((grid_y - 0.84) / 0.15) ** 2))

    img = np.clip(
        navy[None, None, :] * (1.0 - 0.88 * grid_y[..., None])
        + blue[None, None, :] * (0.62 * mix[..., None])
        + purple[None, None, :] * (0.16 * top_glow[..., None])
        + pink[None, None, :] * (0.12 * left_glow[..., None])
        + np.array([255, 255, 255], dtype=np.float32)[None, None, :] * (0.05 * right_glow[..., None])
        + np.array([255, 255, 255], dtype=np.float32)[None, None, :] * (0.03 * bottom_glow[..., None]),
        0,
        255,
    ).astype(np.uint8)
    frame = Image.fromarray(img, mode="RGB").convert("RGBA")

    overlay = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")
    draw.rounded_rectangle((40, 44, WIDTH - 40, HEIGHT - 44), radius=48, outline=(255, 255, 255, 18), width=2)
    draw.line((96, 390, WIDTH - 96, 390), fill=(255, 255, 255, 10), width=2)
    draw.line((96, 1502, WIDTH - 96, 1502), fill=(255, 255, 255, 10), width=2)
    for x in (290, 420, 552, 690, 830, 958):
        draw.line((x, 392, x, 1500), fill=(255, 255, 255, 9), width=2)
    overlay = overlay.filter(ImageFilter.GaussianBlur(radius=3))
    frame.alpha_composite(overlay)
    return frame


def _apply_motion(frame: Image.Image, t: float) -> None:
    overlay = Image.new("RGBA", frame.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")
    pulse_a = 0.5 + 0.5 * math.sin(t * 1.2)
    pulse_b = 0.5 + 0.5 * math.sin(t * 1.7 + 1.1)
    sweep = int(130 + 70 * math.sin(t * 0.9))
    draw.ellipse((10 + sweep, 90, 390 + sweep, 560), fill=(255, 255, 255, int(16 + 12 * pulse_a)))
    draw.ellipse((700 - sweep // 2, 160, 1060 - sweep // 2, 620), fill=(255, 255, 255, int(14 + 10 * pulse_b)))
    draw.ellipse((150, 1300, 930, 1810), fill=(255, 255, 255, int(8 + 8 * pulse_a)))
    overlay = overlay.filter(ImageFilter.GaussianBlur(radius=24))
    frame.alpha_composite(overlay)


def _draw_header(frame: Image.Image, title_font: ImageFont.ImageFont, subtitle_font: ImageFont.ImageFont, phase: float) -> None:
    draw = ImageDraw.Draw(frame, "RGBA")
    title_panel = (48, 58, WIDTH - 48, 360)
    draw.rounded_rectangle(title_panel, radius=42, fill=(10, 18, 34, int(216 * phase)), outline=(255, 255, 255, 18), width=2)
    _draw_glow_text(
        frame,
        (WIDTH // 2, 122),
        TITLE,
        title_font,
        (244, 247, 251),
        (255, 214, 154),
        stroke_width=3,
    )
    pill_alpha = int(255 * phase)
    draw.rounded_rectangle(
        (274, 188, 806, 258),
        radius=26,
        fill=(130, 82, 255, int(160 * phase)),
        outline=(255, 255, 255, int(35 * phase)),
        width=2,
    )
    draw.text((540, 222), SUBTITLE, font=subtitle_font, fill=(245, 250, 255, pill_alpha), anchor="mm")
    draw.text((540, 314), "Open Era career majors", font=_load_font(18, bold=True), fill=(196, 216, 238, pill_alpha), anchor="mm")


def _draw_footer(frame: Image.Image, phase: float) -> None:
    if phase <= 0.0:
        return

    draw = ImageDraw.Draw(frame, "RGBA")
    alpha = int(255 * phase)
    panel = (70, 1460, WIDTH - 70, 1770)
    draw.rounded_rectangle(panel, radius=38, fill=(8, 16, 30, int(210 * phase)), outline=(255, 255, 255, int(22 * phase)), width=2)
    draw.text((540, 1522), FOOTER.upper(), font=_load_font(22, bold=True), fill=(220, 232, 246, alpha), anchor="mm")

    chip_specs = [
        (210, PLAYERS[0]),
        (540, PLAYERS[1]),
        (870, PLAYERS[2]),
    ]
    for center_x, player in chip_specs:
        chip_w = 242
        chip_h = 126
        box = (center_x - chip_w // 2, 1562, center_x + chip_w // 2, 1688)
        fill = player.bar_color
        outline = _mix_rgb(fill, (255, 255, 255), 0.24)
        draw.rounded_rectangle(box, radius=28, fill=(*_hex_to_rgb(fill), int(220 * phase)), outline=(*outline, int(220 * phase)), width=2)
        count_color = _text_on(fill)
        count_font = _load_font(42, bold=True)
        name_font = _fit_font_size(draw, player.full_name, chip_w - 24, 24, 13, bold=True)
        draw.text((center_x, 1606), str(player.count), font=count_font, fill=count_color, anchor="mm")
        draw.text((center_x, 1649), player.surname.title(), font=name_font, fill=count_color, anchor="mm")


def _draw_country_bubble(frame: Image.Image, x: int, y: int, player: PlayerCard, phase: float) -> None:
    draw = ImageDraw.Draw(frame, "RGBA")
    size = 78
    fill = player.bar_color
    outline = _mix_rgb(fill, (255, 255, 255), 0.14)
    draw.ellipse((x, y, x + size, y + size), fill=(*_hex_to_rgb(fill), int(248 * phase)), outline=(*outline, int(240 * phase)), width=2)
    code_font = _load_font(20, bold=True)
    draw.text((x + size // 2, y + size // 2 - 1), player.country_code, font=code_font, fill=_text_on(fill), anchor="mm")


def _draw_row(
    frame: Image.Image,
    player: PlayerCard,
    row_top: int,
    reveal: float,
    avatar: Image.Image,
    name_font: ImageFont.ImageFont,
    count_font: ImageFont.ImageFont,
) -> None:
    draw = ImageDraw.Draw(frame, "RGBA")
    reveal = _ease_out(reveal)
    if reveal <= 0.0:
        return

    x_shift = int(-82 * (1.0 - reveal))
    alpha = int(255 * reveal)

    card_x = 66 + x_shift
    card_y = row_top
    card_box = (card_x, card_y, card_x + 184, card_y + 184)
    draw.rounded_rectangle((card_x + 8, card_y + 8, card_x + 192, card_y + 192), radius=34, fill=(0, 0, 0, int(82 * reveal)))
    draw.rounded_rectangle(card_box, radius=34, fill=(250, 250, 252, alpha), outline=(255, 255, 255, int(120 * reveal)), width=2)
    frame.alpha_composite(avatar, (card_x + 16, card_y + 16))

    name_y = card_y + 200
    draw.text((card_x + 92, name_y), player.full_name, font=name_font, fill=(244, 247, 251, alpha), anchor="mm")

    bar_x = 292 + x_shift
    bar_y = card_y + 30
    bar_h = 96
    bar_track_w = 640
    final_w = int(bar_track_w * (player.count / 24.0))
    fill_w = max(6, int(final_w * reveal))

    track_color = (255, 255, 255, int(16 + 18 * reveal))
    draw.rounded_rectangle((bar_x + 6, bar_y + 8, bar_x + bar_track_w + 6, bar_y + bar_h + 8), radius=bar_h // 2, fill=(0, 0, 0, int(76 * reveal)))
    draw.rounded_rectangle((bar_x, bar_y, bar_x + bar_track_w, bar_y + bar_h), radius=bar_h // 2, fill=track_color, outline=(255, 255, 255, int(24 * reveal)), width=2)

    fill_rgb = _hex_to_rgb(player.bar_color)
    outline = _mix_rgb(player.bar_color, (255, 255, 255), 0.18)
    radius = min(bar_h // 2, max(6, fill_w // 2))
    draw.rounded_rectangle((bar_x, bar_y, bar_x + fill_w, bar_y + bar_h), radius=radius, fill=(*fill_rgb, alpha), outline=(*outline, int(210 * reveal)), width=2)

    sheen_end = min(bar_x + fill_w, bar_x + max(120, int(fill_w * 0.58)))
    draw.rounded_rectangle((bar_x + 10, bar_y + 8, sheen_end, bar_y + 28), radius=10, fill=(255, 255, 255, int(36 * reveal)))

    count_text = str(max(1, int(round(player.count * reveal))))
    text_color = _text_on(player.bar_color)
    text_alpha = int(255 * _clamp((fill_w - 86) / 60.0) * reveal)
    if text_alpha > 0:
        draw.text(
            (bar_x + 26, bar_y + 49),
            count_text,
            font=count_font,
            fill=(*_hex_to_rgb(text_color), text_alpha),
            anchor="lm",
            stroke_width=2,
            stroke_fill=(0, 0, 0, int(120 * reveal)),
        )

    _draw_country_bubble(frame, bar_x + bar_track_w + 18, bar_y + 9, player, reveal)


def render_video(output_path: Path, audio_path: Path, duration: float, fps: int) -> Path:
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    background = _make_background()
    title_font = _fit_font_size(ImageDraw.Draw(background.copy()), TITLE, 920, 70, 30, bold=True)
    subtitle_font = _load_font(24, bold=True)
    name_font = _fit_font_size(ImageDraw.Draw(background.copy()), "Novak Djokovic", 170, 24, 14, bold=True)
    count_font = _load_font(60, bold=True)

    avatars = {
        player.surname: _make_avatar(
        _load_player_photo(player.photo_name, player.surname[:2]),
        _hex_to_rgb(player.ring_color),
            size=144,
        )
        for player in PLAYERS
    }

    row_starts = [1.00, 2.05, 3.10]
    row_span = 0.95
    footer_start = 5.70
    footer_span = 1.00

    def make_frame(t: float) -> np.ndarray:
        frame = background.copy()
        _apply_motion(frame, t)

        title_phase = _ease_out(min(t / 0.80, 1.0))
        _draw_header(frame, title_font, subtitle_font, title_phase)

        for idx, player in enumerate(PLAYERS):
            reveal = _clamp((t - row_starts[idx]) / row_span)
            _draw_row(
                frame,
                player,
                430 + idx * 330,
                reveal,
                avatars[player.surname],
                name_font,
                count_font,
            )

        footer_phase = _ease_in_out(_clamp((t - footer_start) / footer_span))
        _draw_footer(frame, footer_phase)
        return np.array(frame.convert("RGB"))

    clip = VideoClip(make_frame, duration=duration)
    audio_clip, keep_alive = build_audio_track(audio_path, duration)
    clip = clip.with_audio(audio_clip.with_volume_scaled(MUSIC_VOLUME))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    run_id = "gs3"
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
    parser = argparse.ArgumentParser(description="Generate a Federer, Nadal and Djokovic Grand Slam Shorts video.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--audio", type=Path, default=DEFAULT_AUDIO)
    parser.add_argument("--duration", type=float, default=TOTAL_DURATION)
    parser.add_argument("--fps", type=int, default=FPS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = render_video(args.output, args.audio, args.duration, args.fps)
    print(f"[video_generator] Federer Nadal Djokovic Grand Slam Shorts generated -> {output}")


if __name__ == "__main__":
    main()
