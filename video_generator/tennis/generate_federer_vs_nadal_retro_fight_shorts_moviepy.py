from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from moviepy import VideoClip
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageOps

from video_generator.generate_ucl_barchart_race_moviepy import (
    DEFAULT_AUDIO,
    _fit_font_size,
    _load_font,
    build_audio_track,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PHOTOS_DIR = PROJECT_ROOT / "data" / "raw" / "player_photos"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "tennis" / "federer_vs_nadal_retro_fight_shorts.mp4"

WIDTH = 1080
HEIGHT = 1920
FPS = 30
TOTAL_DURATION = 11.5
MUSIC_VOLUME = 0.42

FEDERER = {
    "name": "FEDERER",
    "skin": (227, 205, 188),
    "hair": (110, 84, 58),
    "accent": (118, 202, 255),
    "accent_soft": (188, 232, 255),
    "body": (26, 56, 88),
    "torso_light": (56, 103, 154),
}
NADAL = {
    "name": "NADAL",
    "skin": (214, 170, 138),
    "hair": (92, 70, 48),
    "accent": (255, 124, 72),
    "accent_soft": (255, 200, 146),
    "body": (94, 32, 26),
    "torso_light": (160, 58, 42),
}

STATS = [
    {"label": "GRAND SLAMS", "left_value": 20, "right_value": 22, "winner": "right", "tag": "ROUND 1", "note": "Close battle"},
    {"label": "HEAD-TO-HEAD", "left_value": 16, "right_value": 24, "winner": "right", "tag": "ROUND 2", "note": "Nadal leads"},
    {"label": "CLAY TITLES", "left_value": 11, "right_value": 63, "winner": "right", "tag": "ROUND 3", "note": "CLAY BOSS"},
    {"label": "GRASS TITLES", "left_value": 19, "right_value": 4, "winner": "left", "tag": "ROUND 4", "note": "Federer strike back"},
]

SCENES = [
    ("flash", 0.00, 0.35),
    ("hook", 0.35, 0.90),
    ("intro", 0.90, 1.40),
    ("stat_1", 1.40, 2.20),
    ("stat_2", 2.20, 3.00),
    ("clay_tease", 3.00, 3.80),
    ("stat_3", 3.80, 5.10),
    ("aftershock", 5.10, 5.80),
    ("grass_tease", 5.80, 6.60),
    ("stat_4", 6.60, 7.80),
    ("final_board", 7.80, 8.50),
    ("climax", 8.50, 9.40),
    ("goat_twist", 9.40, 10.20),
    ("ending", 10.20, TOTAL_DURATION),
]


@dataclass(frozen=True)
class SceneState:
    name: str
    start: float
    end: float
    progress: float
    index: int


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _ease_out(value: float) -> float:
    value = _clamp(value)
    return 1.0 - (1.0 - value) ** 3


def _ease_in_out(value: float) -> float:
    value = _clamp(value)
    return value * value * (3.0 - 2.0 * value)


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * _clamp(t)


def _mix_rgb(color_a: tuple[int, int, int], color_b: tuple[int, int, int], amount: float) -> tuple[int, int, int]:
    amount = _clamp(amount)
    return (
        int(color_a[0] + (color_b[0] - color_a[0]) * amount),
        int(color_a[1] + (color_b[1] - color_a[1]) * amount),
        int(color_a[2] + (color_b[2] - color_a[2]) * amount),
    )


def _scene_for_time(t: float) -> SceneState:
    for index, (name, start, end) in enumerate(SCENES):
        if start <= t < end or (index == len(SCENES) - 1 and t >= start):
            span = max(1e-6, end - start)
            return SceneState(name=name, start=start, end=end, progress=_clamp((t - start) / span), index=index)
    name, start, end = SCENES[-1]
    return SceneState(name=name, start=start, end=end, progress=1.0, index=len(SCENES) - 1)


def _score_at_time(t: float) -> tuple[float, float]:
    left = 0.0
    right = 0.0
    for stat, (_, start, end) in zip(STATS, SCENES[3:7]):
        if t >= end:
            if stat["winner"] == "left":
                left += 1.0
            else:
                right += 1.0
        elif start <= t < end:
            local = _ease_out((t - start) / max(1e-6, end - start))
            if stat["winner"] == "left":
                left += local
            else:
                right += local
            break
    return left, right


def _score_text(score: tuple[float, float]) -> str:
    return f"{int(round(score[0]))} - {int(round(score[1]))}"


def _load_portrait(path: Path, initials: str) -> Image.Image:
    # Kept for compatibility with older templates; this template does not use photos.
    img = Image.new("RGBA", (720, 720), (10, 16, 26, 255))
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle((24, 24, 696, 696), radius=60, fill=(22, 34, 48, 255), outline=(255, 255, 255, 72), width=4)
    draw.text((360, 356), initials, font=_load_font(72, bold=True), fill=(245, 248, 252, 255), anchor="mm")
    return img


def _shade_sphere(size: int, core: tuple[int, int, int], edge: tuple[int, int, int], highlight: tuple[int, int, int]) -> Image.Image:
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img, "RGBA")
    for radius in range(size // 2, 0, -2):
        alpha = int(255 * (radius / (size / 2)) ** 1.7)
        color = _mix_rgb(core, edge, 1.0 - radius / (size / 2))
        draw.ellipse((size // 2 - radius, size // 2 - radius, size // 2 + radius, size // 2 + radius), fill=(*color, alpha))
    hl = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    hd = ImageDraw.Draw(hl, "RGBA")
    hd.ellipse((size * 0.18, size * 0.14, size * 0.52, size * 0.46), fill=(*highlight, 92))
    hl = hl.filter(ImageFilter.GaussianBlur(radius=size * 0.04))
    img.alpha_composite(hl)
    return img


def _figurine_head(fighter: dict, size: int = 210) -> Image.Image:
    skin = fighter["skin"]
    hair = fighter["hair"]
    head = _shade_sphere(size, skin, _mix_rgb(skin, (40, 22, 18), 0.26), (255, 255, 255))
    draw = ImageDraw.Draw(head, "RGBA")
    # Jaw and chin.
    jaw = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    jd = ImageDraw.Draw(jaw, "RGBA")
    jd.ellipse((26, 38, size - 26, size - 6), fill=(*_mix_rgb(skin, (175, 144, 124), 0.18), 255))
    jaw = jaw.filter(ImageFilter.GaussianBlur(radius=5))
    head.alpha_composite(jaw)
    # Hair cap.
    hair_layer = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    hd = ImageDraw.Draw(hair_layer, "RGBA")
    if fighter["name"] == "FEDERER":
        hd.pieslice((14, 12, size - 20, size - 6), start=195, end=350, fill=(*hair, 255))
        hd.rectangle((38, 54, size - 54, 112), fill=(*hair, 255))
        hd.arc((16, 18, size - 28, size - 18), start=194, end=350, fill=(*hair, 255), width=18)
    else:
        hd.rounded_rectangle((28, 16, size - 30, 104), radius=40, fill=(*hair, 255))
        hd.ellipse((28, 10, size - 28, 102), fill=(*hair, 255))
        hd.polygon([(38, 72), (62, 44), (88, 70), (114, 38), (142, 72), (166, 44), (182, 68)], fill=(*hair, 255))
    hair_layer = hair_layer.filter(ImageFilter.GaussianBlur(radius=1.6))
    head.alpha_composite(hair_layer)
    # Face features.
    eye_y = int(size * 0.52)
    eye_offset = int(size * 0.11)
    eye_color = (36, 28, 26, 255)
    draw.ellipse((size * 0.30, eye_y, size * 0.36, eye_y + 10), fill=eye_color)
    draw.ellipse((size * 0.64, eye_y, size * 0.70, eye_y + 10), fill=eye_color)
    draw.line((size * 0.38, eye_y + 22, size * 0.62, eye_y + 22), fill=(86, 55, 42, 170), width=3)
    draw.line((size * 0.50, size * 0.56, size * 0.48, size * 0.72), fill=(120, 78, 58, 180), width=3)
    mouth_y = int(size * 0.74)
    if fighter["name"] == "FEDERER":
        draw.arc((size * 0.38, mouth_y - 6, size * 0.62, mouth_y + 16), start=12, end=168, fill=(92, 44, 50, 170), width=3)
    else:
        draw.arc((size * 0.38, mouth_y - 8, size * 0.62, mouth_y + 12), start=8, end=172, fill=(110, 36, 30, 190), width=3)
    # Ear shapes.
    draw.ellipse((size * 0.13, size * 0.50, size * 0.20, size * 0.66), fill=(*_mix_rgb(skin, (185, 150, 120), 0.14), 255))
    draw.ellipse((size * 0.80, size * 0.50, size * 0.87, size * 0.66), fill=(*_mix_rgb(skin, (185, 150, 120), 0.14), 255))
    # Specular sweep.
    shine = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    sd = ImageDraw.Draw(shine, "RGBA")
    sd.ellipse((size * 0.15, size * 0.10, size * 0.56, size * 0.46), fill=(255, 255, 255, 54))
    shine = shine.filter(ImageFilter.GaussianBlur(radius=6))
    head.alpha_composite(shine)
    return head


def _draw_glow_text(
    frame: Image.Image,
    position: tuple[int, int],
    text: str,
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int],
    glow: tuple[int, int, int],
    anchor: str = "mm",
    stroke_width: int = 2,
) -> None:
    blur_layer = Image.new("RGBA", frame.size, (0, 0, 0, 0))
    blur_draw = ImageDraw.Draw(blur_layer, "RGBA")
    blur_draw.text(position, text, font=font, fill=(*glow, 130), anchor=anchor)
    blur_layer = blur_layer.filter(ImageFilter.GaussianBlur(radius=10))
    frame.alpha_composite(blur_layer)
    draw = ImageDraw.Draw(frame, "RGBA")
    draw.text(position, text, font=font, fill=(*fill, 255), anchor=anchor, stroke_width=stroke_width, stroke_fill=(0, 0, 0, 210))


def _make_background(kind: str, t: float) -> Image.Image:
    xx = np.linspace(0, 1, WIDTH, dtype=np.float32)
    yy = np.linspace(0, 1, HEIGHT, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(xx, yy)

    if kind == "clay":
        base = np.array([20, 10, 12], dtype=np.float32)
        left = np.array(FEDERER["accent"], dtype=np.float32)
        right = np.array((255, 150, 94), dtype=np.float32)
        floor = np.array((132, 36, 34), dtype=np.float32)
    elif kind == "grass":
        base = np.array([8, 16, 18], dtype=np.float32)
        left = np.array((114, 214, 142), dtype=np.float32)
        right = np.array(NADAL["accent"], dtype=np.float32)
        floor = np.array((16, 84, 64), dtype=np.float32)
    else:
        base = np.array([8, 10, 20], dtype=np.float32)
        left = np.array(FEDERER["accent"], dtype=np.float32)
        right = np.array(NADAL["accent"], dtype=np.float32)
        floor = np.array((20, 32, 60), dtype=np.float32)

    pulse = 0.5 + 0.5 * math.sin(t * 2.0)
    left_glow = np.exp(-(((grid_x - 0.18) / 0.22) ** 2 + ((grid_y - 0.34) / 0.22) ** 2))
    right_glow = np.exp(-(((grid_x - 0.82) / 0.22) ** 2 + ((grid_y - 0.34) / 0.22) ** 2))
    center_glow = np.exp(-(((grid_x - 0.50) / 0.28) ** 2 + ((grid_y - 0.56) / 0.22) ** 2))
    img = np.clip(
        base[None, None, :] * (1.0 - 0.74 * grid_y[..., None])
        + floor[None, None, :] * (0.55 * grid_y[..., None])
        + left[None, None, :] * (0.20 * left_glow[..., None] * (0.7 + 0.3 * pulse))
        + right[None, None, :] * (0.20 * right_glow[..., None] * (0.7 + 0.3 * (1.0 - pulse)))
        + np.array((255, 255, 255), dtype=np.float32)[None, None, :] * (0.025 * center_glow[..., None]),
        0,
        255,
    ).astype(np.uint8)

    frame = Image.fromarray(img, mode="RGB").convert("RGBA")
    overlay = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")
    vanishing_x = WIDTH // 2
    horizon_y = 1030
    for y in range(horizon_y + 16, HEIGHT, 56):
        alpha = int(46 * (1.0 - (y - horizon_y) / max(1, HEIGHT - horizon_y)))
        draw.line((90, y, WIDTH - 90, y), fill=(120, 210, 255, alpha), width=2)
    for x in range(0, WIDTH + 1, 96):
        draw.line((vanishing_x, horizon_y, x, HEIGHT), fill=(120, 210, 255, 24), width=2)
    draw.rounded_rectangle((36, 54, WIDTH - 36, HEIGHT - 54), radius=52, outline=(255, 255, 255, 16), width=2)
    draw.line((92, 318, WIDTH - 92, 318), fill=(255, 255, 255, 10), width=2)
    draw.ellipse((0, 180, 330, 560), fill=(*FEDERER["accent"], 18))
    draw.ellipse((750, 180, 1080, 560), fill=(*NADAL["accent"], 18))
    draw.ellipse((180, 740, 900, 1620), outline=(255, 255, 255, 10), width=4)
    draw.rectangle((0, 0, WIDTH, 24), fill=(255, 255, 255, 22))
    draw.rectangle((0, 24, WIDTH, 48), fill=(0, 0, 0, 22))
    draw.rectangle((0, HEIGHT - 28, WIDTH, HEIGHT), fill=(0, 0, 0, 64))
    if kind == "clay":
        draw.ellipse((120, 720, 640, 1220), fill=(255, 120, 84, 16))
        draw.ellipse((440, 690, 980, 1230), fill=(255, 154, 90, 10))
    elif kind == "grass":
        draw.ellipse((110, 760, 560, 1210), fill=(120, 214, 142, 18))
        draw.ellipse((520, 720, 980, 1210), fill=(120, 214, 142, 12))
    overlay = overlay.filter(ImageFilter.GaussianBlur(radius=10))
    frame.alpha_composite(overlay)
    return frame


def _scanline_overlay(alpha: int = 34) -> Image.Image:
    layer = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer, "RGBA")
    for y in range(0, HEIGHT, 4):
        draw.line((0, y, WIDTH, y), fill=(0, 0, 0, alpha))
    return layer


def _pixel_burst(center: tuple[int, int], color: tuple[int, int, int], size: int, count: int = 12) -> Image.Image:
    layer = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer, "RGBA")
    cx, cy = center
    for i in range(count):
        angle = i * (2.0 * math.pi / count)
        dx = int(math.cos(angle) * size)
        dy = int(math.sin(angle) * size)
        draw.line((cx, cy, cx + dx, cy + dy), fill=(*color, 255), width=5)
        draw.ellipse((cx + dx - 4, cy + dy - 4, cx + dx + 4, cy + dy + 4), fill=(*color, 255))
    return layer.filter(ImageFilter.GaussianBlur(radius=1.2))


def _draw_sprite(
    frame: Image.Image,
    fighter: dict,
    x: int,
    y: int,
    facing: str,
    attack: float,
    recoil: float,
    damage: float,
    t: float,
    knockout: bool = False,
) -> tuple[int, int]:
    draw = ImageDraw.Draw(frame, "RGBA")
    accent = fighter["accent"]
    accent_soft = fighter["accent_soft"]
    body = fighter["body"]
    torso_light = fighter["torso_light"]
    attack = _clamp(attack)
    recoil = _clamp(recoil)
    damage = _clamp(damage)
    flip = -1 if facing == "left" else 1

    lean = flip * (attack * 26 - recoil * 18)
    bob = int(math.sin(t * 5.0 + (0.0 if facing == "left" else 1.2)) * 8 * (1.0 - damage))
    body_x = x + int(lean)
    body_y = y + bob + int(recoil * 18)

    shadow = Image.new("RGBA", frame.size, (0, 0, 0, 0))
    sd = ImageDraw.Draw(shadow, "RGBA")
    sd.ellipse((body_x - 126, body_y + 262, body_x + 126, body_y + 312), fill=(0, 0, 0, 110))
    shadow = shadow.filter(ImageFilter.GaussianBlur(radius=18))
    frame.alpha_composite(shadow)

    glow = Image.new("RGBA", frame.size, (0, 0, 0, 0))
    gd = ImageDraw.Draw(glow, "RGBA")
    gd.ellipse((body_x - 190, body_y - 280, body_x + 190, body_y + 260), fill=(*accent, 22))
    glow = glow.filter(ImageFilter.GaussianBlur(radius=30))
    frame.alpha_composite(glow)

    # Base pedestal to read as a collectible figurine rather than a cartoon avatar.
    pedestal = (body_x - 126, body_y + 240, body_x + 126, body_y + 310)
    draw.rounded_rectangle(pedestal, radius=26, fill=(12, 16, 24, 230), outline=(255, 255, 255, 18), width=2)
    draw.rounded_rectangle((pedestal[0] + 16, pedestal[1] + 14, pedestal[2] - 16, pedestal[3] - 20), radius=20, fill=(22, 30, 44, 210))

    torso_top = body_y - 132
    torso_bottom = body_y + 64
    chest = (body_x - 88, torso_top, body_x + 88, torso_bottom)
    draw.rounded_rectangle(chest, radius=46, fill=body, outline=(*accent, 210), width=3)
    # Chest highlight and shadow bands.
    draw.ellipse((chest[0] + 12, chest[1] + 18, chest[2] - 28, chest[1] + 92), fill=(*torso_light, 50))
    draw.ellipse((chest[0] + 28, chest[1] + 24, chest[2] - 12, chest[3] - 10), outline=(255, 255, 255, 16), width=2)
    draw.rounded_rectangle((chest[0] + 44, chest[1] + 32, chest[0] + 128, chest[1] + 166), radius=30, fill=(*accent, 35))

    # Neck.
    neck = (body_x - 26, torso_top - 34, body_x + 26, torso_top + 20)
    draw.rounded_rectangle(neck, radius=12, fill=(*accent_soft, 244), outline=(0, 0, 0, 54), width=1)

    # Arms with a more sculpted, articulated shape.
    shoulder_y = torso_top + 22
    punch = 104 * attack * (1.0 if knockout else 0.82)
    recoil_shift = 56 * recoil
    shoulder_span = 80
    if facing == "left":
        punch_arm = [
            (body_x + 50, shoulder_y + 6),
            (body_x + 120 + int(punch), shoulder_y - 14 - int(recoil_shift)),
            (body_x + 194 + int(punch * 1.2), shoulder_y - 24 - int(recoil_shift * 0.4)),
        ]
        back_arm = [
            (body_x - 44, shoulder_y + 8),
            (body_x - 114, shoulder_y - 18),
            (body_x - 160, shoulder_y - 2),
        ]
    else:
        punch_arm = [
            (body_x - 50, shoulder_y + 6),
            (body_x - 120 - int(punch), shoulder_y - 14 - int(recoil_shift)),
            (body_x - 194 - int(punch * 1.2), shoulder_y - 24 - int(recoil_shift * 0.4)),
        ]
        back_arm = [
            (body_x + 44, shoulder_y + 8),
            (body_x + 114, shoulder_y - 18),
            (body_x + 160, shoulder_y - 2),
        ]
    draw.line(back_arm, fill=(*accent, 190), width=20, joint="curve")
    draw.line(punch_arm, fill=(*accent_soft, 245), width=24, joint="curve")
    draw.ellipse((back_arm[-1][0] - 18, back_arm[-1][1] - 18, back_arm[-1][0] + 18, back_arm[-1][1] + 18), fill=(*accent, 220))
    draw.ellipse((punch_arm[-1][0] - 24, punch_arm[-1][1] - 24, punch_arm[-1][0] + 24, punch_arm[-1][1] + 24), fill=(*accent_soft, 255))

    hip_y = torso_bottom - 2
    left_knee = (body_x - 48 * flip, hip_y + 98 + int(recoil * 10))
    right_knee = (body_x + 44 * flip, hip_y + 102 - int(recoil * 8))
    left_foot = (left_knee[0] - 18 * flip, left_knee[1] + 108)
    right_foot = (right_knee[0] + 14 * flip, right_knee[1] + 110)
    draw.line((body_x - 26, hip_y, left_knee[0], left_knee[1]), fill=(*accent_soft, 230), width=20, joint="curve")
    draw.line((left_knee[0], left_knee[1], left_foot[0], left_foot[1]), fill=(12, 14, 18, 255), width=20)
    draw.line((body_x + 28, hip_y, right_knee[0], right_knee[1]), fill=(*accent_soft, 215), width=18, joint="curve")
    draw.line((right_knee[0], right_knee[1], right_foot[0], right_foot[1]), fill=(12, 14, 18, 255), width=18)
    draw.ellipse((left_foot[0] - 18, left_foot[1] - 8, left_foot[0] + 40, left_foot[1] + 18), fill=(16, 18, 22, 255))
    draw.ellipse((right_foot[0] - 18, right_foot[1] - 8, right_foot[0] + 40, right_foot[1] + 18), fill=(16, 18, 22, 255))

    # Head as a sculpted volume, no photo texture.
    head_size = 220
    head_y = body_y - 292 + int(recoil * 16)
    head_x = body_x + int(recoil * 6 * flip)
    head = _figurine_head(fighter, size=head_size)
    head = head.rotate((-8 + attack * 6 - recoil * 5) if facing == "left" else (8 - attack * 6 + recoil * 5), resample=Image.Resampling.BICUBIC)
    frame.alpha_composite(head, (head_x - head.width // 2, head_y - head.height // 2))

    # Cheek/jaw light to sell the sculpted shape.
    contour = Image.new("RGBA", frame.size, (0, 0, 0, 0))
    cd = ImageDraw.Draw(contour, "RGBA")
    cd.ellipse((head_x - 82, head_y - 112, head_x + 82, head_y + 92), outline=(255, 255, 255, 24), width=2)
    cd.ellipse((head_x - 42, head_y - 18, head_x + 42, head_y + 34), fill=(255, 255, 255, 18))
    contour = contour.filter(ImageFilter.GaussianBlur(radius=6))
    frame.alpha_composite(contour)

    if damage > 0.0:
        spark = Image.new("RGBA", frame.size, (0, 0, 0, 0))
        sd = ImageDraw.Draw(spark, "RGBA")
        for i in range(6):
            ang = i * (math.pi / 3.0)
            dx = int(math.cos(ang) * (42 + damage * 38))
            dy = int(math.sin(ang) * (34 + damage * 20))
            sd.line((head_x, head_y - 88, head_x + dx, head_y - 88 + dy), fill=(255, 238, 160, 255), width=5)
        sd.ellipse((head_x - 58, head_y - 130, head_x + 58, head_y - 22), outline=(255, 210, 120, 255), width=8)
        spark = spark.filter(ImageFilter.GaussianBlur(radius=1.6))
        frame.alpha_composite(spark)
    if knockout:
        ko = Image.new("RGBA", frame.size, (0, 0, 0, 0))
        kd = ImageDraw.Draw(ko, "RGBA")
        kd.text((head_x, head_y - 170), "K.O.", font=_load_font(40, bold=True), fill=(255, 240, 178, 255), anchor="mm", stroke_width=3, stroke_fill=(0, 0, 0, 220))
        ko = ko.filter(ImageFilter.GaussianBlur(radius=1.0))
        frame.alpha_composite(ko)
    return head_x, head_y


def _draw_hud(frame: Image.Image, score: tuple[float, float], t: float, scene: SceneState) -> None:
    draw = ImageDraw.Draw(frame, "RGBA")
    left_score, right_score = score
    top_bar = (56, 68, WIDTH - 56, 274)
    draw.rounded_rectangle(top_bar, radius=42, fill=(8, 12, 20, 210), outline=(255, 255, 255, 18), width=2)
    draw.text((120, 114), "FIGURE DUEL", font=_load_font(22, bold=True), fill="#dce7f2")
    draw.text((WIDTH - 120, 114), "COLLECTOR EDITION", font=_load_font(18, bold=True), fill="#ffe0b5", anchor="ra")
    _draw_glow_text(frame, (WIDTH // 2, 116), "FEDERER vs NADAL", _load_font(44, bold=True), (245, 248, 252), FEDERER["accent"], stroke_width=3)

    left_energy = int(280 * (1.0 - 0.22 * left_score))
    right_energy = int(280 * (1.0 - 0.22 * right_score))
    draw.rounded_rectangle((126, 188, 426, 218), radius=12, fill=(255, 255, 255, 20))
    draw.rounded_rectangle((654, 188, 954, 218), radius=12, fill=(255, 255, 255, 20))
    draw.rounded_rectangle((126, 188, 126 + left_energy, 218), radius=12, fill=(*FEDERER["accent"], 240))
    draw.rounded_rectangle((954 - right_energy, 188, 954, 218), radius=12, fill=(*NADAL["accent"], 240))
    draw.text((126, 228), f"{FEDERER['name']} HP", font=_load_font(16, bold=True), fill="#dce7f2")
    draw.text((954, 228), f"{NADAL['name']} HP", font=_load_font(16, bold=True), fill="#ffe2d3", anchor="ra")

    center_box = (418, 148, 662, 242)
    draw.rounded_rectangle(center_box, radius=26, fill=(10, 18, 30, 234), outline=(255, 255, 255, 18), width=2)
    draw.text((540, 182), _score_text(score), font=_load_font(38, bold=True), fill="#f4f7fb", anchor="mm")
    draw.text((540, 220), "STAT ARENA", font=_load_font(18, bold=True), fill="#dce7f2", anchor="mm")

    heat = 0.45 + 0.35 * math.sin(t * 3.0)
    if scene.name in {"stat_3", "climax"}:
        heat += 0.25
    draw.rounded_rectangle((478, 250, 602, 264), radius=7, fill=(255, 255, 255, 18))
    draw.rounded_rectangle((478, 250, 478 + int(124 * _clamp(heat)), 264), radius=7, fill=(255, 208, 138, 220))


def _draw_stat_panel(frame: Image.Image, stat: dict, phase: float, score: tuple[float, float]) -> None:
    draw = ImageDraw.Draw(frame, "RGBA")
    panel = (44, 1216, 1036, 1798)
    draw.rounded_rectangle(panel, radius=48, fill=(8, 12, 20, 230), outline=(255, 255, 255, 18), width=2)
    draw.text((108, 1256), stat["tag"], font=_load_font(18, bold=True), fill="#f4f7fb")
    draw.text((540, 1256), "STAT BATTLE", font=_load_font(18, bold=True), fill="#dce7f2", anchor="mm")
    draw.text((972, 1256), "1 POINT", font=_load_font(18, bold=True), fill="#f4f7fb", anchor="ra")
    _draw_glow_text(frame, (540, 1338), stat["label"], _fit_font_size(draw, stat["label"], 850, 84, 30, bold=True), (245, 248, 252), (255, 214, 156), stroke_width=3)

    reveal = _ease_out(phase)
    left_value = str(int(round(stat["left_value"] * reveal)))
    right_value = str(int(round(stat["right_value"] * reveal)))
    left_font = _fit_font_size(draw, left_value, 200, 88, 36, bold=True)
    right_font = _fit_font_size(draw, right_value, 200, 88, 36, bold=True)
    left_box = (100, 1410, 428, 1602)
    right_box = (652, 1410, 980, 1602)
    draw.rounded_rectangle(left_box, radius=36, fill=(16, 34, 54, 226), outline=(*FEDERER["accent"], 160), width=2)
    draw.rounded_rectangle(right_box, radius=36, fill=(58, 22, 22, 226), outline=(*NADAL["accent"], 180), width=2)
    draw.text((264, 1444), FEDERER["name"], font=_load_font(20, bold=True), fill="#dce7f2", anchor="mm")
    draw.text((816, 1444), NADAL["name"], font=_load_font(20, bold=True), fill="#ffe2d3", anchor="mm")
    _draw_glow_text(frame, (264, 1516), left_value, left_font, (248, 250, 252), FEDERER["accent"], stroke_width=3)
    _draw_glow_text(frame, (816, 1516), right_value, right_font, (248, 250, 252), NADAL["accent"], stroke_width=3)
    _draw_glow_text(frame, (540, 1642), _score_text(score), _load_font(42, bold=True), (245, 248, 252), (255, 220, 160), stroke_width=3)
    _draw_glow_text(frame, (540, 1714), stat["note"].upper(), _load_font(34, bold=True), (255, 242, 220), NADAL["accent"] if stat["winner"] == "right" else FEDERER["accent"], stroke_width=2)


def _draw_scene_text(frame: Image.Image, text: str, y: int, size: int, fill: tuple[int, int, int], glow: tuple[int, int, int]) -> None:
    _draw_glow_text(frame, (WIDTH // 2, y), text, _load_font(size, bold=True), fill, glow, stroke_width=3)


def _fighter_pose(scene: SceneState, fighter_side: str, stat_index: int | None) -> tuple[float, float, float, bool]:
    if scene.name in {"flash", "hook", "intro", "final_board", "goat_twist", "ending"} or stat_index is None:
        return 0.0, 0.0, 0.0, False
    winner = "right" if stat_index in (0, 1, 2) else "left"
    charge = _ease_out(scene.progress)
    if fighter_side == winner:
        return 0.14 * charge, 0.0, 0.02 * charge, stat_index == 2 and fighter_side == "right"
    return -0.08 * charge, 0.36 * charge, 0.10 * charge, stat_index == 2 and fighter_side != winner


def _scene_kind(scene: SceneState) -> str:
    if scene.name in {"clay_tease", "stat_3", "aftershock"}:
        return "clay"
    if scene.name in {"grass_tease", "stat_4", "ending"}:
        return "grass"
    return "split"


def _draw_stage_overlay(frame: Image.Image, t: float) -> None:
    draw = ImageDraw.Draw(frame, "RGBA")
    for i in range(12):
        x = 160 + i * 74
        draw.rectangle((x, 1118, x + 34, 1132), fill=(255, 255, 255, 18))
    if int(t * 4) % 2 == 0:
        draw.text((540, 1180), "SHOWCASE MODE", font=_load_font(18, bold=True), fill="#dce7f2", anchor="mm")


def render_video(output_path: Path, audio_path: Path, duration: float, fps: int) -> Path:
    scanlines = _scanline_overlay()

    def make_frame(t: float) -> np.ndarray:
        scene = _scene_for_time(t)
        frame = _make_background(_scene_kind(scene), t)
        frame.alpha_composite(scanlines)
        score = _score_at_time(t)
        draw = ImageDraw.Draw(frame, "RGBA")

        if scene.name == "flash":
            overlay = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
            od = ImageDraw.Draw(overlay, "RGBA")
            od.rectangle((0, 0, WIDTH, HEIGHT), fill=(0, 0, 0, int(240 * _ease_out(scene.progress))))
            od.text((540, 920), "FEDERER vs NADAL", font=_fit_font_size(od, "FEDERER vs NADAL", 960, 90, 30, bold=True), fill=(255, 248, 236, 255), anchor="mm", stroke_width=3, stroke_fill=(0, 0, 0, 220))
            frame.alpha_composite(overlay)
            return np.array(frame.convert("RGB"))

        _draw_hud(frame, score, t, scene)
        _draw_stage_overlay(frame, t)

        if scene.name == "hook":
            _draw_scene_text(frame, "LE DUEL QUI DIVISE TOUT LE MONDE", 318, 40, (255, 243, 228), (255, 200, 140))
        elif scene.name == "intro":
            _draw_scene_text(frame, "4 STATS. 1 VAINQUEUR.", 336, 38, (230, 238, 246), (255, 220, 160))
            _draw_scene_text(frame, "SHOWDOWN", 392, 34, (255, 242, 220), (255, 208, 140))
        elif scene.name == "clay_tease":
            _draw_scene_text(frame, "ET SUR TERRE BATTUE...", 320, 40, (255, 244, 236), (255, 140, 92))
        elif scene.name == "aftershock":
            _draw_scene_text(frame, "63 SUR CLAY. C'EST ABSURDE.", 330, 36, (255, 244, 236), (255, 140, 92))
        elif scene.name == "grass_tease":
            _draw_scene_text(frame, "MAIS SUR GAZON...", 320, 40, (234, 244, 252), (120, 214, 142))
        elif scene.name == "final_board":
            _draw_scene_text(frame, "SCORE FINAL 1 - 3", 330, 66, (255, 248, 236), (255, 210, 155))
        elif scene.name == "climax":
            draw.rounded_rectangle((172, 948, 908, 1286), radius=44, fill=(108, 20, 28, 190), outline=(255, 124, 124, 230), width=10)
            _draw_scene_text(frame, "NADAL WINS THIS ONE", 1088, 70, (255, 240, 240), NADAL["accent"])
        elif scene.name == "goat_twist":
            _draw_scene_text(frame, "MAIS EST-CE ASSEZ POUR ETRE LE GOAT ?", 1488, 34, (245, 248, 252), (255, 220, 160))
        elif scene.name == "ending":
            draw.rounded_rectangle((194, 1530, 886, 1712), radius=38, fill=(10, 18, 30, 220), outline=(255, 255, 255, 24), width=2)
            draw.text((540, 1596), "TU CHOISIS QUI ?", font=_load_font(32, bold=True), fill="#f4f7fb", anchor="mm")
            draw.text((540, 1646), "TEAM FEDERER OU TEAM NADAL ?", font=_load_font(22, bold=True), fill="#dce7f2", anchor="mm")

        stat_index = {"stat_1": 0, "stat_2": 1, "stat_3": 2, "stat_4": 3}.get(scene.name)
        if stat_index is not None:
            left_attack, left_recoil, left_damage, left_ko = _fighter_pose(scene, "left", stat_index)
            right_attack, right_recoil, right_damage, right_ko = _fighter_pose(scene, "right", stat_index)
        else:
            left_attack = left_recoil = left_damage = 0.0
            right_attack = right_recoil = right_damage = 0.0
            left_ko = right_ko = False

        shake = 0
        if scene.name in {"stat_3", "climax"}:
            shake = int(10 * _ease_out(scene.progress))
        elif scene.name in {"stat_1", "stat_2", "stat_4"}:
            shake = int(3 * _ease_out(scene.progress))

        if scene.name in {"stat_1", "stat_2", "stat_3", "stat_4"}:
            winner = STATS[stat_index]["winner"]
            impact_phase = _ease_in_out(scene.progress)
            if winner == "right":
                cx = int(_lerp(420, 660, impact_phase))
                cy = int(_lerp(1220, 1180, impact_phase))
                frame.alpha_composite(_pixel_burst((cx, cy), NADAL["accent_soft"], int(16 + 28 * impact_phase), count=10))
                if impact_phase > 0.72:
                    draw.text((540, 1168), "HIT!", font=_load_font(48, bold=True), fill="#ffe0b8", anchor="mm", stroke_width=3, stroke_fill=(0, 0, 0, 220))
            else:
                cx = int(_lerp(660, 420, impact_phase))
                cy = int(_lerp(1220, 1180, impact_phase))
                frame.alpha_composite(_pixel_burst((cx, cy), FEDERER["accent_soft"], int(16 + 28 * impact_phase), count=10))
                if impact_phase > 0.72:
                    draw.text((540, 1168), "COUNTER!", font=_load_font(44, bold=True), fill="#dff0ff", anchor="mm", stroke_width=3, stroke_fill=(0, 0, 0, 220))

        if scene.name in {"stat_3", "climax"} and _ease_out(scene.progress) > 0.65:
            frame.alpha_composite(_pixel_burst((742, 1220), NADAL["accent"], int(42 + 36 * scene.progress), count=14))
        if scene.name == "stat_4" and _ease_out(scene.progress) > 0.7:
            frame.alpha_composite(_pixel_burst((338, 1220), FEDERER["accent"], int(34 + 28 * scene.progress), count=12))

        left_center = _draw_sprite(frame, FEDERER, 300 - shake, 1278 + int(6 * math.sin(t * 4.0)), "left", left_attack, left_recoil, left_damage, t, knockout=left_ko)
        right_center = _draw_sprite(frame, NADAL, 780 + shake, 1278 + int(6 * math.cos(t * 4.2)), "right", right_attack, right_recoil, right_damage, t, knockout=right_ko)

        if stat_index is not None:
            _draw_stat_panel(frame, STATS[stat_index], scene.progress, score)

        if scene.name == "final_board":
            draw.text((540, 1150), "STAT BATTLE WINNER", font=_load_font(22, bold=True), fill="#dce7f2", anchor="mm")
        elif scene.name == "climax":
            draw.text((540, 1400), "UNPLAYABLE ON CLAY", font=_load_font(24, bold=True), fill="#ffead6", anchor="mm")
        elif scene.name == "goat_twist":
            draw.text((540, 1546), "RE-PLAY? RE-TWEET? RE-ARGUE.", font=_load_font(20, bold=True), fill="#dce7f2", anchor="mm")

        return np.array(frame.convert("RGB"))

    clip = VideoClip(make_frame, duration=duration)
    audio_track, keep_alive = build_audio_track(audio_path, duration)
    clip = clip.with_audio(audio_track.with_volume_scaled(MUSIC_VOLUME))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_output = output_path.with_name(f"{output_path.stem}.render.mp4")
    tmp_audio = output_path.with_name(f"{output_path.stem}.temp_audio.m4a")
    try:
        clip.write_videofile(
            str(tmp_output),
            fps=fps,
            codec="libx264",
            audio_codec="aac",
            bitrate="12000k",
            preset="slow",
            temp_audiofile=str(tmp_audio),
            remove_temp=False,
        )
        if output_path.exists():
            output_path.unlink()
        tmp_output.replace(output_path)
    finally:
        clip.close()
        audio_track.close()
        for item in keep_alive:
            item.close()
        if tmp_audio.exists():
            try:
                tmp_audio.unlink()
            except OSError:
                pass
        if tmp_output.exists() and tmp_output != output_path:
            try:
                tmp_output.unlink()
            except OSError:
                pass
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a retro Federer vs Nadal fight Shorts video.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--audio", type=Path, default=DEFAULT_AUDIO)
    parser.add_argument("--duration", type=float, default=TOTAL_DURATION)
    parser.add_argument("--fps", type=int, default=FPS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = render_video(args.output, args.audio, args.duration, args.fps)
    print(f"[video_generator] Federer vs Nadal retro fight Shorts generated -> {output}")


if __name__ == "__main__":
    main()
