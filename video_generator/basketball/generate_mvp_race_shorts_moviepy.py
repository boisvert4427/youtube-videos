from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import numpy as np
from moviepy import AudioFileClip, CompositeAudioClip, VideoClip
from moviepy.audio.fx import AudioFadeOut
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageOps

from video_generator.generate_ucl_barchart_race_moviepy import DEFAULT_AUDIO, build_audio_track


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "basketball" / "mvp_race_shorts.mp4"
DEFAULT_PREVIEW = PROJECT_ROOT / "data" / "processed" / "basketball" / "mvp_race_shorts_preview.mp4"
DEFAULT_ASSETS_DIR = PROJECT_ROOT / "data" / "raw" / "mvp_race_assets"
DEFAULT_MUSIC = DEFAULT_AUDIO

WIDTH = 1080
HEIGHT = 1920
FPS = 30
DURATION = 30.0
SAFE_X = 64
FINAL_AUDIO_FADE_OUT = 5.0
PARTICLE_SEED = 17

HOOK_DURATION = 2.0
INTRO_DURATION = 4.0
STATS_DURATION = 12.0
SCORE_DURATION = 6.0
PODIUM_DURATION = 4.0
CTA_DURATION = 2.0

BACKGROUND_TOP = (8, 10, 18)
BACKGROUND_BOTTOM = (20, 26, 40)
WHITE = (247, 247, 242)
MUTED = (176, 188, 210)
GOLD = (255, 208, 98)
SILVER = (186, 200, 214)
BRONZE = (194, 127, 78)
BLACK = (6, 8, 14)


PLAYERS = [
    {
        "name": "Nikola Jokic",
        "short_name": "Jokic",
        "color": "#FDB927",
        "secondary_color": "#0E2240",
        "points": 29.0,
        "assists": 10.2,
        "rebounds": 12.8,
        "team_win_pct": 0.722,
        "score": 96,
        "image": "assets/jokic.png",
    },
    {
        "name": "Shai Gilgeous-Alexander",
        "short_name": "SGA",
        "color": "#007AC1",
        "secondary_color": "#EF3B24",
        "points": 31.1,
        "assists": 6.1,
        "rebounds": 5.5,
        "team_win_pct": 0.611,
        "score": 93,
        "image": "assets/sga.png",
    },
    {
        "name": "Luka Doncic",
        "short_name": "Doncic",
        "color": "#00538C",
        "secondary_color": "#B8C4CA",
        "points": 33.7,
        "assists": 9.8,
        "rebounds": 9.1,
        "team_win_pct": 0.667,
        "score": 91,
        "image": "assets/doncic.png",
    },
]


STAT_DEFS = [
    {"key": "points", "label": "POINTS PER GAME", "suffix": "", "decimals": 1, "inverse": False},
    {"key": "assists", "label": "ASSISTS PER GAME", "suffix": "", "decimals": 1, "inverse": False},
    {"key": "rebounds", "label": "REBOUNDS PER GAME", "suffix": "", "decimals": 1, "inverse": False},
    {"key": "team_win_pct", "label": "TEAM WIN %", "suffix": "%", "decimals": 1, "inverse": False},
]


@dataclass(frozen=True)
class Player:
    name: str
    short_name: str
    color: tuple[int, int, int]
    secondary_color: tuple[int, int, int]
    points: float
    assists: float
    rebounds: float
    team_win_pct: float
    score: int
    image_path: Path | None


@dataclass(frozen=True)
class StatDef:
    key: str
    label: str
    suffix: str
    decimals: int
    inverse: bool = False


def hex_to_rgb(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    return tuple(int(value[index : index + 2], 16) for index in (0, 2, 4))


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def ease_out_cubic(value: float) -> float:
    value = clamp(value)
    return 1.0 - (1.0 - value) ** 3


def ease_in_out_sine(value: float) -> float:
    value = clamp(value)
    return -(math.cos(math.pi * value) - 1.0) / 2.0


def ease_out_back(value: float) -> float:
    value = clamp(value)
    c1 = 1.70158
    c3 = c1 + 1.0
    return 1.0 + c3 * (value - 1.0) ** 3 + c1 * (value - 1.0) ** 2


def mix_color(color_a: tuple[int, int, int], color_b: tuple[int, int, int], amount: float) -> tuple[int, int, int]:
    amount = clamp(amount)
    return tuple(int(a + (b - a) * amount) for a, b in zip(color_a, color_b))


def resolve_path(path_str: str | None, assets_dir: Path) -> Path | None:
    if not path_str:
        return None
    candidate = Path(path_str)
    if candidate.exists():
        return candidate
    local = assets_dir / candidate.name
    if local.exists():
        return local
    stripped = path_str.replace("assets/", "").replace("assets\\", "")
    local = assets_dir / stripped
    if local.exists():
        return local
    return None


def load_players(assets_dir: Path) -> list[Player]:
    players: list[Player] = []
    for raw in PLAYERS:
        players.append(
            Player(
                name=raw["name"],
                short_name=raw["short_name"],
                color=hex_to_rgb(raw["color"]),
                secondary_color=hex_to_rgb(raw["secondary_color"]),
                points=float(raw["points"]),
                assists=float(raw["assists"]),
                rebounds=float(raw["rebounds"]),
                team_win_pct=float(raw["team_win_pct"]),
                score=int(raw["score"]),
                image_path=resolve_path(str(raw.get("image") or ""), assets_dir),
            )
        )
    return players


def load_stats() -> list[StatDef]:
    return [StatDef(**item) for item in STAT_DEFS]


def load_font(size: int, bold: bool = False, custom_path: Path | None = None) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates: list[Path] = []
    if custom_path is not None:
        candidates.append(custom_path)
    candidates.extend(
        [
            Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"),
            Path("C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf"),
            Path("C:/Windows/Fonts/impact.ttf"),
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            try:
                return ImageFont.truetype(str(candidate), size=size)
            except Exception:
                continue
    return ImageFont.load_default()


def fit_font(draw: ImageDraw.ImageDraw, text: str, max_width: int, max_size: int, min_size: int, bold: bool = True) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    size = max_size
    while size >= min_size:
        font = load_font(size=size, bold=bold)
        bbox = draw.textbbox((0, 0), text, font=font)
        if bbox[2] - bbox[0] <= max_width:
            return font
        size -= 2
    return load_font(size=min_size, bold=bold)


def draw_glow_text(
    canvas: Image.Image,
    position: tuple[float, float],
    text: str,
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int] | tuple[int, int, int, int],
    anchor: str = "mm",
    glow_color: tuple[int, int, int] | None = None,
    glow_radius: int = 18,
    glow_strength: int = 180,
) -> None:
    glow_color = glow_color or tuple(fill[:3])  # type: ignore[index]
    glow = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    gd = ImageDraw.Draw(glow)
    gd.text(position, text, font=font, fill=(*glow_color, glow_strength), anchor=anchor)
    glow = glow.filter(ImageFilter.GaussianBlur(radius=glow_radius))
    canvas.alpha_composite(glow)
    draw = ImageDraw.Draw(canvas)
    draw.text(position, text, font=font, fill=fill, anchor=anchor)


def make_round_rect(size: tuple[int, int], radius: int, fill: tuple[int, int, int, int], outline: tuple[int, int, int, int] | None = None, outline_width: int = 2) -> Image.Image:
    image = Image.new("RGBA", size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(image, "RGBA")
    draw.rounded_rectangle((0, 0, size[0] - 1, size[1] - 1), radius=radius, fill=fill, outline=outline, width=outline_width)
    return image


def fit_image(image: Image.Image, size: tuple[int, int], center: tuple[float, float] = (0.5, 0.35)) -> Image.Image:
    return ImageOps.fit(image, size, method=Image.Resampling.LANCZOS, centering=center)


def circular_crop(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    fitted = fit_image(image, size)
    mask = Image.new("L", size, 0)
    ImageDraw.Draw(mask).ellipse((0, 0, size[0], size[1]), fill=255)
    result = Image.new("RGBA", size, (0, 0, 0, 0))
    result.paste(fitted.convert("RGBA"), (0, 0), mask)
    return result


def load_player_portrait(player: Player, size: tuple[int, int]) -> Image.Image:
    if player.image_path and player.image_path.exists():
        try:
            image = ImageOps.exif_transpose(Image.open(player.image_path)).convert("RGBA")
            if "A" not in image.getbands():
                image = fit_image(image.convert("RGB"), size).convert("RGBA")
            else:
                image = ImageOps.contain(image, size, method=Image.Resampling.LANCZOS)
                canvas = Image.new("RGBA", size, (0, 0, 0, 0))
                offset = ((size[0] - image.width) // 2, (size[1] - image.height) // 2)
                canvas.alpha_composite(image, offset)
                image = canvas
            image = ImageEnhance.Contrast(image).enhance(1.05)
            image = ImageEnhance.Color(image).enhance(1.06)
            return image
        except Exception:
            pass

    fallback = Image.new("RGBA", size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(fallback, "RGBA")
    for y in range(size[1]):
        blend = y / max(1, size[1] - 1)
        color = mix_color(player.secondary_color, player.color, blend)
        draw.line((0, y, size[0], y), fill=(*color, 255))
    initials = "".join(part[0] for part in player.short_name.split()[:2]).upper()
    font = load_font(size=max(42, size[0] // 4), bold=True)
    draw.text((size[0] / 2, size[1] / 2), initials, font=font, fill=(255, 255, 255, 235), anchor="mm")
    return fallback


@lru_cache(maxsize=1)
def background_base() -> np.ndarray:
    xx = np.linspace(0.0, 1.0, WIDTH, dtype=np.float32)
    yy = np.linspace(0.0, 1.0, HEIGHT, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(xx, yy)
    top = np.array(BACKGROUND_TOP, dtype=np.float32)
    bottom = np.array(BACKGROUND_BOTTOM, dtype=np.float32)
    vertical = grid_y[..., None]
    base = top * (1.0 - vertical) + bottom * vertical
    vignette = 1.0 - 0.55 * ((grid_x - 0.5) ** 2 + (grid_y - 0.52) ** 2)
    vignette = np.clip(vignette, 0.42, 1.0)[..., None]
    blue_glow = np.exp(-(((grid_x - 0.16) / 0.22) ** 2 + ((grid_y - 0.38) / 0.28) ** 2))[..., None]
    red_glow = np.exp(-(((grid_x - 0.84) / 0.22) ** 2 + ((grid_y - 0.42) / 0.30) ** 2))[..., None]
    center_glow = np.exp(-(((grid_x - 0.5) / 0.35) ** 2 + ((grid_y - 0.58) / 0.42) ** 2))[..., None]
    base += blue_glow * np.array((22, 106, 255), dtype=np.float32) * 0.35
    base += red_glow * np.array((255, 74, 52), dtype=np.float32) * 0.33
    base += center_glow * np.array((255, 199, 128), dtype=np.float32) * 0.10
    base *= vignette
    return np.clip(base, 0, 255).astype(np.uint8)


PARTICLES = [
    (random.Random(PARTICLE_SEED + i).random(), random.Random(PARTICLE_SEED + i * 3).random(), random.Random(PARTICLE_SEED + i * 7).uniform(1.0, 3.2))
    for i in range(42)
]


def render_background_frame(t: float) -> Image.Image:
    image = Image.fromarray(background_base()).convert("RGBA")
    overlay = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")

    sweep_x = int((0.18 + 0.64 * ((math.sin(t * 0.34) + 1.0) / 2.0)) * WIDTH)
    draw.ellipse((sweep_x - 180, 220, sweep_x + 180, HEIGHT - 120), fill=(255, 210, 120, 18))
    draw.ellipse((120, 260 + int(math.sin(t * 0.8) * 40), 420, 720 + int(math.sin(t * 0.8) * 40)), fill=(55, 140, 255, 20))
    draw.ellipse((WIDTH - 420, 290 + int(math.cos(t * 0.74) * 50), WIDTH - 120, 760 + int(math.cos(t * 0.74) * 50)), fill=(255, 74, 52, 20))

    for index, (px, py, radius) in enumerate(PARTICLES):
        travel = (t * (0.04 + index * 0.0018)) % 1.0
        x = int(px * WIDTH)
        y = int((py - travel * 0.22) * HEIGHT) % HEIGHT
        alpha = 30 + int(50 * (0.5 + 0.5 * math.sin(t * 2.0 + index)))
        draw.ellipse((x, y, x + radius * 2, y + radius * 2), fill=(255, 226, 170, alpha))

    overlay = overlay.filter(ImageFilter.GaussianBlur(radius=6))
    image.alpha_composite(overlay)
    return image


def value_for_stat(player: Player, stat: StatDef) -> float:
    return getattr(player, stat.key)


def format_value(stat: StatDef, value: float) -> str:
    raw = value * 100.0 if stat.key == "team_win_pct" else value
    if stat.decimals == 0:
        text = f"{int(round(raw))}"
    else:
        text = f"{raw:.1f}"
    return f"{text}{stat.suffix}"


def count_value(stat: StatDef, target: float, progress: float) -> float:
    return target * ease_out_cubic(progress)


def stat_winner(players: list[Player], stat: StatDef) -> int | None:
    values = [value_for_stat(player, stat) for player in players]
    if max(values) == min(values):
        return None
    if stat.inverse:
        return int(min(range(len(values)), key=lambda idx: values[idx]))
    return int(max(range(len(values)), key=lambda idx: values[idx]))


def make_sfx_timeline(sound_path: Path | None, hits: Iterable[float], duration: float, gain: float = 0.75):
    if sound_path is None or not sound_path.exists():
        return None, []
    try:
        base = AudioFileClip(str(sound_path))
    except Exception:
        return None, []

    clips = []
    for hit_time in hits:
        if hit_time >= duration:
            continue
        clips.append(base.with_start(hit_time).with_volume_scaled(gain))
    if not clips:
        base.close()
        return None, []
    return CompositeAudioClip(clips).with_duration(duration), [base]


def build_audio(music_path: Path | None, swoosh_path: Path | None, hit_path: Path | None, duration: float):
    clips = []
    keep_alive: list[object] = []
    if music_path is not None and music_path.exists():
        music, music_keep = build_audio_track(music_path, duration)
        try:
            music = music.with_effects([AudioFadeOut(min(FINAL_AUDIO_FADE_OUT, duration))])
        except Exception:
            pass
        clips.append(music)
        keep_alive.extend(music_keep)

    swoosh_times = [0.18, 2.25, 3.0, 3.75, 6.2, 9.2, 12.2, 15.2, 18.5, 24.2, 28.2]
    hit_times = [18.0, 23.8, 27.9]
    swoosh_clip, swoosh_keep = make_sfx_timeline(swoosh_path, swoosh_times, duration, gain=0.45)
    hit_clip, hit_keep = make_sfx_timeline(hit_path, hit_times, duration, gain=0.65)
    if swoosh_clip is not None:
        clips.append(swoosh_clip)
        keep_alive.extend(swoosh_keep)
    if hit_clip is not None:
        clips.append(hit_clip)
        keep_alive.extend(hit_keep)

    if not clips:
        return None, keep_alive
    return CompositeAudioClip(clips).with_duration(duration), keep_alive


def render_player_badge(player: Player, size: tuple[int, int], subtitle: str | None = None) -> Image.Image:
    image = Image.new("RGBA", size, (0, 0, 0, 0))
    image.alpha_composite(make_round_rect(size, radius=34, fill=(*player.secondary_color, 210), outline=(*player.color, 235), outline_width=3))
    portrait_size = int(size[1] * 0.68)
    portrait = circular_crop(load_player_portrait(player, (portrait_size, portrait_size)), (portrait_size, portrait_size))
    glow = Image.new("RGBA", size, (0, 0, 0, 0))
    gd = ImageDraw.Draw(glow, "RGBA")
    gd.ellipse((18, 18, 18 + portrait_size + 18, 18 + portrait_size + 18), fill=(*player.color, 100))
    glow = glow.filter(ImageFilter.GaussianBlur(radius=18))
    image.alpha_composite(glow)
    image.alpha_composite(portrait, (22, (size[1] - portrait_size) // 2))
    draw = ImageDraw.Draw(image, "RGBA")
    name_font = fit_font(draw, player.short_name.upper(), size[0] - portrait_size - 80, 42, 24, bold=True)
    draw.text((portrait_size + 42, size[1] * 0.42), player.short_name.upper(), font=name_font, fill=WHITE, anchor="lm")
    if subtitle:
        sub_font = load_font(18, bold=True)
        draw.text((portrait_size + 42, size[1] * 0.67), subtitle.upper(), font=sub_font, fill=(*MUTED, 255), anchor="lm")
    return image


def render_score_chip(left: int, right: int, size: tuple[int, int]) -> Image.Image:
    image = make_round_rect(size, radius=24, fill=(18, 16, 18, 225), outline=(255, 216, 124, 190), outline_width=2)
    draw = ImageDraw.Draw(image, "RGBA")
    font = load_font(42, bold=True)
    draw.text((size[0] / 2, size[1] / 2), f"{left}-{right}", font=font, fill=WHITE, anchor="mm")
    return image


def render_crown(color: tuple[int, int, int], size: tuple[int, int]) -> Image.Image:
    image = Image.new("RGBA", size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(image, "RGBA")
    points = [
        (size[0] * 0.08, size[1] * 0.78),
        (size[0] * 0.18, size[1] * 0.28),
        (size[0] * 0.40, size[1] * 0.68),
        (size[0] * 0.50, size[1] * 0.16),
        (size[0] * 0.60, size[1] * 0.68),
        (size[0] * 0.82, size[1] * 0.28),
        (size[0] * 0.92, size[1] * 0.78),
    ]
    draw.polygon(points, fill=(*color, 230), outline=(255, 240, 170, 255))
    draw.rounded_rectangle((size[0] * 0.1, size[1] * 0.72, size[0] * 0.9, size[1] * 0.9), radius=10, fill=(255, 226, 136, 210))
    return image.filter(ImageFilter.GaussianBlur(radius=0.5))


def render_top_header(canvas: Image.Image, players: list[Player], score_center: int | None = None) -> None:
    badges = [
        render_player_badge(players[0], (304, 132), subtitle="candidate"),
        render_player_badge(players[1], (304, 132), subtitle="candidate"),
        render_player_badge(players[2], (304, 132), subtitle="candidate"),
    ]
    x_positions = [SAFE_X, WIDTH // 2 - badges[1].width // 2, WIDTH - SAFE_X - badges[2].width]
    for badge, x in zip(badges, x_positions):
        canvas.alpha_composite(badge, (x, 90))
    if score_center is not None:
        chip = render_score_chip(score_center, 100, (160, 76))
        canvas.alpha_composite(chip, (WIDTH // 2 - chip.width // 2, 236))


def render_hook(canvas: Image.Image, t_local: float) -> None:
    progress = ease_out_back(min(1.0, t_local / 1.1))
    scale = 0.74 + 0.26 * progress
    title_font = load_font(int(104 * scale), bold=True)
    sub_font = load_font(34, bold=True)
    y = 420 + int((1.0 - progress) * 110)
    draw_glow_text(canvas, (WIDTH / 2, y), "IF MVP WAS", title_font, fill=WHITE, glow_color=GOLD, glow_radius=26, glow_strength=165)
    draw_glow_text(canvas, (WIDTH / 2, y + 118), "DECIDED TODAY...", title_font, fill=GOLD, glow_color=GOLD, glow_radius=28, glow_strength=190)
    draw_glow_text(canvas, (WIDTH / 2, y + 228), "NBA REGULAR SEASON RACE", sub_font, fill=MUTED, glow_color=(70, 130, 255), glow_radius=14, glow_strength=120)
    line_width = int((WIDTH - 2 * SAFE_X - 140) * ease_in_out_sine(min(1.0, t_local / 0.9)))
    line = Image.new("RGBA", (line_width, 6), (255, 220, 140, 240))
    line = line.filter(ImageFilter.GaussianBlur(radius=2))
    canvas.alpha_composite(line, (WIDTH // 2 - line_width // 2, y + 286))


def render_intro(canvas: Image.Image, players: list[Player], t_local: float) -> None:
    title_font = load_font(48, bold=True)
    sub_font = load_font(22, bold=True)
    draw_glow_text(canvas, (WIDTH / 2, 356), "TOP 3 RIGHT NOW", title_font, fill=WHITE, glow_color=GOLD, glow_radius=18)
    draw_glow_text(canvas, (WIDTH / 2, 406), "ONE RACE. THREE SUPERSTARS.", sub_font, fill=MUTED, glow_color=(60, 120, 255), glow_radius=10)

    card_w, card_h = 922, 256
    base_y = 520
    for idx, player in enumerate(players):
        delay = idx * 0.42
        progress = ease_out_back(clamp((t_local - delay) / 0.8))
        if progress <= 0:
            continue
        card = make_round_rect((card_w, card_h), radius=38, fill=(*player.secondary_color, 190), outline=(*player.color, 235), outline_width=3)
        portrait = load_player_portrait(player, (248, 248))
        portrait = fit_image(portrait, (248, 248))
        portrait = ImageOps.expand(portrait.convert("RGBA"), border=4, fill=(*player.color, 255))
        card.alpha_composite(portrait, (26, 4))
        draw = ImageDraw.Draw(card, "RGBA")
        name_font = fit_font(draw, player.name.upper(), 500, 42, 24, bold=True)
        draw.text((310, 74), player.name.upper(), font=name_font, fill=WHITE, anchor="lm")
        score_font = load_font(48, bold=True)
        draw.text((310, 134), f"MVP SCORE {player.score}", font=score_font, fill=(*player.color, 255), anchor="lm")
        pill = make_round_rect((180, 48), radius=24, fill=(*player.color, 220))
        pd = ImageDraw.Draw(pill, "RGBA")
        pd.text((90, 24), "HOT STREAK", font=load_font(20, bold=True), fill=BLACK, anchor="mm")
        card.alpha_composite(pill, (310, 172))
        x = int(SAFE_X + (1.0 - progress) * 180)
        y = int(base_y + idx * 292 - (1.0 - progress) * 40 + math.sin(t_local * 1.8 + idx) * 8)
        alpha_overlay = Image.new("RGBA", (card_w, card_h), (255, 255, 255, int(42 * (1.0 - progress))))
        card.alpha_composite(alpha_overlay)
        canvas.alpha_composite(card, (x, y))


def render_stat_scene(canvas: Image.Image, players: list[Player], stats: list[StatDef], t_local: float) -> None:
    stat_length = STATS_DURATION / len(stats)
    stat_index = min(len(stats) - 1, int(t_local / stat_length))
    stat = stats[stat_index]
    in_stat = t_local - stat_index * stat_length
    stat_progress = clamp(in_stat / stat_length)
    render_top_header(canvas, players)
    draw_glow_text(canvas, (WIDTH / 2, 382), stat.label, load_font(52, bold=True), fill=WHITE, glow_color=GOLD, glow_radius=18)
    draw_glow_text(canvas, (WIDTH / 2, 430), "LIVE COMPARISON", load_font(22, bold=True), fill=MUTED, glow_color=(88, 130, 255), glow_radius=10)

    winner = stat_winner(players, stat)
    values = [value_for_stat(player, stat) for player in players]
    max_value = max(values) if max(values) > 0 else 1.0
    bar_left = SAFE_X + 170
    bar_right = WIDTH - SAFE_X - 90
    track_width = bar_right - bar_left
    row_y = [640, 866, 1092]

    for idx, (player, y, value) in enumerate(zip(players, row_y, values)):
        row_progress = clamp((in_stat - 0.12 - idx * 0.15) / 1.05)
        slide = ease_out_cubic(row_progress)
        if slide <= 0:
            continue
        panel = make_round_rect((WIDTH - 2 * SAFE_X, 164), radius=34, fill=(*player.secondary_color, 164), outline=(*player.color, 150), outline_width=2)
        panel_draw = ImageDraw.Draw(panel, "RGBA")
        panel_draw.rounded_rectangle((170, 84, panel.width - 36, 120), radius=18, fill=(255, 255, 255, 20))
        fill_width = int(track_width * (value / max_value) * slide)
        fill_width = max(10, fill_width) if row_progress > 0 else 0
        if fill_width > 0:
            bar = Image.new("RGBA", (fill_width, 36), (0, 0, 0, 0))
            bd = ImageDraw.Draw(bar, "RGBA")
            bd.rounded_rectangle((0, 0, fill_width - 1, 35), radius=18, fill=(*player.color, 242))
            sweep = Image.new("RGBA", (fill_width, 36), (0, 0, 0, 0))
            sw = ImageDraw.Draw(sweep, "RGBA")
            sweep_x = int((0.15 + 0.70 * stat_progress) * max(28, fill_width))
            sw.rounded_rectangle((max(0, sweep_x - 90), 0, min(fill_width, sweep_x + 18), 35), radius=18, fill=(255, 255, 255, 64))
            bar.alpha_composite(sweep.filter(ImageFilter.GaussianBlur(radius=6)))
            panel.alpha_composite(bar, (170, 84))

        portrait = circular_crop(load_player_portrait(player, (110, 110)), (110, 110))
        panel.alpha_composite(portrait, (22, 27))
        name_font = fit_font(panel_draw, player.short_name.upper(), 350, 34, 20, bold=True)
        panel_draw.text((150, 56), player.short_name.upper(), font=name_font, fill=WHITE, anchor="lm")
        number = count_value(stat, value, row_progress)
        panel_draw.text((panel.width - 42, 58), format_value(stat, number), font=load_font(48, bold=True), fill=WHITE, anchor="rm")
        if winner == idx and row_progress > 0.75:
            glow = Image.new("RGBA", panel.size, (0, 0, 0, 0))
            gd = ImageDraw.Draw(glow, "RGBA")
            gd.rounded_rectangle((8, 8, panel.width - 8, panel.height - 8), radius=30, outline=(*player.color, 120), width=4)
            glow = glow.filter(ImageFilter.GaussianBlur(radius=10))
            panel.alpha_composite(glow)
            panel_draw.text((150, 110), "LEADING", font=load_font(22, bold=True), fill=(*player.color, 255), anchor="lm")
        else:
            panel_draw.text((150, 110), f"{player.score} MVP SCORE", font=load_font(22, bold=True), fill=(*MUTED, 255), anchor="lm")

        x = SAFE_X + int((1.0 - slide) * 46)
        canvas.alpha_composite(panel, (x, y))


def render_score_scene(canvas: Image.Image, players: list[Player], t_local: float) -> None:
    render_top_header(canvas, players)
    draw_glow_text(canvas, (WIDTH / 2, 372), "FINAL MVP SCORE", load_font(56, bold=True), fill=WHITE, glow_color=GOLD, glow_radius=20)
    draw_glow_text(canvas, (WIDTH / 2, 424), "OUT OF 100", load_font(24, bold=True), fill=MUTED, glow_color=(80, 120, 255), glow_radius=8)

    sorted_players = sorted(players, key=lambda player: player.score, reverse=True)
    leader = sorted_players[0]
    xs = [190, 440, 690]
    base_y = 1360
    max_height = 580
    for idx, player in enumerate(sorted_players):
        value_progress = clamp((t_local - idx * 0.22) / 1.6)
        bar_progress = ease_out_back(value_progress)
        score_height = int((player.score / 100.0) * max_height * bar_progress)
        lane = make_round_rect((200, max_height + 140), radius=40, fill=(*player.secondary_color, 140), outline=(*player.color, 145), outline_width=2)
        ld = ImageDraw.Draw(lane, "RGBA")
        ld.rounded_rectangle((72, 70, 128, max_height + 60), radius=28, fill=(255, 255, 255, 18))
        bar_top = max_height + 60 - score_height
        ld.rounded_rectangle((72, bar_top, 128, max_height + 60), radius=28, fill=(*player.color, 240))
        current_score = int(round(player.score * ease_out_cubic(value_progress)))
        ld.text((100, max_height + 96), str(current_score), font=load_font(46, bold=True), fill=WHITE, anchor="mm")
        ld.text((100, max_height + 126), player.short_name.upper(), font=load_font(20, bold=True), fill=MUTED, anchor="mm")
        portrait = circular_crop(load_player_portrait(player, (120, 120)), (120, 120))
        lane.alpha_composite(portrait, (40, 0))
        if player == leader and value_progress > 0.55:
            crown = render_crown(GOLD, (82, 58))
            pulse = 1.0 + 0.06 * math.sin(t_local * 6.0)
            crown = crown.resize((int(crown.width * pulse), int(crown.height * pulse)), Image.Resampling.LANCZOS)
            lane.alpha_composite(crown, ((lane.width - crown.width) // 2, max(0, bar_top - crown.height - 18)))
        canvas.alpha_composite(lane, (xs[idx], base_y - lane.height))


def render_podium_scene(canvas: Image.Image, players: list[Player], t_local: float) -> None:
    ordered = sorted(players, key=lambda player: player.score, reverse=True)
    draw_glow_text(canvas, (WIDTH / 2, 360), "THE PODIUM RIGHT NOW", load_font(56, bold=True), fill=WHITE, glow_color=GOLD, glow_radius=20)
    podium_info = [
        (ordered[1], 126, 1160, 240, SILVER, "#2"),
        (ordered[0], 410, 1040, 320, GOLD, "#1"),
        (ordered[2], 760, 1230, 200, BRONZE, "#3"),
    ]
    for idx, (player, x, top, height, medal_color, tag) in enumerate(podium_info):
        progress = ease_out_back(clamp((t_local - idx * 0.18) / 0.9))
        if progress <= 0:
            continue
        y = int(top + (1.0 - progress) * 120)
        block = make_round_rect((194, height), radius=34, fill=(*player.secondary_color, 214), outline=(*medal_color, 220), outline_width=3)
        portrait = circular_crop(load_player_portrait(player, (132, 132)), (132, 132))
        block.alpha_composite(portrait, (31, 24))
        bd = ImageDraw.Draw(block, "RGBA")
        bd.text((97, 178), tag, font=load_font(46, bold=True), fill=medal_color, anchor="mm")
        bd.text((97, 220), player.short_name.upper(), font=load_font(24, bold=True), fill=WHITE, anchor="mm")
        bd.text((97, 252), f"{player.score}", font=load_font(52, bold=True), fill=(*player.color, 255), anchor="mm")
        canvas.alpha_composite(block, (x, y))


def render_cta(canvas: Image.Image, t_local: float) -> None:
    progress = ease_out_back(clamp(t_local / 0.75))
    scale = 0.9 + 0.1 * progress
    draw_glow_text(canvas, (WIDTH / 2, 920), "DO YOU AGREE?", load_font(int(98 * scale), bold=True), fill=WHITE, glow_color=GOLD, glow_radius=28)
    draw_glow_text(canvas, (WIDTH / 2, 1040), "COMMENT BELOW", load_font(44, bold=True), fill=GOLD, glow_color=(255, 135, 86), glow_radius=16)
    line_width = int((WIDTH - 220) * ease_in_out_sine(clamp(t_local / 0.85)))
    line = Image.new("RGBA", (line_width, 8), (255, 215, 114, 235))
    line = line.filter(ImageFilter.GaussianBlur(radius=2))
    canvas.alpha_composite(line, (WIDTH // 2 - line_width // 2, 1120))


def render_frame(t: float, players: list[Player], stats: list[StatDef]) -> np.ndarray:
    canvas = render_background_frame(t)
    if t < HOOK_DURATION:
        render_hook(canvas, t)
    elif t < HOOK_DURATION + INTRO_DURATION:
        render_intro(canvas, players, t - HOOK_DURATION)
    elif t < HOOK_DURATION + INTRO_DURATION + STATS_DURATION:
        render_stat_scene(canvas, players, stats, t - HOOK_DURATION - INTRO_DURATION)
    elif t < HOOK_DURATION + INTRO_DURATION + STATS_DURATION + SCORE_DURATION:
        render_score_scene(canvas, players, t - HOOK_DURATION - INTRO_DURATION - STATS_DURATION)
    elif t < HOOK_DURATION + INTRO_DURATION + STATS_DURATION + SCORE_DURATION + PODIUM_DURATION:
        render_podium_scene(canvas, players, t - HOOK_DURATION - INTRO_DURATION - STATS_DURATION - SCORE_DURATION)
    else:
        render_cta(canvas, t - HOOK_DURATION - INTRO_DURATION - STATS_DURATION - SCORE_DURATION - PODIUM_DURATION)

    vignette = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    vd = ImageDraw.Draw(vignette, "RGBA")
    vd.rectangle((0, 0, WIDTH, HEIGHT), fill=(0, 0, 0, 24))
    vignette = vignette.filter(ImageFilter.GaussianBlur(radius=28))
    canvas.alpha_composite(vignette)
    return np.array(canvas.convert("RGB"))


def ensure_duration(total_duration: float) -> None:
    expected = HOOK_DURATION + INTRO_DURATION + STATS_DURATION + SCORE_DURATION + PODIUM_DURATION + CTA_DURATION
    if abs(total_duration - expected) > 0.01:
        raise ValueError(f"Configured duration mismatch: expected {expected:.2f}s from phase constants, got {total_duration:.2f}s")


def render_video(
    output_path: Path,
    assets_dir: Path,
    duration: float,
    fps: int,
    music_path: Path | None,
    swoosh_path: Path | None,
    hit_path: Path | None,
) -> None:
    ensure_duration(duration)
    players = load_players(assets_dir)
    stats = load_stats()

    clip = VideoClip(lambda t: render_frame(t, players, stats), duration=duration)
    audio_clip, keep_alive = build_audio(music_path, swoosh_path, hit_path, duration)
    if audio_clip is not None:
        clip = clip.with_audio(audio_clip)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    clip.write_videofile(str(output_path), fps=fps, codec="libx264", audio_codec="aac" if audio_clip else None)

    clip.close()
    if audio_clip is not None:
        audio_clip.close()
    for item in keep_alive:
        try:
            item.close()
        except Exception:
            pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a premium vertical MVP race sports Shorts template.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--assets-dir", type=Path, default=DEFAULT_ASSETS_DIR)
    parser.add_argument("--music", type=Path, default=DEFAULT_MUSIC)
    parser.add_argument("--swoosh-sfx", type=Path, default=None)
    parser.add_argument("--hit-sfx", type=Path, default=None)
    parser.add_argument("--fps", type=int, default=FPS)
    parser.add_argument("--duration", type=float, default=DURATION)
    args = parser.parse_args()

    render_video(
        output_path=args.output,
        assets_dir=args.assets_dir,
        duration=args.duration,
        fps=args.fps,
        music_path=args.music,
        swoosh_path=args.swoosh_sfx,
        hit_path=args.hit_sfx,
    )
    print(f"[video_generator] MVP race Shorts generated -> {args.output}")


if __name__ == "__main__":
    main()
