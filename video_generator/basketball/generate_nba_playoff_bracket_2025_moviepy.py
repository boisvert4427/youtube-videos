from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from moviepy import VideoClip
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps

from video_generator.basketball.generate_nba_titles_shorts_moviepy import TEAM_COLORS
from video_generator.generate_ucl_barchart_race_moviepy import DEFAULT_AUDIO, _fit_font_size, _load_font, build_audio_track


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "basketball" / "nba_playoff_bracket_2025_style.mp4"
LOGO_DIR = PROJECT_ROOT / "data" / "raw" / "nba_team_logos"
TROPHY_PHOTO = PROJECT_ROOT / "data" / "raw" / "nba_trophy_photo.jpg"

WIDTH = 1080
HEIGHT = 1920
FPS = 30
TOTAL_DURATION = 42.0

ROUND1_DURATION = 15.0
SEMI_DURATION = 10.0
CONF_DURATION = 8.0
FINAL_DURATION = 9.0

TITLE = "PLAYOFFS 2025"
SUBTITLE = "Road to the title"
LEFT_LABEL = "EASTERN CONFERENCE"
RIGHT_LABEL = "WESTERN CONFERENCE"
ROUND1_LABEL = "FIRST ROUND"
SEMI_LABEL = "CONFERENCE SEMIFINALS"
CONF_LABEL = "CONFERENCE FINALS"
FINAL_LABEL = "NBA FINALS"
CHAMPION_LABEL = "CHAMPION"

CARD_W = 116
CARD_H = 92
CARD_RADIUS = 14

LEFT_EDGE = 52
RIGHT_EDGE = WIDTH - 52
LEFT_CENTER = 120
RIGHT_CENTER = WIDTH - 120

LEFT_COLS = {
    "seed": 86,
    "round1": 214,
    "semi": 304,
    "conf": 442,
}
RIGHT_COLS = {
    "seed": 994,
    "round1": 866,
    "semi": 776,
    "conf": 638,
}

CENTER_FINAL_BOX = (358, 926, 722, 1066)
CHAMPION_BOX = (326, 1084, 754, 1228)
TROPHY_BOX = (422, 704, 658, 998)

TEAM_LOGO_FILES = {
    "Boston Celtics": "boston_celtics.png",
    "Cleveland Cavaliers": "cleveland_cavaliers.png",
    "Dallas Mavericks": "dallas_mavericks.png",
    "Denver Nuggets": "denver_nuggets.png",
    "Detroit Pistons": "detroit_pistons.png",
    "Golden State Warriors": "golden_state_warriors.png",
    "Houston Rockets": "houston_rockets.png",
    "Indiana Pacers": "indiana_pacers.png",
    "Los Angeles Lakers": "los_angeles_lakers.png",
    "Memphis Grizzlies": "memphis_grizzlies.png",
    "Miami Heat": "miami_heat.png",
    "Milwaukee Bucks": "milwaukee_bucks.png",
    "Minnesota Timberwolves": "minnesota_timberwolves.png",
    "New York Knicks": "new_york_knicks.png",
    "Oklahoma City Thunder": "oklahoma_city_thunder.png",
    "Orlando Magic": "orlando_magic.png",
    "Phoenix Suns": "phoenix_suns.png",
    "Philadelphia 76ers": "philadelphia_76ers.png",
    "LA Clippers": "la_clippers.png",
}

TEAM_ABBRS = {
    "Boston Celtics": "BOS",
    "Cleveland Cavaliers": "CLE",
    "Dallas Mavericks": "DAL",
    "Denver Nuggets": "DEN",
    "Detroit Pistons": "DET",
    "Golden State Warriors": "GSW",
    "Houston Rockets": "HOU",
    "Indiana Pacers": "IND",
    "Los Angeles Lakers": "LAL",
    "Memphis Grizzlies": "MEM",
    "Miami Heat": "MIA",
    "Milwaukee Bucks": "MIL",
    "Minnesota Timberwolves": "MIN",
    "New York Knicks": "NYK",
    "Oklahoma City Thunder": "OKC",
    "Orlando Magic": "ORL",
    "Philadelphia 76ers": "PHI",
    "Phoenix Suns": "PHX",
    "LA Clippers": "LAC",
}

EAST_TEAMS = [
    ("Cleveland Cavaliers", 1),
    ("Miami Heat", 8),
    ("Boston Celtics", 2),
    ("Orlando Magic", 7),
    ("New York Knicks", 3),
    ("Detroit Pistons", 6),
    ("Indiana Pacers", 4),
    ("Milwaukee Bucks", 5),
]

WEST_TEAMS = [
    ("Oklahoma City Thunder", 1),
    ("Memphis Grizzlies", 8),
    ("Houston Rockets", 2),
    ("Golden State Warriors", 7),
    ("Los Angeles Lakers", 3),
    ("Minnesota Timberwolves", 6),
    ("Denver Nuggets", 4),
    ("LA Clippers", 5),
]

ROUND1_WINNERS = [
    "Cleveland Cavaliers",
    "Indiana Pacers",
    "New York Knicks",
    "Boston Celtics",
    "Oklahoma City Thunder",
    "Golden State Warriors",
    "Minnesota Timberwolves",
    "Denver Nuggets",
]

SEMI_WINNERS = [
    "Indiana Pacers",
    "New York Knicks",
    "Oklahoma City Thunder",
    "Minnesota Timberwolves",
]

CONF_WINNERS = [
    "Indiana Pacers",
    "Oklahoma City Thunder",
]

CHAMPION = "Oklahoma City Thunder"

SERIES_SCORES = {
    ("round1", "Cleveland Cavaliers"): "4-0",
    ("round1", "Indiana Pacers"): "4-1",
    ("round1", "New York Knicks"): "4-2",
    ("round1", "Boston Celtics"): "4-1",
    ("round1", "Oklahoma City Thunder"): "4-0",
    ("round1", "Golden State Warriors"): "4-3",
    ("round1", "Minnesota Timberwolves"): "4-1",
    ("round1", "Denver Nuggets"): "4-3",
    ("semi", "Indiana Pacers"): "4-1",
    ("semi", "New York Knicks"): "4-2",
    ("semi", "Oklahoma City Thunder"): "4-3",
    ("semi", "Minnesota Timberwolves"): "4-1",
    ("conf", "Indiana Pacers"): "4-2",
    ("conf", "Oklahoma City Thunder"): "4-1",
    ("final", "Oklahoma City Thunder"): "4-3",
}


@dataclass(frozen=True)
class Move:
    team_name: str
    start: tuple[int, int]
    end: tuple[int, int]
    delay: float
    duration: float
    score: str


def _hex_to_rgb(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    return tuple(int(value[i : i + 2], 16) for i in (0, 2, 4))


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return min(max(value, lo), hi)


def _ease_in_out(value: float) -> float:
    value = _clamp(value)
    return value * value * (3.0 - 2.0 * value)


def _ease_out(value: float) -> float:
    value = _clamp(value)
    return 1.0 - (1.0 - value) ** 3


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _manhattan_point(
    start: tuple[int, int],
    end: tuple[int, int],
    t: float,
    first: str = "horizontal",
) -> tuple[float, float]:
    t = _clamp(t)
    split = 0.5
    if first == "vertical":
        if t <= split:
            local = _ease_in_out(t / split)
            return (start[0], _lerp(start[1], end[1], local))
        local = _ease_in_out((t - split) / split)
        return (_lerp(start[0], end[0], local), end[1])
    if t <= split:
        local = _ease_in_out(t / split)
        return (_lerp(start[0], end[0], local), start[1])
    local = _ease_in_out((t - split) / split)
    return (end[0], _lerp(start[1], end[1], local))


def _text_on(color: str) -> str:
    r, g, b = _hex_to_rgb(color)
    luminance = (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255.0
    return "#101010" if luminance > 0.66 else "#f5f8fd"


def _load_team_logo(team_name: str) -> Image.Image | None:
    logo_name = TEAM_LOGO_FILES.get(team_name)
    if logo_name is None:
        return None
    path = LOGO_DIR / logo_name
    if not path.exists():
        return None
    return ImageOps.contain(Image.open(path).convert("RGBA"), (72, 72), method=Image.Resampling.LANCZOS)


def _make_logo_badge(team_name: str) -> Image.Image:
    primary, secondary = TEAM_COLORS.get(team_name, ("#c5cbd7", "#111111"))
    primary_rgb = _hex_to_rgb(primary)
    secondary_rgb = _hex_to_rgb(secondary)
    badge = Image.new("RGBA", (CARD_W, CARD_H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(badge, "RGBA")
    draw.rounded_rectangle((0, 0, CARD_W - 1, CARD_H - 1), radius=CARD_RADIUS, fill=(14, 14, 18, 245), outline=(255, 255, 255, 38), width=2)
    draw.rounded_rectangle((0, 0, 28, CARD_H - 1), radius=CARD_RADIUS, fill=(*primary_rgb, 235))
    draw.rectangle((28, 0, CARD_W - 1, 8), fill=(*secondary_rgb, 130))
    draw.rectangle((28, CARD_H - 8, CARD_W - 1, CARD_H - 1), fill=(*primary_rgb, 105))
    draw.ellipse((35, 17, 89, 71), outline=(255, 255, 255, 42), width=2)
    abbr = TEAM_ABBRS.get(team_name, "".join(part[0] for part in team_name.split()[:3]).upper())
    font = _fit_font_size(draw, abbr, 34, 24, 12, bold=True)
    draw.text((62, CARD_H // 2 - 1), abbr, font=font, fill="#f5f8fd", anchor="mm")
    return badge


def _make_team_card(team_name: str, seed: int) -> Image.Image:
    primary, secondary = TEAM_COLORS.get(team_name, ("#c5cbd7", "#111111"))
    primary_rgb = _hex_to_rgb(primary)
    secondary_rgb = _hex_to_rgb(secondary)
    card = Image.new("RGBA", (CARD_W, CARD_H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(card, "RGBA")
    draw.rounded_rectangle((1, 1, CARD_W - 2, CARD_H - 2), radius=CARD_RADIUS, fill=(10, 12, 16, 212), outline=(255, 255, 255, 45), width=2)
    draw.rounded_rectangle((0, 0, 28, CARD_H - 1), radius=CARD_RADIUS, fill=(248, 248, 248, 255))
    draw.rectangle((28, 0, CARD_W - 1, 7), fill=(*primary_rgb, 255))
    draw.rectangle((28, CARD_H - 8, CARD_W - 1, CARD_H - 1), fill=(*secondary_rgb, 145))
    draw.ellipse((30, 9, 91, 83), outline=(255, 255, 255, 18), width=2)
    seed_font = _load_font(28, bold=True)
    draw.text((14, CARD_H // 2), str(seed), font=seed_font, fill="#101010", anchor="mm")
    logo = _load_team_logo(team_name)
    if logo is None:
        logo = _make_logo_badge(team_name)
    logo = ImageOps.contain(logo, (72, 72), method=Image.Resampling.LANCZOS)
    card.alpha_composite(logo, ((CARD_W - logo.width) // 2 + 8, (CARD_H - logo.height) // 2))
    return card


def _make_background() -> Image.Image:
    xx = np.linspace(0, 1, WIDTH, dtype=np.float32)
    yy = np.linspace(0, 1, HEIGHT, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(xx, yy)

    black = np.array([0, 0, 0], dtype=np.float32)
    navy = np.array([5, 18, 45], dtype=np.float32)
    blue = np.array([31, 102, 234], dtype=np.float32)
    red = np.array([196, 24, 36], dtype=np.float32)
    gold = np.array([239, 180, 71], dtype=np.float32)

    split = np.clip(0.5 - (grid_x - 0.5) * 1.3, 0, 1)
    left_glow = np.exp(-(((grid_x - 0.11) / 0.19) ** 2 + ((grid_y - 0.37) / 0.22) ** 2))
    right_glow = np.exp(-(((grid_x - 0.89) / 0.19) ** 2 + ((grid_y - 0.38) / 0.22) ** 2))
    top_haze = np.exp(-(((grid_x - 0.5) / 0.35) ** 2 + ((grid_y - 0.08) / 0.12) ** 2))
    center_hole = np.exp(-(((grid_x - 0.5) / 0.18) ** 2 + ((grid_y - 0.46) / 0.26) ** 2))
    floor = np.clip((grid_y - 0.76) / 0.24, 0, 1)
    floor_lines = (np.sin(grid_x * 120.0) ** 2) * floor * 0.10
    spark = (((np.sin(grid_x * 78.0) + 1.0) * (np.sin(grid_y * 60.0) + 1.0)) * 0.008)

    img = np.clip(
        black[None, None, :] * (1.0 - 0.68 * floor[..., None])
        + navy[None, None, :] * (0.80 * (1.0 - split[..., None]) + 0.20 * center_hole[..., None])
        + blue[None, None, :] * (0.58 * (1.0 - split[..., None]) * left_glow[..., None] + 0.10 * top_haze[..., None])
        + red[None, None, :] * (0.58 * split[..., None] * right_glow[..., None] + 0.10 * top_haze[..., None])
        + gold[None, None, :] * (0.12 * top_haze[..., None] + 0.07 * center_hole[..., None])
        + np.array([255, 255, 255], dtype=np.float32)[None, None, :] * (spark[..., None] + floor_lines[..., None]),
        0,
        255,
    ).astype(np.uint8)
    frame = Image.fromarray(img, mode="RGB").convert("RGBA")

    overlay = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")
    draw.rounded_rectangle((22, 18, WIDTH - 22, HEIGHT - 18), radius=38, outline=(255, 255, 255, 12), width=2)
    draw.line((0, 1760, WIDTH, 1760), fill=(255, 255, 255, 10), width=2)
    draw.line((0, 1820, WIDTH, 1820), fill=(255, 180, 80, 16), width=2)
    draw.ellipse((-80, 100, 220, 360), fill=(40, 145, 255, 34))
    draw.ellipse((860, 110, 1160, 360), fill=(255, 52, 68, 34))
    overlay = overlay.filter(ImageFilter.GaussianBlur(radius=10))
    frame.alpha_composite(overlay)
    return frame


def _draw_glow_text(
    frame: Image.Image,
    position: tuple[int, int],
    text: str,
    font: ImageFont.ImageFont,
    fill: str,
    glow: tuple[int, int, int],
    anchor: str = "la",
) -> None:
    glow_layer = Image.new("RGBA", frame.size, (0, 0, 0, 0))
    glow_draw = ImageDraw.Draw(glow_layer, "RGBA")
    glow_draw.text(position, text, font=font, fill=(*glow, 140), anchor=anchor)
    glow_layer = glow_layer.filter(ImageFilter.GaussianBlur(radius=9))
    frame.alpha_composite(glow_layer)
    draw = ImageDraw.Draw(frame, "RGBA")
    draw.text(position, text, font=font, fill=fill, anchor=anchor)


def _draw_card(
    frame: Image.Image,
    card: Image.Image,
    x: float,
    y: float,
    alpha: float,
    scale: float = 1.0,
    glow: bool = False,
    glow_color: tuple[int, int, int] = (255, 218, 118),
) -> None:
    alpha = _clamp(alpha)
    scale = max(0.88, scale)
    image = card
    if scale != 1.0:
        image = image.resize((max(1, int(image.width * scale)), max(1, int(image.height * scale))), Image.Resampling.LANCZOS)
    if alpha < 0.999:
        image = image.copy()
        channel = image.getchannel("A").point(lambda a: int(a * alpha))
        image.putalpha(channel)
    px = int(x - image.width // 2)
    py = int(y - image.height // 2)
    if glow:
        glow_layer = Image.new("RGBA", (image.width + 72, image.height + 72), (0, 0, 0, 0))
        gd = ImageDraw.Draw(glow_layer, "RGBA")
        gd.rounded_rectangle((18, 18, glow_layer.width - 18, glow_layer.height - 18), radius=22, fill=(*glow_color, 100))
        glow_layer = glow_layer.filter(ImageFilter.GaussianBlur(radius=18))
        frame.alpha_composite(glow_layer, (px - 36, py - 36))
    frame.alpha_composite(image, (px, py))


def _draw_score_badge(frame: Image.Image, text: str, x: float, y: float, highlight: bool = False) -> None:
    badge = Image.new("RGBA", (100, 34), (0, 0, 0, 0))
    draw = ImageDraw.Draw(badge, "RGBA")
    fill = (255, 200, 90, 235) if highlight else (16, 18, 22, 228)
    outline = (255, 224, 128, 210) if highlight else (255, 255, 255, 42)
    draw.rounded_rectangle((0, 0, 99, 33), radius=14, fill=fill, outline=outline, width=2)
    font = _load_font(18, bold=True)
    draw.text((50, 17), text, font=font, fill="#101010" if highlight else "#f5f8fd", anchor="mm")
    frame.alpha_composite(badge, (int(x - 50), int(y - 17)))


def _draw_path(frame: Image.Image, start: tuple[int, int], end: tuple[float, float], color: tuple[int, int, int], width: int = 7) -> None:
    draw = ImageDraw.Draw(frame, "RGBA")
    draw.line((start[0], start[1], int(end[0]), start[1]), fill=(*color, 205), width=width)
    draw.line((int(end[0]), start[1], int(end[0]), int(end[1])), fill=(*color, 205), width=width)


def _draw_bracket_pair(
    draw: ImageDraw.ImageDraw,
    top_y: int,
    bottom_y: int,
    mid_y: int,
    left_x: int,
    stem_x: int,
    right_x: int,
    color: tuple[int, int, int, int],
    width: int,
    cap: int = 16,
) -> None:
    draw.line((left_x, top_y, stem_x, top_y), fill=color, width=width)
    draw.line((left_x, bottom_y, stem_x, bottom_y), fill=color, width=width)
    draw.line((stem_x, top_y, stem_x, bottom_y), fill=color, width=width)
    draw.line((stem_x, mid_y, right_x, mid_y), fill=color, width=width)
    draw.line((stem_x - cap, top_y, stem_x + cap, top_y), fill=color, width=width)
    draw.line((stem_x - cap, bottom_y, stem_x + cap, bottom_y), fill=color, width=width)
    draw.line((stem_x - cap, mid_y, stem_x + cap, mid_y), fill=color, width=width)


def _draw_trophy(frame: Image.Image, scale: float = 1.0) -> None:
    if TROPHY_PHOTO.exists():
        photo = Image.open(TROPHY_PHOTO).convert("RGBA")
        photo = ImageOps.contain(photo, (250, 360), method=Image.Resampling.LANCZOS)
        halo = Image.new("RGBA", (photo.width + 54, photo.height + 54), (0, 0, 0, 0))
        halo_draw = ImageDraw.Draw(halo, "RGBA")
        halo_draw.rounded_rectangle((6, 6, halo.width - 7, halo.height - 7), radius=26, fill=(0, 0, 0, 90))
        halo = halo.filter(ImageFilter.GaussianBlur(radius=14))
        frame.alpha_composite(halo, ((WIDTH - halo.width) // 2, 714))
        frame.alpha_composite(photo, ((WIDTH - photo.width) // 2, 714))
        return

    w = int(210 * scale)
    h = int(300 * scale)
    trophy = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(trophy, "RGBA")
    gold = (248, 188, 67, 235)
    bright = (255, 230, 146, 255)
    dark = (122, 80, 20, 255)
    shadow = (35, 24, 10, 180)
    draw.ellipse((52, 12, w - 52, 86), fill=gold, outline=bright, width=3)
    draw.rounded_rectangle((36, 86, w - 36, 182), radius=28, fill=(236, 177, 56, 225), outline=bright, width=3)
    draw.ellipse((14, 56, 64, 146), fill=gold, outline=bright, width=3)
    draw.ellipse((w - 64, 56, w - 14, 146), fill=gold, outline=bright, width=3)
    draw.rectangle((w // 2 - 15, 162, w // 2 + 15, 224), fill=dark)
    draw.rounded_rectangle((w // 2 - 84, 222, w // 2 + 84, 266), radius=12, fill=shadow)
    draw.rounded_rectangle((w // 2 - 104, 266, w // 2 + 104, 300), radius=14, fill=dark, outline=bright, width=2)
    draw.line((w // 2, 18, w // 2, 74), fill=(255, 255, 255, 100), width=4)
    trophy = trophy.filter(ImageFilter.GaussianBlur(radius=0.2))
    frame.alpha_composite(trophy, ((WIDTH - w) // 2, 706))


def _draw_header(draw: ImageDraw.ImageDraw, title_font, subtitle_font, label_font) -> None:
    draw.text((WIDTH // 2 + 110, 72), TITLE, font=title_font, fill="#f6f8fb", anchor="ma", stroke_width=10, stroke_fill=(0, 0, 0, 220))
    draw.text((WIDTH // 2 + 110, 132), SUBTITLE, font=subtitle_font, fill="#d9e7ff", anchor="ma")
    draw.rounded_rectangle((166, 44, 238, 188), radius=16, fill=(20, 70, 180, 250), outline=(255, 255, 255, 180), width=3)
    draw.rounded_rectangle((186, 52, 218, 178), radius=12, fill=(255, 255, 255, 245))
    draw.polygon([(188, 56), (211, 56), (211, 174), (188, 174)], fill=(255, 255, 255, 245))
    draw.polygon([(188, 56), (208, 56), (208, 174), (188, 174)], fill=(220, 37, 37, 245))
    draw.text((202, 158), "NBA", font=_load_font(22, bold=True), fill="#f6f8fb", anchor="mm")
    draw.text((92, 146), LEFT_LABEL, font=label_font, fill="#f6f8fb")
    draw.text((WIDTH - 92, 146), RIGHT_LABEL, font=label_font, fill="#f6f8fb", anchor="ra")


def _draw_scaffold(draw: ImageDraw.ImageDraw) -> None:
    faint = (255, 255, 255, 18)
    white = (255, 255, 255, 88)
    white_dim = (255, 255, 255, 56)
    gold = (255, 215, 108, 72)
    cap = 16
    half = CARD_W // 2
    left_seed_stem = LEFT_COLS["seed"] + half + 20
    right_seed_stem = RIGHT_COLS["seed"] - half - 20
    left_round1_stem = LEFT_COLS["round1"] + half + 16
    right_round1_stem = RIGHT_COLS["round1"] - half - 16
    left_semi_stem = LEFT_COLS["semi"] + half + 14
    right_semi_stem = RIGHT_COLS["semi"] - half - 14

    for y_top, y_bottom in ((262, 402), (574, 714), (886, 1026), (1206, 1346)):
        mid = (y_top + y_bottom) // 2
        _draw_bracket_pair(draw, y_top, y_bottom, mid, LEFT_COLS["seed"] + half, left_seed_stem, LEFT_COLS["round1"] - half, white_dim, 4)
        _draw_bracket_pair(draw, y_top, y_bottom, mid, RIGHT_COLS["seed"] - half, right_seed_stem, RIGHT_COLS["round1"] + half, white_dim, 4)

    _draw_bracket_pair(draw, 332, 714, 523, LEFT_COLS["round1"] + half, left_round1_stem, LEFT_COLS["semi"] - half, white, 4)
    _draw_bracket_pair(draw, 1102, 1484, 1293, LEFT_COLS["round1"] + half, left_round1_stem, LEFT_COLS["semi"] - half, white, 4)
    _draw_bracket_pair(draw, 332, 714, 523, RIGHT_COLS["round1"] - half, right_round1_stem, RIGHT_COLS["semi"] + half, white, 4)
    _draw_bracket_pair(draw, 1102, 1484, 1293, RIGHT_COLS["round1"] - half, right_round1_stem, RIGHT_COLS["semi"] + half, white, 4)

    _draw_bracket_pair(draw, 523, 1293, 908, LEFT_COLS["semi"] + half, left_semi_stem, LEFT_COLS["conf"] - half, white, 5)
    _draw_bracket_pair(draw, 523, 1293, 908, RIGHT_COLS["semi"] - half, right_semi_stem, RIGHT_COLS["conf"] + half, white, 5)

    draw.line((LEFT_COLS["conf"] + half, 908, 520, 972), fill=faint, width=4)
    draw.line((RIGHT_COLS["conf"] - half, 908, 560, 972), fill=faint, width=4)
    draw.line((520, 972, 540, 972), fill=gold, width=4)
    draw.line((540, 972, 560, 972), fill=gold, width=4)

    draw.rounded_rectangle(CENTER_FINAL_BOX, radius=18, fill=(15, 15, 15, 222), outline=(255, 196, 86, 200), width=3)
    draw.rounded_rectangle(CHAMPION_BOX, radius=22, fill=(15, 15, 15, 222), outline=(255, 196, 86, 200), width=4)


def _draw_scaffold_overlay(draw: ImageDraw.ImageDraw) -> None:
    faint = (255, 255, 255, 24)
    white = (255, 255, 255, 104)
    white_dim = (255, 255, 255, 66)
    gold = (255, 215, 108, 82)
    cap = 16
    half = CARD_W // 2
    left_seed_stem = LEFT_COLS["seed"] + half + 20
    right_seed_stem = RIGHT_COLS["seed"] - half - 20
    left_round1_stem = LEFT_COLS["round1"] + half + 16
    right_round1_stem = RIGHT_COLS["round1"] - half - 16
    left_semi_stem = LEFT_COLS["semi"] + half + 14
    right_semi_stem = RIGHT_COLS["semi"] - half - 14

    def hcap(x: int, y: int, color: tuple[int, int, int, int], width: int = 4) -> None:
        draw.line((x - cap, y, x + cap, y), fill=color, width=width)

    for y_top, y_bottom in ((262, 402), (574, 714), (886, 1026), (1206, 1346)):
        mid = (y_top + y_bottom) // 2
        _draw_bracket_pair(draw, y_top, y_bottom, mid, LEFT_COLS["seed"] + half, left_seed_stem, LEFT_COLS["round1"] - half, white_dim, 3)
        _draw_bracket_pair(draw, y_top, y_bottom, mid, RIGHT_COLS["seed"] - half, right_seed_stem, RIGHT_COLS["round1"] + half, white_dim, 3)

    _draw_bracket_pair(draw, 332, 714, 523, LEFT_COLS["round1"] + half, left_round1_stem, LEFT_COLS["semi"] - half, white, 4)
    _draw_bracket_pair(draw, 1102, 1484, 1293, LEFT_COLS["round1"] + half, left_round1_stem, LEFT_COLS["semi"] - half, white, 4)
    _draw_bracket_pair(draw, 332, 714, 523, RIGHT_COLS["round1"] - half, right_round1_stem, RIGHT_COLS["semi"] + half, white, 4)
    _draw_bracket_pair(draw, 1102, 1484, 1293, RIGHT_COLS["round1"] - half, right_round1_stem, RIGHT_COLS["semi"] + half, white, 4)
    hcap(LEFT_COLS["round1"] + half + 16, 332, white)
    hcap(LEFT_COLS["round1"] + half + 16, 714, white)
    hcap(LEFT_COLS["round1"] + half + 16, 1102, white)
    hcap(LEFT_COLS["round1"] + half + 16, 1484, white)
    hcap(RIGHT_COLS["round1"] - half - 16, 332, white)
    hcap(RIGHT_COLS["round1"] - half - 16, 714, white)
    hcap(RIGHT_COLS["round1"] - half - 16, 1102, white)
    hcap(RIGHT_COLS["round1"] - half - 16, 1484, white)

    _draw_bracket_pair(draw, 523, 1293, 908, LEFT_COLS["semi"] + half, left_semi_stem, LEFT_COLS["conf"] - half, white, 4)
    _draw_bracket_pair(draw, 523, 1293, 908, RIGHT_COLS["semi"] - half, right_semi_stem, RIGHT_COLS["conf"] + half, white, 4)
    hcap(LEFT_COLS["semi"] + half + 14, 523, white)
    hcap(LEFT_COLS["semi"] + half + 14, 1293, white)
    hcap(RIGHT_COLS["semi"] - half - 14, 523, white)
    hcap(RIGHT_COLS["semi"] - half - 14, 1293, white)

    draw.line((LEFT_COLS["conf"] + half, 908, 520, 972), fill=faint, width=3)
    draw.line((RIGHT_COLS["conf"] - half, 908, 560, 972), fill=faint, width=3)
    draw.line((520, 972, 540, 972), fill=gold, width=3)
    draw.line((540, 972, 560, 972), fill=gold, width=3)
    draw.rounded_rectangle(CENTER_FINAL_BOX, radius=18, fill=(15, 15, 15, 232), outline=(255, 196, 86, 220), width=3)
    draw.rounded_rectangle(CHAMPION_BOX, radius=22, fill=(15, 15, 15, 232), outline=(255, 196, 86, 220), width=4)


def _positions() -> dict[str, dict[str, tuple[int, int]]]:
    east = {
        "seed": {
            "Cleveland Cavaliers": (LEFT_COLS["seed"], 262),
            "Miami Heat": (LEFT_COLS["seed"], 402),
            "Boston Celtics": (LEFT_COLS["seed"], 574),
            "Orlando Magic": (LEFT_COLS["seed"], 714),
            "New York Knicks": (LEFT_COLS["seed"], 886),
            "Detroit Pistons": (LEFT_COLS["seed"], 1026),
            "Indiana Pacers": (LEFT_COLS["seed"], 1206),
            "Milwaukee Bucks": (LEFT_COLS["seed"], 1346),
        },
        "round1": {
            "Cleveland Cavaliers": (LEFT_COLS["round1"], 332),
            "Indiana Pacers": (LEFT_COLS["round1"], 714),
            "New York Knicks": (LEFT_COLS["round1"], 1102),
            "Boston Celtics": (LEFT_COLS["round1"], 1484),
        },
        "semi": {
            "Indiana Pacers": (LEFT_COLS["semi"], 523),
            "New York Knicks": (LEFT_COLS["semi"], 1293),
        },
        "conf": {
            "Indiana Pacers": (LEFT_COLS["conf"], 908),
        },
    }
    west = {
        "seed": {
            "Oklahoma City Thunder": (RIGHT_COLS["seed"], 262),
            "Memphis Grizzlies": (RIGHT_COLS["seed"], 402),
            "Houston Rockets": (RIGHT_COLS["seed"], 574),
            "Golden State Warriors": (RIGHT_COLS["seed"], 714),
            "Los Angeles Lakers": (RIGHT_COLS["seed"], 886),
            "Minnesota Timberwolves": (RIGHT_COLS["seed"], 1026),
            "Denver Nuggets": (RIGHT_COLS["seed"], 1206),
            "LA Clippers": (RIGHT_COLS["seed"], 1346),
        },
        "round1": {
            "Oklahoma City Thunder": (RIGHT_COLS["round1"], 332),
            "Golden State Warriors": (RIGHT_COLS["round1"], 714),
            "Minnesota Timberwolves": (RIGHT_COLS["round1"], 1102),
            "Denver Nuggets": (RIGHT_COLS["round1"], 1484),
        },
        "semi": {
            "Oklahoma City Thunder": (RIGHT_COLS["semi"], 523),
            "Minnesota Timberwolves": (RIGHT_COLS["semi"], 1293),
        },
        "conf": {
            "Oklahoma City Thunder": (RIGHT_COLS["conf"], 908),
        },
    }
    return {"east": east, "west": west}


POSITIONS = _positions()


def _build_cards() -> dict[str, Image.Image]:
    cards: dict[str, Image.Image] = {}
    for team, seed in EAST_TEAMS + WEST_TEAMS:
        cards[team] = _make_team_card(team, seed)
    cards[CHAMPION] = _make_team_card(CHAMPION, 1)
    return cards


def _build_moves() -> dict[str, list[Move]]:
    moves: dict[str, list[Move]] = {"round1": [], "semi": [], "conf": [], "final": []}

    round1_targets = {
        "Cleveland Cavaliers": POSITIONS["east"]["round1"]["Cleveland Cavaliers"],
        "Indiana Pacers": POSITIONS["east"]["round1"]["Indiana Pacers"],
        "New York Knicks": POSITIONS["east"]["round1"]["New York Knicks"],
        "Boston Celtics": POSITIONS["east"]["round1"]["Boston Celtics"],
        "Oklahoma City Thunder": POSITIONS["west"]["round1"]["Oklahoma City Thunder"],
        "Golden State Warriors": POSITIONS["west"]["round1"]["Golden State Warriors"],
        "Minnesota Timberwolves": POSITIONS["west"]["round1"]["Minnesota Timberwolves"],
        "Denver Nuggets": POSITIONS["west"]["round1"]["Denver Nuggets"],
    }
    round1_delay = 0.0
    for team in ROUND1_WINNERS:
        side = "east" if team in POSITIONS["east"]["seed"] else "west"
        moves["round1"].append(
            Move(
                team_name=team,
                start=POSITIONS[side]["seed"][team],
                end=round1_targets[team],
                delay=round1_delay,
                duration=1.15,
                score=SERIES_SCORES[("round1", team)],
            )
        )
        round1_delay += 1.55

    semi_targets = {
        "Indiana Pacers": POSITIONS["east"]["semi"]["Indiana Pacers"],
        "New York Knicks": POSITIONS["east"]["semi"]["New York Knicks"],
        "Oklahoma City Thunder": POSITIONS["west"]["semi"]["Oklahoma City Thunder"],
        "Minnesota Timberwolves": POSITIONS["west"]["semi"]["Minnesota Timberwolves"],
    }
    semi_delay = 0.0
    for team in SEMI_WINNERS:
        moves["semi"].append(
            Move(
                team_name=team,
                start=round1_targets[team],
                end=semi_targets[team],
                delay=semi_delay,
                duration=1.35,
                score=SERIES_SCORES[("semi", team)],
            )
        )
        semi_delay += 1.9

    conf_targets = {
        "Indiana Pacers": POSITIONS["east"]["conf"]["Indiana Pacers"],
        "Oklahoma City Thunder": POSITIONS["west"]["conf"]["Oklahoma City Thunder"],
    }
    conf_delay = 0.0
    for team in CONF_WINNERS:
        moves["conf"].append(
            Move(
                team_name=team,
                start=semi_targets[team],
                end=conf_targets[team],
                delay=conf_delay,
                duration=1.55,
                score=SERIES_SCORES[("conf", team)],
            )
        )
        conf_delay += 2.1

    moves["final"].append(
        Move(
            team_name=CHAMPION,
            start=(640, 908),
            end=(540, 962),
            delay=0.35,
            duration=2.4,
            score=SERIES_SCORES[("final", CHAMPION)],
        )
    )
    return moves


MOVES = _build_moves()


def _draw_move(frame: Image.Image, cards: dict[str, Image.Image], move: Move, t: float, active: bool, done: bool, highlight: bool = False) -> None:
    card = cards[move.team_name]
    path_color = (245, 247, 251)
    if done:
        _draw_card(frame, card, move.end[0], move.end[1], 1.0, scale=1.0, glow=highlight, glow_color=(255, 225, 126))
        _draw_path(frame, move.start, move.end, path_color)
        _draw_score_badge(frame, move.score, move.end[0], move.end[1] + CARD_H * 0.74, highlight=highlight)
        return
    if active:
        p = _ease_out((t - move.delay) / move.duration)
        x, y = _manhattan_point(move.start, move.end, p, first="horizontal")
        _draw_card(frame, card, x, y, 1.0, scale=1.0 + 0.06 * p, glow=True, glow_color=(255, 225, 126))
        _draw_path(frame, move.start, (x, y), path_color)
        return
    _draw_card(frame, card, move.start[0], move.start[1], 0.34, scale=0.97)


def render_video(output_path: Path, audio_path: Path, duration: float, fps: int) -> Path:
    duration = TOTAL_DURATION
    background = _make_background()
    cards = _build_cards()

    title_font = _load_font(60, bold=True)
    subtitle_font = _load_font(22, bold=True)
    label_font = _load_font(20, bold=True)
    stage_font = _load_font(17, bold=True)
    finals_font = _load_font(22, bold=True)
    champ_font = _load_font(24, bold=True)
    small_font = _load_font(20, bold=False)

    def make_frame(t: float) -> np.ndarray:
        frame = background.copy()
        draw = ImageDraw.Draw(frame, "RGBA")
        _draw_header(draw, title_font, subtitle_font, label_font)
        _draw_scaffold(draw)
        _draw_trophy(frame, scale=1.0)

        # Seed cards stay visible as the bracket builds.
        for team, pos in POSITIONS["east"]["seed"].items():
            alpha = 0.42
            scale = 0.95
            if team in ROUND1_WINNERS:
                alpha = 0.55
                scale = 0.96
            _draw_card(frame, cards[team], pos[0], pos[1], alpha, scale=scale)
        for team, pos in POSITIONS["west"]["seed"].items():
            alpha = 0.42
            scale = 0.95
            if team in ROUND1_WINNERS:
                alpha = 0.55
                scale = 0.96
            _draw_card(frame, cards[team], pos[0], pos[1], alpha, scale=scale)

        # Round 1.
        if t < ROUND1_DURATION:
            draw.text((246, 286), ROUND1_LABEL, font=stage_font, fill="#f6f8fb", anchor="ma")
            draw.text((834, 286), ROUND1_LABEL, font=stage_font, fill="#f6f8fb", anchor="ma")
            draw.text((118, 268), "One logo moves at a time", font=_load_font(14, bold=False), fill="#dbe7ff")
            draw.text((WIDTH - 118, 268), "One logo moves at a time", font=_load_font(14, bold=False), fill="#dbe7ff", anchor="ra")

            for move in MOVES["round1"]:
                active = move.delay <= t < move.delay + move.duration
                done = t >= move.delay + move.duration
                _draw_move(frame, cards, move, t, active, done, highlight=done)

            for team in {name for name, _seed in EAST_TEAMS + WEST_TEAMS}:
                if team not in ROUND1_WINNERS:
                    side = "east" if team in POSITIONS["east"]["seed"] else "west"
                    pos = POSITIONS[side]["seed"][team]
                    _draw_card(frame, cards[team], pos[0], pos[1], 0.22, scale=0.93)

        # Semifinals.
        elif t < ROUND1_DURATION + SEMI_DURATION:
            local_t = t - ROUND1_DURATION
            draw.text((318, 286), SEMI_LABEL, font=stage_font, fill="#f6f8fb", anchor="ma")
            draw.text((WIDTH - 318, 286), SEMI_LABEL, font=stage_font, fill="#f6f8fb", anchor="ma")

            for team in ROUND1_WINNERS[:4]:
                _draw_card(frame, cards[team], *POSITIONS["east"]["round1"].get(team, POSITIONS["west"]["round1"].get(team)), 0.90, scale=1.0)
            for team in ROUND1_WINNERS[4:]:
                _draw_card(frame, cards[team], *POSITIONS["west"]["round1"].get(team, POSITIONS["east"]["round1"].get(team)), 0.90, scale=1.0)

            for move in MOVES["semi"]:
                active = move.delay <= local_t < move.delay + move.duration
                done = local_t >= move.delay + move.duration
                _draw_move(frame, cards, move, local_t, active, done, highlight=done)

        # Conference finals.
        elif t < ROUND1_DURATION + SEMI_DURATION + CONF_DURATION:
            local_t = t - ROUND1_DURATION - SEMI_DURATION
            draw.text((440, 286), CONF_LABEL, font=stage_font, fill="#f6f8fb", anchor="ma")
            for team in SEMI_WINNERS:
                target = POSITIONS["east"]["semi"].get(team) or POSITIONS["west"]["semi"].get(team)
                _draw_card(frame, cards[team], *target, 0.90, scale=1.0)
            for move in MOVES["conf"]:
                active = move.delay <= local_t < move.delay + move.duration
                done = local_t >= move.delay + move.duration
                _draw_move(frame, cards, move, local_t, active, done, highlight=done)

            draw.text((540, 970), FINAL_LABEL, font=finals_font, fill="#f6f8fb", anchor="ma")

        # Finals and champion.
        else:
            local_t = t - ROUND1_DURATION - SEMI_DURATION - CONF_DURATION
            draw.text((540, 970), FINAL_LABEL, font=finals_font, fill="#f6f8fb", anchor="ma")
            _draw_card(frame, cards["Indiana Pacers"], *POSITIONS["east"]["conf"]["Indiana Pacers"], 0.92, scale=1.0)
            _draw_card(frame, cards["Oklahoma City Thunder"], *POSITIONS["west"]["conf"]["Oklahoma City Thunder"], 0.92, scale=1.0)

            champ_move = MOVES["final"][0]
            active = champ_move.delay <= local_t < champ_move.delay + champ_move.duration
            done = local_t >= champ_move.delay + champ_move.duration
            if active:
                p = _ease_out((local_t - champ_move.delay) / champ_move.duration)
                x, y = _manhattan_point(champ_move.start, champ_move.end, p, first="horizontal")
                _draw_card(frame, cards[CHAMPION], x, y, 1.0, scale=1.04 + 0.12 * p, glow=True, glow_color=(255, 229, 143))
            elif done:
                _draw_card(frame, cards[CHAMPION], champ_move.end[0], champ_move.end[1], 1.0, scale=1.14, glow=True, glow_color=(255, 229, 143))
                draw.text((540, 1150), CHAMPION_LABEL, font=champ_font, fill="#d9a543", anchor="ma")
                draw.text((540, 1182), CHAMPION, font=small_font, fill="#f6f8fb", anchor="ma")

        _draw_scaffold_overlay(draw)
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
    parser = argparse.ArgumentParser(description="Generate an NBA playoff bracket 2025 Shorts video.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--audio", type=Path, default=DEFAULT_AUDIO)
    parser.add_argument("--duration", type=float, default=TOTAL_DURATION)
    parser.add_argument("--fps", type=int, default=FPS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = render_video(
        output_path=args.output,
        audio_path=args.audio,
        duration=args.duration,
        fps=args.fps,
    )
    print(f"[video_generator] NBA playoff bracket 2025 Shorts generated -> {output}")


if __name__ == "__main__":
    main()
