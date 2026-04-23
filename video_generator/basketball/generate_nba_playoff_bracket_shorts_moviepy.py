from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from moviepy import VideoClip
from PIL import Image, ImageDraw, ImageFilter, ImageOps

from video_generator.basketball.generate_nba_titles_shorts_moviepy import TEAM_COLORS
from video_generator.generate_ucl_barchart_race_moviepy import (
    DEFAULT_AUDIO,
    _fit_font_size,
    _load_font,
    build_audio_track,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "basketball" / "nba_playoff_bracket_shorts.mp4"
LOGO_DIR = PROJECT_ROOT / "data" / "raw" / "nba_team_logos"

WIDTH = 1080
HEIGHT = 1920
FPS = 30
TOTAL_DURATION = 42.0

ROUND1_DURATION = 16.0
SEMI_DURATION = 10.0
CONF_DURATION = 7.0
FINAL_DURATION = 9.0

TITLE = "2022 NBA PLAYOFFS"
LEFT_LABEL = "EASTERN CONFERENCE"
RIGHT_LABEL = "WESTERN CONFERENCE"
FINAL_LABEL = "NBA FINALS"

CARD_W = 112
CARD_H = 112

LEFT_EDGE = 78
RIGHT_EDGE = WIDTH - 78

LEFT_COLS = {
    "seed": LEFT_EDGE,
    "round1": 150,
    "semi": 286,
    "conf": 414,
}
RIGHT_COLS = {
    "seed": RIGHT_EDGE,
    "round1": 828,
    "semi": 704,
    "conf": 576,
}
FINAL_BOX = (206, 806, 874, 1152)
CHAMPION_BOX = (360, 928, 720, 1070)
TROPHY_PHOTO = PROJECT_ROOT / "data" / "raw" / "nba_trophy_photo.jpg"

TEAM_LOGO_FILES = {
    "Atlanta Hawks": "atlanta_hawks.png",
    "Boston Celtics": "boston_celtics.png",
    "Brooklyn Nets": "brooklyn_nets.png",
    "Chicago Bulls": "chicago_bulls.png",
    "Dallas Mavericks": "dallas_mavericks.png",
    "Denver Nuggets": "denver_nuggets.png",
    "Golden State Warriors": "golden_state_warriors.png",
    "Memphis Grizzlies": "memphis_grizzlies.png",
    "Miami Heat": "miami_heat.png",
    "Milwaukee Bucks": "milwaukee_bucks.png",
    "Minnesota Timberwolves": "minnesota_timberwolves.png",
    "New Orleans Pelicans": "new_orleans_pelicans.png",
    "Philadelphia 76ers": "philadelphia_76ers.png",
    "Phoenix Suns": "phoenix_suns.png",
    "Toronto Raptors": "toronto_raptors.png",
    "Utah Jazz": "utah_jazz.png",
}

EAST_TEAMS = [
    ("Miami Heat", 1),
    ("Atlanta Hawks", 8),
    ("Philadelphia 76ers", 4),
    ("Toronto Raptors", 5),
    ("Milwaukee Bucks", 3),
    ("Chicago Bulls", 6),
    ("Boston Celtics", 2),
    ("Brooklyn Nets", 7),
]

WEST_TEAMS = [
    ("Phoenix Suns", 1),
    ("New Orleans Pelicans", 8),
    ("Memphis Grizzlies", 2),
    ("Minnesota Timberwolves", 7),
    ("Golden State Warriors", 3),
    ("Denver Nuggets", 6),
    ("Dallas Mavericks", 4),
    ("Utah Jazz", 5),
]

ROUND1_EAST_WINNERS = ["Miami Heat", "Philadelphia 76ers", "Milwaukee Bucks", "Boston Celtics"]
ROUND1_WEST_WINNERS = ["Phoenix Suns", "Memphis Grizzlies", "Golden State Warriors", "Dallas Mavericks"]
SEMI_EAST_WINNERS = ["Miami Heat", "Boston Celtics"]
SEMI_WEST_WINNERS = ["Phoenix Suns", "Golden State Warriors"]
CONF_EAST_WINNER = "Boston Celtics"
CONF_WEST_WINNER = "Golden State Warriors"
CHAMPION = "Golden State Warriors"

SERIES_SCORES = {
    "Miami Heat": "4-1",
    "Philadelphia 76ers": "4-2",
    "Milwaukee Bucks": "4-1",
    "Boston Celtics": "4-0",
    "Phoenix Suns": "4-2",
    "Memphis Grizzlies": "4-2",
    "Golden State Warriors": "4-2",
    "Dallas Mavericks": "4-2",
    "SEMI_MIA": "4-2",
    "SEMI_BOS": "4-3",
    "SEMI_PHX": "4-3",
    "SEMI_GSW": "4-2",
    "CONF_BOS": "4-3",
    "CONF_GSW": "4-1",
    "CHAMPION": "4-2",
}


@dataclass(frozen=True)
class TeamCard:
    name: str
    seed: int


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


def _text_on(color: str) -> str:
    r, g, b = _hex_to_rgb(color)
    luminance = (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255.0
    return "#10233f" if luminance > 0.66 else "#f4f7fb"


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


def _point(start: tuple[int, int], end: tuple[int, int], t: float) -> tuple[float, float]:
    return (_lerp(start[0], end[0], t), _lerp(start[1], end[1], t))


def _load_logo(team_name: str) -> Image.Image | None:
    logo_name = TEAM_LOGO_FILES.get(team_name)
    if logo_name is None:
        return None
    path = LOGO_DIR / logo_name
    if not path.exists():
        return None
    return ImageOps.contain(Image.open(path).convert("RGBA"), (58, 58), method=Image.Resampling.LANCZOS)


def _make_background() -> Image.Image:
    xx = np.linspace(0, 1, WIDTH, dtype=np.float32)
    yy = np.linspace(0, 1, HEIGHT, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(xx, yy)

    deep = np.array([6, 12, 24], dtype=np.float32)
    slate = np.array([14, 30, 52], dtype=np.float32)
    gold = np.array([242, 194, 74], dtype=np.float32)
    red = np.array([205, 52, 64], dtype=np.float32)
    blue = np.array([54, 108, 214], dtype=np.float32)

    mix = np.clip(0.8 * grid_y + 0.12 * (1.0 - grid_x), 0, 1)
    center = np.exp(-(((grid_x - 0.5) / 0.22) ** 2 + ((grid_y - 0.20) / 0.12) ** 2))
    left = np.exp(-(((grid_x - 0.14) / 0.16) ** 2 + ((grid_y - 0.52) / 0.34) ** 2))
    right = np.exp(-(((grid_x - 0.86) / 0.16) ** 2 + ((grid_y - 0.52) / 0.34) ** 2))
    mid = np.exp(-(((grid_x - 0.5) / 0.18) ** 2 + ((grid_y - 0.60) / 0.24) ** 2))

    img = np.clip(
        deep[None, None, :] * (1.0 - mix[..., None])
        + slate[None, None, :] * (0.8 * mix[..., None])
        + red[None, None, :] * (0.12 * left[..., None])
        + blue[None, None, :] * (0.12 * right[..., None])
        + gold[None, None, :] * (0.18 * center[..., None] + 0.10 * mid[..., None]),
        0,
        255,
    ).astype(np.uint8)
    frame = Image.fromarray(img, mode="RGB").convert("RGBA")

    overlay = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")
    draw.rounded_rectangle((26, 26, WIDTH - 26, HEIGHT - 26), radius=42, outline=(255, 255, 255, 20), width=2)
    draw.ellipse((270, 280, 810, 840), outline=(255, 204, 82, 24), width=5)
    draw.ellipse((336, 346, 744, 754), outline=(255, 255, 255, 12), width=3)
    draw.line((72, 178, WIDTH - 72, 178), fill=(255, 255, 255, 14), width=2)
    draw.line((72, 1652, WIDTH - 72, 1652), fill=(255, 255, 255, 10), width=2)
    overlay = overlay.filter(ImageFilter.GaussianBlur(radius=3))
    frame.alpha_composite(overlay)
    return frame


def _make_card(team_name: str, seed: int) -> Image.Image:
    primary, secondary = TEAM_COLORS.get(team_name, ("#d7dbe3", "#10233f"))
    primary_rgb = _hex_to_rgb(primary)
    secondary_rgb = _hex_to_rgb(secondary)

    card = Image.new("RGBA", (CARD_W, CARD_H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(card, "RGBA")
    draw.rounded_rectangle((2, 2, CARD_W - 3, CARD_H - 3), radius=22, fill=(10, 12, 18, 88))
    draw.rounded_rectangle((0, 0, CARD_W - 1, CARD_H - 1), radius=22, fill=(246, 248, 251, 255), outline=(255, 255, 255, 34), width=2)
    draw.rounded_rectangle((0, 0, CARD_W - 1, 18), radius=22, fill=(*primary_rgb, 255))
    draw.rounded_rectangle((0, CARD_H - 16, CARD_W - 1, CARD_H - 1), radius=14, fill=(*secondary_rgb, 86))

    logo = _load_logo(team_name)
    if logo is not None:
        logo = ImageOps.contain(logo, (88, 88), method=Image.Resampling.LANCZOS)
        card.alpha_composite(logo, ((CARD_W - logo.width) // 2, (CARD_H - logo.height) // 2 - 1))
    else:
        abbrev = "".join(part[0] for part in team_name.split()[:3]).upper()
        mono = _fit_font_size(draw, abbrev, 48, 24, 16, bold=True)
        draw.text((CARD_W // 2, CARD_H // 2 - 1), abbrev, font=mono, fill="#10233f", anchor="mm")

    return card


def _draw_card(
    frame: Image.Image,
    card: Image.Image,
    x: float,
    y: float,
    alpha: float,
    scale: float = 1.0,
    glow: bool = False,
    glow_color: tuple[int, int, int] = (255, 204, 82),
) -> None:
    alpha = _clamp(alpha)
    scale = max(0.90, scale)
    img = card
    if scale != 1.0:
        img = img.resize((max(1, int(img.width * scale)), max(1, int(img.height * scale))), Image.Resampling.LANCZOS)
    if alpha < 0.999:
        img = img.copy()
        channel = img.getchannel("A").point(lambda a: int(a * alpha))
        img.putalpha(channel)

    px = int(x - img.width // 2)
    py = int(y - img.height // 2)
    if glow:
        glow_layer = Image.new("RGBA", (img.width + 56, img.height + 56), (0, 0, 0, 0))
        gd = ImageDraw.Draw(glow_layer, "RGBA")
        gd.rounded_rectangle((18, 18, glow_layer.width - 18, glow_layer.height - 18), radius=18, fill=(*glow_color, 98))
        glow_layer = glow_layer.filter(ImageFilter.GaussianBlur(radius=16))
        frame.alpha_composite(glow_layer, (px - 28, py - 28))
    frame.alpha_composite(img, (px, py))


def _draw_score_badge(frame: Image.Image, text: str, x: float, y: float, color: tuple[int, int, int]) -> None:
    badge = Image.new("RGBA", (104, 36), (0, 0, 0, 0))
    draw = ImageDraw.Draw(badge, "RGBA")
    draw.rounded_rectangle((0, 0, 103, 35), radius=15, fill=(*color, 224), outline=(255, 255, 255, 44), width=1)
    font = _load_font(20, bold=True)
    draw.text((52, 18), text, font=font, fill="#f4f7fb", anchor="mm")
    frame.alpha_composite(badge, (int(x - 52), int(y - 18)))


def _manhattan_point(
    start: tuple[int, int],
    end: tuple[int, int],
    t: float,
    first: str = "horizontal",
) -> tuple[float, float]:
    t = _clamp(t)
    if first == "vertical":
        split = 0.5
        if t <= split:
            local = _ease_in_out(t / split)
            return (_lerp(start[0], start[0], local), _lerp(start[1], end[1], local))
        local = _ease_in_out((t - split) / split)
        return (_lerp(start[0], end[0], local), end[1])

    split = 0.5
    if t <= split:
        local = _ease_in_out(t / split)
        return (_lerp(start[0], end[0], local), start[1])
    local = _ease_in_out((t - split) / split)
    return (end[0], _lerp(start[1], end[1], local))


def _draw_header(draw: ImageDraw.ImageDraw, title_font, label_font, tag_font) -> None:
    draw.text((WIDTH // 2, 54), TITLE, font=title_font, fill="#f4f7fb", anchor="ma")
    draw.text((70, 138), LEFT_LABEL, font=label_font, fill="#f4f7fb")
    draw.text((WIDTH - 70, 138), RIGHT_LABEL, font=label_font, fill="#f4f7fb", anchor="ra")
    draw.rounded_rectangle((62, 180, 288, 226), radius=16, fill=(205, 52, 64, 255))
    draw.rounded_rectangle((WIDTH - 288, 180, WIDTH - 62, 226), radius=16, fill=(54, 108, 214, 255))
    draw.text((175, 204), "EAST", font=tag_font, fill="#f4f7fb", anchor="ma")
    draw.text((WIDTH - 175, 204), "WEST", font=tag_font, fill="#f4f7fb", anchor="ma")


def _draw_scaffold(draw: ImageDraw.ImageDraw) -> None:
    east = (225, 84, 94)
    west = (67, 121, 223)
    faint = (255, 255, 255, 18)
    semi_line = (255, 255, 255, 34)
    final_line = (255, 214, 102, 54)
    bracket_top = 255
    bracket_bottom = 1680

    # seed column guides
    for y in (260, 450, 640, 830, 1020, 1210, 1400, 1590):
        draw.line((LEFT_COLS["seed"] + CARD_W, y, LEFT_COLS["round1"], y), fill=(*east, 140), width=4)
        draw.line((RIGHT_COLS["round1"] + CARD_W, y, RIGHT_COLS["seed"], y), fill=(*west, 140), width=4)

    # round 1 to semis
    for y in (355, 735, 1115, 1495):
        draw.line((LEFT_COLS["round1"] + CARD_W, y, LEFT_COLS["semi"], y), fill=(*east, 150), width=4)
        draw.line((RIGHT_COLS["semi"] + CARD_W, y, RIGHT_COLS["round1"], y), fill=(*west, 150), width=4)
    draw.line((LEFT_COLS["round1"] + CARD_W + 18, 355, LEFT_COLS["round1"] + CARD_W + 18, 735), fill=semi_line, width=4)
    draw.line((LEFT_COLS["round1"] + CARD_W + 18, 1115, LEFT_COLS["round1"] + CARD_W + 18, 1495), fill=semi_line, width=4)
    draw.line((RIGHT_COLS["round1"] - 18, 355, RIGHT_COLS["round1"] - 18, 735), fill=semi_line, width=4)
    draw.line((RIGHT_COLS["round1"] - 18, 1115, RIGHT_COLS["round1"] - 18, 1495), fill=semi_line, width=4)

    # semis to conference
    draw.line((LEFT_COLS["semi"] + CARD_W, 545, LEFT_COLS["conf"], 545), fill=(*east, 160), width=5)
    draw.line((LEFT_COLS["semi"] + CARD_W, 1355, LEFT_COLS["conf"], 1355), fill=(*east, 160), width=5)
    draw.line((RIGHT_COLS["conf"] + CARD_W, 545, RIGHT_COLS["semi"], 545), fill=(*west, 160), width=5)
    draw.line((RIGHT_COLS["conf"] + CARD_W, 1355, RIGHT_COLS["semi"], 1355), fill=(*west, 160), width=5)
    draw.line((LEFT_COLS["semi"] + CARD_W + 22, 545, LEFT_COLS["semi"] + CARD_W + 22, 1355), fill=semi_line, width=4)
    draw.line((RIGHT_COLS["semi"] - 22, 545, RIGHT_COLS["semi"] - 22, 1355), fill=semi_line, width=4)

    # finals lines
    draw.line((LEFT_COLS["conf"] + CARD_W, 950, 520, 992), fill=faint, width=4)
    draw.line((RIGHT_COLS["conf"], 950, 560, 992), fill=faint, width=4)
    draw.line((520, 992, 540, 992), fill=final_line, width=4)
    draw.line((540, 992, 560, 992), fill=final_line, width=4)
    draw.line((540, bracket_top, 540, bracket_bottom), fill=(255, 255, 255, 10), width=2)
    draw.line((LEFT_COLS["conf"] + CARD_W + 20, 950, LEFT_COLS["conf"] + CARD_W + 20, 1080), fill=semi_line, width=4)
    draw.line((RIGHT_COLS["conf"] - 20, 950, RIGHT_COLS["conf"] - 20, 1080), fill=semi_line, width=4)
    draw.ellipse((384, 684, 696, 1052), outline=(255, 208, 80, 22), width=4)
    draw.ellipse((434, 734, 646, 1010), outline=(255, 255, 255, 10), width=3)


def _draw_round1_guides(draw: ImageDraw.ImageDraw) -> None:
    east = (225, 84, 94, 210)
    west = (67, 121, 223, 210)
    for y in (260, 450, 640, 830, 1020, 1210, 1400, 1590):
        draw.line((LEFT_COLS["seed"] + CARD_W, y, LEFT_COLS["round1"], y), fill=east, width=5)
        draw.line((RIGHT_COLS["round1"] + CARD_W, y, RIGHT_COLS["seed"], y), fill=west, width=5)


def _make_cards() -> dict[str, Image.Image]:
    cards: dict[str, Image.Image] = {}
    for name, seed in EAST_TEAMS + WEST_TEAMS:
        cards[name] = _make_card(name, seed)
    cards[CHAMPION] = _make_card(CHAMPION, 3)
    return cards


def _slot_positions() -> dict[str, dict[str, tuple[int, int]]]:
    east = {
        "seed": {
            "Miami Heat": (LEFT_COLS["seed"], 260),
            "Atlanta Hawks": (LEFT_COLS["seed"], 450),
            "Philadelphia 76ers": (LEFT_COLS["seed"], 640),
            "Toronto Raptors": (LEFT_COLS["seed"], 830),
            "Milwaukee Bucks": (LEFT_COLS["seed"], 1020),
            "Chicago Bulls": (LEFT_COLS["seed"], 1210),
            "Boston Celtics": (LEFT_COLS["seed"], 1400),
            "Brooklyn Nets": (LEFT_COLS["seed"], 1590),
        },
        "round1": {
            "Miami Heat": (LEFT_COLS["round1"], 355),
            "Philadelphia 76ers": (LEFT_COLS["round1"], 735),
            "Milwaukee Bucks": (LEFT_COLS["round1"], 1115),
            "Boston Celtics": (LEFT_COLS["round1"], 1495),
        },
        "semi": {
            "Miami Heat": (LEFT_COLS["semi"], 545),
            "Boston Celtics": (LEFT_COLS["semi"], 1355),
        },
        "conf": {"Boston Celtics": (LEFT_COLS["conf"], 950)},
    }
    west = {
        "seed": {
            "Phoenix Suns": (RIGHT_COLS["seed"], 260),
            "New Orleans Pelicans": (RIGHT_COLS["seed"], 450),
            "Memphis Grizzlies": (RIGHT_COLS["seed"], 640),
            "Minnesota Timberwolves": (RIGHT_COLS["seed"], 830),
            "Golden State Warriors": (RIGHT_COLS["seed"], 1020),
            "Denver Nuggets": (RIGHT_COLS["seed"], 1210),
            "Dallas Mavericks": (RIGHT_COLS["seed"], 1400),
            "Utah Jazz": (RIGHT_COLS["seed"], 1590),
        },
        "round1": {
            "Phoenix Suns": (RIGHT_COLS["round1"], 355),
            "Memphis Grizzlies": (RIGHT_COLS["round1"], 735),
            "Golden State Warriors": (RIGHT_COLS["round1"], 1115),
            "Dallas Mavericks": (RIGHT_COLS["round1"], 1495),
        },
        "semi": {
            "Phoenix Suns": (RIGHT_COLS["semi"], 545),
            "Golden State Warriors": (RIGHT_COLS["semi"], 1355),
        },
        "conf": {"Golden State Warriors": (RIGHT_COLS["conf"], 950)},
    }
    return {"east": east, "west": west}


POSITIONS = _slot_positions()


def _build_moves() -> dict[str, list[Move]]:
    moves: dict[str, list[Move]] = {"round1": [], "semi": [], "conf": [], "final": []}

    round1_order = ROUND1_EAST_WINNERS + ROUND1_WEST_WINNERS
    round1_targets = {
        "Miami Heat": POSITIONS["east"]["round1"]["Miami Heat"],
        "Philadelphia 76ers": POSITIONS["east"]["round1"]["Philadelphia 76ers"],
        "Milwaukee Bucks": POSITIONS["east"]["round1"]["Milwaukee Bucks"],
        "Boston Celtics": POSITIONS["east"]["round1"]["Boston Celtics"],
        "Phoenix Suns": POSITIONS["west"]["round1"]["Phoenix Suns"],
        "Memphis Grizzlies": POSITIONS["west"]["round1"]["Memphis Grizzlies"],
        "Golden State Warriors": POSITIONS["west"]["round1"]["Golden State Warriors"],
        "Dallas Mavericks": POSITIONS["west"]["round1"]["Dallas Mavericks"],
    }
    round1_delay = 0.0
    for team in round1_order:
        side = "east" if team in POSITIONS["east"]["seed"] else "west"
        moves["round1"].append(
            Move(
                team_name=team,
                start=POSITIONS[side]["seed"][team],
                end=round1_targets[team],
                delay=round1_delay,
                duration=1.25,
                score=SERIES_SCORES[team],
            )
        )
        round1_delay += 1.8

    semi_order = SEMI_EAST_WINNERS + SEMI_WEST_WINNERS
    semi_targets = {
        "Miami Heat": POSITIONS["east"]["semi"]["Miami Heat"],
        "Boston Celtics": POSITIONS["east"]["semi"]["Boston Celtics"],
        "Phoenix Suns": POSITIONS["west"]["semi"]["Phoenix Suns"],
        "Golden State Warriors": POSITIONS["west"]["semi"]["Golden State Warriors"],
    }
    semi_delay = 0.0
    for team in semi_order:
        start = round1_targets[team]
        moves["semi"].append(
            Move(
                team_name=team,
                start=start,
                end=semi_targets[team],
                delay=semi_delay,
                duration=1.5,
                score=SERIES_SCORES["SEMI_MIA" if team == "Miami Heat" else "SEMI_BOS" if team == "Boston Celtics" else "SEMI_PHX" if team == "Phoenix Suns" else "SEMI_GSW"],
            )
        )
        semi_delay += 2.0

    conf_order = [CONF_EAST_WINNER, CONF_WEST_WINNER]
    conf_targets = {
        CONF_EAST_WINNER: POSITIONS["east"]["conf"]["Boston Celtics"],
        CONF_WEST_WINNER: POSITIONS["west"]["conf"]["Golden State Warriors"],
    }
    conf_delay = 0.0
    for team in conf_order:
        start = semi_targets[team]
        moves["conf"].append(
            Move(
                team_name=team,
                start=start,
                end=conf_targets[team],
                delay=conf_delay,
                duration=1.6,
                score=SERIES_SCORES["CONF_BOS" if team == CONF_EAST_WINNER else "CONF_GSW"],
            )
        )
        conf_delay += 2.2

    moves["final"].append(
        Move(
            team_name=CHAMPION,
            start=(640, 950),
            end=(540, 988),
            delay=0.4,
            duration=2.4,
            score=SERIES_SCORES["CHAMPION"],
        )
    )
    return moves


MOVES = _build_moves()


def _draw_move(frame: Image.Image, cards: dict[str, Image.Image], move: Move, t: float, active: bool, done: bool, target_alpha: float = 1.0) -> None:
    card = cards[move.team_name]
    if done:
        _draw_card(frame, card, move.end[0], move.end[1], target_alpha, scale=1.0, glow=active)
        badge_y = move.end[1] + CARD_H * 0.72
        badge_x = move.end[0]
        badge_color = _hex_to_rgb(TEAM_COLORS.get(move.team_name, ("#24364f", "#24364f"))[0])
        _draw_score_badge(frame, move.score, badge_x, badge_y, badge_color)
        return
    if active:
        p = _ease_out((t - move.delay) / move.duration)
        x, y = _manhattan_point(move.start, move.end, p, first="horizontal")
        _draw_card(frame, card, x, y, 1.0, scale=1.0 + 0.05 * p, glow=True)
        return
    _draw_card(frame, card, move.start[0], move.start[1], 0.34, scale=0.98)


def render_video(output_path: Path, audio_path: Path, duration: float, fps: int) -> Path:
    duration = TOTAL_DURATION
    background = _make_background()
    cards = _make_cards()
    title_font = _load_font(54, bold=True)
    label_font = _load_font(22, bold=True)
    tag_font = _load_font(18, bold=True)
    stage_font = _load_font(18, bold=True)
    finals_font = _load_font(24, bold=True)
    subtle_font = _load_font(20, bold=False)

    def make_frame(t: float) -> np.ndarray:
        frame = background.copy()
        draw = ImageDraw.Draw(frame, "RGBA")
        _draw_header(draw, title_font, label_font, tag_font)
        _draw_scaffold(draw)
        _draw_trophy(frame, scale=1.0)

        # Seed cards always visible in the background.
        for team, pos in POSITIONS["east"]["seed"].items():
            _draw_card(frame, cards[team], pos[0], pos[1], 0.40, scale=0.97)
        for team, pos in POSITIONS["west"]["seed"].items():
            _draw_card(frame, cards[team], pos[0], pos[1], 0.40, scale=0.97)
        _draw_round1_guides(draw)

        # Round 1.
        if t < ROUND1_DURATION:
            _draw_stage_label(draw, "ROUND 1", stage_font, "One advance at a time")
            for move in MOVES["round1"]:
                active = move.delay <= t < move.delay + move.duration
                done = t >= move.delay + move.duration
                if move.team_name in ROUND1_EAST_WINNERS + ROUND1_WEST_WINNERS:
                    _draw_move(frame, cards, move, t, active, done)
            # losers stay dim at seeds
            for team in set(EAST_TEAMS[i][0] for i in range(8)) | set(WEST_TEAMS[i][0] for i in range(8)):
                if team not in ROUND1_EAST_WINNERS and team not in ROUND1_WEST_WINNERS:
                    pos = POSITIONS["east"]["seed"].get(team) or POSITIONS["west"]["seed"].get(team)
                    if pos:
                        _draw_card(frame, cards[team], pos[0], pos[1], 0.24, scale=0.94)

        # Semis.
        elif t < ROUND1_DURATION + SEMI_DURATION:
            local_t = t - ROUND1_DURATION
            _draw_stage_label(draw, "SEMIFINALS", stage_font, "Winners keep climbing")
            # Keep completed round1 cards in place.
            for team in ROUND1_EAST_WINNERS:
                _draw_card(frame, cards[team], *POSITIONS["east"]["round1"][team], 0.88, scale=1.0)
            for team in ROUND1_WEST_WINNERS:
                _draw_card(frame, cards[team], *POSITIONS["west"]["round1"][team], 0.88, scale=1.0)
            for move in MOVES["semi"]:
                active = move.delay <= local_t < move.delay + move.duration
                done = local_t >= move.delay + move.duration
                _draw_move(frame, cards, move, local_t, active, done)
            for team, pos in [
                ("Miami Heat", POSITIONS["east"]["round1"]["Miami Heat"]),
                ("Philadelphia 76ers", POSITIONS["east"]["round1"]["Philadelphia 76ers"]),
                ("Milwaukee Bucks", POSITIONS["east"]["round1"]["Milwaukee Bucks"]),
                ("Boston Celtics", POSITIONS["east"]["round1"]["Boston Celtics"]),
                ("Phoenix Suns", POSITIONS["west"]["round1"]["Phoenix Suns"]),
                ("Memphis Grizzlies", POSITIONS["west"]["round1"]["Memphis Grizzlies"]),
                ("Golden State Warriors", POSITIONS["west"]["round1"]["Golden State Warriors"]),
                ("Dallas Mavericks", POSITIONS["west"]["round1"]["Dallas Mavericks"]),
            ]:
                if team not in SEMI_EAST_WINNERS + SEMI_WEST_WINNERS:
                    _draw_card(frame, cards[team], pos[0], pos[1], 0.22, scale=0.95)

        # Conference finals.
        elif t < ROUND1_DURATION + SEMI_DURATION + CONF_DURATION:
            local_t = t - ROUND1_DURATION - SEMI_DURATION
            _draw_stage_label(draw, "CONFERENCE FINALS", stage_font, "Only one team per side survives")
            for team in SEMI_EAST_WINNERS:
                _draw_card(frame, cards[team], *POSITIONS["east"]["semi"][team], 0.88, scale=1.0)
            for team in SEMI_WEST_WINNERS:
                _draw_card(frame, cards[team], *POSITIONS["west"]["semi"][team], 0.88, scale=1.0)
            for move in MOVES["conf"]:
                active = move.delay <= local_t < move.delay + move.duration
                done = local_t >= move.delay + move.duration
                _draw_move(frame, cards, move, local_t, active, done)

            draw.text((540, 846), FINAL_LABEL, font=finals_font, fill="#f4f7fb", anchor="ma")

        # Finals and champion.
        else:
            local_t = t - ROUND1_DURATION - SEMI_DURATION - CONF_DURATION
            _draw_stage_label(draw, "FINALS", stage_font, "The title is decided here")
            _draw_card(frame, cards[CONF_EAST_WINNER], *POSITIONS["east"]["conf"][CONF_EAST_WINNER], 0.90, scale=1.0)
            _draw_card(frame, cards[CONF_WEST_WINNER], *POSITIONS["west"]["conf"][CONF_WEST_WINNER], 0.90, scale=1.0)
            champ_move = MOVES["final"][0]
            active = champ_move.delay <= local_t < champ_move.delay + champ_move.duration
            done = local_t >= champ_move.delay + champ_move.duration
            draw.text((540, 846), FINAL_LABEL, font=finals_font, fill="#f4f7fb", anchor="ma")

            if active:
                p = _ease_out((local_t - champ_move.delay) / champ_move.duration)
                x, y = _manhattan_point(champ_move.start, champ_move.end, p, first="horizontal")
                _draw_card(frame, cards[CHAMPION], x, y, 1.0, scale=1.02 + 0.14 * p, glow=True, glow_color=(255, 231, 133))
            elif done:
                _draw_card(frame, cards[CHAMPION], champ_move.end[0], champ_move.end[1], 1.0, scale=1.12, glow=True, glow_color=(255, 231, 133))
                crown = Image.new("RGBA", (430, 110), (0, 0, 0, 0))
                cd = ImageDraw.Draw(crown, "RGBA")
                cd.rounded_rectangle((0, 0, 429, 109), radius=26, fill=(244, 193, 74, 242))
                frame.alpha_composite(crown, (325, 250))
                draw.text((540, 290), "CHAMPION", font=title_font, fill="#10233f", anchor="ma")
                draw.text((540, 338), CHAMPION, font=subtle_font, fill="#10233f", anchor="ma")

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


def _draw_stage_label(draw: ImageDraw.ImageDraw, title: str, stage_font, subtitle: str) -> None:
    draw.text((136, 248), title, font=stage_font, fill="#f4f7fb")
    draw.text((136, 270), subtitle, font=_load_font(14, bold=False), fill="#b7d5f1")


def _draw_trophy(frame: Image.Image, scale: float = 1.0) -> None:
    if TROPHY_PHOTO.exists():
        photo = Image.open(TROPHY_PHOTO).convert("RGBA")
        photo = ImageOps.contain(photo, (360, 480), method=Image.Resampling.LANCZOS)
        back = Image.new("RGBA", (photo.width + 56, photo.height + 56), (0, 0, 0, 0))
        bd = ImageDraw.Draw(back, "RGBA")
        bd.rounded_rectangle((6, 6, back.width - 7, back.height - 7), radius=26, fill=(0, 0, 0, 88))
        back = back.filter(ImageFilter.GaussianBlur(radius=10))
        frame.alpha_composite(back, ((WIDTH - back.width) // 2, 624))
        frame.alpha_composite(photo, ((WIDTH - photo.width) // 2, 624))
        return

    w = int(210 * scale)
    h = int(300 * scale)
    trophy = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(trophy, "RGBA")
    gold = (248, 208, 91, 210)
    bright = (255, 236, 151, 255)
    dark = (178, 127, 24, 255)
    shadow = (60, 42, 10, 150)
    draw.ellipse((58, 12, w - 58, 86), fill=gold, outline=bright, width=3)
    draw.rounded_rectangle((34, 70, w - 34, 164), radius=26, fill=gold, outline=bright, width=3)
    draw.ellipse((14, 54, 62, 142), fill=gold, outline=bright, width=3)
    draw.ellipse((w - 62, 54, w - 14, 142), fill=gold, outline=bright, width=3)
    draw.rectangle((w // 2 - 18, 154, w // 2 + 18, 214), fill=dark)
    draw.rounded_rectangle((w // 2 - 84, 210, w // 2 + 84, 254), radius=12, fill=shadow)
    draw.rounded_rectangle((w // 2 - 108, 252, w // 2 + 108, 288), radius=14, fill=dark, outline=bright, width=2)
    draw.line((w // 2, 16, w // 2, 68), fill=(255, 255, 255, 80), width=4)
    trophy = trophy.filter(ImageFilter.GaussianBlur(radius=0.2))
    frame.alpha_composite(trophy, ((WIDTH - w) // 2, 705))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate an NBA playoff bracket Shorts video.")
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
    print(f"[video_generator] NBA playoff bracket Shorts generated -> {output}")


if __name__ == "__main__":
    main()
