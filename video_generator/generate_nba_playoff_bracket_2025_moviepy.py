from __future__ import annotations

import argparse
from collections import deque
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from moviepy import AudioFileClip, VideoClip
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageOps


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "basketball" / "nba_playoff_bracket_2025_style.mp4"
DEFAULT_AUDIO = PROJECT_ROOT / "data" / "raw" / "audio" / "Midnight_Grip_20260402_0828.mp3"
LOGO_DIR = PROJECT_ROOT / "data" / "raw" / "nba_team_logos"
NBA_LOGO_PATH = PROJECT_ROOT / "data" / "raw" / "nba_logo.png"
TROPHY_PHOTO_PATH = PROJECT_ROOT / "data" / "raw" / "nba_trophy_photo_alt.png"
LEGACY_TROPHY_PHOTO_PATH = PROJECT_ROOT / "data" / "raw" / "nba_trophy_photo.jpg"
BRACKET_SVG = Path.home() / "Downloads" / "bracket_lines_overlay.svg"

WIDTH = 1080
HEIGHT = 1920
FPS = 30
TOTAL_DURATION = 40.0
CHAMPION_HOLD_SECONDS = 5.0
TIMELINE_SCALE = 2.2
TEAM_TIMELINE_OFFSET = 1.05
FINAL_STAGE_RAW_BASE = 16.18
EXPORT_CRF = 16
EXPORT_PRESET = "slow"
EXPORT_SHARPEN = ImageFilter.UnsharpMask(radius=1.1, percent=90, threshold=2)

LEFT_TEAMS = [
    ("Cleveland Cavaliers", 1, "CLE"),
    ("Miami Heat", 8, "MIA"),
    ("Indiana Pacers", 4, "IND"),
    ("Milwaukee Bucks", 5, "MIL"),
    ("New York Knicks", 3, "NYK"),
    ("Detroit Pistons", 6, "DET"),
    ("Boston Celtics", 2, "BOS"),
    ("Orlando Magic", 7, "ORL"),
]

RIGHT_TEAMS = [
    ("Oklahoma City Thunder", 1, "OKC"),
    ("Memphis Grizzlies", 8, "MEM"),
    ("Denver Nuggets", 4, "DEN"),
    ("LA Clippers", 5, "LAC"),
    ("Los Angeles Lakers", 3, "LAL"),
    ("Minnesota Timberwolves", 6, "MIN"),
    ("Houston Rockets", 2, "HOU"),
    ("Golden State Warriors", 7, "GSW"),
]

TEAM_COLORS = {
    "Cleveland Cavaliers": ("#6f263d", "#fdbb30"),
    "Miami Heat": ("#98002e", "#f9a01b"),
    "Boston Celtics": ("#007a33", "#ba9653"),
    "Orlando Magic": ("#0077c0", "#c4ced4"),
    "New York Knicks": ("#006bb6", "#f58426"),
    "Detroit Pistons": ("#1d42ba", "#c8102e"),
    "Indiana Pacers": ("#002d62", "#fdbb30"),
    "Milwaukee Bucks": ("#00471b", "#eee1c6"),
    "Oklahoma City Thunder": ("#007ac1", "#ef3b24"),
    "Memphis Grizzlies": ("#5d76a9", "#12173f"),
    "Houston Rockets": ("#ce1141", "#c4ced4"),
    "Golden State Warriors": ("#1d428a", "#ffc72c"),
    "Los Angeles Lakers": ("#552583", "#fdb927"),
    "Minnesota Timberwolves": ("#0c2340", "#236192"),
    "Denver Nuggets": ("#0e2240", "#fec524"),
    "LA Clippers": ("#c8102e", "#1d428a"),
}

TEAM_ABBR = {name: abbr for name, _seed, abbr in LEFT_TEAMS + RIGHT_TEAMS}

ROUND1_WINNERS = [
    "Cleveland Cavaliers",
    "Indiana Pacers",
    "New York Knicks",
    "Boston Celtics",
    "Oklahoma City Thunder",
    "Denver Nuggets",
    "Minnesota Timberwolves",
    "Golden State Warriors",
]

SEMI_WINNERS = [
    "Indiana Pacers",
    "New York Knicks",
    "Oklahoma City Thunder",
    "Minnesota Timberwolves",
]

CONFERENCE_WINNERS = [
    "Indiana Pacers",
    "Oklahoma City Thunder",
]

CHAMPION = "Oklahoma City Thunder"

SCORES = {
    ("round1", "Cleveland Cavaliers"): "4-0",
    ("round1", "Boston Celtics"): "4-1",
    ("round1", "New York Knicks"): "4-2",
    ("round1", "Indiana Pacers"): "4-1",
    ("round1", "Oklahoma City Thunder"): "4-0",
    ("round1", "Golden State Warriors"): "4-1",
    ("round1", "Denver Nuggets"): "4-2",
    ("round1", "Minnesota Timberwolves"): "4-1",
    ("semi", "Indiana Pacers"): "4-1",
    ("semi", "New York Knicks"): "4-2",
    ("semi", "Oklahoma City Thunder"): "4-3",
    ("semi", "Minnesota Timberwolves"): "4-1",
    ("conf", "Indiana Pacers"): "4-2",
    ("conf", "Oklahoma City Thunder"): "4-1",
    ("final", CHAMPION): "4-3",
}

TEAM_STAGE_BASES_RAW = {
    "round1": 1.05,
    "semi": 7.05,
    "conf": 11.35,
    "final": FINAL_STAGE_RAW_BASE,
}

STAGE_BASES = {
    stage: max(0.0, (base - TEAM_TIMELINE_OFFSET) * TIMELINE_SCALE)
    for stage, base in TEAM_STAGE_BASES_RAW.items()
}


def _scaled_time(seconds: float) -> float:
    return seconds * TIMELINE_SCALE

SEED_Y = [216, 424, 634, 842, 1054, 1262, 1474, 1680]
ROUND1_Y = [292, 710, 1136, 1560]
SEMI_Y = [486, 1360]
CONF_Y = [962]
FINAL_CENTER = (540, 1016)
CHAMPION_CENTER = (540, 1214)

LEFT_SEED_X = 90
RIGHT_SEED_X = WIDTH - 90
LEFT_ROUND1_X = 250
RIGHT_ROUND1_X = 830
LEFT_SEMI_X = 350
RIGHT_SEMI_X = 730
LEFT_CONF_X = 470
RIGHT_CONF_X = 610

TITLE_BOX = (228, 18, 852, 182)
FINALS_BOX = (367, 924, 713, 1080)
CHAMPION_BOX = (389, 1108, 691, 1298)
CARD_W = 164
CARD_H = 58
SEED_LOGO_SIZE = 120
ROUND_LOGO_SIZE = 140
LOGO_RENDER_SCALE = 1.0
LOGO_INSET = 16


@dataclass(frozen=True)
class Slot:
    team: str
    seed: int
    abbr: str
    side: str
    seed_pos: tuple[int, int]
    round1_pos: tuple[int, int]
    semi_pos: tuple[int, int]
    conf_pos: tuple[int, int]


@dataclass(frozen=True)
class Move:
    team: str
    stage: str
    start: tuple[int, int]
    end: tuple[int, int]
    elbow_x: int
    delay: float
    duration: float
    score: str
    side: str


@dataclass(frozen=True)
class SegmentAnim:
    start: tuple[int, int]
    end: tuple[int, int]
    delay: float
    duration: float
    color: tuple[int, int, int, int]
    width: int


@dataclass(frozen=True)
class TeamEvent:
    stage: str
    start: float
    end: float
    source: tuple[int, int]
    dest: tuple[int, int] | None
    winner: bool
    score: str | None
    elbow_x: int


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return min(max(value, lo), hi)


def _ease_out(value: float) -> float:
    value = _clamp(value)
    return 1.0 - (1.0 - value) ** 3


def _ease_in_out(value: float) -> float:
    value = _clamp(value)
    return value * value * (3.0 - 2.0 * value)


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _hex_to_rgb(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    return tuple(int(value[i : i + 2], 16) for i in (0, 2, 4))


def _font_paths(bold: bool) -> list[Path]:
    if bold:
        return [
            Path("C:/Windows/Fonts/arialbd.ttf"),
            Path("C:/Windows/Fonts/segoeuib.ttf"),
            Path("C:/Windows/Fonts/calibrib.ttf"),
            Path("C:/Windows/Fonts/verdanab.ttf"),
        ]
    return [
        Path("C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/segoeui.ttf"),
        Path("C:/Windows/Fonts/calibri.ttf"),
        Path("C:/Windows/Fonts/verdana.ttf"),
    ]


@lru_cache(maxsize=32)
def _load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for candidate in _font_paths(bold):
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


@lru_cache(maxsize=16)
def _load_title_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for candidate in (
        Path("C:/Windows/Fonts/impact.ttf"),
        Path("C:/Windows/Fonts/bahnschrift.ttf"),
        Path("C:/Windows/Fonts/arialbd.ttf"),
        Path("C:/Windows/Fonts/segoeuib.ttf"),
    ):
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size=size)
    return _load_font(size, bold=True)


@lru_cache(maxsize=64)
def _load_logo(team: str, size: int) -> Image.Image:
    filename = team.lower().replace(" ", "_") + ".png"
    path = LOGO_DIR / filename
    if path.exists():
        logo = Image.open(path).convert("RGBA")
        logo = ImageOps.contain(logo, (size, size), method=Image.Resampling.LANCZOS)
        return logo

    primary, secondary = TEAM_COLORS.get(team, ("#d1d5db", "#111111"))
    primary_rgb = _hex_to_rgb(primary)
    secondary_rgb = _hex_to_rgb(secondary)
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img, "RGBA")
    draw.ellipse((0, 0, size - 1, size - 1), fill=(*secondary_rgb, 210), outline=(*primary_rgb, 230), width=3)
    draw.ellipse((6, 6, size - 7, size - 7), outline=(255, 255, 255, 38), width=1)
    abbr = TEAM_ABBR.get(team, "".join(part[0] for part in team.split()[:3]).upper())
    font = _load_font(max(12, int(size * 0.34)), bold=True)
    draw.text((size // 2, size // 2), abbr, font=font, fill="#f7f8fb", anchor="mm")
    return img


@lru_cache(maxsize=4)
def _load_nba_logo(size: int) -> Image.Image | None:
    if not NBA_LOGO_PATH.exists():
        return None
    logo = Image.open(NBA_LOGO_PATH).convert("RGBA")
    logo = ImageOps.contain(logo, (size, size), method=Image.Resampling.LANCZOS)
    return _colorize_nba_logo(logo)


def _colorize_nba_logo(logo: Image.Image) -> Image.Image:
    rgba = logo.convert("RGBA")
    arr = np.asarray(rgba).astype(np.float32)
    rgb = arr[:, :, :3]
    alpha = arr[:, :, 3]
    visible = alpha > 0
    if not np.any(visible):
        return rgba

    # If the asset is already in color, leave it alone.
    channel_spread = np.std(rgb[visible], axis=1).mean()
    if channel_spread > 7.5:
        return rgba

    luminance = rgb.mean(axis=2)
    white_mask = visible & (luminance >= 240.0)
    bg_mask = visible & ~white_mask
    if not np.any(bg_mask):
        return rgba

    x = np.linspace(0.0, 1.0, rgba.width, dtype=np.float32)[None, :, None]
    blue = np.array([24.0, 78.0, 189.0], dtype=np.float32)
    red = np.array([212.0, 43.0, 46.0], dtype=np.float32)
    gradient = blue * (1.0 - x) + red * x
    gradient = np.broadcast_to(gradient, rgb.shape)

    shade = 0.90 + 0.10 * (luminance[..., None] / 255.0)
    rgb[bg_mask] = np.clip(gradient[bg_mask] * shade[bg_mask], 0, 255)
    rgb[white_mask] = 255.0

    return Image.fromarray(np.dstack([rgb.astype(np.uint8), alpha.astype(np.uint8)]))


@lru_cache(maxsize=8)
def _load_trophy_photo(width: int) -> Image.Image | None:
    source_path = TROPHY_PHOTO_PATH if TROPHY_PHOTO_PATH.exists() else LEGACY_TROPHY_PHOTO_PATH
    if not source_path.exists():
        return None

    trophy = Image.open(source_path).convert("RGBA")
    if source_path == TROPHY_PHOTO_PATH:
        alpha = trophy.getchannel("A")
        if alpha.getextrema() == (255, 255):
            trophy = trophy.copy()
            trophy.putalpha(_build_trophy_png_mask(trophy))
        alpha_bbox = trophy.getchannel("A").getbbox()
        if alpha_bbox and alpha_bbox != (0, 0, trophy.width, trophy.height):
            trophy = trophy.crop(alpha_bbox)
        target_height = max(1, int(width * trophy.height / trophy.width))
        trophy = ImageOps.contain(trophy, (width, target_height), method=Image.Resampling.LANCZOS)
        trophy = ImageEnhance.Color(trophy).enhance(0.88)
        trophy = ImageEnhance.Contrast(trophy).enhance(1.04)
        return trophy

    trim = int(trophy.width * 0.14)
    trophy = trophy.crop((trim, 0, trophy.width - trim, trophy.height))

    arr = np.asarray(trophy).astype(np.float32)
    h, w = arr.shape[:2]

    # Key the blue backdrop, then tighten it with a soft matte around the trophy silhouette.
    samples = np.concatenate(
        [
            arr[0:45, 0:35, :3].reshape(-1, 3),
            arr[0:45, -35:, :3].reshape(-1, 3),
        ],
        axis=0,
    )
    ref = np.median(samples, axis=0)
    dist = np.sqrt(((arr[:, :, :3] - ref) ** 2).sum(axis=2))
    key_alpha = np.clip((dist - 48.0) / 88.0, 0.0, 1.0) ** 1.25

    matte = Image.new("L", (w, h), 0)
    md = ImageDraw.Draw(matte, "L")
    md.ellipse((w * 0.44, h * 0.11, w * 0.95, h * 0.48), fill=255)
    md.rounded_rectangle((w * 0.44, h * 0.30, w * 0.80, h * 0.80), radius=max(4, int(w * 0.10)), fill=255)
    md.rectangle((w * 0.52, h * 0.67, w * 0.60, h * 0.87), fill=255)
    md.rounded_rectangle((w * 0.29, h * 0.84, w * 0.85, h * 0.97), radius=max(3, int(w * 0.04)), fill=255)
    md.polygon(
        [
            (w * 0.54, h * 0.44),
            (w * 0.68, h * 0.44),
            (w * 0.67, h * 0.57),
            (w * 0.55, h * 0.57),
        ],
        fill=255,
    )
    matte = matte.filter(ImageFilter.GaussianBlur(8))
    matte_alpha = np.asarray(matte).astype(np.float32) / 255.0

    yy, xx = np.mgrid[0:h, 0:w]
    cx, cy = w * 0.56, h * 0.46
    radial = np.sqrt(((xx - cx) / (w * 0.42)) ** 2 + ((yy - cy) / (h * 0.52)) ** 2)
    radial_alpha = np.clip(1.12 - radial, 0.0, 1.0) ** 1.65

    alpha = np.clip(np.maximum(matte_alpha * key_alpha, radial_alpha * 0.80), 0.0, 1.0)

    rgb = Image.fromarray(np.clip(arr[:, :, :3], 0, 255).astype(np.uint8), "RGB")
    rgb = ImageEnhance.Color(rgb).enhance(0.82)
    rgb = ImageEnhance.Contrast(rgb).enhance(1.06)
    rgb = rgb.convert("RGBA")
    rgb.putalpha(Image.fromarray((alpha * 255).astype(np.uint8), "L"))

    target_height = max(1, int(width * rgb.height / rgb.width))
    return ImageOps.contain(rgb, (width, target_height), method=Image.Resampling.LANCZOS)


def _build_slots() -> dict[str, Slot]:
    slots: dict[str, Slot] = {}
    for (team, seed, abbr), y in zip(LEFT_TEAMS, SEED_Y):
        slots[team] = Slot(
            team=team,
            seed=seed,
            abbr=abbr,
            side="left",
            seed_pos=(LEFT_SEED_X, y),
            round1_pos=(LEFT_ROUND1_X, ROUND1_Y[SEED_Y.index(y) // 2]),
            semi_pos=(LEFT_SEMI_X, SEMI_Y[SEED_Y.index(y) // 4]),
            conf_pos=(LEFT_CONF_X, CONF_Y[0]),
        )
    for (team, seed, abbr), y in zip(RIGHT_TEAMS, SEED_Y):
        slots[team] = Slot(
            team=team,
            seed=seed,
            abbr=abbr,
            side="right",
            seed_pos=(RIGHT_SEED_X, y),
            round1_pos=(RIGHT_ROUND1_X, ROUND1_Y[SEED_Y.index(y) // 2]),
            semi_pos=(RIGHT_SEMI_X, SEMI_Y[SEED_Y.index(y) // 4]),
            conf_pos=(RIGHT_CONF_X, CONF_Y[0]),
        )
    return slots


SLOTS = _build_slots()
TEAM_ORDER = tuple(SLOTS.keys())


def _build_moves() -> dict[str, list[Move]]:
    moves: dict[str, list[Move]] = {"round1": [], "semi": [], "conf": [], "final": []}

    round1_order = [
        "Cleveland Cavaliers",
        "Indiana Pacers",
        "New York Knicks",
        "Boston Celtics",
        "Oklahoma City Thunder",
        "Denver Nuggets",
        "Minnesota Timberwolves",
        "Golden State Warriors",
    ]
    round1_delay = 0.0
    for team in round1_order:
        slot = SLOTS[team]
        moves["round1"].append(
            Move(
                team=team,
                stage="round1",
                start=slot.seed_pos,
                end=slot.round1_pos,
                elbow_x=190 if slot.side == "left" else WIDTH - 190,
                delay=_scaled_time(round1_delay),
                duration=_scaled_time(0.68),
                score=SCORES[("round1", team)],
                side=slot.side,
            )
        )
        round1_delay += 0.68

    semi_order = ["Indiana Pacers", "New York Knicks", "Oklahoma City Thunder", "Minnesota Timberwolves"]
    semi_delay = 0.0
    for team in semi_order:
        slot = SLOTS[team]
        moves["semi"].append(
            Move(
                team=team,
                stage="semi",
                start=slot.round1_pos,
                end=slot.semi_pos,
                elbow_x=300 if slot.side == "left" else WIDTH - 300,
                delay=_scaled_time(semi_delay),
                duration=_scaled_time(0.8),
                score=SCORES[("semi", team)],
                side=slot.side,
            )
        )
        semi_delay += 0.78

    conf_order = ["Indiana Pacers", "Oklahoma City Thunder"]
    conf_delay = 0.0
    for team in conf_order:
        slot = SLOTS[team]
        moves["conf"].append(
            Move(
                team=team,
                stage="conf",
                start=slot.semi_pos,
                end=slot.conf_pos,
                elbow_x=392 if slot.side == "left" else WIDTH - 392,
                delay=_scaled_time(conf_delay),
                duration=_scaled_time(0.9),
                score=SCORES[("conf", team)],
                side=slot.side,
            )
        )
        conf_delay += 0.92

    moves["final"].append(
        Move(
            team=CHAMPION,
            stage="final",
            start=(RIGHT_CONF_X, CONF_Y[0]),
            end=FINAL_CENTER,
            elbow_x=540,
            delay=_scaled_time(0.25),
            duration=_scaled_time(0.8),
            score=SCORES[("final", CHAMPION)],
            side="center",
        )
    )
    return moves


MOVES = _build_moves()

STAGE_ORDER = ("round1", "semi", "conf", "final")
STAGE_MATCHUPS = {
    "round1": [
        ("Cleveland Cavaliers", "Miami Heat", "Cleveland Cavaliers"),
        ("Indiana Pacers", "Milwaukee Bucks", "Indiana Pacers"),
        ("New York Knicks", "Detroit Pistons", "New York Knicks"),
        ("Boston Celtics", "Orlando Magic", "Boston Celtics"),
        ("Oklahoma City Thunder", "Memphis Grizzlies", "Oklahoma City Thunder"),
        ("Denver Nuggets", "LA Clippers", "Denver Nuggets"),
        ("Los Angeles Lakers", "Minnesota Timberwolves", "Minnesota Timberwolves"),
        ("Houston Rockets", "Golden State Warriors", "Golden State Warriors"),
    ],
    "semi": [
        ("Cleveland Cavaliers", "Indiana Pacers", "Indiana Pacers"),
        ("Boston Celtics", "New York Knicks", "New York Knicks"),
        ("Oklahoma City Thunder", "Denver Nuggets", "Oklahoma City Thunder"),
        ("Minnesota Timberwolves", "Golden State Warriors", "Minnesota Timberwolves"),
    ],
    "conf": [
        ("Indiana Pacers", "New York Knicks", "Indiana Pacers"),
        ("Oklahoma City Thunder", "Minnesota Timberwolves", "Oklahoma City Thunder"),
    ],
    "final": [
        ("Indiana Pacers", "Oklahoma City Thunder", "Oklahoma City Thunder"),
    ],
}


def _seed_logo_center(team: str) -> tuple[int, int]:
    slot = SLOTS[team]
    x_offset = 18 if slot.side == "left" else -18
    return slot.seed_pos[0] + x_offset, slot.seed_pos[1]


def _stage_source_pos(team: str, stage: str) -> tuple[int, int]:
    slot = SLOTS[team]
    if stage == "round1":
        return _seed_logo_center(team)
    if stage == "semi":
        return slot.round1_pos
    if stage == "conf":
        return slot.semi_pos
    if stage == "final":
        return slot.conf_pos
    raise ValueError(f"Unknown stage: {stage}")


def _stage_dest_pos(team: str, stage: str) -> tuple[int, int]:
    slot = SLOTS[team]
    if stage == "round1":
        return slot.round1_pos
    if stage == "semi":
        return slot.semi_pos
    if stage == "conf":
        return slot.conf_pos
    if stage == "final":
        return FINAL_CENTER
    raise ValueError(f"Unknown stage: {stage}")


def _path_position(points: list[tuple[int, int]], progress: float) -> tuple[int, int]:
    progress = _clamp(progress)
    if not points:
        return 0, 0
    if len(points) == 1 or progress <= 0.0:
        return points[0]
    total = 0.0
    lengths: list[float] = []
    for a, b in zip(points, points[1:]):
        length = float(np.hypot(b[0] - a[0], b[1] - a[1]))
        lengths.append(length)
        total += length
    if total <= 0.0:
        return points[-1]
    remaining = total * progress
    for (a, b), length in zip(zip(points, points[1:]), lengths):
        if remaining <= length:
            local = remaining / length if length > 0.0 else 1.0
            x = int(_lerp(a[0], b[0], local))
            y = int(_lerp(a[1], b[1], local))
            return x, y
        remaining -= length
    return points[-1]


def _build_team_events() -> dict[str, list[TeamEvent]]:
    move_lookup = {stage: {move.team: move for move in moves} for stage, moves in MOVES.items()}
    events: dict[str, list[TeamEvent]] = {team: [] for team in SLOTS}
    for stage in STAGE_ORDER:
        base = STAGE_BASES[stage]
        for team_a, team_b, winner in STAGE_MATCHUPS[stage]:
            move = move_lookup[stage][winner]
            start = base + move.delay
            end = start + move.duration
            source = _stage_source_pos(winner, stage)
            dest = _stage_dest_pos(winner, stage)
            score = SCORES.get((stage, winner))
            for team in (team_a, team_b):
                events[team].append(
                    TeamEvent(
                        stage=stage,
                        start=start,
                        end=end,
                        source=source,
                        dest=dest if team == winner else None,
                        winner=team == winner,
                        score=score if team == winner else None,
                        elbow_x=move.elbow_x,
                    )
                )
    return events


TEAM_EVENTS = _build_team_events()


def _center_of(pos: tuple[int, int]) -> tuple[int, int]:
    return pos


def _path_points(start: tuple[int, int], end: tuple[int, int], elbow_x: int) -> list[tuple[int, int]]:
    return [start, (elbow_x, start[1]), (elbow_x, end[1]), end]


def _draw_partial_path(draw: ImageDraw.ImageDraw, points: list[tuple[int, int]], progress: float, color: tuple[int, int, int, int], width: int) -> None:
    progress = _clamp(progress)
    if progress <= 0.0:
        return
    segments = []
    lengths: list[float] = []
    total = 0.0
    for a, b in zip(points, points[1:]):
        length = float(np.hypot(b[0] - a[0], b[1] - a[1]))
        segments.append((a, b, length))
        lengths.append(length)
        total += length
    if total <= 0.0:
        return

    remaining = total * progress
    for a, b, length in segments:
        if remaining >= length:
            draw.line((a, b), fill=color, width=width)
            remaining -= length
            continue
        if remaining <= 0.0:
            break
        local = remaining / length
        x = int(_lerp(a[0], b[0], local))
        y = int(_lerp(a[1], b[1], local))
        draw.line((a, (x, y)), fill=color, width=width)
        break


def _draw_glow(frame: Image.Image, center: tuple[int, int], color: tuple[int, int, int], radius: int, alpha: int) -> None:
    glow = Image.new("RGBA", (radius * 4, radius * 4), (0, 0, 0, 0))
    draw = ImageDraw.Draw(glow, "RGBA")
    draw.ellipse((radius // 2, radius // 2, radius * 3.5, radius * 3.5), fill=(*color, alpha))
    glow = glow.filter(ImageFilter.GaussianBlur(radius=max(6, radius // 3)))
    frame.alpha_composite(glow, (int(center[0] - glow.width // 2), int(center[1] - glow.height // 2)))


def _build_trophy_png_mask(source: Image.Image) -> Image.Image:
    rgba = source.convert("RGBA")
    arr = np.asarray(rgba)
    rgb = arr[:, :, :3].astype(np.int16)
    h, w = rgb.shape[:2]

    edge_pixels = np.concatenate(
        [
            rgb[0:25, :, :].reshape(-1, 3),
            rgb[-25:, :, :].reshape(-1, 3),
            rgb[:, 0:25, :].reshape(-1, 3),
            rgb[:, -25:, :].reshape(-1, 3),
        ],
        axis=0,
    )
    edge_pixels_u8 = edge_pixels.astype(np.uint8)
    values, counts = np.unique(edge_pixels_u8, axis=0, return_counts=True)
    bg_colors = values[np.argsort(counts)[::-1][:2]].astype(np.float32)
    dist = np.minimum.reduce([np.sqrt(((rgb - color) ** 2).sum(axis=2)) for color in bg_colors])

    # Remove the light studio background while keeping the trophy's white
    # reflections intact. The source uses two checker-like background tones.
    candidate_bg = dist <= 32.0
    bg_mask = np.zeros((h, w), dtype=bool)
    queue: deque[tuple[int, int]] = deque()

    def seed(y: int, x: int) -> None:
        if candidate_bg[y, x] and not bg_mask[y, x]:
            bg_mask[y, x] = True
            queue.append((y, x))

    for x in range(w):
        seed(0, x)
        seed(h - 1, x)
    for y in range(h):
        seed(y, 0)
        seed(y, w - 1)

    while queue:
        y, x = queue.popleft()
        ny = y - 1
        if ny >= 0 and candidate_bg[ny, x] and not bg_mask[ny, x]:
            bg_mask[ny, x] = True
            queue.append((ny, x))
        ny = y + 1
        if ny < h and candidate_bg[ny, x] and not bg_mask[ny, x]:
            bg_mask[ny, x] = True
            queue.append((ny, x))
        nx = x - 1
        if nx >= 0 and candidate_bg[y, nx] and not bg_mask[y, nx]:
            bg_mask[y, nx] = True
            queue.append((y, nx))
        nx = x + 1
        if nx < w and candidate_bg[y, nx] and not bg_mask[y, nx]:
            bg_mask[y, nx] = True
            queue.append((y, nx))

    alpha = np.where(bg_mask, 0, 255).astype(np.uint8)
    return Image.fromarray(alpha).filter(ImageFilter.GaussianBlur(1.0))


def _draw_background_trophy(frame: Image.Image, t: float) -> None:
    wobble = math.sin(t * 0.85) * 7.0
    pulse = 1.0 + 0.018 * math.sin(t * 0.55)
    trophy = _load_trophy_photo(284)

    if trophy is None:
        # Fallback if the photo asset is missing.
        w, h = 210, 300
        trophy_layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(trophy_layer, "RGBA")

        gold = (248, 188, 67, 170)
        bright = (255, 233, 155, 210)
        dark = (120, 82, 22, 210)
        shadow = (38, 24, 10, 175)

        draw.ellipse((55, 14, w - 55, 92), fill=gold, outline=bright, width=3)
        draw.rounded_rectangle((36, 86, w - 36, 184), radius=28, fill=(236, 177, 56, 180), outline=bright, width=3)
        draw.ellipse((16, 60, 64, 148), fill=gold, outline=bright, width=3)
        draw.ellipse((w - 64, 60, w - 16, 148), fill=gold, outline=bright, width=3)
        draw.rectangle((w // 2 - 15, 164, w // 2 + 15, 226), fill=dark)
        draw.rounded_rectangle((w // 2 - 82, 224, w // 2 + 82, 268), radius=12, fill=shadow)
        draw.rounded_rectangle((w // 2 - 102, 266, w // 2 + 102, 300), radius=14, fill=dark, outline=bright, width=2)
        draw.line((w // 2, 18, w // 2, 78), fill=(255, 255, 255, 110), width=4)
        trophy = trophy_layer.filter(ImageFilter.GaussianBlur(radius=0.2))
    else:
        if abs(pulse - 1.0) > 1e-3:
            scaled_w = max(1, int(round(trophy.width * pulse)))
            scaled_h = max(1, int(round(trophy.height * pulse)))
            trophy = trophy.resize((scaled_w, scaled_h), Image.Resampling.LANCZOS)

    glow = Image.new("RGBA", (trophy.width + 180, trophy.height + 240), (0, 0, 0, 0))
    gd = ImageDraw.Draw(glow, "RGBA")
    gd.ellipse((20, 50, glow.width - 20, glow.height - 18), fill=(255, 201, 92, 62))
    gd.ellipse((54, 96, glow.width - 54, glow.height - 64), fill=(255, 233, 168, 24))
    glow = glow.filter(ImageFilter.GaussianBlur(radius=34))
    glow_x = int((WIDTH - glow.width) // 2)
    glow_y = int(555 + wobble - 18)
    frame.alpha_composite(glow, (glow_x, glow_y))

    x = int((WIDTH - trophy.width) // 2)
    y = int(586 + wobble)
    frame.alpha_composite(trophy, (x, y))


def _draw_segment(draw: ImageDraw.ImageDraw, start: tuple[int, int], end: tuple[int, int], color: tuple[int, int, int, int], width: int) -> None:
    draw.line((start, end), fill=color, width=width)


def _draw_segment_progress(
    draw: ImageDraw.ImageDraw,
    start: tuple[int, int],
    end: tuple[int, int],
    progress: float,
    color: tuple[int, int, int, int],
    width: int,
) -> None:
    progress = _clamp(progress)
    if progress <= 0.0:
        return
    x = _lerp(start[0], end[0], progress)
    y = _lerp(start[1], end[1], progress)
    _draw_segment(draw, start, (int(x), int(y)), color, width)


def _background_frame(t: float) -> Image.Image:
    xx = np.linspace(0, 1, WIDTH, dtype=np.float32)
    yy = np.linspace(0, 1, HEIGHT, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(xx, yy)

    blue = np.array([20, 78, 190], dtype=np.float32)
    red = np.array([164, 38, 62], dtype=np.float32)
    black = np.array([2, 4, 10], dtype=np.float32)
    gold = np.array([216, 166, 72], dtype=np.float32)
    purple = np.array([55, 20, 55], dtype=np.float32)

    left_glow = np.exp(-(((grid_x - 0.10) / 0.18) ** 2 + ((grid_y - 0.28) / 0.22) ** 2))
    right_glow = np.exp(-(((grid_x - 0.90) / 0.18) ** 2 + ((grid_y - 0.28) / 0.22) ** 2))
    center_glow = np.exp(-(((grid_x - 0.50) / 0.22) ** 2 + ((grid_y - 0.48) / 0.36) ** 2))
    top_halo = np.exp(-(((grid_x - 0.50) / 0.28) ** 2 + ((grid_y - 0.08) / 0.10) ** 2))
    floor = np.clip((grid_y - 0.81) / 0.19, 0.0, 1.0)
    scan = (np.sin((grid_x * 104.0) + t * 1.6) ** 2) * floor * 0.08
    grain = (((np.sin(grid_x * 83.0 + t * 0.7) + 1.0) * (np.sin(grid_y * 91.0 + t * 0.9) + 1.0)) * 0.004)

    img = np.clip(
        black[None, None, :] * (1.0 - 0.38 * floor[..., None])
        + purple[None, None, :] * (0.40 * center_glow[..., None])
        + blue[None, None, :] * (0.60 * left_glow[..., None] + 0.14 * top_halo[..., None])
        + red[None, None, :] * (0.60 * right_glow[..., None] + 0.14 * top_halo[..., None])
        + gold[None, None, :] * (0.10 * center_glow[..., None] + 0.08 * top_halo[..., None])
        + np.array([255, 255, 255], dtype=np.float32)[None, None, :] * (scan[..., None] + grain[..., None]),
        0,
        255,
    ).astype(np.uint8)
    frame = Image.fromarray(img).convert("RGBA")

    overlay = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")
    draw.rounded_rectangle((18, 16, WIDTH - 18, HEIGHT - 16), radius=36, outline=(255, 255, 255, 12), width=2)
    draw.ellipse((-120, 120, 320, 520), fill=(34, 120, 255, 28))
    draw.ellipse((760, 120, 1200, 520), fill=(255, 62, 80, 28))
    draw.rectangle((0, HEIGHT - 86, WIDTH, HEIGHT), fill=(0, 0, 0, 94))
    for y in (1750, 1800):
        draw.line((0, y, WIDTH, y), fill=(255, 200, 120, 18), width=3 if y == 1750 else 2)
    overlay = overlay.filter(ImageFilter.GaussianBlur(radius=2))
    frame.alpha_composite(overlay)
    return frame


def _draw_title_panel(frame: Image.Image, t: float) -> None:
    draw = ImageDraw.Draw(frame, "RGBA")
    title_alpha = int(255 * _ease_in_out(_clamp((t - 0.08) / 0.75)))
    panel = Image.new("RGBA", (TITLE_BOX[2] - TITLE_BOX[0], TITLE_BOX[3] - TITLE_BOX[1]), (0, 0, 0, 0))
    pd = ImageDraw.Draw(panel, "RGBA")
    pd.rounded_rectangle((0, 0, panel.width - 1, panel.height - 1), radius=30, fill=(12, 12, 16, 225), outline=(255, 255, 255, 18), width=2)
    pd.rounded_rectangle((15, 14, panel.width - 16, panel.height - 16), radius=26, fill=(8, 8, 12, 170))
    pd.polygon([(34, 16), (184, 16), (152, 145), (4, 145)], fill=(255, 255, 255, 10))
    panel = panel.filter(ImageFilter.GaussianBlur(radius=0.2))
    frame.alpha_composite(panel, (TITLE_BOX[0], TITLE_BOX[1]))

    nba_logo = _load_nba_logo(156)
    badge = Image.new("RGBA", (92, 154), (0, 0, 0, 0))
    bd = ImageDraw.Draw(badge, "RGBA")
    bd.rounded_rectangle((0, 0, 91, 153), radius=16, fill=(21, 82, 189, 255), outline=(255, 255, 255, 190), width=3)
    bd.rectangle((35, 8, 57, 145), fill=(255, 255, 255, 255))
    bd.polygon([(35, 8), (56, 8), (56, 145), (35, 145)], fill=(224, 44, 41, 255))
    if nba_logo is not None:
        if title_alpha < 255:
            nba_logo = nba_logo.copy()
            channel = nba_logo.getchannel("A").point(lambda a: int(a * (title_alpha / 255.0)))
            nba_logo.putalpha(channel)
        badge.alpha_composite(nba_logo, ((badge.width - nba_logo.width) // 2, (badge.height - nba_logo.height) // 2))
    else:
        bd.text((46, 126), "NBA", font=_load_font(18, bold=True), fill="#f7f8fb", anchor="mm")

    if title_alpha < 255:
        badge = badge.copy()
        badge.putalpha(badge.getchannel("A").point(lambda a: int(a * (title_alpha / 255.0))))

    shadow = Image.new("RGBA", (badge.width + 38, badge.height + 38), (0, 0, 0, 0))
    sd = ImageDraw.Draw(shadow, "RGBA")
    sd.rounded_rectangle((10, 10, shadow.width - 11, shadow.height - 11), radius=18, fill=(0, 0, 0, 120))
    shadow = shadow.filter(ImageFilter.GaussianBlur(radius=10))
    frame.alpha_composite(shadow, (238, 14))
    frame.alpha_composite(badge, (244, 26))

    title_font = _load_title_font(70)
    year_font = _load_font(22, bold=True)
    text_x = 611
    draw.text(
        (text_x, 76),
        "PLAYOFFS",
        font=title_font,
        fill=(247, 247, 245, title_alpha),
        anchor="mm",
        stroke_width=9,
        stroke_fill=(0, 0, 0, title_alpha),
    )
    draw.text((text_x + 2, 120), "2025", font=year_font, fill=(255, 213, 125, title_alpha), anchor="mm")
    draw.line((470, 136, 764, 136), fill=(255, 205, 96, int(title_alpha * 0.45)), width=3)
    draw.line((492, 140, 742, 140), fill=(255, 255, 255, int(title_alpha * 0.10)), width=1)


def _draw_side_labels(frame: Image.Image, t: float) -> None:
    draw = ImageDraw.Draw(frame, "RGBA")
    label_font = _load_font(18, bold=True)
    small_font = _load_font(15, bold=True)
    alpha = int(190 * _ease_in_out(_clamp((t - 0.8) / 0.8)))
    left = (245, 110, 255, alpha)
    right = (245, 110, 255, alpha)
    draw.text((110, 1220), "CONFERENCE", font=label_font, fill=left)
    draw.text((110, 1246), "SEMIFINALS", font=label_font, fill=left)
    draw.text((WIDTH - 110, 1220), "CONFERENCE", font=label_font, fill=right, anchor="ra")
    draw.text((WIDTH - 110, 1246), "SEMIFINALS", font=label_font, fill=right, anchor="ra")
    draw.text((110, 1600), "CONFERENCE", font=small_font, fill=left)
    draw.text((110, 1624), "FINALS", font=small_font, fill=left)
    draw.text((WIDTH - 110, 1600), "CONFERENCE", font=small_font, fill=right, anchor="ra")
    draw.text((WIDTH - 110, 1624), "FINALS", font=small_font, fill=right, anchor="ra")


def _draw_finals_box(frame: Image.Image, t: float, highlight: bool = False) -> None:
    draw = ImageDraw.Draw(frame, "RGBA")
    alpha = int(230 * _ease_in_out(_clamp((t - 1.6) / 0.8)))
    shadow = Image.new("RGBA", (FINALS_BOX[2] - FINALS_BOX[0] + 44, FINALS_BOX[3] - FINALS_BOX[1] + 44), (0, 0, 0, 0))
    sd = ImageDraw.Draw(shadow, "RGBA")
    sd.rounded_rectangle((18, 18, shadow.width - 19, shadow.height - 19), radius=26, fill=(0, 0, 0, 124))
    shadow = shadow.filter(ImageFilter.GaussianBlur(radius=12))
    frame.alpha_composite(shadow, (FINALS_BOX[0] - 22, FINALS_BOX[1] - 22))

    draw.rounded_rectangle(FINALS_BOX, radius=22, fill=(8, 8, 10, 230), outline=(255, 197, 96, alpha), width=3)
    draw.rounded_rectangle((FINALS_BOX[0] + 14, FINALS_BOX[1] + 12, FINALS_BOX[2] - 14, FINALS_BOX[3] - 14), radius=16, fill=(0, 0, 0, 120))
    font = _load_font(28, bold=True)
    draw.text(((FINALS_BOX[0] + FINALS_BOX[2]) // 2, FINALS_BOX[1] + 58), "NBA FINALS", font=font, fill=(246, 246, 244, alpha), anchor="mm")
    if highlight:
        glow = Image.new("RGBA", (FINALS_BOX[2] - FINALS_BOX[0] + 160, FINALS_BOX[3] - FINALS_BOX[1] + 160), (0, 0, 0, 0))
        gd = ImageDraw.Draw(glow, "RGBA")
        gd.rounded_rectangle((48, 48, glow.width - 49, glow.height - 49), radius=34, fill=(255, 204, 96, 54))
        glow = glow.filter(ImageFilter.GaussianBlur(radius=18))
        frame.alpha_composite(glow, (FINALS_BOX[0] - 80, FINALS_BOX[1] - 80))


def _draw_champion_box(frame: Image.Image, t: float, start_time: float) -> None:
    progress = _clamp((t - start_time) / 0.8)
    alpha = int(255 * _ease_in_out(progress))
    if alpha <= 0:
        return

    box = CHAMPION_BOX
    shadow = Image.new("RGBA", (box[2] - box[0] + 52, box[3] - box[1] + 52), (0, 0, 0, 0))
    sd = ImageDraw.Draw(shadow, "RGBA")
    sd.rounded_rectangle((20, 20, shadow.width - 21, shadow.height - 21), radius=28, fill=(0, 0, 0, 150))
    shadow = shadow.filter(ImageFilter.GaussianBlur(radius=14))
    frame.alpha_composite(shadow, (box[0] - 26, box[1] - 26))

    draw = ImageDraw.Draw(frame, "RGBA")
    draw.rounded_rectangle(box, radius=26, fill=(10, 10, 14, 242), outline=(255, 207, 104, alpha), width=4)
    draw.rounded_rectangle((box[0] + 12, box[1] + 12, box[2] - 12, box[3] - 12), radius=20, fill=(0, 0, 0, 120))

    center_x = (box[0] + box[2]) // 2
    top_y = box[1] + 30
    title_font = _load_font(22, bold=True)
    subtitle_font = _load_font(18, bold=True)
    small_font = _load_font(16, bold=True)

    draw.text((center_x, top_y), "NBA CHAMPIONS", font=title_font, fill=(255, 242, 184, alpha), anchor="mm")
    draw.line((box[0] + 28, box[1] + 56, box[2] - 28, box[1] + 56), fill=(255, 210, 96, int(alpha * 0.65)), width=2)

    glow = Image.new("RGBA", (136, 136), (0, 0, 0, 0))
    gd = ImageDraw.Draw(glow, "RGBA")
    gd.ellipse((14, 14, 122, 122), fill=(255, 210, 104, 72))
    glow = glow.filter(ImageFilter.GaussianBlur(radius=18))
    frame.alpha_composite(glow, (center_x - 68, box[1] + 58))

    logo = _load_logo(CHAMPION, 100)
    frame.alpha_composite(logo, (center_x - logo.width // 2, box[1] + 68))

    draw.text((center_x, box[3] - 46), CHAMPION.upper(), font=subtitle_font, fill=(255, 236, 178, alpha), anchor="mm")
    draw.text((center_x, box[3] - 20), "2025", font=small_font, fill=(255, 210, 96, alpha), anchor="mm")


@lru_cache(maxsize=32)
def _team_card(team: str, seed: int, side: str) -> Image.Image:
    width = 164
    height = 58
    seed_w = 38
    accent = _hex_to_rgb(TEAM_COLORS.get(team, ("#d1d5db", "#111111"))[0])
    body = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(body, "RGBA")
    draw.rounded_rectangle((2, 2, width - 3, height - 3), radius=7, fill=(0, 0, 0, 112))
    draw.rounded_rectangle((0, 0, width - 1, height - 1), radius=7, fill=(18, 18, 20, 245), outline=(255, 255, 255, 72), width=2)
    if side == "left":
        draw.rounded_rectangle((0, 0, seed_w, height - 1), radius=7, fill=(239, 239, 239, 255))
        draw.rectangle((seed_w - 1, 0, width - 1, 8), fill=(*accent, 255))
    else:
        draw.rounded_rectangle((width - seed_w, 0, width - 1, height - 1), radius=7, fill=(239, 239, 239, 255))
        draw.rectangle((0, 0, width - seed_w, 8), fill=(*accent, 255))

    logo = _load_logo(team, 32)
    if logo.width > 32 or logo.height > 32:
        logo = ImageOps.contain(logo, (32, 32), method=Image.Resampling.LANCZOS)
    if side == "left":
        body.alpha_composite(logo, (seed_w + 10, (height - logo.height) // 2 + 1))
    else:
        body.alpha_composite(logo, (18, (height - logo.height) // 2 + 1))

    seed_font = _load_font(24, bold=True)
    draw.text((seed_w // 2 if side == "left" else width - seed_w // 2, height // 2 + 1), str(seed), font=seed_font, fill="#111111", anchor="mm")
    return body


@lru_cache(maxsize=64)
def _team_square_logo(team: str, size: int = 52) -> Image.Image:
    inner_size = max(1, int((size - LOGO_INSET) * LOGO_RENDER_SCALE))
    logo = _load_logo(team, inner_size)
    tile = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(tile, "RGBA")
    accent = _hex_to_rgb(TEAM_COLORS.get(team, ("#d1d5db", "#111111"))[0])
    draw.rounded_rectangle((1, 1, size - 2, size - 2), radius=10, fill=(13, 13, 16, 245), outline=(*accent, 210), width=2)
    tile.alpha_composite(logo, ((size - logo.width) // 2, (size - logo.height) // 2))
    return tile


def _draw_card(
    frame: Image.Image,
    card: Image.Image,
    x: float,
    y: float,
    alpha: float,
    scale: float = 1.0,
    glow: bool = False,
    glow_color: tuple[int, int, int] = (255, 214, 116),
) -> None:
    alpha = _clamp(alpha)
    scale = max(0.85, scale)
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
        halo = Image.new("RGBA", (image.width + 88, image.height + 88), (0, 0, 0, 0))
        hd = ImageDraw.Draw(halo, "RGBA")
        hd.rounded_rectangle((26, 26, halo.width - 27, halo.height - 27), radius=18, fill=(*glow_color, 92))
        halo = halo.filter(ImageFilter.GaussianBlur(radius=18))
        frame.alpha_composite(halo, (px - 44, py - 44))
    frame.alpha_composite(image, (px, py))


def _draw_score_badge(frame: Image.Image, text: str, x: float, y: float, alpha: int = 255) -> None:
    badge = Image.new("RGBA", (176, 60), (0, 0, 0, 0))
    draw = ImageDraw.Draw(badge, "RGBA")
    draw.rounded_rectangle((0, 0, 175, 59), radius=22, fill=(255, 214, 54, alpha), outline=(255, 250, 202, 135), width=3)
    draw.rounded_rectangle((8, 8, 167, 51), radius=17, fill=(255, 238, 153, int(alpha * 0.28)))
    draw.text(
        (88, 30),
        text,
        font=_load_font(38, bold=True),
        fill=(255, 231, 106, alpha),
        anchor="mm",
        stroke_width=4,
        stroke_fill=(28, 18, 4, alpha),
    )
    frame.alpha_composite(badge, (int(x - 88), int(y - 30)))


def _draw_seed_number(frame: Image.Image, x: float, y: float, seed: int, side: str, alpha: int = 255) -> None:
    panel = Image.new("RGBA", (32, 42), (0, 0, 0, 0))
    draw = ImageDraw.Draw(panel, "RGBA")
    draw.rounded_rectangle((0, 0, 31, 41), radius=5, fill=(240, 240, 240, alpha))
    draw.text((16, 22), str(seed), font=_load_font(20, bold=True), fill="#111111", anchor="mm")
    if side == "left":
        frame.alpha_composite(panel, (int(x - 52), int(y - 21)))
    else:
        frame.alpha_composite(panel, (int(x + 14), int(y - 21)))


def _draw_seed_logo(frame: Image.Image, team: str, x: float, y: float, side: str, alpha: int = 255) -> None:
    tile = _team_square_logo(team, SEED_LOGO_SIZE)
    if alpha < 255:
        tile = tile.copy()
        channel = tile.getchannel("A").point(lambda a: int(a * (alpha / 255.0)))
        tile.putalpha(channel)
    px = int(x - tile.width // 2)
    py = int(y - tile.height // 2)
    if side == "left":
        px += 36
    else:
        px -= 36
    frame.alpha_composite(tile, (px, py))


def _draw_first_round_scaffold(frame: Image.Image, overlay: bool = False) -> None:
    draw = ImageDraw.Draw(frame, "RGBA")
    white = (255, 255, 255, 120 if overlay else 88)
    gold = (255, 205, 96, 138 if overlay else 92)
    left_x = LEFT_SEED_X + 58
    right_x = RIGHT_SEED_X - 58
    left_join_x = 184
    right_join_x = 896
    left_target_x = 350
    right_target_x = 730

    for idx in range(0, 8, 2):
        y1 = SEED_Y[idx]
        y2 = SEED_Y[idx + 1]
        y_mid = ROUND1_Y[idx // 2]
        draw.line((left_x, y1, left_join_x, y1), fill=white, width=4)
        draw.line((left_x, y2, left_join_x, y2), fill=white, width=4)
        draw.line((left_join_x, y_mid, left_target_x, y_mid), fill=gold, width=4)

        draw.line((right_x, y1, right_join_x, y1), fill=white, width=4)
        draw.line((right_x, y2, right_join_x, y2), fill=white, width=4)
        draw.line((right_join_x, y_mid, right_target_x, y_mid), fill=gold, width=4)


def _draw_scaffold(frame: Image.Image, t: float, overlay: bool = False) -> None:
    draw = ImageDraw.Draw(frame, "RGBA")
    white = (255, 255, 255, 90 if overlay else 40)
    gold = (255, 205, 96, 92 if overlay else 48)

    left_seed_right = LEFT_SEED_X + CARD_W // 2
    right_seed_left = RIGHT_SEED_X - CARD_W // 2

    # Round 1 merge lines.
    for idx in range(0, 8, 2):
        y1 = SEED_Y[idx]
        y2 = SEED_Y[idx + 1]
        y_mid = ROUND1_Y[idx // 2]
        draw.line((left_seed_right, y1, 188, y1), fill=white, width=4 if overlay else 3)
        draw.line((left_seed_right, y2, 188, y2), fill=white, width=4 if overlay else 3)
        draw.line((188, y_mid, 350, y_mid), fill=gold, width=4 if overlay else 3)

        draw.line((right_seed_left, y1, 892, y1), fill=white, width=4 if overlay else 3)
        draw.line((right_seed_left, y2, 892, y2), fill=white, width=4 if overlay else 3)
        draw.line((892, y_mid, 730, y_mid), fill=gold, width=4 if overlay else 3)

    # Round 2 merge lines.
    draw.line((350, SEMI_Y[0], 470, SEMI_Y[0]), fill=white, width=4 if overlay else 3)
    draw.line((350, SEMI_Y[1], 470, SEMI_Y[1]), fill=white, width=4 if overlay else 3)
    draw.line((730, SEMI_Y[0], 610, SEMI_Y[0]), fill=white, width=4 if overlay else 3)
    draw.line((730, SEMI_Y[1], 610, SEMI_Y[1]), fill=white, width=4 if overlay else 3)

    # Round 3 merge lines.
    draw.line((470, CONF_Y[0], 540, CONF_Y[0]), fill=gold, width=4 if overlay else 3)
    draw.line((610, CONF_Y[0], 540, CONF_Y[0]), fill=gold, width=4 if overlay else 3)


def _draw_seed_cards(frame: Image.Image, cards: dict[str, Image.Image]) -> None:
    for team, slot in SLOTS.items():
        _draw_seed_logo(frame, team, slot.seed_pos[0], slot.seed_pos[1], slot.side, alpha=235)
        _draw_seed_number(frame, slot.seed_pos[0], slot.seed_pos[1], slot.seed, slot.side, alpha=230)


def _draw_round_logo(
    frame: Image.Image,
    team: str,
    pos: tuple[int, int],
    size: int,
    score: str | None = None,
    alpha: float = 1.0,
    scale: float = 1.0,
    glow: bool = True,
    stage: str | None = None,
) -> None:
    tile = _team_square_logo(team, size)
    _draw_card(frame, tile, pos[0], pos[1], alpha, scale=scale, glow=glow)
    if score and alpha > 0.25 and stage != "conf":
        score_x, score_y = _score_badge_position(pos, size, stage)
        _draw_score_badge(frame, score, score_x, score_y, alpha=int(255 * _clamp(alpha)))


def _score_badge_position(pos: tuple[int, int], size: int, stage: str | None) -> tuple[int, int]:
    score_x = pos[0]
    if stage == "conf":
        if pos[0] < WIDTH * 0.5:
            score_x -= 108
        else:
            score_x += 108
        score_y = pos[1] + size // 2 + 22
    else:
        if pos[0] < WIDTH * 0.5 - 10:
            score_x += 22
        elif pos[0] > WIDTH * 0.5 + 10:
            score_x -= 22
        score_y = pos[1] + size // 2 + 32
    return score_x, score_y


def _draw_conference_score_overlay(frame: Image.Image, t: float, stop_at: float | None = None) -> None:
    if stop_at is not None and t >= stop_at:
        return
    overlay = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    for move in MOVES["conf"]:
        start_time = STAGE_BASES["conf"] + move.delay
        end_time = start_time + move.duration
        if t < end_time:
            continue
        score_x, score_y = _score_badge_position(move.end, ROUND_LOGO_SIZE, "conf")
        _draw_score_badge(overlay, move.score, score_x, score_y, alpha=255)
    frame.alpha_composite(overlay)


def _draw_seed_numbers(frame: Image.Image) -> None:
    for team, slot in SLOTS.items():
        _draw_seed_number(frame, slot.seed_pos[0], slot.seed_pos[1], slot.seed, slot.side, alpha=230)


def _team_visual_state(team: str, t: float) -> dict[str, object]:
    slot = SLOTS[team]
    pos = _seed_logo_center(team)
    level = "seed"
    alpha = 1.0
    moving = False
    score: str | None = None
    for event in TEAM_EVENTS[team]:
        if t < event.start:
            break
        if event.winner:
            if t < event.end:
                progress = _ease_in_out((t - event.start) / max(event.end - event.start, 1e-6))
                path = _path_points(event.source, event.dest or event.source, event.elbow_x)
                pos = _path_position(path, progress)
                return {
                    "mode": "moving",
                    "pos": pos,
                    "level": level,
                    "alpha": 1.0,
                    "scale": 1.02 + 0.05 * (1.0 - abs(2.0 * progress - 1.0)),
                    "score": None,
                }
            pos = event.dest or pos
            level = event.stage
            score = event.score
            continue
        if t < event.end:
            return {
                "mode": "static",
                "pos": pos,
                "level": level,
                "alpha": alpha,
                "scale": 1.0,
                "score": score,
            }
        fade = _ease_in_out(_clamp((t - event.end) / 0.45))
        alpha = 1.0 - 0.38 * fade
        return {
            "mode": "static",
            "pos": pos,
            "level": level,
            "alpha": alpha,
            "scale": 1.0,
            "score": None,
        }
    return {
        "mode": "static",
        "pos": pos,
        "level": level,
        "alpha": alpha,
        "scale": 1.0,
        "score": score,
    }


def _build_line_anims() -> list[SegmentAnim]:
    if not BRACKET_SVG.exists():
        raise FileNotFoundError(f"Bracket SVG not found: {BRACKET_SVG}")

    root = ET.parse(BRACKET_SVG).getroot()
    svg_width = float(root.attrib.get("width", "875"))
    svg_height = float(root.attrib.get("height", "912"))
    x_scale = WIDTH / svg_width
    y_scale = (SEED_Y[-1] - SEED_Y[0]) / (860.0 - 75.0)
    y_offset = SEED_Y[0] - 75.0 * y_scale
    line_tag = "{http://www.w3.org/2000/svg}line"
    skip_gold_segments = {
        (310.0, 467.5, 390.0, 467.5),
        (565.0, 467.5, 485.0, 467.5),
        (390.0, 692.5, 485.0, 692.5),
        (437.5, 757.5, 437.5, 777.5),
    }

    raw_lines: list[tuple[int, float, float, int, SegmentAnim]] = []
    for index, node in enumerate(root.iter(line_tag)):
        raw_x1 = float(node.attrib["x1"])
        raw_y1 = float(node.attrib["y1"])
        raw_x2 = float(node.attrib["x2"])
        raw_y2 = float(node.attrib["y2"])
        stroke = node.attrib.get("stroke", "#f2f2f2").lstrip("#")
        rgb = tuple(int(stroke[i : i + 2], 16) for i in (0, 2, 4))
        if rgb == (230, 189, 85) and (raw_x1, raw_y1, raw_x2, raw_y2) in skip_gold_segments:
            continue

        x1 = raw_x1 * x_scale
        y1 = raw_y1 * y_scale + y_offset
        x2 = raw_x2 * x_scale
        y2 = raw_y2 * y_scale + y_offset
        x_mid = (x1 + x2) / 2.0
        y_mid = (y1 + y2) / 2.0
        alpha = 220 if rgb != (230, 189, 85) else 210
        width = max(1, int(round(float(node.attrib.get("stroke-width", "3")) * ((x_scale + y_scale) / 2.0))))
        if x_mid < WIDTH / 2.0:
            if x_mid < 245 * x_scale:
                stage = 0
            elif x_mid < 310 * x_scale:
                stage = 1
            elif x_mid < 390 * x_scale:
                stage = 2
            else:
                stage = 3
        else:
            if x_mid > 630 * x_scale:
                stage = 0
            elif x_mid > 565 * x_scale:
                stage = 1
            elif x_mid > 485 * x_scale:
                stage = 2
            else:
                stage = 3
        anim = SegmentAnim(start=(int(round(x1)), int(round(y1))), end=(int(round(x2)), int(round(y2))), delay=0.0, duration=0.0, color=(*rgb, alpha), width=width)
        raw_lines.append((stage, y_mid, x_mid, index, anim))

    raw_lines.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
    # Trace the inner bracket lines first so the motion feels anchored to the title from frame one.
    stage_bases = [_scaled_time(value) for value in [5.40, 3.55, 1.80, 0.00]]
    stage_steps = [_scaled_time(value) for value in [0.06, 0.07, 0.08, 0.09]]
    stage_durations = [_scaled_time(value) for value in [0.58, 0.58, 0.60, 0.62]]
    stage_counts = [0, 0, 0, 0]
    lines: list[SegmentAnim] = []
    for stage, _y_mid, _x_mid, _index, anim in raw_lines:
        delay = stage_bases[stage] + stage_steps[stage] * stage_counts[stage]
        duration = stage_durations[stage]
        stage_counts[stage] += 1
        lines.append(SegmentAnim(start=anim.start, end=anim.end, delay=delay, duration=duration, color=anim.color, width=anim.width))
    return lines


LINE_ANIMS = _build_line_anims()


def render_video(output_path: Path, audio_path: Path | None = None, duration: float = TOTAL_DURATION, fps: int = FPS) -> Path:
    audio_exists = bool(audio_path and audio_path.exists())
    temp_audio_path = output_path.with_name(f"{output_path.stem}_temp_audio.m4a")
    champion_hold_start = max(0.0, float(duration) - CHAMPION_HOLD_SECONDS)
    static_base = _background_frame(0.0)

    _draw_title_panel(static_base, 1.0)
    _draw_glow(static_base, (178, 352), (54, 132, 255), 120, 42)
    _draw_glow(static_base, (900, 352), (255, 68, 76), 120, 42)
    _draw_glow(static_base, (540, 1018), (255, 210, 112), 150, 26)

    def _draw_line_overlay(t: float) -> Image.Image:
        overlay = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay, "RGBA")
        for anim in LINE_ANIMS:
            draw.line((anim.start, anim.end), fill=anim.color, width=anim.width)
        return overlay

    static_seed_numbers = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    _draw_seed_numbers(static_seed_numbers)

    def make_frame(t: float) -> np.ndarray:
        frame = static_base.copy()
        _draw_background_trophy(frame, t)
        frame.alpha_composite(_draw_line_overlay(t))
        frame.alpha_composite(static_seed_numbers)
        logo_layer = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
        states = {team: _team_visual_state(team, t) for team in SLOTS}

        for team in TEAM_ORDER:
            state = states[team]
            if state["mode"] != "static":
                continue
            level = state["level"]
            pos = state["pos"]
            alpha = float(state["alpha"])
            score = state["score"]
            if level == "final" and t >= champion_hold_start:
                score = None
            if level == "seed":
                slot = SLOTS[team]
                _draw_seed_logo(logo_layer, team, slot.seed_pos[0], slot.seed_pos[1], slot.side, alpha=int(255 * _clamp(alpha)))
            else:
                size = ROUND_LOGO_SIZE
                _draw_round_logo(
                    logo_layer,
                    team,
                    pos,
                    size=size,
                    score=score if alpha > 0.82 else None,
                    alpha=alpha,
                    scale=1.0,
                    glow=False,
                    stage=str(level),
                )

        for team in TEAM_ORDER:
            state = states[team]
            if state["mode"] != "moving":
                continue
            pos = state["pos"]
            scale = float(state["scale"])
            _draw_round_logo(logo_layer, team, pos, size=ROUND_LOGO_SIZE, score=None, alpha=1.0, scale=scale, glow=True, stage=str(state["level"]))

        frame.alpha_composite(logo_layer)
        if t >= champion_hold_start:
            _draw_champion_box(frame, t, champion_hold_start)
        else:
            _draw_conference_score_overlay(frame, t, champion_hold_start)
        rgb_frame = frame.convert("RGB")
        rgb_frame = rgb_frame.filter(EXPORT_SHARPEN)
        return np.array(rgb_frame)

    clip = VideoClip(make_frame, duration=float(duration))
    audio = None
    if audio_exists:
        audio = AudioFileClip(str(audio_path))
        if audio.duration and audio.duration > duration:
            audio = audio.subclipped(0, duration)
        clip = clip.with_audio(audio)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if audio_exists:
        clip.write_videofile(
            str(output_path),
            fps=fps,
            codec="libx264",
            audio_codec="aac",
            temp_audiofile=str(temp_audio_path),
            remove_temp=False,
            ffmpeg_params=[
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                "-crf",
                str(EXPORT_CRF),
                "-preset",
                EXPORT_PRESET,
            ],
        )
    else:
        clip.write_videofile(
            str(output_path),
            fps=fps,
            codec="libx264",
            audio=False,
            ffmpeg_params=[
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                "-crf",
                str(EXPORT_CRF),
                "-preset",
                EXPORT_PRESET,
            ],
        )
    clip.close()
    if audio is not None:
        audio.close()
    if temp_audio_path.exists():
        temp_audio_path.unlink()
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the NBA playoff bracket 2025 animation.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--audio", type=Path, default=DEFAULT_AUDIO)
    parser.add_argument("--duration", type=float, default=TOTAL_DURATION)
    parser.add_argument("--fps", type=int, default=FPS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = render_video(args.output, args.audio, args.duration, args.fps)
    print(f"[video_generator] NBA playoff bracket 2025 generated -> {output}")


if __name__ == "__main__":
    main()
