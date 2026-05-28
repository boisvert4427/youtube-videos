from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from moviepy import VideoClip
from PIL import Image, ImageDraw, ImageFilter, ImageOps

from video_generator.generate_ucl_barchart_race_moviepy import _fit_font_size, _load_font, build_audio_track


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "basketball" / "nba_titles_franchise_podium_2025_80s.mp4"
DEFAULT_AUDIO = PROJECT_ROOT / "data" / "raw" / "audio" / "Midnight_Grip_20260402_0828.mp3"
NBA_LOGO = PROJECT_ROOT / "data" / "raw" / "nba_logo.png"
TROPHY = PROJECT_ROOT / "data" / "raw" / "nba_trophy_photo_alt.png"
LOGO_DIR = PROJECT_ROOT / "data" / "raw" / "nba_team_logos"

WIDTH = 1080
HEIGHT = 1920
FPS = 30
TOTAL_DURATION = 80.0
OUTRO_HOLD = 7.0

TITLE = "NBA TITLES RANKING"
SUBTITLE = "Active franchises | through 2025"

STAGE_PAD = 110
CARD_W = 250
CARD_GAP = 34
CARD_PITCH = CARD_W + CARD_GAP
FLOOR_Y = 1818
LOGO_BOX = 226
CAP_H = 96
BODY_BASE_H = 560
BODY_SCALE_H = 46
BODY_MAX_H = 1380
DEPTH_X = 38
DEPTH_Y = 26
REVEAL_DISTANCE = 430
INTRO_REVEAL_SECONDS = 0.9

CAP_BROWN = (91, 52, 34)
CAP_GOLD = (186, 135, 12)
COUNT_GOLD = "#ffe176"
YEAR_GOLD = "#f6ce63"
METAL_A = np.array([204, 211, 217], dtype=np.float32)
METAL_B = np.array([231, 236, 241], dtype=np.float32)
METAL_C = np.array([183, 193, 200], dtype=np.float32)


@dataclass(frozen=True)
class TeamRecord:
    team_name: str
    team_abbr: str
    titles: int
    years: tuple[int, ...]


TEAM_RECORDS = [
    TeamRecord("Atlanta Hawks", "ATL", 1, (1958,)),
    TeamRecord("Boston Celtics", "BOS", 18, (1957, 1959, 1960, 1961, 1962, 1963, 1964, 1965, 1966, 1968, 1969, 1974, 1976, 1981, 1984, 1986, 2008, 2024)),
    TeamRecord("Brooklyn Nets", "BKN", 0, ()),
    TeamRecord("Charlotte Hornets", "CHA", 0, ()),
    TeamRecord("Chicago Bulls", "CHI", 6, (1991, 1992, 1993, 1996, 1997, 1998)),
    TeamRecord("Cleveland Cavaliers", "CLE", 1, (2016,)),
    TeamRecord("Dallas Mavericks", "DAL", 1, (2011,)),
    TeamRecord("Denver Nuggets", "DEN", 1, (2023,)),
    TeamRecord("Detroit Pistons", "DET", 3, (1989, 1990, 2004)),
    TeamRecord("Golden State Warriors", "GSW", 7, (1947, 1956, 1975, 2015, 2017, 2018, 2022)),
    TeamRecord("Houston Rockets", "HOU", 2, (1994, 1995)),
    TeamRecord("Indiana Pacers", "IND", 0, ()),
    TeamRecord("LA Clippers", "LAC", 0, ()),
    TeamRecord("Los Angeles Lakers", "LAL", 17, (1949, 1950, 1952, 1953, 1954, 1972, 1980, 1982, 1985, 1987, 1988, 2000, 2001, 2002, 2009, 2010, 2020)),
    TeamRecord("Memphis Grizzlies", "MEM", 0, ()),
    TeamRecord("Miami Heat", "MIA", 3, (2006, 2012, 2013)),
    TeamRecord("Milwaukee Bucks", "MIL", 2, (1971, 2021)),
    TeamRecord("Minnesota Timberwolves", "MIN", 0, ()),
    TeamRecord("New Orleans Pelicans", "NOP", 0, ()),
    TeamRecord("New York Knicks", "NYK", 2, (1970, 1973)),
    TeamRecord("Oklahoma City Thunder", "OKC", 2, (1979, 2025)),
    TeamRecord("Orlando Magic", "ORL", 0, ()),
    TeamRecord("Philadelphia 76ers", "PHI", 3, (1955, 1967, 1983)),
    TeamRecord("Phoenix Suns", "PHX", 0, ()),
    TeamRecord("Portland Trail Blazers", "POR", 1, (1977,)),
    TeamRecord("Sacramento Kings", "SAC", 1, (1951,)),
    TeamRecord("San Antonio Spurs", "SAS", 5, (1999, 2003, 2005, 2007, 2014)),
    TeamRecord("Toronto Raptors", "TOR", 1, (2019,)),
    TeamRecord("Utah Jazz", "UTA", 0, ()),
    TeamRecord("Washington Wizards", "WAS", 1, (1978,)),
]

TEAM_COLORS = {
    "Atlanta Hawks": ("#C8102E", "#FDB927"),
    "Boston Celtics": ("#007A33", "#BA9653"),
    "Brooklyn Nets": ("#111111", "#F4F4F4"),
    "Charlotte Hornets": ("#1D1160", "#00788C"),
    "Chicago Bulls": ("#CE1141", "#111111"),
    "Cleveland Cavaliers": ("#6F263D", "#FFB81C"),
    "Dallas Mavericks": ("#00538C", "#B8C4CA"),
    "Denver Nuggets": ("#0E2240", "#FEC524"),
    "Detroit Pistons": ("#1D42BA", "#C8102E"),
    "Golden State Warriors": ("#1D428A", "#FFC72C"),
    "Houston Rockets": ("#CE1141", "#C4CED4"),
    "Indiana Pacers": ("#002D62", "#FDBB30"),
    "LA Clippers": ("#1D428A", "#C8102E"),
    "Los Angeles Lakers": ("#552583", "#FDB927"),
    "Memphis Grizzlies": ("#5D76A9", "#12173F"),
    "Miami Heat": ("#98002E", "#F9A01B"),
    "Milwaukee Bucks": ("#00471B", "#EEE1C6"),
    "Minnesota Timberwolves": ("#0C2340", "#78BE20"),
    "New Orleans Pelicans": ("#0C2340", "#C8102E"),
    "New York Knicks": ("#006BB6", "#F58426"),
    "Oklahoma City Thunder": ("#007AC1", "#EF3B24"),
    "Orlando Magic": ("#0077C0", "#C4CED4"),
    "Philadelphia 76ers": ("#006BB6", "#ED174C"),
    "Phoenix Suns": ("#1D1160", "#E56020"),
    "Portland Trail Blazers": ("#E03A3E", "#111111"),
    "Sacramento Kings": ("#5A2D81", "#C4CED4"),
    "San Antonio Spurs": ("#111111", "#C4CED4"),
    "Toronto Raptors": ("#CE1141", "#111111"),
    "Utah Jazz": ("#002B5C", "#F9A01B"),
    "Washington Wizards": ("#002B5C", "#E31837"),
}

LOGO_SLUGS = {
    "LA Clippers": "la_clippers",
}


def hex_to_rgb(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    return tuple(int(value[i : i + 2], 16) for i in (0, 2, 4))


def mix_rgb(rgb: tuple[int, int, int], target: tuple[int, int, int], amount: float) -> tuple[int, int, int]:
    amount = min(max(amount, 0.0), 1.0)
    return (
        int(rgb[0] + (target[0] - rgb[0]) * amount),
        int(rgb[1] + (target[1] - rgb[1]) * amount),
        int(rgb[2] + (target[2] - rgb[2]) * amount),
    )


def ease(value: float) -> float:
    value = min(max(value, 0.0), 1.0)
    return value * value * (3.0 - 2.0 * value)


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return min(max(value, low), high)


def ease_out_back(value: float) -> float:
    value = clamp(value)
    c1 = 1.55
    c3 = c1 + 1
    return 1 + c3 * (value - 1) ** 3 + c1 * (value - 1) ** 2


def rgba_hex(value: str, alpha: float) -> tuple[int, int, int, int]:
    return (*hex_to_rgb(value), int(255 * clamp(alpha)))


def slug(team_name: str) -> str:
    if team_name in LOGO_SLUGS:
        return LOGO_SLUGS[team_name]
    value = team_name.lower().replace(".", "").replace("&", "and").replace("-", "_").replace(" ", "_")
    while "__" in value:
        value = value.replace("__", "_")
    return value


def logo_path(team_name: str) -> Path:
    return LOGO_DIR / f"{slug(team_name)}.png"


def ranking_records() -> list[TeamRecord]:
    return sorted(TEAM_RECORDS, key=lambda record: (record.titles, record.team_name))


def build_logo_cache(records: list[TeamRecord]) -> dict[str, Image.Image]:
    cache: dict[str, Image.Image] = {}
    for record in records:
        primary, secondary = TEAM_COLORS.get(record.team_name, ("#47627c", "#f2f2f2"))
        path = logo_path(record.team_name)
        if path.exists():
            raw = Image.open(path).convert("RGBA")
            logo = ImageOps.contain(raw, (LOGO_BOX, LOGO_BOX), method=Image.Resampling.LANCZOS)
        else:
            logo = Image.new("RGBA", (LOGO_BOX, LOGO_BOX), (0, 0, 0, 0))
            draw = ImageDraw.Draw(logo, "RGBA")
            draw.ellipse((12, 12, LOGO_BOX - 12, LOGO_BOX - 12), fill=(*hex_to_rgb(primary), 255), outline=(*hex_to_rgb(secondary), 255), width=8)
            font = _fit_font_size(draw, record.team_abbr, LOGO_BOX - 52, 64, 26, bold=True)
            draw.text(
                (LOGO_BOX // 2, LOGO_BOX // 2 + 4),
                record.team_abbr,
                font=font,
                fill="#ffffff",
                anchor="mm",
                stroke_width=3,
                stroke_fill=(0, 0, 0, 160),
            )

        canvas = Image.new("RGBA", (LOGO_BOX + 28, LOGO_BOX + 28), (0, 0, 0, 0))
        shadow = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
        shadow_draw = ImageDraw.Draw(shadow, "RGBA")
        shadow_draw.ellipse((16, 18, canvas.width - 12, canvas.height - 8), fill=(0, 0, 0, 70))
        shadow = shadow.filter(ImageFilter.GaussianBlur(radius=9))
        canvas.alpha_composite(shadow)
        canvas.alpha_composite(logo, ((canvas.width - logo.width) // 2, (canvas.height - logo.height) // 2))
        cache[record.team_name] = canvas
    return cache


def make_background() -> Image.Image:
    xx = np.linspace(0, 1, WIDTH, dtype=np.float32)
    yy = np.linspace(0, 1, HEIGHT, dtype=np.float32)
    gx, gy = np.meshgrid(xx, yy)

    brushed = 0.5 + 0.5 * np.sin(gx * 25.0 * np.pi + np.sin(gy * 8.0) * 0.7)
    fine = 0.5 + 0.5 * np.sin(gy * 145.0 * np.pi)
    glow = np.exp(-(((gx - 0.66) / 0.35) ** 2 + ((gy - 0.12) / 0.18) ** 2))
    shade = np.clip(0.22 + 0.50 * gy + 0.08 * brushed + 0.03 * fine, 0.0, 1.0)
    img = np.clip(
        METAL_B[None, None, :] * (1.0 - shade[..., None])
        + METAL_A[None, None, :] * shade[..., None]
        + METAL_C[None, None, :] * (0.10 * gy[..., None])
        + np.array([255, 244, 212], dtype=np.float32)[None, None, :] * (0.10 * glow[..., None]),
        0,
        255,
    ).astype(np.uint8)
    frame = Image.fromarray(img, "RGB").convert("RGBA")

    overlay = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")
    for y in range(70, HEIGHT, 34):
        alpha = 12 if y % 68 else 20
        draw.line((0, y, WIDTH, y), fill=(255, 255, 255, alpha), width=2)
    draw.rounded_rectangle((26, 34, WIDTH - 26, HEIGHT - 28), radius=42, outline=(255, 255, 255, 24), width=2)
    overlay = overlay.filter(ImageFilter.GaussianBlur(radius=1.1))
    frame.alpha_composite(overlay)
    return frame


def draw_header(frame: Image.Image, draw: ImageDraw.ImageDraw, nba_logo: Image.Image) -> None:
    title_font = _load_font(38, bold=True)
    sub_font = _load_font(20, bold=True)
    year_font = _load_font(56, bold=True)

    draw.rounded_rectangle((40, 36, WIDTH - 40, 154), radius=26, fill=(238, 243, 247, 150), outline=(255, 255, 255, 92), width=2)
    draw.text((62, 64), TITLE, font=title_font, fill="#121821", stroke_width=3, stroke_fill=(255, 255, 255, 110))
    draw.text((64, 116), SUBTITLE, font=sub_font, fill="#5f6974")

    draw.rounded_rectangle((WIDTH - 244, 58, WIDTH - 86, 140), radius=18, fill=(247, 206, 84, 255), outline=(255, 255, 255, 140), width=2)
    draw.text((WIDTH - 165, 99), "2025", font=year_font, fill="#142235", anchor="mm")

    nba = ImageOps.contain(nba_logo.convert("RGBA"), (38, 64), method=Image.Resampling.LANCZOS)
    nba.putalpha(190)
    frame.alpha_composite(nba, (WIDTH - 67, 66))


def draw_floor(draw: ImageDraw.ImageDraw, camera_x: int) -> None:
    floor_top = 860
    vanishing = (WIDTH // 2, 690)
    draw.polygon(
        [(0, floor_top), (WIDTH, floor_top), (WIDTH, HEIGHT), (0, HEIGHT)],
        fill=(255, 255, 255, 8),
    )
    draw.line((0, floor_top, WIDTH, floor_top), fill=(255, 255, 255, 22), width=2)

    drift = camera_x % 160
    for x in range(-640, WIDTH + 800, 160):
        draw.line((x - drift, HEIGHT, vanishing[0], vanishing[1]), fill=(96, 108, 118, 12), width=2)
    for idx in range(11):
        t = idx / 10.0
        y = int(floor_top + (HEIGHT - floor_top) * (t**1.8))
        alpha = int(8 + 22 * t)
        draw.line((0, y, WIDTH, y), fill=(255, 255, 255, alpha), width=2)

    draw.line((32, FLOOR_Y + 16, WIDTH - 32, FLOOR_Y + 16), fill=(255, 255, 255, 210), width=3)


def draw_years(draw: ImageDraw.ImageDraw, years: tuple[int, ...], x0: int, y0: int, x1: int, y1: int, count: int, alpha: float = 1.0) -> None:
    if not years:
        return

    fill = rgba_hex(YEAR_GOLD, alpha)
    stroke = (42, 24, 12, int(150 * clamp(alpha)))
    available_h = max(1, y1 - y0 - 34)
    if len(years) >= 9:
        left = years[: (len(years) + 1) // 2]
        right = years[(len(years) + 1) // 2 :]
        lines = max(len(left), len(right))
        size = max(20, min(36, int(available_h / max(lines, 1)) - 2))
        font = _load_font(size, bold=True)
        col_w = x1 - x0
        lx = int(x0 + col_w * 0.30)
        rx = int(x0 + col_w * 0.73)
        start_y = y0 + 30
        step = max(34, min(58, int(available_h / max(lines, 1))))
        for idx, year in enumerate(left):
            draw.text(
                (lx, start_y + idx * step),
                str(year),
                font=font,
                fill=fill,
                anchor="mt",
                stroke_width=3,
                stroke_fill=stroke,
            )
        for idx, year in enumerate(right):
            draw.text(
                (rx, start_y + idx * step),
                str(year),
                font=font,
                fill=fill,
                anchor="mt",
                stroke_width=3,
                stroke_fill=stroke,
            )
        return

    size = max(22, min(54, int(available_h / max(len(years), 1)) - 2))
    if count <= 3:
        size = min(size, 46)
    font = _load_font(size, bold=True)
    step = max(38, min(66, int(available_h / max(len(years), 1))))
    start_y = y0 + max(20, (available_h - step * len(years)) // 2 + 34)
    for idx, year in enumerate(years):
        draw.text(
            ((x0 + x1) // 2, start_y + idx * step),
            str(year),
            font=font,
            fill=fill,
            anchor="mm",
            stroke_width=3,
            stroke_fill=stroke,
        )


def draw_column_shadow(draw: ImageDraw.ImageDraw, x0: int, y0: int, x1: int) -> None:
    draw.rounded_rectangle((x0 + 18, y0 + 34, x1 + DEPTH_X + 24, FLOOR_Y + 22), radius=26, fill=(0, 0, 0, 14))
    draw.ellipse((x0 - 22, FLOOR_Y - 28, x1 + DEPTH_X + 56, FLOOR_Y + 38), fill=(0, 0, 0, 14))
    draw.ellipse((x0 + 12, FLOOR_Y - 15, x1 + DEPTH_X + 24, FLOOR_Y + 22), fill=(0, 0, 0, 17))


def fade_image(image: Image.Image, alpha: float) -> Image.Image:
    alpha = clamp(alpha)
    result = image.copy()
    mask = result.getchannel("A").point(lambda value: int(value * alpha))
    result.putalpha(mask)
    return result


def scaled_image(image: Image.Image, scale: float) -> Image.Image:
    scale = max(0.05, scale)
    width = max(1, int(round(image.width * scale)))
    height = max(1, int(round(image.height * scale)))
    return image.resize((width, height), Image.Resampling.LANCZOS)


def draw_ground_burst(draw: ImageDraw.ImageDraw, x0: int, x1: int, reveal: float, accent: tuple[int, int, int]) -> None:
    if reveal <= 0.02 or reveal >= 0.92:
        return
    pulse = clamp((reveal - 0.02) / 0.62)
    fade = 1.0 - clamp((reveal - 0.32) / 0.60)
    cx = (x0 + x1) // 2 + DEPTH_X // 2
    rx = int(70 + 150 * pulse)
    ry = int(10 + 23 * pulse)
    alpha = int(96 * fade * math.sin(min(1.0, pulse) * math.pi))
    if alpha <= 0:
        return
    draw.ellipse((cx - rx, FLOOR_Y - ry, cx + rx, FLOOR_Y + ry), outline=(*accent, alpha), width=3)
    draw.ellipse((cx - rx // 2, FLOOR_Y - ry // 2, cx + rx // 2, FLOOR_Y + ry // 2), fill=(*accent, max(0, alpha // 5)))


def appearance_progress(t: float, slot: int, screen_x: int) -> float:
    entry = clamp((WIDTH + 150 - screen_x) / REVEAL_DISTANCE)
    if t < 1.35 and screen_x < WIDTH + 60:
        intro = clamp((t - min(slot, 6) * 0.08) / INTRO_REVEAL_SECONDS)
        return min(entry, intro)
    return entry


def draw_column(frame: Image.Image, draw: ImageDraw.ImageDraw, record: TeamRecord, logo: Image.Image, trophy: Image.Image, x: int, reveal: float = 1.0) -> None:
    primary_hex, _ = TEAM_COLORS.get(record.team_name, ("#47627c", "#f2f2f2"))
    primary = hex_to_rgb(primary_hex)
    count = record.titles
    target_body_h = int(min(BODY_MAX_H, BODY_BASE_H + count * BODY_SCALE_H))
    height_alpha = min(1.045, ease_out_back(reveal))
    body_h = max(4, int(target_body_h * height_alpha))
    x0 = x
    x1 = x + CARD_W
    body_y0 = FLOOR_Y - body_h
    cap_y0 = body_y0 - CAP_H
    content_alpha = clamp((reveal - 0.50) / 0.42)
    logo_alpha = clamp((reveal - 0.22) / 0.50)

    if count == 0:
        front_fill = mix_rgb(primary, (18, 25, 34), 0.35)
        side_fill = mix_rgb(front_fill, (0, 0, 0), 0.40)
        top_fill = (194, 198, 201)
        cap_fill = (55, 55, 55)
        cap_side_fill = (32, 32, 32)
        count_fill = "#f8fafc"
    else:
        front_fill = primary
        side_fill = mix_rgb(primary, (0, 0, 0), 0.38)
        top_fill = CAP_GOLD
        cap_fill = CAP_BROWN
        cap_side_fill = mix_rgb(CAP_BROWN, (0, 0, 0), 0.32)
        count_fill = COUNT_GOLD

    draw_ground_burst(draw, x0, x1, reveal, top_fill if count else (210, 215, 218))
    draw_column_shadow(draw, x0, cap_y0, x1)

    if logo_alpha > 0:
        logo_scale = 0.74 + 0.30 * min(1.0, ease_out_back(logo_alpha))
        logo_fit = fade_image(scaled_image(logo, logo_scale), logo_alpha)
        logo_x = x0 + (CARD_W - logo_fit.width) // 2
        logo_y = cap_y0 - logo_fit.height + 84
        frame.alpha_composite(logo_fit, (logo_x, logo_y))

    draw.polygon(
        [(x1, body_y0), (x1 + DEPTH_X, body_y0 - DEPTH_Y), (x1 + DEPTH_X, FLOOR_Y - DEPTH_Y), (x1, FLOOR_Y)],
        fill=side_fill + (255,),
    )
    draw.rounded_rectangle((x0, body_y0, x1, FLOOR_Y), radius=18, fill=front_fill + (255,))
    if body_h > 34:
        draw.rectangle((x0, body_y0 + 16, x0 + 6, FLOOR_Y - 12), fill=(255, 255, 255, 38))
        draw.rectangle((x1 - 7, body_y0 + 10, x1, FLOOR_Y), fill=(0, 0, 0, 38))

    draw.polygon(
        [(x1, cap_y0), (x1 + DEPTH_X, cap_y0 - DEPTH_Y), (x1 + DEPTH_X, body_y0 - DEPTH_Y), (x1, body_y0)],
        fill=cap_side_fill + (255,),
    )
    draw.polygon(
        [(x0, cap_y0), (x0 + DEPTH_X, cap_y0 - DEPTH_Y), (x1 + DEPTH_X, cap_y0 - DEPTH_Y), (x1, cap_y0)],
        fill=top_fill + (255,),
    )
    draw.rounded_rectangle((x0, cap_y0, x1, body_y0), radius=10, fill=cap_fill + (255,), outline=(112, 76, 28), width=2)
    draw.line((x1, cap_y0 + 4, x1 + DEPTH_X, cap_y0 - DEPTH_Y + 4), fill=(255, 226, 130, 130), width=2)
    draw.line((x1, body_y0, x1 + DEPTH_X, body_y0 - DEPTH_Y), fill=(0, 0, 0, 74), width=2)

    count_font = _load_font(78 if count < 10 else 70, bold=True)
    draw.text(
        ((x0 + x1) // 2, cap_y0 + 58),
        str(count),
        font=count_font,
        fill=rgba_hex(count_fill, clamp((reveal - 0.25) / 0.35)),
        anchor="mm",
        stroke_width=5,
        stroke_fill=(20, 15, 10, int(168 * clamp((reveal - 0.25) / 0.35))),
    )

    if count > 0 and content_alpha > 0:
        trophy_fit = ImageOps.contain(trophy.convert("RGBA"), (68, 88), method=Image.Resampling.LANCZOS)
        trophy_fit = fade_image(trophy_fit, content_alpha * 0.90)
        frame.alpha_composite(trophy_fit, (x1 - trophy_fit.width - 8, cap_y0 - 16))

    draw_years(draw, record.years, x0 + 10, body_y0 + 8, x1 - 12, FLOOR_Y - 18, count, alpha=content_alpha)


def render_video(output_path: Path, audio_path: Path, duration: float, fps: int) -> Path:
    records = ranking_records()
    logo_cache = build_logo_cache(records)
    background = make_background()
    nba_logo = Image.open(NBA_LOGO).convert("RGBA")
    trophy = Image.open(TROPHY).convert("RGBA")

    stage_width = STAGE_PAD * 2 + len(records) * CARD_PITCH + DEPTH_X
    max_camera_x = max(0, stage_width - WIDTH)
    animated_duration = max(1.0, duration - min(OUTRO_HOLD, max(1.0, duration * 0.20)))

    def make_frame(t: float) -> np.ndarray:
        progress = min(1.0, max(0.0, t / animated_duration))
        camera_x = int(round(ease(progress) * max_camera_x))

        frame = background.copy()
        draw = ImageDraw.Draw(frame, "RGBA")
        draw_floor(draw, camera_x)
        draw_header(frame, draw, nba_logo)

        visible: list[tuple[int, int, TeamRecord, float]] = []
        for slot, record in enumerate(records):
            stage_x = STAGE_PAD + slot * CARD_PITCH
            screen_x = int(round(stage_x - camera_x))
            if screen_x > WIDTH + LOGO_BOX or screen_x + CARD_W + DEPTH_X < -LOGO_BOX:
                continue
            reveal = appearance_progress(t, slot, screen_x)
            visible.append((screen_x, slot, record, reveal))

        for screen_x, _slot, record, reveal in sorted(visible, key=lambda item: item[0]):
            draw_column(frame, draw, record, logo_cache[record.team_name], trophy, screen_x, reveal=reveal)

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
    parser = argparse.ArgumentParser(description="Generate a vertical 2025 NBA titles ranking podium short.")
    parser.add_argument("--input", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--audio", type=Path, default=DEFAULT_AUDIO)
    parser.add_argument("--duration", type=float, default=TOTAL_DURATION)
    parser.add_argument("--fps", type=int, default=FPS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = render_video(args.output, args.audio, args.duration, args.fps)
    print(f"[video_generator] NBA 2025 franchise titles podium generated -> {output}")


if __name__ == "__main__":
    main()
