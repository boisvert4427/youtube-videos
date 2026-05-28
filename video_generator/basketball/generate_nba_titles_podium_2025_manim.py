from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps

from video_generator.generate_ucl_barchart_race_moviepy import _fit_font_size, _load_font

try:
    from manim import (
        BLACK,
        DOWN,
        FadeIn,
        Group,
        ImageMobject,
        LEFT,
        Line,
        ORIGIN,
        Polygon,
        Rectangle,
        RIGHT,
        RoundedRectangle,
        Scene,
        Text,
        UP,
        VGroup,
        WHITE,
        Circle,
        config,
        smooth,
    )

    MANIM_AVAILABLE = True
except Exception:
    MANIM_AVAILABLE = False
    Scene = object  # type: ignore[assignment]
    config = None  # type: ignore[assignment]
    BLACK = "#000000"  # type: ignore[assignment]
    WHITE = "#FFFFFF"  # type: ignore[assignment]
    LEFT = RIGHT = UP = DOWN = ORIGIN = 0  # type: ignore[assignment]

try:
    from moviepy import AudioFileClip, CompositeAudioClip, VideoFileClip
    from moviepy.audio.fx import AudioFadeOut
except Exception:
    AudioFileClip = None  # type: ignore[assignment]
    CompositeAudioClip = None  # type: ignore[assignment]
    VideoFileClip = None  # type: ignore[assignment]
    AudioFadeOut = None  # type: ignore[assignment]


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_MEDIA_DIR = PROJECT_ROOT / "data" / "processed" / "basketball" / "manim_nba_titles_podium"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "basketball" / "nba_titles_franchise_podium_2025_manim.mp4"
DEFAULT_OUTPUT_STEM = "nba_titles_franchise_podium_2025_manim"
DEFAULT_AUDIO = PROJECT_ROOT / "data" / "raw" / "audio" / "Midnight_Grip_20260402_0828.mp3"
DEFAULT_REQUIREMENTS = PROJECT_ROOT / "requirements-manim-mvp-race.txt"
DEFAULT_PLATE = DEFAULT_MEDIA_DIR / "plates" / "nba_titles_franchise_podium_2025_plate.png"
NBA_LOGO = PROJECT_ROOT / "data" / "raw" / "nba_logo.png"
TROPHY = PROJECT_ROOT / "data" / "raw" / "nba_trophy_photo_alt.png"
LOGO_DIR = PROJECT_ROOT / "data" / "raw" / "nba_team_logos"

RENDER_WIDTH = int(os.environ.get("NBA_PODIUM_RENDER_WIDTH", "1080"))
RENDER_HEIGHT = int(os.environ.get("NBA_PODIUM_RENDER_HEIGHT", "1920"))
FPS = int(os.environ.get("NBA_PODIUM_FPS", "30"))
FRAME_W = 9.0
FRAME_H = 16.0
FONT = "Bahnschrift"

if config is not None:
    config.pixel_width = RENDER_WIDTH
    config.pixel_height = RENDER_HEIGHT
    config.frame_width = FRAME_W
    config.frame_height = FRAME_H
    config.frame_rate = FPS
    config.background_color = "#DDE6ED"


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


def mix_hex(color: str, target: str, amount: float) -> str:
    amount = min(max(amount, 0.0), 1.0)
    rgb = hex_to_rgb(color)
    tgt = hex_to_rgb(target)
    mixed = tuple(int(rgb[i] + (tgt[i] - rgb[i]) * amount) for i in range(3))
    return f"#{mixed[0]:02x}{mixed[1]:02x}{mixed[2]:02x}"


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


def make_text(text: str, font_size: int, color: str, weight: str = "BOLD") -> Text:
    return Text(text, font=FONT, font_size=font_size, weight=weight, color=color)


def safe_image(path: Path, max_width: float, max_height: float) -> ImageMobject | None:
    if not path.exists():
        return None
    image = ImageMobject(str(path))
    image.set_width(max_width)
    if image.height > max_height:
        image.set_height(max_height)
    return image


def make_logo_plate(record: TeamRecord, size: int) -> Image.Image:
    primary, secondary = TEAM_COLORS[record.team_name]
    path = logo_path(record.team_name)
    if path.exists():
        logo = Image.open(path).convert("RGBA")
        return ImageOps.contain(logo, (size, size), method=Image.Resampling.LANCZOS)

    fallback = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(fallback, "RGBA")
    draw.ellipse((10, 10, size - 10, size - 10), fill=(*hex_to_rgb(primary), 255), outline=(*hex_to_rgb(secondary), 255), width=7)
    font = _fit_font_size(draw, record.team_abbr, size - 48, 62, 24, bold=True)
    draw.text((size // 2, size // 2 + 4), record.team_abbr, font=font, fill="#ffffff", anchor="mm", stroke_width=3, stroke_fill=(0, 0, 0, 150))
    return fallback


def paste_logo_with_halo(canvas: Image.Image, logo: Image.Image, cx: int, cy: int, halo_color: str) -> None:
    radius = max(logo.width, logo.height) / 2
    outer_x = int(radius * 1.18)
    outer_y = int(radius * 1.08)
    inner = int(radius * 0.95)
    halo = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(halo, "RGBA")
    draw.ellipse((cx - outer_x, cy - outer_y, cx + outer_x, cy + outer_y), fill=(0, 0, 0, 50))
    draw.ellipse((cx - inner, cy - inner, cx + inner, cy + inner), fill=(*hex_to_rgb(halo_color), 34))
    halo = halo.filter(ImageFilter.GaussianBlur(max(2, int(radius * 0.07))))
    canvas.alpha_composite(halo)
    canvas.alpha_composite(logo, (cx - logo.width // 2, cy - logo.height // 2))


def draw_text_with_shadow(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    font: Image.ImageFont.FreeTypeFont,
    fill: str,
    anchor: str = "mm",
    stroke_width: int = 0,
    stroke_fill: tuple[int, int, int, int] | str = (0, 0, 0, 0),
) -> None:
    x, y = xy
    draw.text((x + 3, y + 4), text, font=font, fill=(0, 0, 0, 125), anchor=anchor, stroke_width=stroke_width, stroke_fill=stroke_fill)
    draw.text((x, y), text, font=font, fill=fill, anchor=anchor, stroke_width=stroke_width, stroke_fill=stroke_fill)


def draw_years_plate(draw: ImageDraw.ImageDraw, record: TeamRecord, x0: int, y0: int, x1: int, y1: int) -> None:
    years = record.years
    if not years:
        return

    scale = RENDER_HEIGHT / 1920.0

    def px(value: float, min_value: int = 1) -> int:
        return max(min_value, int(round(value * scale)))

    available_h = max(1, y1 - y0 - 48)
    if len(years) >= 9:
        left = years[: (len(years) + 1) // 2]
        right = years[(len(years) + 1) // 2 :]
        rows = max(len(left), len(right))
        font_size = max(px(20, 9), min(px(34, 11), int(available_h / rows) - px(3)))
        font = _load_font(font_size, bold=True)
        step = max(px(38), min(px(58), int(available_h / rows)))
        start_y = y0 + px(42)
        lx = int(x0 + (x1 - x0) * 0.27)
        rx = int(x0 + (x1 - x0) * 0.78)
        for idx, year in enumerate(left):
            draw_text_with_shadow(draw, (lx, start_y + idx * step), str(year), font, "#F6CE63", stroke_width=px(3), stroke_fill=(42, 24, 12, 160))
        for idx, year in enumerate(right):
            draw_text_with_shadow(draw, (rx, start_y + idx * step), str(year), font, "#F6CE63", stroke_width=px(3), stroke_fill=(42, 24, 12, 160))
        return

    font_size = px(48 if len(years) <= 3 else 42, 12)
    font = _load_font(font_size, bold=True)
    step = px(66) if len(years) > 1 else 0
    total_h = step * max(0, len(years) - 1)
    start_y = int((y0 + y1) / 2 - total_h / 2)
    for idx, year in enumerate(years):
        draw_text_with_shadow(draw, ((x0 + x1) // 2, start_y + idx * step), str(year), font, "#F6CE63", stroke_width=px(3), stroke_fill=(42, 24, 12, 160))


def render_stage_plate(path: Path) -> Path:
    records = ranking_records()
    scale = RENDER_HEIGHT / 1920.0

    def px(value: float, min_value: int = 1) -> int:
        return max(min_value, int(round(value * scale)))

    col_w = px(250)
    gap = px(42)
    pitch = col_w + gap
    pad = px(130)
    depth_x = px(42)
    depth_y = px(30)
    floor_y = px(1810)
    cap_h = px(96)
    body_base_h = px(575)
    body_scale_h = px(48)
    body_max_h = px(1410)
    logo_size = px(232)
    width = pad * 2 + len(records) * pitch + depth_x
    height = RENDER_HEIGHT

    canvas = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(canvas, "RGBA")
    trophy_img = Image.open(TROPHY).convert("RGBA") if TROPHY.exists() else None
    count_font_1 = _load_font(px(82, 16), bold=True)
    count_font_2 = _load_font(px(74, 16), bold=True)

    for slot, record in enumerate(records):
        count = record.titles
        primary, secondary = TEAM_COLORS[record.team_name]
        body_h = min(body_max_h, body_base_h + count * body_scale_h)
        x0 = pad + slot * pitch
        x1 = x0 + col_w
        body_y0 = floor_y - body_h
        cap_y0 = body_y0 - cap_h

        if count == 0:
            front = mix_hex(primary, "#111821", 0.36)
            side = mix_hex(front, "#000000", 0.44)
            cap_front = "#303235"
            cap_side = "#1B1C1E"
            top = "#C9CDD0"
            count_fill = "#F8FAFC"
        else:
            front = primary
            side = mix_hex(primary, "#000000", 0.42)
            cap_front = "#5B3422"
            cap_side = "#382014"
            top = "#BD8910"
            count_fill = "#FFE176"

        shadow = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
        shadow_draw = ImageDraw.Draw(shadow, "RGBA")
        shadow_draw.rounded_rectangle((x0 + px(22), cap_y0 + px(36), x1 + depth_x + px(22), floor_y + px(26)), radius=px(28), fill=(0, 0, 0, 20))
        shadow_draw.ellipse((x0 - px(26), floor_y - px(22), x1 + depth_x + px(56), floor_y + px(30)), fill=(0, 0, 0, 26))
        shadow = shadow.filter(ImageFilter.GaussianBlur(px(3)))
        canvas.alpha_composite(shadow)

        logo = make_logo_plate(record, logo_size)
        paste_logo_with_halo(canvas, logo, x0 + col_w // 2, cap_y0 - px(52), secondary)

        draw.polygon([(x1, body_y0), (x1 + depth_x, body_y0 - depth_y), (x1 + depth_x, floor_y - depth_y), (x1, floor_y)], fill=(*hex_to_rgb(side), 255), outline=(12, 15, 20, 210))
        draw.rounded_rectangle((x0, body_y0, x1, floor_y), radius=px(18), fill=(*hex_to_rgb(front), 255), outline=(15, 19, 26, 230), width=px(2))
        draw.rectangle((x0 + px(10), body_y0 + px(16), x0 + px(16), floor_y - px(14)), fill=(255, 255, 255, 68))
        draw.rectangle((x1 - px(7), body_y0 + px(12), x1, floor_y - px(6)), fill=(0, 0, 0, 48))

        draw.polygon([(x1, cap_y0), (x1 + depth_x, cap_y0 - depth_y), (x1 + depth_x, body_y0 - depth_y), (x1, body_y0)], fill=(*hex_to_rgb(cap_side), 255), outline=(42, 28, 18, 230))
        draw.polygon([(x0, cap_y0), (x0 + depth_x, cap_y0 - depth_y), (x1 + depth_x, cap_y0 - depth_y), (x1, cap_y0)], fill=(*hex_to_rgb(top), 255), outline=(255, 224, 150, 190))
        draw.rounded_rectangle((x0, cap_y0, x1, body_y0), radius=px(10), fill=(*hex_to_rgb(cap_front), 255), outline=(110, 70, 28, 240), width=px(2))

        count_font = count_font_1 if count < 10 else count_font_2
        draw_text_with_shadow(draw, ((x0 + x1) // 2, cap_y0 + px(57)), str(count), count_font, count_fill, stroke_width=px(5), stroke_fill=(20, 15, 10, 180))

        if count > 0 and trophy_img is not None:
            trophy = ImageOps.contain(trophy_img, (px(70), px(86)), method=Image.Resampling.LANCZOS)
            trophy.putalpha(225)
            canvas.alpha_composite(trophy, (x1 - trophy.width - px(10), cap_y0 - px(18)))

        draw_years_plate(draw, record, x0 + px(8), body_y0 + px(8), x1 - px(10), floor_y - px(20))

    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)
    return path


class NBATitlesPodiumManim(Scene):  # type: ignore[misc]
    column_width = 1.95
    column_gap = 0.34
    pitch = column_width + column_gap
    depth_x = 0.34
    depth_y = 0.24
    floor_y = -7.18
    body_base_h = 4.68
    body_scale_h = 0.22
    body_max_h = 8.55
    cap_h = 0.72

    def construct(self) -> None:
        duration = float(os.environ.get("NBA_PODIUM_DURATION", "80"))
        outro_hold = min(5.5, max(0.8, duration * 0.12))
        scroll_duration = max(1.0, duration - outro_hold)

        self.add(self.build_background())
        self.add(self.build_floor())
        header = self.build_header()

        stage = self.build_stage_plate()
        start_shift = -4.65 - stage.get_left()[0]
        stage.shift(RIGHT * start_shift)
        target_right = 4.55
        scroll_distance = stage.get_right()[0] - target_right

        self.add(stage)
        self.add(header)
        self.play(stage.animate.shift(LEFT * scroll_distance), run_time=scroll_duration, rate_func=smooth)
        self.wait(outro_hold)

    def build_background(self) -> VGroup:
        bg = VGroup()
        bg.add(Rectangle(width=9.4, height=16.4, stroke_width=0, fill_color="#DDE6ED", fill_opacity=1).move_to(ORIGIN))

        for idx, color in enumerate(["#F9FCFF", "#EDF3F7", "#D6E1EA", "#C8D5DF"]):
            band = Rectangle(width=9.4, height=4.1, stroke_width=0, fill_color=color, fill_opacity=0.28 - idx * 0.035)
            band.move_to(UP * (5.9 - idx * 2.2))
            bg.add(band)

        for idx, x in enumerate([-3.9, -2.1, 0.0, 2.1, 3.9]):
            ray = Rectangle(width=0.028, height=15.0, stroke_width=0, fill_color=WHITE, fill_opacity=0.08 if idx == 2 else 0.045)
            ray.rotate(0.13 * x).move_to(RIGHT * x * 0.52 + UP * 0.2)
            bg.add(ray)

        for y in [-5.8, -4.7, -3.6, -2.5, -1.4, -0.3, 0.8, 1.9, 3.0, 4.1, 5.2]:
            bg.add(Line(LEFT * 4.5 + UP * y, RIGHT * 4.5 + UP * y, stroke_color=WHITE, stroke_opacity=0.12, stroke_width=1.0))

        glow = Circle(radius=3.2, color="#FFF1B7", fill_color="#FFF1B7", fill_opacity=0.055, stroke_width=0).move_to(RIGHT * 1.2 + UP * 4.3)
        cool_glow = Circle(radius=4.4, color="#BFD4E6", fill_color="#BFD4E6", fill_opacity=0.06, stroke_width=0).move_to(LEFT * 2.7 + DOWN * 1.8)
        bg.add(cool_glow, glow)
        return bg

    def build_header(self) -> Group:
        header = Group()
        panel = RoundedRectangle(width=8.38, height=0.98, corner_radius=0.18, fill_color="#EEF4F8", fill_opacity=0.82, stroke_color=WHITE, stroke_opacity=0.74, stroke_width=1.3)
        panel.move_to(UP * 7.18)
        title = make_text("NBA TITLES RANKING", 34, "#101823").move_to(LEFT * 1.78 + UP * 7.33)
        subtitle = make_text("ACTIVE FRANCHISES | THROUGH 2025", 16, "#5C6875").move_to(LEFT * 2.42 + UP * 7.00)
        badge = RoundedRectangle(width=1.36, height=0.66, corner_radius=0.13, fill_color="#F7CE55", fill_opacity=1, stroke_color=WHITE, stroke_opacity=0.88, stroke_width=1.2)
        badge.move_to(RIGHT * 3.14 + UP * 7.20)
        year = make_text("2025", 36, "#142235").move_to(badge)
        header.add(panel, title, subtitle, badge, year)
        nba = safe_image(NBA_LOGO, 0.30, 0.52)
        if nba is not None:
            nba.move_to(RIGHT * 4.05 + UP * 7.20).set_opacity(0.78)
            header.add(nba)
        return header

    def build_floor(self) -> VGroup:
        floor = VGroup()
        floor_panel = Rectangle(width=9.6, height=6.4, stroke_width=0, fill_color=WHITE, fill_opacity=0.18)
        floor_panel.move_to(DOWN * 4.95)
        floor.add(floor_panel)
        vanishing = UP * 0.16
        for x in np.linspace(-7.2, 7.2, 28):
            floor.add(Line(np.array([x, -8.1, 0]), vanishing, stroke_color="#6E7C86", stroke_opacity=0.14, stroke_width=1.15))
        for idx, y in enumerate(np.linspace(-7.65, -1.05, 11)):
            opacity = 0.08 + idx * 0.008
            floor.add(Line(LEFT * 4.8 + UP * y, RIGHT * 4.8 + UP * y, stroke_color=WHITE, stroke_opacity=opacity, stroke_width=1.2))
        floor.add(Line(LEFT * 4.28 + UP * (self.floor_y - 0.12), RIGHT * 4.28 + UP * (self.floor_y - 0.12), stroke_color=WHITE, stroke_opacity=0.78, stroke_width=2.1))
        return floor

    def build_stage_plate(self) -> ImageMobject:
        plate_path = render_stage_plate(DEFAULT_PLATE)
        stage = ImageMobject(str(plate_path))
        stage.set_height(FRAME_H)
        stage.move_to(ORIGIN)
        return stage

    def build_stage(self, records: list[TeamRecord]) -> Group:
        stage = Group()
        for idx, record in enumerate(records):
            stage.add(self.build_column(record, idx * self.pitch))
        return stage

    def build_logo(self, record: TeamRecord, x_center: float, y_center: float, top_color: str) -> Group:
        group = Group()
        group.add(Circle(radius=0.78, fill_color=BLACK, fill_opacity=0.16, stroke_width=0).move_to(RIGHT * (x_center + 0.06) + UP * (y_center - 0.06)))
        group.add(Circle(radius=0.66, fill_color=top_color, fill_opacity=0.09, stroke_width=0).move_to(RIGHT * x_center + UP * y_center))
        logo = safe_image(logo_path(record.team_name), 1.36, 1.20)
        if logo is None:
            fallback = Circle(radius=0.58, fill_color=top_color, fill_opacity=1, stroke_color=WHITE, stroke_width=2)
            fallback.move_to(RIGHT * x_center + UP * y_center)
            abbr = make_text(record.team_abbr, 25, WHITE).move_to(fallback)
            group.add(fallback, abbr)
        else:
            logo.move_to(RIGHT * x_center + UP * y_center)
            group.add(logo)
        return group

    def build_column(self, record: TeamRecord, x: float) -> Group:
        group = Group()
        count = record.titles
        primary, secondary = TEAM_COLORS[record.team_name]
        body_h = min(self.body_max_h, self.body_base_h + count * self.body_scale_h)
        x0 = x
        x1 = x + self.column_width
        body_y0 = self.floor_y
        body_y1 = body_y0 + body_h
        cap_y0 = body_y1
        cap_y1 = cap_y0 + self.cap_h
        dx = self.depth_x
        dy = self.depth_y

        if count == 0:
            front = mix_hex(primary, "#111821", 0.34)
            side = mix_hex(front, "#000000", 0.42)
            cap_front = "#303235"
            cap_side = "#1B1C1E"
            top = "#C9CDD0"
            count_color = "#F8FAFC"
        else:
            front = primary
            side = mix_hex(primary, "#000000", 0.42)
            cap_front = "#5B3422"
            cap_side = "#382014"
            top = "#BD8910"
            count_color = "#FFE176"

        shadow = VGroup(
            RoundedRectangle(width=self.column_width + dx + 0.22, height=body_h + self.cap_h + 0.18, corner_radius=0.12, stroke_width=0, fill_color=BLACK, fill_opacity=0.10),
            Rectangle(width=self.column_width + dx + 0.45, height=0.18, stroke_width=0, fill_color=BLACK, fill_opacity=0.17),
        )
        shadow[0].move_to(RIGHT * (x0 + self.column_width / 2 + dx / 2 + 0.10) + UP * ((body_y0 + cap_y1) / 2 - 0.12))
        shadow[1].move_to(RIGHT * (x0 + self.column_width / 2 + dx / 2 + 0.08) + UP * (self.floor_y - 0.06))
        group.add(shadow)

        logo_center_y = cap_y1 + 0.42
        group.add(self.build_logo(record, x0 + self.column_width / 2, logo_center_y, secondary))

        side_face = Polygon(
            [x1, body_y0, 0],
            [x1 + dx, body_y0 + dy, 0],
            [x1 + dx, body_y1 + dy, 0],
            [x1, body_y1, 0],
            color=side,
            fill_color=side,
            fill_opacity=1,
            stroke_color="#0E1115",
            stroke_width=1.0,
        )
        front_face = RoundedRectangle(
            width=self.column_width,
            height=body_h,
            corner_radius=0.08,
            fill_color=front,
            fill_opacity=1,
            stroke_color="#10141A",
            stroke_width=1.0,
        ).move_to(RIGHT * (x0 + self.column_width / 2) + UP * (body_y0 + body_h / 2))
        front_highlight = Rectangle(width=0.045, height=max(0.1, body_h - 0.22), stroke_width=0, fill_color=WHITE, fill_opacity=0.35).move_to(RIGHT * (x0 + 0.12) + UP * (body_y0 + body_h / 2))
        side_shade = Rectangle(width=0.055, height=max(0.1, body_h - 0.2), stroke_width=0, fill_color=BLACK, fill_opacity=0.20).move_to(RIGHT * (x1 - 0.04) + UP * (body_y0 + body_h / 2))

        cap_side_face = Polygon(
            [x1, cap_y0, 0],
            [x1 + dx, cap_y0 + dy, 0],
            [x1 + dx, cap_y1 + dy, 0],
            [x1, cap_y1, 0],
            color=cap_side,
            fill_color=cap_side,
            fill_opacity=1,
            stroke_color="#2C1D14",
            stroke_width=1.0,
        )
        cap_top = Polygon(
            [x0, cap_y1, 0],
            [x0 + dx, cap_y1 + dy, 0],
            [x1 + dx, cap_y1 + dy, 0],
            [x1, cap_y1, 0],
            color=top,
            fill_color=top,
            fill_opacity=1,
            stroke_color="#FFE096" if count else "#F2F4F5",
            stroke_width=1.0,
        )
        cap_front_face = RoundedRectangle(
            width=self.column_width,
            height=self.cap_h,
            corner_radius=0.08,
            fill_color=cap_front,
            fill_opacity=1,
            stroke_color="#6E461E",
            stroke_width=1.1,
        ).move_to(RIGHT * (x0 + self.column_width / 2) + UP * (cap_y0 + self.cap_h / 2))

        group.add(side_face, front_face, front_highlight, side_shade, cap_side_face, cap_top, cap_front_face)

        count_text = make_text(str(count), 46 if count < 10 else 40, count_color).move_to(RIGHT * (x0 + self.column_width / 2) + UP * (cap_y0 + self.cap_h * 0.46))
        count_shadow = count_text.copy().set_color(BLACK).set_opacity(0.62).shift(DOWN * 0.035 + RIGHT * 0.035)
        group.add(count_shadow, count_text)

        if count > 0:
            trophy = safe_image(TROPHY, 0.42, 0.52)
            if trophy is not None:
                trophy.move_to(RIGHT * (x1 - 0.28) + UP * (cap_y1 + 0.05)).set_opacity(0.82)
                group.add(trophy)

        group.add(self.build_years(record, x0, body_y0, body_y1))
        return group

    def build_years(self, record: TeamRecord, x0: float, y0: float, y1: float) -> VGroup:
        years = record.years
        group = VGroup()
        if not years:
            return group

        if len(years) >= 9:
            left = years[: (len(years) + 1) // 2]
            right = years[(len(years) + 1) // 2 :]
            rows = max(len(left), len(right))
            font_size = 25 if rows >= 9 else 28
            top = y1 - 0.42
            step = min(0.44, max(0.32, (y1 - y0 - 0.88) / max(rows - 1, 1)))
            for idx, year in enumerate(left):
                text = make_text(str(year), font_size, "#F6CE63").move_to(RIGHT * (x0 + self.column_width * 0.34) + UP * (top - idx * step))
                group.add(text.copy().set_color(BLACK).set_opacity(0.58).shift(DOWN * 0.025 + RIGHT * 0.025), text)
            for idx, year in enumerate(right):
                text = make_text(str(year), font_size, "#F6CE63").move_to(RIGHT * (x0 + self.column_width * 0.73) + UP * (top - idx * step))
                group.add(text.copy().set_color(BLACK).set_opacity(0.58).shift(DOWN * 0.025 + RIGHT * 0.025), text)
            return group

        font_size = 34 if len(years) <= 3 else 31
        total_h = min(0.54 * max(len(years) - 1, 0), y1 - y0 - 1.0)
        start_y = (y0 + y1) / 2 + total_h / 2
        step = 0.54 if len(years) > 1 else 0.0
        for idx, year in enumerate(years):
            text = make_text(str(year), font_size, "#F6CE63").move_to(RIGHT * (x0 + self.column_width / 2) + UP * (start_y - idx * step))
            group.add(text.copy().set_color(BLACK).set_opacity(0.58).shift(DOWN * 0.025 + RIGHT * 0.025), text)
        return group


def run_manim(scene_name: str, media_dir: Path, quality: str, output_stem: str, duration: float, width: int, height: int, fps: int) -> int:
    env = os.environ.copy()
    env["NBA_PODIUM_DURATION"] = str(duration)
    env["NBA_PODIUM_RENDER_WIDTH"] = str(width)
    env["NBA_PODIUM_RENDER_HEIGHT"] = str(height)
    env["NBA_PODIUM_FPS"] = str(fps)
    cmd = [
        sys.executable,
        "-m",
        "manim",
        str(Path(__file__).resolve()),
        scene_name,
        "--format",
        "mp4",
        "--media_dir",
        str(media_dir),
        "--output_file",
        output_stem,
    ]
    if quality:
        cmd.append(f"-q{quality}")
    return subprocess.run(cmd, check=False, env=env).returncode


def find_rendered_video(media_dir: Path, output_stem: str) -> Path:
    matches = list(media_dir.rglob(f"{output_stem}.mp4"))
    if not matches:
        raise FileNotFoundError(f"No Manim render found for {output_stem} in {media_dir}")
    return max(matches, key=lambda path: path.stat().st_mtime)


def attach_audio(video_path: Path, output_path: Path, music_path: Path | None, fade_out_seconds: float = 5.0) -> None:
    if VideoFileClip is None or AudioFileClip is None or CompositeAudioClip is None:
        raise RuntimeError("MoviePy is required for audio assembly.")
    video = VideoFileClip(str(video_path))
    audio = None
    if music_path is not None and music_path.exists():
        base_music = AudioFileClip(str(music_path))
        if base_music.duration >= video.duration:
            audio = base_music.subclipped(0, video.duration)
        else:
            loops = []
            step = max(0.1, base_music.duration - 1.5)
            total = 0.0
            while total < video.duration:
                loops.append(base_music.with_start(total))
                total += step
            audio = CompositeAudioClip(loops).with_duration(video.duration)
        if AudioFadeOut is not None:
            audio = audio.with_effects([AudioFadeOut(min(fade_out_seconds, video.duration))])
        final = video.with_audio(audio)
    else:
        final = video
    output_path.parent.mkdir(parents=True, exist_ok=True)
    final.write_videofile(str(output_path), codec="libx264", audio_codec="aac" if audio else None)
    final.close()
    video.close()
    if audio is not None:
        audio.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a premium Manim NBA 2025 franchise titles podium short.")
    parser.add_argument("--scene", default="NBATitlesPodiumManim")
    parser.add_argument("--render", action="store_true", help="Render the scene with Manim.")
    parser.add_argument("--quality", default="m", help="Manim quality flag: l, m, h, p, k.")
    parser.add_argument("--duration", type=float, default=80.0)
    parser.add_argument("--width", type=int, default=1080)
    parser.add_argument("--height", type=int, default=1920)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--media-dir", type=Path, default=DEFAULT_MEDIA_DIR)
    parser.add_argument("--output-stem", default=DEFAULT_OUTPUT_STEM)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--audio", type=Path, default=DEFAULT_AUDIO)
    parser.add_argument("--mix-audio", action="store_true")
    parser.add_argument("--requirements-file", type=Path, default=DEFAULT_REQUIREMENTS)
    args = parser.parse_args()

    if not MANIM_AVAILABLE:
        if args.render:
            raise SystemExit(f"Manim is not installed. Install it with: python -m pip install -r {args.requirements_file}")
        print(f"Install Manim with: python -m pip install -r {args.requirements_file}")
        return

    if not args.render:
        print("Scene available: NBATitlesPodiumManim")
        print(f"Preview:\n  python {Path(__file__).name} --render --quality l --duration 10 --output data/processed/basketball/nba_titles_franchise_podium_2025_manim_preview.mp4")
        print(f"Final:\n  python {Path(__file__).name} --render --quality h --duration 80 --mix-audio")
        return

    status = run_manim(args.scene, args.media_dir, args.quality, args.output_stem, args.duration, args.width, args.height, args.fps)
    if status != 0:
        raise SystemExit(status)

    rendered = find_rendered_video(args.media_dir, args.output_stem)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.mix_audio:
        attach_audio(rendered, args.output, args.audio)
    else:
        shutil.copy2(rendered, args.output)
    print(f"[manim] NBA titles podium ready -> {args.output}")


if __name__ == "__main__":
    main()
