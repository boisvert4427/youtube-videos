from __future__ import annotations

import argparse
from dataclasses import dataclass
from functools import lru_cache
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
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "tennis" / "federer_vs_djokovic_h2h_scores_shorts.mp4"

WIDTH = 1080
HEIGHT = 1920
FPS = 60
TOTAL_DURATION = 60.0
MUSIC_VOLUME = 0.40

PORTRAIT_TOP = 410
SCOREBOARD_TOP = 750
SCOREBOARD_LINE_Y = SCOREBOARD_TOP + 78
SCOREBOARD_BOTTOM = SCOREBOARD_TOP + 150
SCORE_TOUCH_Y = SCOREBOARD_BOTTOM + 6

TIMELINE_START_Y = 1840
ROW_SPACING = 642
TIMELINE_LEFT = 0
TIMELINE_RIGHT = WIDTH
TIMELINE_END_PADDING = 18

BADGE_W = 166
BADGE_H = 78
BADGE_GAP_X = 28
BADGE_MARGIN_X = 12
BADGE_LANES = 9

THEME = {
    "paper": (244, 248, 255),
    "muted": (166, 179, 200),
    "ink": (4, 8, 16),
    "panel": (10, 17, 31),
    "panel_hi": (27, 39, 63),
    "grid": (201, 218, 245),
    "center": (219, 237, 255),
    "fed": (82, 215, 255),
    "fed_2": (126, 94, 255),
    "djokovic": (243, 201, 79),
    "djokovic_2": (173, 132, 33),
    "nadal": (255, 115, 61),
    "nadal_2": (214, 47, 35),
}

LEFT_PLAYER = {
    "photo": "roger_federer.jpg",
    "accent": THEME["fed"],
}
RIGHT_PLAYER = {
    "photo": "novak_djokovic.jpg",
    "accent": THEME["djokovic"],
}


@dataclass(frozen=True)
class MatchEntry:
    year: int
    month: int
    event: str
    winner: str
    scoreline: str


MATCHES: list[MatchEntry] = [
    MatchEntry(2006, 4, "MONTE CARLO R64", "left", "6-3 2-6 6-3"),
    MatchEntry(2006, 9, "DAVIS CUP WG PO", "left", "6-3 6-2 6-3"),
    MatchEntry(2007, 1, "AUSTRALIAN OPEN R16", "left", "6-2 7-5 6-3"),
    MatchEntry(2007, 2, "DUBAI QF", "left", "6-3 6-7 6-3"),
    MatchEntry(2007, 8, "CANADA F", "right", "7-6 2-6 7-6"),
    MatchEntry(2007, 9, "US OPEN F", "left", "7-6 7-6 6-4"),
    MatchEntry(2008, 1, "AUSTRALIAN OPEN SF", "right", "7-5 6-3 7-6"),
    MatchEntry(2008, 4, "MONTE CARLO SF", "left", "6-3 3-2"),
    MatchEntry(2008, 9, "US OPEN SF", "left", "6-3 5-7 7-5 6-2"),
    MatchEntry(2009, 3, "MIAMI SF", "right", "3-6 6-2 6-3"),
    MatchEntry(2009, 5, "ROME SF", "right", "4-6 6-3 6-3"),
    MatchEntry(2009, 8, "CINCINNATI F", "left", "6-1 7-5"),
    MatchEntry(2009, 9, "US OPEN SF", "left", "7-6 7-5 7-5"),
    MatchEntry(2009, 11, "BASEL F", "right", "6-4 4-6 6-2"),
    MatchEntry(2010, 8, "CANADA SF", "left", "6-1 3-6 7-5"),
    MatchEntry(2010, 9, "US OPEN SF", "right", "5-7 6-1 5-7 6-2 7-5"),
    MatchEntry(2010, 10, "SHANGHAI SF", "left", "7-5 6-4"),
    MatchEntry(2010, 11, "BASEL F", "left", "6-4 3-6 6-1"),
    MatchEntry(2010, 11, "ATP FINALS SF", "left", "6-1 6-4"),
    MatchEntry(2011, 1, "AUSTRALIAN OPEN SF", "right", "7-6 7-5 6-4"),
    MatchEntry(2011, 2, "DUBAI F", "right", "6-3 6-3"),
    MatchEntry(2011, 3, "INDIAN WELLS SF", "right", "6-3 3-6 6-2"),
    MatchEntry(2011, 6, "ROLAND GARROS SF", "left", "7-6 6-3 3-6 7-6"),
    MatchEntry(2011, 9, "US OPEN SF", "right", "6-7 4-6 6-3 6-2 7-5"),
    MatchEntry(2012, 5, "ROME SF", "right", "6-2 7-6"),
    MatchEntry(2012, 6, "ROLAND GARROS SF", "right", "6-4 7-5 6-3"),
    MatchEntry(2012, 7, "WIMBLEDON SF", "left", "6-3 3-6 6-4 6-3"),
    MatchEntry(2012, 8, "CINCINNATI F", "left", "6-0 7-6"),
    MatchEntry(2012, 11, "ATP FINALS F", "right", "7-6 7-5"),
    MatchEntry(2013, 11, "PARIS SF", "right", "4-6 6-3 6-2"),
    MatchEntry(2013, 11, "ATP FINALS RR", "right", "6-4 6-7 6-2"),
    MatchEntry(2014, 2, "DUBAI SF", "left", "3-6 6-3 6-2"),
    MatchEntry(2014, 3, "INDIAN WELLS F", "right", "3-6 6-3 7-6"),
    MatchEntry(2014, 4, "MONTE CARLO SF", "left", "7-5 6-2"),
    MatchEntry(2014, 7, "WIMBLEDON F", "right", "6-7 6-4 7-6 5-7 6-4"),
    MatchEntry(2014, 10, "SHANGHAI SF", "left", "6-4 6-4"),
    MatchEntry(2015, 2, "DUBAI F", "left", "6-3 7-5"),
    MatchEntry(2015, 3, "INDIAN WELLS F", "right", "6-3 6-7 6-2"),
    MatchEntry(2015, 5, "ROME F", "right", "6-4 6-3"),
    MatchEntry(2015, 7, "WIMBLEDON F", "right", "7-6 6-7 6-4 6-3"),
    MatchEntry(2015, 8, "CINCINNATI F", "left", "7-6 6-3"),
    MatchEntry(2015, 9, "US OPEN F", "right", "6-4 5-7 6-4 6-4"),
    MatchEntry(2015, 11, "ATP FINALS RR", "left", "7-5 6-2"),
    MatchEntry(2015, 11, "ATP FINALS F", "right", "6-3 6-4"),
    MatchEntry(2016, 1, "AUSTRALIAN OPEN SF", "right", "6-1 6-2 3-6 6-3"),
    MatchEntry(2018, 8, "CINCINNATI F", "right", "6-4 6-4"),
    MatchEntry(2018, 11, "PARIS SF", "right", "7-6 5-7 7-6"),
    MatchEntry(2019, 7, "WIMBLEDON F", "right", "7-6 1-6 7-6 4-6 13-12"),
    MatchEntry(2019, 11, "ATP FINALS RR", "left", "6-4 6-3"),
    MatchEntry(2020, 1, "AUSTRALIAN OPEN SF", "right", "7-6 6-4 6-3"),
]

YEARS = list(range(2006, 2021))
MONTH_LABELS = ("JAN", "FEV", "MAR", "AVR", "MAI", "JUN", "JUL", "AOU", "SEP", "OCT", "NOV", "DEC")


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return min(max(value, low), high)


def _ease_out(value: float) -> float:
    value = _clamp(value)
    return 1.0 - (1.0 - value) ** 3


def _ease_in_out(value: float) -> float:
    value = _clamp(value)
    return value * value * (3.0 - 2.0 * value)


def _load_sport_font(size: int, bold: bool = False, condensed: bool = False) -> ImageFont.ImageFont:
    candidates = [
        "C:/Windows/Fonts/impact.ttf" if bold and condensed else "",
        "C:/Windows/Fonts/bahnschrift.ttf",
        "C:/Windows/Fonts/ArialNovaCond-Bold.ttf" if bold else "C:/Windows/Fonts/ArialNovaCond.ttf",
        "C:/Windows/Fonts/ArialNova-Bold.ttf" if bold else "C:/Windows/Fonts/ArialNova.ttf",
        "C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf",
    ]
    for font_path in candidates:
        if not font_path:
            continue
        path = Path(font_path)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return _load_font(size, bold=bold)


def _fit_sport_font(
    draw: ImageDraw.ImageDraw,
    text: str,
    max_width: int,
    start_size: int,
    min_size: int,
    bold: bool = True,
    condensed: bool = False,
) -> ImageFont.ImageFont:
    size = start_size
    font = _load_sport_font(size, bold=bold, condensed=condensed)
    while size > min_size and draw.textbbox((0, 0), text, font=font)[2] > max_width:
        size -= 1
        font = _load_sport_font(size, bold=bold, condensed=condensed)
    return font


def _vertical_gradient(size: tuple[int, int], top: tuple[int, int, int], bottom: tuple[int, int, int]) -> Image.Image:
    w, h = size
    y = np.linspace(0, 1, h, dtype=np.float32)[:, None]
    top_arr = np.array(top, dtype=np.float32)
    bottom_arr = np.array(bottom, dtype=np.float32)
    image = (top_arr * (1.0 - y) + bottom_arr * y).astype(np.uint8)
    image = np.repeat(image[:, None, :], w, axis=1)
    return Image.fromarray(image, "RGB").convert("RGBA")


def _glow_blob(size: tuple[int, int], color: tuple[int, int, int], alpha: int, blur: int) -> Image.Image:
    img = Image.new("RGBA", size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(img, "RGBA")
    draw.ellipse((blur, blur, size[0] - blur, size[1] - blur), fill=(*color, alpha))
    return img.filter(ImageFilter.GaussianBlur(radius=blur))


def _rounded_glass(size: tuple[int, int], radius: int, fill_alpha: int = 116) -> Image.Image:
    img = Image.new("RGBA", size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(img, "RGBA")
    draw.rounded_rectangle(
        (0, 0, size[0] - 1, size[1] - 1),
        radius=radius,
        fill=(*THEME["panel_hi"], fill_alpha),
        outline=(255, 255, 255, 36),
        width=1,
    )
    draw.rounded_rectangle((2, 2, size[0] - 3, size[1] - 3), radius=radius - 2, outline=(255, 255, 255, 24), width=1)
    return img


def _load_portrait(path: Path, initials: str) -> Image.Image:
    if path.exists():
        return ImageOps.exif_transpose(Image.open(path)).convert("RGBA")
    img = Image.new("RGBA", (720, 720), (18, 24, 34, 255))
    draw = ImageDraw.Draw(img)
    draw.text((360, 360), initials, font=_load_font(80, bold=True), fill=(245, 248, 252, 255), anchor="mm")
    return img


def _make_background() -> Image.Image:
    xx = np.linspace(0, 1, WIDTH, dtype=np.float32)
    yy = np.linspace(0, 1, HEIGHT, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(xx, yy)

    left_top = np.array([38, 30, 82], dtype=np.float32)
    left_mid = np.array([82, 48, 126], dtype=np.float32)
    left_bottom = np.array([14, 54, 84], dtype=np.float32)
    right_top = np.array([20, 45, 86], dtype=np.float32)
    right_mid = np.array([66, 74, 104], dtype=np.float32)
    right_bottom = np.array([98, 47, 37], dtype=np.float32)
    left = left_top * (1.0 - grid_y[..., None]) + left_bottom * grid_y[..., None]
    right = right_top * (1.0 - grid_y[..., None]) + right_bottom * grid_y[..., None]
    left += left_mid * (0.34 * np.exp(-((grid_y - 0.42) ** 2 + (grid_x - 0.22) ** 2) / 0.12))[..., None]
    right += right_mid * (0.32 * np.exp(-((grid_y - 0.45) ** 2 + (grid_x - 0.76) ** 2) / 0.12))[..., None]

    left_accent = np.array(THEME["nadal"], dtype=np.float32)
    right_accent = np.array(THEME["djokovic"], dtype=np.float32)
    left += left_accent * (0.12 * np.exp(-((grid_y - 0.28) ** 2 + (grid_x - 0.24) ** 2) / 0.026))[..., None]
    right += right_accent * (0.15 * np.exp(-((grid_y - 0.28) ** 2 + (grid_x - 0.76) ** 2) / 0.026))[..., None]

    shade = (1.08 - 0.24 * grid_y)[..., None]
    split = (grid_x < 0.5).astype(np.float32)[..., None]
    image = np.clip((left * split + right * (1.0 - split)) * shade + 12, 0, 255).astype(np.uint8)
    frame = Image.fromarray(image, mode="RGB").convert("RGBA")

    rng = np.random.default_rng(42)
    grain = rng.normal(127, 10, (HEIGHT, WIDTH)).clip(0, 255).astype(np.uint8)
    grain_img = Image.fromarray(grain, "L").convert("RGBA")
    grain_img.putalpha(13)
    frame.alpha_composite(grain_img)

    dist = np.sqrt(((grid_x - 0.5) / 0.72) ** 2 + ((grid_y - 0.47) / 0.90) ** 2)
    vignette_alpha = np.clip((dist - 0.38) / 0.70, 0, 1) * 74
    vignette = Image.fromarray(vignette_alpha.astype(np.uint8), "L")
    dark = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 255))
    dark.putalpha(vignette)
    frame.alpha_composite(dark)
    return frame


def _draw_glow_text(
    frame: Image.Image,
    position: tuple[int, int],
    text: str,
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int],
    stroke_width: int = 2,
    anchor: str = "mm",
    glow_color: tuple[int, int, int] | None = None,
    glow_radius: int = 6,
    glow_alpha: int = 110,
) -> None:
    glow_color = glow_color or fill
    glow = Image.new("RGBA", frame.size, (0, 0, 0, 0))
    gd = ImageDraw.Draw(glow, "RGBA")
    gd.text(position, text, font=font, fill=(*glow_color, glow_alpha), anchor=anchor)
    frame.alpha_composite(glow.filter(ImageFilter.GaussianBlur(radius=glow_radius)))
    shadow = Image.new("RGBA", frame.size, (0, 0, 0, 0))
    sd = ImageDraw.Draw(shadow, "RGBA")
    sd.text((position[0] + 4, position[1] + 6), text, font=font, fill=(0, 0, 0, 150), anchor=anchor)
    frame.alpha_composite(shadow.filter(ImageFilter.GaussianBlur(radius=3)))
    ImageDraw.Draw(frame, "RGBA").text(
        position,
        text,
        font=font,
        fill=(*fill, 255),
        anchor=anchor,
        stroke_width=stroke_width,
        stroke_fill=(2, 5, 12, 210),
    )


def _circle_portrait(source: Image.Image, accent: tuple[int, int, int]) -> Image.Image:
    size = 188
    portrait = ImageOps.fit(source, (size, size), method=Image.Resampling.LANCZOS, centering=(0.5, 0.22))
    portrait = ImageEnhance.Contrast(ImageEnhance.Brightness(portrait).enhance(1.04)).enhance(1.16)
    mask = Image.new("L", (size, size), 0)
    ImageDraw.Draw(mask).ellipse((0, 0, size - 1, size - 1), fill=255)

    tile = Image.new("RGBA", (size + 74, size + 74), (0, 0, 0, 0))
    draw = ImageDraw.Draw(tile, "RGBA")
    tile.alpha_composite(_glow_blob((size + 74, size + 74), accent, 150, 30))
    draw.ellipse((21, 21, size + 52, size + 52), fill=(255, 255, 255, 30), outline=(255, 255, 255, 86), width=2)
    draw.ellipse((29, 29, size + 44, size + 44), outline=(*accent, 245), width=5)
    draw.ellipse((37, 37, size + 36, size + 36), fill=(7, 13, 25, 224), outline=(255, 255, 255, 160), width=2)
    tile.paste(portrait, (37, 37), mask)
    shine = Image.new("RGBA", tile.size, (0, 0, 0, 0))
    sd = ImageDraw.Draw(shine, "RGBA")
    sd.ellipse((49, 43, size + 12, size - 42), fill=(255, 255, 255, 26))
    tile.alpha_composite(shine.filter(ImageFilter.GaussianBlur(radius=8)))
    return tile


def _draw_static_layer(portraits: dict[str, Image.Image]) -> Image.Image:
    frame = _make_background()
    draw = ImageDraw.Draw(frame, "RGBA")
    center_glow = Image.new("RGBA", frame.size, (0, 0, 0, 0))
    cd = ImageDraw.Draw(center_glow, "RGBA")
    cd.line((WIDTH // 2, 0, WIDTH // 2, HEIGHT), fill=(*THEME["center"], 105), width=2)
    frame.alpha_composite(center_glow.filter(ImageFilter.GaussianBlur(radius=4)))
    draw.line((WIDTH // 2, 0, WIDTH // 2, HEIGHT), fill=(*THEME["center"], 94), width=1)

    frame.alpha_composite(_glow_blob((360, 360), LEFT_PLAYER["accent"], 72, 70), (88, PORTRAIT_TOP - 68))
    frame.alpha_composite(_glow_blob((360, 360), RIGHT_PLAYER["accent"], 76, 70), (632, PORTRAIT_TOP - 68))
    frame.alpha_composite(_circle_portrait(portraits["left"], LEFT_PLAYER["accent"]), (132, PORTRAIT_TOP - 26))
    frame.alpha_composite(_circle_portrait(portraits["right"], RIGHT_PLAYER["accent"]), (708, PORTRAIT_TOP - 26))
    return frame


@lru_cache(maxsize=1)
def _header_image() -> Image.Image:
    header = Image.new("RGBA", (WIDTH, 360), (0, 0, 0, 0))
    draw = ImageDraw.Draw(header, "RGBA")

    label_font = _load_sport_font(25, bold=True)
    title_font = _fit_sport_font(draw, "FEDERER vs DJOKOVIC", 944, 94, 58, bold=True, condensed=True)
    sub_font = _load_sport_font(31, bold=False)

    pill_w, pill_h = 326, 42
    pill_x = WIDTH // 2 - pill_w // 2
    draw.rounded_rectangle((pill_x, 42, pill_x + pill_w, 42 + pill_h), radius=21, fill=(255, 255, 255, 22), outline=(255, 255, 255, 46), width=1)
    draw.text((WIDTH // 2, 63), "LEGENDARY RIVALRY", font=label_font, fill=(*THEME["muted"], 255), anchor="mm")

    _draw_glow_text(
        header,
        (WIDTH // 2, 164),
        "FEDERER vs DJOKOVIC",
        title_font,
        THEME["paper"],
        stroke_width=2,
        glow_color=(112, 185, 255),
        glow_radius=9,
        glow_alpha=74,
    )
    draw.text((WIDTH // 2, 254), "HEAD-TO-HEAD TIMELINE", font=sub_font, fill=(224, 233, 246, 255), anchor="mm")
    draw.line((274, 292, 806, 292), fill=(255, 255, 255, 70), width=1)
    draw.line((338, 294, 742, 294), fill=(*THEME["center"], 120), width=2)
    return header


def _draw_header(frame: Image.Image, progress: float) -> None:
    enter = _ease_out(progress / 0.025)
    header = _header_image()
    if enter >= 0.999:
        frame.alpha_composite(header, (0, 8))
        return

    scale = 0.965 + 0.035 * enter

    nw, nh = int(WIDTH * scale), int(360 * scale)
    header = header.resize((nw, nh), Image.Resampling.LANCZOS)
    x = (WIDTH - nw) // 2
    y = int(8 - 12 * (1.0 - enter))
    header.putalpha(ImageEnhance.Brightness(header.getchannel("A")).enhance(enter))
    frame.alpha_composite(header, (x, y))


def _flip_set_score(part: str) -> str:
    left, right = part.split("-", 1)
    return f"{right}-{left}"


def _scoreline_set_count(scoreline: str) -> int:
    return len(scoreline.split())


def _badge_width_for_scoreline(scoreline: str) -> int:
    set_count = _scoreline_set_count(scoreline)
    width_map = {
        2: 186,
        3: 246,
        4: 318,
        5: 410,
    }
    return width_map.get(set_count, min(430, max(BADGE_W, 132 + 58 * set_count)))


def _federer_first_scoreline(scoreline: str, winner: str) -> str:
    parts = scoreline.split()
    if winner == "left":
        ordered = parts
    else:
        ordered = [_flip_set_score(part) for part in parts]
    return "  ".join(ordered)


def _short_event(event: str) -> str:
    replacements = {
        "AUSTRALIAN OPEN": "AO",
        "ROLAND GARROS": "RG",
        "INDIAN WELLS": "IW",
        "MONTE CARLO": "MC",
        "ATP FINALS": "ATP FINALS",
        "MASTERS CUP": "MASTERS",
        "WIMBLEDON": "WIMB",
    }
    for source, short in replacements.items():
        event = event.replace(source, short)
    return event


@lru_cache(maxsize=None)
def _surface_colors(event: str) -> tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]]:
    event = event.upper()
    if any(surface_event in event for surface_event in ("ROLAND GARROS", "MONTE CARLO", "ROME", "HAMBURG", "MADRID")):
        return (90, 38, 22), (26, 13, 13), (230, 92, 38)
    if "WIMBLEDON" in event:
        return (32, 76, 47), (10, 31, 24), (116, 214, 132)
    if any(surface_event in event for surface_event in ("ATP FINALS", "MASTERS CUP", "BASEL")):
        return (55, 38, 91), (14, 17, 39), (168, 124, 255)
    return (31, 56, 88), (7, 17, 34), (88, 188, 255)


@lru_cache(maxsize=None)
def _match_label_font(label: str, badge_width: int) -> ImageFont.ImageFont:
    img = Image.new("RGBA", (1, 1), (0, 0, 0, 0))
    return _fit_sport_font(ImageDraw.Draw(img), label, badge_width + 80, 25, 15, bold=True)


@lru_cache(maxsize=None)
def _score_badge(scoreline: str, winner: str, accent: tuple[int, int, int], surface_top: tuple[int, int, int], surface_bottom: tuple[int, int, int]) -> Image.Image:
    badge_w = _badge_width_for_scoreline(scoreline)
    img = Image.new("RGBA", (badge_w, BADGE_H + 10), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img, "RGBA")
    shadow = Image.new("RGBA", img.size, (0, 0, 0, 0))
    ImageDraw.Draw(shadow, "RGBA").rounded_rectangle((5, 10, badge_w - 5, BADGE_H + 2), radius=18, fill=(0, 0, 0, 145))
    img.alpha_composite(shadow.filter(ImageFilter.GaussianBlur(radius=7)))
    glow = Image.new("RGBA", img.size, (0, 0, 0, 0))
    ImageDraw.Draw(glow, "RGBA").rounded_rectangle((2, 4, badge_w - 2, BADGE_H + 2), radius=18, outline=(*accent, 118), width=4)
    img.alpha_composite(glow.filter(ImageFilter.GaussianBlur(radius=5)))
    panel = _rounded_glass((badge_w, BADGE_H), 18, fill_alpha=176)
    grad = _vertical_gradient((badge_w, BADGE_H), surface_top, surface_bottom)
    grad.putalpha(228)
    panel.alpha_composite(grad)
    pd = ImageDraw.Draw(panel, "RGBA")
    pd.rounded_rectangle((0, 0, badge_w - 1, BADGE_H - 1), radius=18, outline=(*accent, 238), width=2)
    pd.rounded_rectangle((7, 7, badge_w - 8, BADGE_H - 8), radius=13, outline=(255, 255, 255, 42), width=1)
    pd.rounded_rectangle((10, 7, badge_w - 10, 12), radius=3, fill=(*accent, 148))
    img.alpha_composite(panel, (0, 0))

    text = _federer_first_scoreline(scoreline, winner)
    font = _fit_sport_font(draw, text, badge_w - 28, 34, 18, bold=True, condensed=True)
    draw.text((badge_w // 2 + 2, BADGE_H // 2 + 2), text, font=font, fill=(0, 0, 0, 150), anchor="mm")
    draw.text((badge_w // 2, BADGE_H // 2), text, font=font, fill=(248, 251, 255, 255), anchor="mm")
    return img


def _time_offset(year: int, month: int = 1) -> float:
    return (year - YEARS[0]) + (month - 1) / 12.0


@lru_cache(maxsize=1)
def _match_layout() -> tuple[tuple[MatchEntry, int, int], ...]:
    slots: list[tuple[MatchEntry, int, int]] = []
    placed_rects: list[tuple[int, int, int, int]] = []
    y_offsets = (28, -28, 84, -84, 140, -140, 196, -196)
    for match in MATCHES:
        base_y = int(round(_time_offset(match.year, match.month) * ROW_SPACING))
        badge_w = _badge_width_for_scoreline(match.scoreline)
        max_x = WIDTH - BADGE_MARGIN_X - badge_w
        lane_count = max(2, min(BADGE_LANES, int((WIDTH - BADGE_MARGIN_X * 2) / max(1, badge_w * 0.46))))
        x_candidates = [
            round(BADGE_MARGIN_X + idx * (max_x - BADGE_MARGIN_X) / max(1, lane_count - 1))
            for idx in range(lane_count)
        ]
        center_order = sorted(x_candidates, key=lambda item: (abs((item + badge_w / 2) - WIDTH / 2), item))

        best: tuple[float, int, tuple[int, int, int, int], int] | None = None
        for offset in y_offsets:
            y0 = base_y + offset - 42
            y1 = base_y + offset + BADGE_H + 16
            for x in center_order:
                rect = (x - BADGE_GAP_X, y0, x + badge_w + BADGE_GAP_X, y1)
                overlap = sum(
                    max(0, min(rect[2], other[2]) - max(rect[0], other[0]))
                    * max(0, min(rect[3], other[3]) - max(rect[1], other[1]))
                    for other in placed_rects
                )
                vertical_neighbors = [
                    other
                    for other in placed_rects
                    if max(0, min(rect[3] + 170, other[3]) - max(rect[1] - 170, other[1]))
                ]
                left_load = sum(max(0, min(other[2], WIDTH // 2) - max(other[0], 0)) for other in vertical_neighbors)
                right_load = sum(max(0, min(other[2], WIDTH) - max(other[0], WIDTH // 2)) for other in vertical_neighbors)
                side_load = left_load if x + badge_w / 2 < WIDTH / 2 else right_load
                preferred_center = WIDTH * (0.35 if match.winner == "left" else 0.65)
                score = (
                    overlap * 1000
                    + abs(offset) * 4
                    + abs((x + badge_w / 2) - preferred_center) * 0.95
                    + abs((x + badge_w / 2) - WIDTH / 2) * 0.22
                    + side_load * 0.42
                )
                if best is None or score < best[0]:
                    best = (score, x, rect, offset)

        assert best is not None
        _score, x, rect, offset = best
        placed_rects.append(rect)
        slots.append((match, x, offset))
    return tuple(slots)


@lru_cache(maxsize=1)
def _timeline_end_scroll() -> float:
    last_match_y = max(
        _time_offset(match.year, match.month) * ROW_SPACING + y_offset
        for match, _x, y_offset in _match_layout()
    )
    return TIMELINE_START_Y + last_match_y - SCORE_TOUCH_Y + TIMELINE_END_PADDING


def _timeline_y(year: int, month: int, progress: float) -> float:
    return TIMELINE_START_Y + _time_offset(year, month) * ROW_SPACING - _timeline_end_scroll() * progress


@lru_cache(maxsize=1)
def _timeline_layer() -> Image.Image:
    content_height = int(
        max(
            _time_offset(YEARS[-1], 12) * ROW_SPACING + 120,
            max(_time_offset(match.year, match.month) * ROW_SPACING + offset + BADGE_H + 80 for match, _x, offset in _match_layout()),
        )
    )
    layer = Image.new("RGBA", (WIDTH, content_height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer, "RGBA")
    year_font = _load_sport_font(40, bold=True, condensed=True)
    month_font = _load_sport_font(15, bold=True)

    for year in YEARS:
        for month in range(2, 13):
            y = _time_offset(year, month) * ROW_SPACING
            x = TIMELINE_LEFT
            while x < TIMELINE_RIGHT:
                draw.line((x, y, min(x + 5, TIMELINE_RIGHT), y), fill=(*THEME["grid"], 16), width=1)
                x += 30
            if month in (3, 6, 9, 12):
                draw.text(
                    (92, y - 1),
                    f"{month:02d}",
                    font=month_font,
                    fill=(221, 232, 248, 82),
                    anchor="rm",
                )

        y = _time_offset(year, 1) * ROW_SPACING
        year_glow = Image.new("RGBA", layer.size, (0, 0, 0, 0))
        gd = ImageDraw.Draw(year_glow, "RGBA")
        gd.line((0, y, WIDTH, y), fill=(*THEME["center"], 52), width=2)
        layer.alpha_composite(year_glow.filter(ImageFilter.GaussianBlur(radius=3)))
        x = TIMELINE_LEFT
        while x < TIMELINE_RIGHT:
            draw.line((x, y, min(x + 22, TIMELINE_RIGHT), y), fill=(233, 242, 255, 68), width=2)
            x += 34
        draw.text((49, y + 1), str(year), font=year_font, fill=(0, 0, 0, 130), anchor="lm")
        draw.text((46, y - 2), str(year), font=year_font, fill=(246, 250, 255, 228), anchor="lm")

    return layer


def _draw_timeline(frame: Image.Image, progress: float) -> None:
    layer = _timeline_layer()
    scroll = _timeline_end_scroll() * progress
    paste_y = int(round(TIMELINE_START_Y - scroll))
    visible_top = max(SCORE_TOUCH_Y, paste_y)
    visible_bottom = min(HEIGHT, paste_y + layer.height)
    if visible_bottom <= visible_top:
        return

    crop_top = visible_top - paste_y
    crop_bottom = crop_top + (visible_bottom - visible_top)
    crop = layer.crop((0, crop_top, WIDTH, crop_bottom))
    frame.alpha_composite(crop, (0, visible_top))

    draw = ImageDraw.Draw(frame, "RGBA")
    for match, x, offset in _match_layout():
        by = int(_timeline_y(match.year, match.month, progress) + offset)
        fade = _clamp((by - SCORE_TOUCH_Y) / 32.0)
        if fade <= 0 or by > HEIGHT + 190:
            continue

        accent = LEFT_PLAYER["accent"] if match.winner == "left" else RIGHT_PLAYER["accent"]
        surface_top, surface_bottom, surface_accent = _surface_colors(match.event)
        badge = _score_badge(match.scoreline, match.winner, accent, surface_top, surface_bottom)
        spawn_y = HEIGHT - 110
        spawn = _ease_out(_clamp((spawn_y - by) / 92.0))
        scale = 0.94 + 0.06 * spawn
        flash = _clamp((spawn_y - by) / 34.0) * _clamp((by - SCORE_TOUCH_Y) / 360.0)
        if scale < 0.995:
            nw, nh = int(badge.width * scale), int(badge.height * scale)
            badge = badge.resize((nw, nh), Image.Resampling.LANCZOS)
        draw_x = int(x)
        draw_x = max(BADGE_MARGIN_X, min(draw_x, WIDTH - badge.width - BADGE_MARGIN_X))
        if fade < 1:
            badge = badge.copy()
            badge.putalpha(ImageEnhance.Brightness(badge.getchannel("A")).enhance(fade))
        if flash > 0:
            flash_layer = Image.new("RGBA", (badge.width + 36, badge.height + 34), (0, 0, 0, 0))
            fd = ImageDraw.Draw(flash_layer, "RGBA")
            fd.rounded_rectangle((9, 8, badge.width + 27, badge.height + 20), radius=24, outline=(*surface_accent, int(95 * flash)), width=4)
            frame.alpha_composite(flash_layer.filter(ImageFilter.GaussianBlur(radius=8)), (draw_x - 18, by - 17))

        label = f"{match.month:02d}/{match.year} - {_short_event(match.event)}"
        label_alpha = fade * _clamp((HEIGHT - by - 4) / 70.0)
        label_font = _match_label_font(label, badge.width)
        if label_alpha > 0:
            draw.text(
                (draw_x + badge.width // 2 + 2, by - 20),
                label,
                font=label_font,
                fill=(0, 0, 0, int(150 * label_alpha)),
                anchor="mm",
            )
            draw.text(
                (draw_x + badge.width // 2, by - 22),
                label,
                font=label_font,
                fill=(235, 242, 255, int(230 * label_alpha)),
                anchor="mm",
                stroke_width=1,
                stroke_fill=(0, 0, 0, int(125 * label_alpha)),
            )
        frame.alpha_composite(badge, (draw_x, by))


def _scores_for_progress(progress: float) -> tuple[int, int]:
    left = 0
    right = 0
    for match, _x, offset in _match_layout():
        if _timeline_y(match.year, match.month, progress) + offset > SCORE_TOUCH_Y:
            continue
        if match.winner == "left":
            left += 1
        else:
            right += 1
    return left, right


def _touch_progress(match: MatchEntry, offset: int) -> float:
    target = (TIMELINE_START_Y + _time_offset(match.year, match.month) * ROW_SPACING + offset - SCORE_TOUCH_Y) / _timeline_end_scroll()
    return _clamp(target)


def _score_pulse(progress: float, winner: str) -> float:
    pulse = 0.0
    for match, _x, offset in _match_layout():
        if match.winner != winner:
            continue
        dt = progress - _touch_progress(match, offset)
        if 0 <= dt <= 0.045:
            pulse = max(pulse, 1.0 - dt / 0.045)
    return _ease_out(pulse)


@lru_cache(maxsize=None)
def _score_card_image(
    width: int,
    height: int,
    score: int,
    accent: tuple[int, int, int],
    pulse_bucket: int,
) -> Image.Image:
    pulse = pulse_bucket / 10.0
    scale = 1.0 + 0.035 * pulse
    w = int(width * scale)
    h = int(height * scale)
    pad = 34
    img = Image.new("RGBA", (w + pad * 2, h + pad * 2), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img, "RGBA")
    box = (pad, pad, pad + w, pad + h)

    shadow = Image.new("RGBA", img.size, (0, 0, 0, 0))
    sd = ImageDraw.Draw(shadow, "RGBA")
    sd.rounded_rectangle((box[0] + 12, box[1] + 16, box[2] + 12, box[3] + 16), radius=28, fill=(0, 0, 0, 150))
    img.alpha_composite(shadow.filter(ImageFilter.GaussianBlur(radius=12)))

    glow = Image.new("RGBA", img.size, (0, 0, 0, 0))
    gd = ImageDraw.Draw(glow, "RGBA")
    gd.rounded_rectangle((box[0] - 2, box[1] - 2, box[2] + 2, box[3] + 2), radius=30, outline=(*accent, 80 + int(115 * pulse)), width=5)
    img.alpha_composite(glow.filter(ImageFilter.GaussianBlur(radius=8)))

    panel = _vertical_gradient((w, h), (42, 54, 78), (9, 15, 28))
    mask = Image.new("L", panel.size, 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, panel.width - 1, panel.height - 1), radius=28, fill=218)
    panel.putalpha(mask)
    img.alpha_composite(panel, (box[0], box[1]))
    draw.rounded_rectangle(box, radius=28, outline=(255, 255, 255, 58), width=1)
    draw.rounded_rectangle((box[0] + 3, box[1] + 3, box[2] - 3, box[3] - 3), radius=24, outline=(*accent, 214), width=2)
    draw.rounded_rectangle((box[0] + 10, box[1] + 10, box[2] - 10, box[1] + 18), radius=4, fill=(*accent, 176))

    score_font = _fit_sport_font(draw, str(score), int(w - 46), 118, 62, bold=True, condensed=True)
    _draw_glow_text(
        img,
        (img.width // 2, img.height // 2 + 12),
        str(score),
        score_font,
        (248, 251, 255),
        stroke_width=2,
        glow_color=accent,
        glow_radius=8,
        glow_alpha=75 + int(70 * pulse),
    )
    return img


def _draw_score_card(
    frame: Image.Image,
    box: tuple[int, int, int, int],
    score: int,
    accent: tuple[int, int, int],
    pulse: float,
) -> None:
    pulse_bucket = int(round(_clamp(pulse) * 10))
    card = _score_card_image(box[2] - box[0], box[3] - box[1], score, accent, pulse_bucket)
    cx = (box[0] + box[2]) // 2
    cy = (box[1] + box[3]) // 2
    frame.alpha_composite(card, (cx - card.width // 2, cy - card.height // 2))


@lru_cache(maxsize=1)
def _scoreboard_base() -> Image.Image:
    img = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img, "RGBA")
    line = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    ld = ImageDraw.Draw(line, "RGBA")
    ld.line((0, SCOREBOARD_LINE_Y, WIDTH, SCOREBOARD_LINE_Y), fill=(*THEME["center"], 116), width=2)
    img.alpha_composite(line.filter(ImageFilter.GaussianBlur(radius=3)))
    draw.line((0, SCOREBOARD_LINE_Y, WIDTH, SCOREBOARD_LINE_Y), fill=(236, 246, 255, 130), width=1)

    small_font = _load_sport_font(28, bold=True)
    draw.text((271, SCOREBOARD_TOP - 31), "VICTOIRES", font=small_font, fill=(226, 236, 250, 228), anchor="mm")
    draw.text((809, SCOREBOARD_TOP - 31), "VICTOIRES", font=small_font, fill=(226, 236, 250, 228), anchor="mm")
    draw.rounded_rectangle((497, SCOREBOARD_TOP - 45, 583, SCOREBOARD_TOP - 10), radius=17, fill=(*THEME["panel_hi"], 172), outline=(255, 255, 255, 60), width=1)
    draw.text((WIDTH // 2, SCOREBOARD_TOP - 27), "H2H", font=_load_sport_font(22, bold=True), fill=(242, 248, 255, 235), anchor="mm")
    return img


def _draw_scoreboard(frame: Image.Image, left_score: int, right_score: int, progress: float) -> None:
    left_box = (148, SCOREBOARD_TOP - 2, 394, SCOREBOARD_BOTTOM + 10)
    right_box = (686, SCOREBOARD_TOP - 2, 932, SCOREBOARD_BOTTOM + 10)
    frame.alpha_composite(_scoreboard_base())

    _draw_score_card(frame, left_box, left_score, LEFT_PLAYER["accent"], _score_pulse(progress, "left"))
    _draw_score_card(frame, right_box, right_score, RIGHT_PLAYER["accent"], _score_pulse(progress, "right"))


def render_video(output_path: Path, audio_path: Path, duration: float, fps: int) -> Path:
    if duration > 120:
        raise ValueError("La video doit durer 120 secondes maximum.")
    portraits = {
        "left": _load_portrait(PHOTOS_DIR / LEFT_PLAYER["photo"], "RF"),
        "right": _load_portrait(PHOTOS_DIR / RIGHT_PLAYER["photo"], "ND"),
    }
    static_layer = _draw_static_layer(portraits)

    def make_frame(t: float) -> np.ndarray:
        progress = _clamp(t / max(duration, 1e-6))
        frame = static_layer.copy()
        _draw_header(frame, progress)
        _draw_timeline(frame, progress)
        _draw_scoreboard(frame, *_scores_for_progress(progress), progress)
        return np.asarray(frame.convert("RGB"))

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
            bitrate="10000k",
            preset="veryfast",
            threads=4,
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a Federer vs Djokovic H2H score timeline Shorts video.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--audio", type=Path, default=DEFAULT_AUDIO)
    parser.add_argument("--duration", type=float, default=TOTAL_DURATION)
    parser.add_argument("--fps", type=int, default=FPS)
    args = parser.parse_args()
    render_video(args.output, args.audio, args.duration, args.fps)
    print(f"[video_generator] Federer vs Djokovic H2H score timeline generated -> {args.output}")


if __name__ == "__main__":
    main()
