from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from moviepy import AudioFileClip, CompositeAudioClip, VideoClip
from moviepy.audio.fx import AudioFadeIn, AudioFadeOut
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon, box, shape
from shapely.geometry.base import BaseGeometry


MODULE_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = MODULE_ROOT.parent.parent
PERIODS_GEOJSON = MODULE_ROOT / "data" / "raw" / "france_territory_periods.geojson"
BASEMAP_GEOJSON = MODULE_ROOT / "data" / "raw" / "ne_50m_admin_0_countries.geojson"
DEFAULT_AUDIO = REPO_ROOT / "data" / "raw" / "audio" / "audio.mp3"
DEFAULT_OUTPUT = MODULE_ROOT / "data" / "processed" / "france_territory_2000_years_360s_60fps.mp4"
DEFAULT_PREVIEW = MODULE_ROOT / "data" / "processed" / "france_territory_preview.png"

WIDTH = 1920
HEIGHT = 1080
DURATION = 360.0
FPS = 60
MAP_BOUNDS = (-10.5, 14.5, 40.0, 55.8)
MAP_BOX = (62, 174, 1265, 825)
TRANSITION_SHARE = 0.18
GOLD = (232, 190, 103, 255)
CREAM = (247, 241, 230, 255)


@dataclass(frozen=True)
class Period:
    properties: dict[str, str | int]
    geometry: BaseGeometry


CITY_COORDS = {
    "Lugdunum": (4.8357, 45.7640),
    "Soissons": (3.3236, 49.3817),
    "Paris": (2.3522, 48.8566),
    "Versailles": (2.1301, 48.8014),
}


def load_font(size: int, *, bold: bool = False, display: bool = False) -> ImageFont.ImageFont:
    candidates = []
    if display:
        candidates.extend(
            [
                "C:/Windows/Fonts/georgiab.ttf" if bold else "C:/Windows/Fonts/georgia.ttf",
                "C:/Windows/Fonts/constanb.ttf" if bold else "C:/Windows/Fonts/constan.ttf",
            ]
        )
    candidates.extend(
        [
            "C:/Windows/Fonts/bahnschrift.ttf",
            "C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf",
            "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        ]
    )
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def text_width(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> int:
    bounds = draw.textbbox((0, 0), text, font=font)
    return bounds[2] - bounds[0]


def text_height(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> int:
    bounds = draw.textbbox((0, 0), text, font=font)
    return bounds[3] - bounds[1]


def wrap_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.ImageFont,
    max_width: int,
    max_lines: int,
) -> list[str]:
    words = text.split()
    if not words:
        return []
    lines: list[str] = []
    current: list[str] = []
    for word in words:
        candidate = " ".join([*current, word])
        if text_width(draw, candidate, font) <= max_width:
            current.append(word)
            continue
        if current:
            lines.append(" ".join(current))
        current = [word]
    if current:
        lines.append(" ".join(current))
    return lines


def fit_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    max_width: int,
    max_lines: int,
    sizes: tuple[int, ...],
    *,
    bold: bool = False,
    display: bool = False,
) -> tuple[ImageFont.ImageFont, list[str]]:
    for size in sizes:
        font = load_font(size, bold=bold, display=display)
        lines = wrap_text(draw, text, font, max_width, max_lines)
        if lines and all(text_width(draw, line, font) <= max_width for line in lines):
            if len(lines) <= max_lines:
                return font, lines
    for size in range(sizes[-1] - 1, 9, -1):
        font = load_font(size, bold=bold, display=display)
        lines = wrap_text(draw, text, font, max_width, max_lines)
        if lines and len(lines) <= max_lines and all(text_width(draw, line, font) <= max_width for line in lines):
            return font, lines
    font = load_font(10, bold=bold, display=display)
    return font, wrap_text(draw, text, font, max_width, max_lines)


def draw_lines(
    draw: ImageDraw.ImageDraw,
    lines: list[str],
    position: tuple[int, int],
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int, int] | str,
    *,
    line_gap: int = 5,
) -> int:
    x, y = position
    step = text_height(draw, "Ag", font) + line_gap
    for line in lines:
        draw.text((x, y), line, font=font, fill=fill)
        y += step
    return y


def load_basemap(path: Path) -> tuple[list[tuple[str, BaseGeometry]], BaseGeometry]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    viewport = box(*MAP_BOUNDS)
    countries: list[tuple[str, BaseGeometry]] = []
    france_geometry: BaseGeometry | None = None
    for feature in payload["features"]:
        geometry = shape(feature["geometry"])
        if not geometry.intersects(viewport):
            continue
        clipped = geometry.intersection(viewport)
        name = str(feature["properties"].get("ADMIN") or feature["properties"].get("NAME") or "")
        countries.append((name, clipped))
        if name == "France":
            france_geometry = clipped
    if france_geometry is None:
        raise RuntimeError("France geometry not found in Natural Earth basemap.")
    return countries, france_geometry


def load_periods(path: Path, modern_france: BaseGeometry) -> list[Period]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    periods: list[Period] = []
    for feature in payload["features"]:
        properties = feature["properties"]
        if properties.get("geometry_mode") == "modern_france":
            geometry = modern_france
        else:
            geometry = shape(feature["geometry"])
        periods.append(Period(properties=properties, geometry=geometry))
    periods.sort(key=lambda period: int(period.properties["start_year"]))
    return periods


def iter_polygons(geometry: BaseGeometry):
    if isinstance(geometry, Polygon):
        yield geometry
    elif isinstance(geometry, MultiPolygon):
        yield from geometry.geoms
    elif isinstance(geometry, GeometryCollection):
        for item in geometry.geoms:
            yield from iter_polygons(item)


def project(lon: float, lat: float) -> tuple[int, int]:
    min_lon, max_lon, min_lat, max_lat = MAP_BOUNDS
    x0, y0, x1, y1 = MAP_BOX
    x = x0 + ((lon - min_lon) / (max_lon - min_lon)) * (x1 - x0)
    y = y1 - ((lat - min_lat) / (max_lat - min_lat)) * (y1 - y0)
    return int(round(x)), int(round(y))


def geometry_paths(geometry: BaseGeometry) -> list[list[tuple[int, int]]]:
    paths: list[list[tuple[int, int]]] = []
    for polygon in iter_polygons(geometry):
        paths.append([project(lon, lat) for lon, lat in polygon.exterior.coords])
    return paths


def make_background() -> Image.Image:
    xx = np.linspace(0, 1, WIDTH, dtype=np.float32)
    yy = np.linspace(0, 1, HEIGHT, dtype=np.float32)
    gx, gy = np.meshgrid(xx, yy)
    navy = np.array([3, 16, 29], dtype=np.float32)
    lower = np.array([12, 23, 31], dtype=np.float32)
    amber = np.array([115, 72, 24], dtype=np.float32)
    mix = np.clip(0.72 * gy + 0.05 * gx, 0, 1)
    glow = np.exp(-(((gx - 0.58) / 0.25) ** 2 + ((gy - 0.09) / 0.13) ** 2))
    image = (
        navy[None, None, :] * (1 - mix[..., None])
        + lower[None, None, :] * mix[..., None]
        + amber[None, None, :] * (0.14 * glow[..., None])
    )
    frame = Image.fromarray(np.clip(image, 0, 255).astype(np.uint8)).convert("RGBA")
    texture = Image.new("RGBA", frame.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(texture, "RGBA")
    for y in range(0, HEIGHT, 42):
        draw.line((0, y, WIDTH, y + 14), fill=(255, 255, 255, 5), width=1)
    for x in range(0, WIDTH, 56):
        draw.line((x, 0, x + 18, HEIGHT), fill=(233, 193, 110, 4), width=1)
    texture = texture.filter(ImageFilter.GaussianBlur(0.5))
    frame.alpha_composite(texture)
    return frame


def draw_compass(draw: ImageDraw.ImageDraw, center: tuple[int, int], radius: int) -> None:
    cx, cy = center
    draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), outline=GOLD, width=2)
    draw.line((cx, cy - radius - 10, cx, cy + radius + 10), fill=GOLD, width=2)
    draw.line((cx - radius - 10, cy, cx + radius + 10, cy), fill=GOLD, width=2)
    draw.polygon([(cx, cy - radius - 14), (cx - 7, cy - 4), (cx + 7, cy - 4)], fill=GOLD)
    draw.polygon([(cx + radius + 14, cy), (cx + 4, cy - 7), (cx + 4, cy + 7)], fill=(233, 193, 110, 170))
    draw.polygon([(cx, cy + radius + 14), (cx - 7, cy + 4), (cx + 7, cy + 4)], fill=(233, 193, 110, 145))
    draw.polygon([(cx - radius - 14, cy), (cx - 4, cy - 7), (cx - 4, cy + 7)], fill=(233, 193, 110, 170))
    draw.ellipse((cx - 5, cy - 5, cx + 5, cy + 5), fill=CREAM)


def draw_round_panel(
    frame: Image.Image,
    box_coords: tuple[int, int, int, int],
    *,
    radius: int = 24,
    fill: tuple[int, int, int, int] = (7, 24, 39, 222),
    outline: tuple[int, int, int, int] = (233, 193, 110, 150),
) -> None:
    shadow = Image.new("RGBA", frame.size, (0, 0, 0, 0))
    shadow_draw = ImageDraw.Draw(shadow, "RGBA")
    x0, y0, x1, y1 = box_coords
    shadow_draw.rounded_rectangle((x0, y0 + 10, x1, y1 + 10), radius=radius, fill=(0, 0, 0, 110))
    shadow = shadow.filter(ImageFilter.GaussianBlur(14))
    frame.alpha_composite(shadow)
    draw = ImageDraw.Draw(frame, "RGBA")
    draw.rounded_rectangle(box_coords, radius=radius, fill=fill, outline=outline, width=2)
    draw.rounded_rectangle(
        (x0 + 1, y0 + 1, x1 - 1, y1 - 1),
        radius=radius - 1,
        outline=(255, 255, 255, 10),
        width=1,
    )


def draw_map_base(frame: Image.Image, countries: list[tuple[str, BaseGeometry]]) -> None:
    draw_round_panel(frame, MAP_BOX, radius=28, fill=(5, 22, 37, 232), outline=(233, 193, 110, 118))
    content = Image.new("RGBA", frame.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(content, "RGBA")
    x0, y0, x1, y1 = MAP_BOX
    for lon in range(-10, 16, 5):
        x, _ = project(lon, MAP_BOUNDS[2])
        draw.line((x, y0 + 1, x, y1 - 1), fill=(150, 184, 198, 18), width=1)
    for lat in range(40, 57, 4):
        _, y = project(MAP_BOUNDS[0], lat)
        draw.line((x0 + 1, y, x1 - 1, y), fill=(150, 184, 198, 18), width=1)

    for _, geometry in countries:
        for points in geometry_paths(geometry):
            if len(points) >= 3:
                draw.polygon(points, fill=(20, 43, 57, 220), outline=(113, 145, 156, 105))

    sea_font = load_font(18, display=True)
    draw.text((120, 520), "OCÉAN ATLANTIQUE", font=sea_font, fill=(129, 164, 180, 85))
    draw.text((450, 205), "MANCHE", font=sea_font, fill=(129, 164, 180, 80))
    draw.text((720, 775), "MÉDITERRANÉE", font=sea_font, fill=(129, 164, 180, 85))

    mask = Image.new("L", frame.size, 0)
    mask_draw = ImageDraw.Draw(mask)
    mask_draw.rounded_rectangle((x0 + 2, y0 + 2, x1 - 2, y1 - 2), radius=26, fill=255)
    content.putalpha(ImageChops.multiply(content.getchannel("A"), mask))
    frame.alpha_composite(content)


def draw_territory(
    frame: Image.Image,
    geometry: BaseGeometry,
    color: str,
    *,
    alpha: int = 225,
    pulse: float = 0.0,
) -> None:
    rgb = tuple(int(color[index : index + 2], 16) for index in (1, 3, 5))
    glow = Image.new("RGBA", frame.size, (0, 0, 0, 0))
    glow_draw = ImageDraw.Draw(glow, "RGBA")
    paths = geometry_paths(geometry)
    for points in paths:
        if len(points) >= 3:
            glow_draw.polygon(points, fill=(*rgb, 70 + int(25 * pulse)), outline=(*rgb, 180), width=5)
    glow = glow.filter(ImageFilter.GaussianBlur(12 + int(3 * pulse)))
    frame.alpha_composite(glow)

    territory = Image.new("RGBA", frame.size, (0, 0, 0, 0))
    territory_draw = ImageDraw.Draw(territory, "RGBA")
    for points in paths:
        if len(points) >= 3:
            territory_draw.polygon(points, fill=(*rgb, alpha), outline=(255, 222, 144, min(255, alpha + 25)))
            territory_draw.line(points, fill=(255, 229, 168, min(255, alpha + 25)), width=3, joint="curve")
    frame.alpha_composite(territory)


def draw_capital(frame: Image.Image, capital: str) -> None:
    coords = CITY_COORDS.get(capital)
    if coords is None:
        return
    draw = ImageDraw.Draw(frame, "RGBA")
    x, y = project(*coords)
    draw.ellipse((x - 13, y - 13, x + 13, y + 13), fill=(5, 18, 30, 235), outline=GOLD, width=3)
    draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=CREAM)
    font = load_font(17, bold=True)
    label_x = x + 17
    if label_x + text_width(draw, capital, font) > MAP_BOX[2] - 15:
        label_x = x - text_width(draw, capital, font) - 17
    draw.rounded_rectangle(
        (label_x - 7, y - 15, label_x + text_width(draw, capital, font) + 7, y + 14),
        radius=8,
        fill=(4, 18, 31, 205),
    )
    draw.text((label_x, y - 12), capital, font=font, fill=CREAM)


def draw_overseas_inset(frame: Image.Image, period: Period) -> None:
    if period.properties["id"] != "contemporary_france":
        return
    draw = ImageDraw.Draw(frame, "RGBA")
    inset = (86, 690, 405, 796)
    draw.rounded_rectangle(inset, radius=16, fill=(5, 20, 33, 220), outline=(233, 193, 110, 125), width=2)
    title_font = load_font(14, bold=True)
    label_font = load_font(12)
    draw.text((103, 703), "OUTRE-MER", font=title_font, fill=GOLD)
    labels = ["Guyane", "Antilles", "Réunion", "Mayotte", "Pacifique"]
    x = 109
    for label in labels:
        draw.ellipse((x, 741, x + 11, 752), fill=GOLD)
        draw.text((x - 9, 760), label, font=label_font, fill=(215, 225, 232, 235))
        x += 58


def draw_header(frame: Image.Image) -> None:
    draw = ImageDraw.Draw(frame, "RGBA")
    draw_compass(draw, (72, 72), 30)
    brand_font = load_font(50, bold=True, display=True)
    subtitle_font = load_font(16, bold=True)
    draw.text((120, 39), "HISTOVISION", font=brand_font, fill=GOLD)
    draw.text((123, 99), "L'HISTOIRE EN CARTES ET EN CHIFFRES", font=subtitle_font, fill=(218, 191, 132, 235))
    right_font = load_font(18)
    right_bold = load_font(25, bold=True)
    label = "DES GAULES À LA FRANCE"
    draw.text((WIDTH - 72 - text_width(draw, label, right_font), 45), label, font=right_font, fill=CREAM)
    years = "52 AV. J.-C.  |  AUJOURD'HUI"
    draw.text((WIDTH - 72 - text_width(draw, years, right_bold), 75), years, font=right_bold, fill=GOLD)


def draw_info_panel(frame: Image.Image, period: Period, index: int, total: int) -> None:
    panel = (1300, 174, 1858, 825)
    draw_round_panel(frame, panel, radius=28, fill=(7, 24, 39, 232), outline=(233, 193, 110, 150))
    draw = ImageDraw.Draw(frame, "RGBA")
    props = period.properties
    x = panel[0] + 30
    max_width = panel[2] - panel[0] - 60

    badge_font = load_font(15, bold=True)
    badge = f"ÉTAPE {index + 1:02d} / {total:02d}"
    badge_width = text_width(draw, badge, badge_font) + 26
    draw.rounded_rectangle((x, 198, x + badge_width, 230), radius=12, fill=(197, 142, 48, 40), outline=(233, 193, 110, 130))
    draw.text((x + 13, 205), badge, font=badge_font, fill=GOLD)

    year_font, year_lines = fit_text(
        draw,
        str(props["year_label"]),
        max_width,
        1,
        (48, 44, 40, 36, 32, 28),
        bold=True,
        display=True,
    )
    draw.text((x, 246), year_lines[0], font=year_font, fill=(255, 219, 139, 255))

    title_font, title_lines = fit_text(
        draw,
        str(props["title"]),
        max_width,
        2,
        (39, 36, 33, 30, 27),
        bold=True,
        display=True,
    )
    next_y = draw_lines(draw, title_lines, (x, 309), title_font, CREAM, line_gap=2)

    regime_font, regime_lines = fit_text(
        draw,
        str(props["regime"]),
        max_width,
        2,
        (20, 18, 16),
        bold=True,
    )
    next_y = draw_lines(draw, regime_lines, (x, next_y + 8), regime_font, GOLD, line_gap=2)

    divider_y = next_y + 15
    draw.line((x, divider_y, panel[2] - 30, divider_y), fill=(233, 193, 110, 75), width=2)

    section_font = load_font(14, bold=True)
    body_font = load_font(19)
    draw.text((x, divider_y + 18), "CE QUE MONTRE LA CARTE", font=section_font, fill=GOLD)
    scope_font, scope_lines = fit_text(draw, str(props["scope"]), max_width, 2, (20, 18, 16), bold=True)
    next_y = draw_lines(draw, scope_lines, (x, divider_y + 43), scope_font, CREAM, line_gap=4)

    draw.text((x, next_y + 18), "MUTATION MAJEURE", font=section_font, fill=GOLD)
    event_font, event_lines = fit_text(draw, str(props["event"]), max_width, 4, (20, 18, 16))
    next_y = draw_lines(draw, event_lines, (x, next_y + 43), event_font, (220, 229, 235, 255), line_gap=5)

    draw.text((x, next_y + 18), "À RETENIR", font=section_font, fill=GOLD)
    detail_font, detail_lines = fit_text(draw, str(props["detail"]), max_width, 5, (18, 17, 16, 15))
    next_y = draw_lines(draw, detail_lines, (x, next_y + 43), detail_font, (188, 205, 216, 255), line_gap=5)

    capital_font = load_font(16, bold=True)
    capital_text = f"CAPITALE / CENTRE : {props['capital']}"
    capital_y = min(panel[3] - 52, next_y + 22)
    draw.rounded_rectangle((x, capital_y, panel[2] - 30, capital_y + 35), radius=11, fill=(233, 193, 110, 22))
    draw.text((x + 13, capital_y + 8), capital_text, font=capital_font, fill=(240, 210, 145, 255))


def draw_timeline(frame: Image.Image, periods: list[Period], current_index: int) -> None:
    draw = ImageDraw.Draw(frame, "RGBA")
    x0, x1 = 72, WIDTH - 72
    y = 908
    draw.line((x0, y, x1, y), fill=(118, 146, 160, 90), width=4)
    gap = (x1 - x0) / max(1, len(periods) - 1)
    for index, period in enumerate(periods):
        x = int(round(x0 + index * gap))
        active = index == current_index
        past = index < current_index
        radius = 9 if active else 5
        color = GOLD if active else ((205, 218, 225, 190) if past else (95, 120, 133, 130))
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)

    current_x = int(round(x0 + current_index * gap))
    label = str(periods[current_index].properties["year_label"])
    label_font = load_font(18, bold=True)
    label_width = text_width(draw, label, label_font) + 30
    bubble_x = max(x0, min(x1 - label_width, current_x - label_width // 2))
    draw.rounded_rectangle((bubble_x, y - 55, bubble_x + label_width, y - 19), radius=12, fill=(5, 22, 37, 245), outline=GOLD, width=2)
    draw.text((bubble_x + 15, y - 48), label, font=label_font, fill=CREAM)
    draw.line((current_x, y - 19, current_x, y - 8), fill=GOLD, width=2)

    marker_font = load_font(15, bold=True)
    markers = {
        0: "GAULE",
        2: "843",
        6: "1532",
        10: "1812",
        12: "1860",
        14: "AUJ.",
    }
    for index, text in markers.items():
        x = int(round(x0 + index * gap))
        draw.text((x - text_width(draw, text, marker_font) // 2, y + 18), text, font=marker_font, fill=(184, 201, 211, 220))


def draw_footer(frame: Image.Image) -> None:
    draw = ImageDraw.Draw(frame, "RGBA")
    y = 1010
    draw.line((72, y, 760, y), fill=(233, 193, 110, 105), width=2)
    draw.line((1160, y, WIDTH - 72, y), fill=(233, 193, 110, 105), width=2)
    draw_compass(draw, (886, y), 17)
    footer_font = load_font(24, bold=True, display=True)
    draw.text((920, y - 16), "HISTOVISION", font=footer_font, fill=GOLD)
    note_font = load_font(12)
    note = "Reconstitutions schématiques • sources et méthode dans history/france_territory/SOURCES.md"
    draw.text((WIDTH // 2 - text_width(draw, note, note_font) // 2, 1048), note, font=note_font, fill=(147, 166, 178, 210))


def build_audio_track(audio_path: Path, duration: float):
    base = AudioFileClip(str(audio_path))
    if base.duration >= duration:
        clip = base.subclipped(0, duration)
        clip = clip.with_effects([AudioFadeOut(min(8.0, duration))])
        return clip, [base]
    crossfade = min(5.0, max(0.5, base.duration / 6))
    step = max(0.1, base.duration - crossfade)
    loops = int(math.ceil(duration / step)) + 1
    clips = []
    for index in range(loops):
        item = base.with_start(index * step)
        if index:
            item = item.with_effects([AudioFadeIn(crossfade)])
        item = item.with_effects([AudioFadeOut(crossfade)])
        clips.append(item)
    mixed = CompositeAudioClip(clips).with_duration(duration)
    mixed = mixed.with_effects([AudioFadeOut(min(8.0, duration))])
    return mixed, [base]


class TerritoryRenderer:
    def __init__(self) -> None:
        self.countries, modern_france = load_basemap(BASEMAP_GEOJSON)
        self.periods = load_periods(PERIODS_GEOJSON, modern_france)
        self.background = make_background()
        self.frame_cache: dict[int, np.ndarray] = {}

    def frame_for_period(self, index: int, *, pulse_phase: float = 0.0) -> np.ndarray:
        index = max(0, min(len(self.periods) - 1, index))
        cached = self.frame_cache.get(index)
        if cached is not None:
            return cached
        period = self.periods[index]
        frame = self.background.copy()
        draw_header(frame)
        draw_map_base(frame, self.countries)
        pulse = 0.65
        draw_territory(frame, period.geometry, str(period.properties["color"]), pulse=pulse)
        draw_capital(frame, str(period.properties["capital"]))
        draw_overseas_inset(frame, period)
        draw_info_panel(frame, period, index, len(self.periods))
        draw_timeline(frame, self.periods, index)
        draw_footer(frame)
        rendered = np.array(frame.convert("RGB"))
        self.frame_cache[index] = rendered
        return rendered

    def make_frame(self, t: float, duration: float) -> np.ndarray:
        period_duration = duration / len(self.periods)
        index = min(len(self.periods) - 1, int(t / period_duration))
        local = (t - index * period_duration) / period_duration
        frame = self.frame_for_period(index, pulse_phase=local)
        if index >= len(self.periods) - 1 or local < 1.0 - TRANSITION_SHARE:
            return frame

        transition = (local - (1.0 - TRANSITION_SHARE)) / TRANSITION_SHARE
        transition = transition * transition * (3.0 - 2.0 * transition)
        next_frame = self.frame_for_period(index + 1, pulse_phase=transition)
        blended = frame.astype(np.float32) * (1.0 - transition) + next_frame.astype(np.float32) * transition
        return np.clip(blended, 0, 255).astype(np.uint8)

    def period_index_for_year(self, year: int) -> int:
        for index in range(len(self.periods) - 1, -1, -1):
            period = self.periods[index]
            start = int(period.properties["start_year"])
            end = int(period.properties["end_year"])
            if start <= year <= end:
                return index
        return min(
            range(len(self.periods)),
            key=lambda idx: abs(int(self.periods[idx].properties["start_year"]) - year),
        )


def render_preview(renderer: TerritoryRenderer, year: int, output: Path) -> Path:
    index = renderer.period_index_for_year(year)
    output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(renderer.frame_for_period(index, pulse_phase=0.35)).save(output)
    return output


def render_video(renderer: TerritoryRenderer, output: Path, duration: float, fps: int, audio_path: Path) -> Path:
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    output.parent.mkdir(parents=True, exist_ok=True)
    audio_clip, keep_alive = build_audio_track(audio_path, duration)
    clip = VideoClip(lambda t: renderer.make_frame(t, duration), duration=duration).with_audio(audio_clip)
    clip.write_videofile(str(output), fps=fps, codec="libx264", audio_codec="aac")
    clip.close()
    audio_clip.close()
    for item in keep_alive:
        item.close()
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate an animated history of the French territory.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--duration", type=float, default=DURATION)
    parser.add_argument("--fps", type=int, default=FPS)
    parser.add_argument("--audio", type=Path, default=DEFAULT_AUDIO)
    parser.add_argument("--preview-year", type=int)
    parser.add_argument("--preview-output", type=Path, default=DEFAULT_PREVIEW)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    renderer = TerritoryRenderer()
    if args.preview_year is not None:
        output = render_preview(renderer, args.preview_year, args.preview_output)
        print(f"[history] France territory preview generated -> {output}")
        return
    output = render_video(renderer, args.output, args.duration, args.fps, args.audio)
    print(f"[history] France territory video generated -> {output}")


if __name__ == "__main__":
    main()
