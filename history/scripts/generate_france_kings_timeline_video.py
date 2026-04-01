from __future__ import annotations

import argparse
import csv
import re
import unicodedata
from pathlib import Path

import numpy as np
from moviepy import VideoClip
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps


PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_CSV = PROJECT_ROOT / "data" / "raw" / "france_kings_reigns.csv"
PORTRAITS_DIR = PROJECT_ROOT / "data" / "raw" / "portraits"
OUTPUT_MP4 = PROJECT_ROOT / "data" / "processed" / "france_kings_timeline_481_1848_60s.mp4"

WIDTH = 1920
HEIGHT = 1080
FPS = 30
DURATION = 60.0


def slugify(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", "_", normalized.lower()).strip("_")


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf",
    ]
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def hex_to_rgb(color: str) -> tuple[int, int, int]:
    color = color.lstrip("#")
    return tuple(int(color[i : i + 2], 16) for i in (0, 2, 4))


def make_background() -> Image.Image:
    xx = np.linspace(0, 1, WIDTH, dtype=np.float32)
    yy = np.linspace(0, 1, HEIGHT, dtype=np.float32)
    gx, gy = np.meshgrid(xx, yy)
    top = np.array([11, 24, 43], dtype=np.float32)
    bottom = np.array([52, 27, 21], dtype=np.float32)
    gold = np.array([225, 181, 101], dtype=np.float32)
    mix = np.clip(0.56 * gy + 0.18 * gx, 0, 1)
    glow = np.exp(-(((gx - 0.73) / 0.18) ** 2 + ((gy - 0.24) / 0.20) ** 2))
    img = np.clip(
        top[None, None, :] * (1.0 - mix[..., None])
        + bottom[None, None, :] * mix[..., None]
        + gold[None, None, :] * (0.10 * glow[..., None]),
        0,
        255,
    ).astype(np.uint8)
    frame = Image.fromarray(img).convert("RGBA")
    overlay = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")
    draw.rounded_rectangle((24, 24, WIDTH - 24, HEIGHT - 24), radius=36, outline=(255, 255, 255, 22), width=2)
    draw.line((84, 900, WIDTH - 84, 900), fill=(255, 255, 255, 16), width=2)
    draw.arc((1240, 80, WIDTH - 80, 580), start=200, end=25, fill=(240, 201, 120, 26), width=3)
    overlay = overlay.filter(ImageFilter.GaussianBlur(radius=3))
    frame.alpha_composite(overlay)
    return frame


def build_segments(rows: list[dict[str, str]]) -> list[dict[str, str | int]]:
    segments = []
    for row in rows:
        segments.append(
            {
                "start_year": int(row["start_year"]),
                "end_year": int(row["end_year"]),
                "display_name": row["display_name"],
                "dynasty": row["dynasty"],
                "house_color": row["house_color"],
                "notes": row["notes"],
                "fait_1": row.get("fait_1", ""),
                "fait_2": row.get("fait_2", ""),
                "fait_3": row.get("fait_3", ""),
            }
        )
    segments.extend(
        [
            {
                "start_year": 738,
                "end_year": 742,
                "display_name": "Sans roi",
                "dynasty": "Transition",
                "house_color": "#6c757d",
                "notes": "Interrègne avant Childéric III",
                "fait_1": "La succession monarchique est interrompue pendant cet interrègne.",
                "fait_2": "Le pouvoir réel est alors dominé par les maires du palais.",
                "fait_3": "Cette transition prépare le retour d'un roi mérovingien en 743.",
            },
            {
                "start_year": 841,
                "end_year": 842,
                "display_name": "Sans roi",
                "dynasty": "Transition",
                "house_color": "#6c757d",
                "notes": "Conflit de partage après Louis le Pieux",
                "fait_1": "La guerre entre héritiers suit la mort de Louis le Pieux.",
                "fait_2": "Cette crise mène au partage de Verdun en 843.",
                "fait_3": "La Francie occidentale émerge alors comme cadre de la future France.",
            },
            {
                "start_year": 1793,
                "end_year": 1813,
                "display_name": "Sans roi",
                "dynasty": "Révolution et Empire",
                "house_color": "#6c757d",
                "notes": "Monarchie interrompue",
                "fait_1": "La monarchie est abolie après la Révolution française.",
                "fait_2": "La Première République puis le Consulat transforment le régime.",
                "fait_3": "Napoléon Ier fonde ensuite l'Empire avant la Restauration bourbonienne.",
            },
        ]
    )
    segments.sort(key=lambda item: int(item["start_year"]))
    return segments


def find_segment(segments: list[dict[str, str | int]], year: int) -> dict[str, str | int]:
    for segment in segments:
        if int(segment["start_year"]) <= year <= int(segment["end_year"]):
            return segment
    return segments[-1]


def load_portrait(name: str) -> Image.Image | None:
    path = PORTRAITS_DIR / f"{slugify(name)}.png"
    if not path.exists():
        return None
    image = Image.open(path).convert("RGBA")
    return ImageOps.fit(image, (340, 340), method=Image.Resampling.LANCZOS, centering=(0.5, 0.35))


def draw_glow_text(frame: Image.Image, position: tuple[int, int], text: str, font: ImageFont.ImageFont, fill: tuple[int, int, int], glow: tuple[int, int, int]) -> None:
    x, y = position
    glow_layer = Image.new("RGBA", frame.size, (0, 0, 0, 0))
    glow_draw = ImageDraw.Draw(glow_layer, "RGBA")
    for dx, dy in ((-3, 0), (3, 0), (0, -3), (0, 3), (0, 0)):
        glow_draw.text((x + dx, y + dy), text, font=font, fill=(*glow, 70))
    glow_layer = glow_layer.filter(ImageFilter.GaussianBlur(radius=4))
    frame.alpha_composite(glow_layer)
    ImageDraw.Draw(frame, "RGBA").text((x, y), text, font=font, fill=fill)


def wrap_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int, max_lines: int) -> list[str]:
    words = text.split()
    if not words:
        return []
    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        if draw.textbbox((0, 0), candidate, font=font)[2] <= max_width:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    if len(lines) <= max_lines:
        return lines
    trimmed = lines[:max_lines]
    last = trimmed[-1]
    while last and draw.textbbox((0, 0), last + "...", font=font)[2] > max_width:
        last = " ".join(last.split()[:-1])
        if not last:
            break
    trimmed[-1] = (last + "...") if last else "..."
    return trimmed


def render_video(rows: list[dict[str, str]], output_path: Path, duration: float, fps: int) -> Path:
    segments = build_segments(rows)
    background = make_background()
    title_font = load_font(58, bold=True)
    subtitle_font = load_font(24, bold=False)
    year_font = load_font(112, bold=True)
    name_font = load_font(46, bold=True)
    meta_font = load_font(28, bold=True)
    body_font = load_font(21, bold=False)
    axis_font = load_font(18, bold=True)
    legend_font = load_font(22, bold=True)

    start_year = int(segments[0]["start_year"])
    end_year = int(segments[-1]["end_year"])
    total_years = end_year - start_year

    portraits: dict[str, Image.Image | None] = {}
    for segment in segments:
        name = str(segment["display_name"])
        if name not in portraits:
            portraits[name] = load_portrait(name)

    timeline_left = 100
    timeline_right = WIDTH - 100
    timeline_top = 780
    timeline_height = 74
    timeline_width = timeline_right - timeline_left

    def year_to_x(year: float) -> int:
        ratio = (year - start_year) / max(1, total_years)
        return timeline_left + int(ratio * timeline_width)

    def make_frame(t: float) -> np.ndarray:
        year = start_year + int(round((t / duration) * total_years))
        year = max(start_year, min(end_year, year))
        segment = find_segment(segments, year)
        name = str(segment["display_name"])
        dynasty = str(segment["dynasty"])
        color = str(segment["house_color"])
        rgb = hex_to_rgb(color)

        frame = background.copy()
        draw = ImageDraw.Draw(frame, "RGBA")

        draw.text((92, 68), "ROIS DE FRANCE", font=title_font, fill="#f4efe7")
        draw.text((96, 136), "Timeline annuelle canonique de Clovis Ier à Louis-Philippe Ier", font=subtitle_font, fill="#d6dfeb")

        card = (84, 196, 1458, 724)
        draw.rounded_rectangle(card, radius=34, fill=(8, 18, 32, 196), outline=(255, 255, 255, 24), width=2)
        draw.rounded_rectangle((110, 228, 468, 586), radius=34, fill=(*rgb, 255), outline=(255, 255, 255, 22), width=2)
        portrait = portraits.get(name)
        if portrait is not None:
            frame.alpha_composite(portrait, (119, 237))

        draw_glow_text(frame, (540, 244), str(year), year_font, (244, 239, 231), (231, 189, 103))
        draw.text((540, 390), name, font=name_font, fill="#f4efe7")
        draw.text((544, 458), dynasty.upper(), font=meta_font, fill=(rgb[0], rgb[1], rgb[2], 255))
        reign_text = f"{segment['start_year']} - {segment['end_year']}"
        draw.text((544, 506), reign_text, font=meta_font, fill="#d5e0eb")
        facts = [
            str(segment.get("fait_1", "")).strip(),
            str(segment.get("fait_2", "")).strip(),
            str(segment.get("fait_3", "")).strip(),
        ]
        fact_y = 554
        fact_text_x = 576
        fact_text_width = 820
        for fact in facts:
            if not fact:
                continue
            wrapped = wrap_text(draw, fact, body_font, fact_text_width, 2)
            bullet_top = fact_y + 8
            draw.rounded_rectangle((544, bullet_top, 560, bullet_top + 16), radius=6, fill=(rgb[0], rgb[1], rgb[2], 255))
            line_y = fact_y
            for line in wrapped:
                draw.text((fact_text_x, line_y), line, font=body_font, fill="#d5e0eb")
                line_y += 25
            fact_y += max(48, len(wrapped) * 25 + 18)

        right_box = (1568, 196, WIDTH - 120, 664)
        draw.rounded_rectangle(right_box, radius=34, fill=(8, 18, 32, 156), outline=(255, 255, 255, 24), width=2)
        for segment_item in segments:
            sx = year_to_x(int(segment_item["start_year"]))
            ex = year_to_x(int(segment_item["end_year"]) + 1)
            draw.rounded_rectangle((sx, timeline_top, max(sx + 4, ex), timeline_top + timeline_height), radius=14, fill=hex_to_rgb(str(segment_item["house_color"])))

        for tick in (500, 800, 1000, 1200, 1400, 1600, 1800):
            if start_year <= tick <= end_year:
                x = year_to_x(tick)
                draw.line((x, timeline_top - 24, x, timeline_top + timeline_height + 24), fill=(255, 255, 255, 22), width=2)
                draw.text((x - 24, timeline_top + 90), str(tick), font=axis_font, fill="#d6dfeb")

        marker_x = year_to_x(year)
        draw.rounded_rectangle((marker_x - 6, timeline_top - 30, marker_x + 6, timeline_top + timeline_height + 30), radius=6, fill=(244, 239, 231, 255))
        draw.ellipse((marker_x - 16, timeline_top + 18, marker_x + 16, timeline_top + 50), fill=(244, 239, 231, 255))

        legend_y = 252
        dynasties = []
        for item in segments:
            key = str(item["dynasty"])
            if key not in dynasties:
                dynasties.append(key)
        for index, dynasty_name in enumerate(dynasties[:8]):
            source = next(item for item in segments if str(item["dynasty"]) == dynasty_name)
            ly = legend_y + index * 48
            draw.rounded_rectangle((1594, ly, 1622, ly + 28), radius=9, fill=hex_to_rgb(str(source["house_color"])))
            draw.text((1640, ly + 1), dynasty_name, font=legend_font, fill="#f4efe7")

        return np.array(frame.convert("RGB"))

    clip = VideoClip(make_frame, duration=duration)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    clip.write_videofile(str(output_path), fps=fps, codec="libx264", audio=False)
    clip.close()
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a France kings timeline video.")
    parser.add_argument("--input", type=Path, default=INPUT_CSV)
    parser.add_argument("--output", type=Path, default=OUTPUT_MP4)
    parser.add_argument("--duration", type=float, default=DURATION)
    parser.add_argument("--fps", type=int, default=FPS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_rows(args.input)
    output = render_video(rows, args.output, args.duration, args.fps)
    print(f"[history] timeline video generated -> {output}")


if __name__ == "__main__":
    main()
