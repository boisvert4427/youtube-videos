from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from video_generator.generate_ucl_barchart_race_moviepy import _fit_font_size, _load_font


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_INPUT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "video_game_sales"
    / "video_game_sales_publishers_1980_2017.csv"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "video_game_sales"
    / "video_game_sales_publishers_thumbnail_1980_2017.png"
)
DEFAULT_LOGOS_DIR = PROJECT_ROOT / "data" / "raw" / "video_game_sales" / "logos"

WIDTH = 1280
HEIGHT = 720
TOP_N = 5

COLORS = ["#E178D1", "#9B82F3", "#FFB74D", "#66A5F5", "#5ED6D8"]
DISPLAY_NAMES = {
    "sony_computer_entertainment": "Sony Computer Ent.",
    "namco_bandai_games": "Bandai Namco",
    "take_two_interactive": "Take-Two",
    "konami_digital_entertainment": "Konami",
}


def _load_impact(size: int) -> ImageFont.ImageFont:
    path = Path("C:/Windows/Fonts/impact.ttf")
    if path.exists():
        return ImageFont.truetype(str(path), size=size)
    return _load_font(size, bold=True)


def _fit_impact(draw: ImageDraw.ImageDraw, text: str, max_width: int, start: int, minimum: int) -> ImageFont.ImageFont:
    size = start
    font = _load_impact(size)
    while size > minimum and draw.textbbox((0, 0), text, font=font)[2] > max_width:
        size -= 1
        font = _load_impact(size)
    return font


def _make_background() -> Image.Image:
    x_values = np.linspace(0, 1, WIDTH, dtype=np.float32)
    y_values = np.linspace(0, 1, HEIGHT, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(x_values, y_values)

    deep = np.array([5, 9, 18], dtype=np.float32)
    blue = np.array([8, 58, 92], dtype=np.float32)
    cyan = np.array([80, 228, 238], dtype=np.float32)
    orange = np.array([255, 132, 54], dtype=np.float32)
    pink = np.array([235, 80, 190], dtype=np.float32)

    mix = np.clip(0.32 * grid_x + 0.55 * grid_y, 0, 1)
    cyan_glow = np.exp(-(((grid_x - 0.86) / 0.22) ** 2 + ((grid_y - 0.08) / 0.20) ** 2))
    pink_glow = np.exp(-(((grid_x - 0.14) / 0.28) ** 2 + ((grid_y - 0.90) / 0.22) ** 2))
    orange_glow = np.exp(-(((grid_x - 0.92) / 0.18) ** 2 + ((grid_y - 0.76) / 0.16) ** 2))

    pixels = np.clip(
        deep[None, None, :] * (1.0 - mix[..., None])
        + blue[None, None, :] * (0.92 * mix[..., None])
        + cyan[None, None, :] * (0.13 * cyan_glow[..., None])
        + pink[None, None, :] * (0.14 * pink_glow[..., None])
        + orange[None, None, :] * (0.10 * orange_glow[..., None]),
        0,
        255,
    ).astype(np.uint8)
    frame = Image.fromarray(pixels, mode="RGB").convert("RGBA")

    overlay = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")
    draw.rounded_rectangle((18, 18, WIDTH - 18, HEIGHT - 18), radius=32, outline=(92, 231, 238, 76), width=3)
    draw.line((42, 118, 774, 118), fill=(92, 231, 238, 88), width=5)
    draw.line((42, 632, 1228, 512), fill=(255, 255, 255, 18), width=3)
    for x in range(90, WIDTH, 126):
        draw.line((x, 180, x, 680), fill=(255, 255, 255, 9), width=1)
    overlay = overlay.filter(ImageFilter.GaussianBlur(1.4))
    frame.alpha_composite(overlay)
    return frame


def _load_logo(logos_dir: Path, key: str, size: int) -> Image.Image | None:
    path = logos_dir / f"{key}.png"
    if not path.exists():
        return None
    try:
        logo = Image.open(path).convert("RGBA")
        tile = Image.new("RGBA", (size, size), (244, 249, 252, 255))
        draw = ImageDraw.Draw(tile, "RGBA")
        draw.rounded_rectangle((0, 0, size - 1, size - 1), radius=14, fill=(244, 249, 252, 255), outline=(255, 255, 255, 220), width=2)
        logo.thumbnail((size - 12, size - 12), Image.Resampling.LANCZOS)
        tile.alpha_composite(logo, ((size - logo.width) // 2, (size - logo.height) // 2))
        return tile
    except Exception:
        return None


def _read_final_ranking(input_csv: Path, top_n: int) -> tuple[str, list[dict[str, str]]]:
    rows_by_date: dict[str, list[dict[str, str]]] = {}
    with input_csv.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            rows_by_date.setdefault(row["ranking_date"], []).append(row)
    if not rows_by_date:
        raise RuntimeError("No rows found in video game sales CSV.")
    ranking_date = sorted(rows_by_date)[-1]
    rows = sorted(rows_by_date[ranking_date], key=lambda row: -float(row["market_share"]))
    return ranking_date, rows[:top_n]


def render_thumbnail(input_csv: Path, output_path: Path, logos_dir: Path, top_n: int = TOP_N) -> Path:
    ranking_date, ranking = _read_final_ranking(input_csv, top_n)
    year = ranking_date[:4]
    maximum = max(float(row["market_share"]) for row in ranking) if ranking else 1.0

    frame = _make_background()
    draw = ImageDraw.Draw(frame, "RGBA")

    title_font = _fit_impact(draw, "VIDEO GAME", 500, 96, 64)
    wars_font = _fit_impact(draw, "SALES WARS", 520, 104, 68)
    date_font = _load_font(42, bold=True)
    badge_font = _load_font(22, bold=True)
    rank_font = _load_font(30, bold=True)
    name_font = _load_font(32, bold=True)
    value_font = _load_font(29, bold=True)
    small_font = _load_font(20, bold=True)

    draw.text((58, 34), "VIDEO GAME", font=title_font, fill="#FFFFFF", stroke_width=7, stroke_fill=(4, 10, 22, 220))
    draw.text((58, 132), "SALES WARS", font=wars_font, fill="#FFD15C", stroke_width=8, stroke_fill=(4, 10, 22, 230))
    draw.rounded_rectangle((64, 258, 328, 314), radius=18, fill=(255, 132, 54, 245), outline=(255, 233, 194, 200), width=2)
    draw.text((196, 286), "1980-2017", font=date_font, fill="#102033", anchor="mm")
    draw.text((66, 332), "TOP PUBLISHERS", font=badge_font, fill="#71E9EF")
    draw.text((66, 361), "CUMULATIVE GLOBAL SALES", font=small_font, fill=(225, 241, 248, 220))

    nintendo = _load_logo(logos_dir, "nintendo", 136)
    ea = _load_logo(logos_dir, "electronic_arts", 136)
    if nintendo:
        frame.alpha_composite(nintendo, (680, 70))
    if ea:
        frame.alpha_composite(ea, (845, 70))
    draw.text((827, 222), "VS", font=_load_impact(72), fill="#FF5A7A", anchor="mm", stroke_width=5, stroke_fill=(4, 10, 22, 230))

    chart_x = 470
    chart_y = 300
    chart_w = 660
    row_h = 66
    gap = 13
    for index, row in enumerate(ranking):
        y = chart_y + index * (row_h + gap)
        value = float(row["market_share"])
        width = int((value / maximum) * chart_w)
        color = COLORS[index % len(COLORS)]

        draw.rounded_rectangle((chart_x - 14, y - 3, chart_x + chart_w + 34, y + row_h + 3), radius=18, fill=(3, 15, 27, 162))
        draw.rounded_rectangle((chart_x, y, chart_x + max(34, width), y + row_h), radius=18, fill=color, outline=(255, 255, 255, 112), width=2)
        draw.rounded_rectangle((chart_x + 10, y + 9, chart_x + max(26, int(width * 0.62)), y + 20), radius=7, fill=(255, 255, 255, 66))

        logo = _load_logo(logos_dir, row["browser_key"], 54)
        if logo:
            frame.alpha_composite(logo, (chart_x - 72, y + 6))

        rank_box = (chart_x - 132, y + 7, chart_x - 84, y + 55)
        draw.rounded_rectangle(rank_box, radius=15, fill=(255, 132, 54, 245) if index == 0 else (16, 78, 98, 245))
        draw.text(((rank_box[0] + rank_box[2]) // 2, y + 32), str(index + 1), font=rank_font, fill="#102033" if index == 0 else "#F0FBFC", anchor="mm")

        name = DISPLAY_NAMES.get(row["browser_key"], row["browser_name"])
        name_max_width = max(120, min(276, width - 42))
        fitted_name = _fit_font_size(draw, name, name_max_width, 32, 18, bold=True)
        draw.text((chart_x + 18, y + 32), name, font=fitted_name, fill="#FFFFFF", anchor="lm", stroke_width=2, stroke_fill=(4, 10, 22, 160))
        value_text = f"{value:.0f}M"
        value_bbox = draw.textbbox((0, 0), value_text, font=value_font)
        value_width = value_bbox[2] - value_bbox[0]
        value_x = min(chart_x + width + 16, WIDTH - 54 - value_width)
        draw.text((value_x, y + 32), value_text, font=value_font, fill="#FFFFFF", anchor="lm", stroke_width=3, stroke_fill=(4, 10, 22, 180))

    draw.rounded_rectangle((1004, 44, 1220, 112), radius=22, fill=(255, 255, 255, 232), outline=(92, 231, 238, 200), width=3)
    draw.text((1112, 79), "WHO WON?", font=_load_impact(42), fill="#0B2033", anchor="mm")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.convert("RGB").save(output_path, quality=96)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a video game sales publishers thumbnail.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--logos-dir", type=Path, default=DEFAULT_LOGOS_DIR)
    parser.add_argument("--top-n", type=int, default=TOP_N)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = render_thumbnail(args.input, args.output, args.logos_dir, args.top_n)
    print(f"[video_generator] video game sales thumbnail generated -> {output}")


if __name__ == "__main__":
    main()
