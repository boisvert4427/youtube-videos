from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from video_generator.generate_ucl_barchart_race_moviepy import _fit_font_size, _load_font


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_INPUT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "demography"
    / "usa_male_names_top20_by_year_1880_2024.csv"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "demography"
    / "usa_male_names"
    / "usa_male_names_thumbnail_1880_2025.png"
)

WIDTH = 1280
HEIGHT = 720
TOP_N = 5
COLORS = ["#FF8A35", "#56D6E8", "#FFC44D", "#7C8CFF", "#FF5C8A"]


def _load_impact(size: int) -> ImageFont.ImageFont:
    path = Path("C:/Windows/Fonts/impact.ttf")
    if path.exists():
        return ImageFont.truetype(str(path), size=size)
    return _load_font(size, bold=True)


def _fit_impact(
    draw: ImageDraw.ImageDraw,
    text: str,
    max_width: int,
    start_size: int,
    min_size: int,
) -> ImageFont.ImageFont:
    size = start_size
    font = _load_impact(size)
    while size > min_size and draw.textbbox((0, 0), text, font=font)[2] > max_width:
        size -= 1
        font = _load_impact(size)
    return font


def _make_background() -> Image.Image:
    x_values = np.linspace(0, 1, WIDTH, dtype=np.float32)
    y_values = np.linspace(0, 1, HEIGHT, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(x_values, y_values)

    navy = np.array([6, 13, 32], dtype=np.float32)
    blue = np.array([20, 82, 132], dtype=np.float32)
    red = np.array([255, 70, 80], dtype=np.float32)
    cream = np.array([255, 238, 180], dtype=np.float32)
    cyan = np.array([67, 226, 241], dtype=np.float32)

    diagonal = np.clip(0.22 + 0.68 * grid_x + 0.18 * grid_y, 0, 1)
    red_glow = np.exp(-(((grid_x - 0.12) / 0.30) ** 2 + ((grid_y - 0.88) / 0.22) ** 2))
    cyan_glow = np.exp(-(((grid_x - 0.91) / 0.23) ** 2 + ((grid_y - 0.08) / 0.18) ** 2))
    cream_glow = np.exp(-(((grid_x - 0.52) / 0.50) ** 2 + ((grid_y - 0.42) / 0.38) ** 2))

    pixels = np.clip(
        navy[None, None, :] * (1.0 - diagonal[..., None])
        + blue[None, None, :] * diagonal[..., None]
        + red[None, None, :] * (0.14 * red_glow[..., None])
        + cyan[None, None, :] * (0.12 * cyan_glow[..., None])
        + cream[None, None, :] * (0.06 * cream_glow[..., None]),
        0,
        255,
    ).astype(np.uint8)
    frame = Image.fromarray(pixels, mode="RGB").convert("RGBA")

    overlay = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")
    draw.rounded_rectangle((18, 18, WIDTH - 18, HEIGHT - 18), radius=34, outline=(255, 255, 255, 42), width=3)
    draw.polygon([(0, 0), (470, 0), (350, HEIGHT), (0, HEIGHT)], fill=(255, 255, 255, 12))
    draw.polygon([(874, 0), (WIDTH, 0), (WIDTH, HEIGHT), (1030, HEIGHT)], fill=(255, 255, 255, 16))
    for y in range(96, HEIGHT, 96):
        draw.line((40, y, WIDTH - 40, y - 54), fill=(255, 255, 255, 13), width=2)
    overlay = overlay.filter(ImageFilter.GaussianBlur(1.2))
    frame.alpha_composite(overlay)
    return frame


def _read_rankings(input_csv: Path) -> tuple[int, list[dict[str, str]], int, list[dict[str, str]]]:
    by_year: dict[int, list[dict[str, str]]] = {}
    with input_csv.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row.get("sex", "").strip().upper() != "M":
                continue
            year = int(row["year"])
            by_year.setdefault(year, []).append(row)

    if not by_year:
        raise RuntimeError(f"No male-name rows found in {input_csv}")

    first_year = min(by_year)
    final_year = max(by_year)
    first_rows = sorted(by_year[first_year], key=lambda row: int(row["rank"]))
    final_rows = sorted(by_year[final_year], key=lambda row: int(row["rank"]))
    return first_year, first_rows, final_year, final_rows


def _draw_flag_badge(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int]) -> None:
    x0, y0, x1, y1 = box
    draw.rounded_rectangle(box, radius=20, fill=(255, 255, 255, 245), outline=(255, 255, 255, 200), width=2)
    stripe_h = (y1 - y0 - 16) / 7
    for index in range(7):
        y = int(y0 + 8 + index * stripe_h)
        color = (191, 34, 46, 255) if index % 2 == 0 else (255, 255, 255, 255)
        draw.rectangle((x0 + 10, y, x1 - 10, int(y + stripe_h)), fill=color)
    draw.rounded_rectangle((x0 + 10, y0 + 8, x0 + 68, y0 + 50), radius=5, fill=(24, 49, 112, 255))
    for sx in range(x0 + 20, x0 + 63, 16):
        for sy in range(y0 + 16, y0 + 45, 13):
            draw.ellipse((sx, sy, sx + 4, sy + 4), fill=(255, 255, 255, 230))


def render_thumbnail(input_csv: Path, output_path: Path, top_n: int = TOP_N) -> Path:
    first_year, first_rows, final_year, final_rows = _read_rankings(input_csv)
    ranking = final_rows[:top_n]
    leader = ranking[0]
    maximum = max(int(row["births"]) for row in ranking) if ranking else 1

    frame = _make_background()
    draw = ImageDraw.Draw(frame, "RGBA")

    title_font = _fit_impact(draw, "BOY NAME", 510, 104, 70)
    battle_font = _fit_impact(draw, "BATTLE", 520, 122, 78)
    hook_font = _load_font(34, bold=True)
    year_font = _load_font(42, bold=True)
    chip_font = _load_font(24, bold=True)
    rank_font = _load_font(28, bold=True)
    name_font = _load_font(34, bold=True)
    value_font = _load_font(28, bold=True)
    small_font = _load_font(18, bold=True)

    draw.text((56, 36), "BOY NAME", font=title_font, fill="#FFFFFF", stroke_width=7, stroke_fill=(5, 12, 26, 235))
    draw.text((56, 142), "BATTLE", font=battle_font, fill="#FFD84E", stroke_width=8, stroke_fill=(5, 12, 26, 240))

    draw.rounded_rectangle((62, 275, 392, 334), radius=20, fill=(255, 86, 91, 245), outline=(255, 255, 255, 145), width=2)
    draw.text((227, 304), f"{first_year}-{final_year}", font=year_font, fill="#FFFFFF", anchor="mm")
    draw.rounded_rectangle((62, 352, 392, 404), radius=18, fill=(8, 22, 45, 220), outline=(91, 226, 241, 170), width=2)
    draw.text((227, 378), "POPULAR IN THE USA", font=chip_font, fill="#77EFF8", anchor="mm")

    _draw_flag_badge(draw, (66, 466, 178, 536))
    draw.text((194, 490), "TOP BABY", font=hook_font, fill="#FFFFFF", stroke_width=4, stroke_fill=(5, 12, 26, 210))
    draw.text((194, 527), "NAMES", font=hook_font, fill="#FFFFFF", stroke_width=4, stroke_fill=(5, 12, 26, 210))

    draw.rounded_rectangle((828, 42, 1208, 146), radius=28, fill=(255, 255, 255, 238), outline=(255, 216, 78, 220), width=4)
    draw.text((1018, 73), "WHO WON?", font=_load_impact(48), fill="#0A1730", anchor="mm")
    draw.text(
        (1018, 118),
        f"#{leader['rank']} {leader['name'].upper()}",
        font=_load_font(31, bold=True),
        fill="#F04A55",
        anchor="mm",
    )

    chart_x = 520
    chart_y = 202
    chart_w = 650
    row_h = 67
    gap = 12
    for index, row in enumerate(ranking):
        y = chart_y + index * (row_h + gap)
        births = int(row["births"])
        width = int((births / maximum) * chart_w)
        color = COLORS[index % len(COLORS)]

        draw.rounded_rectangle((chart_x - 116, y - 4, chart_x + chart_w + 44, y + row_h + 4), radius=22, fill=(3, 13, 28, 170))
        rank_box = (chart_x - 102, y + 9, chart_x - 50, y + 61)
        draw.rounded_rectangle(rank_box, radius=16, fill=(255, 216, 78, 248) if index == 0 else (22, 91, 122, 245))
        draw.text(((rank_box[0] + rank_box[2]) // 2, y + 36), str(index + 1), font=rank_font, fill="#0A1730" if index == 0 else "#FFFFFF", anchor="mm")

        draw.rounded_rectangle((chart_x, y, chart_x + max(42, width), y + row_h), radius=22, fill=color, outline=(255, 255, 255, 130), width=2)
        draw.rounded_rectangle((chart_x + 14, y + 11, chart_x + max(34, int(width * 0.58)), y + 24), radius=8, fill=(255, 255, 255, 70))

        fitted = _fit_font_size(draw, row["name"], 230, 38, 24, bold=True)
        draw.text((chart_x + 22, y + 36), row["name"], font=fitted, fill="#FFFFFF", anchor="lm", stroke_width=3, stroke_fill=(5, 12, 26, 180))

        value_text = f"{births:,}"
        value_bbox = draw.textbbox((0, 0), value_text, font=value_font)
        value_width = value_bbox[2] - value_bbox[0]
        value_x = min(chart_x + width + 16, WIDTH - 54 - value_width)
        draw.text((value_x, y + 36), value_text, font=value_font, fill="#FFFFFF", anchor="lm", stroke_width=3, stroke_fill=(5, 12, 26, 190))

    old_leader = first_rows[0]
    then_now_box = (60, 596, 392, 674)
    draw.rounded_rectangle(then_now_box, radius=22, fill=(255, 255, 255, 232), outline=(91, 226, 241, 180), width=3)
    draw.text((82, 622), f"{first_year}: {old_leader['name']}", font=_load_font(25, bold=True), fill="#0A1730", anchor="lm")
    draw.text((82, 652), f"{final_year}: {leader['name']} takes over", font=small_font, fill=(240, 74, 85, 235), anchor="lm")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.convert("RGB").save(output_path, quality=96)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a US boy names thumbnail.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--top-n", type=int, default=TOP_N)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = render_thumbnail(args.input, args.output, args.top_n)
    print(f"[video_generator] US boy names thumbnail generated -> {output}")


if __name__ == "__main__":
    main()
