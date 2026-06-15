from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from video_generator.demography.generate_world_population_race_moviepy import (
    _build_flag_cache,
    _fit_font_size,
    _load_font,
    load_snapshots,
)
from video_generator.finance.forbes_billionaires_theme import (
    COLOR_PALETTE,
    format_money,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_INPUT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "finance"
    / "forbes_billionaires"
    / "forbes_billionaires_1997_2024.csv"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "finance"
    / "forbes_billionaires"
    / "forbes_billionaires_thumbnail_1997_2024.png"
)
DEFAULT_FLAGS_DIR = PROJECT_ROOT / "data" / "raw" / "flags"

WIDTH = 1280
HEIGHT = 720
TOP_N = 5
TITLE_TAG = "BILLIONAIRE BATTLE"
HOOK = "WHO'S #1?"
SUBTITLE = "FORBES TOP 5 | 1997-2024"
YEAR_LABEL = "2024"

MONTH_LABELS_EN = {
    1: "JAN",
    2: "FEB",
    3: "MAR",
    4: "APR",
    5: "MAY",
    6: "JUN",
    7: "JUL",
    8: "AUG",
    9: "SEP",
    10: "OCT",
    11: "NOV",
    12: "DEC",
}


def _make_background() -> Image.Image:
    x_values = np.linspace(0, 1, WIDTH, dtype=np.float32)
    y_values = np.linspace(0, 1, HEIGHT, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(x_values, y_values)
    base = np.array([18, 24, 44], dtype=np.float32)
    warm = np.array([255, 179, 71], dtype=np.float32)
    cool = np.array([84, 165, 255], dtype=np.float32)
    blush = np.array([255, 83, 111], dtype=np.float32)

    top_glow = np.exp(-(((grid_x - 0.08) / 0.22) ** 2 + ((grid_y - 0.06) / 0.16) ** 2))
    right_glow = np.exp(-(((grid_x - 0.92) / 0.16) ** 2 + ((grid_y - 0.18) / 0.22) ** 2))
    lower_glow = np.exp(-(((grid_x - 0.78) / 0.22) ** 2 + ((grid_y - 0.86) / 0.16) ** 2))
    center_vignette = np.exp(-(((grid_x - 0.48) / 0.62) ** 2 + ((grid_y - 0.48) / 0.54) ** 2))

    pixels = np.clip(
        base[None, None, :]
        + warm[None, None, :] * (0.16 * top_glow[..., None])
        + cool[None, None, :] * (0.07 * right_glow[..., None])
        + blush[None, None, :] * (0.06 * lower_glow[..., None])
        - np.array([10, 12, 18], dtype=np.float32)[None, None, :] * (0.42 * center_vignette[..., None]),
        0,
        255,
    ).astype(np.uint8)
    frame = Image.fromarray(pixels, mode="RGB").convert("RGBA")

    overlay = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")
    draw.rounded_rectangle((18, 18, WIDTH - 18, HEIGHT - 18), radius=34, outline=(255, 255, 255, 20), width=2)
    draw.ellipse((-40, -60, 420, 320), fill=(255, 205, 90, 45))
    draw.ellipse((880, -40, 1320, 300), fill=(95, 161, 255, 30))
    draw.ellipse((640, 440, 1260, 900), fill=(255, 66, 102, 28))
    draw.polygon([(720, 16), (770, 16), (1230, 372), (1180, 372)], fill=(255, 255, 255, 14))
    draw.line((78, 104, 1188, 104), fill=(255, 205, 90, 70), width=5)
    draw.line((60, 626, 1220, 500), fill=(255, 255, 255, 10), width=2)
    overlay = overlay.filter(ImageFilter.GaussianBlur(radius=2))
    frame.alpha_composite(overlay)
    return frame


def _format_snapshot_label_en(snapshot) -> str:
    try:
        month = int(str(snapshot.ranking_date)[5:7])
    except (TypeError, ValueError):
        month = 0
    month_label = MONTH_LABELS_EN.get(month)
    if month_label:
        return f"{month_label} {snapshot.year}"
    return str(snapshot.year)


def _load_impact_font(size: int) -> ImageFont.ImageFont:
    path = Path("C:/Windows/Fonts/impact.ttf")
    if path.exists():
        return ImageFont.truetype(str(path), size=size)
    return _load_font(size, bold=True)


def _fit_impact_font(
    draw: ImageDraw.ImageDraw,
    text: str,
    max_width: int,
    start_size: int,
    min_size: int,
) -> ImageFont.ImageFont:
    size = start_size
    font = _load_impact_font(size)
    while size > min_size and draw.textbbox((0, 0), text, font=font)[2] > max_width:
        size -= 1
        font = _load_impact_font(size)
    return font


def _hex_to_rgb(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    return tuple(int(value[index : index + 2], 16) for index in (0, 2, 4))


def _mix_rgb(color: str, target: tuple[int, int, int], amount: float) -> tuple[int, int, int]:
    red, green, blue = _hex_to_rgb(color)
    amount = min(max(amount, 0.0), 1.0)
    return (
        int(red + (target[0] - red) * amount),
        int(green + (target[1] - green) * amount),
        int(blue + (target[2] - blue) * amount),
    )


def _stable_color(key: str) -> str:
    digest = __import__("hashlib").sha1(key.encode("utf-8")).digest()
    return COLOR_PALETTE[int.from_bytes(digest[:2], "big") % len(COLOR_PALETTE)]


def _load_flag(flags_dir: Path, country_code: str) -> Image.Image | None:
    if not country_code:
        return None
    path = flags_dir / f"{country_code.lower()}.png"
    if not path.exists():
        return None
    try:
        image = Image.open(path).convert("RGBA")
        return ImageOps.fit(image, (46, 30), method=Image.Resampling.LANCZOS)
    except Exception:
        return None


def render_thumbnail(
    input_csv: Path,
    output_path: Path,
    flags_dir: Path,
    ranking_date: str | None = None,
    top_n: int = TOP_N,
) -> Path:
    snapshots = load_snapshots(input_csv)
    if not snapshots:
        raise RuntimeError("No Forbes billionaires snapshots were found.")

    if ranking_date is None:
        snapshot = snapshots[-1]
    else:
        match = next((candidate for candidate in snapshots if candidate.ranking_date == ranking_date), None)
        if match is None:
            available = ", ".join(item.ranking_date for item in snapshots[-5:])
            raise RuntimeError(f"Ranking date {ranking_date} not found. Available tail: {available}")
        snapshot = match

    ranked = snapshot.states[:top_n]
    colors = {state.country_iso3: _stable_color(state.country_iso3) for state in ranked}
    flags = _build_flag_cache(ranked, flags_dir)
    background = _make_background()
    draw = ImageDraw.Draw(background, "RGBA")

    title_font = _fit_impact_font(draw, "BILLIONAIRE", 400, 102, 74)
    battle_font = _fit_impact_font(draw, "WAR", 400, 110, 78)
    hook_font = _load_font(24, bold=True)
    subtitle_font = _load_font(22, bold=True)
    range_font = _load_font(30, bold=True)
    year_font = _load_font(96, bold=True)
    month_font = _load_font(15, bold=True)
    rank_font = _load_font(28, bold=True)
    name_font = _load_font(36, bold=True)
    value_font = _load_font(30, bold=True)
    badge_font = _load_font(20, bold=True)

    title_x = 832
    title_shadow = (20, 11, 32, 140)
    draw.rounded_rectangle((title_x - 8, 20, 1248, 214), radius=28, fill=(255, 255, 255, 28), outline=(255, 255, 255, 34), width=2)
    draw.rounded_rectangle((title_x - 10, 22, 1250, 216), radius=28, fill=(9, 18, 40, 138))
    draw.text((title_x, 30), "BILLIONAIRE", font=title_font, fill="#ffffff", stroke_width=7, stroke_fill=title_shadow)
    draw.text((title_x + 2, 136), "WAR", font=battle_font, fill="#ffdc5f", stroke_width=8, stroke_fill=title_shadow)
    hook_box = (title_x + 6, 222, title_x + 196, 260)
    draw.rounded_rectangle(hook_box, radius=16, fill=(255, 61, 94, 235), outline=(255, 255, 255, 120), width=2)
    hook_bbox = draw.textbbox((0, 0), HOOK, font=hook_font)
    draw.text(
        (hook_box[0] + (hook_box[2] - hook_box[0] - (hook_bbox[2] - hook_bbox[0])) // 2, hook_box[1] + 7),
        HOOK,
        font=hook_font,
        fill="#ffffff",
    )
    draw.text((title_x + 210, 228), "FORBES TOP 5", font=subtitle_font, fill="#ffffff")
    draw.text((title_x + 210, 258), "1997-2024", font=range_font, fill="#ffdc5f")
    draw.line((title_x + 204, 294, WIDTH - 44, 294), fill=(255, 61, 94, 220), width=5)

    year_box = (836, 336, 1240, 686)
    draw.rounded_rectangle(year_box, radius=42, fill=(12, 23, 48, 255), outline=(255, 208, 90, 200), width=2)
    draw.rounded_rectangle((year_box[0] + 16, year_box[1] + 16, year_box[2] - 16, year_box[3] - 16), radius=34, outline=(255, 255, 255, 28), width=2)
    month_label = _format_snapshot_label_en(snapshot)
    month_bbox = draw.textbbox((0, 0), month_label, font=month_font)
    draw.text(
        (year_box[0] + (year_box[2] - year_box[0] - (month_bbox[2] - month_bbox[0])) // 2, 356),
        month_label,
        font=month_font,
        fill=(255, 224, 157, 240),
    )
    year_bbox = draw.textbbox((0, 0), YEAR_LABEL, font=year_font)
    draw.text(
        (year_box[0] + (year_box[2] - year_box[0] - (year_bbox[2] - year_bbox[0])) // 2, 392),
        YEAR_LABEL,
        font=year_font,
        fill="#ffffff",
    )
    burst = [(year_box[0] + 26, year_box[1] + 88), (year_box[0] + 102, year_box[1] + 54), (year_box[0] + 92, year_box[1] + 128)]
    draw.polygon(burst, fill=(255, 61, 94, 220))
    draw.polygon([(year_box[0] + 52, year_box[1] + 96), (year_box[0] + 145, year_box[1] + 72), (year_box[0] + 134, year_box[1] + 134)], fill=(255, 208, 90, 200))

    leader = ranked[0]
    leader_flag = flags.get(leader.country_code.upper())
    if leader_flag is not None:
        flag_x = year_box[0] + 112
        flag_y = 346
        background.alpha_composite(leader_flag, (flag_x, flag_y))
        draw.rounded_rectangle((flag_x - 4, flag_y - 4, flag_x + leader_flag.width + 4, flag_y + leader_flag.height + 4), radius=8, outline=(255, 255, 255, 150), width=2)

    badge = (986, 596, 1188, 654)
    draw.rounded_rectangle(badge, radius=22, fill=(255, 61, 94, 236), outline=(255, 255, 255, 90), width=2)
    draw.text((badge[0] + 18, badge[1] + 12), "TOP 5", font=badge_font, fill="#ffffff")

    bar_left = 38
    bar_right = 806
    bar_width = bar_right - bar_left
    base_y = 184
    pitch = 92
    bar_height = 68

    for index, state in enumerate(ranked):
        row_y = base_y + index * pitch
        color = colors[state.country_iso3]
        fill = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
        bar_draw = ImageDraw.Draw(fill, "RGBA")
        bar_draw.rounded_rectangle(
            (bar_left + 5, row_y + 5, bar_right + 5, row_y + bar_height + 5),
            radius=22,
            fill=(0, 0, 0, 34),
        )
        bar_draw.rounded_rectangle(
            (bar_left, row_y, bar_right, row_y + bar_height),
            radius=22,
            fill=color,
            outline=(255, 255, 255, 210),
            width=3,
        )
        bar_draw.rounded_rectangle((bar_left, row_y, bar_left + 18, row_y + bar_height), radius=22, fill=_mix_rgb(color, (255, 255, 255), 0.10))
        bar_draw.rounded_rectangle((bar_left + 20, row_y + 10, bar_right - 18, row_y + 18), radius=10, fill=(255, 255, 255, 22))
        background.alpha_composite(fill)

        rank_box = (bar_left + 4, row_y + 4, bar_left + 72, row_y + bar_height - 4)
        draw.rounded_rectangle(rank_box, radius=18, fill=(12, 23, 48, 255))
        rank_bbox = draw.textbbox((0, 0), str(index + 1), font=rank_font)
        draw.text(
            (rank_box[0] + (rank_box[2] - rank_box[0] - (rank_bbox[2] - rank_bbox[0])) // 2, row_y + 15),
            str(index + 1),
            font=rank_font,
            fill="#ffffff",
        )

        flag = flags.get(state.country_code.upper())
        flag_x = bar_left + 86
        if flag is not None:
            flag_y = row_y + (bar_height - flag.height) // 2
            draw.rounded_rectangle((flag_x - 4, flag_y - 4, flag_x + flag.width + 4, flag_y + flag.height + 4), radius=8, fill=(255, 255, 255, 240))
            background.alpha_composite(flag, (flag_x, flag_y))
            name_x = flag_x + 60
        else:
            name_x = flag_x

        name = state.country_name
        value_text = format_money(state.population)
        value_bbox = draw.textbbox((0, 0), value_text, font=value_font)
        max_name_width = max(130, bar_right - name_x - (value_bbox[2] - value_bbox[0]) - 40)
        name_font_fit = _fit_font_size(draw, name, max_name_width, 36, 20, bold=True)
        draw.text((name_x + 1, row_y + 14), name, font=name_font_fit, fill="#ffffff", stroke_width=2, stroke_fill=(0, 0, 0, 140))

        draw.text(
            (bar_right - 22 - (value_bbox[2] - value_bbox[0]), row_y + 14),
            value_text,
            font=value_font,
            fill="#ffffff",
            stroke_width=2,
            stroke_fill=(0, 0, 0, 140),
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    background.convert("RGB").save(output_path)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a Forbes billionaires thumbnail.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--flags-dir", type=Path, default=DEFAULT_FLAGS_DIR)
    parser.add_argument("--ranking-date", type=str, default=None)
    parser.add_argument("--top-n", type=int, default=TOP_N)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = render_thumbnail(
        input_csv=args.input,
        output_path=args.output,
        flags_dir=args.flags_dir,
        ranking_date=args.ranking_date,
        top_n=args.top_n,
    )
    print(f"[video_generator] Forbes billionaires thumbnail generated -> {output}")


if __name__ == "__main__":
    main()
