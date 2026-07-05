from __future__ import annotations

import argparse
import csv
import re
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps

from video_generator.demography import generate_world_population_race_moviepy as base
from video_generator.generate_ucl_barchart_race_moviepy import DEFAULT_AUDIO


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_INPUT = Path(r"C:\Users\leona\Downloads\oil-consumption-by-country.csv")
DEFAULT_BACKGROUND_IMAGE = Path(r"C:\Users\leona\Downloads\ChatGPT Image 5 juil. 2026, 20_51_42.png")
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "technology"
    / "oil_consumption"
    / "oil_consumption_race_1965_2024_3min_uhd.mp4"
)
DEFAULT_FLAGS_DIR = PROJECT_ROOT / "data" / "raw" / "flags"

TITLE = "OIL CONSUMPTION RACE"
SUBTITLE = "TOP 12 COUNTRIES | 1965-2024 | thousand barrels per day"
LEFT_HEADER_LABEL = ""
RIGHT_HEADER_LABEL = "OIL CONSUMPTION"
FOOTER = "SOURCE: OUR WORLD IN DATA"
TOP_N = 12
FPS = 60
TOTAL_DURATION = 180.0
FINAL_HOLD_DURATION = 10.0

AGGREGATE_CODES = {
    "OWID_AFR",
    "OWID_ASI",
    "OWID_EUR",
    "OWID_EU27",
    "OWID_HIC",
    "OWID_LMC",
    "OWID_NAM",
    "OWID_OCE",
    "OWID_SAM",
    "OWID_UMC",
    "OWID_WRL",
    "OWID_USS",
}

ISO3_TO_ALPHA2 = {
    "ARG": "AR",
    "AUS": "AU",
    "BEL": "BE",
    "BRA": "BR",
    "CAN": "CA",
    "CHN": "CN",
    "DEU": "DE",
    "ESP": "ES",
    "FRA": "FR",
    "GBR": "GB",
    "IND": "IN",
    "IRN": "IR",
    "ITA": "IT",
    "JPN": "JP",
    "KOR": "KR",
    "MEX": "MX",
    "NLD": "NL",
    "RUS": "RU",
    "SAU": "SA",
    "SWE": "SE",
    "UKR": "UA",
    "USA": "US",
}

EXTRA_COLORS = {
    "USA": "#67c8ff",
    "CHN": "#ff9d4d",
    "IND": "#7cc36a",
    "RUS": "#9f8bff",
    "SAU": "#f4c15d",
    "JPN": "#77d7c5",
    "DEU": "#f38aa7",
    "CAN": "#85b8ff",
    "BRA": "#7bd36a",
    "KOR": "#c38bff",
    "MEX": "#ffb56a",
    "IRN": "#8ed0a7",
}

WIDTH = 1920
HEIGHT = 1080


def _slugify_key(text: str) -> str:
    text = text.strip().upper()
    text = re.sub(r"[^A-Z0-9]+", "_", text)
    return text.strip("_") or "UNKNOWN"


def _format_oil(value: float) -> str:
    return f"{value:,.0f}".replace(",", " ")


def _is_country_row(row: dict[str, str]) -> bool:
    code = row.get("Code", "").strip().upper()
    if not code or code in AGGREGATE_CODES:
        return False
    return len(code) == 3 and code.isalpha()


def _build_season_summary(year: str, rows: list[dict[str, str]]) -> str:
    if not rows:
        return f"No data available | {year}"
    leader = max(rows, key=lambda row: float(row["Oil"]))
    leader_name = leader["Entity"].strip()
    leader_value = _format_oil(float(leader["Oil"]))
    return f"Leader: {leader_name} {leader_value} | Oil consumption"


def _make_background(background_image: Path | None) -> Image.Image:
    if background_image is not None and background_image.exists():
        try:
            base_image = Image.open(background_image).convert("RGBA")
            base_image = ImageOps.fit(base_image, (WIDTH, HEIGHT), method=Image.Resampling.LANCZOS, centering=(0.5, 0.5))
            base_image = base_image.filter(ImageFilter.GaussianBlur(radius=0.5))
            veil = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
            draw = ImageDraw.Draw(veil, "RGBA")
            draw.rectangle((0, 0, WIDTH, HEIGHT), fill=(0, 0, 0, 54))
            draw.rectangle((1040, 0, WIDTH, HEIGHT), fill=(0, 0, 0, 78))
            draw.ellipse((760, -180, 1920, 640), fill=(16, 52, 84, 26))
            draw.ellipse((1040, 80, 1920, 820), fill=(0, 0, 0, 48))
            draw.ellipse((-240, 650, 520, 1220), fill=(0, 0, 0, 42))
            return Image.alpha_composite(base_image, veil)
        except Exception:
            pass

    x_values = np.linspace(0, 1, WIDTH, dtype=np.float32)
    y_values = np.linspace(0, 1, HEIGHT, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(x_values, y_values)
    deep = np.array([2, 10, 20], dtype=np.float32)
    navy = np.array([9, 34, 62], dtype=np.float32)
    teal = np.array([19, 91, 109], dtype=np.float32)
    amber = np.array([222, 170, 85], dtype=np.float32)
    mix = np.clip(0.34 * grid_x + 0.68 * grid_y, 0, 1)
    glow = np.exp(-(((grid_x - 0.84) / 0.26) ** 2 + ((grid_y - 0.18) / 0.18) ** 2))
    pixels = np.clip(
        deep[None, None, :] * (1.0 - mix[..., None])
        + navy[None, None, :] * (0.85 * mix[..., None])
        + teal[None, None, :] * (0.18 * glow[..., None])
        + amber[None, None, :] * (0.04 * glow[..., None]),
        0,
        255,
    ).astype(np.uint8)
    return Image.fromarray(pixels, mode="RGB").convert("RGBA")


def _transform_csv(input_csv: Path, output_csv: Path) -> Path:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    with input_csv.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        required = {"Entity", "Code", "Year", "Oil"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise RuntimeError(f"Missing required columns in {input_csv}: {sorted(missing)}")
        for row in reader:
            if not _is_country_row(row):
                continue
            year = row["Year"].strip()
            grouped[year].append(row)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "ranking_date",
                "country_name",
                "country_code",
                "country_iso3",
                "population",
                "season_summary",
                "source",
            ]
        )
        for year in sorted(grouped, key=lambda item: int(item)):
            rows = sorted(grouped[year], key=lambda row: float(row["Oil"]), reverse=True)
            summary = _build_season_summary(year, rows)
            ranking_date = f"{int(year):04d}-12-31"
            for row in rows:
                entity = row["Entity"].strip()
                code = row.get("Code", "").strip().upper()
                value = float(row["Oil"])
                writer.writerow(
                    [
                        ranking_date,
                        entity,
                        ISO3_TO_ALPHA2.get(code, ""),
                        code if code else _slugify_key(entity),
                        value,
                        summary,
                        "Our World in Data",
                    ]
                )
    return output_csv


def _patch_theme() -> dict[str, object]:
    previous = {
        "TITLE": base.TITLE,
        "SUBTITLE": base.SUBTITLE,
        "LEFT_HEADER_LABEL": base.LEFT_HEADER_LABEL,
        "RIGHT_HEADER_LABEL": base.RIGHT_HEADER_LABEL,
        "FOOTER": base.FOOTER,
        "_format_population": base._format_population,
        "COUNTRY_COLORS": base.COUNTRY_COLORS,
        "SNAP_TO_CURRENT_RANKS": getattr(base, "SNAP_TO_CURRENT_RANKS", False),
        "SHOW_INSIGHT_BOX": getattr(base, "SHOW_INSIGHT_BOX", True),
        "SHOW_ROW_BACKPLATE": getattr(base, "SHOW_ROW_BACKPLATE", True),
        "NAME_IN_BAR": getattr(base, "NAME_IN_BAR", False),
        "_make_background": base._make_background,
    }
    base.TITLE = TITLE
    base.SUBTITLE = SUBTITLE
    base.LEFT_HEADER_LABEL = LEFT_HEADER_LABEL
    base.RIGHT_HEADER_LABEL = RIGHT_HEADER_LABEL
    base.FOOTER = FOOTER
    base._format_population = _format_oil
    base.COUNTRY_COLORS = {**base.COUNTRY_COLORS, **EXTRA_COLORS}
    base.SNAP_TO_CURRENT_RANKS = False
    base.SHOW_INSIGHT_BOX = False
    base.SHOW_ROW_BACKPLATE = False
    base.NAME_IN_BAR = True
    base._make_background = lambda: _make_background(DEFAULT_BACKGROUND_IMAGE)
    return previous


def _restore_theme(previous: dict[str, object]) -> None:
    base.TITLE = previous["TITLE"]  # type: ignore[assignment]
    base.SUBTITLE = previous["SUBTITLE"]  # type: ignore[assignment]
    base.LEFT_HEADER_LABEL = previous["LEFT_HEADER_LABEL"]  # type: ignore[assignment]
    base.RIGHT_HEADER_LABEL = previous["RIGHT_HEADER_LABEL"]  # type: ignore[assignment]
    base.FOOTER = previous["FOOTER"]  # type: ignore[assignment]
    base._format_population = previous["_format_population"]  # type: ignore[assignment]
    base.COUNTRY_COLORS = previous["COUNTRY_COLORS"]  # type: ignore[assignment]
    base.SNAP_TO_CURRENT_RANKS = previous["SNAP_TO_CURRENT_RANKS"]  # type: ignore[assignment]
    base.SHOW_INSIGHT_BOX = previous["SHOW_INSIGHT_BOX"]  # type: ignore[assignment]
    base.SHOW_ROW_BACKPLATE = previous["SHOW_ROW_BACKPLATE"]  # type: ignore[assignment]
    base.NAME_IN_BAR = previous["NAME_IN_BAR"]  # type: ignore[assignment]
    base._make_background = previous["_make_background"]  # type: ignore[assignment]


def render_video(
    input_csv: Path,
    output_path: Path,
    flags_dir: Path,
    audio_path: Path,
    background_image: Path | None,
    duration: float,
    final_hold_duration: float,
    fps: int,
    top_n: int,
) -> Path:
    with tempfile.TemporaryDirectory(prefix="oil_consumption_timeseries_") as temp_dir:
        transformed_csv = _transform_csv(input_csv, Path(temp_dir) / "oil_consumption_timeseries.csv")
        previous = _patch_theme()
        try:
            if background_image is not None and background_image.exists():
                base._make_background = lambda: _make_background(background_image)
            return base.render_video(
                transformed_csv,
                output_path,
                flags_dir,
                audio_path,
                duration,
                final_hold_duration,
                fps,
                top_n,
            )
        finally:
            _restore_theme(previous)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a landscape oil consumption bar chart race.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--flags-dir", type=Path, default=DEFAULT_FLAGS_DIR)
    parser.add_argument("--background-image", type=Path, default=DEFAULT_BACKGROUND_IMAGE)
    parser.add_argument("--audio", type=Path, default=DEFAULT_AUDIO)
    parser.add_argument("--duration", type=float, default=TOTAL_DURATION)
    parser.add_argument("--final-hold", type=float, default=FINAL_HOLD_DURATION)
    parser.add_argument("--fps", type=int, default=FPS)
    parser.add_argument("--top-n", type=int, default=TOP_N)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = render_video(
        input_csv=args.input,
        output_path=args.output,
        flags_dir=args.flags_dir,
        audio_path=args.audio,
        background_image=args.background_image,
        duration=args.duration,
        final_hold_duration=args.final_hold,
        fps=args.fps,
        top_n=args.top_n,
    )
    print(f"[video_generator] Oil consumption race generated -> {output}")


if __name__ == "__main__":
    main()
