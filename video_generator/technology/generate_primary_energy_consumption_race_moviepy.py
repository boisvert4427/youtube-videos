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

from video_generator.demography import generate_world_population_race_moviepy as base
from video_generator.generate_ucl_barchart_race_moviepy import DEFAULT_AUDIO


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_INPUT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "technology"
    / "browser_market_share"
    / "primary-energy-cons.csv"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "technology"
    / "browser_market_share"
    / "primary_energy_consumption_race_1965_2024_3min.mp4"
)
DEFAULT_FLAGS_DIR = PROJECT_ROOT / "data" / "raw" / "flags"

TITLE = "PRIMARY ENERGY CONSUMPTION RACE"
SUBTITLE = "TOP 12 COUNTRIES | 1965-2024 | kWh"
LEFT_HEADER_LABEL = "ENTITY"
RIGHT_HEADER_LABEL = "PRIMARY ENERGY"
FOOTER = "PRIMARY ENERGY CONSUMPTION | 1965-2024"
TOP_N = 12
FPS = 60
TOTAL_DURATION = 180.0
FINAL_HOLD_DURATION = 10.0

EXTRA_COLORS = {
    "CHN": "#FF8F42",
    "USA": "#5CE1E6",
    "IND": "#6CA7FF",
    "RUS": "#A98BFF",
    "JPN": "#66D19E",
    "DEU": "#FF6B8A",
    "BRA": "#FFB84D",
    "CAN": "#E4CB58",
    "FRA": "#E481D8",
    "GBR": "#40CECE",
    "ITA": "#8AC84A",
    "KOR": "#7B8CFF",
}


def _slugify_key(text: str) -> str:
    text = text.strip().upper()
    text = re.sub(r"[^A-Z0-9]+", "_", text)
    return text.strip("_") or "UNKNOWN"


def _format_energy(value: float) -> str:
    rounded = int(round(float(value)))
    return f"{rounded:,}".replace(",", " ")


def _build_season_summary(year: str, rows: list[dict[str, str]]) -> str:
    if not rows:
        return f"No data available | {year}"
    leader = max(rows, key=lambda row: float(row["Primary energy consumption"]))
    leader_name = leader["Entity"].strip()
    leader_value = _format_energy(float(leader["Primary energy consumption"]))
    return f"Leader: {leader_name} {leader_value} | Primary energy consumption"


def _transform_csv(input_csv: Path, output_csv: Path) -> Path:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    with input_csv.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        required = {"Entity", "Year", "Primary energy consumption"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise RuntimeError(f"Missing required columns in {input_csv}: {sorted(missing)}")
        for row in reader:
            code = row.get("Code", "").strip().upper()
            if not (len(code) == 3 and code.isalpha()):
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
            rows = sorted(
                grouped[year],
                key=lambda row: float(row["Primary energy consumption"]),
                reverse=True,
            )
            summary = _build_season_summary(year, rows)
            ranking_date = f"{int(year):04d}-12-31"
            for row in rows:
                entity = row["Entity"].strip()
                code = row.get("Code", "").strip().upper()
                iso3 = code if code else _slugify_key(entity)
                value = float(row["Primary energy consumption"])
                writer.writerow(
                    [
                        ranking_date,
                        entity,
                        "",
                        iso3,
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
        "SHOW_INSIGHT_BOX": getattr(base, "SHOW_INSIGHT_BOX", True),
    }
    base.TITLE = TITLE
    base.SUBTITLE = SUBTITLE
    base.LEFT_HEADER_LABEL = LEFT_HEADER_LABEL
    base.RIGHT_HEADER_LABEL = RIGHT_HEADER_LABEL
    base.FOOTER = FOOTER
    base._format_population = _format_energy
    base.COUNTRY_COLORS = {**base.COUNTRY_COLORS, **EXTRA_COLORS}
    base.SHOW_INSIGHT_BOX = False
    return previous


def _restore_theme(previous: dict[str, object]) -> None:
    base.TITLE = previous["TITLE"]  # type: ignore[assignment]
    base.SUBTITLE = previous["SUBTITLE"]  # type: ignore[assignment]
    base.LEFT_HEADER_LABEL = previous["LEFT_HEADER_LABEL"]  # type: ignore[assignment]
    base.RIGHT_HEADER_LABEL = previous["RIGHT_HEADER_LABEL"]  # type: ignore[assignment]
    base.FOOTER = previous["FOOTER"]  # type: ignore[assignment]
    base._format_population = previous["_format_population"]  # type: ignore[assignment]
    base.COUNTRY_COLORS = previous["COUNTRY_COLORS"]  # type: ignore[assignment]
    base.SHOW_INSIGHT_BOX = previous["SHOW_INSIGHT_BOX"]  # type: ignore[assignment]


def render_video(
    input_csv: Path,
    output_path: Path,
    flags_dir: Path,
    audio_path: Path,
    duration: float,
    final_hold_duration: float,
    fps: int,
    top_n: int,
) -> Path:
    with tempfile.TemporaryDirectory(prefix="primary_energy_timeseries_") as temp_dir:
        transformed_csv = _transform_csv(input_csv, Path(temp_dir) / "primary_energy_timeseries.csv")
        previous = _patch_theme()
        try:
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
    parser = argparse.ArgumentParser(description="Generate a landscape primary energy consumption bar chart race.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--flags-dir", type=Path, default=DEFAULT_FLAGS_DIR)
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
        duration=args.duration,
        final_hold_duration=args.final_hold,
        fps=args.fps,
        top_n=args.top_n,
    )
    print(f"[video_generator] Primary energy consumption race generated -> {output}")


if __name__ == "__main__":
    main()
