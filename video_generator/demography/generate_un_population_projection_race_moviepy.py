from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from video_generator.demography import generate_world_population_race_moviepy as base


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_INPUT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "demography"
    / "population-with-un-projections.csv"
)
DEFAULT_NORMALIZED_INPUT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "demography"
    / "un_population_projection"
    / "un_population_projection_2026_2100.csv"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "demography"
    / "un_population_projection"
    / "un_population_projection_race_2026_2100_3min.mp4"
)

TITLE = "FUTURE POPULATION RACE"
SUBTITLE = "TOP 12 COUNTRIES | 2026-2100 | UN PROJECTIONS"
FOOTER = "UN POPULATION PROJECTIONS | 2026-2100"

FPS = 60
TOTAL_DURATION = 180.0
FINAL_HOLD_DURATION = 10.0
TOP_N = 12
START_YEAR = 2026

ISO3_TO_ALPHA2 = {
    "AFG": "AF",
    "AGO": "AO",
    "ARG": "AR",
    "AUS": "AU",
    "BGD": "BD",
    "BRA": "BR",
    "CAN": "CA",
    "CHN": "CN",
    "COD": "CD",
    "DEU": "DE",
    "DZA": "DZ",
    "EGY": "EG",
    "ESP": "ES",
    "ETH": "ET",
    "FRA": "FR",
    "GBR": "GB",
    "IDN": "ID",
    "IND": "IN",
    "IRN": "IR",
    "ITA": "IT",
    "JPN": "JP",
    "MEX": "MX",
    "MMR": "MM",
    "NGA": "NG",
    "PAK": "PK",
    "PHL": "PH",
    "RUS": "RU",
    "THA": "TH",
    "TUR": "TR",
    "TZA": "TZ",
    "USA": "US",
    "VNM": "VN",
}

DISPLAY_NAMES = {
    "Democratic Republic of Congo": "DR Congo",
    "Tanzania": "Tanzania",
    "United States": "United States",
    "United Kingdom": "United Kingdom",
}


def _format_population(value: float) -> str:
    if value >= 1_000_000_000:
        return f"{value / 1_000_000_000:,.1f}B"
    return f"{value / 1_000_000:,.1f}M"


def normalize_input(input_csv: Path, output_csv: Path, start_year: int = START_YEAR) -> Path:
    grouped: dict[int, list[dict[str, str]]] = {}
    previous_values: dict[str, float] = {}
    with input_csv.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        for row in reader:
            iso3 = row["Code"].strip().upper()
            if len(iso3) != 3 or iso3.startswith("OWID"):
                continue
            value_text = row["Population"].strip() or row["Population (Projected)"].strip()
            if not value_text:
                continue
            year = int(row["Year"])
            if year < start_year:
                continue
            population = float(value_text)
            grouped.setdefault(year, []).append(
                {
                    "country_name": DISPLAY_NAMES.get(row["Entity"].strip(), row["Entity"].strip()),
                    "country_code": ISO3_TO_ALPHA2.get(iso3, ""),
                    "country_iso3": iso3,
                    "population": f"{population:.0f}",
                    "is_projection": "1" if row["Population (Projected)"].strip() else "0",
                }
            )

    if not grouped:
        raise RuntimeError(f"No country population rows found in {input_csv}")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as file:
        fieldnames = [
            "ranking_date",
            "country_name",
            "country_code",
            "country_iso3",
            "population",
            "yearly_change",
            "season_summary",
            "data_source",
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for year in sorted(grouped):
            rows = sorted(
                grouped[year],
                key=lambda row: (-float(row["population"]), row["country_name"]),
            )
            leader = rows[0]
            phase = "UN projection" if leader["is_projection"] == "1" else "historical data"
            summary = (
                f"Leader: {leader['country_name']} {_format_population(float(leader['population']))}"
                f"|{phase}"
            )
            for row in rows:
                previous = previous_values.get(row["country_iso3"])
                population = float(row["population"])
                writer.writerow(
                    {
                        "ranking_date": f"{year}-12-31",
                        "country_name": row["country_name"],
                        "country_code": row["country_code"],
                        "country_iso3": row["country_iso3"],
                        "population": row["population"],
                        "yearly_change": "" if previous is None else f"{population - previous:.0f}",
                        "season_summary": summary,
                        "data_source": "UN World Population Prospects",
                    }
                )
                previous_values[row["country_iso3"]] = population
    return output_csv


def _patch_theme() -> dict[str, object]:
    previous = {
        "TITLE": base.TITLE,
        "SUBTITLE": base.SUBTITLE,
        "FOOTER": base.FOOTER,
    }
    base.TITLE = TITLE
    base.SUBTITLE = SUBTITLE
    base.FOOTER = FOOTER
    return previous


def _restore_theme(previous: dict[str, object]) -> None:
    base.TITLE = previous["TITLE"]  # type: ignore[assignment]
    base.SUBTITLE = previous["SUBTITLE"]  # type: ignore[assignment]
    base.FOOTER = previous["FOOTER"]  # type: ignore[assignment]


def render_video(
    input_csv: Path,
    normalized_input: Path,
    output_path: Path,
    flags_dir: Path,
    audio_path: Path,
    duration: float,
    final_hold_duration: float,
    fps: int,
    top_n: int,
    start_year: int,
) -> Path:
    normalized = normalize_input(input_csv, normalized_input, start_year=start_year)
    previous = _patch_theme()
    try:
        return base.render_video(
            input_csv=normalized,
            output_path=output_path,
            flags_dir=flags_dir,
            audio_path=audio_path,
            duration=duration,
            final_hold_duration=final_hold_duration,
            fps=fps,
            top_n=top_n,
        )
    finally:
        _restore_theme(previous)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a UN population projections race.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--normalized-input", type=Path, default=DEFAULT_NORMALIZED_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--flags-dir", type=Path, default=base.DEFAULT_FLAGS_DIR)
    parser.add_argument("--audio", type=Path, default=base.DEFAULT_AUDIO)
    parser.add_argument("--duration", type=float, default=TOTAL_DURATION)
    parser.add_argument("--final-hold", type=float, default=FINAL_HOLD_DURATION)
    parser.add_argument("--fps", type=int, default=FPS)
    parser.add_argument("--top-n", type=int, default=TOP_N)
    parser.add_argument("--start-year", type=int, default=START_YEAR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = render_video(
        input_csv=args.input,
        normalized_input=args.normalized_input,
        output_path=args.output,
        flags_dir=args.flags_dir,
        audio_path=args.audio,
        duration=args.duration,
        final_hold_duration=args.final_hold,
        fps=args.fps,
        top_n=args.top_n,
        start_year=args.start_year,
    )
    print(f"[video_generator] UN population projection race generated -> {output}")


if __name__ == "__main__":
    main()
