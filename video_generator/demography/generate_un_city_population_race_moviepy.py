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
    / "WUP2025-DB-DEGURBA-Cities-Population-Surface-Data.csv"
)
DEFAULT_NORMALIZED_INPUT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "demography"
    / "un_city_population"
    / "un_city_population_1975_2026.csv"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "demography"
    / "un_city_population"
    / "un_city_population_race_1975_2026_2m30.mp4"
)

TITLE = "CITY POPULATION RACE"
SUBTITLE = "TOP 12 CITIES | 1975-2026 | UN WUP 2025"
FOOTER = "UN WORLD URBANIZATION PROSPECTS | CITIES | 1975-2026"

FPS = 60
TOTAL_DURATION = 150.0
FINAL_HOLD_DURATION = 10.0
TOP_N = 12
START_YEAR = 1975
END_YEAR = 2026
PROJECTION_START_YEAR = 2026

CITY_NAME_ALIASES = {
    "Al-Qahirah (Cairo)": "Cairo",
    "Ciudad de México (Mexico City)": "Mexico City",
    "Moskva (Moscow)": "Moscow",
    "Tōkyō (Tokyo)": "Tokyo",
}


def _format_population(value: float) -> str:
    return f"{value / 1_000_000:,.2f}M"


def _display_city_name(city_name: str) -> str:
    return CITY_NAME_ALIASES.get(city_name.strip(), city_name.strip())


def normalize_input(
    input_csv: Path,
    output_csv: Path,
    start_year: int = START_YEAR,
    end_year: int = END_YEAR,
) -> Path:
    grouped: dict[int, list[dict[str, str]]] = {}
    previous_values: dict[str, float] = {}

    with input_csv.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        for row in reader:
            year = int(row["Year"])
            if year < start_year or year > end_year:
                continue
            pop_text = row["Pop"].strip()
            if not pop_text:
                continue
            iso3 = row["ISO3_Code"].strip().upper()
            iso2 = row["ISO2_Code"].strip().upper()
            city_code = row["City_Code"].strip()
            if not iso3 or not iso2 or not city_code:
                continue
            population = float(pop_text) * 1000.0
            city_name = _display_city_name(row["City_Name"])
            city_key = f"{iso3}_{city_code}"
            grouped.setdefault(year, []).append(
                {
                    "city_name": city_name,
                    "country_code": iso2,
                    "city_key": city_key,
                    "population": f"{population:.0f}",
                    "phase": "UN projection" if year >= PROJECTION_START_YEAR else "historical estimate",
                }
            )

    if not grouped:
        raise RuntimeError(f"No city population rows found in {input_csv}")

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
                key=lambda row: (-float(row["population"]), row["city_name"]),
            )
            leader = rows[0]
            summary = (
                f"Leader: {leader['city_name']} {_format_population(float(leader['population']))}"
                f"|{leader['phase']}"
            )
            for row in rows:
                city_key = row["city_key"]
                population = float(row["population"])
                previous = previous_values.get(city_key)
                writer.writerow(
                    {
                        "ranking_date": f"{year}-12-31",
                        "country_name": row["city_name"],
                        "country_code": row["country_code"],
                        "country_iso3": city_key,
                        "population": row["population"],
                        "yearly_change": "" if previous is None else f"{population - previous:.0f}",
                        "season_summary": summary,
                        "data_source": "UN World Urbanization Prospects 2025",
                    }
                )
                previous_values[city_key] = population
    return output_csv


def _patch_theme() -> dict[str, object]:
    previous = {
        "TITLE": base.TITLE,
        "SUBTITLE": base.SUBTITLE,
        "LEFT_HEADER_LABEL": base.LEFT_HEADER_LABEL,
        "RIGHT_HEADER_LABEL": base.RIGHT_HEADER_LABEL,
        "FOOTER": base.FOOTER,
        "SNAP_TO_CURRENT_RANKS": base.SNAP_TO_CURRENT_RANKS,
    }
    base.TITLE = TITLE
    base.SUBTITLE = SUBTITLE
    base.LEFT_HEADER_LABEL = "CITY"
    base.RIGHT_HEADER_LABEL = "POPULATION"
    base.FOOTER = FOOTER
    base.SNAP_TO_CURRENT_RANKS = False
    return previous


def _restore_theme(previous: dict[str, object]) -> None:
    base.TITLE = previous["TITLE"]  # type: ignore[assignment]
    base.SUBTITLE = previous["SUBTITLE"]  # type: ignore[assignment]
    base.LEFT_HEADER_LABEL = previous["LEFT_HEADER_LABEL"]  # type: ignore[assignment]
    base.RIGHT_HEADER_LABEL = previous["RIGHT_HEADER_LABEL"]  # type: ignore[assignment]
    base.FOOTER = previous["FOOTER"]  # type: ignore[assignment]
    base.SNAP_TO_CURRENT_RANKS = previous["SNAP_TO_CURRENT_RANKS"]  # type: ignore[assignment]


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
    end_year: int,
) -> Path:
    normalized = normalize_input(input_csv, normalized_input, start_year=start_year, end_year=end_year)
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
    parser = argparse.ArgumentParser(description="Generate a UN city population race.")
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
    parser.add_argument("--end-year", type=int, default=END_YEAR)
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
        end_year=args.end_year,
    )
    print(f"[video_generator] UN city population race generated -> {output}")


if __name__ == "__main__":
    main()
