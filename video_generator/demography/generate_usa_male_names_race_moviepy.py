from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from video_generator.demography import generate_france_female_first_names_race_moviepy as base


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_INPUT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "demography"
    / "usa_male_names_top20_by_year_1880_2024.csv"
)
DEFAULT_NORMALIZED_INPUT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "demography"
    / "usa_male_names"
    / "usa_male_names_1880_2025.csv"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "demography"
    / "usa_male_names"
    / "usa_male_names_race_1880_2025_3min.mp4"
)

TITLE = "US BOY NAMES"
SUBTITLE = "MOST POPULAR BOY NAMES | UNITED STATES | 1880-2025"
FOOTER = "US BOY NAMES | SSA BABY NAMES | 1880-2025"
FPS = 60
TOTAL_DURATION = 180.0
FINAL_HOLD_DURATION = 10.0
TOP_N = 12


def normalize_input(input_csv: Path, output_csv: Path) -> Path:
    grouped: dict[int, list[dict[str, str]]] = {}
    with input_csv.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row.get("sex", "").strip().upper() != "M":
                continue
            year = int(row["year"])
            grouped.setdefault(year, []).append(row)

    if not grouped:
        raise RuntimeError(f"No male name rows found in {input_csv}")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as file:
        fieldnames = ["ranking_date", "first_name", "births", "season_summary"]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for year in sorted(grouped):
            rows = sorted(
                grouped[year],
                key=lambda row: (int(row["rank"]), -int(row["births"]), row["name"]),
            )
            leader = rows[0]
            summary = (
                f"Most popular: {leader['name']} | "
                f"{int(leader['births']):,} boys named {leader['name']} | "
                f"Top {len(rows)} tracked"
            )
            for row in rows:
                writer.writerow(
                    {
                        "ranking_date": f"{year}-01-01",
                        "first_name": row["name"].strip(),
                        "births": row["births"].strip(),
                        "season_summary": summary,
                    }
                )
    return output_csv


def render_video(
    input_csv: Path,
    normalized_input: Path,
    output_path: Path,
    audio_path: Path,
    duration: float,
    final_hold_duration: float,
    fps: int,
    top_n: int,
) -> Path:
    normalized = normalize_input(input_csv, normalized_input)
    previous_title = base.TITLE
    previous_subtitle = base.SUBTITLE
    previous_footer = base.FOOTER
    previous_leader_label = base.LEADER_LABEL
    previous_value_suffix = base.VALUE_SUFFIX
    previous_left_header = base.LEFT_HEADER_LABEL
    previous_right_header = base.RIGHT_HEADER_LABEL
    base.TITLE = TITLE
    base.SUBTITLE = SUBTITLE
    base.FOOTER = FOOTER
    base.LEADER_LABEL = "#1"
    base.VALUE_SUFFIX = "births"
    base.LEFT_HEADER_LABEL = "NAME"
    base.RIGHT_HEADER_LABEL = "BIRTHS"
    try:
        return base.render_video(
            input_csv=normalized,
            output_path=output_path,
            audio_path=audio_path,
            duration=duration,
            final_hold_duration=final_hold_duration,
            fps=fps,
            top_n=top_n,
        )
    finally:
        base.TITLE = previous_title
        base.SUBTITLE = previous_subtitle
        base.FOOTER = previous_footer
        base.LEADER_LABEL = previous_leader_label
        base.VALUE_SUFFIX = previous_value_suffix
        base.LEFT_HEADER_LABEL = previous_left_header
        base.RIGHT_HEADER_LABEL = previous_right_header


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a landscape US boy names bar chart race."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--normalized-input", type=Path, default=DEFAULT_NORMALIZED_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--audio", type=Path, default=base.DEFAULT_AUDIO)
    parser.add_argument("--duration", type=float, default=TOTAL_DURATION)
    parser.add_argument("--final-hold", type=float, default=FINAL_HOLD_DURATION)
    parser.add_argument("--fps", type=int, default=FPS)
    parser.add_argument("--top-n", type=int, default=TOP_N)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = render_video(
        input_csv=args.input,
        normalized_input=args.normalized_input,
        output_path=args.output,
        audio_path=args.audio,
        duration=args.duration,
        final_hold_duration=args.final_hold,
        fps=args.fps,
        top_n=args.top_n,
    )
    print(f"[video_generator] US boy names race generated -> {output}")


if __name__ == "__main__":
    main()
