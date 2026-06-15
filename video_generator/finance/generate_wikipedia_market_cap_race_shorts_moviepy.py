from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from video_generator.demography import generate_world_population_race_shorts_moviepy as base
from video_generator.finance.forbes_billionaires_theme import (
    build_stable_color_map,
    make_background,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_INPUT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "finance"
    / "public_corporations_market_capitalization"
    / "public_corporations_market_capitalization_2000_2026.csv"
)
DEFAULT_ADAPTED_INPUT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "finance"
    / "public_corporations_market_capitalization"
    / "public_corporations_market_capitalization_2000_2026_short_input.csv"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "finance"
    / "public_corporations_market_capitalization"
    / "public_corporations_market_capitalization_2000_2026_shorts_v3.mp4"
)
DEFAULT_FLAGS_DIR = PROJECT_ROOT / "data" / "raw" / "flags"

TITLE = "BIGGEST COMPANIES"
SUBTITLE = "TOP 12 | MARKET CAP | 2000-2026"
LEFT_HEADER_LABEL = "COMPANY"
RIGHT_HEADER_LABEL = "MARKET CAP"
FOOTER = "WORLD'S BIGGEST PUBLIC COMPANIES | 2000-2026"

FPS = 60
TOTAL_DURATION = 100.0
FINAL_HOLD_DURATION = 5.0
TOP_N = 12


def _company_id(name: str) -> str:
    return re.sub(r"[^A-Z0-9]+", "_", name.upper()).strip("_")


def _format_market_cap(value: float) -> str:
    if abs(value) >= 1_000_000_000_000:
        text = f"${value / 1_000_000_000_000:,.2f}T"
        return text.replace(".00T", "T")
    if abs(value) >= 1_000_000_000:
        text = f"${value / 1_000_000_000:,.1f}B"
        return text.replace(".0B", "B")
    return f"${value / 1_000_000:,.0f}M"


def _format_snapshot_label(snapshot: object) -> str:
    ranking_date = str(getattr(snapshot, "ranking_date", "")).strip()
    year = ranking_date[:4]
    if ranking_date.endswith("-12-31"):
        return year
    try:
        month = int(ranking_date[5:7])
    except ValueError:
        return year or ranking_date
    quarter = ((month - 1) // 3) + 1
    return f"Q{quarter} {year}"


def build_short_input(source_path: Path, output_path: Path) -> Path:
    source_rows: list[dict[str, str]] = []
    with source_path.open("r", newline="", encoding="utf-8-sig") as source:
        source_rows = list(csv.DictReader(source))

    latest_date_by_year: dict[str, str] = {}
    for row in source_rows:
        ranking_date = row.get("ranking_date", "").strip()
        if ranking_date:
            year = ranking_date[:4]
            latest_date_by_year[year] = max(latest_date_by_year.get(year, ""), ranking_date)

    rows: list[dict[str, str]] = []
    for row in source_rows:
            company_name = row.get("company_name", "").strip()
            ranking_date = row.get("ranking_date", "").strip()
            if not company_name or not ranking_date:
                continue
            if latest_date_by_year.get(ranking_date[:4]) != ranking_date:
                continue
            try:
                market_cap_usd = float(row.get("market_cap_usd_m", "0")) * 1_000_000
            except ValueError:
                continue
            if market_cap_usd <= 0:
                continue
            rows.append(
                {
                    "ranking_date": ranking_date,
                    "country_name": company_name,
                    "country_code": "",
                    "country_iso3": _company_id(company_name),
                    "population": str(int(round(market_cap_usd))),
                    "yearly_change": "",
                    "season_summary": (
                        f"Leader: {company_name} {_format_market_cap(market_cap_usd)}"
                        if row.get("rank", "").strip() == "1"
                        else ""
                    ),
                    "data_source": "Wikipedia",
                }
            )

    summaries: dict[str, str] = {}
    for row in rows:
        if row["season_summary"]:
            summaries[row["ranking_date"]] = row["season_summary"]
    for row in rows:
        row["season_summary"] = summaries.get(row["ranking_date"], "")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=[
                "ranking_date",
                "country_name",
                "country_code",
                "country_iso3",
                "population",
                "yearly_change",
                "season_summary",
                "data_source",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def _patch_theme() -> dict[str, object]:
    previous = {
        "TITLE": base.TITLE,
        "SUBTITLE": base.SUBTITLE,
        "LEFT_HEADER_LABEL": base.LEFT_HEADER_LABEL,
        "RIGHT_HEADER_LABEL": base.RIGHT_HEADER_LABEL,
        "FOOTER": base.FOOTER,
        "YEAR_LABEL": base.YEAR_LABEL,
        "_format_population": base._format_population,
        "_make_background": base._make_background,
        "_build_color_map": base._build_color_map,
    }
    base.TITLE = TITLE
    base.SUBTITLE = SUBTITLE
    base.LEFT_HEADER_LABEL = LEFT_HEADER_LABEL
    base.RIGHT_HEADER_LABEL = RIGHT_HEADER_LABEL
    base.FOOTER = FOOTER
    base.YEAR_LABEL = _format_snapshot_label
    base._format_population = _format_market_cap
    base._make_background = lambda: make_background(base.WIDTH, base.HEIGHT)
    base._build_color_map = build_stable_color_map
    return previous


def _restore_theme(previous: dict[str, object]) -> None:
    base.TITLE = previous["TITLE"]  # type: ignore[assignment]
    base.SUBTITLE = previous["SUBTITLE"]  # type: ignore[assignment]
    base.LEFT_HEADER_LABEL = previous["LEFT_HEADER_LABEL"]  # type: ignore[assignment]
    base.RIGHT_HEADER_LABEL = previous["RIGHT_HEADER_LABEL"]  # type: ignore[assignment]
    base.FOOTER = previous["FOOTER"]  # type: ignore[assignment]
    base.YEAR_LABEL = previous["YEAR_LABEL"]  # type: ignore[assignment]
    base._format_population = previous["_format_population"]  # type: ignore[assignment]
    base._make_background = previous["_make_background"]  # type: ignore[assignment]
    base._build_color_map = previous["_build_color_map"]  # type: ignore[assignment]


def render_video(
    input_csv: Path,
    adapted_input_csv: Path,
    output_path: Path,
    flags_dir: Path,
    audio_path: Path,
    duration: float,
    final_hold_duration: float,
    fps: int,
    top_n: int,
) -> Path:
    short_input = build_short_input(input_csv, adapted_input_csv)
    previous = _patch_theme()
    try:
        return base.render_video(
            input_csv=short_input,
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
    parser = argparse.ArgumentParser(description="Generate the market-cap race with the Forbes short layout.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--adapted-input", type=Path, default=DEFAULT_ADAPTED_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--flags-dir", type=Path, default=DEFAULT_FLAGS_DIR)
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
        adapted_input_csv=args.adapted_input,
        output_path=args.output,
        flags_dir=args.flags_dir,
        audio_path=args.audio,
        duration=args.duration,
        final_hold_duration=args.final_hold,
        fps=args.fps,
        top_n=args.top_n,
    )
    print(f"[video_generator] market cap race short generated -> {output}")


if __name__ == "__main__":
    main()
