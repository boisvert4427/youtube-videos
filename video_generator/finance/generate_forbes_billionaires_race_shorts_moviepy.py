from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from video_generator.demography import generate_world_population_race_shorts_moviepy as base
from video_generator.finance.forbes_billionaires_theme import (
    build_stable_color_map,
    format_money,
    format_snapshot_label,
    make_background,
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
    / "forbes_billionaires_race_1997_2024_shorts.mp4"
)
DEFAULT_FLAGS_DIR = PROJECT_ROOT / "data" / "raw" / "flags"

TITLE = "LES MILLIARDAIRES"
SUBTITLE = "TOP 12 | 1997-2024"
LEFT_HEADER_LABEL = "NOM"
RIGHT_HEADER_LABEL = "FORTUNE"
FOOTER = "MILLIARDAIRES | 1997-2024"

FPS = 60
TOTAL_DURATION = 100.0
FINAL_HOLD_DURATION = 5.0
TOP_N = 12


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
    base.YEAR_LABEL = format_snapshot_label
    base._format_population = format_money
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
    output_path: Path,
    flags_dir: Path,
    audio_path: Path,
    duration: float,
    final_hold_duration: float,
    fps: int,
    top_n: int,
) -> Path:
    previous = _patch_theme()
    try:
        return base.render_video(
            input_csv=input_csv,
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
    parser = argparse.ArgumentParser(description="Generate a vertical Forbes billionaires bar chart race Short.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
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
        output_path=args.output,
        flags_dir=args.flags_dir,
        audio_path=args.audio,
        duration=args.duration,
        final_hold_duration=args.final_hold,
        fps=args.fps,
        top_n=args.top_n,
    )
    print(f"[video_generator] Forbes billionaires race short generated -> {output}")


if __name__ == "__main__":
    main()
