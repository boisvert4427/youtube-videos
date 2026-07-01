from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from video_generator.cycling import generate_tour_de_france_stage_wins_race_moviepy as base


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_INPUT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "cycling"
    / "tour_de_france"
    / "tour_de_france_yellow_jersey_days_postwar_1947_2025.csv"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "cycling"
    / "tour_de_france"
    / "tour_de_france_yellow_jersey_days_postwar_1947_2025.mp4"
)

TITLE = "TOUR DE FRANCE YELLOW JERSEY"
SUBTITLE = "MOST DAYS IN YELLOW | 1947-2025"
INTRO_SUMMARY = "Post-war race|Days in yellow|Since 1947"
YEAR_BOX_RECT = (1518, 58, 1798, 136)
SUMMARY_BOX_RECT = (1236, 182, 1848, 404)
INFO_BOX_RECT = (1252, 706, 1848, 886)


def _patch_theme() -> dict[str, object]:
    previous = {
        "TITLE": base.TITLE,
        "SUBTITLE": base.SUBTITLE,
        "INTRO_SUMMARY": base.INTRO_SUMMARY,
        "YEAR_BOX_RECT": base.YEAR_BOX_RECT,
        "SUMMARY_BOX_RECT": base.SUMMARY_BOX_RECT,
        "INFO_BOX_RECT": base.INFO_BOX_RECT,
    }
    base.TITLE = TITLE
    base.SUBTITLE = SUBTITLE
    base.INTRO_SUMMARY = INTRO_SUMMARY
    base.YEAR_BOX_RECT = YEAR_BOX_RECT
    base.SUMMARY_BOX_RECT = SUMMARY_BOX_RECT
    base.INFO_BOX_RECT = INFO_BOX_RECT
    return previous


def _restore_theme(previous: dict[str, object]) -> None:
    base.TITLE = previous["TITLE"]  # type: ignore[assignment]
    base.SUBTITLE = previous["SUBTITLE"]  # type: ignore[assignment]
    base.INTRO_SUMMARY = previous["INTRO_SUMMARY"]  # type: ignore[assignment]
    base.YEAR_BOX_RECT = previous["YEAR_BOX_RECT"]  # type: ignore[assignment]
    base.SUMMARY_BOX_RECT = previous["SUMMARY_BOX_RECT"]  # type: ignore[assignment]
    base.INFO_BOX_RECT = previous["INFO_BOX_RECT"]  # type: ignore[assignment]


def render_video(
    input_csv: Path,
    output_path: Path,
    flags_dir: Path,
    photos_dir: Path,
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
            photos_dir=photos_dir,
            audio_path=audio_path,
            duration=duration,
            final_hold_duration=final_hold_duration,
            fps=fps,
            top_n=top_n,
        )
    finally:
        _restore_theme(previous)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a landscape Tour de France yellow jersey days race video.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--flags-dir", type=Path, default=base.DEFAULT_FLAGS_DIR)
    parser.add_argument("--photos-dir", type=Path, default=base.DEFAULT_PHOTOS_DIR)
    parser.add_argument("--audio", type=Path, default=base.DEFAULT_AUDIO)
    parser.add_argument("--duration", type=float, default=base.TOTAL_DURATION)
    parser.add_argument("--final-hold", type=float, default=base.FINAL_HOLD_DURATION)
    parser.add_argument("--fps", type=int, default=base.FPS)
    parser.add_argument("--top-n", type=int, default=base.TOP_N)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = render_video(
        input_csv=args.input,
        output_path=args.output,
        flags_dir=args.flags_dir,
        photos_dir=args.photos_dir,
        audio_path=args.audio,
        duration=args.duration,
        final_hold_duration=args.final_hold,
        fps=args.fps,
        top_n=args.top_n,
    )
    print(f"[video_generator] Tour de France yellow jersey days landscape race generated -> {output}")


if __name__ == "__main__":
    main()
