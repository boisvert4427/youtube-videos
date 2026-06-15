from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from video_generator.demography import generate_france_female_first_names_race_shorts_moviepy as base


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_INPUT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "demography"
    / "france_male_first_names"
    / "france_male_first_names_1900_2024.csv"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "demography"
    / "france_male_first_names"
    / "france_male_first_names_race_1900_2024_shorts.mp4"
)

TITLE = "PRÉNOMS MASCULINS"
SUBTITLE = "FRANCE | 1900-2024"
FPS = 60
TOTAL_DURATION = 100.0
FINAL_HOLD_DURATION = 8.0
INTRO_HOLD_DURATION = 4.0
TOP_N = 12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a vertical French male first-name bar chart race Short."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--audio", type=Path, default=base.DEFAULT_AUDIO)
    parser.add_argument("--duration", type=float, default=TOTAL_DURATION)
    parser.add_argument("--final-hold", type=float, default=FINAL_HOLD_DURATION)
    parser.add_argument("--intro-hold", type=float, default=INTRO_HOLD_DURATION)
    parser.add_argument("--fps", type=int, default=FPS)
    parser.add_argument("--top-n", type=int, default=TOP_N)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    previous_title = base.TITLE
    previous_subtitle = base.SUBTITLE
    previous_footer = base.FOOTER
    base.TITLE = TITLE
    base.SUBTITLE = SUBTITLE
    base.FOOTER = "PRÉNOMS MASCULINS | FRANCE | 1900-2024"
    try:
        output = base.render_video(
            input_csv=args.input,
            output_path=args.output,
            audio_path=args.audio,
            duration=args.duration,
            final_hold_duration=args.final_hold,
            intro_hold_duration=args.intro_hold,
            fps=args.fps,
            top_n=args.top_n,
        )
    finally:
        base.TITLE = previous_title
        base.SUBTITLE = previous_subtitle
        base.FOOTER = previous_footer

    print(f"[video_generator] France male first-name short generated -> {output}")


if __name__ == "__main__":
    main()
