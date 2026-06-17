from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from video_generator.technology import generate_browser_market_share_race_shorts_moviepy as base


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_INPUT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "video_game_sales"
    / "video_game_sales_publishers_1980_2017.csv"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "video_game_sales"
    / "video_game_sales_publishers_race_1980_2017_short.mp4"
)
DEFAULT_LOGOS_DIR = PROJECT_ROOT / "data" / "raw" / "video_game_sales" / "logos"

TITLE = "VIDEO GAME SALES WARS"
SUBTITLE = "TOP PUBLISHERS | 1980-2017"
FOOTER = "KAGGLE / VGCHARTZ | CUMULATIVE GLOBAL SALES"
TITLE_FONT_SIZE = 38
SUBTITLE_FONT_SIZE = 18
START_DATE = "1980-01-01"
EXCLUDED_KEYS = {"unknown"}
FPS = 60
TOTAL_DURATION = 60.0
FINAL_HOLD_DURATION = 5.0
TOP_N = 10


def _axis_value(value: float) -> str:
    if value >= 1000.0:
        return f"{value / 1000.0:.1f}B"
    return f"{value:.0f}M"


def _nice_number(value: float) -> float:
    if value <= 0:
        return 1.0
    exponent = math.floor(math.log10(value))
    fraction = value / (10 ** exponent)
    if fraction <= 1:
        nice_fraction = 1
    elif fraction <= 2:
        nice_fraction = 2
    elif fraction <= 2.5:
        nice_fraction = 2.5
    elif fraction <= 5:
        nice_fraction = 5
    else:
        nice_fraction = 10
    return nice_fraction * (10 ** exponent)


def _era_label(ranking_date: str) -> str:
    if ranking_date < "1990-01-01":
        return "Arcade and 8-bit era"
    if ranking_date < "2000-01-01":
        return "16-bit and PlayStation era"
    if ranking_date < "2010-01-01":
        return "PS2, Wii and Xbox era"
    return "HD and digital era"


def _patch_theme() -> dict[str, object]:
    previous = {
        "TITLE": base.TITLE,
        "SUBTITLE": base.SUBTITLE,
        "TITLE_FONT_SIZE": base.TITLE_FONT_SIZE,
        "SUBTITLE_FONT_SIZE": base.SUBTITLE_FONT_SIZE,
        "FOOTER": base.FOOTER,
        "ERA_LABEL": base.ERA_LABEL,
        "TICK_LABEL_FORMAT": base.TICK_LABEL_FORMAT,
        "VALUE_LABEL_FORMAT": base.VALUE_LABEL_FORMAT,
    }
    base.TITLE = TITLE
    base.SUBTITLE = SUBTITLE
    base.TITLE_FONT_SIZE = TITLE_FONT_SIZE
    base.SUBTITLE_FONT_SIZE = SUBTITLE_FONT_SIZE
    base.FOOTER = FOOTER
    base.ERA_LABEL = _era_label
    base.TICK_LABEL_FORMAT = _axis_value
    base.VALUE_LABEL_FORMAT = _axis_value
    return previous


def _restore_theme(previous: dict[str, object]) -> None:
    base.TITLE = previous["TITLE"]  # type: ignore[assignment]
    base.SUBTITLE = previous["SUBTITLE"]  # type: ignore[assignment]
    base.TITLE_FONT_SIZE = previous["TITLE_FONT_SIZE"]  # type: ignore[assignment]
    base.SUBTITLE_FONT_SIZE = previous["SUBTITLE_FONT_SIZE"]  # type: ignore[assignment]
    base.FOOTER = previous["FOOTER"]  # type: ignore[assignment]
    base.ERA_LABEL = previous["ERA_LABEL"]  # type: ignore[assignment]
    base.TICK_LABEL_FORMAT = previous["TICK_LABEL_FORMAT"]  # type: ignore[assignment]
    base.VALUE_LABEL_FORMAT = previous["VALUE_LABEL_FORMAT"]  # type: ignore[assignment]


def render_video(
    input_csv: Path,
    output_path: Path,
    logos_dir: Path,
    audio_path: Path,
    duration: float,
    final_hold_duration: float,
    fps: int,
    top_n: int,
) -> Path:
    snapshots = base.load_snapshots(input_csv)
    snapshots = [
        snapshot
        for snapshot in snapshots
        if snapshot.ranking_date >= START_DATE
    ]
    snapshots = [
        base.Snapshot(
            ranking_date=snapshot.ranking_date,
            season_summary=snapshot.season_summary,
            states=[
                state for state in snapshot.states if state.browser_key not in EXCLUDED_KEYS
            ],
        )
        for snapshot in snapshots
    ]
    max_value = max(
        (state.market_share for snapshot in snapshots for state in snapshot.states),
        default=1.0,
    )
    axis_step = _nice_number(max_value * 1.08 / 6.0)
    axis_cap = math.ceil(max_value * 1.08 / axis_step) * axis_step
    previous = _patch_theme()
    try:
        return base.render_video(
            input_csv=input_csv,
            output_path=output_path,
            logos_dir=logos_dir,
            audio_path=audio_path,
            duration=duration,
            final_hold_duration=final_hold_duration,
            fps=fps,
            top_n=top_n,
            start_date=START_DATE,
            excluded_keys=EXCLUDED_KEYS,
            axis_cap=axis_cap,
            tick_step=axis_step,
        )
    finally:
        _restore_theme(previous)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a video game sales publisher Short.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--logos-dir", type=Path, default=DEFAULT_LOGOS_DIR)
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
        logos_dir=args.logos_dir,
        audio_path=args.audio,
        duration=args.duration,
        final_hold_duration=args.final_hold,
        fps=args.fps,
        top_n=args.top_n,
    )
    print(f"[video_generator] video game sales publisher Short generated -> {output}")


if __name__ == "__main__":
    main()
