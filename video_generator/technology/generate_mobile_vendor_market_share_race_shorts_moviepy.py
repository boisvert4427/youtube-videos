from __future__ import annotations

import argparse
import sys
from pathlib import Path
import math

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from video_generator.technology import generate_browser_market_share_race_shorts_moviepy as base


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_INPUT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "technology"
    / "mobile_vendor_market_share"
    / "mobile_vendor_market_share_2009_2025.csv"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "technology"
    / "mobile_vendor_market_share"
    / "mobile_vendor_market_share_race_2010_2025_top10_short.mp4"
)
DEFAULT_LOGOS_DIR = (
    PROJECT_ROOT / "data" / "raw" / "technology" / "mobile_vendor_market_share" / "logos"
)

TITLE = "MOBILE BRAND WARS"
SUBTITLE = "WORLDWIDE MARKET SHARE | 2010-2025"
FOOTER = "MOBILE VENDOR MARKET SHARE | APR 2010-2025"
START_DATE = "2010-04-01"
EXCLUDED_KEYS = {"unknown"}
FPS = 60
TOTAL_DURATION = 60.0
FINAL_HOLD_DURATION = 5.0
TOP_N = 10


def _era_label(ranking_date: str) -> str:
    if ranking_date < "2013-01-01":
        return "Nokia, Apple and BlackBerry"
    if ranking_date < "2017-01-01":
        return "Samsung takes the lead"
    if ranking_date < "2021-01-01":
        return "The rise of Chinese brands"
    return "Apple vs Samsung"


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
        snapshot for snapshot in snapshots if snapshot.ranking_date >= START_DATE
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
    max_share = max(
        (state.market_share for snapshot in snapshots for state in snapshot.states),
        default=100.0,
    )
    visual_cap = max(10.0, float(math.ceil(max_share)))

    previous = (base.TITLE, base.SUBTITLE, base.FOOTER, base.ERA_LABEL)
    base.TITLE = TITLE
    base.SUBTITLE = SUBTITLE
    base.FOOTER = FOOTER
    base.ERA_LABEL = _era_label
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
            axis_cap=visual_cap,
        )
    finally:
        base.TITLE, base.SUBTITLE, base.FOOTER, base.ERA_LABEL = previous


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a mobile vendor market share Short.")
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
    print(f"[video_generator] mobile vendor market share Short generated -> {output}")


if __name__ == "__main__":
    main()
