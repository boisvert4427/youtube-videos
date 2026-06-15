from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from video_generator.technology import generate_browser_market_share_race_moviepy as base


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
    / "mobile_vendor_market_share_race_2010_2025_top10_2min.mp4"
)
DEFAULT_LOGOS_DIR = (
    PROJECT_ROOT / "data" / "raw" / "technology" / "mobile_vendor_market_share" / "logos"
)

TITLE = "MOBILE BRAND WARS"
SUBTITLE = "WORLDWIDE MARKET SHARE | 2010-2025"
TITLE_FONT_SIZE = 52
SUBTITLE_FONT_SIZE = 22
LEFT_HEADER_LABEL = "BRAND"
RIGHT_HEADER_LABEL = "MARKET SHARE"
FOOTER = "MOBILE VENDOR MARKET SHARE | APR 2010-2025"

FPS = 60
TOTAL_DURATION = 120.0
FINAL_HOLD_DURATION = 10.0
TOP_N = 10
START_DATE = "2010-04-01"
EXCLUDED_KEYS = {"unknown"}


def _era_label(ranking_date: str) -> str:
    if ranking_date < "2013-01-01":
        return "Nokia, Apple and BlackBerry"
    if ranking_date < "2017-01-01":
        return "Samsung takes the lead"
    if ranking_date < "2021-01-01":
        return "The rise of Chinese brands"
    return "Apple vs Samsung"


def _patch_theme() -> dict[str, object]:
    previous = {
        "TITLE": base.TITLE,
        "SUBTITLE": base.SUBTITLE,
        "TITLE_FONT_SIZE": base.TITLE_FONT_SIZE,
        "SUBTITLE_FONT_SIZE": base.SUBTITLE_FONT_SIZE,
        "LEFT_HEADER_LABEL": base.LEFT_HEADER_LABEL,
        "RIGHT_HEADER_LABEL": base.RIGHT_HEADER_LABEL,
        "FOOTER": base.FOOTER,
        "ERA_LABEL": base.ERA_LABEL,
    }
    base.TITLE = TITLE
    base.SUBTITLE = SUBTITLE
    base.TITLE_FONT_SIZE = TITLE_FONT_SIZE
    base.SUBTITLE_FONT_SIZE = SUBTITLE_FONT_SIZE
    base.LEFT_HEADER_LABEL = LEFT_HEADER_LABEL
    base.RIGHT_HEADER_LABEL = RIGHT_HEADER_LABEL
    base.FOOTER = FOOTER
    base.ERA_LABEL = _era_label
    return previous


def _restore_theme(previous: dict[str, object]) -> None:
    base.TITLE = previous["TITLE"]  # type: ignore[assignment]
    base.SUBTITLE = previous["SUBTITLE"]  # type: ignore[assignment]
    base.TITLE_FONT_SIZE = previous["TITLE_FONT_SIZE"]  # type: ignore[assignment]
    base.SUBTITLE_FONT_SIZE = previous["SUBTITLE_FONT_SIZE"]  # type: ignore[assignment]
    base.LEFT_HEADER_LABEL = previous["LEFT_HEADER_LABEL"]  # type: ignore[assignment]
    base.RIGHT_HEADER_LABEL = previous["RIGHT_HEADER_LABEL"]  # type: ignore[assignment]
    base.FOOTER = previous["FOOTER"]  # type: ignore[assignment]
    base.ERA_LABEL = previous["ERA_LABEL"]  # type: ignore[assignment]


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
        )
    finally:
        _restore_theme(previous)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a mobile vendor market share race.")
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
    print(f"[video_generator] mobile vendor market share race generated -> {output}")


if __name__ == "__main__":
    main()
