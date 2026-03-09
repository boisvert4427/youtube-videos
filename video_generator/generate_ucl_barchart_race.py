from __future__ import annotations

import argparse
from pathlib import Path
import sys

try:
    from video_generator.generate_atp_barchart_race import (
        PROJECT_ROOT,
        interpolate_snapshots,
        load_snapshots,
        render_video,
    )
except ModuleNotFoundError:
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from video_generator.generate_atp_barchart_race import (
        PROJECT_ROOT,
        interpolate_snapshots,
        load_snapshots,
        render_video,
    )


DEFAULT_INPUT = PROJECT_ROOT / "data" / "processed" / "ucl_titles_timeseries_1956_2025.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "ucl_titles_race.mp4"
DEFAULT_FLAGS_DIR = PROJECT_ROOT / "data" / "raw" / "flags"
DEFAULT_PHOTOS_DIR = PROJECT_ROOT / "data" / "raw" / "club_logos"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate UEFA Champions League titles bar chart race from CSV")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="CSV path")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output .mp4 or .gif path")
    parser.add_argument("--photos-dir", type=Path, default=DEFAULT_PHOTOS_DIR, help="Optional folder with club logos/photos")
    parser.add_argument("--flags-dir", type=Path, default=DEFAULT_FLAGS_DIR, help="Folder for cached country flag PNGs")
    parser.add_argument("--fps", type=int, default=60, help="Video fps")
    parser.add_argument("--frames-per-period", type=int, default=52, help="Interpolation frames between ranking dates")
    parser.add_argument("--top-n", type=int, default=15, help="Number of bars displayed")
    parser.add_argument("--title", type=str, default="UEFA Champions League Titles", help="Chart title")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    snapshots = load_snapshots(args.input)
    if not snapshots:
        raise RuntimeError(f"No data found in {args.input}")

    frames = interpolate_snapshots(snapshots, frames_per_period=args.frames_per_period)
    output = render_video(
        frames=frames,
        output_path=args.output,
        top_n=args.top_n,
        title=args.title,
        fps=args.fps,
        photos_dir=args.photos_dir,
        flags_dir=args.flags_dir,
        year_in_media_box=True,
        leader_logo_in_header=True,
        show_bottom_date=False,
        position_lerp=0.22,
    )
    print(f"[video_generator] UCL bar chart race generated -> {output}")


if __name__ == "__main__":
    main()
