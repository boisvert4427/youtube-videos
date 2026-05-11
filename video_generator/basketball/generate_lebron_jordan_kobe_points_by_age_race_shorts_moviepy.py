from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from video_generator.tennis import generate_federer_nadal_djokovic_age_race_shorts_moviepy as age_race


DEFAULT_INPUT = PROJECT_ROOT / "data" / "processed" / "basketball" / "nba_points_by_age_lebron_jordan_kobe.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "basketball" / "lebron_jordan_kobe_points_by_age_race_shorts.mp4"
DEFAULT_PHOTOS_DIR = PROJECT_ROOT / "data" / "raw" / "nba_goat_assets"
DEFAULT_AUDIO = PROJECT_ROOT / "data" / "raw" / "audio" / "Midnight_Grip_20260402_0828.mp3"
DEFAULT_PREVIEW = PROJECT_ROOT / "tmp_frames" / "lebron_jordan_kobe_points_by_age_race_preview.png"


PLAYER_CONFIG = {
    "LeBron James": {
        "short": "LEBRON",
        "photo_name": "lebron_portrait.jpg",
        "color": "#F6C945",
    },
    "Michael Jordan": {
        "short": "JORDAN",
        "photo_name": "jordan_1997_color.jpg",
        "color": "#F01832",
    },
    "Kobe Bryant": {
        "short": "KOBE",
        "photo_name": "kobe_wiki.jpg",
        "color": "#552583",
    },
}


def load_counts(input_csv: Path) -> dict[str, dict[int, int]]:
    counts: dict[str, dict[int, int]] = {}
    with input_csv.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            player = row["player"]
            counts.setdefault(player, {})[int(row["age"])] = int(row["cumulative_points"])
    return counts


def configure_age_race(input_csv: Path) -> None:
    counts = load_counts(input_csv)
    age_race.DEFAULT_OUTPUT = DEFAULT_OUTPUT
    age_race.DEFAULT_PHOTOS_DIR = DEFAULT_PHOTOS_DIR
    age_race.DEFAULT_PREVIEW = DEFAULT_PREVIEW
    age_race.AGE_MIN = min(age for player_counts in counts.values() for age in player_counts)
    age_race.AGE_MAX = max(age for player_counts in counts.values() for age in player_counts)
    age_race.TITLE = "LEBRON vs JORDAN vs KOBE"
    age_race.SUBTITLE = "Points by Age"
    age_race.TICK_AGES = [18, 21, 24, 27, 30, 33, 36, 39, 41]
    age_race.VALUE_SCALE_MAX = max(max(player_counts.values()) for player_counts in counts.values()) * 1.03
    age_race.CHART_LEFT = 350
    age_race.CARD_X = 68
    age_race.ROW_CENTER_OFFSET = 35
    age_race.PLAYER_TIE_ORDER = {
        "LeBron James": 0,
        "Michael Jordan": 1,
        "Kobe Bryant": 2,
    }
    age_race.PLAYERS = [
        age_race.Player(
            name=player,
            short=config["short"],
            photo_name=config["photo_name"],
            color=config["color"],
            counts=counts[player],
        )
        for player, config in PLAYER_CONFIG.items()
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate LeBron vs Jordan vs Kobe points-by-age race Short.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--photos-dir", type=Path, default=DEFAULT_PHOTOS_DIR)
    parser.add_argument("--audio", type=Path, default=DEFAULT_AUDIO)
    parser.add_argument("--duration", type=float, default=40.0)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--preview-image", type=Path, default=DEFAULT_PREVIEW)
    parser.add_argument("--preview-age", type=float, default=33.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_age_race(args.input)
    output = age_race.render_video(
        output_path=args.output,
        photos_dir=args.photos_dir,
        audio_path=args.audio,
        duration=args.duration,
        fps=args.fps,
        preview_image=args.preview_image,
        preview_age=args.preview_age,
    )
    print(f"[video_generator] LeBron vs Jordan vs Kobe points-by-age race generated -> {output}")


if __name__ == "__main__":
    main()
