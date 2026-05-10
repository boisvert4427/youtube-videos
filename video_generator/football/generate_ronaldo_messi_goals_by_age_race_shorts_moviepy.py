from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from video_generator.tennis import generate_federer_nadal_djokovic_age_race_shorts_moviepy as age_race

DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "football" / "ronaldo_messi_goals_by_age_race_midnight.mp4"
DEFAULT_PHOTOS_DIR = PROJECT_ROOT / "data" / "raw" / "player_photos"
DEFAULT_AUDIO = PROJECT_ROOT / "data" / "raw" / "audio" / "Midnight_Grip_20260402_0828.mp3"
DEFAULT_PREVIEW = PROJECT_ROOT / "tmp_frames" / "ronaldo_messi_goals_by_age_race_preview.png"


MESSI_GOALS = {
    18: 1,
    19: 11,
    20: 30,
    21: 51,
    22: 92,
    23: 140,
    24: 197,
    25: 279,
    26: 348,
    27: 394,
    28: 458,
    29: 508,
    30: 565,
    31: 616,
    32: 671,
    33: 699,
    34: 745,
    35: 769,
    36: 807,
    37: 837,
    38: 866,
    39: 907,
}

RONALDO_GOALS = {
    18: 5,
    19: 6,
    20: 23,
    21: 36,
    22: 62,
    23: 102,
    24: 132,
    25: 160,
    26: 213,
    27: 272,
    28: 339,
    29: 403,
    30: 463,
    31: 521,
    32: 575,
    33: 628,
    34: 677,
    35: 722,
    36: 762,
    37: 803,
    38: 820,
    39: 873,
}


def configure_age_race() -> None:
    age_race.DEFAULT_OUTPUT = DEFAULT_OUTPUT
    age_race.DEFAULT_PHOTOS_DIR = DEFAULT_PHOTOS_DIR
    age_race.DEFAULT_PREVIEW = DEFAULT_PREVIEW
    age_race.AGE_MIN = 18
    age_race.AGE_MAX = 39
    age_race.TITLE = "RONALDO vs MESSI"
    age_race.SUBTITLE = "Goals by Age"
    age_race.TICK_AGES = [18, 21, 24, 27, 30, 33, 36, 39]
    age_race.VALUE_SCALE_MAX = 950.0
    age_race.CHART_LEFT = 390
    age_race.CARD_X = 92
    age_race.ROW_CENTER_OFFSET = 70
    age_race.PLAYER_TIE_ORDER = {
        "Cristiano Ronaldo": 0,
        "Lionel Messi": 1,
    }
    age_race.PLAYERS = [
        age_race.Player(
            name="Cristiano Ronaldo",
            short="RONALDO",
            photo_name="cristiano_ronaldo.jpg",
            color="#F13D47",
            counts=RONALDO_GOALS,
        ),
        age_race.Player(
            name="Lionel Messi",
            short="MESSI",
            photo_name="lionel_messi.jpg",
            color="#34A6FF",
            counts=MESSI_GOALS,
        ),
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a Ronaldo vs Messi goals-by-age Shorts video.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--photos-dir", type=Path, default=DEFAULT_PHOTOS_DIR)
    parser.add_argument("--audio", type=Path, default=DEFAULT_AUDIO)
    parser.add_argument("--duration", type=float, default=40.0)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--preview-image", type=Path, default=DEFAULT_PREVIEW)
    parser.add_argument("--preview-age", type=float, default=30.0)
    return parser.parse_args()


def main() -> None:
    configure_age_race()
    args = parse_args()
    output = age_race.render_video(
        output_path=args.output,
        photos_dir=args.photos_dir,
        audio_path=args.audio,
        duration=args.duration,
        fps=args.fps,
        preview_image=args.preview_image,
        preview_age=args.preview_age,
    )
    print(f"[video_generator] Ronaldo vs Messi goals-by-age race generated -> {output}")


if __name__ == "__main__":
    main()
