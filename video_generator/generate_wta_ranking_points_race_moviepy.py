from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from video_generator.tennis.generate_wta_ranking_points_race_moviepy import main


if __name__ == "__main__":
    main()
