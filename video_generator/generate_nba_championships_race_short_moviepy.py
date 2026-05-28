from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from video_generator.basketball.generate_nba_championships_race_short_moviepy import main


if __name__ == "__main__":
    main()
