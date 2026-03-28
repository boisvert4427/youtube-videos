from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from video_generator.basketball.generate_lebron_vs_jordan_goat_split_shorts_moviepy import main


if __name__ == "__main__":
    main()
