from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from video_generator.football.generate_messi_vs_ronaldo_career_shorts_moviepy import main


if __name__ == "__main__":
    main()
