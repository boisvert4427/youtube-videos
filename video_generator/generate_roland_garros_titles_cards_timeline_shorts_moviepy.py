from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from video_generator.tennis.generate_roland_garros_titles_cards_timeline_shorts_moviepy import main


if __name__ == "__main__":
    main()
