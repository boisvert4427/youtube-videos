from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from video_generator.cycling.generate_tour_de_france_through_the_years_moviepy import main


if __name__ == "__main__":
    main()
