from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from video_generator.cycling.generate_tour_of_flanders_titles_cards_shorts_moviepy import main


if __name__ == "__main__":
    main()
