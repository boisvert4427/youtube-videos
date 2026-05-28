from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
NESTED_PROJECT_ROOT = PROJECT_ROOT / "youtube-videos"
if str(NESTED_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(NESTED_PROJECT_ROOT))

from video_generator.tennis._generate_roland_garros_titles_cards_matchpoint_preview_impl import main


if __name__ == "__main__":
    main()
