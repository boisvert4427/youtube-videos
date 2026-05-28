from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scraper.tennis.build_roland_garros_titles_cards_csv import run


if __name__ == "__main__":
    output = run()
    print(f"[scraper] Roland-Garros titles cards CSV generated -> {output}")

