from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scraper.cycling.download_paris_nice_winner_photos import main


if __name__ == "__main__":
    main()
