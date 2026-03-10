from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scraper.cycling.build_paris_nice_postwar_csv import main


if __name__ == "__main__":
    main()
