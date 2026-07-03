from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scraper.cycling.build_tour_de_france_through_the_years_csv import main


if __name__ == "__main__":
    main()
