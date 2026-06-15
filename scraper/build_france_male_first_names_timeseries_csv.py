from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scraper.demography.build_france_male_first_names_timeseries_csv import main


if __name__ == "__main__":
    main()
