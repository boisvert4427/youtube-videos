from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scraper.technology.build_browser_market_share_timeseries_csv import main


if __name__ == "__main__":
    main()
