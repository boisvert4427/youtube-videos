import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scraper.games.build_video_game_sales_publishers_timeseries_csv import main


if __name__ == "__main__":
    main()
