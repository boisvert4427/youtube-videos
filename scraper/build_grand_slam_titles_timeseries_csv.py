from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scraper.tennis.build_grand_slam_titles_timeseries_csv import run


if __name__ == "__main__":
    output = run()
    print(f"[scraper] Grand Slam titles timeseries generated -> {output}")
