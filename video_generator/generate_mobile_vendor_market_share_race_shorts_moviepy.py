import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from video_generator.technology.generate_mobile_vendor_market_share_race_shorts_moviepy import main


if __name__ == "__main__":
    main()
