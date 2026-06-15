from __future__ import annotations

import argparse
import csv
import re
import shutil
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw" / "technology" / "browser_market_share"
DEFAULT_RAW_CSV = DEFAULT_RAW_DIR / "statcounter_browser_market_share_monthly.csv"
DEFAULT_LOGOS_DIR = DEFAULT_RAW_DIR / "logos"
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "technology"
    / "browser_market_share"
    / "browser_market_share_1995_2026.csv"
)

SOURCE_PAGE = "https://gs.statcounter.com/browser-market-share"
HISTORICAL_REFERENCE_PAGE = "https://en.wikipedia.org/wiki/Usage_share_of_web_browsers"
CHART_ENDPOINT = "https://gs.statcounter.com/chart.php"
START_MONTH = "2009-01"
USER_AGENT = "Mozilla/5.0 (compatible; Codex Browser Market Share Builder)"

# Pre-2009 measurements use different panels and methodologies. These sparse
# published points are interpolated monthly and labeled as historical estimates.
EARLY_HISTORICAL_POINTS = [
    ("1995-01", {"IE": 2.90, "Netscape": 80.10}, "Dataquest"),
    ("1996-06", {"IE": 9.60, "Netscape": 82.77, "Mosaic": 6.93}, "EWS"),
    ("1996-09", {"IE": 13.97, "Netscape": 80.37, "Mosaic": 2.47}, "EWS"),
    ("1996-12", {"IE": 19.07, "Netscape": 77.13, "Mosaic": 1.20}, "EWS"),
    ("1997-03", {"IE": 22.87, "Netscape": 74.33, "Mosaic": 0.60}, "EWS"),
    ("1997-06", {"IE": 27.67, "Netscape": 69.77, "Mosaic": 0.37}, "EWS"),
    ("1997-09", {"IE": 32.40, "Netscape": 64.93}, "EWS"),
    ("1997-12", {"IE": 35.53, "Netscape": 62.23}, "EWS"),
    ("1998-03", {"IE": 39.67, "Netscape": 57.63}, "EWS"),
    ("1998-06", {"IE": 43.17, "Netscape": 53.57}, "EWS"),
    ("1998-09", {"IE": 47.90, "Netscape": 48.97}, "EWS"),
    ("1998-12", {"IE": 50.43, "Netscape": 46.87}, "EWS"),
    ("1999-02", {"IE": 64.60, "Netscape": 33.43}, "WebSideStory"),
    ("1999-03", {"IE": 66.90, "Netscape": 31.21}, "WebSideStory"),
    ("1999-04", {"IE": 68.75, "Netscape": 29.46}, "WebSideStory"),
    ("1999-08", {"IE": 75.31, "Netscape": 24.68}, "WebSideStory"),
]

THECOUNTER_QUARTERLY_POINTS = [
    ("2000-03", 79.09, 0.00, 0.00, 0.13, 19.25),
    ("2000-06", 80.30, 0.02, 0.00, 0.12, 17.54),
    ("2000-09", 82.76, 0.04, 0.00, 0.14, 14.35),
    ("2000-12", 83.95, 0.14, 0.00, 0.14, 12.61),
    ("2001-03", 86.80, 0.30, 0.00, 0.22, 9.84),
    ("2001-06", 87.99, 0.27, 0.00, 0.28, 7.46),
    ("2001-09", 88.43, 0.26, 0.00, 0.31, 6.49),
    ("2001-12", 90.83, 0.71, 0.00, 0.36, 5.23),
    ("2002-03", 92.40, 0.93, 0.00, 0.52, 4.67),
    ("2002-06", 92.47, 1.13, 0.00, 0.82, 4.13),
    ("2002-09", 93.32, 1.36, 0.00, 0.94, 3.04),
    ("2002-12", 93.94, 1.67, 0.00, 0.83, 2.31),
    ("2003-03", 94.18, 2.15, 0.00, 0.65, 1.77),
    ("2003-06", 94.43, 2.22, 0.00, 0.66, 1.45),
    ("2004-03", 94.28, 2.70, 0.00, 0.52, 0.36),
    ("2004-06", 95.04, 2.37, 0.67, 0.51, 0.32),
    ("2004-09", 92.70, 3.57, 0.73, 0.65, 0.20),
    ("2004-12", 90.98, 5.10, 0.77, 0.68, 0.18),
    ("2005-03", 90.77, 5.73, 1.00, 0.54, 0.11),
    ("2005-06", 90.90, 6.02, 0.99, 0.51, 0.09),
    ("2005-09", 87.58, 8.42, 1.60, 0.67, 0.07),
    ("2005-12", 87.25, 8.60, 1.83, 0.71, 0.07),
    ("2006-03", 90.01, 6.77, 1.40, 0.58, 0.05),
    ("2006-06", 86.32, 9.03, 1.89, 0.70, 0.05),
    ("2006-09", 84.48, 10.56, 2.27, 0.73, 0.06),
    ("2006-12", 84.11, 11.13, 2.80, 0.60, 0.05),
    ("2007-03", 83.69, 11.57, 2.92, 0.57, 0.06),
    ("2007-06", 82.97, 12.41, 2.87, 0.64, 0.06),
    ("2007-09", 81.63, 13.49, 3.00, 0.66, 0.06),
    ("2007-12", 81.14, 13.81, 3.21, 0.67, 0.06),
    ("2008-03", 78.80, 15.87, 3.32, 0.79, 0.06),
    ("2008-06", 78.30, 16.36, 3.41, 0.81, 0.06),
    ("2008-09", 76.33, 17.97, 3.76, 0.84, 0.07),
    ("2008-12", 74.24, 18.66, 4.52, 0.89, 0.07),
]

PNG_LOGO_URLS = {
    "brave": "https://cdn.jsdelivr.net/gh/alrra/browser-logos@master/src/brave/brave_128x128.png",
    "chrome": "https://cdn.jsdelivr.net/gh/alrra/browser-logos@master/src/chrome/chrome_128x128.png",
    "chromium": "https://cdn.jsdelivr.net/gh/alrra/browser-logos@master/src/chromium/chromium_128x128.png",
    "edge": "https://cdn.jsdelivr.net/gh/alrra/browser-logos@master/src/edge/edge_128x128.png",
    "edge_legacy": "https://cdn.jsdelivr.net/gh/alrra/browser-logos@master/src/edge/edge_128x128.png",
    "firefox": "https://cdn.jsdelivr.net/gh/alrra/browser-logos@master/src/firefox/firefox_128x128.png",
    "opera": "https://cdn.jsdelivr.net/gh/alrra/browser-logos@master/src/opera/opera_128x128.png",
    "safari": "https://cdn.jsdelivr.net/gh/alrra/browser-logos@master/src/safari/safari_128x128.png",
    "samsung_internet": (
        "https://cdn.jsdelivr.net/gh/alrra/browser-logos@master/"
        "src/samsung-internet/samsung-internet_128x128.png"
    ),
    "uc_browser": "https://cdn.jsdelivr.net/gh/alrra/browser-logos@master/src/uc/uc_128x128.png",
    "yandex_browser": "https://cdn.jsdelivr.net/gh/alrra/browser-logos@master/src/yandex/yandex_128x128.png",
}

SVG_LOGO_URLS = {
    "android": "https://cdn.simpleicons.org/android/3DDC84",
    "blackberry": "https://cdn.simpleicons.org/blackberry/FFFFFF",
    "internet_explorer": (
        "https://commons.wikimedia.org/wiki/Special:Redirect/file/"
        "Internet_Explorer_10%2B11_logo.svg"
    ),
    "nokia": "https://cdn.simpleicons.org/nokia/3B7EC1",
}


def _download(url: str, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".download")
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Referer": SOURCE_PAGE,
        },
    )
    with urllib.request.urlopen(request, timeout=120) as response, temporary_path.open("wb") as output:
        shutil.copyfileobj(response, output)
    temporary_path.replace(output_path)
    return output_path


def _latest_complete_month() -> str:
    request = urllib.request.Request(SOURCE_PAGE, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:
        html = response.read().decode("utf-8", errors="replace")
    match = re.search(
        r"Browser Market Share Worldwide\s*-\s*([A-Za-z]+)\s+(\d{4})",
        html,
        flags=re.IGNORECASE,
    )
    if not match:
        raise RuntimeError("Could not determine the latest complete Statcounter month.")
    parsed = datetime.strptime(f"{match.group(1)} {match.group(2)}", "%B %Y")
    return parsed.strftime("%Y-%m")


def _chart_url(start_month: str, end_month: str) -> str:
    start_compact = start_month.replace("-", "")
    end_compact = end_month.replace("-", "")
    query = {
        "device": "Desktop & Mobile & Tablet & Console",
        "device_hidden": "desktop+mobile+tablet+console",
        "multi-device": "true",
        "statType_hidden": "browser",
        "region_hidden": "ww",
        "granularity": "monthly",
        "statType": "Browser",
        "region": "Worldwide",
        "fromInt": start_compact,
        "toInt": end_compact,
        "fromMonthYear": start_month,
        "toMonthYear": end_month,
        "csv": "1",
    }
    return f"{CHART_ENDPOINT}?{urllib.parse.urlencode(query)}"


def _browser_key(name: str) -> str:
    aliases = {
        "IE": "internet_explorer",
        "IEMobile": "ie_mobile",
    }
    if name in aliases:
        return aliases[name]
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def _load_wide_rows(raw_csv: Path) -> tuple[list[str], list[dict[str, str]]]:
    with raw_csv.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        rows = list(reader)
        fieldnames = reader.fieldnames or []
    browser_names = [name for name in fieldnames if name != "Date"]
    if not rows or not browser_names:
        raise RuntimeError("The Statcounter CSV is empty or has no browser columns.")
    return browser_names, rows


def _month_index(month: str) -> int:
    year, month_number = (int(part) for part in month.split("-"))
    return year * 12 + month_number - 1


def _month_from_index(index: int) -> str:
    year, zero_based_month = divmod(index, 12)
    return f"{year:04d}-{zero_based_month + 1:02d}"


def _complete_shares(values: dict[str, float]) -> dict[str, float]:
    shares = {name: max(0.0, float(value)) for name, value in values.items()}
    shares["Other"] = max(0.0, 100.0 - sum(shares.values()))
    return shares


def _historical_points() -> list[tuple[str, dict[str, float], str]]:
    points = [
        (month, _complete_shares(values), source)
        for month, values, source in EARLY_HISTORICAL_POINTS
    ]
    for month, ie, combined, safari, opera, netscape in THECOUNTER_QUARTERLY_POINTS:
        combined_name = "Firefox" if month >= "2004-10" else "Mozilla"
        values = {
            "IE": ie,
            combined_name: combined,
            "Safari": safari,
            "Opera": opera,
            "Netscape": netscape,
        }
        points.append((month, _complete_shares(values), "TheCounter.com"))
    return sorted(points, key=lambda point: point[0])


def _historical_monthly_snapshots() -> list[tuple[str, dict[str, float], str, str]]:
    points = _historical_points()
    start_index = _month_index("1995-01")
    end_index = _month_index("2008-12")
    snapshots: list[tuple[str, dict[str, float], str, str]] = []
    point_index = 0

    for month_index in range(start_index, end_index + 1):
        while (
            point_index + 1 < len(points)
            and _month_index(points[point_index + 1][0]) <= month_index
        ):
            point_index += 1
        left_month, left_values, left_source = points[point_index]
        if point_index + 1 < len(points):
            right_month, right_values, right_source = points[point_index + 1]
        else:
            right_month, right_values, right_source = left_month, left_values, left_source

        left_index = _month_index(left_month)
        right_index = _month_index(right_month)
        alpha = 0.0 if right_index == left_index else (month_index - left_index) / (right_index - left_index)
        values = {
            browser: left_values.get(browser, 0.0)
            + (right_values.get(browser, 0.0) - left_values.get(browser, 0.0)) * alpha
            for browser in set(left_values) | set(right_values)
        }
        source = left_source if left_source == right_source else f"{left_source} / {right_source}"
        snapshots.append(
            (
                _month_from_index(month_index),
                values,
                source,
                HISTORICAL_REFERENCE_PAGE,
            )
        )
    return snapshots


def _harmonize_2008_transition(
    historical: list[tuple[str, dict[str, float], str, str]],
    first_statcounter: tuple[str, dict[str, float], str, str],
) -> list[tuple[str, dict[str, float], str, str]]:
    _, official_values, _, _ = first_statcounter
    common_names = {"IE", "Firefox", "Safari", "Opera", "Chrome"}
    target = {name: official_values.get(name, 0.0) for name in common_names}
    target["Netscape"] = 0.0
    target["Mozilla"] = 0.0
    target["Mosaic"] = 0.0
    target["Other"] = max(0.0, 100.0 - sum(target.values()))

    harmonized: list[tuple[str, dict[str, float], str, str]] = []
    for month, values, source, source_url in historical:
        if not month.startswith("2008-"):
            harmonized.append((month, values, source, source_url))
            continue

        month_number = int(month[5:7])
        alpha = month_number / 12.0
        month_target = dict(target)
        if month < "2008-09":
            month_target["Other"] += month_target["Chrome"]
            month_target["Chrome"] = 0.0

        blended = {
            browser: values.get(browser, 0.0) * (1.0 - alpha)
            + month_target.get(browser, 0.0) * alpha
            for browser in set(values) | set(month_target)
        }
        blended["Other"] += 100.0 - sum(blended.values())
        harmonized.append((month, blended, source, source_url))
    return harmonized


def _statcounter_snapshots(raw_csv: Path) -> list[tuple[str, dict[str, float], str, str]]:
    browser_names, wide_rows = _load_wide_rows(raw_csv)
    return [
        (
            wide_row["Date"].strip(),
            {
                browser: float(wide_row.get(browser, "0") or 0)
                for browser in browser_names
            },
            "Statcounter Global Stats",
            SOURCE_PAGE,
        )
        for wide_row in wide_rows
    ]


def build_rows(raw_csv: Path) -> list[dict[str, str]]:
    statcounter = _statcounter_snapshots(raw_csv)
    historical = _harmonize_2008_transition(
        _historical_monthly_snapshots(),
        statcounter[0],
    )
    snapshots = historical + statcounter
    output: list[dict[str, str]] = []
    previous_shares: dict[str, float] = {}

    for month, shares, source, source_url in snapshots:
        ranked = sorted(shares.items(), key=lambda item: (-item[1], item[0]))
        leader_name, leader_share = ranked[0]
        positive_changes = sorted(
            (
                (share - previous_shares.get(browser, share), browser)
                for browser, share in shares.items()
            ),
            reverse=True,
        )
        if month < "2009-01":
            summary = (
                f"Leader: {leader_name} {leader_share:.1f}%"
                "|The browser wars"
            )
        elif month == "2009-01":
            summary = (
                f"Leader: {leader_name} {leader_share:.1f}%"
                "|Statcounter monthly series begins"
            )
        elif previous_shares and positive_changes and positive_changes[0][0] > 0:
            gain, gain_name = positive_changes[0]
            summary = (
                f"Leader: {leader_name} {leader_share:.1f}%"
                f"|Biggest monthly gain: {gain_name} +{gain:.1f} pp"
            )
        else:
            summary = f"Leader: {leader_name} {leader_share:.1f}%|No positive monthly change"

        for browser, share in ranked:
            previous = previous_shares.get(browser)
            output.append(
                {
                    "ranking_date": f"{month}-01",
                    "browser_name": browser,
                    "browser_key": _browser_key(browser),
                    "market_share": f"{share:.2f}",
                    "monthly_change": "" if previous is None else f"{share - previous:.2f}",
                    "season_summary": summary,
                    "data_source": source,
                    "source_url": source_url,
                }
            )
        previous_shares = shares
    return output


def write_csv(rows: list[dict[str, str]], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "ranking_date",
                "browser_name",
                "browser_key",
                "market_share",
                "monthly_change",
                "season_summary",
                "data_source",
                "source_url",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def download_logos(logos_dir: Path, refresh: bool) -> list[Path]:
    logos_dir.mkdir(parents=True, exist_ok=True)
    downloaded: list[Path] = []
    for key, url in PNG_LOGO_URLS.items():
        output_path = logos_dir / f"{key}.png"
        if output_path.exists() and not refresh:
            continue
        try:
            _download(url, output_path)
            downloaded.append(output_path)
        except Exception as error:
            print(f"[scraper] logo download skipped for {key}: {error}")

    try:
        import cairosvg
    except (ImportError, OSError):
        print("[scraper] Cairo SVG support unavailable; SVG browser logos skipped")
        return downloaded

    for key, url in SVG_LOGO_URLS.items():
        output_path = logos_dir / f"{key}.png"
        if output_path.exists() and not refresh:
            continue
        svg_path = logos_dir / f"{key}.svg.download"
        try:
            _download(url, svg_path)
            cairosvg.svg2png(
                url=str(svg_path),
                write_to=str(output_path),
                output_width=128,
                output_height=128,
            )
            svg_path.unlink(missing_ok=True)
            downloaded.append(output_path)
        except Exception as error:
            svg_path.unlink(missing_ok=True)
            print(f"[scraper] logo download skipped for {key}: {error}")
    return downloaded


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the official worldwide monthly browser market share timeseries."
    )
    parser.add_argument("--raw-csv", type=Path, default=DEFAULT_RAW_CSV)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--logos-dir", type=Path, default=DEFAULT_LOGOS_DIR)
    parser.add_argument("--start-month", default=START_MONTH)
    parser.add_argument("--end-month", default="")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--skip-logos", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    end_month = args.end_month or _latest_complete_month()
    if args.start_month > end_month:
        raise ValueError("--start-month must be earlier than or equal to --end-month.")

    if args.refresh or not args.raw_csv.exists():
        _download(_chart_url(args.start_month, end_month), args.raw_csv)
    rows = build_rows(args.raw_csv)
    output = write_csv(rows, args.output)
    downloaded = [] if args.skip_logos else download_logos(args.logos_dir, args.refresh)

    print(f"[scraper] Browser market share CSV generated -> {output}")
    print(
        f"[scraper] {len(rows)} rows, 1995-01 to {end_month}, "
        f"{len(downloaded)} logos downloaded"
    )


if __name__ == "__main__":
    main()
