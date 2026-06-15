from __future__ import annotations

import argparse
import csv
import re
import shutil
import urllib.parse
import urllib.request
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw" / "technology" / "mobile_vendor_market_share"
DEFAULT_RAW_CSV = DEFAULT_RAW_DIR / "statcounter_mobile_vendor_market_share_monthly.csv"
DEFAULT_LOGOS_DIR = DEFAULT_RAW_DIR / "logos"
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "technology"
    / "mobile_vendor_market_share"
    / "mobile_vendor_market_share_2009_2025.csv"
)

SOURCE_PAGE = "https://gs.statcounter.com/vendor-market-share/mobile/worldwide/"
CHART_ENDPOINT = "https://gs.statcounter.com/chart.php"
START_MONTH = "2009-01"
END_MONTH = "2025-12"
USER_AGENT = "Mozilla/5.0 (compatible; Codex Mobile Vendor Market Share Builder)"

LOGO_DOMAINS = {
    "acer": "acer.com",
    "alcatel": "alcatelmobile.com",
    "apple": "apple.com",
    "asus": "asus.com",
    "blackberry": "blackberry.com",
    "google": "google.com",
    "honor": "honor.com",
    "htc": "htc.com",
    "huawei": "huawei.com",
    "lenovo": "lenovo.com",
    "lg": "lg.com",
    "micromax": "micromaxinfo.com",
    "motorola": "motorola.com",
    "nintendo": "nintendo.com",
    "nokia": "nokia.com",
    "oneplus": "oneplus.com",
    "oppo": "oppo.com",
    "realme": "realme.com",
    "rim": "blackberry.com",
    "samsung": "samsung.com",
    "sony": "sony.com",
    "tecno": "tecno-mobile.com",
    "t_mobile": "t-mobile.com",
    "vivo": "vivo.com",
    "xiaomi": "mi.com",
    "zte": "zte.com.cn",
}


def _download(url: str, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".download")
    request = urllib.request.Request(
        url,
        headers={"User-Agent": USER_AGENT, "Referer": SOURCE_PAGE},
    )
    with urllib.request.urlopen(request, timeout=120) as response, temporary_path.open("wb") as output:
        shutil.copyfileobj(response, output)
    temporary_path.replace(output_path)
    return output_path


def _chart_url(start_month: str, end_month: str) -> str:
    query = {
        "device": "Mobile",
        "device_hidden": "mobile",
        "statType_hidden": "vendor",
        "region_hidden": "ww",
        "granularity": "monthly",
        "statType": "Mobile Vendor",
        "region": "Worldwide",
        "fromInt": start_month.replace("-", ""),
        "toInt": end_month.replace("-", ""),
        "fromMonthYear": start_month,
        "toMonthYear": end_month,
        "csv": "1",
    }
    return f"{CHART_ENDPOINT}?{urllib.parse.urlencode(query)}"


def _vendor_key(name: str) -> str:
    aliases = {
        "RIM": "rim",
        "Sony Ericsson": "sony",
    }
    if name in aliases:
        return aliases[name]
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def build_rows(raw_csv: Path) -> list[dict[str, str]]:
    with raw_csv.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        wide_rows = list(reader)
        vendor_names = [name for name in (reader.fieldnames or []) if name != "Date"]
    if not wide_rows or not vendor_names:
        raise RuntimeError("The StatCounter mobile vendor CSV is empty.")

    output: list[dict[str, str]] = []
    previous_shares: dict[str, float] = {}
    for wide_row in wide_rows:
        month = wide_row["Date"].strip()
        shares = {
            vendor: max(0.0, float(wide_row.get(vendor, "0") or 0))
            for vendor in vendor_names
        }
        ranked = sorted(shares.items(), key=lambda item: (-item[1], item[0]))
        leader_name, leader_share = ranked[0]
        gains = sorted(
            (
                (share - previous_shares.get(vendor, share), vendor)
                for vendor, share in shares.items()
            ),
            reverse=True,
        )
        if previous_shares and gains and gains[0][0] > 0:
            gain, gain_name = gains[0]
            summary = (
                f"Leader: {leader_name} {leader_share:.1f}%"
                f"|Biggest monthly gain: {gain_name} +{gain:.1f} pp"
            )
        else:
            summary = f"Leader: {leader_name} {leader_share:.1f}%|Worldwide mobile vendors"

        for vendor, share in ranked:
            previous = previous_shares.get(vendor)
            output.append(
                {
                    "ranking_date": f"{month}-01",
                    "browser_name": vendor,
                    "browser_key": _vendor_key(vendor),
                    "market_share": f"{share:.2f}",
                    "monthly_change": "" if previous is None else f"{share - previous:.2f}",
                    "season_summary": summary,
                    "data_source": "StatCounter Global Stats",
                    "source_url": SOURCE_PAGE,
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
    for key, domain in LOGO_DOMAINS.items():
        output_path = logos_dir / f"{key}.png"
        if output_path.exists() and not refresh:
            continue
        try:
            url = f"https://www.google.com/s2/favicons?domain={domain}&sz=128"
            _download(url, output_path)
            downloaded.append(output_path)
        except Exception as error:
            print(f"[scraper] logo download skipped for {key}: {error}")
    return downloaded


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build worldwide mobile vendor market share data.")
    parser.add_argument("--raw-csv", type=Path, default=DEFAULT_RAW_CSV)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--logos-dir", type=Path, default=DEFAULT_LOGOS_DIR)
    parser.add_argument("--start-month", default=START_MONTH)
    parser.add_argument("--end-month", default=END_MONTH)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--skip-logos", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.start_month > args.end_month:
        raise ValueError("--start-month must be earlier than or equal to --end-month.")
    if args.refresh or not args.raw_csv.exists():
        _download(_chart_url(args.start_month, args.end_month), args.raw_csv)
    rows = build_rows(args.raw_csv)
    output = write_csv(rows, args.output)
    downloaded = [] if args.skip_logos else download_logos(args.logos_dir, args.refresh)
    dates = sorted({row["ranking_date"] for row in rows})
    print(f"[scraper] Mobile vendor market share CSV generated -> {output}")
    print(
        f"[scraper] {len(rows)} rows, {dates[0]} to {dates[-1]}, "
        f"{len(downloaded)} logos downloaded"
    )


if __name__ == "__main__":
    main()
