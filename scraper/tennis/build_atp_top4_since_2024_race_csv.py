from __future__ import annotations

import argparse
import csv
import json
import re
import urllib.parse
import urllib.request
from datetime import date, datetime
from html import unescape
from html.parser import HTMLParser
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ATP_API_BASE = "https://atp-rankings-data-visualization.onrender.com/api"
USER_AGENT = "youtube-videos-local/1.0"

DEFAULT_START_DATE = "2024-01-01"
DEFAULT_END_DATE = date.today().isoformat()
DEFAULT_RANKINGS_OUTPUT = PROJECT_ROOT / "data" / "processed" / "tennis" / "atp_top4_points_since_2024.csv"
DEFAULT_TITLES_OUTPUT = PROJECT_ROOT / "data" / "processed" / "tennis" / "atp_top4_titles_since_2024.csv"
DEFAULT_LOGOS_DIR = PROJECT_ROOT / "data" / "raw" / "tennis_tournament_logos"
LOCAL_SLAM_LOGOS_DIR = PROJECT_ROOT / "data" / "raw" / "tennis_logos"

COUNTRY_FALLBACK = {
    "Alexander Zverev": "GER",
    "Andrey Rublev": "RUS",
    "Carlos Alcaraz": "ESP",
    "Daniil Medvedev": "RUS",
    "Jannik Sinner": "ITA",
    "Novak Djokovic": "SRB",
    "Taylor Fritz": "USA",
    "Casper Ruud": "NOR",
    "Jack Draper": "GBR",
    "Lorenzo Musetti": "ITA",
    "Holger Rune": "DEN",
    "Alex de Minaur": "AUS",
}

SLAM_LOGO_FILES = {
    "Australian Open": "australian_open.png",
    "French Open": "roland_garros.jpg",
    "Roland Garros": "roland_garros.jpg",
    "Wimbledon": "wimbledon.png",
    "US Open": "us_open.png",
}

COUNTRY_NAMES = {
    "Argentina", "Australia", "Austria", "Belgium", "Brazil", "Bulgaria", "Canada", "Chile", "China",
    "Croatia", "Czech Republic", "Denmark", "Finland", "France", "Germany", "Great Britain", "Greece",
    "Hungary", "Italy", "Japan", "Kazakhstan", "Netherlands", "Norway", "Poland", "Portugal", "Russia",
    "Serbia", "Spain", "Sweden", "Switzerland", "United Kingdom", "United States",
}


class TableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.in_cell = False
        self.current_text: list[str] = []
        self.current_links: list[tuple[str, str]] = []
        self.current_cells: list[dict[str, object]] = []
        self.rows: list[list[dict[str, object]]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_dict = {key: value or "" for key, value in attrs}
        if tag == "tr":
            self.current_cells = []
        elif tag in {"td", "th"}:
            self.in_cell = True
            self.current_text = []
            self.current_links = []
        elif self.in_cell and tag == "br":
            self.current_text.append("\n")
        elif self.in_cell and tag == "a":
            self.current_links.append((attrs_dict.get("href", ""), attrs_dict.get("title", "")))

    def handle_data(self, data: str) -> None:
        if self.in_cell:
            self.current_text.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag in {"td", "th"} and self.in_cell:
            text = unescape("".join(self.current_text))
            text = re.sub(r"[ \t\r\f\v]+", " ", text)
            text = re.sub(r"\n+", "\n", text).strip()
            self.current_cells.append({"text": text, "links": self.current_links[:]})
            self.in_cell = False
        elif tag == "tr" and self.current_cells:
            self.rows.append(self.current_cells[:])


def _fetch_json(url: str) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def _fetch_text(url: str) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read().decode("utf-8", "replace")


def _slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _to_int_points(value: str) -> int:
    return int(str(value).replace(",", "").strip())


def _parse_week_date(text: str, year: int, section_month: str = "") -> str | None:
    match = re.search(r"(\d{1,2})\s+([A-Z][a-z]+)", text)
    if not match:
        return None
    parsed_year = year
    if section_month == "January" and match.group(2) == "Dec":
        parsed_year = year - 1
    parsed = None
    for pattern in ("%d %B %Y", "%d %b %Y"):
        try:
            parsed = datetime.strptime(f"{match.group(1)} {match.group(2)} {parsed_year}", pattern).date()
            break
        except ValueError:
            continue
    if parsed is None:
        return None
    return parsed.isoformat()


def build_rankings(start_date: str, end_date: str, top_limit: int) -> list[dict[str, str | int]]:
    weeks_payload = _fetch_json(f"{ATP_API_BASE}/weeks")
    weeks = weeks_payload.get("weeks", []) if isinstance(weeks_payload, dict) else weeks_payload
    week_dates = sorted(str(item.get("date", item) if isinstance(item, dict) else item) for item in weeks)
    week_dates = [item for item in week_dates if item >= start_date and item <= end_date]

    rows: list[dict[str, str | int]] = []
    for ranking_date in week_dates:
        weekly = _fetch_json(f"{ATP_API_BASE}/week/{ranking_date}")
        for rank_row in weekly.get("rankings", [])[:top_limit]:
            player = str(rank_row["name"]).strip()
            rows.append(
                {
                    "ranking_date": ranking_date,
                    "player_name": player,
                    "country_code": COUNTRY_FALLBACK.get(player, ""),
                    "points": _to_int_points(str(rank_row["points"])),
                    "rank": int(rank_row["rank"]),
                }
            )
    return rows


def _first_player_link(cell: dict[str, object]) -> str:
    for _href, title in cell.get("links", []):
        title = str(title).strip()
        if title in COUNTRY_NAMES:
            continue
        if title and not any(blocked in title for blocked in ["Open", "Classic", "Championship", "Cup", "Masters", "Finals"]):
            return title
    return ""


def _clean_player_name(name: str) -> str:
    return re.sub(r"\s+\([^)]*\)$", "", name).strip()


def parse_atp_tour_titles(year: int) -> list[dict[str, str]]:
    html = _fetch_text(f"https://en.wikipedia.org/wiki/{year}_ATP_Tour")
    rows: list[dict[str, str]] = []
    month_positions = [(m.start(), m.group(1)) for m in re.finditer(r'<h3 id="([A-Z][a-z]+)"', html)]
    for idx, (start, _month) in enumerate(month_positions):
        end = month_positions[idx + 1][0] if idx + 1 < len(month_positions) else html.find('<h2 id="Statistical_information"', start)
        if end < 0:
            end = len(html)
        parser = TableParser()
        parser.feed(html[start:end])
        current_week = ""
        for table_row in parser.rows:
            if not table_row or table_row[0]["text"] == "Week":
                continue

            first_text = str(table_row[0]["text"])
            if _parse_week_date(first_text, year, _month):
                current_week = _parse_week_date(first_text, year, _month) or current_week
                cells = table_row[1:]
            else:
                cells = table_row

            if len(cells) < 2:
                continue
            tournament_cell = cells[0]
            champion_cell = cells[1]
            tournament_text = str(tournament_cell["text"])
            if "Singles" not in tournament_text:
                continue

            tournament_name = tournament_text.split("\n", 1)[0].strip()
            if not tournament_name or not current_week:
                continue
            champion = _clean_player_name(_first_player_link(champion_cell))
            if not champion:
                continue
            tournament_page = ""
            for href, title in tournament_cell.get("links", []):
                if href.startswith("/wiki/") and title:
                    tournament_page = title
                    break
            rows.append(
                {
                    "event_date": current_week,
                    "player_name": champion,
                    "tournament": tournament_name,
                    "tournament_page": tournament_page,
                    "source": f"https://en.wikipedia.org/wiki/{year}_ATP_Tour",
                }
            )
    return rows


def _copy_local_slam_logo(tournament: str, out_path: Path) -> bool:
    filename = SLAM_LOGO_FILES.get(tournament)
    if not filename:
        return False
    source = LOCAL_SLAM_LOGOS_DIR / filename
    if not source.exists():
        return False
    out_path.write_bytes(source.read_bytes())
    return True


def _wikipedia_thumbnail(title: str) -> str:
    if not title:
        return ""
    params = urllib.parse.urlencode(
        {
            "action": "query",
            "titles": title,
            "prop": "pageimages",
            "format": "json",
            "pithumbsize": "500",
        }
    )
    try:
        payload = _fetch_json(f"https://en.wikipedia.org/w/api.php?{params}")
    except Exception:
        return ""
    for page in payload.get("query", {}).get("pages", {}).values():
        thumb = page.get("thumbnail", {}).get("source", "")
        if thumb:
            return str(thumb)
    return ""


def _wikipedia_summary_thumbnail(title: str) -> str:
    if not title:
        return ""
    safe_title = urllib.parse.quote(title.replace(" ", "_"), safe="")
    try:
        payload = _fetch_json(f"https://en.wikipedia.org/api/rest_v1/page/summary/{safe_title}")
    except Exception:
        return ""
    return str(payload.get("thumbnail", {}).get("source", "") or "")


def _download_file(url: str, path: Path) -> bool:
    if not url:
        return False
    parsed = urllib.parse.urlsplit(url)
    safe_path = urllib.parse.quote(urllib.parse.unquote(parsed.path), safe="/%")
    safe_url = urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, safe_path, parsed.query, parsed.fragment))
    request = urllib.request.Request(safe_url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            path.write_bytes(response.read())
        return True
    except Exception:
        return False


def _load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    for path in [
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf",
    ]:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _create_fallback_logo(tournament: str, out_path: Path) -> None:
    words = [part for part in re.split(r"[^A-Za-z0-9]+", tournament) if part]
    initials = "".join(word[0] for word in words[:3]).upper() or "ATP"
    img = Image.new("RGBA", (360, 360), (31, 68, 120, 255))
    draw = ImageDraw.Draw(img)
    for y in range(360):
        shade = int(35 + y * 0.18)
        draw.line((0, y, 360, y), fill=(25, shade, 150, 255))
    draw.ellipse((18, 18, 342, 342), outline=(255, 255, 255, 110), width=5)
    draw.text((180, 166), initials, font=_load_font(74, True), fill=(255, 255, 255, 255), anchor="mm")
    draw.text((180, 232), "ATP", font=_load_font(38, True), fill=(245, 210, 88, 255), anchor="mm")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


def ensure_logo(tournament: str, tournament_page: str, logos_dir: Path) -> str:
    logos_dir.mkdir(parents=True, exist_ok=True)
    logo_path = logos_dir / f"{_slugify(tournament)}.png"
    if logo_path.exists():
        return str(logo_path.relative_to(PROJECT_ROOT))
    if _copy_local_slam_logo(tournament, logo_path):
        return str(logo_path.relative_to(PROJECT_ROOT))
    thumb = (
        _wikipedia_thumbnail(tournament_page)
        or _wikipedia_summary_thumbnail(tournament_page)
        or _wikipedia_thumbnail(tournament)
        or _wikipedia_summary_thumbnail(tournament)
    )
    if _download_file(thumb, logo_path):
        return str(logo_path.relative_to(PROJECT_ROOT))
    _create_fallback_logo(tournament, logo_path)
    return str(logo_path.relative_to(PROJECT_ROOT))


def write_csv(path: Path, rows: list[dict[str, str | int]], fieldnames: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def run(args: argparse.Namespace) -> tuple[Path, Path, int, int]:
    ranking_rows = build_rankings(args.start_date, args.end_date, args.top_limit)
    top4_players = {
        str(row["player_name"])
        for row in ranking_rows
        if int(row["rank"]) <= 4
    }
    raw_titles: list[dict[str, str]] = []
    start_year = int(args.start_date[:4])
    end_year = int(args.end_date[:4])
    for year in range(start_year, end_year + 1):
        raw_titles.extend(parse_atp_tour_titles(year))

    title_rows: list[dict[str, str]] = []
    for item in raw_titles:
        if item["event_date"] < args.start_date or item["event_date"] > args.end_date:
            continue
        if item["player_name"] not in top4_players:
            continue
        logo_file = ensure_logo(item["tournament"], item["tournament_page"], args.logos_dir)
        title_rows.append({**item, "logo_file": logo_file})

    title_rows.sort(key=lambda row: (row["event_date"], row["player_name"], row["tournament"]))
    write_csv(args.rankings_output, ranking_rows, ["ranking_date", "player_name", "country_code", "points", "rank"])
    write_csv(args.titles_output, title_rows, ["event_date", "player_name", "tournament", "tournament_page", "logo_file", "source"])
    return args.rankings_output, args.titles_output, len(ranking_rows), len(title_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build ATP top 4 race data since 2024 with title logos.")
    parser.add_argument("--start-date", default=DEFAULT_START_DATE)
    parser.add_argument("--end-date", default=DEFAULT_END_DATE)
    parser.add_argument("--top-limit", type=int, default=10)
    parser.add_argument("--rankings-output", type=Path, default=DEFAULT_RANKINGS_OUTPUT)
    parser.add_argument("--titles-output", type=Path, default=DEFAULT_TITLES_OUTPUT)
    parser.add_argument("--logos-dir", type=Path, default=DEFAULT_LOGOS_DIR)
    return parser.parse_args()


if __name__ == "__main__":
    rankings_path, titles_path, ranking_count, title_count = run(parse_args())
    print(f"[scraper] ATP rankings CSV -> {rankings_path} ({ranking_count} rows)")
    print(f"[scraper] ATP title events CSV -> {titles_path} ({title_count} rows)")
