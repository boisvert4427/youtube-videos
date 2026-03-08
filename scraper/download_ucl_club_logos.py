from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOGOS_DIR = PROJECT_ROOT / "data" / "raw" / "club_logos"

CLUB_TITLES = {
    "Real Madrid": "Real Madrid CF",
    "Benfica": "S.L. Benfica",
    "AC Milan": "AC Milan",
    "Inter Milan": "Inter Milan",
    "Celtic": "Celtic F.C.",
    "Manchester United": "Manchester United F.C.",
    "Feyenoord": "Feyenoord",
    "Ajax": "AFC Ajax",
    "Bayern Munich": "FC Bayern Munich",
    "Liverpool": "Liverpool F.C.",
    "Nottingham Forest": "Nottingham Forest F.C.",
    "Aston Villa": "Aston Villa F.C.",
    "Hamburger SV": "Hamburger SV",
    "Juventus": "Juventus FC",
    "Steaua Bucuresti": "FCSB",
}


def _slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")


def _fetch_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "youtube-videos-local/1.0"})
    with urllib.request.urlopen(req, timeout=60) as response:
        return json.load(response)


def _find_logo_url(wiki_title: str) -> str:
    params = urllib.parse.urlencode(
        {
            "action": "query",
            "titles": wiki_title,
            "prop": "pageimages",
            "format": "json",
            "pithumbsize": "700",
        }
    )
    payload = _fetch_json(f"https://en.wikipedia.org/w/api.php?{params}")
    pages = payload.get("query", {}).get("pages", {})
    for page in pages.values():
        url = page.get("thumbnail", {}).get("source", "")
        if url:
            return str(url)
    return ""


def _wikidata_logo_url(label: str) -> str:
    query = f"""
SELECT ?logo WHERE {{
  ?club rdfs:label "{label}"@en .
  ?club wdt:P31/wdt:P279* wd:Q476028 .
  OPTIONAL {{ ?club wdt:P154 ?logo . }}
}}
LIMIT 1
"""
    url = "https://query.wikidata.org/sparql?" + urllib.parse.urlencode({"query": query, "format": "json"})
    payload = _fetch_json(url)
    bindings = payload.get("results", {}).get("bindings", [])
    if not bindings:
        return ""
    logo_uri = bindings[0].get("logo", {}).get("value", "")
    if not logo_uri:
        return ""
    # Convert commons file URI to direct file path endpoint.
    filename = logo_uri.rsplit("/", 1)[-1]
    return "https://commons.wikimedia.org/wiki/Special:FilePath/" + urllib.parse.quote(filename)


def _download(url: str, out_path: Path) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "youtube-videos-local/1.0"})
    with urllib.request.urlopen(req, timeout=60) as response:
        out_path.write_bytes(response.read())


def run() -> tuple[int, int]:
    LOGOS_DIR.mkdir(parents=True, exist_ok=True)
    saved = 0
    total = len(CLUB_TITLES)
    for club_name, wiki_title in CLUB_TITLES.items():
        slug = _slugify(club_name)
        out_path = LOGOS_DIR / f"{slug}.png"
        if out_path.exists():
            saved += 1
            continue
        logo_url = _wikidata_logo_url(wiki_title) or _find_logo_url(wiki_title)
        if not logo_url:
            continue
        try:
            _download(logo_url, out_path)
            saved += 1
        except Exception:
            continue
    return saved, total


if __name__ == "__main__":
    ok, total = run()
    print(f"[scraper] club logos saved: {ok}/{total} -> {LOGOS_DIR}")
