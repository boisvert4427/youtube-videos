from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request
from pathlib import Path

import requests


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
LOGOS_DIR = PROJECT_ROOT / "data" / "raw" / "company_logos"


COMPANY_LABELS = {
    "000660.KS": "SK hynix",
    "005930.KS": "Samsung Electronics",
    "0700.HK": "Tencent",
    "0939.HK": "China Construction Bank",
    "1288.HK": "Agricultural Bank of China",
    "1398.HK": "ICBC",
    "2222.SR": "Saudi Aramco",
    "AAPL": "Apple Inc.",
    "ABBV": "AbbVie",
    "AMD": "Advanced Micro Devices",
    "AMZN": "Amazon",
    "ASML": "ASML",
    "AVGO": "Broadcom",
    "AZN": "AstraZeneca",
    "BABA": "Alibaba Group",
    "BAC": "Bank of America",
    "BRK-B": "Berkshire Hathaway",
    "CAT": "Caterpillar",
    "COST": "Costco Wholesale",
    "CSCO": "Cisco",
    "CVX": "Chevron Corporation",
    "GE": "General Electric",
    "GOOGL": "Alphabet Inc.",
    "HD": "The Home Depot",
    "HSBC": "HSBC",
    "JNJ": "Johnson & Johnson",
    "JPM": "JPMorgan Chase",
    "KO": "The Coca-Cola Company",
    "LLY": "Eli Lilly and Company",
    "LRCX": "Lam Research",
    "MA": "Mastercard",
    "MC.PA": "LVMH",
    "META": "Meta Platforms",
    "MRK": "Merck & Co.",
    "MSFT": "Microsoft",
    "MU": "Micron Technology",
    "NFLX": "Netflix",
    "NOVN.SW": "Novartis",
    "NVDA": "NVIDIA",
    "ORCL": "Oracle Corporation",
    "PG": "Procter & Gamble",
    "PLTR": "Palantir Technologies",
    "ROG.SW": "Roche Holding AG",
    "TM": "Toyota Motor Corporation",
    "TSLA": "Tesla, Inc.",
    "TSM": "Taiwan Semiconductor Manufacturing Company",
    "V": "Visa Inc.",
    "WMT": "Walmart",
    "XOM": "Exxon Mobil",
}

COMPANY_PAGE_SLUGS = {
    "000660.KS": "sk-hynix",
    "005930.KS": "samsung",
    "0700.HK": "tencent",
    "0939.HK": "china-construction-bank",
    "1288.HK": "agricultural-bank-of-china",
    "1398.HK": "icbc",
    "2222.SR": "saudi-aramco",
    "AAPL": "apple",
    "ABBV": "abbvie",
    "AMD": "amd",
    "AMZN": "amazon",
    "ASML": "asml",
    "AVGO": "broadcom",
    "AZN": "astrazeneca",
    "BABA": "alibaba",
    "BAC": "bank-of-america",
    "BRK-B": "berkshire-hathaway",
    "CAT": "caterpillar",
    "COST": "costco",
    "CSCO": "cisco",
    "CVX": "chevron",
    "GE": "general-electric",
    "GOOGL": "alphabet-google",
    "HD": "home-depot",
    "HSBC": "hsbc",
    "JNJ": "johnson-and-johnson",
    "JPM": "jp-morgan-chase",
    "KO": "coca-cola",
    "LLY": "eli-lilly",
    "LRCX": "lam-research",
    "MA": "mastercard",
    "MC.PA": "lvmh",
    "META": "meta-platforms",
    "MRK": "merck",
    "MSFT": "microsoft",
    "MU": "micron-technology",
    "NFLX": "netflix",
    "NOVN.SW": "novartis",
    "NVDA": "nvidia",
    "ORCL": "oracle",
    "PG": "procter-and-gamble",
    "PLTR": "palantir",
    "ROG.SW": "roche",
    "TM": "toyota",
    "TSLA": "tesla",
    "TSM": "tsmc",
    "V": "visa",
    "WMT": "walmart",
    "XOM": "exxon-mobil",
}


def _slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")


def _fetch_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "youtube-videos-local/1.0"})
    with urllib.request.urlopen(req, timeout=60) as response:
        return json.load(response)


def _download_bytes(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "youtube-videos-local/1.0"})
    with urllib.request.urlopen(req, timeout=60) as response:
        return response.read()


def _companiesmarketcap_logo_url(ticker: str) -> str:
    slug = COMPANY_PAGE_SLUGS.get(ticker.strip().upper())
    if not slug:
        return ""
    urls = [
        f"https://companiesmarketcap.com/{slug}/marketcap/",
        f"https://companiesmarketcap.com/{slug}/shares-outstanding/",
    ]
    for url in urls:
        try:
            html = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=30).text
        except Exception:
            continue
        match = re.search(r'class="company-profile-logo"[^>]+src="([^"]+)"', html)
        if match:
            src = match.group(1)
            if src.startswith("/"):
                src = "https://companiesmarketcap.com" + src
            return src
    return ""


def _wikidata_logo_url(label: str) -> str:
    query = f"""
SELECT ?logo WHERE {{
  ?item rdfs:label "{label}"@en .
  OPTIONAL {{ ?item wdt:P154 ?logo . }}
}}
LIMIT 5
"""
    url = "https://query.wikidata.org/sparql?" + urllib.parse.urlencode({"query": query, "format": "json"})
    try:
        payload = _fetch_json(url)
    except Exception:
        return ""
    bindings = payload.get("results", {}).get("bindings", [])
    for binding in bindings:
        logo_uri = binding.get("logo", {}).get("value", "")
        if logo_uri:
            filename = logo_uri.rsplit("/", 1)[-1]
            return "https://commons.wikimedia.org/wiki/Special:FilePath/" + urllib.parse.quote(filename) + "?width=700"
    return ""


def _wikipedia_logo_url(title: str) -> str:
    summary_url = "https://en.wikipedia.org/api/rest_v1/page/summary/" + urllib.parse.quote(title)
    try:
        summary = _fetch_json(summary_url)
        thumb = summary.get("thumbnail", {}).get("source", "")
        if thumb:
            return str(thumb)
    except Exception:
        pass

    params = urllib.parse.urlencode(
        {
            "action": "query",
            "titles": title,
            "prop": "pageimages",
            "format": "json",
            "pithumbsize": "700",
        }
    )
    try:
        payload = _fetch_json(f"https://en.wikipedia.org/w/api.php?{params}")
    except Exception:
        return ""
    pages = payload.get("query", {}).get("pages", {})
    for page in pages.values():
        thumb = page.get("thumbnail", {}).get("source", "")
        if thumb:
            return str(thumb)
    return ""


def run() -> tuple[int, int]:
    LOGOS_DIR.mkdir(parents=True, exist_ok=True)
    saved = 0
    total = len(COMPANY_LABELS)
    for ticker, label in COMPANY_LABELS.items():
        out_path = LOGOS_DIR / f"{_slugify(ticker)}.png"
        if out_path.exists() and out_path.stat().st_size >= 1200:
            saved += 1
            continue
        url = _companiesmarketcap_logo_url(ticker) or _wikidata_logo_url(label) or _wikipedia_logo_url(label)
        if not url:
            continue
        try:
            out_path.write_bytes(_download_bytes(url))
            saved += 1
        except Exception:
            continue
    return saved, total


if __name__ == "__main__":
    ok, total = run()
    print(f"[scraper] company logos saved: {ok}/{total} -> {LOGOS_DIR}")
