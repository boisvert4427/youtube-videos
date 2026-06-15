from __future__ import annotations

import argparse
import csv
import hashlib
import io
import re
import shutil
import unicodedata
from collections import defaultdict
from pathlib import Path

import requests
import fitz
from bs4 import BeautifulSoup


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "finance"
    / "forbes_billionaires"
    / "forbes_billionaires_1997_2024.csv"
)
DEFAULT_FLAGS_DIR = PROJECT_ROOT / "data" / "raw" / "flags"

SOURCE_URL = "https://en.wikipedia.org/w/index.php?title=The_World%27s_Billionaires&action=raw"
GAPMINDER_BASE_URL = "https://raw.githubusercontent.com/open-numbers/ddf--gapminder--billionaires/master/"
GAPMINDER_PERSONS_URL = GAPMINDER_BASE_URL + "ddf--entities--person.csv"
GAPMINDER_WORTH_URL = GAPMINDER_BASE_URL + "ddf--datapoints--worth--by--person--time.csv"
USER_AGENT = "Mozilla/5.0 (compatible; Codex Forbes Billionaires Builder)"

START_YEAR = 1997
END_YEAR = 2024
TOP_N = 12

COUNTRY_ALIASES = {
    "czech republic": "czechia",
    "eswatini swaziland": "eswatini",
    "macao": "macau",
    "st kitts and nevis": "saint kitts and nevis",
    "swaziland": "eswatini",
}

PERSON_NAME_ALIASES = {
    "al waleed bin talal": "Al-Waleed bin Talal",
    "alwaleed bin talal": "Al-Waleed bin Talal",
    "alwaleed bin talal alsaud": "Al-Waleed bin Talal",
    "alice l walton": "Alice Walton",
    "allen paul": "Paul Allen",
    "allen paul gardner": "Paul Allen",
    "albrecht theo karl": "Karl and Theo Albrecht",
    "albrecht theo karl and": "Karl and Theo Albrecht",
    "alsaud prince alwaleed bin talal": "Al-Waleed bin Talal",
    "bill gates": "Bill Gates",
    "gates william h iii": "Bill Gates",
    "gates william henry iii": "Bill Gates",
    "william gates iii": "Bill Gates",
    "william henry gates iii": "Bill Gates",
    "carlos slim": "Carlos Slim Helu",
    "carlos slim helu": "Carlos Slim Helu",
    "buffett warren edward": "Warren Buffett",
    "buffett warren e": "Warren Buffett",
    "buffett warren": "Warren Buffett",
    "cargill": "Cargill family",
    "donald ands i newhouse": "Donald Newhouse & S.I. Newhouse",
    "ellison lawrence joseph": "Larry Ellison",
    "forrest edward mars sr and": "Forrest Mars Sr.",
    "elon musk": "Elon Musk",
    "jeff bezos": "Jeff Bezos",
    "paul allen": "Paul Allen",
    "prince alwaleed bin talal alsaud": "Al-Waleed bin Talal",
    "john t walton": "John T. Walton",
    "johanna quandt": "Johanna Quandt & family",
    "jay and robert pritzker": "Jay and Robert Pritzker",
    "kenneth thomson 2nd baron thomson of fleet": "Kenneth Thomson",
    "larry ellison": "Larry Ellison",
    "larry page": "Larry Page",
    "mark zuckerberg": "Mark Zuckerberg",
    "michael bloomberg": "Michael Bloomberg",
    "michael dell": "Michael Dell",
    "sergey brin": "Sergey Brin",
    "steve ballmer": "Steve Ballmer",
    "warren buffett": "Warren Buffett",
    "bernard arnault": "Bernard Arnault",
    "mukesh ambani": "Mukesh Ambani",
    "philip f anschutz": "Philip Anschutz",
    "philip anschutz": "Philip Anschutz",
    "quandt johanna family": "Johanna Quandt & family",
    "quandt johanna": "Johanna Quandt & family",
    "robson walton s": "S. Robson Walton",
    "s robson walton": "S. Robson Walton",
    "steven a ballmer": "Steve Ballmer",
    "steven ballmer": "Steve Ballmer",
    "walton": "Walton family",
    "walton alice l": "Alice Walton",
    "walton family": "Walton family",
    "walton helen r": "Helen Walton",
    "walton jim c": "Jim Walton",
    "walton john t": "John T. Walton",
    "walton s robson": "S. Robson Walton",
    "rob walton": "S. Robson Walton",
    "warren e buffett": "Warren Buffett",
    "warren buffett": "Warren Buffett",
    "william henry gates iii": "Bill Gates",
    "jim walton": "Jim Walton",
}

PERSON_COUNTRY_CODE_ALIASES = {
    "Alice Walton": "US",
    "Al-Waleed bin Talal": "SA",
    "Bill Gates": "US",
    "Cargill family": "US",
    "Donald Newhouse & S.I. Newhouse": "US",
    "Forrest Mars Sr.": "US",
    "Haas": "US",
    "Jay and Robert Pritzker": "US",
    "Jim Walton": "US",
    "Johanna Quandt & family": "DE",
    "Karl and Theo Albrecht": "DE",
    "Larry Ellison": "US",
    "Michael Dell": "US",
    "Paul Allen": "US",
    "Philip Anschutz": "US",
    "S. Robson Walton": "US",
    "Steve Ballmer": "US",
    "Walton family": "US",
    "Warren Buffett": "US",
    "Helen Walton": "US",
    "John T. Walton": "US",
}

PERSON_ID_ALIASES = {
    "Alice Walton": "alice_walton",
    "Al-Waleed bin Talal": "alwaleed_bin_talal_alsaud",
    "Bill Gates": "bill_gates",
    "Forrest Mars Sr.": "forrest_mars_sr",
    "Jim Walton": "jim_walton",
    "Johanna Quandt & family": "johanna_quandt",
    "Karl and Theo Albrecht": "karl_albrecht",
    "Larry Ellison": "larry_ellison",
    "Masayoshi Son": "masayoshi_son",
    "Michael Dell": "michael_dell",
    "Paul Allen": "paul_allen",
    "Philip Anschutz": "philip_anschutz",
    "S. Robson Walton": "rob_walton",
    "Steve Ballmer": "steve_ballmer",
    "Warren Buffett": "warren_buffett",
    "Kenneth Thomson": "kenneth_thomson",
    "Liliane Bettencourt": "liliane_bettencourt",
    "Sultan Hassanal Bolkiah": "sultan_hassanal_bolkiah",
    "Walton family": "walton_family",
    "Oeri, Hoffman and Sacher families": "oeri_hoffman_and_sacher_families",
    "Haas": "haas_family",
    "Kwok brothers": "kwok_brothers",
    "King Fahd bin Abdul Aziz Al Saud": "king_fahd_bin_abdul_aziz_al_saud",
    "Suharto": "suharto",
    "Sheikh Jaber Al-Ahmad Al-Sabah": "sheikh_jaber_al_ahmad_al_sabah",
}

ISO3_TO_ALPHA2 = {
    "arg": "AR",
    "aus": "AU",
    "aut": "AT",
    "bel": "BE",
    "bgr": "BG",
    "bra": "BR",
    "brn": "BN",
    "can": "CA",
    "che": "CH",
    "chn": "CN",
    "deu": "DE",
    "dnk": "DK",
    "esp": "ES",
    "fra": "FR",
    "gbr": "GB",
    "hkg": "HK",
    "idn": "ID",
    "ind": "IN",
    "irl": "IE",
    "isr": "IL",
    "ita": "IT",
    "jpn": "JP",
    "kwt": "KW",
    "mex": "MX",
    "nld": "NL",
    "nor": "NO",
    "nzl": "NZ",
    "pak": "PK",
    "phl": "PH",
    "pry": "PY",
    "qat": "QA",
    "rus": "RU",
    "sau": "SA",
    "swe": "SE",
    "sgp": "SG",
    "tha": "TH",
    "tur": "TR",
    "twn": "TW",
    "ukr": "UA",
    "usa": "US",
    "ven": "VE",
    "zaf": "ZA",
}

LEGACY_YEAR_ROWS = {
    1997: [
        ("Sultan Hassanal Bolkiah", 38.0, "BN"),
        ("Bill Gates", 36.4, "US"),
        ("Walton family", 27.6, "US"),
        ("Warren Buffett", 23.2, "US"),
        ("King Fahd bin Abdul Aziz Al Saud", 20.0, "SA"),
        ("Suharto", 16.0, "ID"),
        ("Paul Allen", 15.31, "US"),
        ("Sheikh Jaber Al-Ahmad Al-Sabah", 15.0, "KW"),
        ("Lee Shau Kee", 14.7, "HK"),
        ("Oeri, Hoffman and Sacher families", 14.3, "CH"),
        ("Haas family", 12.3, "US"),
        ("Kwok brothers", 12.3, "HK"),
    ],
    1998: [
        ("Bill Gates", 51.0, "US"),
        ("Walton family", 48.0, "US"),
        ("Sultan Hassanal Bolkiah", 36.0, "BN"),
        ("Warren Buffett", 33.0, "US"),
        ("King Fahd bin Abdul Aziz Al Saud", 25.0, "SA"),
        ("Paul Allen", 21.0, "US"),
        ("Sheikh Zayed bin Sultan Al Nahyan", 15.0, "AE"),
        ("Sheikh Jaber Al-Ahmad Al-Sabah", 15.0, "KW"),
        ("Kenneth Thomson", 14.4, "CA"),
        ("Forrest Mars Sr. & family", 13.5, "US"),
        ("Jay and Robert Pritzker", 13.5, "US"),
        ("Al-Waleed bin Talal", 13.3, "SA"),
    ],
    1999: [
        ("Bill Gates", 90.0, "US"),
        ("Warren Buffett", 36.0, "US"),
        ("Paul Allen", 30.0, "US"),
        ("Steve Ballmer", 19.5, "US"),
        ("Oeri, Hoffmann and Sacher families", 17.0, "CH"),
        ("Philip Anschutz", 16.5, "US"),
        ("Michael Dell", 16.5, "US"),
        ("S. Robson Walton", 15.8, "US"),
        ("Al-Waleed bin Talal", 15.0, "SA"),
        ("Liliane Bettencourt", 13.9, "FR"),
        ("Karl and Theo Albrecht", 13.6, "DE"),
        ("Li Ka-shing", 12.7, "HK"),
    ],
    2000: [
        ("Bill Gates", 60.0, "US"),
        ("Larry Ellison", 47.0, "US"),
        ("Paul Allen", 28.0, "US"),
        ("Warren Buffett", 25.6, "US"),
        ("Karl and Theo Albrecht", 20.0, "DE"),
        ("Al-Waleed bin Talal", 20.0, "SA"),
        ("S. Robson Walton", 20.0, "US"),
        ("Masayoshi Son", 19.4, "JP"),
        ("Michael Dell", 19.1, "US"),
        ("Kenneth Thomson", 16.1, "CA"),
        ("Philip Anschutz", 15.5, "US"),
        ("Steve Ballmer", 15.5, "US"),
    ],
}

TEMPLATE_COUNTRY_NAMES = {
    "AFG": "Afghanistan",
    "ARE": "United Arab Emirates",
    "ARG": "Argentina",
    "AUS": "Australia",
    "AUT": "Austria",
    "BEL": "Belgium",
    "BGD": "Bangladesh",
    "BRA": "Brazil",
    "CAN": "Canada",
    "CHE": "Switzerland",
    "CHN": "China",
    "COL": "Colombia",
    "DEU": "Germany",
    "DNK": "Denmark",
    "ESP": "Spain",
    "FIN": "Finland",
    "FRA": "France",
    "GBR": "United Kingdom",
    "HKG": "Hong Kong",
    "IDN": "Indonesia",
    "IND": "India",
    "IRL": "Ireland",
    "ISR": "Israel",
    "ITA": "Italy",
    "JPN": "Japan",
    "KOR": "South Korea",
    "KWT": "Kuwait",
    "LIE": "Liechtenstein",
    "MEX": "Mexico",
    "MYS": "Malaysia",
    "NLD": "Netherlands",
    "NOR": "Norway",
    "NZL": "New Zealand",
    "PAK": "Pakistan",
    "PHL": "Philippines",
    "PRT": "Portugal",
    "QAT": "Qatar",
    "RUS": "Russia",
    "SAU": "Saudi Arabia",
    "SGP": "Singapore",
    "SWE": "Sweden",
    "THA": "Thailand",
    "TUR": "Turkey",
    "TWN": "Taiwan",
    "UKR": "Ukraine",
    "USA": "United States",
    "VEN": "Venezuela",
}

TEMPLATE_COUNTRY_CODES = {
    "AFG": "AF",
    "ARE": "AE",
    "ARG": "AR",
    "AUS": "AU",
    "AUT": "AT",
    "BEL": "BE",
    "BGD": "BD",
    "BRA": "BR",
    "CAN": "CA",
    "CHE": "CH",
    "CHN": "CN",
    "COL": "CO",
    "DEU": "DE",
    "DNK": "DK",
    "ESP": "ES",
    "FIN": "FI",
    "FRA": "FR",
    "GBR": "GB",
    "HKG": "HK",
    "IDN": "ID",
    "IND": "IN",
    "IRL": "IE",
    "ISR": "IL",
    "ITA": "IT",
    "JPN": "JP",
    "KOR": "KR",
    "KWT": "KW",
    "LIE": "LI",
    "MEX": "MX",
    "MYS": "MY",
    "NLD": "NL",
    "NOR": "NO",
    "NZL": "NZ",
    "PAK": "PK",
    "PHL": "PH",
    "PRT": "PT",
    "QAT": "QA",
    "RUS": "RU",
    "SAU": "SA",
    "SGP": "SG",
    "SWE": "SE",
    "THA": "TH",
    "TUR": "TR",
    "TWN": "TW",
    "UKR": "UA",
    "USA": "US",
    "VEN": "VE",
}

YEAR_SOURCES = {
    1998: ("html", "https://images.forbes.com/forbes/1998/0706/6201210a_print.html"),
    2001: ("pdf", "https://i.forbesimg.com/media/lists/10/2001/billpdf.pdf"),
    2024: ("article", "https://www.forbes.com/sites/chasewithorn/2024/04/02/forbes-worlds-billionaires-list-2024-the-top-200/"),
}


def _download_text(url: str) -> str:
    response = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=120)
    response.raise_for_status()
    response.encoding = "utf-8"
    return response.text


def _download_bytes(url: str) -> bytes:
    response = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=120)
    response.raise_for_status()
    return response.content


def _normalize(text: str) -> str:
    cleaned = unicodedata.normalize("NFKD", text or "")
    cleaned = "".join(character for character in cleaned if not unicodedata.combining(character))
    cleaned = cleaned.lower().strip()
    cleaned = re.sub(r"[^a-z0-9]+", " ", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def _resolve_country_code(raw_name: str) -> str:
    normalized = _normalize(raw_name)
    if not normalized or normalized in {"na", "n a", "none", "unknown", "_"}:
        return ""
    normalized = COUNTRY_ALIASES.get(normalized, normalized)
    for code, name in TEMPLATE_COUNTRY_NAMES.items():
        if _normalize(name) == normalized:
            return TEMPLATE_COUNTRY_CODES.get(code, code[:2])
    if len(normalized) == 2:
        return normalized.upper()
    if len(normalized) == 3 and normalized.upper() in TEMPLATE_COUNTRY_NAMES:
        return TEMPLATE_COUNTRY_CODES.get(normalized.upper(), "")
    return ""


def _identity_name(full_name: str) -> str:
    text = full_name.strip()
    text = text.rstrip("*").strip()
    text = re.sub(r"\s*&\s*family$", "", text, flags=re.I)
    text = re.sub(r"\s+family$", "", text, flags=re.I)
    return text.strip()


def _canonical_person_name(full_name: str) -> str:
    normalized = _normalize(_identity_name(full_name))
    return PERSON_NAME_ALIASES.get(normalized, _identity_name(full_name))


def _build_person_id(full_name: str) -> str:
    identity = _normalize(_canonical_person_name(full_name))
    return hashlib.sha1(identity.encode("utf-8")).hexdigest()[:16]


def _parse_net_worth(raw_value: str) -> float | None:
    cleaned = raw_value.strip()
    cleaned = cleaned.replace("\xa0", " ").replace("$", "").replace(",", "").replace(" ", "")
    cleaned = cleaned.replace("&nbsp;", "")
    if not cleaned or cleaned.lower() in {"n/a", "na", "-", "_"}:
        return None
    match = re.search(r"([0-9]+(?:\.[0-9]+)?)", cleaned)
    if not match:
        return None
    return float(match.group(1)) * 1_000_000_000


def _format_money(value: float) -> str:
    billions = value / 1_000_000_000
    if billions >= 100:
        return f"${billions:,.0f}B"
    text = f"${billions:,.1f}B"
    return text.replace(".0B", "B")


def _strip_refs(text: str) -> str:
    text = re.sub(r"<ref[^>]*>.*?</ref>", "", text, flags=re.S)
    text = re.sub(r"<ref[^/]*/>", "", text, flags=re.S)
    return text


def _replace_templates(text: str) -> str:
    def replace_sortname(match: re.Match[str]) -> str:
        parts = [part.strip() for part in match.group(1).split("|")]
        if not parts:
            return ""
        if len(parts) >= 3 and parts[-1]:
            return parts[-1]
        if len(parts) >= 2:
            return " ".join(part for part in parts[:2] if part)
        return parts[0]

    def replace_flag(match: re.Match[str]) -> str:
        code = match.group(1).strip()
        if len(code) == 3 and code.upper() in TEMPLATE_COUNTRY_NAMES:
            return TEMPLATE_COUNTRY_NAMES[code.upper()]
        if len(code) == 2:
            return code.upper()
        return code

    text = re.sub(r"\{\{sortname\|([^{}]+)\}\}", replace_sortname, text)
    text = re.sub(r"\{\{(?:Flagu|Flag)\|([^{}|]+)\}\}", replace_flag, text)

    def generic(match: re.Match[str]) -> str:
        inner = match.group(1).strip()
        if "|" not in inner:
            return inner
        parts = [part.strip() for part in inner.split("|") if part.strip()]
        if not parts:
            return ""
        head = parts[0].lower()
        if head.upper() in TEMPLATE_COUNTRY_NAMES:
            return TEMPLATE_COUNTRY_NAMES[head.upper()]
        if head == "nowrap" and len(parts) > 1:
            return parts[1]
        return parts[-1]

    text = re.sub(r"\{\{([^{}]+)\}\}", generic, text)
    return text


def _strip_wiki_markup(text: str) -> str:
    text = text.replace("&nbsp;", " ")
    text = text.replace("''", "")
    text = _strip_refs(text)
    text = re.sub(r"\[\[(?:[^|\]]+\|)?([^\]]+)\]\]", r"\1", text)
    text = _replace_templates(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip(" |")


def _parse_table_rows(section_text: str) -> list[list[str]]:
    table_start = section_text.find("{|")
    if table_start < 0:
        return []
    table_end = section_text.find("\n|}", table_start)
    if table_end < 0:
        return []

    table = section_text[table_start:table_end]
    rows: list[str] = []
    current = ""
    for raw_line in table.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("{|") or line.startswith("|+") or line.startswith("!") or line == "|-":
            if line == "|-" and current:
                rows.append(current)
                current = ""
            continue
        if line.startswith("|-"):
            if current:
                rows.append(current)
                current = ""
            continue
        if current:
            if line.startswith("|"):
                current += " || " + line.lstrip("|").strip()
            else:
                current += " " + line
        else:
            current = line.lstrip("|").strip()
    if current:
        rows.append(current)

    parsed_rows: list[list[str]] = []
    for row_text in rows:
        row_text = row_text.replace("\n", " ")
        row_text = re.sub(r"\s+\|\s*$", "", row_text)
        parts = [part.strip() for part in row_text.split("||")]
        parts = [part for part in parts if part]
        if len(parts) < 5:
            continue
        if len(parts) > 6:
            parts = parts[:5] + [" || ".join(parts[5:])]
        parsed_rows.append(parts)
    return parsed_rows


def _extract_year_sections(raw_text: str) -> dict[int, str]:
    matches = list(re.finditer(r"^===\s*(\d{4})\s*===\s*$", raw_text, flags=re.M))
    sections: dict[int, str] = {}
    for index, match in enumerate(matches):
        year = int(match.group(1))
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(raw_text)
        sections[year] = raw_text[start:end]
    return sections


def _parse_ranked_names_from_lines(lines: list[str], year: int, limit: int) -> list[dict[str, str]]:
    entries: list[dict[str, str]] = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line.lower().startswith("net worth"):
            i += 1
            continue

        name = ""
        j = i - 1
        while j >= 0:
            candidate = lines[j].strip()
            if candidate and not candidate.lower().startswith(("page", "rank", "name", "net worth")):
                name = candidate
                break
            j -= 1

        value = ""
        if i + 1 < len(lines):
            value = lines[i + 1].strip()

        wealth = _parse_net_worth(value or line)
        if name and wealth is not None:
            entries.append(
                {
                    "year": str(year),
                    "rank": str(len(entries) + 1),
                    "full_name": _canonical_person_name(_strip_wiki_markup(name)),
                    "net_worth": str(wealth),
                    "age": "",
                    "country_of_citizenship": "",
                    "country_of_residence": "",
                    "source": "",
                }
            )
            if len(entries) >= limit:
                break
        i += 1
    return entries


def _parse_pdf_ranked_list(lines: list[str], year: int, limit: int) -> list[dict[str, str]]:
    entries: list[dict[str, str]] = []
    cleaned = [line.strip() for line in lines if line.strip()]
    start_index = None
    for index, line in enumerate(cleaned):
        if line.lower() == "rank":
            start_index = index + 1
            break
    if start_index is None:
        return entries

    index = start_index
    while index < len(cleaned) and len(entries) < limit:
        rank_line = cleaned[index]
        if not rank_line.isdigit():
            index += 1
            continue

        name_index = index + 1
        wealth_index = index + 2
        if wealth_index >= len(cleaned):
            break
        name = _canonical_person_name(cleaned[name_index])
        if name.lower() in {"networth", "($billions)", "age", "origin of wealth"}:
            index += 1
            continue

        wealth_token = cleaned[wealth_index]
        wealth = _parse_net_worth(wealth_token) if "." in wealth_token else None
        if wealth is None:
            index += 1
            continue

        entries.append(
            {
                "year": str(year),
                "rank": rank_line,
                "full_name": _canonical_person_name(_strip_wiki_markup(name)),
                "net_worth": str(wealth),
                "age": "",
                "country_of_citizenship": "",
                "country_of_residence": "",
                "source": "",
            }
        )
        index = wealth_index + 1
    return entries


def _load_year_specific_rows(year: int, limit: int) -> list[dict[str, str]]:
    source = YEAR_SOURCES.get(year)
    if source is None:
        return []
    kind, url = source
    if kind == "pdf":
        content = _download_bytes(url)
        pdf = fitz.open(stream=content, filetype="pdf")
        lines: list[str] = []
        for page_index in range(pdf.page_count):
            lines.extend((pdf.load_page(page_index).get_text() or "").splitlines())
        return _parse_pdf_ranked_list(lines, year, limit)
    if kind == "html":
        html = _download_text(url)
        text = BeautifulSoup(html, "html.parser").get_text("\n", strip=True)
        lines = text.splitlines()
        return _parse_ranked_names_from_lines(lines, year, limit)
    if kind == "article":
        html = _download_text(url)
        text = BeautifulSoup(html, "html.parser").get_text("\n", strip=True)
        pattern = re.compile(
            r"(?ms)^(\d+)\.\s+(.+?)\nNet Worth:\n\$([0-9,.]+)\s+Billion\n\|\s*Age:\n([^\n]+)\n\|\s*Country/Territory:\n([^\n]+)\n\|\s*Industry:\n([^\n]+)",
        )
        entries: list[dict[str, str]] = []
        for rank, name, wealth_text, age, country, source in pattern.findall(text):
            wealth = _parse_net_worth(f"${wealth_text}B")
            if wealth is None:
                continue
            entries.append(
                {
                    "year": str(year),
                    "rank": rank,
                    "full_name": _canonical_person_name(_strip_wiki_markup(name)),
                    "net_worth": str(wealth),
                    "age": age.strip(),
                    "country_of_citizenship": country.strip(),
                    "country_of_residence": "",
                    "source": source.strip(),
                }
            )
            if len(entries) >= limit:
                break
        return entries
    return []


def _parse_year_rows(year: int, section_text: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for raw_parts in _parse_table_rows(section_text):
        if len(raw_parts) < 6:
            continue
        rank_text = _strip_wiki_markup(raw_parts[0])
        name_text = _canonical_person_name(_strip_wiki_markup(raw_parts[1]))
        worth_text = _strip_wiki_markup(raw_parts[2])
        age_text = _strip_wiki_markup(raw_parts[3])
        country_text = _strip_wiki_markup(raw_parts[4].replace("<br />", " | ").replace("<br/>", " | "))
        source_text = _strip_wiki_markup(" || ".join(raw_parts[5:]))

        rank_match = re.search(r"\d+", rank_text)
        if not rank_match:
            continue
        wealth = _parse_net_worth(worth_text)
        if wealth is None or wealth <= 0:
            continue

        if not name_text or name_text.lower() in {"rank", "name"}:
            continue

        country_code = ""
        for candidate in [part.strip() for part in re.split(r"\s+\|\s+|\s*/\s+|,", country_text) if part.strip()]:
            country_code = _resolve_country_code(candidate)
            if country_code:
                break
        if not country_code:
            country_code = _resolve_country_code(country_text)

        rows.append(
            {
                "year": str(year),
                "rank": rank_match.group(0),
                "full_name": name_text,
                "net_worth": str(wealth),
                "age": age_text,
                "country_of_citizenship": country_text,
                "country_of_residence": country_code,
                "source": source_text,
            }
        )
    return rows


def _rank_from_text(value: str) -> int:
    match = re.search(r"\d+", value)
    return int(match.group(0)) if match else 10**9


def _download_csv_rows(url: str) -> list[dict[str, str]]:
    response = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=120)
    response.raise_for_status()
    return list(csv.DictReader(io.StringIO(response.text)))


def _alpha2_for_gapminder_country(country_code: str) -> str:
    normalized = (country_code or "").strip().lower()
    if not normalized:
        return ""
    if len(normalized) == 2:
        return normalized.upper()
    return ISO3_TO_ALPHA2.get(normalized, normalized[:2].upper())


def _legacy_gapminder_name(name: str) -> str:
    return PERSON_NAME_ALIASES.get(_normalize(name), name)


def _load_gapminder_rankings(top_n: int) -> dict[int, list[dict[str, str]]]:
    people_rows = _download_csv_rows(GAPMINDER_PERSONS_URL)
    worth_rows = _download_csv_rows(GAPMINDER_WORTH_URL)

    people = {row["person"]: row for row in people_rows}
    by_year: dict[int, list[dict[str, str]]] = defaultdict(list)

    for row in worth_rows:
        try:
            year = int(row["time"])
            worth = float(row["worth"])
        except (TypeError, ValueError):
            continue
        person = row["person"].strip()
        person_row = people.get(person)
        if not person_row:
            continue
        name = _canonical_person_name(_legacy_gapminder_name(person_row["name"].strip()))
        country_code = _alpha2_for_gapminder_country(person_row.get("country", ""))
        by_year[year].append(
            {
                "ranking_date": f"{year:04d}-12-31",
                "country_name": name,
                "country_code": country_code,
                "country_iso3": person,
                "population": str(int(round(worth * 1_000_000))),
                "yearly_change": "",
                "season_summary": "",
                "data_source": "Gapminder billionaires dataset (Forbes/Hurun)",
            }
        )

    for year, rows in by_year.items():
        rows.sort(key=lambda row: (-float(row["population"]), row["country_name"]))
        by_year[year] = rows[:top_n]
    return by_year


def _legacy_rows_for_year(year: int) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for name, worth_billion, country_code in LEGACY_YEAR_ROWS.get(year, []):
        display_name = _canonical_person_name(name)
        rows.append(
            {
                "ranking_date": f"{year:04d}-12-31",
                "country_name": display_name,
                "country_code": country_code,
                "country_iso3": PERSON_ID_ALIASES.get(display_name, _build_person_id(display_name)),
                "population": str(int(round(worth_billion * 1_000_000_000))),
                "yearly_change": "",
                "season_summary": "",
                "data_source": "Forbes World's Billionaires legacy tables",
            }
        )
    return rows


def build_rows(start_year: int, end_year: int, top_n: int) -> list[dict[str, str]]:
    gapminder_rows = _load_gapminder_rankings(top_n)
    snapshots: dict[int, dict[str, dict[str, str]]] = defaultdict(dict)
    previous_values: dict[str, float] = {}

    for year in range(start_year, end_year + 1):
        if year <= 2000:
            current_rows = _legacy_rows_for_year(year)
        else:
            current_rows = gapminder_rows.get(year, [])

        if not current_rows:
            continue

        current_rows = sorted(current_rows, key=lambda row: (-float(row["population"]), row["country_name"]))[:top_n]
        leader = current_rows[0]
        gains: list[tuple[float, str]] = []

        for row in current_rows:
            person_id = row["country_iso3"]
            current_value = float(row["population"])
            previous_value = previous_values.get(person_id)
            if previous_value is not None:
                delta = current_value - previous_value
                row["yearly_change"] = str(int(round(delta)))
                if delta > 0:
                    gains.append((delta, row["country_name"]))
            else:
                row["yearly_change"] = ""

        gains.sort(key=lambda item: (-item[0], item[1]))
        if gains:
            biggest_gain_value, biggest_gain_name = gains[0]
            summary = (
                f"Leader: {leader['country_name']} {_format_money(float(leader['population']))}"
                f"|Biggest gain: {biggest_gain_name} +{_format_money(biggest_gain_value)}"
            )
        else:
            summary = f"Leader: {leader['country_name']} {_format_money(float(leader['population']))}|Dataset begins {year}"

        for row in current_rows:
            row["season_summary"] = summary
            snapshots[year][row["country_iso3"]] = row

        previous_values = {row["country_iso3"]: float(row["population"]) for row in current_rows}

    rows: list[dict[str, str]] = []
    for year in sorted(snapshots):
        ranked = list(snapshots[year].values())
        ranked.sort(key=lambda row: (-float(row["population"]), row["country_name"]))
        rows.extend(ranked)
    return rows


def _download_flags(country_codes: set[str], flags_dir: Path) -> list[Path]:
    flags_dir.mkdir(parents=True, exist_ok=True)
    downloaded: list[Path] = []
    for alpha2 in sorted(code.lower() for code in country_codes if code):
        output_path = flags_dir / f"{alpha2}.png"
        if output_path.exists():
            continue
        try:
            response = requests.get(f"https://flagcdn.com/w80/{alpha2}.png", headers={"User-Agent": USER_AGENT}, timeout=60)
            response.raise_for_status()
            output_path.write_bytes(response.content)
            downloaded.append(output_path)
        except Exception as error:
            print(f"[scraper] flag download skipped for {alpha2}: {error}")
    return downloaded


def write_csv(rows: list[dict[str, str]], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "ranking_date",
                "country_name",
                "country_code",
                "country_iso3",
                "population",
                "yearly_change",
                "season_summary",
                "data_source",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the Forbes billionaires timeseries from merged billionaire datasets.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--flags-dir", type=Path, default=DEFAULT_FLAGS_DIR)
    parser.add_argument("--start-year", type=int, default=START_YEAR)
    parser.add_argument("--end-year", type=int, default=END_YEAR)
    parser.add_argument("--top-n", type=int, default=TOP_N)
    parser.add_argument("--skip-flags", action="store_true", help="Skip downloading flag PNGs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.start_year > args.end_year:
        raise ValueError("--start-year must be less than or equal to --end-year.")
    if args.top_n <= 0:
        raise ValueError("--top-n must be greater than 0.")

    rows = build_rows(args.start_year, args.end_year, args.top_n)
    output = write_csv(rows, args.output)

    downloaded_flags = [] if args.skip_flags else _download_flags(
        {row["country_code"] for row in rows if row["country_code"]},
        args.flags_dir,
    )

    snapshots = len({row["ranking_date"] for row in rows})
    print(f"[scraper] Forbes billionaires CSV generated -> {output}")
    print(
        f"[scraper] {len(rows)} rows, {snapshots} snapshots, {args.start_year}-{args.end_year}, "
        f"{len(downloaded_flags)} flags downloaded"
    )


if __name__ == "__main__":
    main()
