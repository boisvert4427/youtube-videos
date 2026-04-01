from __future__ import annotations

import csv
import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_CSV = PROJECT_ROOT / "data" / "raw" / "france_kings_reigns.csv"
OUTPUT_CSV = INPUT_CSV
HEADERS = {"User-Agent": "youtube-videos-history/1.0"}

SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def fetch_json(url: str) -> dict:
    request = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def resolve_french_title(display_name: str) -> str | None:
    try:
        url = (
            "https://fr.wikipedia.org/w/api.php?action=opensearch&limit=5&namespace=0&format=json"
            f"&search={urllib.parse.quote(display_name)}"
        )
        data = fetch_json(url)
        candidates = data[1]
        if candidates:
            return candidates[0]
    except Exception:
        return None
    return None


def fetch_french_extract(title: str) -> tuple[str, str]:
    url = (
        "https://fr.wikipedia.org/w/api.php?action=query&prop=extracts&exintro=1&explaintext=1&redirects=1"
        f"&titles={urllib.parse.quote(title)}&format=json"
    )
    for attempt in range(4):
        try:
            data = fetch_json(url)
            page = next(iter(data["query"]["pages"].values()))
            return str(page.get("title") or "").strip(), (page.get("extract") or "").strip()
        except Exception:
            time.sleep(1.2 + attempt * 1.5)
    return "", ""


def clean_sentences(text: str) -> list[str]:
    text = text.replace("\n", " ").strip()
    parts = [part.strip() for part in SENTENCE_SPLIT_RE.split(text) if part.strip()]
    cleaned: list[str] = []
    for part in parts:
        part = re.sub(r"\s+", " ", part)
        if len(part) < 28:
            continue
        if part not in cleaned:
            cleaned.append(part)
    return cleaned


def normalize_text(value: str) -> str:
    replacements = {
        "à": "a",
        "â": "a",
        "ä": "a",
        "é": "e",
        "è": "e",
        "ê": "e",
        "ë": "e",
        "î": "i",
        "ï": "i",
        "ô": "o",
        "ö": "o",
        "ù": "u",
        "û": "u",
        "ü": "u",
        "ç": "c",
        "œ": "oe",
    }
    value = value.lower()
    for source, target in replacements.items():
        value = value.replace(source, target)
    return re.sub(r"[^a-z0-9\s]", " ", value)


def extract_matches_row(row: dict[str, str], page_title: str, extract: str) -> bool:
    if not extract:
        return False
    title_stack = normalize_text(page_title)
    haystack = normalize_text(extract)
    tokens = [token for token in normalize_text(row["display_name"]).split() if len(token) >= 4]
    if not tokens:
        tokens = [token for token in normalize_text(row["ruler_name"]).split() if len(token) >= 4]
    if not tokens:
        return True
    if not any(token in title_stack for token in tokens):
        return False
    return any(token in haystack for token in tokens)


def build_facts(row: dict[str, str], page_title: str, extract: str) -> tuple[str, str, str]:
    if not extract_matches_row(row, page_title, extract):
        extract = ""
    sentences = clean_sentences(extract)
    facts: list[str] = []
    for sentence in sentences:
        if sentence.lower().startswith("cet article"):
            continue
        facts.append(sentence)
        if len(facts) == 3:
            break

    fallback = [
        f"Règne de {row['display_name']} de {row['start_year']} à {row['end_year']}.",
        f"{row['display_name']} appartient à la dynastie des {row['dynasty']}.",
        row["notes"].strip() or f"{row['display_name']} figure dans la succession canonique retenue pour la timeline.",
    ]
    while len(facts) < 3:
        candidate = fallback[len(facts)]
        if candidate not in facts:
            facts.append(candidate)
        else:
            facts.append(f"{row['display_name']} est inclus dans la chronologie des rois de France.")
    return facts[0], facts[1], facts[2]


def write_rows(rows: list[dict[str, str]], path: Path) -> None:
    fieldnames = [
        "start_year",
        "end_year",
        "ruler_name",
        "display_name",
        "dynasty",
        "house_color",
        "wiki_title",
        "notes",
        "fait_1",
        "fait_2",
        "fait_3",
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def normalize_existing_fact(row: dict[str, str], key: str) -> str:
    return row.get(key, "").strip()


def main() -> None:
    rows = load_rows(INPUT_CSV)
    enriched_rows: list[dict[str, str]] = []
    for row in rows:
        title = resolve_french_title(row["display_name"].strip()) or row["display_name"].strip()
        page_title, extract = fetch_french_extract(title)
        fact_1, fact_2, fact_3 = build_facts(row, page_title, extract)
        row["fait_1"] = fact_1
        row["fait_2"] = fact_2
        row["fait_3"] = fact_3
        enriched_rows.append(row)
        write_rows([*enriched_rows, *rows[len(enriched_rows) :]], OUTPUT_CSV)
        print(f"[history] faits prêts -> {row['display_name']}")
        time.sleep(0.35)
    write_rows(enriched_rows, OUTPUT_CSV)
    print(f"[history] dataset enrichi en français -> {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
