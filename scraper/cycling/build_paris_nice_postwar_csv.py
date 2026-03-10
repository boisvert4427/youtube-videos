from __future__ import annotations

import csv
import json
import re
import time
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_CSV = PROJECT_ROOT / "data" / "processed" / "cycling" / "paris_nice" / "paris_nice_timeline_postwar_template.csv"
START_YEAR = 1946
END_YEAR = 2025
USER_AGENT = "Mozilla/5.0 (compatible; Codex Paris-Nice builder)"
EN_DASH = "\u2013"

FIELDNAMES = [
    "year",
    "winner_name",
    "winner_team",
    "winner_country",
    "image_path",
    "card_bg_color",
    "accent_color",
    "badge_label",
    "gc1_name",
    "gc1_team",
    "gc1_country",
    "gc1_gap",
    "gc2_name",
    "gc2_team",
    "gc2_country",
    "gc2_gap",
    "gc3_name",
    "gc3_team",
    "gc3_country",
    "gc3_gap",
    "gc4_name",
    "gc4_team",
    "gc4_country",
    "gc4_gap",
    "gc5_name",
    "gc5_team",
    "gc5_country",
    "gc5_gap",
    "points_name",
    "points_team",
    "points_country",
    "mountains_name",
    "mountains_team",
    "mountains_country",
    "notes",
]


def _fetch_api(url: str) -> dict:
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8", "ignore"))


def _fetch_wikitext_pages(titles: list[str]) -> dict[str, str]:
    if not titles:
        return {}
    encoded_titles = "|".join(quote(title, safe="") for title in titles)
    url = (
        "https://en.wikipedia.org/w/api.php?action=query&prop=revisions"
        f"&rvprop=content&titles={encoded_titles}&format=json&formatversion=2"
    )
    data = _fetch_api(url)
    result: dict[str, str] = {}
    for page in data.get("query", {}).get("pages", []):
        title = page.get("title", "")
        if page.get("missing"):
            result[title] = ""
        else:
            result[title] = page.get("revisions", [{}])[0].get("content", "")
    return result


def _chunked(values: list[str], size: int) -> list[list[str]]:
    return [values[i : i + size] for i in range(0, len(values), size)]


def _strip_refs(text: str) -> str:
    text = re.sub(r"<ref[^>/]*/>", "", text)
    text = re.sub(r"<ref[^>]*>.*?</ref>", "", text, flags=re.S)
    return text


def _extract_balanced(text: str, start: int, open_token: str = "{{", close_token: str = "}}") -> tuple[str, int]:
    depth = 0
    i = start
    end = len(text)
    while i < end:
        if text.startswith(open_token, i):
            depth += 1
            i += len(open_token)
            continue
        if text.startswith(close_token, i):
            depth -= 1
            i += len(close_token)
            if depth == 0:
                return text[start:i], i
            continue
        i += 1
    return text[start:end], end


def _split_top_level_fields(text: str, sep: str = "|") -> list[str]:
    parts: list[str] = []
    current: list[str] = []
    brace_depth = 0
    bracket_depth = 0
    i = 0
    while i < len(text):
        if text.startswith("{{", i):
            brace_depth += 1
            current.append("{{")
            i += 2
            continue
        if text.startswith("}}", i):
            brace_depth = max(0, brace_depth - 1)
            current.append("}}")
            i += 2
            continue
        if text.startswith("[[", i):
            bracket_depth += 1
            current.append("[[")
            i += 2
            continue
        if text.startswith("]]", i):
            bracket_depth = max(0, bracket_depth - 1)
            current.append("]]")
            i += 2
            continue
        if text[i] == sep and brace_depth == 0 and bracket_depth == 0:
            parts.append("".join(current))
            current = []
            i += 1
            continue
        current.append(text[i])
        i += 1
    parts.append("".join(current))
    return parts


def _clean_wiki_value(text: str) -> str:
    if not text:
        return ""
    value = _strip_refs(text).strip()
    value = value.replace("&nbsp;", " ")
    value = re.sub(r"\{\{nowrap\|(.*?)\}\}", r"\1", value)
    value = re.sub(r"\{\{sortname\|([^|{}]+)\|([^|{}]+)(?:\|[^{}]*)?\}\}", r"\1 \2", value)

    def replace_link(match: re.Match[str]) -> str:
        inner = match.group(1)
        parts = inner.split("|")
        return parts[-1]

    value = re.sub(r"\[\[([^][]+)\]\]", replace_link, value)

    def replace_template(match: re.Match[str]) -> str:
        inner = match.group(1)
        parts = [p.strip() for p in inner.split("|")]
        name = parts[0].lower()
        if name == "uci team code" and len(parts) >= 2:
            return parts[1].replace(" men", "").replace(" women", "").strip()
        if name in {"flag", "flagu", "flagicon", "country data", "flagg"}:
            return ""
        if len(parts) >= 2:
            return parts[-1]
        return parts[0]

    while "{{" in value and "}}" in value:
        new_value = re.sub(r"\{\{([^{}]+)\}\}", replace_template, value)
        if new_value == value:
            break
        value = new_value

    value = re.sub(r"''+", "", value)
    value = re.sub(r"\s+", " ", value).strip(" ,;")
    return value


def _extract_infobox_field(text: str, key: str) -> str:
    match = re.search(rf"^\|\s*{re.escape(key)}\s*=\s*(.+)$", text, flags=re.M)
    return _clean_wiki_value(match.group(1)) if match else ""


def _parse_winners_table(main_text: str) -> dict[int, dict[str, str]]:
    winners: dict[int, dict[str, str]] = {}
    for match in re.finditer(r"\{\{Cycling past winner rider\|", main_text):
        template, _ = _extract_balanced(main_text, match.start())
        inner = template[2:-2]
        fields = _split_top_level_fields(inner)
        values: dict[str, str] = {}
        for field in fields[1:]:
            if "=" not in field:
                continue
            key, value = field.split("=", 1)
            values[key.strip()] = value.strip()
        year_match = re.search(r"(\d{4})", values.get("year", ""))
        if not year_match:
            continue
        year = int(year_match.group(1))
        winners[year] = {
            "winner_name": _clean_wiki_value(values.get("name", "")),
            "winner_country": values.get("nat", ""),
            "winner_team": _clean_wiki_value(values.get("team", "")),
        }
    return winners


def _parse_general_top5(text: str) -> list[dict[str, str]]:
    starts = list(re.finditer(r"\{\{cyclingresult start\|title=.*?General classification", text, flags=re.I))
    if not starts:
        return []
    last_start = starts[-1].start()
    end_marker = text.find("{{cyclingresult end}}", last_start)
    if end_marker == -1:
        return []
    block = text[last_start:end_marker]
    results: list[dict[str, str]] = []
    i = 0
    while i < len(block) and len(results) < 5:
        start = block.find("{{cyclingresult|", i)
        if start == -1:
            break
        template, end = _extract_balanced(block, start)
        i = end
        inner = template[2:-2]
        fields = _split_top_level_fields(inner)
        if len(fields) < 6 or fields[0].strip().lower() != "cyclingresult":
            continue
        try:
            position = int(fields[1].strip())
        except ValueError:
            continue
        if position < 1 or position > 5:
            continue
        while len(results) < position - 1:
            results.append({"name": "-", "country": "", "team": "", "gap": ""})
        results.append(
            {
                "name": _clean_wiki_value(fields[2]),
                "country": _clean_wiki_value(fields[3]),
                "team": _clean_wiki_value(fields[4]),
                "gap": _clean_wiki_value(fields[5]),
            }
        )
    return results[:5]


def build_rows() -> list[dict[str, str]]:
    main_title = f"Paris{EN_DASH}Nice"
    main_text = _fetch_wikitext_pages([main_title]).get(main_title, "")
    winners_map = _parse_winners_table(main_text)

    titles = [f"{year} Paris{EN_DASH}Nice" for year in range(START_YEAR, END_YEAR + 1)]
    page_texts: dict[str, str] = {}
    for chunk in _chunked(titles, 10):
        page_texts.update(_fetch_wikitext_pages(chunk))
        time.sleep(0.25)

    rows: list[dict[str, str]] = []
    for year in range(START_YEAR, END_YEAR + 1):
        title = f"{year} Paris{EN_DASH}Nice"
        text = page_texts.get(title, "")
        winner_seed = winners_map.get(year, {})
        row = {field: "" for field in FIELDNAMES}
        row["year"] = str(year)
        row["card_bg_color"] = "#1f6f78"
        row["accent_color"] = "#d8c06a"
        row["badge_label"] = "PN #1"

        row["winner_name"] = winner_seed.get("winner_name", "") or _extract_infobox_field(text, "first")
        row["winner_team"] = winner_seed.get("winner_team", "") or _extract_infobox_field(text, "first_team")
        row["winner_country"] = winner_seed.get("winner_country", "") or _extract_infobox_field(text, "first_nat")

        top5 = _parse_general_top5(text)
        if top5:
            for idx, item in enumerate(top5, start=1):
                row[f"gc{idx}_name"] = item["name"]
                row[f"gc{idx}_team"] = item["team"]
                row[f"gc{idx}_country"] = item["country"]
                row[f"gc{idx}_gap"] = item["gap"]
        else:
            row["gc1_name"] = row["winner_name"]
            row["gc1_team"] = row["winner_team"]
            row["gc1_country"] = row["winner_country"]
            row["gc1_gap"] = "0:00" if row["winner_name"] else ""
            for idx, prefix in ((2, "second"), (3, "third")):
                row[f"gc{idx}_name"] = _extract_infobox_field(text, prefix)
                row[f"gc{idx}_team"] = _extract_infobox_field(text, f"{prefix}_team")
                row[f"gc{idx}_country"] = _extract_infobox_field(text, f"{prefix}_nat")
                row[f"gc{idx}_gap"] = ""

        row["points_name"] = _extract_infobox_field(text, "points")
        row["points_team"] = _extract_infobox_field(text, "points_team")
        row["points_country"] = _extract_infobox_field(text, "points_nat")
        row["mountains_name"] = _extract_infobox_field(text, "mountains")
        row["mountains_team"] = _extract_infobox_field(text, "mountains_team")
        row["mountains_country"] = _extract_infobox_field(text, "mountains_nat")

        note_parts: list[str] = []
        if not text:
            note_parts.append("no annual Wikipedia page")
        if not row["gc4_name"]:
            note_parts.append("top 5 partial")
        if not row["points_name"]:
            note_parts.append("points unavailable")
        if not row["mountains_name"]:
            note_parts.append("mountains unavailable")
        row["notes"] = "; ".join(note_parts)
        rows.append(row)
    return rows


def write_rows(rows: list[dict[str, str]], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    rows = build_rows()
    write_rows(rows, OUTPUT_CSV)
    total = len(rows)
    with_top5 = sum(1 for row in rows if row["gc5_name"])
    with_points = sum(1 for row in rows if row["points_name"])
    with_mountains = sum(1 for row in rows if row["mountains_name"])
    print(f"[scraper] Paris-Nice rows written -> {OUTPUT_CSV}")
    print(f"[scraper] coverage: rows={total} full_top5={with_top5} points={with_points} mountains={with_mountains}")


if __name__ == "__main__":
    main()
