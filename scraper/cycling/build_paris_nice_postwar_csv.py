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
START_YEAR = 1951
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

MANUAL_GC_TOP5 = {
    2004: [
        {"name": "Jörg Jaksche", "country": "GER", "team": "Team CSC", "gap": "28:00:01"},
        {"name": "Davide Rebellin", "country": "ITA", "team": "Gerolsteiner", "gap": "+ 15"},
        {"name": "Bobby Julich", "country": "USA", "team": "Team CSC", "gap": "+ 43"},
        {"name": "Jens Voigt", "country": "GER", "team": "Team CSC", "gap": "+ 43"},
        {"name": "George Hincapie", "country": "USA", "team": "US Postal p/b Berry Floor", "gap": "+ 46"},
    ],
    2005: [
        {"name": "Bobby Julich", "country": "USA", "team": "Team CSC", "gap": "22:32:13"},
        {"name": "Alejandro Valverde", "country": "ESP", "team": "Illes Balears - Caisse d'Epargne", "gap": "+ 10"},
        {"name": "Constantino Zaballa", "country": "ESP", "team": "Saunier Duval - Prodir", "gap": "+ 19"},
        {"name": "Jens Voigt", "country": "GER", "team": "Team CSC", "gap": "+ 44"},
        {"name": "Jörg Jaksche", "country": "GER", "team": "Liberty Seguros - Würth Team", "gap": "+ 45"},
    ],
    2006: [
        {"name": "Floyd Landis", "country": "USA", "team": "Phonak Hearing Systems", "gap": "31:54:41"},
        {"name": "Patxi Vila", "country": "ESP", "team": "Lampre-Fondital", "gap": "+ 09"},
        {"name": "Antonio Colom", "country": "ESP", "team": "Caisse d'Epargne - Illes Balears", "gap": "+ 01:05"},
        {"name": "Samuel Sánchez", "country": "ESP", "team": "Euskaltel-Euskadi", "gap": "+ 01:13"},
        {"name": "Fränk Schleck", "country": "LUX", "team": "Team CSC", "gap": "+ 01:13"},
    ],
}

SUPPLEMENTAL_POINTS_WINNERS = {
    1964: "Jan Janssen",
    1965: "Rudi Altig",
    1969: "Marino Basso",
    1974: "Rik Van Linden",
    2001: "Danilo Hondo",
    2002: "Alessandro Petacchi",
    2003: "Laurent Brochard",
    2004: "Davide Rebellin",
    2005: "Jens Voigt",
    2006: "Samuel Sánchez",
    2007: "Franco Pellizotti",
    2008: "Thor Hushovd",
    2009: "Sylvain Chavanel",
}

SUPPLEMENTAL_MOUNTAINS_WINNERS = {
    1964: "Guy Ignolin",
    1965: "Gianni Motta",
    1969: "Gilbert Bellone",
    1974: "Jean-Pierre Danguillaume",
    1989: "Julio César Cadena",
    2001: "Piotr Wadecki",
    2003: "Tyler Hamilton",
    2004: "Aitor Osa",
    2005: "David Moncoutié",
    2006: "David Moncoutié",
    2007: "Thomas Voeckler",
    2008: "Clément Lhotellerie",
    2009: "Tony Martin",
}


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


def _extract_table_after(text: str, marker_patterns: list[str]) -> str:
    for pattern in marker_patterns:
        match = re.search(pattern, text, flags=re.I | re.S)
        if not match:
            continue
        table_start = text.find("{|", match.end())
        if table_start == -1:
            continue
        table_text, _ = _extract_balanced(text, table_start, open_token="{|", close_token="|}")
        if table_text:
            return table_text
    return ""


def _extract_country_from_rider_cell(cell: str) -> str:
    match = re.search(r"\{\{flagathlete\|", cell, flags=re.I)
    if not match:
        return ""
    template, _ = _extract_balanced(cell, match.start())
    fields = _split_top_level_fields(template[2:-2])
    if len(fields) >= 3:
        return _clean_wiki_value(fields[2])
    return ""


def _extract_rider_name(cell: str) -> str:
    match = re.search(r"\{\{flagathlete\|", cell, flags=re.I)
    if match:
        template, _ = _extract_balanced(cell, match.start())
        fields = _split_top_level_fields(template[2:-2])
        if len(fields) >= 2:
            return _clean_wiki_value(fields[1])
    cleaned = _clean_wiki_value(cell)
    cleaned = re.sub(r"\s+\([A-Z]{2,3}\)$", "", cleaned)
    return cleaned


def _parse_table_cell(line: str) -> str:
    stripped = line.strip()
    if not stripped or stripped == "|-" or stripped == "|}":
        return ""
    marker = stripped[0]
    if marker not in {"|", "!"}:
        return ""
    content = stripped[1:].strip()
    if "||" in content or "!!" in content:
        return content
    if "|" in content:
        left, right = content.split("|", 1)
        if "=" in left or left.strip().endswith(("scope", "align", "style")) or "scope" in left or "style" in left or "align" in left:
            return right.strip()
    return content.strip()


def _parse_wikitable_rows(table_text: str) -> list[list[str]]:
    rows: list[list[str]] = []
    current: list[str] = []
    for raw_line in table_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("|-"):
            if current:
                rows.append(current)
                current = []
            continue
        if line.startswith("|}") or line.startswith("{|"):
            continue
        if not line.startswith(("|", "!")):
            continue
        content = _parse_table_cell(line)
        if not content:
            continue
        splitter = "||" if "||" in content else "!!" if "!!" in content else None
        if splitter:
            current.extend(part.strip() for part in content.split(splitter) if part.strip())
        else:
            current.append(content)
    if current:
        rows.append(current)
    return rows


def _parse_general_top5_wikitable(text: str) -> list[dict[str, str]]:
    table_text = _extract_table_after(
        text,
        [
            r"Final general classification(?:\s*\(1-10\))?",
            r"=+\s*(?:\[\[)?General classification(?:\|[^]]+)?(?:\]\])?\s*=+",
            r"==+\s*General classification\s*==+",
        ],
    )
    if not table_text:
        return []
    results: list[dict[str, str]] = []
    for row in _parse_wikitable_rows(table_text):
        if len(row) < 3:
            continue
        rank_text = _clean_wiki_value(row[0])
        rank_match = re.search(r"\b([1-9]|10)\b", rank_text)
        if not rank_match:
            continue
        rank = int(rank_match.group(1))
        if rank < 1 or rank > 5:
            continue
        rider_cell = row[1]
        team_cell = row[2] if len(row) >= 4 else ""
        gap_cell = row[3] if len(row) >= 4 else row[2]
        while len(results) < rank - 1:
            results.append({"name": "-", "country": "", "team": "", "gap": ""})
        results.append(
            {
                "name": _extract_rider_name(rider_cell),
                "country": _extract_country_from_rider_cell(rider_cell),
                "team": _clean_wiki_value(team_cell) if len(row) >= 4 else "",
                "gap": _clean_wiki_value(gap_cell),
            }
        )
    return results[:5]


def _parse_single_leader_from_table(text: str, marker_patterns: list[str]) -> dict[str, str]:
    table_text = _extract_table_after(text, marker_patterns)
    if not table_text:
        return {}
    for row in _parse_wikitable_rows(table_text):
        if len(row) < 2:
            continue
        rank_text = _clean_wiki_value(row[0])
        rank_match = re.search(r"\b([1-9]|10)\b", rank_text)
        if not rank_match or rank_match.group(1) != "1":
            continue
        rider_cell = row[1]
        team_cell = row[2] if len(row) >= 4 else ""
        return {
            "name": _extract_rider_name(rider_cell),
            "country": _extract_country_from_rider_cell(rider_cell),
            "team": _clean_wiki_value(team_cell) if len(row) >= 4 else "",
        }
    return {}


def _parse_leadership_table_winners(text: str) -> tuple[dict[str, str], dict[str, str]]:
    table_text = _extract_table_after(text, [r"Classification leadership table"])
    if not table_text:
        return {}, {}
    final_match = re.search(r"!\s*colspan\s*=\s*2\s*\|\s*Final.*?(?=\n\|\}|$)", table_text, flags=re.I | re.S)
    if not final_match:
        return {}, {}
    lines = [line.strip() for line in final_match.group(0).splitlines() if line.strip().startswith(("|", "!"))]
    cells = [_parse_table_cell(line) for line in lines]
    cells = [cell for cell in cells if cell]
    if len(cells) < 5:
        return {}, {}
    points = {
        "name": _clean_wiki_value(cells[2]),
        "team": "",
        "country": "",
    }
    mountains = {
        "name": _clean_wiki_value(cells[3]),
        "team": "",
        "country": "",
    }
    return points, mountains


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
        if not text and not winner_seed:
            continue
        row = {field: "" for field in FIELDNAMES}
        row["year"] = str(year)
        row["card_bg_color"] = "#1f6f78"
        row["accent_color"] = "#d8c06a"
        row["badge_label"] = "PN #1"

        row["winner_name"] = winner_seed.get("winner_name", "") or _extract_infobox_field(text, "first")
        row["winner_team"] = winner_seed.get("winner_team", "") or _extract_infobox_field(text, "first_team")
        row["winner_country"] = winner_seed.get("winner_country", "") or _extract_infobox_field(text, "first_nat")

        top5 = MANUAL_GC_TOP5.get(year) or _parse_general_top5(text) or _parse_general_top5_wikitable(text)
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

        points = {
            "name": _extract_infobox_field(text, "points"),
            "team": _extract_infobox_field(text, "points_team"),
            "country": _extract_infobox_field(text, "points_nat"),
        }
        mountains = {
            "name": _extract_infobox_field(text, "mountains"),
            "team": _extract_infobox_field(text, "mountains_team"),
            "country": _extract_infobox_field(text, "mountains_nat"),
        }
        if not points["name"]:
            points = _parse_single_leader_from_table(
                text,
                [
                    r"Final points classification(?:\s*\(1-10\))?",
                    r"=+\s*(?:\[\[)?Points classification(?:\|[^]]+)?(?:\]\])?\s*=+",
                    r"==+\s*Points classification\s*==+",
                ],
            ) or points
        if not mountains["name"]:
            mountains = _parse_single_leader_from_table(
                text,
                [
                    r"Final mountains classification(?:\s*\(1-10\))?",
                    r"=+\s*(?:\[\[)?(?:King of the Mountains\|)?Mountains classification(?:\]\])?\s*=+",
                    r"==+\s*Mountains classification\s*==+",
                ],
            ) or mountains
        if not points["name"] or not mountains["name"]:
            leadership_points, leadership_mountains = _parse_leadership_table_winners(text)
            if not points["name"]:
                points = leadership_points or points
            if not mountains["name"]:
                mountains = leadership_mountains or mountains

        row["points_name"] = points.get("name", "")
        row["points_team"] = points.get("team", "")
        row["points_country"] = points.get("country", "")
        row["mountains_name"] = mountains.get("name", "")
        row["mountains_team"] = mountains.get("team", "")
        row["mountains_country"] = mountains.get("country", "")

        if not row["points_name"] and year in SUPPLEMENTAL_POINTS_WINNERS:
            row["points_name"] = SUPPLEMENTAL_POINTS_WINNERS[year]
        if not row["mountains_name"] and year in SUPPLEMENTAL_MOUNTAINS_WINNERS:
            row["mountains_name"] = SUPPLEMENTAL_MOUNTAINS_WINNERS[year]

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
