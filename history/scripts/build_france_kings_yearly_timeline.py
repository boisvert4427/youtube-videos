from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_CSV = PROJECT_ROOT / "data" / "raw" / "france_kings_reigns.csv"
OUTPUT_CSV = PROJECT_ROOT / "data" / "processed" / "france_kings_yearly_timeline.csv"

YEAR_START = 481
YEAR_END = 1848
GAP_YEARS = {
    (738, 742): ("No king", "No king", "Transition", "#6c757d", "", "Interregnum before Childeric III"),
    (841, 842): ("No king", "No king", "Transition", "#6c757d", "", "Partition struggle after Louis I"),
    (1793, 1813): ("No king", "No king", "Revolution and Empire", "#6c757d", "", "Monarchy interrupted"),
}


def load_reigns(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def build_yearly_rows(reigns: list[dict[str, str]]) -> list[dict[str, str]]:
    by_year: defaultdict[int, list[dict[str, str]]] = defaultdict(list)
    for reign in reigns:
        start_year = int(reign["start_year"])
        end_year = int(reign["end_year"])
        for year in range(start_year, end_year + 1):
            by_year[year].append(reign)

    for (start_year, end_year), gap_row in GAP_YEARS.items():
        for year in range(start_year, end_year + 1):
            if year not in by_year:
                by_year[year].append(
                    {
                        "start_year": str(start_year),
                        "end_year": str(end_year),
                        "ruler_name": gap_row[0],
                        "display_name": gap_row[1],
                        "dynasty": gap_row[2],
                        "house_color": gap_row[3],
                        "wiki_title": gap_row[4],
                        "notes": gap_row[5],
                    }
                )

    rows: list[dict[str, str]] = []
    for year in range(YEAR_START, YEAR_END + 1):
        rulers = sorted(by_year.get(year, []), key=lambda item: (int(item["start_year"]), item["display_name"]))
        if not rulers:
            rulers = [
                {
                    "start_year": str(year),
                    "end_year": str(year),
                    "ruler_name": "Unknown",
                    "display_name": "Unknown",
                    "dynasty": "Unknown",
                    "house_color": "#6c757d",
                    "wiki_title": "",
                    "notes": "",
                }
            ]
        rows.append(
            {
                "year": str(year),
                "ruler_count": str(len(rulers)),
                "ruler_name": " | ".join(ruler["ruler_name"] for ruler in rulers),
                "display_name": " & ".join(ruler["display_name"] for ruler in rulers),
                "dynasty": " & ".join(dict.fromkeys(ruler["dynasty"] for ruler in rulers)),
                "house_color": " & ".join(dict.fromkeys(ruler["house_color"] for ruler in rulers)),
                "start_year": " & ".join(dict.fromkeys(ruler["start_year"] for ruler in rulers)),
                "end_year": " & ".join(dict.fromkeys(ruler["end_year"] for ruler in rulers)),
                "wiki_title": " | ".join(filter(None, dict.fromkeys(ruler["wiki_title"] for ruler in rulers))),
                "notes": " | ".join(filter(None, dict.fromkeys(ruler["notes"] for ruler in rulers))),
                "fait_1": " | ".join(filter(None, dict.fromkeys(ruler.get("fait_1", "") for ruler in rulers))),
                "fait_2": " | ".join(filter(None, dict.fromkeys(ruler.get("fait_2", "") for ruler in rulers))),
                "fait_3": " | ".join(filter(None, dict.fromkeys(ruler.get("fait_3", "") for ruler in rulers))),
            }
        )
    return rows


def write_rows(rows: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "year",
                "ruler_count",
                "ruler_name",
                "display_name",
                "dynasty",
                "house_color",
                "start_year",
                "end_year",
                "wiki_title",
                "notes",
                "fait_1",
                "fait_2",
                "fait_3",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    reigns = load_reigns(INPUT_CSV)
    rows = build_yearly_rows(reigns)
    write_rows(rows, OUTPUT_CSV)
    print(f"[history] yearly timeline generated -> {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
