from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_CSV = PROJECT_ROOT / "data" / "raw" / "france_kings_reigns.csv"
OUTPUT_PNG = PROJECT_ROOT / "data" / "processed" / "france_kings_timeline_preview.png"


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def main() -> None:
    rows = load_rows(INPUT_CSV)
    fig, ax = plt.subplots(figsize=(22, 11), dpi=150)
    ax.set_facecolor("#f5efe4")
    fig.patch.set_facecolor("#f5efe4")

    for index, row in enumerate(rows):
        start_year = int(row["start_year"])
        end_year = int(row["end_year"])
        width = end_year - start_year + 1
        y = len(rows) - index - 1
        ax.barh(y, width, left=start_year, height=0.86, color=row["house_color"], edgecolor="white", linewidth=1.25)
        label = f"{row['display_name']} ({start_year}-{end_year})"
        ax.text(start_year + 0.35, y, label, va="center", ha="left", fontsize=8.5, color="white", fontweight="bold")

    ax.set_title("Rois de France - timeline canonique 481-1848", fontsize=22, fontweight="bold", pad=18)
    ax.text(481, len(rows) + 1.2, "Merovingiens, Carolingiens, Capetiens, Valois, Bourbons, Orleans", fontsize=12, color="#334e68")
    ax.set_xlabel("Annee", fontsize=12)
    ax.set_xlim(476, 1852)
    ax.set_yticks([])
    ax.grid(axis="x", linestyle="--", alpha=0.22)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)

    OUTPUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(OUTPUT_PNG, bbox_inches="tight")
    plt.close(fig)
    print(f"[history] timeline preview generated -> {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
