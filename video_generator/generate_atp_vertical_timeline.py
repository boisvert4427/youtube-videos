from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import time
import unicodedata
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation, writers
from matplotlib.patches import FancyBboxPatch, Rectangle


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = PROJECT_ROOT / "data" / "processed" / "atp_vertical_timeline_v1.sample.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "atp_vertical_timeline.mp4"
DEFAULT_TOP_N = 5
DEFAULT_PHOTOS_DIR = PROJECT_ROOT / "data" / "raw" / "player_photos"


@dataclass(frozen=True)
class TimelineEntry:
    year: int
    player_name: str
    subtitle: str
    image_path: str
    name_bg_color: str
    card_bg_color: str
    rank_label: str
    results: list[str]


def parse_results(value: str) -> list[str]:
    if not value.strip():
        return []
    chunks = [part.strip() for part in value.split("|")]
    return [c for c in chunks if c]


def _slugify(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", normalized.strip().lower())
    return cleaned.strip("_")


def _extract_result_parts(line: str) -> tuple[str, str, str]:
    text = line.strip()
    if not text:
        return ("", "", "")
    tokens = text.split()
    rnd = tokens[0] if tokens else ""
    rest = " ".join(tokens[1:]) if len(tokens) > 1 else ""
    if rnd not in {"R128", "R64", "R32", "R16", "QF", "SF", "F"}:
        return ("", text, "")

    if not rest:
        return (rnd, "", "")

    parts = rest.split()
    score_parts: list[str] = []
    while parts:
        t = parts[-1]
        if any(ch.isdigit() for ch in t) or "-" in t or "(" in t or ")" in t:
            score_parts.insert(0, parts.pop())
            continue
        break
    opponent = " ".join(parts).strip()
    score = " ".join(score_parts).strip()
    if not opponent and score:
        return (rnd, rest, "")
    return (rnd, opponent or rest, score)


def _normalize_parcours_rows(results: list[str]) -> list[str]:
    rounds = ["R128", "R64", "R32", "R16", "QF", "SF", "F"]
    mapped: dict[str, str] = {}
    extra: list[str] = []
    for line in results:
        rnd, opponent, score = _extract_result_parts(line)
        if rnd:
            payload = opponent
            if score:
                payload = f"{payload} {score}".strip()
            mapped[rnd] = payload.strip() or "-"
        elif line.strip():
            extra.append(line.strip())

    normalized: list[str] = []
    for rnd in rounds:
        value = mapped.get(rnd, "-")
        normalized.append(f"{rnd} {value}".strip())

    if extra:
        normalized[-1] = f"F {extra[0]}"
    return normalized


def _wiki_api(params: dict[str, str]) -> dict | None:
    query = urllib.parse.urlencode(params)
    url = f"https://en.wikipedia.org/w/api.php?{query}"
    req = urllib.request.Request(url, headers={"User-Agent": "youtube-videos-local/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=25) as response:
            return json.loads(response.read().decode("utf-8"))
    except Exception:
        return None


def _find_wikipedia_title(player_name: str) -> str | None:
    direct_candidates = [
        player_name,
        f"{player_name} (tennis)",
        f"{player_name} (tennis player)",
    ]
    if player_name.endswith(" Sr."):
        base = player_name.replace(" Sr.", "").strip()
        direct_candidates.extend([base, f"{base} (tennis)", f"{base} (tennis player)"])

    for candidate in direct_candidates:
        data = _wiki_api(
            {
                "action": "query",
                "format": "json",
                "titles": candidate,
            }
        )
        if not data:
            continue
        pages = data.get("query", {}).get("pages", {})
        for page in pages.values():
            if "missing" not in page and page.get("title"):
                return str(page["title"])

    for q in (f"{player_name} tennis", player_name):
        data = _wiki_api(
            {
                "action": "opensearch",
                "search": q,
                "limit": "5",
                "namespace": "0",
                "format": "json",
            }
        )
        if not isinstance(data, list) or len(data) < 2:
            continue
        titles = data[1]
        if not titles:
            continue
        preferred = None
        for title in titles:
            lower = str(title).lower()
            if "(tennis)" in lower or "tennis player" in lower:
                preferred = str(title)
                break
        return preferred or str(titles[0])
    return None


def _find_wiki_image_url(title: str) -> str | None:
    thumb_data = _wiki_api(
        {
            "action": "query",
            "format": "json",
            "prop": "pageimages",
            "piprop": "thumbnail",
            "pithumbsize": "1200",
            "titles": title,
        }
    )
    if thumb_data:
        pages = thumb_data.get("query", {}).get("pages", {})
        for page in pages.values():
            thumbnail = page.get("thumbnail")
            if thumbnail and "source" in thumbnail:
                return str(thumbnail["source"])

    summary_url = "https://en.wikipedia.org/api/rest_v1/page/summary/" + urllib.parse.quote(title.replace(" ", "_"))
    req = urllib.request.Request(summary_url, headers={"User-Agent": "youtube-videos-local/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=25) as response:
            summary = json.loads(response.read().decode("utf-8"))
        thumb = summary.get("thumbnail", {}).get("source")
        if thumb:
            return str(thumb)
        original = summary.get("originalimage", {}).get("source")
        if original:
            return str(original)
    except Exception:
        pass

    data = _wiki_api(
        {
            "action": "query",
            "format": "json",
            "prop": "pageimages",
            "piprop": "original",
            "titles": title,
        }
    )
    if data:
        pages = data.get("query", {}).get("pages", {})
        for page in pages.values():
            original = page.get("original")
            if original and "source" in original:
                return str(original["source"])
    return None


def _download_file(url: str, dest: Path) -> bool:
    req = urllib.request.Request(url, headers={"User-Agent": "youtube-videos-local/1.0"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=30) as response:
                data = response.read()
            dest.write_bytes(data)
            return True
        except Exception:
            if attempt < 2:
                time.sleep(0.6 * (attempt + 1))
    return False


def _guess_extension(image_url: str) -> str:
    lower = image_url.lower()
    if ".png" in lower:
        return ".png"
    if ".webp" in lower:
        return ".webp"
    return ".jpg"


def _to_wikimedia_thumb_url(image_url: str, size: int = 640) -> str | None:
    marker = "/wikipedia/commons/"
    if marker not in image_url or "/wikipedia/commons/thumb/" in image_url:
        return None
    base, _, tail = image_url.partition(marker)
    parts = tail.split("/")
    if len(parts) < 3:
        return None
    filename = parts[-1]
    thumb_tail = "thumb/" + "/".join(parts) + f"/{size}px-{filename}"
    return base + marker + thumb_tail


def _resolve_or_fetch_image_path(
    entry: TimelineEntry,
    photos_dir: Path,
    auto_fetch_images: bool,
    fetch_cache: dict[str, Path | None],
) -> Path | None:
    if entry.image_path:
        direct = PROJECT_ROOT / entry.image_path
        if direct.exists():
            return direct
        return None

    slug = _slugify(entry.player_name)
    for ext in (".jpg", ".jpeg", ".png", ".webp"):
        candidate = photos_dir / f"{slug}{ext}"
        if candidate.exists():
            return candidate

    if not auto_fetch_images:
        return None

    if entry.player_name in fetch_cache:
        return fetch_cache[entry.player_name]

    photos_dir.mkdir(parents=True, exist_ok=True)
    title = _find_wikipedia_title(entry.player_name)
    if not title:
        fetch_cache[entry.player_name] = None
        return None
    image_url = _find_wiki_image_url(title)
    if not image_url:
        fetch_cache[entry.player_name] = None
        return None

    dest = photos_dir / f"{slug}{_guess_extension(image_url)}"
    if _download_file(image_url, dest):
        fetch_cache[entry.player_name] = dest
        return dest

    alt_url = _to_wikimedia_thumb_url(image_url, size=640)
    if alt_url and _download_file(alt_url, dest):
        fetch_cache[entry.player_name] = dest
        return dest

    fetch_cache[entry.player_name] = None
    return None


def load_entries(input_csv: Path) -> list[TimelineEntry]:
    entries: list[TimelineEntry] = []
    with input_csv.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            entries.append(
                TimelineEntry(
                    year=int(row["year"]),
                    player_name=row["player_name"].strip(),
                    subtitle=row.get("subtitle", "").strip(),
                    image_path=row.get("image_path", "").strip(),
                    name_bg_color=row.get("name_bg_color", "#f4df26").strip() or "#f4df26",
                    card_bg_color=row.get("card_bg_color", "#5f3518").strip() or "#5f3518",
                    rank_label=row.get("rank_label", "RG #1").strip() or "RG #1",
                    results=parse_results(row.get("results", "")),
                )
            )
    entries.sort(key=lambda e: e.year)
    return entries


def _read_image(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    try:
        return plt.imread(str(path))
    except Exception:
        return None


def _draw_results_table(ax: plt.Axes, entry: TimelineEntry, x: float, y: float, w: float, h: float) -> None:
    panel_x = x + w * 0.02
    panel_y = y + h * 0.015
    panel_w = w * 0.96
    panel_h = h * 0.285
    ax.add_patch(
        FancyBboxPatch(
            (panel_x, panel_y),
            panel_w,
            panel_h,
            boxstyle="round,pad=0.004,rounding_size=0.01",
            facecolor="#2a1509",
            edgecolor="#b06d30",
            linewidth=1.0,
            zorder=5,
        )
    )

    rows = _normalize_parcours_rows(entry.results)[:7]

    row_h = panel_h / 7.0
    rnd_x = panel_x + panel_w * 0.03
    opp_x = panel_x + panel_w * 0.16
    score_x = panel_x + panel_w * 0.98
    for i in range(8):
        y0 = panel_y + panel_h - (i + 1) * row_h
        ax.plot([panel_x, panel_x + panel_w], [y0, y0], color="#8b5124", linewidth=0.7, zorder=6, alpha=0.8)

    for idx, line in enumerate(rows):
        rnd, opponent, score = _extract_result_parts(line)
        y_center = panel_y + panel_h - (idx + 0.5) * row_h
        if rnd:
            ax.text(
                rnd_x,
                y_center,
                rnd,
                ha="left",
                va="center",
                fontsize=10.8,
                color="#f1c738",
                weight="bold",
                zorder=7,
            )
        ax.text(
            opp_x,
            y_center,
            opponent or line,
            ha="left",
            va="center",
            fontsize=10.2,
            color="#e6e6e6",
            zorder=7,
        )
        if score:
            ax.text(
                score_x,
                y_center,
                score,
                ha="right",
                va="center",
                fontsize=10.1,
                color="#9de37a",
                zorder=7,
            )


def _draw_card(
    ax: plt.Axes,
    entry: TimelineEntry,
    x: float,
    y: float,
    w: float,
    h: float,
    image_cache: dict[str, np.ndarray | None],
    resolved_image_path: Path | None,
) -> None:
    card = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.005,rounding_size=0.018",
        facecolor=entry.card_bg_color,
        edgecolor="#a36836",
        linewidth=2.1,
        zorder=2,
    )
    ax.add_patch(card)

    year_w = w * 0.36
    year_h = h * 0.075
    year_x = x + w * 0.03
    year_y = y + h * 0.90
    ax.add_patch(
        FancyBboxPatch(
            (year_x, year_y),
            year_w,
            year_h,
            boxstyle="round,pad=0.01,rounding_size=0.012",
            facecolor="black",
            edgecolor="black",
            linewidth=1.0,
            zorder=5,
        )
    )
    ax.text(
        year_x + year_w / 2,
        year_y + year_h / 2,
        f"{entry.year}",
        ha="center",
        va="center",
        fontsize=22,
        color="#f7dd2d",
        weight="bold",
        zorder=6,
    )

    img_x = x + w * 0.02
    img_y = y + h * 0.50
    img_w = w * 0.96
    img_h = h * 0.42

    image = None
    if resolved_image_path:
        cache_key = str(resolved_image_path.resolve())
        if cache_key not in image_cache:
            image_cache[cache_key] = _read_image(resolved_image_path)
        image = image_cache[cache_key]

    if image is not None:
        ax.imshow(image, extent=[img_x, img_x + img_w, img_y, img_y + img_h], aspect="auto", zorder=3)
    else:
        ax.add_patch(
            Rectangle((img_x, img_y), img_w, img_h, facecolor="#cfcfcf", edgecolor="none", zorder=3)
        )
        ax.text(
            img_x + img_w / 2,
            img_y + img_h / 2,
            entry.player_name,
            ha="center",
            va="center",
            fontsize=10,
            color="#212121",
            weight="bold",
            zorder=4,
        )

    name_h = h * 0.12
    name_y = y + h * 0.38
    ax.add_patch(
        Rectangle((img_x, name_y), img_w, name_h, facecolor=entry.name_bg_color, edgecolor="none", zorder=4)
    )
    ax.text(
        x + w / 2,
        name_y + name_h / 2,
        entry.player_name,
        ha="center",
        va="center",
        fontsize=20,
        color="#101010",
        weight="bold",
        zorder=6,
    )

    rank_w = w * 0.76
    rank_h = h * 0.065
    rank_x = x + (w - rank_w) / 2
    rank_y = y + h * 0.30
    ax.add_patch(
        FancyBboxPatch(
            (rank_x, rank_y),
            rank_w,
            rank_h,
            boxstyle="round,pad=0.008,rounding_size=0.012",
            facecolor="black",
            edgecolor="#f7dd2d",
            linewidth=1.2,
            zorder=5,
        )
    )
    ax.text(
        x + w / 2,
        rank_y + rank_h / 2,
        entry.rank_label,
        ha="center",
        va="center",
        fontsize=18,
        color="#f7dd2d",
        weight="bold",
        zorder=6,
    )

    subtitle = entry.subtitle.strip()
    if subtitle:
        ax.text(
            x + w / 2,
            y + h * 0.274,
            subtitle,
            ha="center",
            va="center",
            fontsize=10.0,
            color="#d8d8d8",
            zorder=6,
        )

    _draw_results_table(ax, entry, x, y, w, h)


def _layout_params(cards_visible: int) -> tuple[float, float, float, float]:
    usable_width = 1.0
    left = 0.0
    gap = 0.004
    cards_visible = max(3, cards_visible)
    card_w = (usable_width - gap * (cards_visible - 1)) / cards_visible
    pitch = card_w + gap
    return left, card_w, gap, pitch


def render_timeline_video(
    entries: list[TimelineEntry],
    output_path: Path,
    fps: int,
    frames_per_year: int,
    cards_visible: int,
    title: str,
    photos_dir: Path,
    auto_fetch_images: bool,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fetch_cache: dict[str, Path | None] = {}
    resolved_path_by_player: dict[str, Path | None] = {}
    for entry in entries:
        if entry.player_name in resolved_path_by_player:
            continue
        resolved_path_by_player[entry.player_name] = _resolve_or_fetch_image_path(
            entry=entry,
            photos_dir=photos_dir,
            auto_fetch_images=auto_fetch_images,
            fetch_cache=fetch_cache,
        )

    # 1920x1080, standard YouTube 16:9
    fig, ax = plt.subplots(figsize=(19.2, 10.8), dpi=100, facecolor="#1f1208")
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.set_position([0, 0, 1, 1])
    ax.set_facecolor("#1f1208")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    total_years = len(entries)
    if total_years == 1:
        total_frames = 1
    else:
        total_frames = (total_years - 1) * frames_per_year + 1

    x_start, card_w, gap, pitch = _layout_params(cards_visible)
    card_h = 0.99
    card_y = 0.005
    image_cache: dict[str, np.ndarray | None] = {}

    def update(frame_index: int) -> None:
        ax.clear()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.set_facecolor("#1f1208")

        base_index = frame_index // max(frames_per_year, 1)
        local_step = frame_index % max(frames_per_year, 1)
        progress = local_step / max(frames_per_year, 1)
        if frame_index == total_frames - 1:
            progress = 0.0
            base_index = total_years - 1

        shift = base_index + progress
        start_x = x_start - shift * pitch

        visible_cutoff_left = -0.05
        visible_cutoff_right = 1.02
        for i, entry in enumerate(entries):
            x = start_x + i * pitch
            if x > visible_cutoff_right or x + card_w < visible_cutoff_left:
                continue
            resolved_image_path = resolved_path_by_player.get(entry.player_name)
            _draw_card(ax, entry, x, card_y, card_w, card_h, image_cache, resolved_image_path)

        _ = title  # kept for CLI compatibility

    animation = FuncAnimation(fig, update, frames=total_frames, interval=1000 / fps, repeat=False)
    ffmpeg_available = bool(shutil.which("ffmpeg")) and writers.is_available("ffmpeg")
    if output_path.suffix.lower() != ".mp4":
        output_path = output_path.with_suffix(".mp4")
    if not ffmpeg_available:
        raise RuntimeError("ffmpeg is required for MP4 output but is not available.")
    writer = FFMpegWriter(fps=fps, bitrate=3000)
    animation.save(output_path, writer=writer)
    plt.close(fig)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate ATP vertical timeline video from CSV")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="CSV path")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output .mp4 path")
    parser.add_argument("--photos-dir", type=Path, default=DEFAULT_PHOTOS_DIR, help="Folder for player photos")
    parser.add_argument("--auto-fetch-images", action="store_true", help="Auto-download missing player photos")
    parser.add_argument("--fps", type=int, default=30, help="Video fps")
    parser.add_argument("--frames-per-year", type=int, default=24, help="Animation frames between years")
    parser.add_argument("--cards-visible", type=int, default=4, help="Number of cards visible on screen")
    parser.add_argument("--title", type=str, default="Roland-Garros Champions Timeline", help="Video title")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    entries = load_entries(args.input)
    if not entries:
        raise RuntimeError(f"No data found in {args.input}")

    output = render_timeline_video(
        entries=entries,
        output_path=args.output,
        fps=args.fps,
        frames_per_year=args.frames_per_year,
        cards_visible=args.cards_visible,
        title=args.title,
        photos_dir=args.photos_dir,
        auto_fetch_images=args.auto_fetch_images,
    )
    print(f"[video_generator] vertical timeline generated -> {output}")


if __name__ == "__main__":
    main()
