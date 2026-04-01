from __future__ import annotations

import argparse
import csv
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import re
import urllib.request

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patheffects
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter, writers
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Rectangle


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = PROJECT_ROOT / "data" / "processed" / "atp_ranking_timeseries_v1.sample.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "atp_ranking_race.mp4"
DEFAULT_PHOTOS_DIR = PROJECT_ROOT / "data" / "raw" / "player_photos"
DEFAULT_FLAGS_DIR = PROJECT_ROOT / "data" / "raw" / "flags"


@dataclass(frozen=True)
class Snapshot:
    ranking_date: datetime
    points_by_player: dict[str, int]
    country_by_player: dict[str, str]


ALPHA3_TO_ALPHA2 = {
    "ARG": "AR",
    "AUS": "AU",
    "AUT": "AT",
    "BEL": "BE",
    "BGR": "BG",
    "BRA": "BR",
    "CAN": "CA",
    "CHI": "CL",
    "CRO": "HR",
    "CZE": "CZ",
    "DEN": "DK",
    "ESP": "ES",
    "FRA": "FR",
    "GBR": "GB",
    "GER": "DE",
    "GRE": "GR",
    "ITA": "IT",
    "JPN": "JP",
    "MEX": "MX",
    "NED": "NL",
    "NOR": "NO",
    "POL": "PL",
    "POR": "PT",
    "ROU": "RO",
    "RUS": "RU",
    "SRB": "RS",
    "SUI": "CH",
    "SWE": "SE",
    "UKR": "UA",
    "USA": "US",
    "ZAF": "ZA",
}

EMOJI_FONT_PATH = Path("C:/Windows/Fonts/seguiemj.ttf")
EMOJI_FONT = FontProperties(fname=str(EMOJI_FONT_PATH)) if EMOJI_FONT_PATH.exists() else None
TITLE_FONT_PATH = Path("C:/Windows/Fonts/arialbd.ttf")
LABEL_FONT_PATH = Path("C:/Windows/Fonts/arialbd.ttf")
TITLE_FONT = FontProperties(fname=str(TITLE_FONT_PATH)) if TITLE_FONT_PATH.exists() else None
LABEL_FONT = FontProperties(fname=str(LABEL_FONT_PATH)) if LABEL_FONT_PATH.exists() else None


def country_to_flag(country_code: str) -> str:
    code = country_code.strip().upper()
    if not code:
        return ""

    if len(code) == 2 and code.isalpha():
        alpha2 = code
    elif len(code) == 3 and code.isalpha():
        alpha2 = ALPHA3_TO_ALPHA2.get(code, "")
    else:
        alpha2 = ""

    if len(alpha2) != 2:
        return ""
    offset = 127397
    return chr(ord(alpha2[0]) + offset) + chr(ord(alpha2[1]) + offset)


def to_alpha2(country_code: str) -> str:
    code = country_code.strip().upper()
    if len(code) == 2 and code.isalpha():
        return code
    if len(code) == 3 and code.isalpha():
        return ALPHA3_TO_ALPHA2.get(code, "")
    return ""


def _download_flag_png(alpha2: str, out_path: Path) -> bool:
    # 64px wide PNG gives clean rendering even after downscaling in chart.
    url = f"https://flagcdn.com/w80/{alpha2.lower()}.png"
    req = urllib.request.Request(url, headers={"User-Agent": "youtube-videos-local/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            data = response.read()
        out_path.write_bytes(data)
        return True
    except Exception:
        return False


def _apply_gradient_background(fig: plt.Figure) -> None:
    bg = fig.add_axes([0, 0, 1, 1], zorder=0)
    bg.axis("off")

    width, height = 1600, 900
    x = np.linspace(0, 1, width)
    y = np.linspace(0, 1, height)
    xx, yy = np.meshgrid(x, y)
    base = 0.20 + 0.80 * (0.62 * (1 - xx) + 0.38 * (1 - yy))
    vignette = 1 - 0.17 * ((xx - 0.50) ** 2 + (yy - 0.53) ** 2)
    spotlight = np.exp(-(((xx - 0.30) / 0.42) ** 2 + ((yy - 0.50) / 0.58) ** 2))
    lum = np.clip(base * vignette + 0.08 * spotlight, 0, 1)

    dark = np.array([18, 36, 62]) / 255.0
    light = np.array([46, 78, 112]) / 255.0
    img = dark + (light - dark)[None, None, :] * lum[..., None]
    bg.imshow(img, aspect="auto", interpolation="bicubic")


def load_snapshots(input_csv: Path) -> list[Snapshot]:
    raw: dict[str, dict[str, int]] = {}
    countries: dict[str, str] = {}

    with input_csv.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            ranking_date = row["ranking_date"].strip()
            player = row["player_name"].strip()
            country = row.get("country_code", "").strip().upper()
            points = int(row["points"])

            raw.setdefault(ranking_date, {})[player] = points
            countries[player] = country

    snapshots: list[Snapshot] = []
    for date_str in sorted(raw.keys()):
        snapshots.append(
            Snapshot(
                ranking_date=datetime.strptime(date_str, "%Y-%m-%d"),
                points_by_player=raw[date_str],
                country_by_player=countries.copy(),
            )
        )
    return snapshots


def interpolate_snapshots(snapshots: list[Snapshot], frames_per_period: int) -> list[tuple[datetime, dict[str, float], dict[str, str]]]:
    if len(snapshots) < 2:
        only = snapshots[0]
        return [(only.ranking_date, {k: float(v) for k, v in only.points_by_player.items()}, only.country_by_player)]

    frames: list[tuple[datetime, dict[str, float], dict[str, str]]] = []
    for i in range(len(snapshots) - 1):
        current = snapshots[i]
        nxt = snapshots[i + 1]
        players = sorted(set(current.points_by_player) | set(nxt.points_by_player))

        for step in range(frames_per_period):
            t = step / frames_per_period
            interpolated: dict[str, float] = {}
            for player in players:
                start = float(current.points_by_player.get(player, 0))
                end = float(nxt.points_by_player.get(player, 0))
                interpolated[player] = start + (end - start) * t
            frames.append((current.ranking_date, interpolated, current.country_by_player))

    last = snapshots[-1]
    frames.append((last.ranking_date, {k: float(v) for k, v in last.points_by_player.items()}, last.country_by_player))
    return frames


def _slugify_player_name(name: str) -> str:
    lowered = name.strip().lower()
    cleaned = re.sub(r"[^a-z0-9]+", "_", lowered)
    return cleaned.strip("_")


def _find_player_photo(player_name: str, photos_dir: Path) -> Path | None:
    if not photos_dir.exists():
        return None

    slug = _slugify_player_name(player_name)
    candidates = [
        photos_dir / f"{slug}.jpg",
        photos_dir / f"{slug}.jpeg",
        photos_dir / f"{slug}.png",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _nice_axis_max(current_max: float) -> float:
    if current_max <= 0:
        return 10.0
    if current_max <= 20:
        return float(max(6, int(np.ceil(current_max)) + 1))
    if current_max <= 100:
        return float(int(np.ceil(current_max / 10.0) * 10))
    if current_max <= 1000:
        return float(int(np.ceil(current_max / 100.0) * 100))
    rounded = int(np.ceil(current_max / 1000.0) * 1000)
    return float(max(rounded, 1000))


def render_video(
    frames: list[tuple[datetime, dict[str, float], dict[str, str]]],
    output_path: Path,
    top_n: int,
    title: str,
    fps: int,
    photos_dir: Path,
    flags_dir: Path,
    year_in_media_box: bool = False,
    leader_logo_in_header: bool = False,
    show_bottom_date: bool = True,
    position_lerp: float = 0.36,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(13.66, 7.02), dpi=100, facecolor="#0b1930")
    ax.set_position([0.045, 0.07, 0.92, 0.84])
    photo_ax = fig.add_axes([0.825, 0.18, 0.10, 0.265], zorder=8)
    logo_ax = fig.add_axes([0.86, 0.84, 0.08, 0.11], zorder=9)
    if year_in_media_box and leader_logo_in_header:
        logo_ax.set_position([0.83, 0.48, 0.09, 0.13])
    _apply_gradient_background(fig)
    ax.set_zorder(2)
    ax.set_facecolor((0, 0, 0, 0))
    photo_ax.set_facecolor((0, 0, 0, 0))
    logo_ax.set_facecolor((0, 0, 0, 0))

    palette = [
        "#f0bfd5",
        "#cfc3e3",
        "#9fb6d6",
        "#8ec5d6",
        "#f5bc6a",
        "#2ba63e",
        "#f29f96",
        "#ef3526",
        "#98d18a",
        "#86c9e1",
        "#ef9e97",
        "#e7b0ca",
        "#8f6152",
        "#8f6152",
        "#2ca13a",
    ]
    photo_cache: dict[str, np.ndarray] = {}
    flag_img_cache: dict[str, np.ndarray] = {}
    display_y_by_player: dict[str, float] = {}
    display_axis_max: float | None = None
    flags_dir.mkdir(parents=True, exist_ok=True)

    global_peak = max((max(points.values()) if points else 0.0) for _, points, _ in frames)
    global_axis_cap = _nice_axis_max(float(global_peak))

    def update(frame_index: int) -> None:
        ax.clear()
        photo_ax.clear()
        logo_ax.clear()
        frame_date, points, countries = frames[frame_index]
        ranked = sorted(points.items(), key=lambda x: x[1], reverse=True)[:top_n]

        entries: list[dict] = []

        target_y_by_player = {player: float(top_n - 1 - idx) for idx, (player, _) in enumerate(ranked)}
        for idx, (player, value) in enumerate(ranked):
            country_code = countries.get(player, "")
            alpha2 = to_alpha2(country_code)
            label = player.strip()
            target_y = target_y_by_player[player]
            previous_y = display_y_by_player.get(player, target_y)
            current_y = previous_y + (target_y - previous_y) * position_lerp
            display_y_by_player[player] = current_y
            entries.append(
                {
                    "rank": idx + 1,
                    "player": player,
                    "label": label,
                    "alpha2": alpha2,
                    "value": float(value),
                    "color": palette[idx % len(palette)],
                    "y": current_y,
                }
            )

        for stale_player in list(display_y_by_player.keys()):
            if stale_player not in target_y_by_player:
                del display_y_by_player[stale_player]

        bars_by_player: dict[str, object] = {}
        draw_order = sorted(entries, key=lambda item: item["y"])  # draw from bottom to top
        for item in draw_order:
            bar = ax.barh(
                item["y"],
                item["value"],
                color=item["color"],
                height=0.78,
                edgecolor=(0, 0, 0, 0),
                zorder=3 + (item["y"] * 0.01),
            )[0]
            bars_by_player[item["player"]] = bar

        values = [item["value"] for item in entries]
        max_val = max(values) if values else 100.0
        target_axis_max = _nice_axis_max(max_val)
        nonlocal display_axis_max
        if display_axis_max is None:
            display_axis_max = target_axis_max
        else:
            # Smooth scale transitions to avoid abrupt zoom jumps.
            display_axis_max = display_axis_max + (target_axis_max - display_axis_max) * 0.22
        x_axis_max = max(display_axis_max, 1.0)
        small_scale = x_axis_max <= 200
        if small_scale:
            x_right = x_axis_max + max(0.9, x_axis_max * 0.10)
            x_left = -max(1.8, x_axis_max * 0.22)
        else:
            x_right = x_axis_max + max(260.0, x_axis_max * 0.035)
            x_left = -max(180.0, x_axis_max * 0.02)
        ax.set_xlim(x_left, x_right)
        ax.set_ylim(-0.6, len(entries) - 0.1)

        ax.xaxis.set_ticks_position("top")
        ax.xaxis.set_label_position("top")
        if global_axis_cap <= 20:
            tick_step = 1
        elif global_axis_cap <= 50:
            tick_step = 5
        elif global_axis_cap <= 100:
            tick_step = 10
        elif global_axis_cap <= 500:
            tick_step = 50
        elif global_axis_cap <= 2000:
            tick_step = 200
        elif global_axis_cap <= 4000:
            tick_step = 500
        else:
            tick_step = 1000
        ax.set_xticks(np.arange(0, global_axis_cap + 1, tick_step))
        ax.tick_params(axis="x", colors="#7f90aa", labelsize=10, length=0, pad=8)
        ax.set_yticks([])
        ax.grid(axis="x", alpha=0.08, color="#87a0bd", linewidth=0.9)
        ax.set_axisbelow(True)

        # Keep a clear fixed gap between rank badges and the start of bars (x=0).
        if small_scale:
            rank_circle_x = -max(0.9, x_axis_max * 0.12)
            label_start_x = x_axis_max * 0.03
        else:
            rank_circle_x = -max(90.0, x_axis_max * 0.012)
            label_start_x = 34
        for item in sorted(entries, key=lambda e: e["rank"]):
            idx = int(item["rank"])
            bar = bars_by_player[item["player"]]
            val = float(item["value"])
            label = str(item["label"])
            alpha2 = str(item["alpha2"])
            x = bar.get_width()
            y = bar.get_y() + bar.get_height() / 2

            ax.scatter(
                [rank_circle_x],
                [y],
                s=260,
                c="#f6c445",
                edgecolors="#f8d46d",
                linewidths=1.3,
                zorder=5,
                clip_on=False,
            )
            rank_text = ax.text(
                rank_circle_x,
                y,
                f"{idx}",
                ha="center",
                va="center",
                fontsize=12,
                color="#172033",
                fontweight="bold",
                zorder=6,
            )
            rank_text.set_clip_on(False)

            flag_x = label_start_x
            flag_width = max(max_val * 0.026, 0.22 if small_scale else 0.0)
            label_gap = max(max_val * 0.008, 0.08 if small_scale else 10.0)
            flag_loaded = False
            if alpha2:
                flag_path = flags_dir / f"{alpha2.lower()}.png"
                if not flag_path.exists():
                    _download_flag_png(alpha2, flag_path)
                if flag_path.exists():
                    cache_key = str(flag_path.resolve())
                    flag_img = flag_img_cache.get(cache_key)
                    if flag_img is None:
                        flag_img = plt.imread(str(flag_path))
                        flag_img_cache[cache_key] = flag_img
                    fh = 0.42
                    ax.imshow(
                        flag_img,
                        extent=[flag_x, flag_x + flag_width, y - fh / 2, y + fh / 2],
                        aspect="auto",
                        zorder=6,
                    )
                    flag_loaded = True
            if not flag_loaded and alpha2:
                fallback = ax.text(
                    flag_x,
                    y,
                    alpha2,
                    va="center",
                    ha="left",
                    fontsize=11,
                    color="#e8edf5",
                    zorder=6,
                    fontproperties=LABEL_FONT,
                )
                fallback.set_path_effects([patheffects.withStroke(linewidth=2.0, foreground="#1c283d")])

            label_artist = ax.text(
                flag_x + flag_width + label_gap,
                y,
                label,
                va="center",
                ha="left",
                fontsize=11.5 if small_scale else 13.5,
                color="#e8edf5",
                fontweight="bold",
                zorder=6,
                fontproperties=LABEL_FONT,
            )
            label_artist.set_path_effects([patheffects.withStroke(linewidth=1.6, foreground="#1c283d")])

            value_artist = ax.text(
                x + (max(max_val * 0.02, 0.12) if small_scale else max_val * 0.008),
                y,
                f"{int(round(val))}",
                va="center",
                ha="left",
                fontsize=13 if small_scale else 17,
                color="#e6ebf3",
                fontweight="bold",
                zorder=6,
                fontproperties=LABEL_FONT,
            )
            value_artist.set_path_effects([patheffects.withStroke(linewidth=1.8, foreground="#162238")])
            value_artist.set_clip_on(False)

        title_artist = ax.set_title(title, fontsize=27, color="#f2f5fa", fontweight="bold", pad=12, fontproperties=TITLE_FONT)
        title_artist.set_path_effects([patheffects.withStroke(linewidth=2.2, foreground="#162238")])
        for spine in ax.spines.values():
            spine.set_visible(False)

        ax.margins(x=0)

        photo_ax.set_xticks([])
        photo_ax.set_yticks([])
        photo_ax.set_xlim(0, 1)
        photo_ax.set_ylim(0, 1)
        for spine in photo_ax.spines.values():
            spine.set_visible(False)

        logo_ax.set_xticks([])
        logo_ax.set_yticks([])
        logo_ax.set_xlim(0, 1)
        logo_ax.set_ylim(0, 1)
        for spine in logo_ax.spines.values():
            spine.set_visible(False)

        if ranked:
            leader_name = ranked[0][0]
            photo_path = _find_player_photo(leader_name, photos_dir)
            if year_in_media_box:
                year_artist = photo_ax.text(
                    0.5,
                    0.5,
                    frame_date.strftime("%Y"),
                    ha="center",
                    va="center",
                    fontsize=38,
                    color="#edf1f7",
                    fontweight="bold",
                    zorder=6,
                    fontproperties=TITLE_FONT,
                )
                year_artist.set_path_effects([patheffects.withStroke(linewidth=2.8, foreground="#0f172a")])
            else:
                if photo_path:
                    cache_key = str(photo_path.resolve())
                    img = photo_cache.get(cache_key)
                    if img is None:
                        img = plt.imread(str(photo_path))
                        photo_cache[cache_key] = img
                    photo_ax.imshow(img, extent=[0, 1, 0, 1], zorder=5, aspect="auto")
                    photo_ax.add_patch(
                        Rectangle((0, 0), 1, 1, fill=False, edgecolor="#7e8ea4", linewidth=1.6, zorder=6)
                    )
                else:
                    photo_ax.add_patch(
                        Rectangle((0, 0), 1, 1, facecolor=(1, 1, 1, 0.05), edgecolor="#7e8ea4", linewidth=1.6, zorder=5)
                    )
                    fallback = photo_ax.text(
                        0.5,
                        0.5,
                        leader_name,
                        ha="center",
                        va="center",
                        color="#d6deea",
                        fontsize=10,
                        fontweight="bold",
                        zorder=6,
                        wrap=True,
                    )
                    fallback.set_path_effects([patheffects.withStroke(linewidth=2.2, foreground="#132136")])

            if leader_logo_in_header:
                if photo_path:
                    cache_key = str(photo_path.resolve())
                    logo_img = photo_cache.get(cache_key)
                    if logo_img is None:
                        logo_img = plt.imread(str(photo_path))
                        photo_cache[cache_key] = logo_img
                    logo_ax.imshow(logo_img, extent=[0, 1, 0, 1], zorder=6, aspect="auto")
                    logo_ax.add_patch(Rectangle((0, 0), 1, 1, fill=False, edgecolor="#7e8ea4", linewidth=1.4, zorder=7))
                else:
                    logo_ax.add_patch(
                        Rectangle((0, 0), 1, 1, facecolor=(1, 1, 1, 0.06), edgecolor="#7e8ea4", linewidth=1.4, zorder=6)
                    )
                    mini = logo_ax.text(
                        0.5,
                        0.5,
                        leader_name,
                        ha="center",
                        va="center",
                        fontsize=7.5,
                        color="#e8edf5",
                        fontweight="bold",
                        zorder=7,
                        wrap=True,
                    )
                    mini.set_path_effects([patheffects.withStroke(linewidth=1.5, foreground="#1c283d")])

        if show_bottom_date:
            date_text = frame_date.strftime("%Y-%m")
            date_artist = ax.text(
                0.998,
                0.052,
                date_text,
                transform=ax.transAxes,
                ha="right",
                va="center",
                fontsize=40,
                fontweight="bold",
                color="#edf1f7",
                alpha=0.94,
                zorder=7,
                fontproperties=TITLE_FONT,
            )
            date_artist.set_path_effects([patheffects.withStroke(linewidth=2.8, foreground="#0f172a")])

    animation = FuncAnimation(fig, update, frames=len(frames), interval=1000 / fps, repeat=False)

    writer = None
    ffmpeg_available = bool(shutil.which("ffmpeg")) and writers.is_available("ffmpeg")
    if output_path.suffix.lower() == ".mp4" and ffmpeg_available:
        writer = FFMpegWriter(fps=fps, bitrate=2800)
    if writer is None:
        output_path = output_path.with_suffix(".gif")
        writer = PillowWriter(fps=fps)

    animation.save(output_path, writer=writer)
    plt.close(fig)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate ATP bar chart race video from CSV")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="CSV path")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output .mp4 or .gif path")
    parser.add_argument("--photos-dir", type=Path, default=DEFAULT_PHOTOS_DIR, help="Optional folder with player photos")
    parser.add_argument("--flags-dir", type=Path, default=DEFAULT_FLAGS_DIR, help="Folder for cached country flag PNGs")
    parser.add_argument("--fps", type=int, default=30, help="Video fps")
    parser.add_argument("--frames-per-period", type=int, default=18, help="Interpolation frames between ranking dates")
    parser.add_argument("--top-n", type=int, default=15, help="Number of bars displayed")
    parser.add_argument("--title", type=str, default="ATP Ranking", help="Chart title")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    snapshots = load_snapshots(args.input)
    if not snapshots:
        raise RuntimeError(f"No data found in {args.input}")

    frames = interpolate_snapshots(snapshots, frames_per_period=args.frames_per_period)
    output = render_video(
        frames=frames,
        output_path=args.output,
        top_n=args.top_n,
        title=args.title,
        fps=args.fps,
        photos_dir=args.photos_dir,
        flags_dir=args.flags_dir,
        year_in_media_box=False,
        leader_logo_in_header=False,
        show_bottom_date=True,
    )
    print(f"[video_generator] bar chart race generated -> {output}")


if __name__ == "__main__":
    main()
