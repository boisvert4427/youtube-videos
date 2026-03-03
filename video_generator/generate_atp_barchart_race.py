from __future__ import annotations

import argparse
import csv
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patheffects
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter, writers
from matplotlib.font_manager import FontProperties


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = PROJECT_ROOT / "data" / "processed" / "atp_ranking_timeseries_v1.sample.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "atp_ranking_race.mp4"


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
    "NED": "NL",
    "NOR": "NO",
    "POL": "PL",
    "POR": "PT",
    "ROU": "RO",
    "RUS": "RU",
    "SRB": "RS",
    "SUI": "CH",
    "SWE": "SE",
    "USA": "US",
}

EMOJI_FONT_PATH = Path("C:/Windows/Fonts/seguiemj.ttf")
EMOJI_FONT = FontProperties(fname=str(EMOJI_FONT_PATH)) if EMOJI_FONT_PATH.exists() else None


def country_to_flag(country_code_alpha3: str) -> str:
    alpha2 = ALPHA3_TO_ALPHA2.get(country_code_alpha3.strip().upper(), "")
    if len(alpha2) != 2:
        code = country_code_alpha3.strip().upper()
        return f"[{code}]" if code else ""
    offset = 127397
    return chr(ord(alpha2[0]) + offset) + chr(ord(alpha2[1]) + offset)


def _apply_gradient_background(fig: plt.Figure) -> None:
    bg = fig.add_axes([0, 0, 1, 1], zorder=0)
    bg.axis("off")

    width, height = 1600, 900
    x = np.linspace(0, 1, width)
    y = np.linspace(0, 1, height)
    xx, yy = np.meshgrid(x, y)
    base = 0.22 + 0.78 * (0.6 * (1 - xx) + 0.4 * (1 - yy))
    vignette = 1 - 0.35 * ((xx - 0.45) ** 2 + (yy - 0.5) ** 2)
    lum = np.clip(base * vignette, 0, 1)

    dark = np.array([11, 20, 40]) / 255.0
    light = np.array([35, 61, 92]) / 255.0
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


def render_video(frames: list[tuple[datetime, dict[str, float], dict[str, str]]], output_path: Path, top_n: int, title: str, fps: int) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(14, 8), dpi=120, facecolor="#0b1930")
    _apply_gradient_background(fig)
    ax.set_zorder(2)
    ax.set_facecolor((0, 0, 0, 0))

    palette = [
        "#d8b7a8",
        "#9bb3d9",
        "#2a9d3a",
        "#2e86de",
        "#9b7acb",
        "#86bed0",
        "#b4a6ca",
        "#8aca7f",
        "#f09c90",
        "#f44336",
        "#80bed8",
        "#ffafcc",
        "#f2e394",
        "#84a59d",
        "#f6bd60",
    ]

    def update(frame_index: int) -> None:
        ax.clear()
        frame_date, points, countries = frames[frame_index]
        ranked = sorted(points.items(), key=lambda x: x[1], reverse=True)[:top_n]

        labels: list[str] = []
        flags: list[str] = []
        values: list[float] = []
        colors: list[str] = []
        for idx, (player, value) in enumerate(ranked):
            country_code = countries.get(player, "")
            flag = country_to_flag(country_code)
            label = player.strip()
            labels.append(label)
            flags.append(flag)
            values.append(value)
            colors.append(palette[idx % len(palette)])

        y_pos = list(range(len(labels)))[::-1]
        bars = ax.barh(y_pos, values, color=colors, height=0.78, edgecolor=(0, 0, 0, 0), zorder=3)

        max_val = max(values) if values else 100
        x_right = max_val * 1.04
        x_left = -max_val * 0.025
        ax.set_xlim(x_left, x_right)
        ax.set_ylim(-0.6, len(labels) - 0.1)

        ax.xaxis.set_ticks_position("top")
        ax.xaxis.set_label_position("top")
        ax.tick_params(axis="x", colors="#8da2bd", labelsize=9, length=0, pad=8)
        ax.set_yticks([])
        ax.grid(axis="x", alpha=0.16, color="#87a0bd", linewidth=0.8)
        ax.set_axisbelow(True)

        rank_circle_x = x_left + (x_right - x_left) * 0.01
        for idx, (bar, val, label, flag) in enumerate(zip(bars, values, labels, flags), start=1):
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
            )
            ax.text(rank_circle_x, y, f"{idx}", ha="center", va="center", fontsize=12, color="#172033", fontweight="bold", zorder=6)

            flag_x = x * 0.01 + max_val * 0.01
            flag_artist = ax.text(
                flag_x,
                y,
                flag,
                va="center",
                ha="left",
                fontsize=18,
                color="#e8edf5",
                zorder=6,
                fontproperties=EMOJI_FONT,
            )
            flag_artist.set_path_effects([patheffects.withStroke(linewidth=2.0, foreground="#1c283d")])

            label_artist = ax.text(
                flag_x + max_val * 0.035,
                y,
                label,
                va="center",
                ha="left",
                fontsize=19,
                color="#e8edf5",
                fontweight="bold",
                zorder=6,
            )
            label_artist.set_path_effects([patheffects.withStroke(linewidth=2.6, foreground="#1c283d")])

            value_artist = ax.text(
                x + max_val * 0.008,
                y,
                f"{int(round(val))}",
                va="center",
                ha="left",
                fontsize=14,
                color="#e6ebf3",
                fontweight="bold",
                zorder=6,
            )
            value_artist.set_path_effects([patheffects.withStroke(linewidth=2.4, foreground="#162238")])

        date_text = frame_date.strftime("%Y-%m")
        date_artist = ax.text(
            0.985,
            0.055,
            date_text,
            transform=ax.transAxes,
            ha="right",
            va="center",
            fontsize=38,
            fontweight="bold",
            color="#edf1f7",
            alpha=0.92,
            zorder=7,
        )
        date_artist.set_path_effects([patheffects.withStroke(linewidth=2.8, foreground="#0f172a")])

        ax.set_title(title, fontsize=27, color="#f2f5fa", fontweight="bold", pad=12)
        for spine in ax.spines.values():
            spine.set_visible(False)

        ax.margins(x=0)

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
    )
    print(f"[video_generator] bar chart race generated -> {output}")


if __name__ == "__main__":
    main()
