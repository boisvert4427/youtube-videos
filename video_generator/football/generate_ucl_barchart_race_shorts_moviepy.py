from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation, writers
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib import patheffects

from video_generator.generate_ucl_barchart_race_moviepy import (
    CLUB_COLORS,
    DEFAULT_AUDIO,
    DEFAULT_INPUT,
    DEFAULT_LOGOS_DIR,
    TOP_N as HORIZONTAL_TOP_N,
    _build_logo_cache,
    _filter_snapshots,
    _slugify,
    load_snapshots,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "ucl_titles_race_shorts.mp4"

FPS = 30
TOP_N = 8
TOTAL_DURATION = 35.0
FIGSIZE = (10.8, 19.2)
DPI = 100
TITLE = "UCL TITLES RACE"


def interpolate_snapshots(snapshots, frames_per_period: int):
    frames: list[tuple[object, dict[str, float], object]] = []
    if len(snapshots) < 2:
        only = snapshots[0]
        return [(only, {s.club_name: float(s.titles) for s in only.states}, only)]

    for i in range(len(snapshots) - 1):
        current = snapshots[i]
        nxt = snapshots[i + 1]
        club_names = sorted({state.club_name for state in current.states} | {state.club_name for state in nxt.states})
        current_map = {state.club_name: state for state in current.states}
        next_map = {state.club_name: state for state in nxt.states}
        for step in range(frames_per_period):
            t = step / frames_per_period
            values: dict[str, float] = {}
            for club_name in club_names:
                start = float(current_map.get(club_name).titles if club_name in current_map else 0.0)
                end = float(next_map.get(club_name).titles if club_name in next_map else 0.0)
                values[club_name] = start + (end - start) * t
            frames.append((current, values, nxt))

    last = snapshots[-1]
    frames.append((last, {state.club_name: float(state.titles) for state in last.states}, last))
    return frames


def _nice_axis_max(current_max: float) -> float:
    if current_max <= 0:
        return 6.0
    if current_max <= 20:
        return float(max(6, int(np.ceil(current_max)) + 1))
    return float(int(np.ceil(current_max / 2.0) * 2))


def render_video(
    input_csv: Path,
    output_path: Path,
    logos_dir: Path,
    duration: float,
    fps: int,
    top_n: int,
    start_year: int | None = None,
    end_year: int | None = None,
) -> Path:
    snapshots = load_snapshots(input_csv)
    snapshots = _filter_snapshots(snapshots, start_year, end_year)
    if len(snapshots) < 2:
        raise RuntimeError("Not enough UCL snapshots to render.")

    periods = len(snapshots) - 1
    frames_per_period = max(2, int(round((duration * fps) / periods)))
    frames = interpolate_snapshots(snapshots, frames_per_period=frames_per_period)
    logo_cache = _build_logo_cache(logos_dir)

    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI, facecolor="#07152f")
    fig.subplots_adjust(left=0.08, right=0.95, top=0.83, bottom=0.06)
    ax.set_facecolor("#0a1c3d")

    display_y_by_club: dict[str, float] = {}
    global_max = 1.0
    for _, values, _ in frames:
        if values:
            global_max = max(global_max, max(values.values()))
    axis_cap = _nice_axis_max(global_max)
    title_fx = [patheffects.withStroke(linewidth=3, foreground="#071327", alpha=0.65)]

    def update(frame_index: int) -> None:
        ax.clear()
        ax.set_facecolor("#0a1c3d")
        ax.set_xlim(-2.9, axis_cap + 0.9)
        ax.set_ylim(-0.8, top_n - 0.2)
        ax.spines[:].set_visible(False)
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        ax.grid(axis="x", color=(1, 1, 1, 0.07), linewidth=1)
        ax.set_axisbelow(True)

        prev_snapshot, values, next_snapshot = frames[frame_index]
        ranked = sorted(values.items(), key=lambda item: (-item[1], item[0]))[:top_n]
        target_y = {club_name: float(top_n - 1 - idx) for idx, (club_name, _) in enumerate(ranked)}

        items: list[tuple[str, float, float]] = []
        for club_name, value in ranked:
            prev_y = display_y_by_club.get(club_name, target_y[club_name])
            current_y = prev_y + (target_y[club_name] - prev_y) * 0.30
            display_y_by_club[club_name] = current_y
            items.append((club_name, value, current_y))

        for stale in list(display_y_by_club):
            if stale not in target_y:
                del display_y_by_club[stale]

        # Header
        header_panel = plt.Rectangle((0.0, 0.905), 1.0, 0.14, transform=ax.transAxes, facecolor=(1, 1, 1, 0.03), edgecolor="none", zorder=0, clip_on=False)
        ax.add_patch(header_panel)
        ax.text(
            0.02,
            1.07,
            "MOST UCL TITLES",
            transform=ax.transAxes,
            fontsize=29,
            fontweight="bold",
            color="#f4f7fb",
            path_effects=title_fx,
        )
        ax.text(
            0.02,
            1.025,
            "European Cup + Champions League history",
            transform=ax.transAxes,
            fontsize=12.5,
            color="#9fb6d8",
        )
        ax.text(
            0.02,
            0.962,
            str(next_snapshot.year),
            transform=ax.transAxes,
            fontsize=31,
            fontweight="bold",
            color="#0f2238",
            bbox=dict(boxstyle="round,pad=0.34", fc="#f3c454", ec="none"),
        )
        if next_snapshot.final_score_line:
            ax.text(
                0.98,
                0.975,
                next_snapshot.final_score_line,
                transform=ax.transAxes,
                ha="right",
                va="center",
                fontsize=13.5,
                fontweight="bold",
                color="#0f2238",
                bbox=dict(boxstyle="round,pad=0.42", fc="#f3c454", ec="none"),
            )
        if next_snapshot.final_runner_up:
            ax.text(
                0.16,
                0.922,
                f"Final vs {next_snapshot.final_runner_up}",
                transform=ax.transAxes,
                fontsize=11.5,
                color="#dce7f7",
            )

        for tick in range(1, int(axis_cap) + 1):
            ax.text(tick, top_n - 0.08, str(tick), ha="center", va="bottom", fontsize=9.5, color=(1, 1, 1, 0.42))

        for rank_idx in range(top_n):
            y_lane = top_n - 1 - rank_idx
            ax.barh(y_lane, axis_cap, height=0.72, color=(1, 1, 1, 0.040), edgecolor="none", zorder=1)
            ax.barh(y_lane, axis_cap * 0.99, height=0.72, left=0.02, color=(1, 1, 1, 0.018), edgecolor="none", zorder=1)

        ax.plot([-0.02, -0.02], [-0.5, top_n - 0.2], color=(1, 1, 1, 0.10), linewidth=1.2, zorder=2)

        for idx, (club_name, value, y) in enumerate(sorted(items, key=lambda item: item[2])):
            color = CLUB_COLORS.get(club_name, "#39c0ff")
            ax.barh(y, value, height=0.60, color=color, edgecolor="none", zorder=3)
            ax.barh(y + 0.18, value * 0.995, height=0.06, color=(1, 1, 1, 0.14), edgecolor="none", zorder=4)

            rank = idx + 1
            ax.text(
                -2.35,
                y,
                str(rank),
                va="center",
                ha="center",
                fontsize=13,
                fontweight="bold",
                color="#10233f",
                bbox=dict(boxstyle="round,pad=0.40", fc="#f3c454", ec="none"),
                clip_on=False,
                zorder=6,
            )

            logo = logo_cache.get(_slugify(club_name))
            is_light_bar = color in {"#d4af37", "#f4d000", "#f5f5f5", "#ff7f50", "#6cabdd"}
            label_color = "#10233f" if is_light_bar else "#eef4ff"
            text_x = 0.34
            if logo is not None:
                imagebox = OffsetImage(logo, zoom=0.78)
                ab = AnnotationBbox(
                    imagebox,
                    (0.20, y),
                    frameon=True,
                    box_alignment=(0.5, 0.5),
                    bboxprops=dict(fc="white", ec="none", boxstyle="round,pad=0.16"),
                )
                ax.add_artist(ab)
                text_x = 0.42

            max_name_chars = 24 if value >= 6 else 18 if value >= 4 else 14
            display_name = club_name if len(club_name) <= max_name_chars else club_name[: max_name_chars - 3].rstrip() + "..."
            ax.text(
                text_x,
                y,
                display_name,
                va="center",
                ha="left",
                fontsize=13.5,
                fontweight="bold",
                color=label_color,
                path_effects=[patheffects.withStroke(linewidth=3, foreground="#081632", alpha=0.65)],
                zorder=6,
            )
            value_x = min(value + 0.14, axis_cap + 0.28)
            ax.text(
                value_x,
                y,
                f"{int(round(value))}",
                va="center",
                ha="left",
                fontsize=15,
                fontweight="bold",
                color="#f4f7fb",
                path_effects=[patheffects.withStroke(linewidth=3, foreground="#081632", alpha=0.70)],
                zorder=6,
            )

    animation = FuncAnimation(fig, update, frames=len(frames), interval=1000 / fps, repeat=False)
    ffmpeg_available = writers.is_available("ffmpeg")
    if not ffmpeg_available:
        raise RuntimeError("ffmpeg is required for MP4 output but is not available.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = FFMpegWriter(fps=fps, bitrate=2400)
    animation.save(output_path, writer=writer)
    plt.close(fig)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a new Shorts UCL bar chart race.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--logos-dir", type=Path, default=DEFAULT_LOGOS_DIR)
    parser.add_argument("--audio", type=Path, default=DEFAULT_AUDIO)
    parser.add_argument("--duration", type=float, default=TOTAL_DURATION)
    parser.add_argument("--fps", type=int, default=FPS)
    parser.add_argument("--top-n", type=int, default=min(TOP_N, HORIZONTAL_TOP_N))
    parser.add_argument("--start-year", type=int, default=None)
    parser.add_argument("--end-year", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _ = args.audio
    output = render_video(
        input_csv=args.input,
        output_path=args.output,
        logos_dir=args.logos_dir,
        duration=args.duration,
        fps=args.fps,
        top_n=args.top_n,
        start_year=args.start_year,
        end_year=args.end_year,
    )
    print(f"[video_generator] new UCL Shorts race generated -> {output}")


if __name__ == "__main__":
    main()
