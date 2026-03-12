from __future__ import annotations

import argparse
import csv
import math
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from moviepy import AudioFileClip, CompositeAudioClip, VideoClip
from moviepy.audio.fx import AudioFadeIn, AudioFadeOut
from PIL import Image, ImageDraw, ImageFont, ImageOps


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = PROJECT_ROOT / "data" / "processed" / "ucl_titles_timeseries_1956_2025.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "ucl_titles_race_moviepy_1min.mp4"
DEFAULT_LOGOS_DIR = PROJECT_ROOT / "data" / "raw" / "club_logos"
DEFAULT_TROPHY_PATH = PROJECT_ROOT / "data" / "raw" / "ucl_trophy.png"
DEFAULT_AUDIO = PROJECT_ROOT / "data" / "raw" / "audio" / "audio.mp3"
DEFAULT_FLAGS_DIR = PROJECT_ROOT / "data" / "raw" / "flags"

WIDTH = 1920
HEIGHT = 1080
TOP_N = 10
FPS = 60
TOTAL_DURATION = 60.0
HOLD_SECONDS = 0.0
TITLE = "UEFA Champions League Titles"
FINAL_AUDIO_FADE_OUT = 10.0
LOOP_CROSSFADE = 5.0

CLUB_COLORS = {
    "Real Madrid": "#d4af37",
    "Benfica": "#c91f37",
    "AC Milan": "#8b1e3f",
    "Inter Milan": "#0057b8",
    "Celtic": "#0a8f3d",
    "Manchester United": "#b22234",
    "Feyenoord": "#e36b5d",
    "Ajax": "#f5f5f5",
    "Bayern Munich": "#ff315c",
    "Liverpool": "#7a1029",
    "Nottingham Forest": "#ff5a5f",
    "Aston Villa": "#7a003c",
    "Hamburger SV": "#3f51b5",
    "Juventus": "#111111",
    "Steaua Bucuresti": "#4b6cb7",
    "Barcelona": "#8c1d5b",
    "Borussia Dortmund": "#f4d000",
    "Chelsea": "#034694",
    "Manchester City": "#6cabdd",
    "Marseille": "#00a3e0",
    "PSV Eindhoven": "#ff7f50",
    "Paris Saint-Germain": "#2b2d72",
    "Porto": "#1d5fd0",
    "Red Star Belgrade": "#d4001f",
}

COUNTRY_FLAG_CODES = {
    "ESP": "es",
    "FRA": "fr",
    "GBR": "gb",
    "GER": "de",
    "ITA": "it",
    "NED": "nl",
    "POR": "pt",
    "ROU": "ro",
    "SRB": "rs",
}


@dataclass(frozen=True)
class ClubState:
    club_name: str
    country_code: str
    titles: float
    won_this_year: int
    entered_top10: int


@dataclass(frozen=True)
class Snapshot:
    ranking_date: str
    year: int
    final_score: str
    final_runner_up: str
    final_score_line: str
    states: list[ClubState]


def build_audio_track(audio_path: Path, duration: float):
    base = AudioFileClip(str(audio_path))
    if base.duration >= duration:
        return base.subclipped(0, duration).with_effects([AudioFadeOut(FINAL_AUDIO_FADE_OUT)]), [base]

    clips = []
    keep_alive = [base]
    step = max(0.1, base.duration - LOOP_CROSSFADE)
    loops = int(math.ceil(max(0.0, duration - LOOP_CROSSFADE) / step))
    for index in range(loops):
        segment = (
            base.with_start(index * step)
            .with_effects([AudioFadeIn(LOOP_CROSSFADE), AudioFadeOut(LOOP_CROSSFADE)])
        )
        clips.append(segment)
    mixed = CompositeAudioClip(clips).with_duration(duration).with_effects([AudioFadeOut(FINAL_AUDIO_FADE_OUT)])
    return mixed, keep_alive


def _load_font(size: int, bold: bool = False):
    candidates = [
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf",
    ]
    for font_path in candidates:
        path = Path(font_path)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def _slugify(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-zA-Z0-9]+", "_", normalized.strip().lower()).strip("_")


def _truncate(draw: ImageDraw.ImageDraw, text: str, font, max_width: int) -> str:
    if draw.textbbox((0, 0), text, font=font)[2] <= max_width:
        return text
    suffix = "..."
    lo, hi = 0, len(text)
    best = suffix
    while lo <= hi:
        mid = (lo + hi) // 2
        candidate = text[:mid].rstrip() + suffix
        if draw.textbbox((0, 0), candidate, font=font)[2] <= max_width:
            best = candidate
            lo = mid + 1
        else:
            hi = mid - 1
    return best


def _fit_font_size(
    draw: ImageDraw.ImageDraw,
    text: str,
    max_width: int,
    start_size: int,
    min_size: int,
    bold: bool = True,
):
    size = start_size
    font = _load_font(size, bold=bold)
    while size > min_size and draw.textbbox((0, 0), text, font=font)[2] > max_width:
        size -= 1
        font = _load_font(size, bold=bold)
    return font


def load_snapshots(input_csv: Path) -> list[Snapshot]:
    by_date: dict[str, list[ClubState]] = {}
    meta_by_date: dict[str, tuple[str, str, str]] = {}
    with input_csv.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            date = row["ranking_date"]
            meta_by_date.setdefault(
                date,
                (
                    row.get("final_score", "").strip(),
                    row.get("final_runner_up", "").strip(),
                    row.get("final_score_line", "").strip(),
                ),
            )
            by_date.setdefault(date, []).append(
                ClubState(
                    club_name=row["club_name"].strip(),
                    country_code=row["country_code"].strip(),
                    titles=float(row["titles"]),
                    won_this_year=int(row.get("won_this_year", "0") or 0),
                    entered_top10=int(row.get("entered_top10", "0") or 0),
                )
            )
    snapshots: list[Snapshot] = []
    for ranking_date in sorted(by_date):
        year = int(ranking_date[:4])
        ranked = sorted(by_date[ranking_date], key=lambda item: (-item.titles, item.club_name))
        final_score, final_runner_up, final_score_line = meta_by_date.get(ranking_date, ("", "", ""))
        snapshots.append(
            Snapshot(
                ranking_date=ranking_date,
                year=year,
                final_score=final_score,
                final_runner_up=final_runner_up,
                final_score_line=final_score_line,
                states=ranked,
            )
        )
    return snapshots


def _build_logo_cache(logos_dir: Path) -> dict[str, np.ndarray]:
    cache: dict[str, np.ndarray] = {}
    for path in logos_dir.glob("*"):
        if path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
            continue
        key = path.stem.lower()
        try:
            img = Image.open(path).convert("RGBA")
            fitted = ImageOps.contain(img, (54, 54), method=Image.Resampling.LANCZOS)
            cache[key] = np.array(fitted)
        except Exception:
            continue
    return cache


def _build_flag_cache(flags_dir: Path) -> dict[str, np.ndarray]:
    cache: dict[str, np.ndarray] = {}
    for country_code, flag_code in COUNTRY_FLAG_CODES.items():
        path = flags_dir / f"{flag_code}.png"
        if not path.exists():
            continue
        img = Image.open(path).convert("RGBA")
        # Small, clean flag that fits inside the bar without dominating the label.
        img = ImageOps.contain(img, (34, 24))
        cache[country_code] = np.array(img)
    return cache


def _hex_to_rgb(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    return tuple(int(value[i : i + 2], 16) for i in (0, 2, 4))


def _darken(value: str, ratio: float) -> str:
    r, g, b = _hex_to_rgb(value)
    return "#{:02x}{:02x}{:02x}".format(int(r * ratio), int(g * ratio), int(b * ratio))


def _lighten(value: str, ratio: float) -> str:
    r, g, b = _hex_to_rgb(value)
    return "#{:02x}{:02x}{:02x}".format(
        min(255, int(r + (255 - r) * ratio)),
        min(255, int(g + (255 - g) * ratio)),
        min(255, int(b + (255 - b) * ratio)),
    )


def _text_on(color: str) -> str:
    r, g, b = _hex_to_rgb(color)
    luminance = (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255.0
    return "#10233f" if luminance > 0.62 else "#f4f7fb"


def _draw_centered(draw: ImageDraw.ImageDraw, rect: tuple[int, int, int, int], text: str, font, fill: str) -> None:
    x0, y0, x1, y1 = rect
    bbox = draw.textbbox((0, 0), text, font=font)
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    draw.text((x0 + (x1 - x0 - w) // 2, y0 + (y1 - y0 - h) // 2), text, font=font, fill=fill)


def _smoothstep(value: float) -> float:
    value = min(max(value, 0.0), 1.0)
    return value * value * (3.0 - 2.0 * value)


def _smootherstep(value: float) -> float:
    value = min(max(value, 0.0), 1.0)
    return value * value * value * (value * (value * 6.0 - 15.0) + 10.0)


def _slow_cross_ease(value: float) -> float:
    value = min(max(value, 0.0), 1.0)
    if value <= 0.25:
        return (value / 0.25) * 0.30
    if value >= 0.75:
        return 0.70 + ((value - 0.75) / 0.25) * 0.30
    middle = (value - 0.25) / 0.50
    return 0.30 + _smoothstep(middle) * 0.40


def _ease_out_cubic(value: float) -> float:
    value = min(max(value, 0.0), 1.0)
    return 1.0 - (1.0 - value) ** 3


def _ease_out_quint(value: float) -> float:
    value = min(max(value, 0.0), 1.0)
    return 1.0 - (1.0 - value) ** 5


def _ease_out_back_soft(value: float) -> float:
    value = min(max(value, 0.0), 1.0)
    c1 = 1.15
    c3 = c1 + 1.0
    return 1 + c3 * (value - 1) ** 3 + c1 * (value - 1) ** 2


def _ease_in_out_cubic(value: float) -> float:
    value = min(max(value, 0.0), 1.0)
    if value < 0.5:
        return 4.0 * value * value * value
    return 1.0 - ((-2.0 * value + 2.0) ** 3) / 2.0


def _phase_delay(value: float, delay: float, span: float) -> float:
    if span <= 0.0:
        return 1.0 if value >= delay else 0.0
    return min(max((value - delay) / span, 0.0), 1.0)


def _stepwise_rank_position(prev_idx: int, next_idx: int, alpha: float) -> float:
    alpha = min(max(alpha, 0.0), 1.0)
    if prev_idx == next_idx:
        return float(prev_idx)
    steps = abs(next_idx - prev_idx)
    direction = 1 if next_idx > prev_idx else -1
    scaled = alpha * steps
    completed_steps = min(steps - 1, int(math.floor(scaled)))
    local_alpha = scaled - completed_steps
    eased_local = _smootherstep(local_alpha)
    return float(prev_idx + direction * completed_steps + direction * eased_local)


def _continuous_rank_position(prev_idx: float, next_idx: float, alpha: float) -> float:
    alpha = min(max(alpha, 0.0), 1.0)
    if math.isclose(prev_idx, next_idx):
        return float(next_idx)
    total_distance = abs(next_idx - prev_idx)
    steps = max(1, int(math.ceil(total_distance)))
    direction = 1.0 if next_idx > prev_idx else -1.0
    gap = 1.0 / steps
    # Let adjacent crossed ranks overlap slightly so the velocity stays
    # continuous instead of stalling on every rank boundary.
    span = min(0.9, gap * 1.35)
    travelled = 0.0
    end_travel = 0.0
    for step in range(steps):
        start = step * gap
        segment_distance = min(1.0, max(0.0, total_distance - step))
        local = min(max((alpha - start) / span, 0.0), 1.0)
        end_local = min(max((1.0 - start) / span, 0.0), 1.0)
        travelled += _smoothstep(local) * segment_distance
        end_travel += _smoothstep(end_local) * segment_distance
    if end_travel > 1e-9:
        travelled *= total_distance / end_travel
    travelled = min(total_distance, travelled)
    return float(prev_idx + direction * travelled)


def _interp_positions(states: list[ClubState], top_n: int) -> dict[str, int]:
    ranked = sorted((state for state in states if state.titles > 0), key=lambda item: (-item.titles, item.club_name))
    return {state.club_name: idx for idx, state in enumerate(ranked[:top_n])}


def _rank_with_tie_priority(
    states: list[ClubState],
    top_n: int,
    priority_order: dict[str, int] | None = None,
) -> dict[str, int]:
    priority_order = priority_order or {}
    ranked = sorted(
        (state for state in states if state.titles > 0),
        key=lambda item: (
            -item.titles,
            priority_order.get(item.club_name, 10_000),
            item.club_name,
        ),
    )
    return {state.club_name: idx for idx, state in enumerate(ranked[:top_n])}


def _priority_order_map(states: list[ClubState]) -> dict[str, int]:
    ranked = sorted(
        (state for state in states if state.titles > 0),
        key=lambda item: (-item.titles, item.club_name),
    )
    return {state.club_name: idx for idx, state in enumerate(ranked)}


def _interp_values(prev: Snapshot, nxt: Snapshot, alpha: float) -> list[ClubState]:
    prev_map = {s.club_name: s for s in prev.states}
    next_map = {s.club_name: s for s in nxt.states}
    clubs = sorted(next_map)
    states: list[ClubState] = []
    for club in clubs:
        a = prev_map.get(club) or next_map[club]
        b = next_map[club]
        titles = a.titles + (b.titles - a.titles) * alpha
        states.append(
            ClubState(
                club_name=club,
                country_code=b.country_code or a.country_code,
                titles=titles,
                won_this_year=b.won_this_year if alpha > 0.5 else a.won_this_year,
                entered_top10=b.entered_top10 if alpha > 0.5 else 0,
            )
        )
    return states


def _filter_snapshots(snapshots: list[Snapshot], start_year: int | None, end_year: int | None) -> list[Snapshot]:
    if start_year is None and end_year is None:
        return snapshots

    selected: list[Snapshot] = []
    for idx, snapshot in enumerate(snapshots):
        if start_year is not None and snapshot.year < start_year - 1:
            continue
        if end_year is not None and snapshot.year > end_year:
            continue
        selected.append(snapshot)

    if start_year is not None and selected and selected[0].year >= start_year:
        original_index = snapshots.index(selected[0])
        if original_index > 0:
            selected.insert(0, snapshots[original_index - 1])

    return selected


def render_video(
    input_csv: Path,
    output_path: Path,
    logos_dir: Path,
    trophy_path: Path,
    flags_dir: Path,
    audio_path: Path | None,
    duration: float,
    fps: int,
    top_n: int,
    start_year: int | None = None,
    end_year: int | None = None,
) -> None:
    snapshots = load_snapshots(input_csv)
    snapshots = _filter_snapshots(snapshots, start_year, end_year)
    if len(snapshots) < 2:
        raise RuntimeError("Not enough UCL snapshots to render.")

    logo_cache = _build_logo_cache(logos_dir)
    flag_cache = _build_flag_cache(flags_dir)
    trophy_img = None
    if trophy_path.exists():
        try:
            trophy_img = Image.open(trophy_path).convert("RGBA")
            trophy_img = ImageOps.contain(trophy_img, (150, 150), method=Image.Resampling.LANCZOS)
        except Exception:
            trophy_img = None
    periods = len(snapshots) - 1
    seconds_per_period = duration / periods
    score_reveal_seconds = min(1.15, max(0.45, seconds_per_period * 0.34))
    min_transition_seconds = 1.05
    effective_hold_seconds = min(HOLD_SECONDS, max(0.0, seconds_per_period - score_reveal_seconds - min_transition_seconds))
    hold_ratio = effective_hold_seconds / seconds_per_period if seconds_per_period > 0 else 0.0
    score_ratio = score_reveal_seconds / seconds_per_period if seconds_per_period > 0 else 0.0
    rank_center_x = 110
    rank_radius = 34
    bar_left = 180
    bar_right = 1640
    bar_max_w = bar_right - bar_left
    base_y = 205
    row_h = 70
    row_gap = 8
    header_font = _load_font(56, bold=True)
    sub_font = _load_font(22, bold=False)
    rank_font = _load_font(34, bold=True)
    name_font = _load_font(32, bold=True)
    value_font = _load_font(34, bold=True)
    year_font = _load_font(86, bold=True)

    xx = np.linspace(0, 1, WIDTH, dtype=np.float32)
    yy = np.linspace(0, 1, HEIGHT, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(xx, yy)
    night = np.array([8, 20, 46], dtype=np.float32)
    blue = np.array([10, 71, 138], dtype=np.float32)
    cyan = np.array([33, 151, 219], dtype=np.float32)
    mix = np.clip(0.62 * grid_y + 0.22 * grid_x, 0, 1)
    bg = np.clip(
        night[None, None, :] * (1 - mix[..., None])
        + blue[None, None, :] * (0.7 * mix[..., None])
        + cyan[None, None, :] * (0.3 * mix[..., None]),
        0,
        255,
    ).astype(np.uint8)

    def make_frame(t: float) -> np.ndarray:
        frame = Image.fromarray(bg.copy()).convert("RGBA")
        draw = ImageDraw.Draw(frame)
        period_index = min(int(t / seconds_per_period), periods - 1)
        local_t = (t - period_index * seconds_per_period) / seconds_per_period
        local_t = min(max(local_t, 0.0), 1.0)
        prev = snapshots[period_index]
        nxt = snapshots[period_index + 1]
        score_phase_end = min(1.0, hold_ratio + score_ratio)
        score_alpha = 0.0
        if local_t > hold_ratio and score_ratio > 0.0:
            score_alpha = min(1.0, (local_t - hold_ratio) / max(1e-6, score_ratio))
        score_alpha = _ease_out_cubic(score_alpha)
        transition_alpha = 0.0
        if local_t > score_phase_end and score_phase_end < 1.0:
            transition_alpha = (local_t - score_phase_end) / max(1e-6, 1.0 - score_phase_end)
        transition_alpha = min(max(transition_alpha, 0.0), 1.0)
        value_alpha = _ease_in_out_cubic(_smootherstep(transition_alpha))
        move_alpha = _smoothstep(_phase_delay(transition_alpha, 0.02, 0.96))
        interp = _interp_values(prev, nxt, value_alpha)
        interp_map = {state.club_name: state for state in interp}
        prev_priority = _priority_order_map(prev.states)
        prev_rank = _rank_with_tie_priority(prev.states, top_n, prev_priority)
        next_rank = _rank_with_tie_priority(nxt.states, top_n, prev_priority)
        prev_map = {state.club_name: state for state in prev.states}
        next_map = {state.club_name: state for state in nxt.states}
        animated_clubs = sorted(set(prev_rank) | set(next_rank))
        top_states = [
            interp_map[name]
            for name in animated_clubs
            if interp_map.get(name) is not None
        ]
        max_titles = max(1.0, max((state.titles for state in top_states), default=1.0))

        draw.text((90, 54), TITLE, font=header_font, fill="#f4f7fb")
        draw.text((93, 118), "European Cup + Champions League", font=sub_font, fill="#b7c7df")
        tick_font = _load_font(18, bold=True)
        tick_count = max(1, int(math.ceil(max_titles)))
        for tick in range(1, tick_count + 1):
            x = bar_left + int((tick / max_titles) * bar_max_w)
            draw.line((x, 204, x, HEIGHT - 120), fill=(0, 0, 0, 88), width=2)
            draw.text((x - 8, 176), str(tick), font=tick_font, fill=(0, 0, 0, 120))

        year_center_x = 1640
        score_box_max_w = 520
        score_text = nxt.final_score_line or ""
        score_font = _fit_font_size(draw, score_text, score_box_max_w - 40, 34, 22, bold=True) if score_text else _load_font(34, bold=True)
        score_bbox = draw.textbbox((0, 0), score_text or " ", font=score_font)
        trophy_h = trophy_img.height if trophy_img is not None else 0
        year_anchor_y = 705
        trophy_gap = 24
        trophy_top_y = year_anchor_y - trophy_h - trophy_gap
        score_anchor_y = 850
        if trophy_img is not None:
            trophy_x = year_center_x - trophy_img.width // 2 - 18
            frame.alpha_composite(trophy_img, (trophy_x, trophy_top_y))
        draw.text((year_center_x, year_anchor_y), str(nxt.year), font=year_font, fill="#f4c84b", anchor="ma")
        if nxt.final_score_line:
            box_w = max(280, min(score_box_max_w, score_bbox[2] - score_bbox[0] + 54))
            score_layer = ImageDraw.Draw(frame, "RGBA")
            score_layer.rounded_rectangle(
                (year_center_x - box_w // 2, score_anchor_y - 40, year_center_x + box_w // 2, score_anchor_y + 20),
                radius=20,
                fill=(9, 26, 52, int(175 * score_alpha)),
                outline=(244, 200, 75, int(135 * score_alpha)),
                width=2,
            )
            score_layer.text(
                (year_center_x, score_anchor_y - 28),
                score_text,
                font=score_font,
                fill=(244, 247, 251, int(255 * score_alpha)),
                anchor="ma",
            )

        render_items: list[tuple[float, float, ClubState, int, str, str, int, bool, float, float, int, float]] = []
        for state in top_states:
            prev_idx = prev_rank.get(state.club_name, top_n + 1)
            next_idx = next_rank.get(state.club_name, top_n + 1)
            entering = prev_idx > top_n and next_idx <= top_n
            effective_prev_idx = float(top_n + 2.4) if entering else float(prev_idx)
            effective_move_alpha = move_alpha
            places_moved = abs(float(next_idx) - effective_prev_idx)
            if places_moved > 0.0:
                extra_delay = min(0.08, max(0.0, (places_moved - 1.0) * 0.015))
                active_span = max(0.82, 0.98 - extra_delay)
                effective_move_alpha = _smoothstep(_phase_delay(effective_move_alpha, extra_delay, active_span))
            if entering:
                effective_move_alpha = _smoothstep(_phase_delay(effective_move_alpha, 0.02, 0.96))
            y_idx = _continuous_rank_position(effective_prev_idx, float(next_idx), effective_move_alpha)
            y = base_y + y_idx * (row_h + row_gap)
            prev_state = prev_map.get(state.club_name)
            previous_titles = prev_state.titles if prev_state is not None else 0.0
            target_titles = next_map[state.club_name].titles if state.club_name in next_map else state.titles
            base_color = CLUB_COLORS.get(state.club_name, "#39c0ff")
            total_rank_delta = abs(float(next_idx) - effective_prev_idx)
            if total_rank_delta > 1e-6:
                # Keep horizontal growth locked to the same motion curve as the vertical move.
                width_alpha = effective_move_alpha
            else:
                width_alpha = value_alpha
            current_titles_for_width = previous_titles + (target_titles - previous_titles) * width_alpha
            bar_w = max(120, int((current_titles_for_width / max_titles) * bar_max_w))
            current_rank_label = next_idx + 1 if next_idx <= top_n else top_n + 1
            direction = 0
            if next_idx > prev_idx:
                direction = 1
            elif next_idx < prev_idx:
                direction = -1
            render_items.append(
                (
                    0.0,
                    y,
                    state,
                    current_rank_label,
                    base_color,
                    "",
                    bar_w,
                    next_idx == 0,
                    previous_titles,
                    target_titles,
                    direction,
                    current_titles_for_width,
                )
            )

        render_items.sort(key=lambda item: (item[10] != 1, item[10] == -1, item[1]))
        frame_draw = ImageDraw.Draw(frame, "RGBA")

        for rank_idx in range(top_n):
            rank_center_y = base_y + rank_idx * (row_h + row_gap) + row_h // 2
            frame_draw.ellipse(
                (
                    rank_center_x - rank_radius + 4,
                    rank_center_y - rank_radius + 6,
                    rank_center_x + rank_radius + 4,
                    rank_center_y + rank_radius + 6,
                ),
                fill=(0, 0, 0, 100),
            )
            frame_draw.ellipse(
                (
                    rank_center_x - rank_radius,
                    rank_center_y - rank_radius,
                    rank_center_x + rank_radius,
                    rank_center_y + rank_radius,
                ),
                fill="#f4c84b",
                outline="#fff2b8",
                width=3,
            )
            _draw_centered(
                draw,
                (
                    rank_center_x - rank_radius,
                    rank_center_y - rank_radius,
                    rank_center_x + rank_radius,
                    rank_center_y + rank_radius,
                ),
                f"{rank_idx + 1}",
                rank_font,
                "#10233f",
            )

        for _, y, state, rank_label, bar_color, label_bg, bar_w, is_leader, previous_titles, target_titles, direction, current_titles_for_width in render_items:
            y0 = int(y)
            y1 = y0 + row_h
            text_color = _text_on(bar_color)
            shadow_rect = (bar_left + 8, y0 + 8, bar_left + bar_w + 8, y1 + 8)
            frame_draw.rounded_rectangle(shadow_rect, radius=22, fill=(0, 0, 0, 92))
            if state.won_this_year:
                frame_draw.rounded_rectangle(
                    (bar_left - 8, y0 - 8, bar_left + bar_w + 8, y1 + 8),
                    radius=28,
                    outline=_lighten(bar_color, 0.25),
                    width=6,
                )
            draw.rounded_rectangle((bar_left, y0, bar_left + bar_w, y1), radius=22, fill=bar_color)
            frame_draw.line((bar_left + 18, y0 + 12, bar_left + bar_w - 18, y0 + 12), fill=(255, 255, 255, 38), width=3)

            flag = flag_cache.get(state.country_code.strip())
            flag_width = int(flag.shape[1]) if flag is not None else 0
            flag_gap = 16 if flag is not None else 0
            name_x = bar_left + 24 + flag_width + flag_gap
            logo = logo_cache.get(_slugify(state.club_name))
            logo_width = int(logo.shape[1]) if logo is not None else 0
            value_number = int(round(current_titles_for_width))
            value_text = f"{value_number}"
            value_bbox = draw.textbbox((0, 0), value_text, font=value_font)
            outside_gap = 24
            value_pad = value_bbox[2] + outside_gap + 8
            name_max_w = min(720, max(180, bar_w - 56 - value_pad - logo_width - flag_width - flag_gap))
            name = _truncate(draw, state.club_name, name_font, name_max_w)
            if flag is not None:
                flag_img = Image.fromarray(flag)
                fx = bar_left + 24
                fy = y0 + (row_h - flag_img.height) // 2
                frame_draw.rounded_rectangle(
                    (fx - 4, fy - 4, fx + flag_img.width + 4, fy + flag_img.height + 4),
                    radius=8,
                    fill=(244, 247, 251, 230),
                    outline=(19, 39, 70, 150),
                    width=1,
                )
                frame.alpha_composite(flag_img, (fx, fy))
            draw.text((name_x, y0 + 18), name, font=name_font, fill=text_color)
            if logo is not None:
                logo_img = Image.fromarray(logo)
                lx = bar_left + bar_w - logo_img.width - 12
                lx = max(bar_left + 120, lx)
                ly = y0 + (row_h - logo_img.height) // 2
                frame_draw.rounded_rectangle(
                    (lx - 8, y0 + 8, lx + logo_img.width + 8, y1 - 8),
                    radius=16,
                    fill=(244, 247, 251, 232),
                    outline=(19, 39, 70, 185),
                    width=2,
                )
                frame.alpha_composite(logo_img, (lx, ly))
                value_x = bar_left + bar_w + outside_gap
            else:
                value_x = bar_left + bar_w + outside_gap
            draw.text((value_x, y0 + 17), value_text, font=value_font, fill="#f4f7fb")

        return np.array(frame.convert("RGB"))

    clip = VideoClip(make_frame, duration=duration)
    audio_clip = None
    keep_alive = []
    if audio_path is not None:
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        audio_clip, keep_alive = build_audio_track(audio_path, duration)
        clip = clip.with_audio(audio_clip)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    clip.write_videofile(str(output_path), fps=fps, codec="libx264", audio_codec="aac")
    clip.close()
    if audio_clip is not None:
        audio_clip.close()
    for item in keep_alive:
        item.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a MoviePy UCL bar chart race preview.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--logos-dir", type=Path, default=DEFAULT_LOGOS_DIR)
    parser.add_argument("--trophy-path", type=Path, default=DEFAULT_TROPHY_PATH)
    parser.add_argument("--flags-dir", type=Path, default=DEFAULT_FLAGS_DIR)
    parser.add_argument("--audio", type=Path, default=DEFAULT_AUDIO)
    parser.add_argument("--duration", type=float, default=TOTAL_DURATION)
    parser.add_argument("--fps", type=int, default=FPS)
    parser.add_argument("--top-n", type=int, default=TOP_N)
    parser.add_argument("--start-year", type=int, default=None)
    parser.add_argument("--end-year", type=int, default=None)
    args = parser.parse_args()

    render_video(
        args.input,
        args.output,
        args.logos_dir,
        args.trophy_path,
        args.flags_dir,
        args.audio,
        args.duration,
        args.fps,
        args.top_n,
        args.start_year,
        args.end_year,
    )
    print(f"[video_generator] UCL MoviePy race generated -> {args.output}")


if __name__ == "__main__":
    main()
