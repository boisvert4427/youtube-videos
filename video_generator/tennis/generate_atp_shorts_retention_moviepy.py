from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from moviepy import VideoClip
from PIL import Image, ImageDraw, ImageFont, ImageOps

from video_generator.generate_atp_vertical_timeline_moviepy import load_entries
from video_generator.tennis.generate_atp_shorts_timeline_moviepy import (
    DEFAULT_AUDIO,
    DEFAULT_INPUT,
    DEFAULT_PHOTOS_DIR,
    FINAL_AUDIO_FADE_OUT,
    FPS,
    HEIGHT,
    HOLD_END,
    HOLD_START,
    LOOP_CROSSFADE,
    TOTAL_DURATION,
    WIDTH,
    _extract_result_parts,
    _fit_font,
    _load_font,
    _resolve_player_image,
    _truncate_to_width,
    build_audio_track,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "atp_shorts_retention_indian_wells.mp4"


def _draw_text(draw: ImageDraw.ImageDraw, x: int, y: int, text: str, font, fill: str, anchor: str = "la") -> None:
    draw.text((x, y), text, font=font, fill=fill, anchor=anchor)


def _pick_key_result(results: list[str]) -> tuple[str, str]:
    priority = ["F", "SF", "QF", "R16", "R32", "R64", "R128"]
    parsed: dict[str, tuple[str, str]] = {}
    for row in results:
        rnd, opp, score = _extract_result_parts(row)
        if rnd:
            parsed[rnd] = (opp, score)
    for rnd in priority:
        opp, score = parsed.get(rnd, ("", ""))
        if opp and opp != "-":
            return rnd, f"beat {opp}"
    return "CHAMPION", "Indian Wells winner"


def _wrap_two_lines(draw: ImageDraw.ImageDraw, text: str, font, max_width: int) -> list[str]:
    words = text.split()
    if not words:
        return []
    best = [text]
    for idx in range(1, len(words)):
        left = " ".join(words[:idx])
        right = " ".join(words[idx:])
        if (
            draw.textbbox((0, 0), left, font=font)[2] <= max_width
            and draw.textbbox((0, 0), right, font=font)[2] <= max_width
        ):
            best = [left, right]
    if len(best) == 1:
        best = [_truncate_to_width(draw, text, font, max_width)]
    return best[:2]


def render_card(entry, photos_dir: Path) -> np.ndarray:
    xx = np.linspace(0, 1, WIDTH, dtype=np.float32)
    yy = np.linspace(0, 1, HEIGHT, dtype=np.float32)
    _, grid_y = np.meshgrid(xx, yy)
    paper = np.array([240, 233, 217], dtype=np.float32)
    sand = np.array([212, 170, 98], dtype=np.float32)
    court = np.array([54, 118, 132], dtype=np.float32)
    deep = np.array([15, 39, 44], dtype=np.float32)
    mix = np.clip(0.48 * grid_y + 0.08, 0, 1)
    bg = np.clip(
        paper[None, None, :] * (1.0 - mix[..., None])
        + sand[None, None, :] * (0.62 * mix[..., None])
        + court[None, None, :] * (0.24 * (1.0 - grid_y[..., None]))
        + deep[None, None, :] * (0.10 * (1.0 - grid_y[..., None])),
        0,
        255,
    ).astype(np.uint8)
    frame = Image.fromarray(np.dstack([bg, np.full((HEIGHT, WIDTH), 255, dtype=np.uint8)]), mode="RGBA")
    draw = ImageDraw.Draw(frame, "RGBA")

    source_path = _resolve_player_image(entry.image_path, entry.player_name, photos_dir)
    photo_rect = (52, 170, WIDTH - 52, 1260)
    if source_path:
        try:
            photo = ImageOps.exif_transpose(Image.open(source_path)).convert("RGB")
            photo = ImageOps.fit(
                photo,
                (photo_rect[2] - photo_rect[0], photo_rect[3] - photo_rect[1]),
                method=Image.Resampling.LANCZOS,
                centering=(0.5, 0.14),
            )
        except Exception:
            photo = Image.new("RGB", (photo_rect[2] - photo_rect[0], photo_rect[3] - photo_rect[1]), "#d3d3d3")
    else:
        photo = Image.new("RGB", (photo_rect[2] - photo_rect[0], photo_rect[3] - photo_rect[1]), "#d3d3d3")

    photo_rgba = photo.convert("RGBA")
    mask = Image.new("L", photo_rgba.size, 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, photo_rgba.width, photo_rgba.height), radius=42, fill=255)
    photo_rgba.putalpha(mask)
    frame.alpha_composite(photo_rgba, (photo_rect[0], photo_rect[1]))

    shadow = Image.new("RGBA", (photo_rect[2] - photo_rect[0], 360), (0, 0, 0, 0))
    shadow_draw = ImageDraw.Draw(shadow, "RGBA")
    for i in range(360):
        alpha = int(210 * (i / 359) ** 1.6)
        shadow_draw.line((0, i, shadow.width, i), fill=(6, 12, 15, alpha))
    frame.alpha_composite(shadow, (photo_rect[0], photo_rect[3] - 360))

    label_font = _load_font(40, bold=True)
    year_font = _load_font(92, bold=True)
    _draw_text(draw, 68, 78, "INDIAN WELLS", label_font, "#2f7685")
    draw.rounded_rectangle((68, 122, 312, 132), radius=5, fill=(229, 191, 101, 255))
    _draw_text(draw, WIDTH - 68, 88, str(entry.year), year_font, "#102a31", anchor="ra")

    name_font = _fit_font(draw, entry.player_name.upper(), WIDTH - 160, 96, 44)
    name_lines = _wrap_two_lines(draw, entry.player_name.upper(), name_font, WIDTH - 160)
    name_y = 1030
    for idx, line in enumerate(name_lines):
        _draw_text(draw, 84, name_y + idx * 86, line, name_font, "#fff8ee")

    rank_rect = (84, 1232, 330, 1302)
    draw.rounded_rectangle(rank_rect, radius=18, fill=(223, 185, 97, 255))
    rank_font = _fit_font(draw, entry.rank_label, rank_rect[2] - rank_rect[0] - 30, 34, 18)
    bbox = draw.textbbox((0, 0), entry.rank_label, font=rank_font)
    draw.text(
        ((rank_rect[0] + rank_rect[2] - (bbox[2] - bbox[0])) // 2, rank_rect[1] + 17),
        entry.rank_label,
        font=rank_font,
        fill="#12262d",
    )

    key_round, key_line = _pick_key_result(entry.results)
    story_rect = (52, 1335, WIDTH - 52, 1770)
    draw.rounded_rectangle(story_rect, radius=40, fill=(18, 56, 66, 236))
    accent_rect = (52, 1335, 76, 1770)
    draw.rounded_rectangle(accent_rect, radius=20, fill=(225, 186, 96, 255))

    micro_font = _load_font(28, bold=True)
    headline_font = _fit_font(draw, key_line.upper(), story_rect[2] - story_rect[0] - 88, 62, 28)
    support_font = _load_font(26, bold=True)
    _draw_text(draw, 108, 1408, key_round, micro_font, "#e6c274")
    headline_lines = _wrap_two_lines(draw, key_line.upper(), headline_font, story_rect[2] - story_rect[0] - 112)
    for idx, line in enumerate(headline_lines):
        _draw_text(draw, 108, 1484 + idx * 64, line, headline_font, "#fff8ee")

    score_text = ""
    for row in entry.results:
        rnd, opp, score = _extract_result_parts(row)
        if rnd == key_round:
            score_text = score
            break
    if score_text:
        score_label_font = _load_font(20, bold=True)
        score_box = (108, 1608, 600, 1716)
        draw.rounded_rectangle(score_box, radius=26, fill=(240, 232, 213, 255))
        _draw_text(draw, score_box[0] + 24, score_box[1] + 26, "FINAL SCORE", score_label_font, "#8a6a38")
        score_big_font = _fit_font(draw, score_text, score_box[2] - score_box[0] - 48, 44, 24)
        _draw_text(draw, score_box[0] + 24, score_box[1] + 72, score_text, score_big_font, "#245b68")

    hook_rect = (52, 1794, WIDTH - 52, 1868)
    draw.rounded_rectangle(hook_rect, radius=24, fill=(241, 234, 218, 236))
    hook_font = _load_font(28, bold=True)
    _draw_text(draw, WIDTH // 2, 1831, "ONE CHAMPION. ONE YEAR. ONE GLANCE.", hook_font, "#2d6774", anchor="ma")

    return np.array(frame.convert("RGB"))


def render_video(entries: list, output_path: Path, photos_dir: Path, audio_path: Path, fps: int, duration: float) -> Path:
    if not entries:
        raise RuntimeError("No entries to render.")
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    photos_dir.mkdir(parents=True, exist_ok=True)
    card_arrays = [render_card(entry, photos_dir) for entry in entries]

    scroll_duration = duration - HOLD_START - HOLD_END
    if scroll_duration <= 0:
        raise RuntimeError("Invalid Shorts timing configuration.")
    total_shift = max(0.0, len(entries) - 1)

    def make_frame(t: float) -> np.ndarray:
        if t <= HOLD_START:
            progress = 0.0
        elif t >= duration - HOLD_END:
            progress = total_shift
        else:
            progress = total_shift * ((t - HOLD_START) / scroll_duration)

        base_index = min(int(progress), max(0, len(card_arrays) - 1))
        next_index = min(base_index + 1, len(card_arrays) - 1)
        alpha = progress - base_index

        current = card_arrays[base_index].astype(np.float32)
        if next_index == base_index or alpha <= 0:
            return current.astype(np.uint8)

        eased = min(max(alpha, 0.0), 1.0)
        mixed = current * (1.0 - eased) + card_arrays[next_index].astype(np.float32) * eased
        return np.clip(mixed, 0, 255).astype(np.uint8)

    clip = VideoClip(make_frame, duration=duration)
    audio_clip, keep_alive = build_audio_track(audio_path, duration)
    clip = clip.with_audio(audio_clip)
    output_path = output_path.with_suffix(".mp4")
    clip.write_videofile(str(output_path), fps=fps, codec="libx264", audio_codec="aac")
    clip.close()
    audio_clip.close()
    for item in keep_alive:
        item.close()
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate an ATP Shorts retention template video")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="CSV path")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output .mp4 path")
    parser.add_argument("--photos-dir", type=Path, default=DEFAULT_PHOTOS_DIR, help="Player photos folder")
    parser.add_argument("--audio", type=Path, default=DEFAULT_AUDIO, help="Background music path")
    parser.add_argument("--fps", type=int, default=FPS, help="Video fps")
    parser.add_argument("--duration", type=float, default=TOTAL_DURATION, help="Video duration in seconds")
    parser.add_argument("--start-year", type=int, default=None, help="First year to include")
    parser.add_argument("--end-year", type=int, default=None, help="Last year to include")
    parser.add_argument("--last-n", type=int, default=None, help="Keep only the last N entries")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    entries = load_entries(args.input)
    if args.start_year is not None:
        entries = [entry for entry in entries if entry.year >= args.start_year]
    if args.end_year is not None:
        entries = [entry for entry in entries if entry.year <= args.end_year]
    if args.last_n is not None and args.last_n > 0:
        entries = entries[-args.last_n :]
    output = render_video(
        entries=entries,
        output_path=args.output,
        photos_dir=args.photos_dir,
        audio_path=args.audio,
        fps=args.fps,
        duration=args.duration,
    )
    print(f"[video_generator] ATP Shorts retention video generated -> {output}")


if __name__ == "__main__":
    main()
