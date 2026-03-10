from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
from moviepy import AudioFileClip, CompositeAudioClip, VideoClip
from moviepy.audio.fx import AudioFadeIn, AudioFadeOut

from video_generator.cycling.generate_paris_nice_timeline_preview import (
    DEFAULT_INPUT,
    DEFAULT_PHOTOS_DIR,
    HEIGHT,
    WIDTH,
    load_entries,
    render_card,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "cycling" / "paris_nice" / "paris_nice_timeline_postwar_1946_2025.mp4"
DEFAULT_AUDIO = PROJECT_ROOT / "data" / "raw" / "audio" / "audio.mp3"

TOTAL_DURATION = 240.0
HOLD_START = 5.0
HOLD_END = 15.0
FPS = 120
FINAL_AUDIO_FADE_OUT = 10.0
LOOP_CROSSFADE = 5.0


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


def render_video(output_path: Path, input_csv: Path, photos_dir: Path, audio_path: Path, cards_visible: int, fps: int) -> None:
    entries = load_entries(input_csv, start_year=None, end_year=None)
    if not entries:
        raise RuntimeError("No Paris-Nice entries to render.")
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    gap = 10
    card_w = (WIDTH - gap * (cards_visible - 1)) // cards_visible
    pitch = card_w + gap
    total_shift = max(0.0, (len(entries) - cards_visible) * pitch)
    card_arrays = [render_card(entry, card_w, HEIGHT, photos_dir) for entry in entries]

    xx = np.linspace(0, 1, WIDTH, dtype=np.float32)
    yy = np.linspace(0, 1, HEIGHT, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(xx, yy)
    sea = np.array([11, 96, 143], dtype=np.float32)
    sky = np.array([75, 165, 213], dtype=np.float32)
    sun = np.array([244, 194, 48], dtype=np.float32)
    mix = np.clip(0.58 * grid_y + 0.28 * (1.0 - grid_x), 0, 1)
    bg = np.clip(
        sea[None, None, :] * (1 - mix[..., None])
        + sky[None, None, :] * (0.55 * mix[..., None])
        + sun[None, None, :] * (0.45 * mix[..., None]),
        0,
        255,
    ).astype(np.uint8)

    scroll_duration = TOTAL_DURATION - HOLD_START - HOLD_END
    if scroll_duration <= 0:
        raise RuntimeError("Invalid timeline duration configuration.")

    def make_frame(t: float) -> np.ndarray:
        frame = bg.copy()
        if t <= HOLD_START:
            shift = 0.0
        elif t >= TOTAL_DURATION - HOLD_END:
            shift = total_shift
        else:
            progress = (t - HOLD_START) / scroll_duration
            shift = total_shift * progress
        for idx, card in enumerate(card_arrays):
            x = int(idx * pitch - shift)
            if x >= WIDTH or x + card_w <= 0:
                continue
            src_x0 = 0 if x >= 0 else -x
            dst_x0 = 0 if x < 0 else x
            visible_w = min(card_w - src_x0, WIDTH - dst_x0)
            frame[:, dst_x0 : dst_x0 + visible_w] = card[:, src_x0 : src_x0 + visible_w]
        return frame

    clip = VideoClip(make_frame, duration=TOTAL_DURATION)
    audio_clip, keep_alive = build_audio_track(audio_path, TOTAL_DURATION)
    clip = clip.with_audio(audio_clip)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    clip.write_videofile(str(output_path), fps=fps, codec="libx264", audio_codec="aac")

    clip.close()
    audio_clip.close()
    for item in keep_alive:
        item.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the final Paris-Nice timeline MP4.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--photos-dir", type=Path, default=DEFAULT_PHOTOS_DIR)
    parser.add_argument("--audio", type=Path, default=DEFAULT_AUDIO)
    parser.add_argument("--cards-visible", type=int, default=4)
    parser.add_argument("--fps", type=int, default=FPS)
    args = parser.parse_args()

    render_video(
        output_path=args.output,
        input_csv=args.input,
        photos_dir=args.photos_dir,
        audio_path=args.audio,
        cards_visible=args.cards_visible,
        fps=args.fps,
    )
    print(f"[video_generator] Paris-Nice final timeline generated -> {args.output}")


if __name__ == "__main__":
    main()
