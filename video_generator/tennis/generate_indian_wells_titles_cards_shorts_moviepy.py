from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from moviepy import VideoClip
from PIL import Image, ImageDraw, ImageOps

from video_generator.generate_atp_vertical_timeline_moviepy import load_entries
from video_generator.tennis.generate_atp_shorts_timeline_moviepy import (
    DEFAULT_AUDIO,
    DEFAULT_INPUT,
    DEFAULT_PHOTOS_DIR,
    FPS,
    HEIGHT,
    HOLD_END,
    HOLD_START,
    TOTAL_DURATION,
    WIDTH,
    _fit_font,
    _load_font,
    _resolve_player_image,
    _truncate_to_width,
    build_audio_track,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "tennis" / "indian_wells_titles_cards_shorts.mp4"
VISIBLE_CARDS = 3
CARD_W = 304
CARD_H = 1368
CARD_GAP = 24


@dataclass(frozen=True)
class PlayerCard:
    player_name: str
    titles: int
    years: list[int]
    latest_year: int
    image_path: str


def _draw_text(draw: ImageDraw.ImageDraw, x: int, y: int, text: str, font, fill: str, anchor: str = "la") -> None:
    draw.text((x, y), text, font=font, fill=fill, anchor=anchor)


def _wrap_lines(draw: ImageDraw.ImageDraw, text: str, font, max_width: int, max_lines: int) -> list[str]:
    words = text.split()
    if not words:
        return []
    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        if draw.textbbox((0, 0), candidate, font=font)[2] <= max_width:
            current = candidate
        else:
            lines.append(current)
            current = word
            if len(lines) == max_lines - 1:
                break
    if len(lines) < max_lines:
        remaining = words[len(" ".join(lines + [current]).split()):]
        tail = " ".join([current] + remaining).strip()
        lines.append(_truncate_to_width(draw, tail, font, max_width))
    return lines[:max_lines]


def build_player_cards(input_csv: Path) -> list[PlayerCard]:
    entries = load_entries(input_csv)
    grouped: dict[str, list] = {}
    for entry in entries:
        grouped.setdefault(entry.player_name, []).append(entry)

    cards: list[PlayerCard] = []
    for player_name, player_entries in grouped.items():
        player_entries = sorted(player_entries, key=lambda item: item.year)
        years = [entry.year for entry in player_entries]
        latest = player_entries[-1]
        cards.append(
            PlayerCard(
                player_name=player_name,
                titles=len(years),
                years=years,
                latest_year=years[-1],
                image_path=latest.image_path,
            )
        )

    cards.sort(key=lambda item: (item.titles, item.latest_year, item.player_name))
    return cards


def _render_track_card(card: PlayerCard, photos_dir: Path) -> Image.Image:
    xx = np.linspace(0, 1, CARD_W, dtype=np.float32)
    yy = np.linspace(0, 1, CARD_H, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(xx, yy)
    sand = np.array([214, 178, 113], dtype=np.float32)
    cream = np.array([244, 238, 225], dtype=np.float32)
    court = np.array([33, 94, 113], dtype=np.float32)
    deep = np.array([10, 28, 36], dtype=np.float32)
    mix = np.clip(0.56 * grid_y + 0.12 * grid_x, 0, 1)
    bg = np.clip(
        cream[None, None, :] * (1.0 - mix[..., None])
        + sand[None, None, :] * (0.72 * mix[..., None])
        + court[None, None, :] * (0.20 * (1.0 - grid_y[..., None]))
        + deep[None, None, :] * (0.12 * (1.0 - grid_y[..., None])),
        0,
        255,
    ).astype(np.uint8)
    frame = Image.fromarray(np.dstack([bg, np.full((CARD_H, CARD_W), 255, dtype=np.uint8)]), mode="RGBA")
    draw = ImageDraw.Draw(frame, "RGBA")

    photo_rect = (18, 106, CARD_W - 18, 860)
    source_path = _resolve_player_image(card.image_path, card.player_name, photos_dir)
    if source_path:
        try:
            photo = ImageOps.exif_transpose(Image.open(source_path)).convert("RGB")
            photo = ImageOps.fit(
                photo,
                (photo_rect[2] - photo_rect[0], photo_rect[3] - photo_rect[1]),
                method=Image.Resampling.LANCZOS,
                centering=(0.5, 0.18),
            )
        except Exception:
            photo = Image.new("RGB", (photo_rect[2] - photo_rect[0], photo_rect[3] - photo_rect[1]), "#d6d6d6")
    else:
        photo = Image.new("RGB", (photo_rect[2] - photo_rect[0], photo_rect[3] - photo_rect[1]), "#d6d6d6")

    photo_rgba = photo.convert("RGBA")
    mask = Image.new("L", photo_rgba.size, 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, photo_rgba.width, photo_rgba.height), radius=40, fill=255)
    photo_rgba.putalpha(mask)
    frame.alpha_composite(photo_rgba, (photo_rect[0], photo_rect[1]))

    shadow = Image.new("RGBA", (photo_rect[2] - photo_rect[0], 350), (0, 0, 0, 0))
    shadow_draw = ImageDraw.Draw(shadow, "RGBA")
    for i in range(350):
        alpha = int(220 * (i / 349) ** 1.7)
        shadow_draw.line((0, i, shadow.width, i), fill=(5, 12, 16, alpha))
    frame.alpha_composite(shadow, (photo_rect[0], photo_rect[3] - 350))

    label_font = _load_font(22, bold=True)
    badge_font = _load_font(56, bold=True)
    name_font = _fit_font(draw, card.player_name.upper(), CARD_W - 52, 42, 22)
    body_font = _load_font(26, bold=True)
    years_font = _fit_font(draw, ", ".join(str(year) for year in card.years), CARD_W - 56, 26, 16)
    footer_font = _load_font(18, bold=True)

    _draw_text(draw, 22, 26, "INDIAN WELLS", label_font, "#2e7685")
    draw.rounded_rectangle((22, 60, 146, 66), radius=3, fill=(229, 191, 101, 255))
    title_badge = (CARD_W - 116, 22, CARD_W - 20, 96)
    draw.rounded_rectangle(title_badge, radius=24, fill=(17, 48, 57, 236))
    _draw_text(draw, (title_badge[0] + title_badge[2]) // 2, 58, str(card.titles), badge_font, "#f9f0db", anchor="mm")
    _draw_text(draw, title_badge[0] + 14, 104, "TITLES", _load_font(16, bold=True), "#17343d")

    name_lines = _wrap_lines(draw, card.player_name.upper(), name_font, CARD_W - 42, 3)
    base_name_y = 734
    for idx, line in enumerate(name_lines):
        _draw_text(draw, 22, base_name_y + idx * 38, line, name_font, "#fff7ea")

    info_rect = (18, 880, CARD_W - 18, 1228)
    draw.rounded_rectangle(info_rect, radius=40, fill=(18, 56, 66, 236))
    draw.rounded_rectangle((18, 880, 34, 1228), radius=14, fill=(225, 186, 96, 255))

    _draw_text(draw, 46, 918, "CHAMPION YEARS", _load_font(18, bold=True), "#e6c274")
    years_text = ", ".join(str(year) for year in card.years)
    for idx, line in enumerate(_wrap_lines(draw, years_text, years_font, info_rect[2] - info_rect[0] - 56, 5)):
        _draw_text(draw, 46, 960 + idx * 34, line, years_font, "#fff8ee")

    stat_box = (46, 1138, CARD_W - 34, 1208)
    draw.rounded_rectangle(stat_box, radius=24, fill=(240, 232, 213, 255))
    _draw_text(draw, stat_box[0] + 16, stat_box[1] + 12, "LAST TITLE", _load_font(15, bold=True), "#8a6a38")
    _draw_text(draw, stat_box[0] + 16, stat_box[1] + 34, str(card.latest_year), body_font, "#245b68")

    footer_rect = (18, 1250, CARD_W - 18, 1318)
    draw.rounded_rectangle(footer_rect, radius=24, fill=(241, 234, 218, 236))
    footer_text = "DESERT KINGS"
    _draw_text(draw, CARD_W // 2, 1283, footer_text, footer_font, "#2d6774", anchor="ma")

    return frame


def _make_background() -> Image.Image:
    xx = np.linspace(0, 1, WIDTH, dtype=np.float32)
    yy = np.linspace(0, 1, HEIGHT, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(xx, yy)
    sand = np.array([214, 178, 113], dtype=np.float32)
    cream = np.array([244, 238, 225], dtype=np.float32)
    court = np.array([25, 82, 100], dtype=np.float32)
    deep = np.array([8, 21, 28], dtype=np.float32)
    mix = np.clip(0.66 * grid_y + 0.14 * grid_x, 0, 1)
    bg = np.clip(
        cream[None, None, :] * (1.0 - mix[..., None])
        + sand[None, None, :] * (0.68 * mix[..., None])
        + court[None, None, :] * (0.18 * (1.0 - grid_y[..., None]))
        + deep[None, None, :] * (0.16 * (1.0 - grid_y[..., None])),
        0,
        255,
    ).astype(np.uint8)
    frame = Image.fromarray(np.dstack([bg, np.full((HEIGHT, WIDTH), 255, dtype=np.uint8)]), mode="RGBA")
    overlay = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay, "RGBA")
    od.rounded_rectangle((40, 44, WIDTH - 40, HEIGHT - 44), radius=42, outline=(255, 255, 255, 18), width=2)
    od.ellipse((170, 160, WIDTH - 170, 820), outline=(229, 191, 101, 18), width=3)
    frame.alpha_composite(overlay)
    return frame


def render_video(cards: list[PlayerCard], output_path: Path, photos_dir: Path, audio_path: Path, fps: int, duration: float) -> Path:
    if not cards:
        raise RuntimeError("No player cards to render.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    photos_dir.mkdir(parents=True, exist_ok=True)
    card_images = [_render_track_card(card, photos_dir) for card in cards]
    background = _make_background()
    header_title_font = _load_font(54, bold=True)
    header_subtitle_font = _load_font(24, bold=False)
    rail_title_font = _load_font(28, bold=True)
    track_w = len(card_images) * CARD_W + max(0, len(card_images) - 1) * CARD_GAP
    track = Image.new("RGBA", (track_w, CARD_H), (0, 0, 0, 0))
    x = 0
    for image in card_images:
        track.alpha_composite(image, (x, 0))
        x += CARD_W + CARD_GAP

    scroll_duration = duration - HOLD_START - HOLD_END
    visible_w = VISIBLE_CARDS * CARD_W + (VISIBLE_CARDS - 1) * CARD_GAP
    rail_left = (WIDTH - visible_w) // 2
    rail_top = 376
    max_offset = max(0, track_w - visible_w)

    def make_frame(t: float) -> np.ndarray:
        frame = background.copy()
        draw = ImageDraw.Draw(frame, "RGBA")
        draw.text((64, 72), "INDIAN WELLS TITLES", font=header_title_font, fill="#f6f2e8")
        draw.text((66, 136), "champions sorted from one title to the record holders", font=header_subtitle_font, fill="#d8e5ec")
        draw.rounded_rectangle((60, 188, WIDTH - 60, 320), radius=34, fill=(11, 35, 51, 214), outline=(255, 255, 255, 16), width=1)
        draw.text((WIDTH // 2, 222), "3 CARDS ON SCREEN", font=rail_title_font, fill="#e6c274", anchor="ma")
        draw.text((WIDTH // 2, 266), "horizontal scroll through every Indian Wells champion", font=header_subtitle_font, fill="#f5f0e6", anchor="ma")

        if t <= HOLD_START:
            offset = 0.0
        elif t >= duration - HOLD_END:
            offset = float(max_offset)
        else:
            alpha = (t - HOLD_START) / scroll_duration
            alpha = alpha * alpha * (3.0 - 2.0 * alpha)
            offset = max_offset * alpha
        viewport = track.crop((int(offset), 0, int(offset) + visible_w, CARD_H))
        frame.alpha_composite(viewport, (rail_left, rail_top))
        return np.array(frame.convert("RGB"))

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
    parser = argparse.ArgumentParser(description="Generate Indian Wells player cards Shorts video.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--photos-dir", type=Path, default=DEFAULT_PHOTOS_DIR)
    parser.add_argument("--audio", type=Path, default=DEFAULT_AUDIO)
    parser.add_argument("--fps", type=int, default=FPS)
    parser.add_argument("--duration", type=float, default=TOTAL_DURATION)
    parser.add_argument("--last-n", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cards = build_player_cards(args.input)
    if args.last_n is not None and args.last_n > 0:
        cards = cards[-args.last_n :]
    output = render_video(
        cards=cards,
        output_path=args.output,
        photos_dir=args.photos_dir,
        audio_path=args.audio,
        fps=args.fps,
        duration=args.duration,
    )
    print(f"[video_generator] Indian Wells player cards Shorts generated -> {output}")


if __name__ == "__main__":
    main()
