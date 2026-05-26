from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from moviepy import VideoClip
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageOps

from video_generator.generate_ucl_barchart_race_moviepy import DEFAULT_AUDIO, build_audio_track
from video_generator.tennis.generate_atp_shorts_timeline_moviepy import HEIGHT, WIDTH, _fit_font, _load_font, _truncate_to_width


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "basketball" / "nba_playoff_wins_without_title_refstyle_shorts.mp4"
DEFAULT_PHOTOS_DIR = PROJECT_ROOT / "data" / "raw" / "player_photos"
DEFAULT_LOGOS_DIR = PROJECT_ROOT / "data" / "raw" / "nba_team_logos"
NBA_LOGO = PROJECT_ROOT / "data" / "raw" / "nba_logo.png"

TOTAL_DURATION = 60.0
HOLD_START_SECONDS = 5.0
HOLD_END_SECONDS = 5.0
FPS = 60

CARD_W = 720
CARD_H = 1560
CARD_GAP = 0
VISIBLE_CARDS = 1.5
CARD_TOP = 246
CARD_RADIUS = 38
PHOTO_H = 900
NAME_H = 160

TITLE = "PLAYOFF WINS, NO RING"
SOURCE_NOTE = "Players with the most postseason wins without an NBA title"


@dataclass(frozen=True)
class PlayerEntry:
    rank: int
    player_name: str
    team_name: str
    team_slug: str
    wins: int
    accent_color: str
    secondary_color: str


ENTRIES: tuple[PlayerEntry, ...] = (
    PlayerEntry(1, "James Harden", "Cleveland Cavaliers", "cleveland_cavaliers", 98, "#6f263d", "#ffb81c"),
    PlayerEntry(2, "Karl Malone", "Utah Jazz", "utah_jazz", 98, "#2f0a66", "#00a3e0"),
    PlayerEntry(3, "John Stockton", "Utah Jazz", "utah_jazz", 89, "#2f0a66", "#00a3e0"),
    PlayerEntry(4, "Sam Perkins", "Dallas Mavericks", "dallas_mavericks", 88, "#00538c", "#b8c4ca"),
    PlayerEntry(5, "Derrick McKey", "Indiana Pacers", "indiana_pacers", 77, "#002d62", "#fdbb30"),
    PlayerEntry(6, "Chris Paul", "LA Clippers", "la_clippers", 76, "#1d428a", "#c8102e"),
    PlayerEntry(7, "Jeff Hornacek", "Utah Jazz", "utah_jazz", 76, "#2f0a66", "#00a3e0"),
    PlayerEntry(8, "Reggie Miller", "Indiana Pacers", "indiana_pacers", 76, "#002d62", "#fdbb30"),
    PlayerEntry(9, "Dale Davis", "Indiana Pacers", "indiana_pacers", 75, "#002d62", "#fdbb30"),
    PlayerEntry(10, "George Hill", "Indiana Pacers", "indiana_pacers", 75, "#002d62", "#fdbb30"),
)


def _hex_to_rgb(value: str) -> tuple[int, int, int]:
    value = value.strip().lstrip("#")
    return tuple(int(value[i : i + 2], 16) for i in (0, 2, 4))


def _slug_name(name: str) -> str:
    value = name.lower().replace(".", "").replace("'", "").replace("-", "_").replace(" ", "_")
    while "__" in value:
        value = value.replace("__", "_")
    return value.strip("_")


def _resolve_player_image(name: str, photos_dir: Path) -> Path | None:
    slug = _slug_name(name)
    for suffix in (".jpg", ".jpeg", ".png", ".webp"):
        candidate = photos_dir / f"{slug}{suffix}"
        if candidate.exists():
            return candidate
    return None


def _logo_path(entry: PlayerEntry, logos_dir: Path) -> Path | None:
    candidate = logos_dir / f"{entry.team_slug}.png"
    return candidate if candidate.exists() else None


def _load_logo(path: Path | None, size: int) -> Image.Image | None:
    if path is None:
        return None
    try:
        logo = ImageOps.exif_transpose(Image.open(path)).convert("RGBA")
    except Exception:
        return None
    logo.thumbnail((size, size), Image.Resampling.LANCZOS)
    canvas = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    canvas.alpha_composite(logo, ((size - logo.width) // 2, (size - logo.height) // 2))
    return canvas


def _make_background() -> Image.Image:
    xx = np.linspace(0, 1, WIDTH, dtype=np.float32)
    yy = np.linspace(0, 1, HEIGHT, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(xx, yy)

    charcoal = np.array([18, 20, 24], dtype=np.float32)
    slate = np.array([26, 34, 46], dtype=np.float32)
    wine = np.array([72, 20, 48], dtype=np.float32)
    blue = np.array([12, 72, 122], dtype=np.float32)
    gold = np.array([245, 196, 78], dtype=np.float32)

    base_mix = np.clip(0.72 * grid_y + 0.10 * grid_x, 0, 1)
    left_glow = np.exp(-(((grid_x - 0.15) / 0.22) ** 2 + ((grid_y - 0.38) / 0.26) ** 2))
    right_glow = np.exp(-(((grid_x - 0.88) / 0.25) ** 2 + ((grid_y - 0.62) / 0.28) ** 2))
    top_glow = np.exp(-(((grid_x - 0.55) / 0.46) ** 2 + ((grid_y - 0.10) / 0.18) ** 2))

    img = np.clip(
        charcoal[None, None, :] * (1.0 - base_mix[..., None])
        + slate[None, None, :] * (0.78 * base_mix[..., None])
        + wine[None, None, :] * (0.24 * left_glow[..., None])
        + blue[None, None, :] * (0.22 * right_glow[..., None])
        + gold[None, None, :] * (0.07 * top_glow[..., None]),
        0,
        255,
    ).astype(np.uint8)
    frame = Image.fromarray(img, mode="RGB").convert("RGBA")

    lines = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    draw = ImageDraw.Draw(lines, "RGBA")
    for y in range(220, HEIGHT, 168):
        draw.line((46, y, WIDTH - 46, y), fill=(255, 255, 255, 9), width=1)
    draw.rounded_rectangle((42, 82, WIDTH - 42, HEIGHT - 72), radius=52, outline=(255, 255, 255, 15), width=2)
    frame.alpha_composite(lines)
    return frame


def _make_placeholder(frame: Image.Image, draw: ImageDraw.ImageDraw, rect: tuple[int, int, int, int], entry: PlayerEntry, logo: Image.Image | None) -> None:
    accent = _hex_to_rgb(entry.accent_color)
    draw.rounded_rectangle(rect, radius=32, fill=(29, 34, 43, 255), outline=(*accent, 160), width=2)
    if logo is not None:
        faded = logo.copy()
        alpha = faded.getchannel("A").point(lambda value: int(value * 0.32))
        faded.putalpha(alpha)
        frame.alpha_composite(faded, ((rect[0] + rect[2] - faded.width) // 2, rect[1] + 140))
    initials = "".join(part[0] for part in entry.player_name.split()[:2]).upper()
    font = _fit_font(draw, initials, rect[2] - rect[0] - 40, 140, 64, bold=True)
    draw.text(((rect[0] + rect[2]) // 2, (rect[1] + rect[3]) // 2 + 40), initials, font=font, fill="#f5f7fb", anchor="mm")


def _wrap_name(draw: ImageDraw.ImageDraw, text: str, max_width: int) -> tuple[ImageFont.ImageFont, list[str]]:
    font = _fit_font(draw, text, max_width, 70, 34, bold=True)
    if draw.textbbox((0, 0), text, font=font)[2] <= max_width:
        return font, [text]

    parts = [part for part in text.split() if part]
    if len(parts) <= 1:
        return font, [_truncate_to_width(draw, text, font, max_width)]
    first_line = " ".join(parts[:-1])
    second_line = parts[-1]
    while draw.textbbox((0, 0), first_line, font=font)[2] > max_width and len(first_line) > 1:
        pieces = first_line.split()
        if len(pieces) <= 1:
            break
        second_line = " ".join([pieces[-1], second_line])
        first_line = " ".join(pieces[:-1])
    return font, [
        _truncate_to_width(draw, first_line, font, max_width),
        _truncate_to_width(draw, second_line, font, max_width),
    ]


def _draw_centered_lines(draw: ImageDraw.ImageDraw, center_x: int, top_y: int, lines: list[str], font, fill: str) -> None:
    bboxes = [draw.textbbox((0, 0), line, font=font) for line in lines]
    y = top_y
    for idx, line in enumerate(lines):
        draw.text((center_x, y), line, font=font, fill=fill, anchor="ma")
        y += (bboxes[idx][3] - bboxes[idx][1]) + 4


def _render_card(entry: PlayerEntry, photos_dir: Path, logos_dir: Path) -> Image.Image:
    frame = Image.new("RGBA", (CARD_W, CARD_H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(frame, "RGBA")
    accent = _hex_to_rgb(entry.accent_color)
    secondary = _hex_to_rgb(entry.secondary_color)
    logo = _load_logo(_logo_path(entry, logos_dir), 180)

    shadow = Image.new("RGBA", frame.size, (0, 0, 0, 0))
    sdraw = ImageDraw.Draw(shadow, "RGBA")
    sdraw.rounded_rectangle((12, 12, CARD_W - 12, CARD_H - 12), radius=CARD_RADIUS, fill=(0, 0, 0, 120))
    shadow = shadow.filter(ImageFilter.GaussianBlur(radius=18))
    frame.alpha_composite(shadow)

    draw.rounded_rectangle((0, 0, CARD_W - 1, CARD_H - 1), radius=CARD_RADIUS, fill=(23, 28, 38, 255), outline=(*accent, 245), width=4)
    draw.rounded_rectangle((12, 12, CARD_W - 13, CARD_H - 13), radius=CARD_RADIUS - 6, outline=(255, 255, 255, 18), width=1)

    glow = Image.new("RGBA", frame.size, (0, 0, 0, 0))
    gdraw = ImageDraw.Draw(glow, "RGBA")
    gdraw.ellipse((CARD_W - 360, -120, CARD_W + 120, 270), fill=(*secondary, 48))
    gdraw.ellipse((-140, 230, 280, 650), fill=(*accent, 40))
    gdraw.ellipse((CARD_W - 480, CARD_H - 380, CARD_W + 140, CARD_H + 140), fill=(*secondary, 26))
    frame.alpha_composite(glow.filter(ImageFilter.GaussianBlur(radius=44)))

    photo_rect = (18, 18, CARD_W - 18, PHOTO_H)
    photo_path = _resolve_player_image(entry.player_name, photos_dir)
    if photo_path is not None:
        try:
            photo = ImageOps.exif_transpose(Image.open(photo_path)).convert("RGB")
            photo = ImageOps.fit(photo, (photo_rect[2] - photo_rect[0], photo_rect[3] - photo_rect[1]), method=Image.Resampling.LANCZOS, centering=(0.5, 0.16))
            photo = ImageEnhance.Contrast(photo).enhance(1.06)
            photo = ImageEnhance.Brightness(photo).enhance(1.02)
            frame.alpha_composite(photo.convert("RGBA"), (photo_rect[0], photo_rect[1]))
        except Exception:
            _make_placeholder(frame, draw, photo_rect, entry, logo)
    else:
        _make_placeholder(frame, draw, photo_rect, entry, logo)

    fade = Image.new("RGBA", frame.size, (0, 0, 0, 0))
    fdraw = ImageDraw.Draw(fade, "RGBA")
    fdraw.rectangle((photo_rect[0], photo_rect[3] - 220, photo_rect[2], photo_rect[3]), fill=(8, 13, 20, 156))
    frame.alpha_composite(fade.filter(ImageFilter.GaussianBlur(radius=24)))

    rank_box = (30, 38, 126, 126)
    draw.rounded_rectangle(rank_box, radius=24, fill=(8, 12, 18, 236), outline=(*secondary, 140), width=2)
    rank_font = _fit_font(draw, str(entry.rank), 78, 58, 28, bold=True)
    draw.text(((rank_box[0] + rank_box[2]) // 2, 76), str(entry.rank), font=rank_font, fill="#f7fbff", anchor="mm")
    draw.text((78, 112), "RANG", font=_load_font(14, bold=True), fill="#dce6f1", anchor="mm")

    if logo is not None:
        frame.alpha_composite(_load_logo(_logo_path(entry, logos_dir), 76), (CARD_W - 112, 32))

    name_band = (18, PHOTO_H - 22, CARD_W - 18, PHOTO_H + NAME_H)
    band = Image.new("RGBA", frame.size, (0, 0, 0, 0))
    bdraw = ImageDraw.Draw(band, "RGBA")
    bdraw.rounded_rectangle(name_band, radius=28, fill=(12, 18, 30, 238), outline=(255, 255, 255, 13), width=1)
    bdraw.rounded_rectangle((name_band[0] + 10, name_band[1] + 10, name_band[2] - 10, name_band[3] - 10), radius=22, outline=(255, 255, 255, 15), width=1)
    frame.alpha_composite(band)

    name_font, lines = _wrap_name(draw, entry.player_name.upper(), CARD_W - 92)
    _draw_centered_lines(draw, CARD_W // 2, PHOTO_H + 14, lines[:2], name_font, "#fff8ef")

    bottom_top = PHOTO_H + NAME_H - 10
    bottom_rect = (18, bottom_top, CARD_W - 18, CARD_H - 18)
    panel = Image.new("RGBA", frame.size, (0, 0, 0, 0))
    pdraw = ImageDraw.Draw(panel, "RGBA")
    pdraw.rounded_rectangle(bottom_rect, radius=28, fill=(37, 43, 54, 248), outline=(*secondary, 150), width=2)
    pdraw.rounded_rectangle((bottom_rect[0] + 10, bottom_rect[1] + 10, bottom_rect[2] - 10, bottom_rect[3] - 10), radius=22, outline=(255, 255, 255, 12), width=1)
    frame.alpha_composite(panel)

    summary_box = (46, bottom_top + 26, CARD_W - 46, bottom_top + 250)
    draw.rounded_rectangle(summary_box, radius=24, fill=(244, 246, 249, 232), outline=(255, 255, 255, 20), width=1)
    draw.text((summary_box[0] + 18, summary_box[1] + 18), "PLAYOFF WINS", font=_load_font(18, bold=True), fill="#5a6471")
    value_font = _fit_font(draw, str(entry.wins), 170, 104, 64, bold=True)
    value_box = (summary_box[0] + 18, summary_box[1] + 48, summary_box[0] + 186, summary_box[1] + 166)
    draw.rounded_rectangle(value_box, radius=20, fill=(*accent, 220), outline=(255, 255, 255, 18), width=1)
    draw.text(((value_box[0] + value_box[2]) // 2, (value_box[1] + value_box[3]) // 2 - 2), str(entry.wins), font=value_font, fill="#fffaf3", anchor="mm")

    if logo is not None:
        logo_small = _load_logo(_logo_path(entry, logos_dir), 92)
        frame.alpha_composite(logo_small, (summary_box[0] + 212, summary_box[1] + 56))
        team_x = summary_box[0] + 320
    else:
        team_x = summary_box[0] + 212
    team_font = _fit_font(draw, entry.team_name.upper(), summary_box[2] - team_x - 18, 30, 18, bold=True)
    draw.text((team_x, summary_box[1] + 68), entry.team_name.upper(), font=team_font, fill="#1c2530")
    draw.text((team_x, summary_box[1] + 108), "0 NBA TITLES", font=_load_font(17, bold=True), fill="#6a7682")

    context_box = (46, summary_box[3] + 18, CARD_W - 46, CARD_H - 88)
    draw.rounded_rectangle(context_box, radius=24, fill=(245, 230, 187, 228), outline=(255, 255, 255, 14), width=1)
    draw.text((context_box[0] + 18, context_box[1] + 18), "STAT", font=_load_font(18, bold=True), fill="#72562c")
    context_font = _fit_font(draw, "MOST POSTSEASON WINS WITHOUT A CHAMPIONSHIP", context_box[2] - context_box[0] - 36, 28, 16, bold=True)
    draw.text((context_box[0] + 18, context_box[1] + 56), "MOST POSTSEASON WINS WITHOUT A CHAMPIONSHIP", font=context_font, fill="#162230")
    draw.text((context_box[0] + 18, context_box[1] + 104), "Classement base sur la stat fournie", font=_load_font(17, bold=True), fill="#5d5f64")

    return frame


def render_video(output_path: Path, photos_dir: Path, logos_dir: Path, audio_path: Path, duration: float, fps: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    photos_dir.mkdir(parents=True, exist_ok=True)

    card_images = [_render_card(entry, photos_dir, logos_dir) for entry in ENTRIES]
    track_w = len(card_images) * CARD_W + max(0, len(card_images) - 1) * CARD_GAP
    track = Image.new("RGBA", (track_w, CARD_H), (0, 0, 0, 0))
    x = 0
    for image in card_images:
        track.alpha_composite(image, (x, 0))
        x += CARD_W + CARD_GAP

    viewport_src_w = int(round(VISIBLE_CARDS * CARD_W + max(0.0, VISIBLE_CARDS - 1.0) * CARD_GAP))
    max_offset = max(0, track_w - viewport_src_w)
    scroll_duration = max(0.1, duration - HOLD_START_SECONDS - HOLD_END_SECONDS)

    background = _make_background()
    header = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    hdraw = ImageDraw.Draw(header, "RGBA")
    hdraw.rounded_rectangle((36, 30, WIDTH - 36, 122), radius=28, fill=(8, 12, 18, 170), outline=(255, 255, 255, 20), width=1)
    hdraw.text((58, 48), TITLE, font=_load_font(45, bold=True), fill="#fff6ef")
    if NBA_LOGO.exists():
        nba_logo = _load_logo(NBA_LOGO, 58)
        if nba_logo is not None:
            header.alpha_composite(nba_logo, (WIDTH - 98, 47))

    footer = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    fdraw = ImageDraw.Draw(footer, "RGBA")
    fdraw.ellipse((-120, HEIGHT - 520, 440, HEIGHT + 120), fill=(245, 196, 78, 18))
    fdraw.ellipse((WIDTH - 520, HEIGHT - 420, WIDTH + 120, HEIGHT + 180), fill=(0, 83, 140, 20))
    footer = footer.filter(ImageFilter.GaussianBlur(radius=48))

    static_base = background.copy()
    static_base.alpha_composite(header)
    static_base.alpha_composite(footer)

    def make_frame(t: float) -> np.ndarray:
        if t <= HOLD_START_SECONDS:
            shift = 0.0
        elif t >= duration - HOLD_END_SECONDS:
            shift = float(max_offset)
        else:
            progress = (t - HOLD_START_SECONDS) / scroll_duration
            shift = max_offset * progress

        canvas = static_base.copy()
        viewport = track.crop((int(shift), 0, int(shift) + viewport_src_w, CARD_H))
        canvas.alpha_composite(viewport, (0, CARD_TOP))
        return np.array(canvas.convert("RGB"))

    clip = VideoClip(make_frame, duration=duration)
    if audio_path.exists():
        audio_clip, keep_alive = build_audio_track(audio_path, duration)
        clip = clip.with_audio(audio_clip)
    else:
        audio_clip, keep_alive = None, []

    clip.write_videofile(str(output_path), fps=fps, codec="libx264", audio_codec="aac" if audio_clip else None)
    clip.close()
    if audio_clip is not None:
        audio_clip.close()
    for item in keep_alive:
        item.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate NBA playoff wins without a title ref-style cards Shorts video.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--photos-dir", type=Path, default=DEFAULT_PHOTOS_DIR)
    parser.add_argument("--logos-dir", type=Path, default=DEFAULT_LOGOS_DIR)
    parser.add_argument("--audio", type=Path, default=DEFAULT_AUDIO)
    parser.add_argument("--duration", type=float, default=TOTAL_DURATION)
    parser.add_argument("--fps", type=int, default=FPS)
    args = parser.parse_args()

    render_video(
        output_path=args.output,
        photos_dir=args.photos_dir,
        logos_dir=args.logos_dir,
        audio_path=args.audio,
        duration=args.duration,
        fps=min(args.fps, 60),
    )
    print(f"[video_generator] NBA playoff wins without title ref-style Shorts generated -> {args.output}")


if __name__ == "__main__":
    main()
