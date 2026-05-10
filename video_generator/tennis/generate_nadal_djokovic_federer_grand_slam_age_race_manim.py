from __future__ import annotations

import argparse
import math
import subprocess
import sys
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

try:
    from manim import (
        BLACK,
        BLUE,
        Circle,
        Create,
        DecimalNumber,
        DOWN,
        FadeIn,
        FadeOut,
        Group,
        ImageMobject,
        LEFT,
        Line,
        ORIGIN,
        RIGHT,
        RoundedRectangle,
        Scene,
        Text,
        UP,
        VGroup,
        ValueTracker,
        always_redraw,
        config,
        config as manim_config,
    )

    MANIM_AVAILABLE = True
except Exception:
    MANIM_AVAILABLE = False
    Scene = object  # type: ignore[assignment]
    config = None  # type: ignore[assignment]
    manim_config = None  # type: ignore[assignment]

try:
    from moviepy import AudioFileClip, CompositeAudioClip, VideoFileClip
except Exception:  # pragma: no cover
    AudioFileClip = None  # type: ignore[assignment]
    CompositeAudioClip = None  # type: ignore[assignment]
    VideoFileClip = None  # type: ignore[assignment]


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ASSETS_DIR = PROJECT_ROOT / "assets"
FALLBACK_PHOTOS_DIR = PROJECT_ROOT / "data" / "raw" / "player_photos"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "tennis" / "manim_age_race"
DEFAULT_OUTPUT_STEM = "nadal_djokovic_federer_grand_slam_age_race"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "tennis" / "nadal_djokovic_federer_grand_slam_age_race_manim.mp4"
DEFAULT_AUDIO = PROJECT_ROOT / "data" / "raw" / "audio" / "Midnight_Grip_20260402_0828.mp3"

WIDTH = 1080
HEIGHT = 1920
FPS = 15

AGE_MIN = 17
AGE_MAX = 37

INTRO_DURATION = 2.0
AGE_STEP_DURATION = 1.25
OUTRO_DURATION = 4.0
TOTAL_DURATION = INTRO_DURATION + (AGE_MAX - AGE_MIN) * AGE_STEP_DURATION + OUTRO_DURATION

TITLE = "NADAL vs DJOKOVIC vs FEDERER"
SUBTITLE = "Grand Slam Titles by Age"
FOOTER = "Grand Slam titles by age"
CTA = "Who dominated by age?"

BACKGROUND_TOP = "#7E59FF"
BACKGROUND_MID = "#9A58F3"
BACKGROUND_BOTTOM = "#3A5CFF"

FEDERER_COLOR = "#F2C94C"
NADAL_COLOR = "#FF4B3A"
DJOKOVIC_COLOR = "#3E86FF"

PLAYER_TIE_ORDER = {"Roger Federer": 0, "Rafael Nadal": 1, "Novak Djokovic": 2}


if manim_config is not None:
    manim_config.pixel_width = WIDTH
    manim_config.pixel_height = HEIGHT
    manim_config.frame_width = 9
    manim_config.frame_height = 16
    manim_config.frame_rate = FPS
    manim_config.background_color = "#090B18"


@dataclass(frozen=True)
class Player:
    name: str
    short: str
    asset_name: str
    color: str
    title_counts: dict[int, int]


PLAYERS = [
    Player(
        name="Roger Federer",
        short="FEDERER",
        asset_name="federer.png",
        color=FEDERER_COLOR,
        title_counts={
            17: 0, 18: 0, 19: 0, 20: 0, 21: 1, 22: 3, 23: 6, 24: 7, 25: 10, 26: 12,
            27: 13, 28: 16, 29: 16, 30: 16, 31: 17, 32: 17, 33: 17, 34: 17, 35: 18,
            36: 20, 37: 20,
        },
    ),
    Player(
        name="Rafael Nadal",
        short="NADAL",
        asset_name="nadal.png",
        color=NADAL_COLOR,
        title_counts={
            17: 0, 18: 0, 19: 1, 20: 2, 21: 3, 22: 5, 23: 6, 24: 9, 25: 10, 26: 11,
            27: 13, 28: 14, 29: 14, 30: 14, 31: 16, 32: 17, 33: 19, 34: 20, 35: 20,
            36: 22, 37: 22,
        },
    ),
    Player(
        name="Novak Djokovic",
        short="DJOKOVIC",
        asset_name="djokovic.png",
        color=DJOKOVIC_COLOR,
        title_counts={
            17: 0, 18: 0, 19: 0, 20: 1, 21: 1, 22: 1, 23: 1, 24: 4, 25: 5, 26: 6,
            27: 7, 28: 10, 29: 12, 30: 12, 31: 14, 32: 16, 33: 17, 34: 20, 35: 21,
            36: 24, 37: 24,
        },
    ),
]


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def smoothstep(value: float) -> float:
    value = clamp(value)
    return value * value * (3.0 - 2.0 * value)


def ease_out_cubic(value: float) -> float:
    value = clamp(value)
    return 1.0 - (1.0 - value) ** 3


def hex_to_rgb(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    return tuple(int(value[i : i + 2], 16) for i in (0, 2, 4))


def mix_color(color: str, target: tuple[int, int, int], amount: float) -> tuple[int, int, int]:
    amount = clamp(amount)
    r, g, b = hex_to_rgb(color)
    return (
        int(r + (target[0] - r) * amount),
        int(g + (target[1] - g) * amount),
        int(b + (target[2] - b) * amount),
    )


def text_on(color: str) -> str:
    r, g, b = hex_to_rgb(color)
    luminance = (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255.0
    return "#08111E" if luminance > 0.66 else "#F7F9FC"


def build_background() -> np.ndarray:
    xx = np.linspace(0, 1, WIDTH, dtype=np.float32)
    yy = np.linspace(0, 1, HEIGHT, dtype=np.float32)
    x, y = np.meshgrid(xx, yy)

    top = np.array(hex_to_rgb(BACKGROUND_TOP), dtype=np.float32)
    mid = np.array(hex_to_rgb(BACKGROUND_MID), dtype=np.float32)
    bottom = np.array(hex_to_rgb(BACKGROUND_BOTTOM), dtype=np.float32)
    rose = np.array([232, 157, 255], dtype=np.float32)
    court = np.array([34, 135, 102], dtype=np.float32)
    ball = np.array([255, 228, 103], dtype=np.float32)

    base = np.clip(
        top[None, None, :] * (1.0 - y[..., None])
        + bottom[None, None, :] * y[..., None],
        0,
        255,
    )

    glow_center = np.exp(-(((x - 0.52) / 0.32) ** 2 + ((y - 0.20) / 0.18) ** 2))
    glow_left = np.exp(-(((x - 0.16) / 0.22) ** 2 + ((y - 0.45) / 0.26) ** 2))
    glow_right = np.exp(-(((x - 0.84) / 0.20) ** 2 + ((y - 0.50) / 0.24) ** 2))
    glow_bottom = np.exp(-(((x - 0.52) / 0.52) ** 2 + ((y - 0.88) / 0.14) ** 2))
    court_glow = np.exp(-(((x - 0.52) / 0.58) ** 2 + ((y - 0.60) / 0.28) ** 2))
    ball_glow = np.exp(-(((x - 0.80) / 0.10) ** 2 + ((y - 0.18) / 0.08) ** 2))

    image = np.clip(
        base
        + rose[None, None, :] * (0.22 * glow_center[..., None] + 0.10 * glow_right[..., None])
        + mid[None, None, :] * (0.16 * glow_left[..., None])
        + court[None, None, :] * (0.16 * court_glow[..., None])
        + ball[None, None, :] * (0.32 * ball_glow[..., None])
        + np.array([255, 255, 255], dtype=np.float32)[None, None, :] * (0.05 * glow_bottom[..., None]),
        0,
        255,
    ).astype(np.uint8)

    # Add a soft dark vignette.
    vignette = np.exp(-(((x - 0.5) / 0.58) ** 2 + ((y - 0.55) / 0.68) ** 2))
    image = np.clip(image * (0.72 + 0.28 * vignette[..., None]), 0, 255).astype(np.uint8)
    return image


def resolve_asset(asset_name: str) -> Path | None:
    candidates = [
        ASSETS_DIR / asset_name,
        FALLBACK_PHOTOS_DIR / asset_name.replace(".png", ".jpg"),
        FALLBACK_PHOTOS_DIR / asset_name.replace(".png", ".jpeg"),
        FALLBACK_PHOTOS_DIR / asset_name.replace(".png", ".webp"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def make_avatar_image(player: Player, size: int = 640) -> np.ndarray:
    asset = resolve_asset(player.asset_name)
    if asset is not None:
        with Image.open(asset) as img:
            source = img.convert("RGBA")
        source = ImageOps.fit(source, (size, size), method=Image.Resampling.LANCZOS, centering=(0.5, 0.24))
    else:
        source = Image.new("RGBA", (size, size), (16, 18, 30, 255))
        draw = ImageDraw.Draw(source)
        draw.rounded_rectangle((10, 10, size - 10, size - 10), radius=70, fill=(20, 24, 38, 255))
        initials_font = ImageFont.load_default()
        initials = "".join(part[0] for part in player.short.split()).upper()[:2]
        draw.text((size // 2, size // 2), initials, font=initials_font, fill=(255, 255, 255, 255), anchor="mm")

    # Create a rounded-square avatar with a soft border.
    mask = Image.new("L", (size, size), 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, size - 1, size - 1), radius=90, fill=255)
    out = Image.new("RGBA", (size + 24, size + 24), (0, 0, 0, 0))
    drawer = ImageDraw.Draw(out)
    drawer.rounded_rectangle((0, 0, size + 23, size + 23), radius=108, fill=(255, 255, 255, 240))
    drawer.rounded_rectangle((4, 4, size + 19, size + 19), radius=100, fill=(*hex_to_rgb(player.color), 255))
    drawer.rounded_rectangle((10, 10, size + 13, size + 13), radius=92, fill=(12, 18, 30, 255))
    out.paste(source, (12, 12), mask)
    return np.array(out)


def count_for_age(player: Player, age: float) -> float:
    low = int(math.floor(age))
    high = min(AGE_MAX, low + 1)
    alpha = smoothstep(age - low)
    low_count = player.title_counts[low]
    high_count = player.title_counts[high]
    return low_count + (high_count - low_count) * alpha


def rank_positions(age: float) -> dict[str, float]:
    low = int(math.floor(age))
    high = min(AGE_MAX, low + 1)
    alpha = smoothstep(age - low)

    counts_low = {player.name: player.title_counts[low] for player in PLAYERS}
    counts_high = {player.name: player.title_counts[high] for player in PLAYERS}

    def order_for(counts: dict[str, float]) -> dict[str, int]:
        order = sorted(
            PLAYERS,
            key=lambda player: (-counts[player.name], PLAYER_TIE_ORDER[player.name]),
        )
        return {player.name: idx for idx, player in enumerate(order)}

    ranks_low = order_for(counts_low)
    ranks_high = order_for(counts_high)

    return {
        player.name: ranks_low[player.name] + (ranks_high[player.name] - ranks_low[player.name]) * alpha
        for player in PLAYERS
    }


def interpolated_counts(age: float) -> dict[str, float]:
    return {player.name: count_for_age(player, age) for player in PLAYERS}


def build_title() -> Text:
    title = Text(
        TITLE,
        font_size=46,
        weight="BOLD",
        color="#F8FBFF",
        font="DejaVu Sans",
    )
    title.set_stroke(BLACK, width=6, background=True)
    return title


def build_subtitle() -> VGroup:
    pill = RoundedRectangle(width=4.8, height=0.62, corner_radius=0.25)
    pill.set_fill("#8E56F5", opacity=0.96)
    pill.set_stroke("#FFFFFF", width=1.5, opacity=0.35)
    text = Text(
        SUBTITLE,
        font_size=26,
        weight="BOLD",
        color="#FBFBFF",
        font="DejaVu Sans",
    ).move_to(pill)
    text.set_stroke(BLACK, width=4, background=True)
    return VGroup(pill, text)


def build_footer_line() -> Text:
    text = Text(
        FOOTER,
        font_size=20,
        weight="BOLD",
        color="#E8ECF8",
        font="DejaVu Sans",
    )
    text.set_stroke(BLACK, width=4, background=True)
    return text


def build_axes() -> VGroup:
    left_x = -3.15
    right_x = 3.35
    top_y = 4.35
    bottom_y = -3.55

    elements = VGroup()
    for age in range(AGE_MIN, AGE_MAX + 1):
        x = left_x + (age - AGE_MIN) / (AGE_MAX - AGE_MIN) * (right_x - left_x)
        line = Line([x, top_y, 0], [x, bottom_y, 0], color="#FFFFFF", stroke_width=2)
        line.set_stroke(opacity=0.10)
        elements.add(line)

        if age in {17, 20, 23, 26, 29, 32, 35, 37}:
            label = Text(f"Age {age}", font_size=16, weight="BOLD", color="#F4F7FF", font="DejaVu Sans")
            label.set_stroke(BLACK, width=3, background=True)
            label.move_to([x, top_y + 0.22, 0])
            elements.add(label)

    current_line = Line([0, top_y, 0], [0, bottom_y, 0], color="#FFE366", stroke_width=4)
    current_line.set_stroke(opacity=0.0)
    current_age = RoundedRectangle(width=1.12, height=0.42, corner_radius=0.18)
    current_age.set_fill("#F5C94B", opacity=1.0)
    current_age.set_stroke("#FFFFFF", width=1.5, opacity=0.7)
    current_label = Text("Age 17", font_size=24, weight="BOLD", color="#08111E", font="DejaVu Sans")
    current_label.move_to(current_age)

    return VGroup(elements, current_line, current_age, current_label)


def build_player_row(player: Player, age_tracker: ValueTracker) -> VGroup:
    avatar_img = make_avatar_image(player, size=480)
    avatar = ImageMobject(avatar_img).scale_to_fit_width(1.32)

    name = Text(
        player.short,
        font_size=20,
        weight="BOLD",
        color="#F4F8FF",
        font="DejaVu Sans",
    )
    name.set_stroke(BLACK, width=4, background=True)

    def make_row() -> VGroup:
        age = age_tracker.get_value()
        ranks = rank_positions(age)
        counts = interpolated_counts(age)
        rank = ranks[player.name]
        count = counts[player.name]
        row_y = 1.95 - rank * 2.20

        card = RoundedRectangle(width=1.46, height=1.46, corner_radius=0.28)
        card.set_fill("#F6F7FB", opacity=1.0)
        card.set_stroke("#FFFFFF", width=1.8, opacity=0.95)

        avatar_group = Group(card, avatar.copy()).move_to([0.0, row_y, 0.0])
        avatar_group[1].scale_to_fit_height(1.22)
        avatar_group[1].move_to(avatar_group[0].get_center())

        name_text = Text(
            player.short,
            font_size=20,
            weight="BOLD",
            color="#F4F8FF",
            font="DejaVu Sans",
        )
        name_text.set_stroke(BLACK, width=4, background=True)
        name_text.next_to(avatar_group, RIGHT, buff=0.22)

        bar_left = -0.25
        bar_y = row_y
        max_width = 4.95
        bar_h = 0.72
        fill_w = max(0.62, max_width * (count / 24.0))

        shadow = RoundedRectangle(width=max_width, height=bar_h, corner_radius=0.36)
        shadow.set_fill("#000000", opacity=0.30)
        shadow.set_stroke(opacity=0.0)
        shadow.shift([0.12, -0.08, 0])
        shadow.move_to([bar_left + max_width / 2, bar_y, 0])

        track = RoundedRectangle(width=max_width, height=bar_h, corner_radius=0.36)
        track.set_fill("#FFFFFF", opacity=0.16)
        track.set_stroke("#FFFFFF", width=1.0, opacity=0.20)
        track.move_to([bar_left + max_width / 2, bar_y, 0])

        fill = RoundedRectangle(width=fill_w, height=bar_h, corner_radius=0.36)
        fill.set_fill(player.color, opacity=1.0)
        fill.set_stroke(mix_color(player.color, (255, 255, 255), 0.20), width=1.5, opacity=0.9)
        fill.move_to([bar_left + fill_w / 2, bar_y, 0])

        highlight = RoundedRectangle(width=min(fill_w, 2.2), height=0.14, corner_radius=0.07)
        highlight.set_fill("#FFFFFF", opacity=0.20)
        highlight.set_stroke(opacity=0.0)
        highlight.move_to([bar_left + min(fill_w * 0.55, 1.6), bar_y + 0.22, 0])

        count_text = Text(
            f"{int(round(count))}",
            font_size=42,
            weight="BOLD",
            color=text_on(player.color),
            font="DejaVu Sans",
        )
        count_text.set_stroke(BLACK, width=4, background=True)
        count_text.move_to([bar_left + 0.72, bar_y, 0])

        label = Text(
            player.name.upper(),
            font_size=18,
            weight="BOLD",
            color="#DCE3F2",
            font="DejaVu Sans",
        )
        label.set_stroke(BLACK, width=3, background=True)
        label.next_to(fill, RIGHT, buff=0.18)

        row = Group(shadow, track, fill, highlight, count_text, label, avatar_group, name_text)
        return row

    return always_redraw(make_row)


class GrandSlamAgeRace(Scene):
    def construct(self) -> None:
        bg = ImageMobject(build_background()).scale_to_fit_width(9).scale_to_fit_height(16)
        bg.move_to(ORIGIN)
        self.add(bg)

        title = build_title()
        title.move_to([0, 6.95, 0])
        subtitle = build_subtitle()
        subtitle.move_to([0, 6.12, 0])
        footer = build_footer_line()
        footer.move_to([0, -7.35, 0])

        title_ball = Circle(radius=0.19)
        title_ball.set_fill("#FFE766", opacity=1.0)
        title_ball.set_stroke("#FFFFFF", width=1.2, opacity=0.8)
        title_ball.move_to([3.7, 7.02, 0])

        seam1 = Line([-0.11, 0, 0], [0.11, 0, 0], color="#FFFFFF", stroke_width=2).rotate(math.pi / 4).move_to(title_ball)
        seam2 = Line([-0.11, 0, 0], [0.11, 0, 0], color="#FFFFFF", stroke_width=2).rotate(-math.pi / 4).move_to(title_ball)

        axes = build_axes()
        axes.move_to([0, 0.05, 0])
        age_line = axes[1]
        age_box = axes[2]
        age_label = axes[3]

        self.play(FadeIn(title, shift=UP * 0.25), FadeIn(title_ball, scale=0.8), FadeIn(subtitle, shift=UP * 0.15), run_time=1.0)
        self.play(Create(seam1), Create(seam2), run_time=0.35)
        self.play(FadeIn(axes[0], lag_ratio=0.02), run_time=0.5)

        age_tracker = ValueTracker(AGE_MIN)
        current_age_line = always_redraw(
            lambda: Line(
                [
                    -3.15 + (age_tracker.get_value() - AGE_MIN) / (AGE_MAX - AGE_MIN) * (3.35 - (-3.15)),
                    4.35,
                    0,
                ],
                [
                    -3.15 + (age_tracker.get_value() - AGE_MIN) / (AGE_MAX - AGE_MIN) * (3.35 - (-3.15)),
                    -3.55,
                    0,
                ],
                color="#FFF2A1",
                stroke_width=4,
            )
        )
        current_age_badge = always_redraw(
            lambda: RoundedRectangle(width=1.08, height=0.42, corner_radius=0.18)
            .set_fill("#F5C94B", opacity=1.0)
            .set_stroke("#FFFFFF", width=1.5, opacity=0.75)
            .move_to(
                [
                    -3.15 + (age_tracker.get_value() - AGE_MIN) / (AGE_MAX - AGE_MIN) * (3.35 - (-3.15)),
                    4.72,
                    0,
                ]
            )
        )
        current_age_text = always_redraw(
            lambda: Text(
                f"{int(round(age_tracker.get_value()))}",
                font_size=24,
                weight="BOLD",
                color="#08111E",
                font="DejaVu Sans",
            ).move_to(current_age_badge)
        )
        age_header = always_redraw(
            lambda: Text(
                f"Age {int(round(age_tracker.get_value()))}",
                font_size=26,
                weight="BOLD",
                color="#F7FAFF",
                font="DejaVu Sans",
            ).set_stroke(BLACK, width=4, background=True).move_to([0, 5.05, 0])
        )

        self.add(current_age_line, current_age_badge, current_age_text, age_header)

        row_groups = Group(*[build_player_row(player, age_tracker) for player in PLAYERS])
        row_groups.set_opacity(0)
        self.play(FadeIn(row_groups, shift=UP * 0.12), run_time=0.7)

        self.play(
            age_tracker.animate.set_value(AGE_MAX),
            row_groups.animate.set_opacity(1),
            run_time=TOTAL_DURATION - INTRO_DURATION - OUTRO_DURATION,
            rate_func=linear,
        )

        final_panel = RoundedRectangle(width=7.25, height=2.85, corner_radius=0.28)
        final_panel.set_fill("#0C1222", opacity=0.90)
        final_panel.set_stroke("#FFFFFF", width=1.2, opacity=0.12)
        final_panel.move_to([0, -4.95, 0])

        cta = Text(
            CTA,
            font_size=28,
            weight="BOLD",
            color="#F7FAFF",
            font="DejaVu Sans",
        )
        cta.set_stroke(BLACK, width=5, background=True)
        cta.move_to([0, -4.10, 0])

        final_rows = VGroup(
            Text("24  Djokovic", font_size=28, weight="BOLD", color="#F7FAFF", font="DejaVu Sans").set_stroke(BLACK, width=4, background=True),
            Text("22  Nadal", font_size=28, weight="BOLD", color="#F7FAFF", font="DejaVu Sans").set_stroke(BLACK, width=4, background=True),
            Text("20  Federer", font_size=28, weight="BOLD", color="#F7FAFF", font="DejaVu Sans").set_stroke(BLACK, width=4, background=True),
        ).arrange(DOWN, buff=0.16).move_to([0, -4.82, 0])

        self.play(FadeIn(final_panel, shift=UP * 0.1), FadeIn(cta, shift=UP * 0.1), FadeIn(final_rows, shift=UP * 0.1), run_time=1.0)
        self.play(FadeIn(footer, shift=UP * 0.05), run_time=0.3)
        self.wait(1.2)


def attach_audio(video_path: Path, output_path: Path, music_path: Path | None) -> None:
    if VideoFileClip is None or AudioFileClip is None or CompositeAudioClip is None:
        raise RuntimeError("MoviePy is required for audio assembly.")
    if music_path is None or not music_path.exists():
        raise FileNotFoundError(f"Audio file not found: {music_path}")

    video = VideoFileClip(str(video_path))
    music = AudioFileClip(str(music_path))
    if music.duration >= video.duration:
        audio = music.subclipped(0, video.duration)
    else:
        parts = []
        step = max(0.1, music.duration - 1.5)
        current = 0.0
        while current < video.duration:
            parts.append(music.with_start(current))
            current += step
        audio = CompositeAudioClip(parts).with_duration(video.duration)
    final = video.with_audio(audio)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    final.write_videofile(str(output_path), codec="libx264", audio_codec="aac")
    final.close()
    audio.close()
    video.close()


def upscale_video(input_path: Path, output_path: Path, width: int = 1080, height: int = 1920) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-vf",
        f"scale={width}:{height}:flags=lanczos",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "18",
        "-c:a",
        "copy",
        str(output_path),
    ]
    subprocess.run(cmd, check=True)


def run_manim(scene_name: str, media_dir: Path, quality: str, output_stem: str) -> int:
    cmd = [
        sys.executable,
        "-m",
        "manim",
        str(Path(__file__).resolve()),
        scene_name,
        "--format",
        "mp4",
        "--media_dir",
        str(media_dir),
        "--output_file",
        output_stem,
    ]
    if quality:
        cmd.append(f"-q{quality}")
    return subprocess.run(cmd, check=False).returncode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a vertical Manim bar chart race for Nadal, Djokovic and Federer by age.")
    parser.add_argument("--scene", default="GrandSlamAgeRace", help="Manim scene to render.")
    parser.add_argument("--render", action="store_true", help="Render the scene with Manim.")
    parser.add_argument("--quality", default="l", help="Manim quality flag: l, m, h, p, k.")
    parser.add_argument("--media-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-stem", default=DEFAULT_OUTPUT_STEM)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Final MP4 path to copy the render to.")
    parser.add_argument("--audio", type=Path, default=DEFAULT_AUDIO)
    parser.add_argument("--mix-audio", action="store_true", help="Attach music after Manim render using MoviePy.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not MANIM_AVAILABLE:
        raise SystemExit("Manim is not installed. Install it with: pip install -r requirements-manim-mvp-race.txt")

    if not args.render:
        print("Scene available: GrandSlamAgeRace")
        print("Render command example:")
        print(f"  python {Path(__file__).name} --render --scene GrandSlamAgeRace --quality h")
        return

    status = run_manim(args.scene, args.media_dir, args.quality, args.output_stem)
    if status != 0:
        raise SystemExit(status)

    quality_folder = {
        "l": "480p15",
        "m": "720p30",
        "h": "1080p30",
        "p": "1440p60",
        "k": "2160p60",
    }.get(args.quality, "1080p30")
    rendered = args.media_dir / "videos" / Path(__file__).stem / quality_folder / f"{args.output_stem}.mp4"
    if args.output:
        if rendered.exists():
            upscale_video(rendered, args.output)
            print(f"[manim] upscaled final video -> {args.output}")

    if args.mix_audio:
        mixed = rendered.with_name(f"{rendered.stem}_audio.mp4")
        attach_audio(rendered, mixed, args.audio)
        print(f"[manim] audio mix ready -> {mixed}")


if __name__ == "__main__":
    main()
