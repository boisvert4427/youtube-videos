from __future__ import annotations

import argparse
import math
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

try:
    from manim import (
        AnimationGroup,
        BLACK,
        BLUE,
        DOWN,
        FadeIn,
        FadeOut,
        GOLD,
        GRAY_B,
        GRAY_C,
        Group,
        ImageMobject,
        LEFT,
        Line,
        Mobject,
        ORIGIN,
        RED,
        RIGHT,
        RoundedRectangle,
        Scene,
        Square,
        SurroundingRectangle,
        Text,
        UP,
        VGroup,
        VMobject,
        ValueTracker,
        WHITE,
        always_redraw,
        config,
    )
    MANIM_AVAILABLE = True
except Exception:
    MANIM_AVAILABLE = False
    Scene = object  # type: ignore[assignment]
    config = None  # type: ignore[assignment]
    WHITE = "#F7F7F2"  # type: ignore[assignment]
    BLACK = "#06080E"  # type: ignore[assignment]
    GOLD = "#FFD062"  # type: ignore[assignment]
    BLUE = "#1F6BFF"  # type: ignore[assignment]
    RED = "#FF5B45"  # type: ignore[assignment]
    GRAY_B = "#B0BCD2"  # type: ignore[assignment]
    GRAY_C = "#8C97A8"  # type: ignore[assignment]
    LEFT = RIGHT = UP = DOWN = ORIGIN = 0  # type: ignore[assignment]

try:
    from moviepy import AudioFileClip, CompositeAudioClip, VideoFileClip
    from moviepy.audio.fx import AudioFadeOut
except Exception:
    AudioFileClip = None  # type: ignore[assignment]
    CompositeAudioClip = None  # type: ignore[assignment]
    VideoFileClip = None  # type: ignore[assignment]
    AudioFadeOut = None  # type: ignore[assignment]


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_OUTPUT_STEM = "mvp_race_shorts_manim"
DEFAULT_MEDIA_DIR = PROJECT_ROOT / "data" / "processed" / "basketball" / "manim_mvp_race"
DEFAULT_ASSETS_DIR = PROJECT_ROOT / "data" / "raw" / "mvp_race_assets"
DEFAULT_AUDIO = PROJECT_ROOT / "data" / "raw" / "audio" / "audio.mp3"
DEFAULT_REQUIREMENTS = PROJECT_ROOT / "requirements-manim-mvp-race.txt"

RENDER_WIDTH = 1080
RENDER_HEIGHT = 1920
FPS = 30

if config is not None:
    config.pixel_width = RENDER_WIDTH
    config.pixel_height = RENDER_HEIGHT
    config.frame_width = 9
    config.frame_height = 16
    config.frame_rate = FPS
    config.background_color = "#080b12"

SILVER = "#BAC8D6"
BRONZE = "#C27F4E"


@dataclass(frozen=True)
class PlayerData:
    name: str
    short_name: str
    color: str
    secondary_color: str
    points: float
    assists: float
    rebounds: float
    team_win_pct: float
    score: int
    image: str


@dataclass(frozen=True)
class StatSpec:
    key: str
    label: str
    decimals: int
    suffix: str = ""
    inverse: bool = False


@dataclass(frozen=True)
class TimingConfig:
    hook: float = 2.0
    intro: float = 4.0
    stats: float = 12.0
    score: float = 6.0
    podium: float = 4.0
    cta: float = 2.0

    @property
    def total(self) -> float:
        return self.hook + self.intro + self.stats + self.score + self.podium + self.cta


PLAYERS = [
    PlayerData(
        name="Nikola Jokic",
        short_name="JOKIC",
        color="#FDB927",
        secondary_color="#0E2240",
        points=29.0,
        assists=10.2,
        rebounds=12.8,
        team_win_pct=0.722,
        score=96,
        image="jokic.png",
    ),
    PlayerData(
        name="Shai Gilgeous-Alexander",
        short_name="SGA",
        color="#007AC1",
        secondary_color="#EF3B24",
        points=31.1,
        assists=6.1,
        rebounds=5.5,
        team_win_pct=0.611,
        score=93,
        image="sga.png",
    ),
    PlayerData(
        name="Luka Doncic",
        short_name="DONCIC",
        color="#00538C",
        secondary_color="#B8C4CA",
        points=33.7,
        assists=9.8,
        rebounds=9.1,
        team_win_pct=0.667,
        score=91,
        image="doncic.png",
    ),
]


STATS = [
    StatSpec("points", "POINTS PER GAME", 1),
    StatSpec("assists", "ASSISTS PER GAME", 1),
    StatSpec("rebounds", "REBOUNDS PER GAME", 1),
    StatSpec("team_win_pct", "TEAM WIN %", 1, suffix="%"),
]


def ease_out_cubic(value: float) -> float:
    value = max(0.0, min(1.0, value))
    return 1 - (1 - value) ** 3


def ease_out_back(value: float) -> float:
    value = max(0.0, min(1.0, value))
    c1 = 1.70158
    c3 = c1 + 1
    return 1 + c3 * (value - 1) ** 3 + c1 * (value - 1) ** 2


def resolve_asset(path_str: str, assets_dir: Path) -> Path | None:
    candidate = Path(path_str)
    if candidate.exists():
        return candidate
    local = assets_dir / path_str
    if local.exists():
        return local
    return None


def format_stat_text(stat: StatSpec, value: float) -> str:
    if stat.key == "team_win_pct":
        value *= 100
    if stat.decimals == 0:
        return f"{int(round(value))}{stat.suffix}"
    return f"{value:.1f}{stat.suffix}"


class MVPRaceBase(Scene):  # type: ignore[misc]
    timings = TimingConfig()
    players = PLAYERS
    stats = STATS
    assets_dir = DEFAULT_ASSETS_DIR

    def construct_background(self) -> VGroup:
        background = RoundedRectangle(
            width=9.4,
            height=16.4,
            corner_radius=0.25,
            stroke_width=0,
            fill_color="#080b12",
            fill_opacity=1,
        )
        blue_orb = Square(4.8, stroke_width=0, fill_color="#1F6BFF", fill_opacity=0.18).move_to(LEFT * 2.6 + UP * 1.9)
        red_orb = Square(5.2, stroke_width=0, fill_color="#FF5B45", fill_opacity=0.18).move_to(RIGHT * 2.5 + UP * 1.2)
        gold_orb = Square(3.6, stroke_width=0, fill_color="#FFD46C", fill_opacity=0.08).move_to(DOWN * 2.5)
        for mob in (blue_orb, red_orb, gold_orb):
            mob.rotate(math.radians(45))
        return VGroup(background, blue_orb, red_orb, gold_orb)

    def glow_text(self, text: str, size: int, color=WHITE, glow_color=None, weight="BOLD") -> VGroup:
        glow_color = glow_color or color
        core = Text(text, font_size=size, weight=weight, color=color)
        glow = Text(text, font_size=size, weight=weight, color=glow_color, fill_opacity=0.18, stroke_opacity=0)
        glow.scale(1.03)
        return VGroup(glow, core)

    def make_player_card(self, player: PlayerData, width: float = 2.55, height: float = 1.5) -> VGroup:
        panel = RoundedRectangle(
            width=width,
            height=height,
            corner_radius=0.22,
            fill_color=player.secondary_color,
            fill_opacity=0.88,
            stroke_color=player.color,
            stroke_opacity=0.9,
            stroke_width=2,
        )
        image_path = resolve_asset(player.image, self.assets_dir)
        if image_path is not None:
            portrait: Mobject = ImageMobject(str(image_path)).scale_to_fit_height(height - 0.22)
            portrait.set_x(panel.get_left()[0] + 0.66)
        else:
            portrait = RoundedRectangle(
                width=0.95,
                height=0.95,
                corner_radius=0.48,
                fill_color=player.color,
                fill_opacity=0.95,
                stroke_width=0,
            )
            initials = Text(player.short_name[:2], font_size=28, weight="BOLD", color=BLACK).move_to(portrait)
            portrait = VGroup(portrait, initials)
            portrait.set_x(panel.get_left()[0] + 0.66)
        name = Text(player.short_name, font_size=24, weight="BOLD", color=WHITE).move_to(panel.get_center() + RIGHT * 0.32 + UP * 0.12)
        subtitle = Text("MVP CANDIDATE", font_size=12, weight="BOLD", color=GRAY_B).move_to(panel.get_center() + RIGHT * 0.38 + DOWN * 0.28)
        return VGroup(panel, portrait, name, subtitle)

    def make_stat_row(self, label: str) -> VGroup:
        row = RoundedRectangle(
            width=7.6,
            height=1.45,
            corner_radius=0.24,
            fill_color="#122034",
            fill_opacity=0.84,
            stroke_color="#355784",
            stroke_opacity=0.45,
            stroke_width=1.8,
        )
        track = RoundedRectangle(
            width=5.25,
            height=0.26,
            corner_radius=0.13,
            fill_color="#DCE4F0",
            fill_opacity=0.12,
            stroke_width=0,
        ).move_to(row.get_center() + DOWN * 0.2)
        label_box = RoundedRectangle(
            width=2.7,
            height=0.48,
            corner_radius=0.18,
            fill_color="#F0D49A",
            fill_opacity=0.98,
            stroke_width=0,
        ).move_to(row.get_center() + UP * 0.14)
        label_text = Text(label, font_size=20, weight="BOLD", color=BLACK).move_to(label_box)
        return VGroup(row, track, label_box, label_text)

    def make_score_bar(self, player: PlayerData, height: float) -> VGroup:
        lane = RoundedRectangle(
            width=1.45,
            height=height + 1.2,
            corner_radius=0.22,
            fill_color=player.secondary_color,
            fill_opacity=0.78,
            stroke_color=player.color,
            stroke_opacity=0.85,
            stroke_width=2,
        )
        track = RoundedRectangle(width=0.42, height=height - 0.2, corner_radius=0.18, fill_color=WHITE, fill_opacity=0.08, stroke_width=0)
        portrait_path = resolve_asset(player.image, self.assets_dir)
        if portrait_path is not None:
            portrait = ImageMobject(str(portrait_path)).scale_to_fit_height(0.82)
        else:
            portrait = Text(player.short_name[:2], font_size=22, weight="BOLD", color=WHITE)
        portrait.move_to(lane.get_top() + DOWN * 0.52)
        return VGroup(lane, track, portrait)

    def make_crown(self, color=GOLD) -> VGroup:
        left = Line(LEFT * 0.46 + DOWN * 0.18, LEFT * 0.18 + UP * 0.24, color=color, stroke_width=8)
        middle_left = Line(LEFT * 0.18 + UP * 0.24, ORIGIN + DOWN * 0.06, color=color, stroke_width=8)
        spike = Line(ORIGIN + DOWN * 0.06, ORIGIN + UP * 0.4, color=color, stroke_width=8)
        middle_right = Line(ORIGIN + UP * 0.4, RIGHT * 0.18 + DOWN * 0.06, color=color, stroke_width=8)
        right = Line(RIGHT * 0.18 + DOWN * 0.06, RIGHT * 0.46 + UP * 0.24, color=color, stroke_width=8)
        base = Line(LEFT * 0.58 + DOWN * 0.28, RIGHT * 0.58 + DOWN * 0.28, color=color, stroke_width=10)
        return VGroup(left, middle_left, spike, middle_right, right, base)

    def build_hook(self) -> VGroup:
        title = self.glow_text("IF MVP WAS", 38, WHITE, GOLD).move_to(UP * 2.6)
        subtitle = self.glow_text("DECIDED TODAY...", 38, GOLD, GOLD).next_to(title, DOWN, buff=0.28)
        kicker = self.glow_text("THIS IS THE RACE", 18, GRAY_B, BLUE).next_to(subtitle, DOWN, buff=0.42)
        underline = Line(LEFT * 2.8, RIGHT * 2.8, stroke_width=4, color=GOLD).next_to(kicker, DOWN, buff=0.35)
        return VGroup(title, subtitle, kicker, underline)

    def build_intro_group(self) -> VGroup:
        title = self.glow_text("TOP 3 RIGHT NOW", 26, WHITE, GOLD).move_to(UP * 4.45)
        cards = VGroup(
            self.make_player_card(self.players[0]).move_to(UP * 2.7),
            self.make_player_card(self.players[1]).move_to(ORIGIN + UP * 0.65),
            self.make_player_card(self.players[2]).move_to(DOWN * 1.4),
        )
        return VGroup(title, cards)

    def build_stats_scene(self) -> tuple[VGroup, list[tuple[Text, VMobject, PlayerData, StatSpec]]]:
        title = self.glow_text("STAT BATTLE", 28, WHITE, GOLD).move_to(UP * 4.55)
        rows = VGroup()
        animated: list[tuple[Text, VMobject, PlayerData, StatSpec]] = []
        ys = [2.35, 0.8, -0.75, -2.3]
        for stat, y in zip(self.stats, ys):
            row = self.make_stat_row(stat.label).move_to(UP * y)
            values = [getattr(player, stat.key) * (100 if stat.key == "team_win_pct" else 1) for player in self.players]
            max_val = max(values) if max(values) else 1
            left_x = row[1].get_left()[0]
            for index, player in enumerate(self.players):
                bar_color = player.color
                width = 5.1 * (values[index] / max_val)
                bar = RoundedRectangle(
                    width=max(0.05, width),
                    height=0.22,
                    corner_radius=0.11,
                    fill_color=bar_color,
                    fill_opacity=0.95,
                    stroke_width=0,
                )
                bar.align_to(row[1], LEFT)
                bar.shift(RIGHT * (width / 2))
                if index == 1:
                    bar.next_to(row[1], DOWN, buff=0.18)
                elif index == 2:
                    bar.next_to(row[1], DOWN, buff=0.48)
                else:
                    bar.next_to(row[1], DOWN, buff=-0.12)
                value = Text("0" if stat.decimals == 0 else "0.0", font_size=22, weight="BOLD", color=WHITE)
                value.next_to(bar, RIGHT, buff=0.16)
                tag = Text(player.short_name, font_size=12, weight="BOLD", color=GRAY_C).next_to(bar, LEFT, buff=0.16)
                row.add(bar, value, tag)
                animated.append((value, bar, player, stat))
            rows.add(row)
        return VGroup(title, rows), animated

    def build_score_group(self) -> tuple[VGroup, list[tuple[Text, VMobject, PlayerData]]]:
        title = self.glow_text("FINAL MVP SCORE", 30, WHITE, GOLD).move_to(UP * 4.45)
        bars = VGroup()
        animated: list[tuple[Text, VMobject, PlayerData]] = []
        for x, player in zip([-2.35, 0, 2.35], sorted(self.players, key=lambda item: item.score, reverse=True)):
            column = self.make_score_bar(player, 4.5).move_to(DOWN * 0.5 + RIGHT * x)
            fill = RoundedRectangle(
                width=0.42,
                height=max(0.1, 4.3 * (player.score / 100)),
                corner_radius=0.18,
                fill_color=player.color,
                fill_opacity=0.96,
                stroke_width=0,
            )
            fill.align_to(column[1], DOWN)
            score = Text("0", font_size=28, weight="BOLD", color=WHITE).next_to(column[0], DOWN, buff=0.24)
            name = Text(player.short_name, font_size=16, weight="BOLD", color=GRAY_B).next_to(score, DOWN, buff=0.12)
            crown = self.make_crown().scale(0.72).next_to(column[0], UP, buff=0.22) if player.score == max(item.score for item in self.players) else VGroup()
            column.add(fill, score, name, crown)
            bars.add(column)
            animated.append((score, fill, player))
        return VGroup(title, bars), animated

    def build_podium_group(self) -> VGroup:
        title = self.glow_text("THE PODIUM", 30, WHITE, GOLD).move_to(UP * 4.45)
        ordered = sorted(self.players, key=lambda item: item.score, reverse=True)
        specs = [
            (ordered[1], -2.5, -2.15, 2.3, SILVER, "#2"),
            (ordered[0], 0, -1.55, 3.1, GOLD, "#1"),
            (ordered[2], 2.5, -2.35, 2.0, BRONZE, "#3"),
        ]
        blocks = VGroup()
        for player, x, y, height, medal_color, rank in specs:
            block = RoundedRectangle(
                width=1.8,
                height=height,
                corner_radius=0.22,
                fill_color=player.secondary_color,
                fill_opacity=0.88,
                stroke_color=medal_color,
                stroke_opacity=1,
                stroke_width=2.2,
            ).move_to(RIGHT * x + UP * y)
            label = Text(rank, font_size=34, weight="BOLD", color=medal_color).next_to(block.get_top(), DOWN, buff=0.28)
            name = Text(player.short_name, font_size=18, weight="BOLD", color=WHITE).move_to(block.get_center() + DOWN * 0.35)
            score = Text(str(player.score), font_size=30, weight="BOLD", color=player.color).move_to(block.get_center() + DOWN * 0.8)
            blocks.add(VGroup(block, label, name, score))
        return VGroup(title, blocks)

    def build_cta_group(self) -> VGroup:
        title = self.glow_text("DO YOU AGREE?", 40, WHITE, GOLD).move_to(UP * 0.8)
        subtitle = self.glow_text("COMMENT BELOW", 20, GOLD, RED).next_to(title, DOWN, buff=0.4)
        return VGroup(title, subtitle)


class HookScene(MVPRaceBase):
    def construct(self) -> None:
        self.add(self.construct_background())
        hook = self.build_hook()
        self.play(FadeIn(hook, shift=UP * 0.3), run_time=self.timings.hook)
        self.wait(0.1)


class Top3IntroScene(MVPRaceBase):
    def construct(self) -> None:
        self.add(self.construct_background())
        intro = self.build_intro_group()
        self.play(FadeIn(intro[0], shift=UP * 0.2), run_time=0.45)
        self.play(AnimationGroup(*[FadeIn(card, shift=RIGHT * 0.3) for card in intro[1]], lag_ratio=0.18), run_time=self.timings.intro)
        self.wait(0.1)


class StatsBarsScene(MVPRaceBase):
    def construct(self) -> None:
        self.add(self.construct_background())
        group, animated = self.build_stats_scene()
        self.play(FadeIn(group[0], shift=UP * 0.2), FadeIn(group[1], shift=DOWN * 0.12), run_time=0.7)
        step = self.timings.stats / len(self.stats)
        for stat_index, stat in enumerate(self.stats):
            related = [item for item in animated if item[3] == stat]
            trackers = [ValueTracker(0) for _ in related]
            for tracker, (number, _bar, player, spec) in zip(trackers, related):
                target = getattr(player, spec.key) * (100 if spec.key == "team_win_pct" else 1)
                replacement = always_redraw(
                    lambda tr=tracker, sp=spec, mob=number: Text(
                        format_stat_text(sp, tr.get_value() if sp.key != "team_win_pct" else tr.get_value() / 100),
                        font_size=22,
                        weight="BOLD",
                        color=WHITE,
                    ).move_to(mob)
                )
                self.add(replacement)
                self.play(tracker.animate.set_value(target), run_time=step * 0.72)
                self.remove(replacement)
        self.wait(0.1)


class ScoreRevealScene(MVPRaceBase):
    def construct(self) -> None:
        self.add(self.construct_background())
        group, animated = self.build_score_group()
        self.play(FadeIn(group[0], shift=UP * 0.2), FadeIn(group[1], shift=DOWN * 0.2), run_time=0.7)
        trackers = [ValueTracker(0) for _ in animated]
        for tracker, (number, _fill, player) in zip(trackers, animated):
            replacement = always_redraw(
                lambda tr=tracker, mob=number: Text(
                    str(int(round(tr.get_value()))),
                    font_size=28,
                    weight="BOLD",
                    color=WHITE,
                ).move_to(mob)
            )
            self.add(replacement)
            self.play(tracker.animate.set_value(player.score), run_time=self.timings.score * 0.75)
            self.remove(replacement)
        self.wait(0.1)


class PodiumScene(MVPRaceBase):
    def construct(self) -> None:
        self.add(self.construct_background())
        podium = self.build_podium_group()
        self.play(FadeIn(podium[0], shift=UP * 0.2), run_time=0.4)
        self.play(AnimationGroup(*[FadeIn(block, shift=UP * 0.3) for block in podium[1]], lag_ratio=0.15), run_time=self.timings.podium)
        self.wait(0.1)


class CTAScene(MVPRaceBase):
    def construct(self) -> None:
        self.add(self.construct_background())
        cta = self.build_cta_group()
        self.play(FadeIn(cta, shift=UP * 0.2), run_time=self.timings.cta)
        self.wait(0.1)


class MVPRaceShort(MVPRaceBase):
    def construct(self) -> None:
        background = self.construct_background()
        self.add(background)

        hook = self.build_hook()
        self.play(FadeIn(hook, shift=UP * 0.35), run_time=0.65)
        self.wait(self.timings.hook - 0.95)
        self.play(FadeOut(hook, shift=UP * 0.15), run_time=0.28)

        intro = self.build_intro_group()
        self.play(FadeIn(intro[0], shift=UP * 0.2), run_time=0.35)
        self.play(AnimationGroup(*[FadeIn(card, shift=RIGHT * 0.35) for card in intro[1]], lag_ratio=0.18), run_time=self.timings.intro - 0.85)
        self.play(FadeOut(intro, shift=UP * 0.1), run_time=0.35)

        stats_group, animated = self.build_stats_scene()
        self.play(FadeIn(stats_group[0], shift=UP * 0.15), FadeIn(stats_group[1], shift=DOWN * 0.15), run_time=0.45)
        per_stat = self.timings.stats / len(self.stats)
        for stat in self.stats:
            stat_items = [item for item in animated if item[3] == stat]
            trackers = [ValueTracker(0) for _ in stat_items]
            replacements = []
            for tracker, (number, _bar, player, spec) in zip(trackers, stat_items):
                target = getattr(player, spec.key) * (100 if spec.key == "team_win_pct" else 1)
                replacement = always_redraw(
                    lambda tr=tracker, sp=spec, mob=number: Text(
                        format_stat_text(sp, tr.get_value() if sp.key != "team_win_pct" else tr.get_value() / 100),
                        font_size=22,
                        weight="BOLD",
                        color=WHITE,
                    ).move_to(mob)
                )
                replacements.append(replacement)
                self.add(replacement)
            self.play(
                *[
                    tracker.animate.set_value(getattr(player, spec.key) * (100 if spec.key == "team_win_pct" else 1))
                    for tracker, (_number, _bar, player, spec) in zip(trackers, stat_items)
                ],
                run_time=per_stat * 0.72,
            )
            for replacement in replacements:
                self.remove(replacement)
        self.play(FadeOut(stats_group, shift=DOWN * 0.1), run_time=0.38)

        score_group, score_items = self.build_score_group()
        self.play(FadeIn(score_group[0], shift=UP * 0.18), FadeIn(score_group[1], shift=DOWN * 0.2), run_time=0.45)
        trackers = [ValueTracker(0) for _ in score_items]
        replacements = []
        for tracker, (number, _fill, _player) in zip(trackers, score_items):
            replacement = always_redraw(
                lambda tr=tracker, mob=number: Text(
                    str(int(round(tr.get_value()))),
                    font_size=28,
                    weight="BOLD",
                    color=WHITE,
                ).move_to(mob)
            )
            replacements.append(replacement)
            self.add(replacement)
        self.play(*[tracker.animate.set_value(player.score) for tracker, (_number, __, player) in zip(trackers, score_items)], run_time=self.timings.score - 0.95)
        for replacement in replacements:
            self.remove(replacement)
        self.play(FadeOut(score_group, shift=DOWN * 0.1), run_time=0.35)

        podium = self.build_podium_group()
        self.play(FadeIn(podium[0], shift=UP * 0.2), run_time=0.3)
        self.play(AnimationGroup(*[FadeIn(block, shift=UP * 0.28) for block in podium[1]], lag_ratio=0.12), run_time=self.timings.podium - 0.7)
        self.play(FadeOut(podium, shift=DOWN * 0.12), run_time=0.32)

        cta = self.build_cta_group()
        self.play(FadeIn(cta, shift=UP * 0.2), run_time=self.timings.cta - 0.2)
        self.wait(0.2)


def attach_audio(video_path: Path, output_path: Path, music_path: Path | None, fade_out_seconds: float = 5.0) -> None:
    if VideoFileClip is None or AudioFileClip is None or CompositeAudioClip is None:
        raise RuntimeError("MoviePy is required for audio assembly.")
    video = VideoFileClip(str(video_path))
    audio = None
    if music_path is not None and music_path.exists():
        base_music = AudioFileClip(str(music_path))
        if base_music.duration >= video.duration:
            audio = base_music.subclipped(0, video.duration)
        else:
            loops = []
            step = max(0.1, base_music.duration - 1.5)
            total = 0.0
            while total < video.duration:
                loops.append(base_music.with_start(total))
                total += step
            audio = CompositeAudioClip(loops).with_duration(video.duration)
        if AudioFadeOut is not None:
            audio = audio.with_effects([AudioFadeOut(min(fade_out_seconds, video.duration))])
        final = video.with_audio(audio)
    else:
        final = video
    output_path.parent.mkdir(parents=True, exist_ok=True)
    final.write_videofile(str(output_path), codec="libx264", audio_codec="aac" if audio else None)
    final.close()
    video.close()
    if audio is not None:
        audio.close()


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Manim-based premium MVP race Shorts generator.")
    parser.add_argument("--scene", default="MVPRaceShort", help="Manim scene to render.")
    parser.add_argument("--render", action="store_true", help="Render the scene with manim.")
    parser.add_argument("--quality", default="h", help="Manim quality flag: l, m, h, p, k.")
    parser.add_argument("--media-dir", type=Path, default=DEFAULT_MEDIA_DIR)
    parser.add_argument("--output-stem", default=DEFAULT_OUTPUT_STEM)
    parser.add_argument("--audio", type=Path, default=DEFAULT_AUDIO)
    parser.add_argument("--mix-audio", action="store_true", help="Attach music after manim render using MoviePy.")
    parser.add_argument("--requirements-file", type=Path, default=DEFAULT_REQUIREMENTS)
    args = parser.parse_args()

    if not MANIM_AVAILABLE:
        if args.render:
            raise SystemExit(
                "Manim is not installed in this environment. Install it with: "
                f"pip install -r {args.requirements_file}"
            )
        print("Manim is not installed in this environment.")
        print(f"Install it with:\n  pip install -r {args.requirements_file}")
        print("Then render with:")
        print(f"  python {Path(__file__).name} --render --scene MVPRaceShort --quality h")
        return

    if not args.render:
        print("Scene classes available: HookScene, Top3IntroScene, StatsBarsScene, ScoreRevealScene, PodiumScene, CTAScene, MVPRaceShort")
        print(f"Example render:\n  python {Path(__file__).name} --render --scene MVPRaceShort --quality h")
        print(f"Requirements file:\n  {args.requirements_file}")
        return

    status = run_manim(args.scene, args.media_dir, args.quality, args.output_stem)
    if status != 0:
        raise SystemExit(status)

    if args.mix_audio:
        quality_folder = {
            "l": "480p15",
            "m": "720p30",
            "h": "1920p30",
            "p": "1440p60",
            "k": "2160p60",
        }.get(args.quality, "1920p30")
        rendered = args.media_dir / "videos" / Path(__file__).stem / quality_folder / f"{args.output_stem}.mp4"
        mixed = rendered.with_name(f"{rendered.stem}_audio.mp4")
        attach_audio(rendered, mixed, args.audio)
        print(f"[manim] audio mix ready -> {mixed}")


if __name__ == "__main__":
    main()
