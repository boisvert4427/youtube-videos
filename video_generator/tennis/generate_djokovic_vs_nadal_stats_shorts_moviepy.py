from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageOps

from video_generator.tennis import generate_federer_vs_nadal_stats_shorts_moviepy as base


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PHOTOS_DIR = PROJECT_ROOT / "data" / "raw" / "player_photos"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "tennis" / "djokovic_vs_nadal_stats_shorts.mp4"

DJOKOVIC_BLUE = (35, 76, 160)
NADAL_ORANGE = (238, 123, 41)

STAT_ROWS = [
    ("TOTAL SLAMS", "24", "22"),
    ("AUSTRALIAN OPEN", "10", "2"),
    ("ROLAND-GARROS", "3", "14"),
    ("WIMBLEDON", "7", "2"),
    ("US OPEN", "4", "4"),
    ("ATP FINALS", "7", "0"),
    ("OLYMPIC TITLES", "1", "2"),
    ("YEAR-END #1", "8", "5"),
    ("WEEKS #1", "428", "209"),
    ("WIN %", "83.5", "82.6"),
    ("H2H", "31", "29"),
    ("TOP 10 WINS", "265", "186"),
    ("WIN STREAK", "43", "32"),
    ("MASTERS 1000", "40", "36"),
]


def _load_portraits() -> dict[str, Image.Image]:
    mapping = {
        "djokovic": PHOTOS_DIR / "novak_djokovic.jpg",
        "nadal": PHOTOS_DIR / "rafael_nadal.jpg",
    }
    cache: dict[str, Image.Image] = {}
    for key, path in mapping.items():
        img = ImageOps.exif_transpose(Image.open(path)).convert("RGBA")
        cache[key] = img
    return cache


def _final_stamp_text() -> str:
    left_total, right_total = base._score_totals()
    if left_total == right_total:
        return "YOU DECIDE"
    return "DJOKOVIC WINS" if left_total > right_total else "NADAL WINS"


def _winner_word(left_value: str, right_value: str) -> str:
    winner = base._winner_side(left_value, right_value)
    if winner == "left":
        return "Djokovic"
    if winner == "right":
        return "Nadal"
    return "Tie"


def _build_narration_text() -> str:
    final_text = _final_stamp_text()
    ending = (
        "Et au final, Djokovic gagne ce duel."
        if final_text == "DJOKOVIC WINS"
        else "Et au final, Nadal gagne ce duel."
        if final_text == "NADAL WINS"
        else "Et au final, c'est à vous de décider."
    )
    return (
        "Djokovic contre Nadal. "
        "Djokovic domine à l'Open d'Australie, au Masters, en semaines numéro un et au total des Grands Chelems. "
        "Nadal reste intouchable à Roland Garros et garde l'avantage sur les titres olympiques. "
        "Le face à face est serré, le pourcentage de victoires est très proche, et le débat reste énorme. "
        f"{ending}"
    )


def _disable_voiceover(text: str):
    return None


def _disable_presenter_overlay(frame, t: float, voice_active: bool, title_font, sub_font) -> None:
    return None


def _draw_top_player_cards(
    frame: Image.Image,
    portraits: dict[str, Image.Image],
    title_font,
    sub_font,
) -> None:
    draw = base.ImageDraw.Draw(frame, "RGBA")
    left_box = (42, 104, 500, 422)
    right_box = (580, 104, 1038, 422)
    draw.rounded_rectangle(left_box, radius=42, fill=(10, 26, 60, 222), outline=(*DJOKOVIC_BLUE, 168), width=2)
    draw.rounded_rectangle(right_box, radius=42, fill=(64, 28, 10, 222), outline=(*NADAL_ORANGE, 168), width=2)

    left_portrait = base._make_top_portrait(portraits["djokovic"], DJOKOVIC_BLUE)
    right_portrait = base._make_top_portrait(portraits["nadal"], NADAL_ORANGE)
    frame.alpha_composite(left_portrait, (74, 146))
    frame.alpha_composite(right_portrait, (612, 146))

    base._draw_glow_text(frame, (354, 214), "DJOKOVIC", title_font, base.WHITE, DJOKOVIC_BLUE)
    base._draw_glow_text(frame, (892, 214), "NADAL", title_font, base.WHITE, NADAL_ORANGE)
    draw.text((354, 266), "NOVAK", font=sub_font, fill="#deebff", anchor="ma")
    draw.text((892, 266), "RAFAEL", font=sub_font, fill="#ffe7d7", anchor="ma")


def _draw_center_stat_scene(
    frame: Image.Image,
    small_font,
    list_font,
    value_font_map: dict[str, object],
    stat_index: int | None,
    phase: float,
) -> None:
    draw = base.ImageDraw.Draw(frame, "RGBA")
    panel = (34, 476, base.WIDTH - 34, 1658)
    draw.rounded_rectangle(panel, radius=48, fill=(7, 18, 36, 220), outline=(255, 255, 255, 20), width=2)
    draw.text((170, 540), "DJOKOVIC", font=small_font, fill="#deebff", anchor="ma")
    draw.text((base.WIDTH // 2, 540), "STATS", font=small_font, fill="#f4f7fb", anchor="ma")
    draw.text((base.WIDTH - 170, 540), "NADAL", font=small_font, fill="#ffe5d8", anchor="ma")

    left_target_x = 238
    center_x = base.WIDTH // 2
    right_target_x = base.WIDTH - 238
    left_start_x = 238
    right_start_x = base.WIDTH - 238
    final_board = stat_index is None

    row_h = 76
    for idx, (stat_name, left_value, right_value) in enumerate(base.STAT_ROWS):
        row_top = 592 + idx * row_h
        row_bottom = row_top + 70
        row_center_y = (row_top + row_bottom) // 2
        active = stat_index is not None and idx == stat_index
        revealed = final_board or (stat_index is not None and idx < stat_index)
        fill = (20, 50, 86, 255) if (active or revealed) else (255, 255, 255, 10)
        outline = (*base.WHITE, 32) if (active or revealed) else (255, 255, 255, 12)
        draw.rounded_rectangle((84, row_top, base.WIDTH - 84, row_bottom), radius=18, fill=fill, outline=outline, width=2)
        row_phase = 1.0 if revealed else max(0.0, min(1.0, phase / 0.72))
        row_visible = revealed or active

        if row_visible:
            stat_y = row_center_y - 14 + int((1.0 - row_phase) * 10)
            label_fill = base.WHITE if active else (220, 229, 239)
            base._draw_glow_text(frame, (center_x, stat_y), stat_name, list_font, label_fill, base.WHITE)
            left_font = value_font_map[left_value]
            right_font = value_font_map[right_value]

            if active and not revealed:
                left_x = int(left_start_x + (left_target_x - left_start_x) * row_phase)
                right_x = int(right_start_x + (right_target_x - right_start_x) * row_phase)
            else:
                left_x = left_target_x
                right_x = right_target_x

            base._draw_glow_text(frame, (left_x, stat_y), left_value, left_font, base.WHITE, DJOKOVIC_BLUE)
            base._draw_glow_text(frame, (right_x, stat_y), right_value, right_font, base.WHITE, NADAL_ORANGE)
            winner_side = base._winner_side(left_value, right_value)

            check_phase = 1.0 if revealed else max(0.0, min(1.0, (phase - 0.56) / 0.34))
            if check_phase > 0 and winner_side != "tie":
                if winner_side == "left":
                    x0, y0 = 118, row_top + 34
                    x1, y1 = 128, row_top + 46
                    x2, y2 = 148, row_top + 20
                else:
                    x0, y0 = base.WIDTH - 148, row_top + 34
                    x1, y1 = base.WIDTH - 138, row_top + 46
                    x2, y2 = base.WIDTH - 118, row_top + 20

                if check_phase < 0.5:
                    mid = check_phase / 0.5
                    xe = int(x0 + (x1 - x0) * mid)
                    ye = int(y0 + (y1 - y0) * mid)
                    draw.line((x0, y0, xe, ye), fill=(72, 255, 142, 255), width=6)
                else:
                    draw.line((x0, y0, x1, y1), fill=(72, 255, 142, 255), width=6)
                    mid = (check_phase - 0.5) / 0.5
                    xe = int(x1 + (x2 - x1) * mid)
                    ye = int(y1 + (y2 - y1) * mid)
                    draw.line((x1, y1, xe, ye), fill=(72, 255, 142, 255), width=6)

    if final_board:
        left_total, right_total = base._score_totals()
        score_phase = max(0.0, min(1.0, phase / 0.42))
        score_y = 1714
        draw.rounded_rectangle((74, score_y, base.WIDTH - 74, score_y + 154), radius=30, fill=(10, 28, 54, 220), outline=(255, 255, 255, 18), width=2)
        draw.text((208, score_y + 38), "POINTS", font=base._load_font(24, bold=True), fill="#deebff", anchor="ma")
        draw.text((base.WIDTH - 208, score_y + 38), "POINTS", font=base._load_font(24, bold=True), fill="#ffe5d8", anchor="ma")
        left_score = int(round(left_total * score_phase))
        right_score = int(round(right_total * score_phase))
        left_score_font = base._fit_font_size(draw, str(left_total), 80, 52, 22, bold=True)
        right_score_font = base._fit_font_size(draw, str(right_total), 80, 52, 22, bold=True)
        base._draw_glow_text(frame, (208, score_y + 102), str(left_score), left_score_font, base.WHITE, DJOKOVIC_BLUE)
        base._draw_glow_text(frame, (base.WIDTH - 208, score_y + 102), str(right_score), right_score_font, base.WHITE, NADAL_ORANGE)
        draw.text((base.WIDTH // 2, score_y + 102), "SCORE", font=base._load_font(30, bold=True), fill="#f4f7fb", anchor="ma")

        stamp_phase = max(0.0, min(1.0, (phase - 0.22) / 0.50))
        if stamp_phase > 0:
            stamp = base.Image.new("RGBA", (base.WIDTH, base.HEIGHT), (0, 0, 0, 0))
            sd = base.ImageDraw.Draw(stamp, "RGBA")
            stamp_text = _final_stamp_text()
            stamp_box = (168, 1226, 912, 1498)
            stamp_red = (155, 30, 42)
            fill_red = (120, 10, 14, int(44 * stamp_phase))
            sd.rounded_rectangle(stamp_box, radius=38, fill=fill_red, outline=(*stamp_red, int(252 * stamp_phase)), width=14)
            sd.rounded_rectangle((198, 1180, 882, 1402), radius=32, outline=(255, 214, 214, int(84 * stamp_phase)), width=3)
            stamp_font = base._fit_font_size(sd, stamp_text, 600, 108, 36, bold=True)
            sd.text((540, 1362), stamp_text, font=stamp_font, fill=(255, 234, 234, int(255 * stamp_phase)), anchor="ma", stroke_width=3, stroke_fill=(*stamp_red, int(255 * stamp_phase)))
            sd.text((540, 1362), stamp_text, font=stamp_font, fill=(*stamp_red, int(255 * stamp_phase)), anchor="ma")
            stamp = stamp.rotate(-10, resample=base.Image.Resampling.BICUBIC, center=(540, 1362))
            frame.alpha_composite(stamp)


def _patch_base_module() -> None:
    base.DEFAULT_OUTPUT = DEFAULT_OUTPUT
    base.FEDERER_RED = DJOKOVIC_BLUE
    base.NADAL_ORANGE = NADAL_ORANGE
    base.STAT_ROWS = STAT_ROWS
    base.MUSIC_VOLUME = 1.0
    base._load_portraits = _load_portraits
    base._final_stamp_text = _final_stamp_text
    base._winner_word = _winner_word
    base._build_narration_text = _build_narration_text
    base._synthesize_voiceover = _disable_voiceover
    base._draw_presenter_overlay = _disable_presenter_overlay
    base._draw_top_player_cards = _draw_top_player_cards
    base._draw_center_stat_scene = _draw_center_stat_scene


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a Djokovic vs Nadal tennis stats Shorts video.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--audio", type=Path, default=base.DEFAULT_AUDIO)
    parser.add_argument("--duration", type=float, default=base.TOTAL_DURATION)
    parser.add_argument("--fps", type=int, default=base.FPS)
    return parser.parse_args()


def main() -> None:
    _patch_base_module()
    args = parse_args()
    output = base.render_video(args.output, args.audio, args.duration, args.fps)
    print(f"[video_generator] Djokovic vs Nadal stats Shorts generated -> {output}")


if __name__ == "__main__":
    main()
