from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from moviepy import VideoClip
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps

from video_generator.generate_ucl_barchart_race_moviepy import _fit_font_size, _load_font, build_audio_track

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_INPUT = PROJECT_ROOT / 'data' / 'processed' / 'basketball' / 'nba_championships_timeseries_1947_2025.csv'
DEFAULT_OUTPUT = PROJECT_ROOT / 'data' / 'processed' / 'basketball' / 'nba_championship_leaders_short_1947_2025_80s.mp4'
DEFAULT_AUDIO = PROJECT_ROOT / 'data' / 'raw' / 'audio' / 'Midnight_Grip_20260402_0828.mp3'
NBA_LOGO = PROJECT_ROOT / 'data' / 'raw' / 'nba_logo.png'
TROPHY = PROJECT_ROOT / 'data' / 'raw' / 'nba_trophy_photo_alt.png'
LOGO_DIR = PROJECT_ROOT / 'data' / 'raw' / 'nba_team_logos'

WIDTH, HEIGHT = 1080, 1920
FPS = 30
TOTAL_DURATION = 80.0
INTRO_DURATION = 5.5
OUTRO_DURATION = 9.0
TOP_N = 10
TITLE = 'NBA CHAMPIONSHIP LEADERS'
SUBTITLE = 'Official NBA history | 1947-2025'

TEAM_COLORS = {
    'Atlanta Hawks': ('#E03A3E', '#C1D32F'),
    'Baltimore Bullets': ('#6C7A89', '#D9E2EC'),
    'Boston Celtics': ('#007A33', '#BA9653'),
    'Brooklyn Nets': ('#111111', '#F4F4F4'),
    'Chicago Bulls': ('#CE1141', '#111111'),
    'Cleveland Cavaliers': ('#6F263D', '#FFB81C'),
    'Dallas Mavericks': ('#00538C', '#B8C4CA'),
    'Denver Nuggets': ('#0E2240', '#FEC524'),
    'Detroit Pistons': ('#1D42BA', '#C8102E'),
    'Golden State Warriors': ('#1D428A', '#FFC72C'),
    'Houston Rockets': ('#CE1141', '#C4CED4'),
    'Los Angeles Lakers': ('#552583', '#FDB927'),
    'Miami Heat': ('#98002E', '#F9A01B'),
    'Milwaukee Bucks': ('#00471B', '#EEE1C6'),
    'New York Knicks': ('#006BB6', '#F58426'),
    'Oklahoma City Thunder': ('#007AC1', '#EF3B24'),
    'Orlando Magic': ('#0077C0', '#C4CED4'),
    'Philadelphia 76ers': ('#006BB6', '#ED174C'),
    'Phoenix Suns': ('#1D1160', '#E56020'),
    'Portland Trail Blazers': ('#E03A3E', '#111111'),
    'Sacramento Kings': ('#5A2D81', '#C4CED4'),
    'San Antonio Spurs': ('#111111', '#C4CED4'),
    'Toronto Raptors': ('#CE1141', '#111111'),
    'Utah Jazz': ('#002B5C', '#F9A01B'),
    'Washington Wizards': ('#002B5C', '#E31837'),
}
TEAM_ABBR = {
    'Atlanta Hawks': 'ATL', 'Baltimore Bullets': 'BLT', 'Boston Celtics': 'BOS', 'Brooklyn Nets': 'BKN',
    'Chicago Bulls': 'CHI', 'Cleveland Cavaliers': 'CLE', 'Dallas Mavericks': 'DAL', 'Denver Nuggets': 'DEN',
    'Detroit Pistons': 'DET', 'Golden State Warriors': 'GSW', 'Houston Rockets': 'HOU', 'Los Angeles Lakers': 'LAL',
    'Miami Heat': 'MIA', 'Milwaukee Bucks': 'MIL', 'New York Knicks': 'NYK', 'Oklahoma City Thunder': 'OKC',
    'Orlando Magic': 'ORL', 'Philadelphia 76ers': 'PHI', 'Phoenix Suns': 'PHX', 'Portland Trail Blazers': 'POR',
    'Sacramento Kings': 'SAC', 'San Antonio Spurs': 'SAS', 'Toronto Raptors': 'TOR', 'Utah Jazz': 'UTA',
    'Washington Wizards': 'WAS',
}

@dataclass(frozen=True)
class TeamState:
    team_name: str
    team_abbr: str
    titles: float

@dataclass(frozen=True)
class Snapshot:
    ranking_date: str
    year: int
    season_summary: str
    states: list[TeamState]


def hex_to_rgb(value: str) -> tuple[int, int, int]:
    value = value.lstrip('#')
    return tuple(int(value[i:i+2], 16) for i in (0, 2, 4))


def mix_rgb(color: str, target: tuple[int, int, int], amount: float) -> tuple[int, int, int]:
    r, g, b = hex_to_rgb(color)
    amount = min(max(amount, 0.0), 1.0)
    return (int(r + (target[0] - r) * amount), int(g + (target[1] - g) * amount), int(b + (target[2] - b) * amount))


def text_on(color: str) -> str:
    r, g, b = hex_to_rgb(color)
    lum = (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255.0
    return '#10233F' if lum > 0.67 else '#F4F7FB'


def ease(value: float) -> float:
    value = min(max(value, 0.0), 1.0)
    return value * value * (3.0 - 2.0 * value)


def smooth(value: float) -> float:
    value = min(max(value, 0.0), 1.0)
    return value * value * (3.0 - 2.0 * value)


def rank_y(prev_idx: float, next_idx: float, alpha: float) -> float:
    alpha = min(max(alpha, 0.0), 1.0)
    if math.isclose(prev_idx, next_idx):
        return float(next_idx)
    dist = abs(next_idx - prev_idx)
    steps = max(1, int(math.ceil(dist)))
    direction = 1.0 if next_idx > prev_idx else -1.0
    gap = 1.0 / steps
    span = min(0.9, gap * 1.25)
    moved = 0.0
    end_moved = 0.0
    for step in range(steps):
        start = step * gap
        seg = min(1.0, max(0.0, dist - step))
        local = min(max((alpha - start) / span, 0.0), 1.0)
        end_local = min(max((1.0 - start) / span, 0.0), 1.0)
        moved += smooth(local) * seg
        end_moved += smooth(end_local) * seg
    if end_moved > 1e-9:
        moved *= dist / end_moved
    return float(prev_idx + direction * min(dist, moved))


def truncate(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int) -> str:
    if max_width <= 0:
        return ''
    if draw.textbbox((0, 0), text, font=font)[2] <= max_width:
        return text
    lo, hi = 0, len(text)
    best = '...'
    while lo <= hi:
        mid = (lo + hi) // 2
        candidate = text[:mid].rstrip() + '...'
        if draw.textbbox((0, 0), candidate, font=font)[2] <= max_width:
            best = candidate
            lo = mid + 1
        else:
            hi = mid - 1
    return best


def wrap_years(years: list[int], per_line: int = 6) -> str:
    if not years:
        return ''
    return '\n'.join(' · '.join(str(y) for y in years[i:i+per_line]) for i in range(0, len(years), per_line))


def slug(team_name: str) -> str:
    s = team_name.lower().replace('.', '').replace('&', 'and').replace('-', '_').replace(' ', '_')
    while '__' in s:
        s = s.replace('__', '_')
    return s


def logo_path(team_name: str) -> Path:
    return LOGO_DIR / f'{slug(team_name)}.png'


def background() -> Image.Image:
    xx = np.linspace(0, 1, WIDTH, dtype=np.float32)
    yy = np.linspace(0, 1, HEIGHT, dtype=np.float32)
    gx, gy = np.meshgrid(xx, yy)
    navy = np.array([6, 14, 34], dtype=np.float32)
    blue = np.array([14, 46, 86], dtype=np.float32)
    gold = np.array([239, 193, 86], dtype=np.float32)
    purple = np.array([111, 60, 173], dtype=np.float32)
    mix = np.clip(0.7 * gy + 0.15 * gx, 0, 1)
    glow = np.exp(-(((gx - 0.55) / 0.35) ** 2 + ((gy - 0.12) / 0.12) ** 2))
    side = np.exp(-(((gx - 0.12) / 0.16) ** 2 + ((gy - 0.52) / 0.25) ** 2)) + np.exp(-(((gx - 0.88) / 0.16) ** 2 + ((gy - 0.52) / 0.25) ** 2))
    img = np.clip(
        navy[None, None, :] * (1.0 - mix[..., None])
        + blue[None, None, :] * (0.82 * mix[..., None])
        + gold[None, None, :] * (0.07 * glow[..., None])
        + purple[None, None, :] * (0.06 * side[..., None]),
        0, 255
    ).astype(np.uint8)
    frame = Image.fromarray(img, 'RGB').convert('RGBA')
    overlay = Image.new('RGBA', (WIDTH, HEIGHT), (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay, 'RGBA')
    d.rounded_rectangle((48, 84, WIDTH - 48, HEIGHT - 102), radius=50, outline=(255, 255, 255, 18), width=2)
    d.line((136, 1498, WIDTH - 136, 1498), fill=(255, 255, 255, 12), width=2)
    d.line((WIDTH // 2, 320, WIDTH // 2, HEIGHT - 240), fill=(255, 255, 255, 8), width=2)
    d.ellipse((188, 248, WIDTH - 188, 790), outline=(255, 255, 255, 14), width=3)
    return frame.alpha_composite(overlay)


def load_snapshots(csv_path: Path) -> list[Snapshot]:
    grouped: dict[str, list[TeamState]] = defaultdict(list)
    summaries: dict[str, str] = {}
    with csv_path.open('r', newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            date = row['ranking_date'].strip()
            grouped[date].append(TeamState(row['team_name'].strip(), row['team_abbr'].strip(), float(row['titles'])))
            summaries[date] = row.get('season_summary', '').strip()
    snaps = []
    for date in sorted(grouped):
        snaps.append(Snapshot(date, int(date[:4]), summaries.get(date, ''), sorted(grouped[date], key=lambda s: (-s.titles, s.team_name))))
    return snaps


def build_priorities(snaps: list[Snapshot]) -> list[dict[str, int]]:
    out: list[dict[str, int]] = []
    prev: dict[str, int] | None = None
    for snap in snaps:
        ranked = sorted((s for s in snap.states if s.titles > 0), key=lambda s: (-s.titles, (prev or {}).get(s.team_name, 10_000), s.team_name))
        cur = {s.team_name: i for i, s in enumerate(ranked)}
        out.append(cur)
        prev = cur
    return out


def interp(prev: Snapshot, nxt: Snapshot, alpha: float) -> list[TeamState]:
    a = {s.team_name: s for s in prev.states}
    b = {s.team_name: s for s in nxt.states}
    names = sorted(set(a) | set(b))
    vals = []
    for name in names:
        x = a.get(name) or b[name]
        y = b.get(name) or a[name]
        titles = x.titles + (y.titles - x.titles) * alpha
        vals.append(TeamState(name, y.team_abbr or x.team_abbr, titles))
    return vals


def title_years(snaps: list[Snapshot]) -> dict[str, list[int]]:
    years: dict[str, list[int]] = defaultdict(list)
    prev: dict[str, int] = defaultdict(int)
    for snap in snaps:
        cur = {s.team_name: int(round(s.titles)) for s in snap.states}
        for team, count in cur.items():
            if count > prev.get(team, 0):
                years[team].append(snap.year)
        prev = cur
    return years


def build_badges(states: list[TeamState], size: int = 66) -> dict[str, Image.Image]:
    cache: dict[str, Image.Image] = {}
    for s in states:
        if s.team_name in cache:
            continue
        badge = Image.new('RGBA', (size + 8, size + 8), (0, 0, 0, 0))
        d = ImageDraw.Draw(badge, 'RGBA')
        primary, secondary = TEAM_COLORS.get(s.team_name, ('#39C0FF', '#F4F7FB'))
        d.rounded_rectangle((2, 2, size + 6, size + 6), radius=18, fill=(11, 22, 42, 235), outline=mix_rgb(primary, (0, 0, 0), 0.28) + (255,), width=2)
        d.rounded_rectangle((7, 7, size + 1, size + 1), radius=14, fill=(255, 255, 255, 18))
        path = logo_path(s.team_name)
        if path.exists():
            logo = Image.open(path).convert('RGBA')
            logo = ImageOps.contain(logo, (size - 10, size - 10))
            badge.alpha_composite(logo, ((badge.width - logo.width) // 2, (badge.height - logo.height) // 2))
        else:
            font = _fit_font_size(d, s.team_abbr, size - 12, 22, 12, bold=True)
            bbox = d.textbbox((0, 0), s.team_abbr, font=font)
            d.text(((badge.width - (bbox[2] - bbox[0])) // 2, (badge.height - (bbox[3] - bbox[1])) // 2 - 2), s.team_abbr, font=font, fill=text_on(primary))
        cache[s.team_name] = badge
    return cache


def draw_header(frame: Image.Image, draw: ImageDraw.ImageDraw, title_font, subtitle_font, year_font, summary_font, year: int, summary: str, trophy: Image.Image, nba_logo: Image.Image):
    draw.rounded_rectangle((54, 94, WIDTH - 54, 394), radius=42, fill=(6, 16, 34, 232), outline=(255, 255, 255, 18), width=2)
    draw.text((80, 128), TITLE, font=title_font, fill='#F4F7FB')
    draw.text((82, 200), SUBTITLE, font=subtitle_font, fill='#B8D2EE')
    draw.rounded_rectangle((78, 244, 220, 330), radius=26, fill=(255, 204, 82, 255))
    bbox = draw.textbbox((0, 0), str(year), font=year_font)
    draw.text((149 - (bbox[2] - bbox[0]) // 2, 255), str(year), font=year_font, fill='#12223D')
    draw.rounded_rectangle((244, 238, WIDTH - 76, 336), radius=26, fill=(14, 40, 76, 225), outline=(255, 255, 255, 18), width=2)
    line = truncate(draw, summary, summary_font, WIDTH - 360)
    sb = draw.textbbox((0, 0), line, font=summary_font)
    draw.text((266, 252), line, font=summary_font, fill='#EEF7FF')
    trophy_fit = ImageOps.contain(trophy, (220, 280)).copy()
    trophy_fit.putalpha(210)
    frame.alpha_composite(trophy_fit, (WIDTH - 286, 116))
    logo = ImageOps.contain(nba_logo, (190, 430)).copy()
    logo.putalpha(34)
    frame.alpha_composite(logo, (60, 700))


def draw_intro(frame: Image.Image, draw: ImageDraw.ImageDraw, snaps: list[Snapshot], badges: dict[str, Image.Image], title_font, subtitle_font, chip_font, trophy: Image.Image, nba_logo: Image.Image):
    latest = snaps[-1]
    draw_header(frame, draw, title_font, subtitle_font, _load_font(86, bold=True), _load_font(28, bold=True), latest.year, 'Watch the all-time NBA title race evolve by franchise.', trophy, nba_logo)
    leaders = latest.states[:3]
    start_x, chip_w, gap, y = 74, 294, 24, 560
    for i, state in enumerate(leaders):
        x = start_x + i * (chip_w + gap)
        p, _ = TEAM_COLORS.get(state.team_name, ('#39C0FF', '#F4F7FB'))
        draw.rounded_rectangle((x, y, x + chip_w, y + 132), radius=28, fill=(12, 26, 48, 232), outline=mix_rgb(p, (255, 255, 255), 0.22) + (255,), width=2)
        frame.alpha_composite(badges[state.team_name], (x + 12, y + 15))
        draw.text((x + 98, y + 18), f'#{i+1}', font=chip_font, fill='#F4F7FB')
        draw.text((x + 98, y + 56), state.team_name, font=_load_font(24, bold=True), fill='#F4F7FB')
        draw.text((x + 246, y + 18), str(int(round(state.titles))), font=_load_font(32, bold=True), fill='#FEC524', anchor='rm')
    draw.text((82, 472), 'Championship leaders, franchise by franchise.', font=_load_font(22, bold=True), fill='#D7E7F7')
    draw.rounded_rectangle((72, 1470, WIDTH - 72, 1808), radius=32, fill=(7, 18, 34, 228), outline=(255, 255, 255, 18), width=2)
    draw.text((96, 1510), 'TITLE YEARS', font=_load_font(24, bold=True), fill='#B8D2EE')
    leader = leaders[0]
    draw.text((96, 1558), f'{leader.team_name} — {int(round(leader.titles))} titles', font=_load_font(28, bold=True), fill='#F4F7FB')
    years = wrap_years(title_years(snaps)[leader.team_name], 6)
    years_font = _fit_font_size(draw, years, WIDTH - 180, 24, 14, bold=False)
    draw.multiline_text((96, 1618), years, font=years_font, fill='#ECF5FF', spacing=4)


def draw_race(frame: Image.Image, draw: ImageDraw.ImageDraw, snaps: list[Snapshot], priorities: list[dict[str, int]], badges: dict[str, Image.Image], years_map: dict[str, list[int]], title_font, subtitle_font, year_font, summary_font, row_font, value_font, rank_font, t: float):
    periods = len(snaps) - 1
    race_duration = TOTAL_DURATION - INTRO_DURATION - OUTRO_DURATION
    sec = race_duration / periods
    idx = min(int(t / sec), periods - 1)
    local = (t - idx * sec) / sec
    alpha = ease(local)
    prev, nxt = snaps[idx], snaps[idx + 1]
    states = interp(prev, nxt, alpha)
    state_map = {s.team_name: s for s in states}
    axis_cap = max(1.0, max((s.titles for s in prev.states[:TOP_N]), default=1.0) + 1.0)
    axis_cap = axis_cap + (max(1.0, max((s.titles for s in nxt.states[:TOP_N]), default=1.0) + 1.0) - axis_cap) * alpha
    prev_rank = {s.team_name: i for i, s in enumerate(sorted((s for s in prev.states if s.titles > 0), key=lambda s: (-s.titles, priorities[idx].get(s.team_name, 10_000), s.team_name))[:TOP_N])}
    nxt_rank = {s.team_name: i for i, s in enumerate(sorted((s for s in nxt.states if s.titles > 0), key=lambda s: (-s.titles, priorities[idx].get(s.team_name, 10_000), s.team_name))[:TOP_N])}
    visible = sorted(set(prev_rank) | set(nxt_rank))
    top_states = [state_map[n] for n in visible if n in state_map]
    draw_header(frame, draw, title_font, subtitle_font, year_font, summary_font, nxt.year, nxt.season_summary, TROPHY_IMG, NBA_IMG)
    draw.rounded_rectangle((72, 1470, WIDTH - 72, 1808), radius=32, fill=(7, 18, 34, 228), outline=(255, 255, 255, 18), width=2)
    leader = sorted((s for s in states if s.titles > 0), key=lambda s: (-s.titles, priorities[idx].get(s.team_name, 10_000), s.team_name))[0]
    draw.text((96, 1510), 'TITLE YEARS', font=_load_font(24, bold=True), fill='#B8D2EE')
    draw.text((96, 1558), f'{leader.team_name} — {int(round(leader.titles))} titles', font=_load_font(28, bold=True), fill='#F4F7FB')
    years = wrap_years(years_map[leader.team_name], 6)
    years_font = _fit_font_size(draw, years, WIDTH - 180, 24, 14, bold=False)
    draw.multiline_text((96, 1618), years, font=years_font, fill='#ECF5FF', spacing=4)
    row_h, gap, base_y = 100, 8, 410
    rank_w, bar_left, bar_max = 58, 254, 698
    for i in range(TOP_N):
        y = base_y + i * (row_h + gap)
        draw.rounded_rectangle((72, y + 10, 128, y + 86), radius=18, fill=(255, 204, 82, 255))
        text = str(i + 1)
        bb = draw.textbbox((0, 0), text, font=rank_font)
        draw.text((100 - (bb[2] - bb[0]) // 2, y + 34), text, font=rank_font, fill='#12223D')
    items = []
    for s in top_states:
        p = prev_rank.get(s.team_name, TOP_N + 1)
        n = nxt_rank.get(s.team_name, TOP_N + 1)
        y = base_y + rank_y(float(p), float(n), alpha) * (row_h + gap)
        bw = max(116, int((s.titles / axis_cap) * bar_max))
        items.append((1 if n < p else 0, y, s, bw, s.team_name == leader.team_name))
    items.sort(key=lambda x: (x[0], x[1]))
    for _, y, s, bw, is_leader in items:
        yy = int(y)
        primary, secondary = TEAM_COLORS.get(s.team_name, ('#39C0FF', '#F4F7FB'))
        fill = mix_rgb(primary, (10, 18, 34), 0.70)
        draw.rounded_rectangle((66, yy, WIDTH - 66, yy + row_h), radius=30, fill=(*fill, 226), outline=mix_rgb(primary, (255, 255, 255), 0.14) + (255,), width=2)
        if is_leader:
            draw.rounded_rectangle((60, yy - 2, WIDTH - 60, yy + row_h + 2), radius=34, outline=mix_rgb(primary, (255, 255, 255), 0.42) + (255,), width=4)
        frame.alpha_composite(badges[s.team_name], (106, yy + 12))
        draw.text((194, yy + 18), s.team_name, font=row_font, fill='#F4F7FB')
        draw.text((194, yy + 58), f'{int(round(s.titles))} CHAMPIONSHIPS', font=_load_font(18, bold=False), fill='#D7E7F7')
        draw.rounded_rectangle((bar_left, yy + 14, bar_left + bw, yy + row_h - 14), radius=22, fill=hex_to_rgb(primary), outline=mix_rgb(primary, (255, 255, 255), 0.18), width=2)
        draw.rounded_rectangle((bar_left + 10, yy + 20, bar_left + max(86, int(bw * 0.6)), yy + 32), radius=8, fill=(*mix_rgb(primary, (255, 255, 255), 0.32), 72))
        draw.line((bar_left + 16, yy + row_h - 18, bar_left + max(30, int(bw * 0.76)), yy + row_h - 18), fill=(*mix_rgb(primary, (0, 0, 0), 0.18), 88), width=3)
        value = str(int(round(s.titles)))
        vb = draw.textbbox((0, 0), value, font=value_font)
        vx = min(bar_left + bw + 18, WIDTH - 112 - (vb[2] - vb[0]))
        draw.text((vx, yy + 48), value, font=value_font, fill='#F4F7FB', anchor='mm')
        if is_leader:
            draw.text((WIDTH - 116, yy + 26), '★', font=rank_font, fill='#FEC524', anchor='mm')


def draw_outro(frame: Image.Image, draw: ImageDraw.ImageDraw, snaps: list[Snapshot], priorities: list[dict[str, int]], badges: dict[str, Image.Image], years_map: dict[str, list[int]], title_font, subtitle_font, year_font, summary_font, row_font, value_font, rank_font):
    latest = snaps[-1]
    draw_header(frame, draw, title_font, subtitle_font, year_font, summary_font, latest.year, latest.season_summary, TROPHY_IMG, NBA_IMG)
    draw.rounded_rectangle((64, 418, WIDTH - 64, 1808), radius=34, fill=(7, 18, 34, 230), outline=(255, 255, 255, 18), width=2)
    draw.text((86, 440), 'FINAL LEADERBOARD', font=_load_font(24, bold=True), fill='#B8D2EE')
    top_states = latest.states[:TOP_N]
    row_h, gap, base_y = 110, 10, 480
    bar_left, bar_max = 282, 650
    max_titles = max(s.titles for s in top_states)
    for i, s in enumerate(top_states):
        y = base_y + i * (row_h + gap)
        p, sec = TEAM_COLORS.get(s.team_name, ('#39C0FF', '#F4F7FB'))
        draw.rounded_rectangle((80, y, WIDTH - 80, y + row_h), radius=28, fill=(*mix_rgb(p, (8, 18, 34), 0.70), 228), outline=mix_rgb(p, (255, 255, 255), 0.14) + (255,), width=2)
        draw.rounded_rectangle((96, y + 18, 148, y + 86), radius=18, fill=(255, 204, 82, 255))
        text = str(i + 1)
        bb = draw.textbbox((0, 0), text, font=rank_font)
        draw.text((122 - (bb[2] - bb[0]) // 2, y + 31), text, font=rank_font, fill='#12223D')
        frame.alpha_composite(badges[s.team_name], (168, y + 13))
        draw.text((252, y + 18), s.team_name, font=row_font, fill='#F4F7FB')
        draw.text((252, y + 60), f'{int(round(s.titles))} TITLES', font=_load_font(18, bold=False), fill='#D7E7F7')
        bw = max(116, int((s.titles / max_titles) * bar_max))
        draw.rounded_rectangle((bar_left, y + 16, bar_left + bw, y + row_h - 16), radius=22, fill=hex_to_rgb(p), outline=mix_rgb(p, (255, 255, 255), 0.18), width=2)
        years = wrap_years(years_map[s.team_name], 5)
        years_font = _fit_font_size(draw, years, 560, 17, 12, bold=False)
        draw.multiline_text((bar_left + bw + 20, y + 18), years, font=years_font, fill='#ECF5FF', spacing=3)
        draw.text((WIDTH - 124, y + 52), str(int(round(s.titles))), font=value_font, fill='#FEC524', anchor='mm')
    draw.rounded_rectangle((66, 1618, WIDTH - 66, 1850), radius=30, fill=(11, 22, 42, 235), outline=(255, 255, 255, 16), width=2)
    leader = top_states[0]
    draw.text((90, 1652), 'TITLE YEARS', font=_load_font(24, bold=True), fill='#B8D2EE')
    draw.text((90, 1694), f'{leader.team_name} — {int(round(leader.titles))} titles', font=_load_font(28, bold=True), fill='#F4F7FB')
    yrs = wrap_years(years_map[leader.team_name], 8)
    yrs_font = _fit_font_size(draw, yrs, WIDTH - 160, 23, 14, bold=False)
    draw.multiline_text((90, 1750), yrs, font=yrs_font, fill='#ECF5FF', spacing=4)


def render(input_csv: Path, output_path: Path, audio_path: Path, duration: float, fps: int) -> Path:
    snaps = load_snapshots(input_csv)
    if len(snaps) < 2:
        raise RuntimeError('Not enough snapshots to render.')
    global badges, years_map, NBA_IMG, TROPHY_IMG
    badges = build_badges([s for snap in snaps for s in snap.states])
    years_map = title_years(snaps)
    priorities = build_priorities(snaps)
    NBA_IMG = Image.open(NBA_LOGO).convert('RGBA')
    TROPHY_IMG = Image.open(TROPHY).convert('RGBA')
    bg = background()
    title_font = _load_font(56, bold=True)
    subtitle_font = _load_font(24, bold=False)
    year_font = _load_font(80, bold=True)
    summary_font = _load_font(28, bold=True)
    row_font = _load_font(28, bold=True)
    value_font = _load_font(30, bold=True)
    rank_font = _load_font(26, bold=True)
    def make_frame(t: float) -> np.ndarray:
        frame = bg.copy()
        draw = ImageDraw.Draw(frame, 'RGBA')
        if t < INTRO_DURATION:
            draw_intro(frame, draw, snaps, badges, title_font, subtitle_font, _load_font(22, bold=True), TROPHY_IMG, NBA_IMG)
        elif t > duration - OUTRO_DURATION:
            draw_outro(frame, draw, snaps, priorities, badges, years_map, title_font, subtitle_font, year_font, summary_font, row_font, value_font, rank_font)
        else:
            draw_race(frame, draw, snaps, priorities, badges, years_map, title_font, subtitle_font, year_font, summary_font, row_font, value_font, rank_font, t - INTRO_DURATION)
        return np.array(frame.convert('RGB'))
    clip = VideoClip(make_frame, duration=duration)
    audio_clip, keep_alive = build_audio_track(audio_path, duration)
    clip = clip.with_audio(audio_clip)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    clip.write_videofile(str(output_path), fps=fps, codec='libx264', audio_codec='aac')
    clip.close(); audio_clip.close()
    for item in keep_alive:
        item.close()
    return output_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Generate a vertical NBA championships short inspired by the reference video.')
    p.add_argument('--input', type=Path, default=DEFAULT_INPUT)
    p.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    p.add_argument('--audio', type=Path, default=DEFAULT_AUDIO)
    p.add_argument('--duration', type=float, default=TOTAL_DURATION)
    p.add_argument('--fps', type=int, default=FPS)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    output = render(args.input, args.output, args.audio, args.duration, args.fps)
    print(f'[video_generator] NBA championships short generated -> {output}')


if __name__ == '__main__':
    main()
