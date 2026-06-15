from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
from moviepy import VideoClip
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps
import requests

from video_generator.generate_ucl_barchart_race_moviepy import (
    DEFAULT_AUDIO,
    _fit_font_size,
    _load_font,
    build_audio_track,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_INPUT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "finance"
    / "forbes_billionaires"
    / "top_companies_20y_daily_combined.csv"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "finance"
    / "company_values"
    / "company_value_race_2006_2026_4min.mp4"
)
DEFAULT_LOGOS_DIR = PROJECT_ROOT / "data" / "raw" / "company_logos"

WIDTH = 1920
HEIGHT = 1080
TOP_N = 12
FPS = 60
TOTAL_DURATION = 240.0
FINAL_HOLD_DURATION = 10.0

TITLE = "COMPANY VALUE RACE"
SUBTITLE = "TOP 12 | ESTIMATED MARKET CAP | 2006-2026"
LEFT_HEADER_LABEL = "COMPANY"
RIGHT_HEADER_LABEL = "VALUE (USD)"
FOOTER = "ESTIMATED MARKET CAP FROM PRICE X SHARES"

DISPLAY_NAME_ALIASES = {
    "000660.KS": "SK hynix",
    "005930.KS": "Samsung Electronics",
    "0700.HK": "Tencent",
    "0939.HK": "China Construction Bank",
    "1288.HK": "Agricultural Bank of China",
    "1398.HK": "ICBC",
    "2222.SR": "Saudi Aramco",
    "AAPL": "Apple",
    "ABBV": "AbbVie",
    "AMD": "Advanced Micro Devices",
    "AMZN": "Amazon",
    "ASML": "ASML",
    "AVGO": "Broadcom",
    "AZN": "AstraZeneca",
    "BABA": "Alibaba",
    "BAC": "Bank of America",
    "BRK-B": "Berkshire Hathaway",
    "CAT": "Caterpillar",
    "COST": "Costco",
    "CSCO": "Cisco",
    "CVX": "Chevron",
    "GE": "General Electric",
    "GOOGL": "Alphabet",
    "HD": "Home Depot",
    "HSBC": "HSBC",
    "JNJ": "Johnson & Johnson",
    "JPM": "JPMorgan Chase",
    "KO": "Coca-Cola",
    "LLY": "Eli Lilly",
    "LRCX": "Lam Research",
    "MA": "Mastercard",
    "MC.PA": "LVMH",
    "META": "Meta Platforms",
    "MRK": "Merck",
    "MSFT": "Microsoft",
    "MU": "Micron Technology",
    "NFLX": "Netflix",
    "NOVN.SW": "Novartis",
    "NVDA": "Nvidia",
    "ORCL": "Oracle",
    "PG": "Procter & Gamble",
    "PLTR": "Palantir",
    "ROG.SW": "Roche",
    "TM": "Toyota",
    "TSLA": "Tesla",
    "TSM": "TSMC",
    "V": "Visa",
    "WMT": "Walmart",
    "XOM": "Exxon Mobil",
}

COMPANY_PAGE_SLUGS = {
    "000660.KS": "sk-hynix",
    "005930.KS": "samsung",
    "0700.HK": "tencent",
    "0939.HK": "china-construction-bank",
    "1288.HK": "agricultural-bank-of-china",
    "1398.HK": "icbc",
    "2222.SR": "saudi-aramco",
    "AAPL": "apple",
    "ABBV": "abbvie",
    "AMD": "amd",
    "AMZN": "amazon",
    "ASML": "asml",
    "AVGO": "broadcom",
    "AZN": "astrazeneca",
    "BABA": "alibaba",
    "BAC": "bank-of-america",
    "BRK-B": "berkshire-hathaway",
    "CAT": "caterpillar",
    "COST": "costco",
    "CSCO": "cisco",
    "CVX": "chevron",
    "GE": "general-electric",
    "GOOGL": "alphabet-google",
    "HD": "home-depot",
    "HSBC": "hsbc",
    "JNJ": "johnson-and-johnson",
    "JPM": "jp-morgan-chase",
    "KO": "coca-cola",
    "LLY": "eli-lilly",
    "LRCX": "lam-research",
    "MA": "mastercard",
    "MC.PA": "lvmh",
    "META": "meta-platforms",
    "MRK": "merck",
    "MSFT": "microsoft",
    "MU": "micron-technology",
    "NFLX": "netflix",
    "NOVN.SW": "novartis",
    "NVDA": "nvidia",
    "ORCL": "oracle",
    "PG": "procter-and-gamble",
    "PLTR": "palantir",
    "ROG.SW": "roche",
    "TM": "toyota",
    "TSLA": "tesla",
    "TSM": "tsmc",
    "V": "visa",
    "WMT": "walmart",
    "XOM": "exxon-mobil",
}

EXCLUDED_TICKERS = {
    "000660.KS",
    "005930.KS",
}

MONTH_LABELS_EN = {
    1: "JAN",
    2: "FEB",
    3: "MAR",
    4: "APR",
    5: "MAY",
    6: "JUN",
    7: "JUL",
    8: "AUG",
    9: "SEP",
    10: "OCT",
    11: "NOV",
    12: "DEC",
}


@dataclass(frozen=True)
class CompanyState:
    ticker: str
    display_name: str
    index_value: float


@dataclass(frozen=True)
class Snapshot:
    ranking_date: str
    label: str
    states: list[CompanyState]


def _ease_in_out(value: float) -> float:
    value = min(max(value, 0.0), 1.0)
    return value * value * (3.0 - 2.0 * value)


def _phase_delay(value: float, delay: float, span: float) -> float:
    if span <= 0.0:
        return 1.0 if value >= delay else 0.0
    return min(max((value - delay) / span, 0.0), 1.0)


def _continuous_rank_position(previous: float, target: float, alpha: float) -> float:
    alpha = min(max(alpha, 0.0), 1.0)
    if math.isclose(previous, target):
        return target
    distance = abs(target - previous)
    steps = max(1, int(math.ceil(distance)))
    direction = 1.0 if target > previous else -1.0
    gap = 1.0 / steps
    span = min(0.9, gap * 1.35)
    travelled = 0.0
    end_travel = 0.0
    for step in range(steps):
        start = step * gap
        segment = min(1.0, max(0.0, distance - step))
        local = _ease_in_out(min(max((alpha - start) / span, 0.0), 1.0))
        end_local = _ease_in_out(min(max((1.0 - start) / span, 0.0), 1.0))
        travelled += local * segment
        end_travel += end_local * segment
    if end_travel > 1e-9:
        travelled *= distance / end_travel
    return previous + direction * min(distance, travelled)


def _hex_to_rgb(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    return tuple(int(value[index : index + 2], 16) for index in (0, 2, 4))


def _mix_rgb(color: str, target: tuple[int, int, int], amount: float) -> tuple[int, int, int]:
    red, green, blue = _hex_to_rgb(color)
    amount = min(max(amount, 0.0), 1.0)
    return (
        int(red + (target[0] - red) * amount),
        int(green + (target[1] - green) * amount),
        int(blue + (target[2] - blue) * amount),
    )


def _center_text(
    draw: ImageDraw.ImageDraw,
    rect: tuple[int, int, int, int],
    text: str,
    font: ImageFont.ImageFont,
    fill: str,
) -> None:
    bbox = draw.textbbox((0, 0), text, font=font)
    x = (rect[0] + rect[2] - (bbox[2] - bbox[0])) // 2 - bbox[0]
    y = (rect[1] + rect[3] - (bbox[3] - bbox[1])) // 2 - bbox[1]
    draw.text((x, y), text, font=font, fill=fill)


def _format_index(value: float) -> str:
    if value >= 1_000:
        return f"{value:,.0f}"
    if value >= 100:
        return f"{value:,.1f}"
    if value >= 10:
        return f"{value:,.2f}"
    return f"{value:,.3f}"


def _nice_number(value: float) -> float:
    if value <= 0:
        return 1.0
    exponent = math.floor(math.log10(value))
    fraction = value / (10**exponent)
    if fraction <= 1:
        nice_fraction = 1
    elif fraction <= 2:
        nice_fraction = 2
    elif fraction <= 2.5:
        nice_fraction = 2.5
    elif fraction <= 5:
        nice_fraction = 5
    else:
        nice_fraction = 10
    return nice_fraction * (10**exponent)


def _axis_scale(maximum: float) -> tuple[float, float]:
    step = _nice_number(maximum * 1.08 / 6.0)
    cap = math.ceil(maximum * 1.08 / step) * step
    return max(step, cap), step


def _month_label(date_text: str) -> str:
    year = int(date_text[:4])
    month = int(date_text[5:7])
    return f"{MONTH_LABELS_EN.get(month, 'UNK')} {year}"


def _display_name_for_ticker(ticker: str) -> str:
    ticker = ticker.strip().upper()
    if ticker in DISPLAY_NAME_ALIASES:
        return DISPLAY_NAME_ALIASES[ticker]
    if "." in ticker:
        return ticker.split(".", 1)[0]
    return ticker


def _logo_slug(ticker: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", ticker.strip().lower()).strip("_")


def _logo_initials(text: str) -> str:
    parts = [part for part in re.split(r"[^A-Za-z0-9]+", text) if part]
    if not parts:
        return "?"
    if len(parts) == 1:
        token = parts[0]
        return token[:2].upper()
    return "".join(part[0] for part in parts[:2]).upper()


def _slugify_company(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.strip().lower()).strip("-")


def _company_slug(ticker: str) -> str:
    ticker = ticker.strip().upper()
    if ticker in COMPANY_PAGE_SLUGS:
        return COMPANY_PAGE_SLUGS[ticker]
    return _slugify_company(_display_name_for_ticker(ticker))


def _parse_market_cap_value(text: str) -> float:
    cleaned = text.replace("$", "").replace(",", "").strip()
    if not cleaned:
        return 0.0
    parts = cleaned.split()
    if len(parts) == 1:
        try:
            return float(parts[0])
        except ValueError:
            return 0.0
    try:
        value = float(parts[0])
    except ValueError:
        return 0.0
    suffix = parts[1].upper()
    multipliers = {"T": 1_000_000_000_000.0, "B": 1_000_000_000.0, "M": 1_000_000.0, "K": 1_000.0}
    return value * multipliers.get(suffix, 1.0)


def _format_market_cap(value: float) -> str:
    abs_value = abs(value)
    if abs_value >= 1_000_000_000_000:
        scaled = value / 1_000_000_000_000
        text = f"${scaled:,.2f}T"
    elif abs_value >= 1_000_000_000:
        scaled = value / 1_000_000_000
        text = f"${scaled:,.2f}B"
    elif abs_value >= 1_000_000:
        scaled = value / 1_000_000
        text = f"${scaled:,.1f}M"
    else:
        text = f"${value:,.0f}"
    return text.replace(".00T", "T").replace(".00B", "B").replace(".0M", "M")


def _fetch_html(url: str) -> str:
    response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
    response.raise_for_status()
    return response.text


def _extract_marketcap_history_from_html(html: str) -> dict[int, float]:
    history: dict[int, float] = {}
    for year_text, value_text in re.findall(r"<tr><td>(20\d{2})</td><td>(\$[^<]+)</td><td", html):
        try:
            history[int(year_text)] = _parse_market_cap_value(value_text)
        except ValueError:
            continue
    return history


def _extract_current_marketcap_and_price(html: str) -> tuple[float | None, float | None]:
    market_cap = None
    share_price = None
    market_match = re.search(r'<div class="line1">\s*([^<]+?)\s*</div>\s*<div class="line2">Marketcap</div>', html, re.S)
    if market_match:
        market_cap = _parse_market_cap_value(market_match.group(1))
    price_match = re.search(r'<div class="line1">\s*([^<]+?)\s*</div>\s*<div class="line2">Share price</div>', html, re.S)
    if price_match:
        try:
            share_price = float(price_match.group(1).replace("$", "").replace(",", "").strip())
        except ValueError:
            share_price = None
    return market_cap, share_price


def _build_logo_cache(logos_dir: Path, logo_size: int) -> dict[str, Image.Image]:
    cache: dict[str, Image.Image] = {}
    if not logos_dir.exists():
        return cache
    for path in logos_dir.glob("*.png"):
        try:
            image = Image.open(path).convert("RGBA")
        except Exception:
            continue
        fitted = ImageOps.contain(image, (logo_size - 8, logo_size - 8), Image.Resampling.LANCZOS)
        canvas = Image.new("RGBA", (logo_size, logo_size), (0, 0, 0, 0))
        canvas.alpha_composite(
            fitted,
            ((logo_size - fitted.width) // 2, (logo_size - fitted.height) // 2),
        )
        cache[path.stem] = canvas
    return cache


def _make_background() -> Image.Image:
    x_values = np.linspace(0, 1, WIDTH, dtype=np.float32)
    y_values = np.linspace(0, 1, HEIGHT, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(x_values, y_values)

    deep = np.array([5, 9, 20], dtype=np.float32)
    navy = np.array([10, 25, 51], dtype=np.float32)
    gold = np.array([244, 194, 107], dtype=np.float32)
    cyan = np.array([102, 224, 210], dtype=np.float32)
    pink = np.array([255, 95, 130], dtype=np.float32)

    mix = np.clip(0.18 * grid_x + 0.70 * grid_y, 0.0, 1.0)
    top_glow = np.exp(-(((grid_x - 0.83) / 0.26) ** 2 + ((grid_y - 0.10) / 0.17) ** 2))
    right_glow = np.exp(-(((grid_x - 0.90) / 0.22) ** 2 + ((grid_y - 0.76) / 0.18) ** 2))
    left_glow = np.exp(-(((grid_x - 0.10) / 0.24) ** 2 + ((grid_y - 0.90) / 0.20) ** 2))

    pixels = np.clip(
        deep[None, None, :] * (1.0 - mix[..., None])
        + navy[None, None, :] * (0.88 * mix[..., None])
        + gold[None, None, :] * (0.10 * top_glow[..., None])
        + cyan[None, None, :] * (0.09 * right_glow[..., None])
        + pink[None, None, :] * (0.06 * left_glow[..., None]),
        0,
        255,
    ).astype(np.uint8)
    frame = Image.fromarray(pixels, mode="RGB").convert("RGBA")

    overlay = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")
    draw.rounded_rectangle((28, 24, WIDTH - 28, HEIGHT - 24), radius=42, outline=(255, 255, 255, 20), width=2)
    draw.ellipse((1460, -140, 2020, 420), outline=(102, 224, 210, 24), width=3)
    draw.ellipse((-120, 760, 440, 1320), outline=(244, 194, 107, 20), width=3)
    draw.line((54, 104, 1820, 104), fill=(244, 194, 107, 45), width=4)
    draw.line((92, 970, 980, 860), fill=(102, 224, 210, 12), width=2)
    draw.line((980, 940, 1840, 790), fill=(255, 95, 130, 12), width=2)
    overlay = overlay.filter(ImageFilter.GaussianBlur(radius=2))
    frame.alpha_composite(overlay)
    return frame


def load_snapshots(input_csv: Path) -> list[Snapshot]:
    rows_by_date: dict[str, dict[str, float]] = defaultdict(dict)

    with input_csv.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        for row in reader:
            date_text = row["Date"].strip()
            ticker = row["Ticker"].strip().upper()
            if ticker in EXCLUDED_TICKERS:
                continue
            try:
                value = float(row["Adj Close"])
            except (TypeError, ValueError):
                continue
            rows_by_date[date_text][ticker] = value

    if not rows_by_date:
        return []

    ordered_dates = sorted(rows_by_date)
    month_end_dates: dict[str, str] = {}
    year_end_prices: dict[str, dict[int, float]] = defaultdict(dict)
    for date_text in ordered_dates:
        month_end_dates[date_text[:7]] = date_text
        year = int(date_text[:4])
        for ticker, value in rows_by_date[date_text].items():
            year_end_prices[ticker][year] = value

    month_end_set = set(month_end_dates.values())
    all_tickers = sorted({ticker for day_values in rows_by_date.values() for ticker in day_values})
    available_years = sorted({int(date_text[:4]) for date_text in ordered_dates})

    share_factors: dict[str, dict[int, float]] = {}
    for ticker in all_tickers:
        slug = _company_slug(ticker)
        url = f"https://companiesmarketcap.com/{slug}/marketcap/"
        try:
            html = _fetch_html(url)
        except Exception:
            html = ""
        history = _extract_marketcap_history_from_html(html)
        current_cap, current_price = _extract_current_marketcap_and_price(html)
        factors: dict[int, float] = {}
        for year, market_cap in history.items():
            price = year_end_prices.get(ticker, {}).get(year)
            if price is None or price <= 0:
                continue
            factors[year] = market_cap / price
        if current_cap and current_price and current_price > 0 and available_years:
            factors[max(available_years)] = current_cap / current_price
        if not factors:
            factors[max(available_years)] = 1.0
        share_factors[ticker] = factors

    def _factor_for_year(ticker: str, year: int) -> float:
        factors = share_factors.get(ticker)
        if not factors:
            return 1.0
        if year in factors:
            return factors[year]
        eligible = [candidate for candidate in factors if candidate <= year]
        if eligible:
            return factors[max(eligible)]
        return factors[min(factors)]

    snapshots: list[Snapshot] = []
    latest_values: dict[str, float] = {}
    for date_text in ordered_dates:
        year = int(date_text[:4])
        for ticker, value in rows_by_date[date_text].items():
            factor = _factor_for_year(ticker, year)
            if factor > 0:
                latest_values[ticker] = value * factor

        if date_text not in month_end_set:
            continue
        states: list[CompanyState] = []
        for ticker, value in latest_values.items():
            if value <= 0:
                continue
            states.append(
                CompanyState(
                    ticker=ticker,
                    display_name=_display_name_for_ticker(ticker),
                    index_value=value,
                )
            )
        snapshots.append(
            Snapshot(
                ranking_date=date_text,
                label=_month_label(date_text),
                states=sorted(states, key=lambda state: (-state.index_value, state.display_name)),
            )
        )
    return snapshots


def _interpolate(prev: Snapshot, nxt: Snapshot, alpha: float) -> list[CompanyState]:
    previous = {state.ticker: state for state in prev.states}
    target = {state.ticker: state for state in nxt.states}
    states: list[CompanyState] = []
    for ticker in sorted(set(previous) | set(target)):
        before = previous.get(ticker)
        after = target.get(ticker)
        meta = after or before
        if meta is None:
            continue
        before_value = before.index_value if before else 0.0
        after_value = after.index_value if after else 0.0
        states.append(
            CompanyState(
                ticker=ticker,
                display_name=meta.display_name,
                index_value=before_value + (after_value - before_value) * alpha,
            )
        )
    return states


def _rank(states: list[CompanyState], top_n: int, priority: dict[str, int] | None = None) -> dict[str, int]:
    priority = priority or {}
    ranked = sorted(
        (state for state in states if state.index_value > 0),
        key=lambda state: (-state.index_value, priority.get(state.ticker, 10_000), state.display_name),
    )
    return {state.ticker: index for index, state in enumerate(ranked[:top_n])}


def _build_priorities(snapshots: list[Snapshot]) -> list[dict[str, int]]:
    priorities: list[dict[str, int]] = []
    previous: dict[str, int] = {}
    for snapshot in snapshots:
        ranked = sorted(
            snapshot.states,
            key=lambda state: (-state.index_value, previous.get(state.ticker, 10_000), state.display_name),
        )
        current = {state.ticker: index for index, state in enumerate(ranked)}
        priorities.append(current)
        previous = current
    return priorities


def _build_color_map(snapshots: list[Snapshot]) -> dict[str, str]:
    palette = [
        "#ef4444",
        "#f97316",
        "#f59e0b",
        "#eab308",
        "#84cc16",
        "#22c55e",
        "#14b8a6",
        "#06b6d4",
        "#3b82f6",
        "#6366f1",
        "#8b5cf6",
        "#d946ef",
        "#ec4899",
        "#fb7185",
    ]
    colors: dict[str, str] = {}
    fallback_index = 0
    for snapshot in snapshots:
        for state in snapshot.states:
            if state.ticker in colors:
                continue
            digest = __import__("hashlib").sha1(state.ticker.encode("utf-8")).digest()
            color = palette[int.from_bytes(digest[:2], "big") % len(palette)]
            colors[state.ticker] = color
            fallback_index += 1
    return colors


def render_video(
    input_csv: Path,
    output_path: Path,
    audio_path: Path,
    logos_dir: Path,
    duration: float,
    final_hold_duration: float,
    fps: int,
    top_n: int,
) -> Path:
    snapshots = load_snapshots(input_csv)
    if len(snapshots) < 2:
        raise RuntimeError("Not enough company snapshots to render.")

    first = snapshots[0]
    snapshots = [
        Snapshot(
            ranking_date=(datetime.strptime(first.ranking_date, "%Y-%m-%d") - timedelta(days=1)).strftime("%Y-%m-%d"),
            label=first.label,
            states=[],
        ),
        *snapshots,
    ]

    colors = _build_color_map(snapshots)
    priorities = _build_priorities(snapshots)
    periods = len(snapshots) - 1
    transition_duration = max(0.1, duration - max(0.0, final_hold_duration))
    seconds_per_period = transition_duration / periods

    axis_scales = [
        _axis_scale(max((state.index_value for state in snapshot.states[:top_n]), default=1.0))
        for snapshot in snapshots
    ]
    axis_scales[0] = axis_scales[1]

    background = _make_background()
    title_font = _load_font(58, bold=True)
    subtitle_font = _load_font(23, bold=True)
    insight_font = _load_font(22, bold=True)
    year_font = _load_font(62, bold=True)
    label_font = _load_font(18, bold=True)
    name_font = _load_font(28, bold=True)
    value_font = _load_font(28, bold=True)
    rank_font = _load_font(24, bold=True)
    tick_font = _load_font(17, bold=True)
    footer_font = _load_font(16, bold=True)
    logo_font = _load_font(16, bold=True)
    measure_canvas = Image.new("RGB", (1, 1))
    measure_draw = ImageDraw.Draw(measure_canvas)
    name_font_cache: dict[str, ImageFont.ImageFont] = {}
    logo_size = 36
    logo_gap = 14
    logo_cache = _build_logo_cache(logos_dir, logo_size)

    header_box = (38, 34, WIDTH - 38, 170)
    insight_box = (840, 54, 1450, 150)
    year_box = (1518, 50, 1848, 154)

    rank_left = 48
    name_left = 182
    name_text_left = name_left + logo_size + logo_gap
    bar_left = 430
    bar_right = 1762
    bar_max_width = bar_right - bar_left
    name_max_width = bar_left - name_text_left - 12
    base_y = 246
    pitch = 64
    row_height = 48
    ranking_bottom = base_y + (top_n - 1) * pitch + row_height

    def _name_font_for(text: str) -> ImageFont.ImageFont:
        font = name_font_cache.get(text)
        if font is None:
            font = _fit_font_size(measure_draw, text, name_max_width, 28, 15, bold=True)
            name_font_cache[text] = font
        return font

    def make_frame(t: float) -> np.ndarray:
        frame = background.copy()
        draw = ImageDraw.Draw(frame, "RGBA")

        if t >= transition_duration:
            period_index = periods - 1
            value_alpha = 1.0
            rank_alpha = 1.0
        else:
            period_index = min(int(t / seconds_per_period), periods - 1)
            local_time = (t - period_index * seconds_per_period) / seconds_per_period
            value_alpha = min(max(local_time, 0.0), 1.0)
            rank_alpha = _ease_in_out(value_alpha)

        prev = snapshots[period_index]
        nxt = snapshots[period_index + 1]
        axis_cap = axis_scales[period_index][0] + (axis_scales[period_index + 1][0] - axis_scales[period_index][0]) * value_alpha
        tick_step = axis_scales[period_index][1] + (axis_scales[period_index + 1][1] - axis_scales[period_index][1]) * value_alpha

        interpolated = _interpolate(prev, nxt, value_alpha)
        states_by_ticker = {state.ticker: state for state in interpolated}
        priority = priorities[period_index]
        previous_rank = _rank(prev.states, top_n, priority)
        target_rank = _rank(nxt.states, top_n, priority)
        visible_tickers = sorted(set(previous_rank) | set(target_rank))

        draw.rounded_rectangle(header_box, radius=34, fill=(3, 16, 30, 220), outline=(255, 255, 255, 24), width=2)
        draw.text((68, 55), TITLE, font=title_font, fill="#f6fbff")
        draw.text((70, 119), SUBTITLE, font=subtitle_font, fill="#83d9d0")

        draw.rounded_rectangle(insight_box, radius=26, fill=(9, 43, 58, 230), outline=(102, 224, 210, 54), width=2)
        draw.text((insight_box[0] + 20, insight_box[1] + 16), "DAILY ADJ CLOSE", font=insight_font, fill="#f4d39a")
        draw.text((insight_box[0] + 20, insight_box[1] + 58), "RAW VALUES, SHOWN PER TICKER", font=insight_font, fill="#eaf7f5")

        draw.rounded_rectangle(year_box, radius=30, fill=(244, 194, 107, 255), outline=(255, 235, 184, 180), width=2)
        _center_text(draw, year_box, nxt.label, year_font, "#10273a")

        draw.text((name_left, 195), LEFT_HEADER_LABEL, font=label_font, fill=(177, 210, 219, 205))
        draw.text((bar_left + 18, 195), RIGHT_HEADER_LABEL, font=label_font, fill=(177, 210, 219, 205))

        tick_count = max(1, int(math.floor(axis_cap / max(tick_step, 1.0))))
        for tick_index in range(tick_count + 1):
            value = min(axis_cap, tick_index * tick_step)
            x = bar_left + int((value / axis_cap) * bar_max_width)
            draw.line((x, 224, x, ranking_bottom + 12), fill=(1, 9, 18, 82), width=2 if tick_index else 3)
            tick_text = _format_market_cap(value) if tick_index else "$0"
            bbox = draw.textbbox((0, 0), tick_text, font=tick_font)
            draw.text((x - (bbox[2] - bbox[0]) // 2, 194), tick_text, font=tick_font, fill=(3, 15, 27, 165))

        for rank_index in range(top_n):
            y0 = base_y + rank_index * pitch
            fill = (244, 194, 107, 255) if rank_index == 0 else (20, 73, 87, 245)
            text_fill = "#10273a" if rank_index == 0 else "#edf8f7"
            draw.rounded_rectangle((rank_left, y0, rank_left + 52, y0 + row_height), radius=17, fill=fill)
            _center_text(draw, (rank_left, y0, rank_left + 52, y0 + row_height), str(rank_index + 1), rank_font, text_fill)

        render_items: list[tuple[int, float, CompanyState, int]] = []
        for ticker in visible_tickers:
            state = states_by_ticker.get(ticker)
            if state is None:
                continue
            previous_index = previous_rank.get(ticker, top_n + 1)
            target_index = target_rank.get(ticker, top_n + 1)
            entering = previous_index > top_n and target_index <= top_n
            effective_previous = float(top_n + 1.8) if entering else float(previous_index)
            movement_alpha = rank_alpha
            places_moved = abs(float(target_index) - effective_previous)
            if places_moved > 0:
                delay = min(0.08, max(0.0, (places_moved - 1.0) * 0.015))
                movement_alpha = _ease_in_out(_phase_delay(movement_alpha, delay, max(0.82, 0.98 - delay)))
            if entering:
                movement_alpha = _ease_in_out(_phase_delay(movement_alpha, 0.02, 0.96))
            y_index = _continuous_rank_position(effective_previous, float(target_index), movement_alpha)
            y = base_y + y_index * pitch
            bar_width = max(8, int((state.index_value / axis_cap) * bar_max_width))
            moving_up = 1 if target_index < previous_index else 0
            render_items.append((moving_up, y, state, bar_width))

        render_items.sort(key=lambda item: (item[0], item[1]))
        for _, y, state, bar_width in render_items:
            y0 = int(y)
            y1 = y0 + row_height
            if y1 < base_y - pitch or y0 > ranking_bottom + pitch:
                continue

            color = colors[state.ticker]
            highlight = _mix_rgb(color, (255, 255, 255), 0.28)
            shadow = _mix_rgb(color, (0, 0, 0), 0.24)

            draw.rounded_rectangle((108, y0 - 3, WIDTH - 64, y1 + 3), radius=19, fill=(3, 15, 27, 62))

            logo_box = (name_left, y0 + 6, name_left + logo_size, y0 + 6 + logo_size)
            draw.rounded_rectangle(logo_box, radius=12, fill=(255, 255, 255, 22), outline=(255, 255, 255, 34), width=1)
            logo_key = _logo_slug(state.ticker)
            logo_image = logo_cache.get(logo_key)
            if logo_image is not None:
                frame.paste(logo_image, (name_left, y0 + 6), logo_image)
            else:
                initials = _logo_initials(state.display_name if state.display_name else state.ticker)
                initials_bbox = draw.textbbox((0, 0), initials, font=logo_font)
                initials_x = name_left + (logo_size - (initials_bbox[2] - initials_bbox[0])) // 2 - initials_bbox[0]
                initials_y = y0 + 6 + (logo_size - (initials_bbox[3] - initials_bbox[1])) // 2 - initials_bbox[1]
                draw.text((initials_x, initials_y), initials, font=logo_font, fill="#f4f8fb")

            name_text = state.display_name
            name_font_state = _name_font_for(name_text)
            name_bbox = draw.textbbox((0, 0), name_text, font=name_font_state)
            name_y = y0 + (row_height - (name_bbox[3] - name_bbox[1])) // 2 - name_bbox[1]
            draw.text((name_text_left, name_y), name_text, font=name_font_state, fill="#f1f7f8")

            draw.rounded_rectangle((bar_left + 6, y0 + 6, bar_left + bar_width + 6, y1 + 6), radius=20, fill=(0, 0, 0, 76))
            draw.rounded_rectangle(
                (bar_left, y0, bar_left + bar_width, y1),
                radius=20,
                fill=color,
                outline=highlight,
                width=2,
            )
            if bar_width > 42:
                draw.rounded_rectangle(
                    (bar_left + 9, y0 + 8, bar_left + max(30, int(bar_width * 0.64)), y0 + 17),
                    radius=6,
                    fill=(*highlight, 58),
                )
                draw.line(
                    (bar_left + 18, y1 - 8, bar_left + max(28, int(bar_width * 0.72)), y1 - 8),
                    fill=(*shadow, 92),
                    width=3,
                )

            value_text = _format_market_cap(state.index_value)
            value_bbox = draw.textbbox((0, 0), value_text, font=value_font)
            value_x = min(bar_left + bar_width + 14, WIDTH - 38 - (value_bbox[2] - value_bbox[0]))
            value_y = y0 + (row_height - (value_bbox[3] - value_bbox[1])) // 2 - value_bbox[1]
            draw.text((value_x, value_y), value_text, font=value_font, fill="#f7fbfc")

        footer_bbox = draw.textbbox((0, 0), FOOTER, font=footer_font)
        draw.text(
            ((WIDTH - (footer_bbox[2] - footer_bbox[0])) // 2, 1041),
            FOOTER,
            font=footer_font,
            fill=(174, 207, 214, 175),
        )

        return np.array(frame.convert("RGB"))

    clip = VideoClip(make_frame, duration=duration)
    audio_clip = None
    keep_alive: list[object] = []
    if audio_path.exists():
        audio_clip, keep_alive = build_audio_track(audio_path, duration)
        clip = clip.with_audio(audio_clip)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_kwargs = {
        "fps": fps,
        "codec": "libx264",
        "ffmpeg_params": ["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
    }
    if audio_clip is not None:
        write_kwargs["audio_codec"] = "aac"
    else:
        write_kwargs["audio"] = False
    clip.write_videofile(str(output_path), **write_kwargs)

    clip.close()
    if audio_clip is not None:
        audio_clip.close()
    for item in keep_alive:
        close = getattr(item, "close", None)
        if callable(close):
            close()
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a landscape company value bar chart race.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--audio", type=Path, default=DEFAULT_AUDIO)
    parser.add_argument("--logos-dir", type=Path, default=DEFAULT_LOGOS_DIR)
    parser.add_argument("--duration", type=float, default=TOTAL_DURATION)
    parser.add_argument("--final-hold", type=float, default=FINAL_HOLD_DURATION)
    parser.add_argument("--fps", type=int, default=FPS)
    parser.add_argument("--top-n", type=int, default=TOP_N)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = render_video(
        input_csv=args.input,
        output_path=args.output,
        audio_path=args.audio,
        logos_dir=args.logos_dir,
        duration=args.duration,
        final_hold_duration=args.final_hold,
        fps=args.fps,
        top_n=args.top_n,
    )
    print(f"[video_generator] company value race generated -> {output}")


if __name__ == "__main__":
    main()
