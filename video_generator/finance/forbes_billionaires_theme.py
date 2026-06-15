from __future__ import annotations

import hashlib
from dataclasses import is_dataclass
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


MONTH_LABELS = {
    1: "JAN",
    2: "FEV",
    3: "MAR",
    4: "AVR",
    5: "MAI",
    6: "JUIN",
    7: "JUIL",
    8: "AOU",
    9: "SEP",
    10: "OCT",
    11: "NOV",
    12: "DEC",
}

COLOR_PALETTE = [
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
    "#0f766e",
    "#1d4ed8",
]


def format_money(value: float) -> str:
    billions = value / 1_000_000_000
    if billions >= 100:
        return f"${billions:,.0f}B"
    text = f"${billions:,.1f}B"
    return text.replace(".0B", "B")


def format_snapshot_label(snapshot: Any) -> str:
    ranking_date = str(getattr(snapshot, "ranking_date", "")).strip()
    year = getattr(snapshot, "year", None)
    try:
        month = int(ranking_date[5:7])
    except (TypeError, ValueError):
        month = 0
    if year is None:
        try:
            year = int(ranking_date[:4])
        except (TypeError, ValueError):
            year = None
    if ranking_date.endswith("-12-31") and year is not None:
        return str(year)
    if month in MONTH_LABELS and year is not None:
        return f"{MONTH_LABELS[month]} {year}"
    if year is not None:
        return str(year)
    return ranking_date or "?"


def make_background(width: int, height: int) -> Image.Image:
    x_values = np.linspace(0, 1, width, dtype=np.float32)
    y_values = np.linspace(0, 1, height, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(x_values, y_values)

    deep = np.array([6, 10, 21], dtype=np.float32)
    navy = np.array([12, 23, 48], dtype=np.float32)
    gold = np.array([201, 165, 82], dtype=np.float32)
    emerald = np.array([22, 101, 89], dtype=np.float32)
    ruby = np.array([120, 18, 18], dtype=np.float32)

    base_mix = np.clip(0.20 * grid_x + 0.72 * grid_y, 0.0, 1.0)
    top_glow = np.exp(-(((grid_x - 0.82) / 0.28) ** 2 + ((grid_y - 0.10) / 0.18) ** 2))
    left_glow = np.exp(-(((grid_x - 0.08) / 0.24) ** 2 + ((grid_y - 0.92) / 0.18) ** 2))
    central_glow = np.exp(-(((grid_x - 0.62) / 0.22) ** 2 + ((grid_y - 0.46) / 0.24) ** 2))

    pixels = np.clip(
        deep[None, None, :] * (1.0 - base_mix[..., None])
        + navy[None, None, :] * (0.78 * base_mix[..., None])
        + gold[None, None, :] * (0.18 * top_glow[..., None])
        + emerald[None, None, :] * (0.10 * left_glow[..., None])
        + ruby[None, None, :] * (0.06 * central_glow[..., None]),
        0,
        255,
    ).astype(np.uint8)
    frame = Image.fromarray(pixels, mode="RGB").convert("RGBA")

    overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")
    draw.rounded_rectangle((26, 24, width - 26, height - 24), radius=42, outline=(255, 255, 255, 18), width=2)
    draw.ellipse((int(width * 0.70), -int(height * 0.10), int(width * 1.05), int(height * 0.28)), outline=(255, 208, 90, 26), width=3)
    draw.ellipse((int(width * -0.08), int(height * 0.64), int(width * 0.34), int(height * 1.08)), outline=(34, 197, 94, 20), width=3)
    draw.line((int(width * 0.10), int(height * 0.14), int(width * 0.90), int(height * 0.06)), fill=(255, 255, 255, 10), width=2)
    draw.line((int(width * 0.18), int(height * 0.86), int(width * 0.92), int(height * 0.72)), fill=(255, 208, 90, 9), width=2)
    draw.rounded_rectangle(
        (int(width * 0.62), int(height * 0.20), int(width * 0.98), int(height * 0.72)),
        radius=44,
        outline=(255, 255, 255, 8),
        width=2,
    )
    overlay = overlay.filter(ImageFilter.GaussianBlur(radius=2))
    frame.alpha_composite(overlay)
    return frame


def build_stable_color_map(snapshots: list[Any]) -> dict[str, str]:
    colors: dict[str, str] = {}
    for snapshot in snapshots:
        for state in getattr(snapshot, "states", []):
            key = str(getattr(state, "country_iso3", "")).strip()
            if not key or key in colors:
                continue
            digest = hashlib.sha1(key.encode("utf-8")).digest()
            colors[key] = COLOR_PALETTE[int.from_bytes(digest[:2], "big") % len(COLOR_PALETTE)]
    return colors
