from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from video_generator.tennis import generate_federer_nadal_djokovic_age_race_shorts_moviepy as base


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "tennis" / "serena_graf_evert_navratilova_age_race_shorts.mp4"
DEFAULT_PREVIEW = PROJECT_ROOT / "tmp_frames" / "serena_graf_evert_navratilova_age_race_preview.png"

TITLE = "SERENA • GRAF • EVERT • NAVRATILOVA"
SUBTITLE = "Grand Slam Titles by Age"
FOOTER = "WTA Legends"

SERENA_COLOR = "#0E6B43"
GRAF_COLOR = "#E8D8AD"
EVERT_COLOR = "#C75B6A"
NAVRATILOVA_COLOR = "#6D4BC9"

TOURNAMENT_MONTHS = {
    "AO": 1,
    "RG": 6,
    "WIM": 7,
    "USO": 9,
}

TOURNAMENT_ORDER = {
    "AO": 0,
    "RG": 1,
    "WIM": 2,
    "USO": 3,
}


@dataclass(frozen=True)
class TitleEvent:
    year: int
    tournament: str


PLAYER_TITLE_EVENTS: dict[str, tuple[TitleEvent, ...]] = {
    "Serena Williams": (
        TitleEvent(1999, "USO"),
        TitleEvent(2002, "RG"),
        TitleEvent(2002, "WIM"),
        TitleEvent(2002, "USO"),
        TitleEvent(2003, "AO"),
        TitleEvent(2005, "AO"),
        TitleEvent(2005, "WIM"),
        TitleEvent(2007, "AO"),
        TitleEvent(2007, "WIM"),
        TitleEvent(2008, "USO"),
        TitleEvent(2009, "AO"),
        TitleEvent(2010, "AO"),
        TitleEvent(2010, "WIM"),
        TitleEvent(2012, "WIM"),
        TitleEvent(2012, "USO"),
        TitleEvent(2013, "AO"),
        TitleEvent(2013, "RG"),
        TitleEvent(2014, "USO"),
        TitleEvent(2015, "AO"),
        TitleEvent(2015, "RG"),
        TitleEvent(2015, "WIM"),
        TitleEvent(2016, "WIM"),
        TitleEvent(2017, "AO"),
    ),
    "Steffi Graf": (
        TitleEvent(1987, "RG"),
        TitleEvent(1988, "AO"),
        TitleEvent(1988, "RG"),
        TitleEvent(1988, "WIM"),
        TitleEvent(1988, "USO"),
        TitleEvent(1989, "AO"),
        TitleEvent(1989, "WIM"),
        TitleEvent(1989, "USO"),
        TitleEvent(1990, "RG"),
        TitleEvent(1990, "WIM"),
        TitleEvent(1990, "USO"),
        TitleEvent(1991, "AO"),
        TitleEvent(1991, "WIM"),
        TitleEvent(1991, "USO"),
        TitleEvent(1992, "RG"),
        TitleEvent(1992, "USO"),
        TitleEvent(1993, "WIM"),
        TitleEvent(1994, "USO"),
        TitleEvent(1995, "AO"),
        TitleEvent(1995, "RG"),
        TitleEvent(1995, "WIM"),
        TitleEvent(1996, "AO"),
    ),
    "Chris Evert": (
        TitleEvent(1974, "RG"),
        TitleEvent(1974, "WIM"),
        TitleEvent(1975, "RG"),
        TitleEvent(1975, "USO"),
        TitleEvent(1976, "WIM"),
        TitleEvent(1976, "USO"),
        TitleEvent(1977, "USO"),
        TitleEvent(1978, "USO"),
        TitleEvent(1979, "RG"),
        TitleEvent(1980, "RG"),
        TitleEvent(1980, "USO"),
        TitleEvent(1981, "WIM"),
        TitleEvent(1982, "AO"),
        TitleEvent(1982, "USO"),
        TitleEvent(1983, "RG"),
        TitleEvent(1984, "AO"),
        TitleEvent(1985, "RG"),
        TitleEvent(1986, "RG"),
    ),
    "Martina Navratilova": (
        TitleEvent(1978, "WIM"),
        TitleEvent(1979, "WIM"),
        TitleEvent(1981, "AO"),
        TitleEvent(1982, "RG"),
        TitleEvent(1982, "WIM"),
        TitleEvent(1983, "WIM"),
        TitleEvent(1983, "USO"),
        TitleEvent(1984, "WIM"),
        TitleEvent(1984, "USO"),
        TitleEvent(1985, "WIM"),
        TitleEvent(1985, "USO"),
        TitleEvent(1986, "WIM"),
        TitleEvent(1986, "USO"),
        TitleEvent(1987, "WIM"),
        TitleEvent(1987, "USO"),
        TitleEvent(1988, "WIM"),
        TitleEvent(1989, "WIM"),
        TitleEvent(1990, "WIM"),
    ),
}

BIRTH_DATES = {
    "Serena Williams": (1981, 9),
    "Steffi Graf": (1969, 6),
    "Chris Evert": (1954, 12),
    "Martina Navratilova": (1956, 10),
}

PLAYER_COLORS = {
    "Serena Williams": SERENA_COLOR,
    "Steffi Graf": GRAF_COLOR,
    "Chris Evert": EVERT_COLOR,
    "Martina Navratilova": NAVRATILOVA_COLOR,
}

PLAYER_PHOTOS = {
    "Serena Williams": "serena_williams.jpg",
    "Steffi Graf": "steffi_graf.jpg",
    "Chris Evert": "chris_evert.jpg",
    "Martina Navratilova": "martina_navratilova.jpg",
}


def _title_age(event: TitleEvent, birth_year: int, birth_month: int) -> int:
    age = event.year - birth_year
    if TOURNAMENT_MONTHS[event.tournament] < birth_month:
        age -= 1
    return max(base.AGE_MIN, min(base.AGE_MAX, age))


def _build_counts_and_segments(name: str) -> tuple[dict[int, int], dict[int, list[str]]]:
    birth_year, birth_month = BIRTH_DATES[name]
    counts = {age: 0 for age in range(base.AGE_MIN, base.AGE_MAX + 1)}
    segments = {age: [] for age in range(base.AGE_MIN, base.AGE_MAX)}
    running = 0

    for event in sorted(PLAYER_TITLE_EVENTS[name], key=lambda item: (item.year, TOURNAMENT_ORDER[item.tournament])):
        age = _title_age(event, birth_year, birth_month)
        running += 1
        for current_age in range(age, base.AGE_MAX + 1):
            counts[current_age] = running
        if age < base.AGE_MAX:
            segments.setdefault(age, []).append(event.tournament)

    return counts, segments


def _install_template_overrides() -> None:
    players = []
    segments: dict[str, dict[int, list[str]]] = {}

    for name in ("Serena Williams", "Steffi Graf", "Chris Evert", "Martina Navratilova"):
        counts, player_segments = _build_counts_and_segments(name)
        players.append(
            base.Player(
                name=name,
                short=name.split()[0].upper() if name != "Martina Navratilova" else "NAVRATILOVA",
                photo_name=PLAYER_PHOTOS[name],
                color=PLAYER_COLORS[name],
                counts=counts,
            )
        )
        segments[name] = player_segments

    base.DEFAULT_OUTPUT = DEFAULT_OUTPUT
    base.DEFAULT_PREVIEW = DEFAULT_PREVIEW
    base.TITLE = TITLE
    base.SUBTITLE = SUBTITLE
    base.FOOTER = FOOTER
    base.VALUE_SCALE_MAX = 24.0
    base.PLAYERS = players
    base.PLAYER_TIE_ORDER = {player.name: index for index, player in enumerate(players)}
    base.SEGMENT_SLAMS = segments


def main() -> None:
    _install_template_overrides()
    base.main()


if __name__ == "__main__":
    main()
