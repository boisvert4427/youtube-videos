# ATP Prize Money Leaders CSV

CSV contract for the ATP prize money leaders dataset used by the tennis prize-money video generator.

## Source

- Official ATP prize money leaders PDF from `protennislive.com`
- Parsed from the current rankings table on page 1

## Columns

- `ranking_date`: ISO date of the ATP ranking snapshot, for example `2026-05-25`
- `tour`: always `ATP`
- `rank`: official ATP rank in the prize-money leaderboard
- `player_name`: player name normalized to `Given Surname`
- `country_code`: ISO 3166-1 alpha-2 country code used for flags
- `career_usd`: career prize money in US dollars, as an integer
- `ytd_usd`: prize money won in the current year to date, as an integer
- `singles_usd`: current year singles prize money, as an integer
- `doubles_usd`: current year doubles prize money, as an integer

## Intended Use

- The generator sorts the rows by `career_usd` and renders the top leaders in a landscape side-scrolling layout.
- Keep the `country_code` field populated for the players that appear in the default top-10 video.
