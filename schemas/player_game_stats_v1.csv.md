# player_game_stats_v1

Colonnes obligatoires:

1. `game_date` (YYYY-MM-DD)
2. `league` (ex: nba)
3. `season` (ex: 2025-26)
4. `team`
5. `opponent`
6. `player_name`
7. `minutes` (entier)
8. `points` (entier)
9. `rebounds` (entier)
10. `assists` (entier)

Notes:

- Ce schema est le contrat entre `scraper` et `video_generator`.
- Toute nouvelle colonne doit etre ajoutee sans casser les colonnes existantes.
