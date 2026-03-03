# atp_ranking_timeseries_v1

Colonnes obligatoires:

1. `ranking_date` (YYYY-MM-DD)
2. `player_name` (nom complet)
3. `country_code` (ISO alpha-3, ex: FRA, SRB, ESP)
4. `points` (entier >= 0)

Colonnes optionnelles:

1. `rank` (entier, utile pour QA)

Notes:

- Une ligne = un joueur pour une date de classement ATP.
- Le generateur video trie automatiquement par `points` a chaque frame.
