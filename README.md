# youtube-videos

Projet structure en 2 modules:

- `scraper/`: collecte, normalisation, export CSV.
- `video_generator/`: lecture du CSV et preparation des donnees video.

## Structure

- `data/raw/`: donnees brutes temporaires.
- `data/processed/`: CSV normalises consommes par la video.
- `schemas/`: contrat de colonnes CSV entre scraper et generateur video.

## Utilisation

1. Generer un CSV de stats:
   - `python scraper/run_scraper.py`
2. Verifier la lecture cote video:
   - `python video_generator/load_csv.py`
3. Generer une video bar chart race ATP (a partir d'un CSV timeseries):
   - `python video_generator/generate_atp_barchart_race.py`

Le pipeline actuel utilise un jeu de donnees de demonstration local pour etablir
le contrat. Ensuite, tu pourras remplacer la source par une API officielle.

## ATP et dates

Le fichier `data/processed/atp_ranking_timeseries_v1.sample.csv` contient un exemple
avec plusieurs dates de classement.

Pour produire `data/processed/atp_ranking_timeseries_v1.csv` depuis des snapshots
bruts, depose des CSV dans `data/raw/atp_rankings/` avec les colonnes:

- `ranking_date`
- `player_name`
- `country_code`
- `points`

Puis lance:

- `python scraper/build_atp_timeseries_csv.py`
