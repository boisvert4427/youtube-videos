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
4. Generer une video timeline verticale ATP (template cartes par annee):
   - `python video_generator/generate_atp_vertical_timeline.py`
5. Generer une video timeline verticale ATP en MoviePy (rendu template):
   - `python video_generator/generate_atp_vertical_timeline_moviepy.py`

Regles fixes du template Indian Wells (toujours):

- Duree totale: `90s`
- Freeze debut: `5s` sur la premiere image
- Freeze fin: `15s` sur la derniere image
- Audio de fond: `data/raw/audio/audio.mp3`
- Fade out audio final: `10s`
- Format sortie: `1920x1080` MP4

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

## Nouveau template: timeline verticale

Le fichier `data/processed/atp_vertical_timeline_v1.sample.csv` contient un exemple
de timeline par annee (carte joueur + badge + liste de resultats).

Commande:

- `python video_generator/generate_atp_vertical_timeline.py`
