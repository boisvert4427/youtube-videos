# youtube-videos

Projet structure en 2 modules:

- `scraper/`: collecte, normalisation, export CSV.
- `video_generator/`: lecture du CSV et preparation des donnees video.

## Structure

- `data/raw/`: donnees brutes temporaires.
- `data/processed/`: CSV normalises consommes par la video.
- `schemas/`: contrat de colonnes CSV entre scraper et generateur video.

Regle d'organisation:

- A chaque creation de fichier, il faut penser a l'organisation globale du projet.
- Ne pas ajouter un fichier "au plus vite" dans un dossier racine si un sous-dossier metier existe deja ou doit etre cree.
- Ranger les fichiers par domaine (`cycling`, `tennis`, `football`, etc.), puis par competition ou usage si necessaire.
- Privilegier une structure stable des le depart pour eviter l'accumulation de fichiers disperses.
- Si un nouveau bloc fonctionnel apparait, creer les dossiers adaptes avant d'ajouter les fichiers.

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
6. Generer une video ATP en format Shorts:
   - `python video_generator/generate_atp_shorts_timeline_moviepy.py`
7. Generer la video timeline Paris-Nice:
   - `python video_generator/generate_paris_nice_timeline_moviepy.py`

Regles fixes du template Indian Wells (toujours):

- Duree totale: `210s`
- Freeze debut: `5s` sur la premiere image
- Freeze fin: `15s` sur les `4 dernieres cartes` visibles
- Audio de fond: `data/raw/audio/audio.mp3`
- Fade out audio final: `10s`
- Frame rate: `120 fps`
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

## Nouveau template: ATP Shorts

Fichiers principaux:

- Video finale: `python video_generator/tennis/generate_atp_shorts_timeline_moviepy.py`
- Wrapper compatible: `python video_generator/generate_atp_shorts_timeline_moviepy.py`

Regles fixes du template Shorts:

- Format sortie: `1080x1920` MP4
- Duree totale: `75s`
- Freeze debut: `2.5s`
- Freeze fin: `4s`
- Audio source: `data/raw/audio/audio.mp3`
- Fade out audio final: `6s`

## Template Paris-Nice

Fichiers principaux:

- CSV: `data/processed/cycling/paris_nice/paris_nice_timeline_postwar_template.csv`
- Schema: `schemas/cycling/paris_nice_timeline_postwar_v1.csv.md`
- Builder CSV: `python scraper/cycling/build_paris_nice_postwar_csv.py`
- Photos vainqueurs: `python scraper/cycling/download_paris_nice_winner_photos.py`
- Preview: `python video_generator/cycling/generate_paris_nice_timeline_preview.py`
- Video finale: `python video_generator/cycling/generate_paris_nice_timeline_moviepy.py`

Compatibilite:

- Les anciens chemins `scraper/build_paris_nice_postwar_csv.py`, `scraper/download_paris_nice_winner_photos.py`,
  `video_generator/generate_paris_nice_timeline_preview.py` et
  `video_generator/generate_paris_nice_timeline_moviepy.py` restent utilisables comme wrappers.

Regles fixes du template Paris-Nice:

- Duree totale video: `240s`
- Freeze debut: `5s`
- Freeze fin: `15s` sur les 4 dernieres cartes visibles
- Frame rate: `120 fps`
- Format sortie: `1920x1080` MP4
- Audio source: `data/raw/audio/audio.mp3`
- Fade out audio final: `10s`
- Si la musique est plus courte que la video:
  - boucles avec recouvrement
  - fade in `5s` sur chaque nouvelle boucle
  - fade out `5s` sur chaque boucle

Notes data:

- Le CSV est rempli depuis l'API Wikipedia en best-effort.
- Les annees recentes ont plus souvent le top 5 GC complet.
- Les annees anciennes peuvent rester partielles si Wikipedia ne fournit pas de tableau final exploitable.
