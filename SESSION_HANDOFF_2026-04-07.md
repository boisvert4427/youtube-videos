# Session Handoff - 2026-04-07

## Ce qui a ete fait

- Ajout de plusieurs generateurs Shorts basket/cyclisme/football/tennis bases sur MoviePy.
- Ajout du template Ronaldo vs Messi buts par age:
  - `video_generator/football/generate_ronaldo_messi_goals_by_age_race_shorts_moviepy.py`
  - sortie finale avec audio Midnight: `data/processed/football/ronaldo_messi_goals_by_age_race_midnight.mp4`
  - audio par defaut: `data/raw/audio/Midnight_Grip_20260402_0828.mp3`
- Ajout du template Federer/Nadal/Djokovic Grand Slam par age:
  - `video_generator/tennis/generate_federer_nadal_djokovic_age_race_shorts_moviepy.py`
  - wrappers racine compatibles dans `video_generator/`
- Ajout de l'outil snapshots YouTube Shorts:
  - `video_tools/extract_youtube_short_snapshots.py`
- Ajout d'un generateur MVP Race vertical en Manim :
  - `video_generator/basketball/generate_mvp_race_shorts_manim.py`
  - wrapper `video_generator/generate_mvp_race_shorts_manim.py`
  - `requirements-manim-mvp-race.txt`
- Setup local Manim fonctionnel avec Python 3.12 dans `\.venv-manim`.
- Rendu Manim deja sorti ici :
  - `data/processed/basketball/manim_mvp_race/videos/generate_mvp_race_shorts_manim/1920p30/mvp_race_shorts_manim.mp4`
  - version audio mixee : `.../mvp_race_shorts_manim_audio.mp4`

## Point exact ou on s'arrete

On reprend demain sur **le template MVP Race Manim**.

Problemes remontes par l'utilisateur :

- la video fait trop "cheap"
- il faut **ajouter de vraies images joueurs**
- les blocs de data/statistiques sont **trop serres et se chevauchent**
- il faut rendre l'ensemble plus premium / plus broadcast / plus lisible sur mobile

## Diagnostic deja fait

- Le dossier `data/raw/mvp_race_assets` n'existe pas encore, donc le rendu Manim utilise surtout des fallbacks.
- Les visuels Jokic / SGA / Doncic n'ont pas encore ete branches localement.
- Le layout de la scene stats est trop compact :
  - rows trop serrees
  - cartes joueurs trop petites
  - hiérarchie visuelle pas assez forte

## Priorites pour demain

1. Creer `data/raw/mvp_race_assets`
2. Ajouter 3 vrais visuels :
   - Jokic
   - Shai Gilgeous-Alexander
   - Luka Doncic
3. Refaire le layout Manim :
   - cartes joueurs plus grosses
   - scene stats plus aeree
   - meilleure separation barres / labels / valeurs
   - look plus premium
4. Regenerer la video Manim avec audio

## Commandes utiles

Activer le venv Manim :

```powershell
cd C:\Users\leona\Documents\python\youtube-videos\youtube-videos
.\.venv-manim\Scripts\Activate.ps1
```

Render Manim :

```powershell
python video_generator\generate_mvp_race_shorts_manim.py --render --scene MVPRaceShort --quality h --mix-audio --audio data\raw\audio\audio.mp3
```

## Remarque

Le but demain n'est pas de refaire tout le systeme : la base Manim tourne deja. Il faut surtout **ameliorer les assets et le layout**.

## Note plus recente

Le workflow visuel recent utilise souvent:

- une preview PNG avant rendu complet
- un controle de frames intermediaires
- un rendu final vertical `1080x1920`
- une musique de fond locale dans `data/raw/audio/`

## Note WTA plus recente

- Le workflow WTA rankings race est maintenant base sur l'API officielle WTA uniquement.
- CSV de reference: `data/processed/tennis/wta_rankings_weekly_top12_api_only_2000_2026.csv`
- Video finale: `data/processed/tennis/wta_ranking_points_race_2000_2026_38min_60fps_1080p.mp4`
- Preview PNG: `data/processed/tennis/wta_ranking_points_race_preview.png`
- Le template utilise le top 12 hebdomadaire entre `2000-11-27` et `2026-06-29`.
