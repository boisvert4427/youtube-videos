# Session Handoff - 2026-04-07

## Ce qui a ete fait

- Ajout de plusieurs generateurs Shorts basket/cyclisme/football/tennis bases sur MoviePy.
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
