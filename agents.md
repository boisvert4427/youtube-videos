# Agents Handbook

Ce fichier sert de source de verite pour les agents qui travaillent dans ce repo.
Il resume l'organisation du projet, les regles de travail et les points sensibles
du template NBA bracket 2025.

## Principes

- Utiliser `rg` pour les recherches.
- Utiliser `apply_patch` pour les modifications de fichiers.
- Ne jamais ecraser des changements utilisateur non relies au sujet.
- Ne pas recreer des generateurs supprimes sans besoin explicite.
- Garder les previews PNG et les MP4 temporaires hors du commit.
- Si un script change, mettre a jour la doc associee dans `README.md` et ici.

## Carte Du Repo

- `scraper/`: collecte et normalisation des donnees.
- `video_generator/`: generateurs MoviePy et Manim.
- `video_tools/`: utilitaires video et extraction de snapshots.
- `video_generator/basketball/`: anciens scripts et wrappers basketball.
- `video_generator/cycling/`: templates cyclisme.
- `video_generator/football/`: templates football.
- `video_generator/tennis/`: templates tennis.
- `history/`: module historique separe avec ses propres scripts et donnees.
- `schemas/`: contrats CSV.
- `data/raw/`: assets locaux et sources brutes. Le dossier est ignore par git.
- `data/processed/`: exports finaux et intermediaires. Le dossier est ignore par git.
- `README.md`: vue d'ensemble utilisateur.
- `SESSION_HANDOFF_2026-04-07.md`: ancien contexte de travail, utile seulement pour reference.

## Conventions Globales

- Preferer des noms de fichiers stables et explicites.
- Regrouper les assets par domaine avant de les ajouter.
- Lorsque le rendu est important, verifier un frame intermediaire avant de lancer le mp4 complet.
- Supprimer les fichiers de preview creats pour le debug avant de terminer.
- Pour les gros changements visuels, garder une hierarchie claire: titre, fond, lignes, logos, scores.
- Le format de sortie est souvent vertical 1080x1920 pour les shorts.

## NBA Playoff Bracket 2025

- Script canonique: `video_generator/generate_nba_playoff_bracket_2025_moviepy.py`
- Sortie par defaut: `data/processed/basketball/nba_playoff_bracket_2025_style.mp4`
- Commande de rendu: `python video_generator/generate_nba_playoff_bracket_2025_moviepy.py`
- Assets locaux attendus:
  - `data/raw/nba_logo.png`
  - `data/raw/nba_team_logos/orlando_magic.png`
  - `data/raw/nba_trophy_photo_alt.png`
  - `~/Downloads/bracket_lines_overlay.svg`

### Objectif Visuel

- Rendu premium, type trailer / broadcast haut de gamme.
- Export encode en definition renforcee pour garder les lignes, les logos et les chiffres nets.
- Les seeds et les affiches suivent le bracket 2025 reel, pas un arbre generique.
- Titre compose avec le vrai logo NBA en couleur.
- `PLAYOFFS 2025` doit rester lisible et ne pas chevaucher le badge.
- Trophee central prefere en image decoupee propre (PNG) sans fond visible, avec glow.
- Les score badges doivent rester jaune vif, etre plus grands, et afficher des chiffres agrandis pour rester lisibles sur mobile.
- Lignes du bracket visibles des le debut, avec le centre du bracket deja raccorde au titre.
- Le fond doit lire comme un split bleu / rouge premium avec halo central et ambience cine.
- Les segments jaunes parasites au centre-bas doivent rester supprimes.
- Les logos doivent etre entiers, non coupes, et plus grands si possible.
- Les seeds doivent rester a l'exterieur des logos pour rester lisibles.
- Les vainqueurs doivent garder des score badges lisibles.
- Les scores des finales de conference sont redessines au-dessus de la finale pour rester visibles jusqu'au bout.
- La timeline est etiree pour que le dernier plan tienne seulement les 5 dernieres secondes.
- Audio de fond par defaut: `data/raw/audio/Midnight_Grip_20260402_0828.mp3`

### Points Sensibles

- Ne pas recreer le vieux doublon `video_generator/basketball/generate_nba_playoff_bracket_2025_moviepy.py`.
- Ne pas reintroduire les traits jaunes centraux dans le bracket.
- Ne pas remettre un logo NBA trop petit ou un titre serre.
- Ne pas couper le trophee ou les logos avec des masques trop agressifs.

## Tennis Grand Slam Age Race Shorts

- Script canonique: `video_generator/generate_federer_nadal_djokovic_age_race_shorts_moviepy.py`
- Sortie par defaut: `data/processed/tennis/federer_nadal_djokovic_age_race_shorts.mp4`
- Format vertical `1080x1920`
- Contenu: Federer 20, Nadal 22, Djokovic 24 avec portraits, barres animees et recap par age
- Audio de fond par defaut: `data/raw/audio/audio.mp3`

## Football Goals By Age Race Shorts

- Script canonique: `video_generator/football/generate_ronaldo_messi_goals_by_age_race_shorts_moviepy.py`
- Sortie par defaut: `data/processed/football/ronaldo_messi_goals_by_age_race_midnight.mp4`
- Format vertical `1080x1920`, duree par defaut `40s`, `60 fps`
- Contenu: Ronaldo vs Messi, buts cumules par age, donnees age `18` a `39`
- Audio de fond par defaut: `data/raw/audio/Midnight_Grip_20260402_0828.mp3`
- Le template reutilise le moteur Tennis age race, avec override des joueurs, couleurs, photos, axe et echelle.

### Points Sensibles

- Garder la barre de depart juste apres les photos, pas au milieu de l'ecran.
- La graduation des ages defile de droite a gauche et ne doit pas apparaitre a gauche de la barre de depart.
- La ligne horizontale doit continuer jusqu'au bord droit et rester derriere les ronds, pas s'arreter au rond.
- Les ronds de gains annuels doivent disparaitre progressivement derriere la barre quand ils la touchent.
- Lors d'un changement de rang, la barre qui monte doit passer entierement au-dessus de l'autre.

## Basketball MVP Ladder Weekly Race Shorts

- Builder CSV: `scraper/basketball/build_nba_kia_mvp_ladder_weekly_csv.py`
- Script canonique: `video_generator/basketball/generate_nba_mvp_ladder_weekly_race_shorts_moviepy.py`
- CSV par defaut: `data/processed/basketball/nba_kia_mvp_ladder_2025_26_weekly.csv`
- Sortie par defaut: `data/processed/basketball/nba_mvp_ladder_cumulative_race_shorts.mp4`
- Format vertical `1080x1920`, duree par defaut `40s`, `60 fps`
- Contenu: top 5 Kia MVP Ladder 2025-26, score cumule semaine par semaine
- Audio de fond par defaut: `data/raw/audio/Midnight_Grip_20260402_0828.mp3`
- Assets joueurs attendus dans `data/raw/mvp_race_assets/`

### Points Sensibles

- Garder seulement les 5 premiers visibles.
- Les noms des joueurs ne doivent pas etre affiches sur les barres.
- Le score est cumule depuis les rangs hebdomadaires, pas une probabilite officielle.
- Les photos et barres doivent rester centrees dans le layout short.
- Lors d'un changement de rang, la barre qui monte doit passer visiblement au-dessus.

## Basketball LeBron Jordan Kobe Points By Age Race Shorts

- Builder CSV: `scraper/basketball/build_nba_points_by_age_csv.py`
- Script canonique: `video_generator/basketball/generate_lebron_jordan_kobe_points_by_age_race_shorts_moviepy.py`
- CSV par defaut: `data/processed/basketball/nba_points_by_age_lebron_jordan_kobe.csv`
- Sortie par defaut: `data/processed/basketball/lebron_jordan_kobe_points_by_age_race_shorts.mp4`
- Format vertical `1080x1920`, duree par defaut `40s`, `60 fps`
- Contenu: LeBron vs Jordan vs Kobe, points NBA de saison reguliere cumules par age
- Audio de fond par defaut: `data/raw/audio/Midnight_Grip_20260402_0828.mp3`
- Assets joueurs attendus dans `data/raw/nba_goat_assets/`
- Le template reutilise le moteur Tennis age race, avec override des joueurs, couleurs, photos, axe et echelle.

### Points Sensibles

- Verifier une frame intermediaire avant le rendu final, surtout autour des ages `32-34`.
- Garder la barre de depart juste apres les photos.
- Les ronds de gains annuels doivent rester lisibles sur la barre d'age qui defile.
- La ligne horizontale doit continuer jusqu'au bord droit et rester derriere les ronds, pas s'arreter au rond.
- Les valeurs dans les barres sont les points cumules; les ronds affichent les points de la saison suivante par age.
- Les donnees viennent de Basketball-Reference et peuvent evoluer pour LeBron tant que sa carriere continue.

## Snapshots Short YouTube

- Script canonique: `video_tools/extract_youtube_short_snapshots.py`
- Source: URL YouTube Shorts ou fichier video local
- Sortie par defaut: `data/processed/youtube_short_snapshots/<short_id>_<timestamp>/`
- Format de sortie: PNG + `manifest.json`
- Dependence URL: `yt-dlp`

## Workflow Conseille

- Lire le code cible et lister les fichiers touches avant d'ecrire.
- Faire les edits par petits blocs coherents.
- Lancer `python -m py_compile` sur le script modifie quand il y a du code.
- Lancer le rendu ou un frame de verification quand le visuel change.
- Verifier `git status` avant commit.
- Commit puis push vers `origin main` sauf instruction contraire.

## Notes Pratiques

- Les assets NBA locaux sont des fichiers de travail, pas des sources de verite versionnees.
- Si un asset change de nom ou de dossier, mettre a jour `README.md` et ce fichier ensemble.
- Si une nouvelle entree de rendu apparait, privilegier un seul moteur canonique par template.
