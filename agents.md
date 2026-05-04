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
  - `data/raw/nba_trophy_photo.jpg`
  - `~/Downloads/bracket_lines_overlay.svg`

### Objectif Visuel

- Rendu premium, type trailer / broadcast haut de gamme.
- Titre compose avec le vrai logo NBA.
- `PLAYOFFS 2025` doit rester lisible et ne pas chevaucher le badge.
- Trophee central prefere en photo reelle, pas en icone plate.
- Lignes du bracket animees a partir du trace SVG.
- Les segments jaunes parasites au centre-bas doivent rester supprimes.
- Les logos doivent etre entiers, non coupes, et plus grands si possible.
- Les seeds doivent rester a l'exterieur des logos pour rester lisibles.
- Les vainqueurs doivent garder des score badges lisibles.

### Points Sensibles

- Ne pas recreer le vieux doublon `video_generator/basketball/generate_nba_playoff_bracket_2025_moviepy.py`.
- Ne pas reintroduire les traits jaunes centraux dans le bracket.
- Ne pas remettre un logo NBA trop petit ou un titre serre.
- Ne pas couper le trophee ou les logos avec des masques trop agressifs.

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
