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
- Pour les timelines historiques, privilegier une legende integree a la frise et garder les textes contenus dans leur carte.
- Pour le module history, la sortie canonique est `history/data/processed/france_kings_timeline_481_1870_300s_60fps_audio.mp4`.
- La timeline des rois et empereurs couvre maintenant `481-1870`.
- Le générateur de carte historique du territoire français est
  `history/france_territory/scripts/generate_france_territory_video.py`.
- La carte du territoire français repose sur des jalons datés fixes et une
  interpolation visuelle entre deux cartes de reference ; ne pas inventer de
  frontieres continues comme si elles etaient exactes d'une annee a l'autre.
- Les surfaces anciennes de cette carte sont des reconstitutions schématiques ;
  conserver les avertissements de méthode dans l'image et dans `SOURCES.md`.

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

## ATP Prize Money Leaders Landscape Race

- Builder CSV: `scraper/tennis/build_atp_prize_money_leaders_csv.py`
- Script canonique: `video_generator/tennis/generate_atp_prize_money_leaders_race_moviepy.py`
- Wrapper: `video_generator/generate_atp_prize_money_leaders_race_moviepy.py`
- CSV par defaut: `data/processed/tennis/atp_prize_money_leaders_current.csv`
- Sortie par defaut: `data/processed/tennis/atp_prize_money_leaders_current.mp4`
- Format paysage `1920x1080`, duree par defaut `40s`, `60 fps`
- Source officielle: `https://www.protennislive.com/posting/ramr/career_prize.pdf`
- Assets locaux attendus:
  - `data/raw/player_photos/`
  - `data/raw/flags/`
  - `data/raw/audio/audio.mp3`

### Objectif Visuel

- Coller au maximum au montage de reference: fond ciel bleu, skyline sombre, colonnes 3D beige, portraits circulaires, drapeaux et montants rouge fonce.
- Conserver une lecture `portrait -> nom -> montant` dans chaque colonne.
- Garder un scroll horizontal lent avec des proportions compactes.
- Le CSV doit rester alimente depuis le PDF officiel ATP et le schema doit suivre si la source change.

## Tour de France Stage Wins Post-War Race

- Builder CSV: `scraper/cycling/build_tour_de_france_postwar_stage_wins_csv.py`
- Script canonique: `video_generator/cycling/generate_tour_de_france_stage_wins_race_moviepy.py`
- Wrapper: `video_generator/generate_tour_de_france_stage_wins_race_moviepy.py`
- CSV par defaut: `data/processed/cycling/tour_de_france/tour_de_france_stage_wins_postwar_1947_2025.csv`
- Sortie par defaut: `data/processed/cycling/tour_de_france/tour_de_france_stage_wins_postwar_1947_2025.mp4`
- Format paysage `1920x1080`, duree par defaut `240s`, `30 fps`
- Contenu: victoires d'etape cumulees du Tour de France depuis 1947
- Les prologues et les contre-la-montre individuels sont comptes, mais les team time trials sont exclus.
- Source de donnees: pages annuelles du Tour de France sur Wikipedia, avec validation sur la page des records du Tour.
- Les portraits et drapeaux locaux sont utilises si disponibles dans `data/raw/player_photos/` et `data/raw/flags/`.

### Points Sensibles

- Verifier que le top 12 final reste lisible avec les grands noms historiques.
- Garder le panneau de droite lisible et compact, avec les leaders annuels.
- Ne pas compter les team time trials comme des victoires d'etape individuelles.

## Federer vs Nadal H2H Score Timeline Shorts

- Script canonique: `video_generator/tennis/generate_federer_vs_nadal_h2h_scores_shorts_moviepy.py`
- Wrapper: `video_generator/generate_federer_vs_nadal_h2h_scores_shorts_moviepy.py`
- Sortie par defaut: `data/processed/tennis/federer_vs_nadal_h2h_scores_shorts.mp4`
- Format vertical `1080x1920`, duree cible `45s`
- Contenu: timeline de matchs Federer/Nadal avec badges de score a la place des trophées
- Scoreboard final de reference: `16-24`
- Le layout doit rester lisible sur mobile avec une timeline aeree et un hook tres court

## Roland-Garros Cards + Timeline Shorts

- Script canonique: `video_generator/tennis/generate_roland_garros_titles_cards_timeline_shorts_moviepy.py`
- Wrapper: `video_generator/generate_roland_garros_titles_cards_timeline_shorts_moviepy.py`
- CSV par defaut: `data/processed/tennis/roland_garros_titles_top12_cards.csv`
- Sortie par defaut: `data/processed/tennis/roland_garros_titles_cards_timeline_shorts.mp4`
- Format vertical `1080x1920`, duree par defaut `48s`, `60 fps`
- Contenu: top 12 vainqueurs Open Era classes par nombre de titres, cards horizontales et panneau timeline des titres du joueur central
- Audio de fond par defaut: `data/raw/audio/audio.mp3`

### Points Sensibles

- Garder le scroll des cards et le panneau timeline synchronises sur la carte centrale.
- La timeline doit rester lisible sur mobile, avec les annees de titres en ordre chronologique.
- Conserver les cartes existantes, le rang et le trio `premier titre / dernier titre / pays` comme repere principal.

## NBA Playoff Wins Without Title Ref-Style Cards

- Script canonique: `video_generator/basketball/generate_nba_playoff_wins_without_title_refstyle_shorts_moviepy.py`
- Wrapper: `video_generator/generate_nba_playoff_wins_without_title_refstyle_shorts_moviepy.py`
- Sortie par defaut: `data/processed/basketball/nba_playoff_wins_without_title_refstyle_shorts.mp4`

Caractere du template:

- Format vertical `1080x1920`
- Scroll horizontal linéaire de cards premium
- Classement des joueurs NBA avec le plus de victoires en playoffs sans titre NBA
- Gros bloc statistique pour les victoires, avec le nom du joueur et l'equipe associee
- Header large et style ref-style proche des autres shorts cards

## Roland-Garros Ref-Style Cards Shorts

- Script canonique: `video_generator/tennis/generate_roland_garros_titles_refstyle_shorts_moviepy.py`
- Wrapper: `video_generator/generate_roland_garros_titles_cards_refstyle_shorts_moviepy.py`
- CSV par defaut: `data/processed/tennis/roland_garros_titles_top12_cards.csv`
- Sortie par defaut: `data/processed/tennis/roland_garros_titles_cards_refstyle_shorts.mp4`
- Format vertical `1080x1920`, duree par defaut `40s`, `60 fps`
- Contenu: top 12 vainqueurs Open Era classes par nombre de titres, avec cartes larges inspirees du debut du montage de reference
- Audio de fond par defaut: `data/raw/audio/audio.mp3`

### Points Sensibles

- Garder le scroll horizontal lent et les cartes quasiment bord a bord, sans espace entre elles.
- Conserver la lisibilite mobile avec la hiérarchie `photo / nom / statistiques`.
- Les photos ne doivent jamais etre ecrasees; si l'asset manque, garder un placeholder propre et premium.
- Les champs `premier titre` et `dernier titre` doivent rester dans le bloc du bas sans chevauchement.
- La zone vide sous les cartes doit rester volontairement aeree pour rappeler l'espace respirable du montage de reference.

## Roland-Garros Cards + Match-Point Preview

- Script canonique: `video_generator/tennis/generate_roland_garros_titles_cards_matchpoint_preview_moviepy.py`
- Wrapper: `video_generator/generate_roland_garros_titles_cards_matchpoint_preview_moviepy.py`
- Sortie par defaut: `data/processed/tennis/roland_garros_titles_cards_matchpoint_preview.mp4`
- Format vertical `1080x1920`, preview cible sur un extrait YouTube telecharge avec `yt-dlp`
- Par defaut: `https://www.youtube.com/watch?v=Fkv_NJLsvAU`, segment `9:08 -> 9:25`, focus `Nadal`
- Montage sequence `5 -> 1`, avec un ecran par rang
- Layout 1/3 carte en haut, 2/3 video en bas, sur toute la largeur
- Le clip de match n'est branche que sur Nadal; les autres ecrans gardent le bas noir

### Points Sensibles

- Ne pas remplir les autres ecrans avec des clips de remplacement tant que le montage n'est pas valide.
- Le clip de match doit etre branche dans l'ecran du joueur cible, pas sur la carte du classement.
- Le rendu doit rester lisible en mobile: cartes nettes en haut, panneau video large en bas.
- Garder le preview court et reproductible, avec cache local dans `tmp/roland_garros_matchpoint_preview/`.

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

## Basketball NBA Titles Franchise Podium 2025

- Script canonique: `video_generator/basketball/generate_nba_championship_podium_short_moviepy.py`
- Wrapper: `video_generator/generate_nba_championship_podium_short_moviepy.py`
- Version Manim premium: `video_generator/basketball/generate_nba_titles_podium_2025_manim.py`
- Wrapper Manim: `video_generator/generate_nba_titles_podium_2025_manim.py`
- Sortie par defaut: `data/processed/basketball/nba_titles_franchise_podium_2025_80s.mp4`
- Sortie Manim par defaut: `data/processed/basketball/nba_titles_franchise_podium_2025_manim.mp4`
- Format vertical `1080x1920`, duree par defaut `80s`
- Contenu: classement statique 2025 des franchises NBA actives par titres, pas une bar chart race.
- Les anciennes villes sont rattachees aux franchises actives quand c'est l'usage NBA: Lakers, Warriors, 76ers, Kings, Hawks, Wizards et Thunder/SuperSonics.
- Les franchises actives sans titre restent visibles avec `0`; les franchises defuntes comme Baltimore Bullets 1948 ne sont pas incluses dans ce classement.
- Le rendu doit garder un effet podium profond: faces laterales, top faces, ombres et grille de sol subtile.
- La version Manim utilise un plate podium transparent haute resolution anime par Manim; le rendu final 1080p peut etre lent, donc verifier d'abord une preview 540x960/12fps.

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

## Snapshots et frames YouTube

- Script canonique: `video_tools/extract_youtube_short_snapshots.py`
- Source: URL YouTube Shorts / watch / youtu.be ou fichier video local
- Sortie par defaut: `data/processed/youtube_short_snapshots/<short_id>_<timestamp>/`
- Format de sortie: PNG + `manifest.json`
- Dependence URL: `yt-dlp`
- Mode complet: `--all-frames` pour extraire chaque frame decodee
- Mode queue: `--tail-seconds 40` pour limiter l'extraction aux 40 dernieres secondes, par exemple avec `--all-frames`

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
