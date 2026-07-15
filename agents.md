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

## Tennis WTA Legends Age Race Shorts

- Script canonique: `video_generator/tennis/generate_serena_graf_evert_navratilova_age_race_shorts_moviepy.py`
- Wrapper: `video_generator/generate_serena_graf_evert_navratilova_age_race_shorts_moviepy.py`
- Sortie par defaut: `data/processed/tennis/serena_graf_evert_navratilova_age_race_shorts.mp4`
- Format vertical `1080x1920`
- Contenu: Serena Williams, Steffi Graf, Chris Evert et Martina Navratilova avec portraits, barres animees et recap par age
- Audio de fond par defaut: `data/raw/audio/audio.mp3`
- Le template reutilise le moteur Tennis age race, avec override des joueurs, couleurs, photos, axe et echelle.

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

## WTA Rankings Race

- Builder API: `scraper/tennis/build_wta_rankings_api_timeseries.py`
- Script canonique: `video_generator/tennis/generate_wta_ranking_points_race_moviepy.py`
- Wrapper: `video_generator/generate_wta_ranking_points_race_moviepy.py`
- CSV par defaut: `data/processed/tennis/wta_rankings_weekly_top12_api_only_2000_2026.csv`
- Sortie par defaut: `data/processed/tennis/wta_ranking_points_race_2000_2026_38min_60fps_1080p.mp4`
- Preview PNG: `data/processed/tennis/wta_ranking_points_race_preview.png`
- Format paysage `1920x1080`, duree finale `2280s`, `60 fps`, top 12
- Source de donnees: API officielle WTA uniquement

### Objectif Visuel

- Conserver le style broadcast premium, avec la meme hierarchie que les autres bar chart races tennis.
- Ne pas revenir au vieux CSV mixte Kaggle/API.
- Le top 12 hebdomadaire doit rester lisible de bout en bout entre `2000-11-27` et `2026-06-29`.
- Garder le detourage et l'ecart entre photos joueurs, drapeaux et barres.

## Tour de France Stage Wins Post-War Race

- Builder CSV: `scraper/cycling/build_tour_de_france_postwar_stage_wins_csv.py`
- Script canonique: `video_generator/cycling/generate_tour_de_france_stage_wins_race_moviepy.py`
- Wrapper: `video_generator/generate_tour_de_france_stage_wins_race_moviepy.py`
- CSV par defaut: `data/processed/cycling/tour_de_france/tour_de_france_stage_wins_postwar_1947_2025.csv`
- Sortie par defaut: `data/processed/cycling/tour_de_france/tour_de_france_stage_wins_postwar_1947_2025.mp4`
- Preview rapide: `python video_generator/cycling/generate_tour_de_france_stage_wins_race_moviepy.py --preview --preview-time 120`
- Preview PNG: `data/processed/cycling/tour_de_france/tour_de_france_stage_wins_preview.png`
- Format paysage `1920x1080`, duree par defaut `240s`, `30 fps`
- Contenu: victoires d'etape cumulees du Tour de France depuis 1947
- Les prologues et les contre-la-montre individuels sont comptes, mais les team time trials sont exclus.
- Source de donnees: pages annuelles du Tour de France sur Wikipedia, avec validation sur la page des records du Tour.
- Les portraits et drapeaux locaux sont utilises si disponibles dans `data/raw/player_photos/` et `data/raw/flags/`.
- Faire d'abord une preview avant le rendu final pour verifier le panneau de droite et le placement de l'annee.

### Points Sensibles

- Verifier que le top 12 final reste lisible avec les grands noms historiques.
- Garder le panneau de droite lisible et compact, avec les leaders annuels.
- Ne pas compter les team time trials comme des victoires d'etape individuelles.

## Tour de France Stage Wins Race Shorts

- Script canonique: `video_generator/cycling/generate_tour_de_france_stage_wins_race_shorts_moviepy.py`
- Wrapper: `video_generator/generate_tour_de_france_stage_wins_race_shorts_moviepy.py`
- Sortie par defaut: `data/processed/cycling/tour_de_france/tour_de_france_stage_wins_postwar_1947_2025_shorts.mp4`
- Format vertical `1080x1920`, duree par defaut `60s`, `60 fps`
- Contenu: top 12 des victoires d'etape cumulees depuis 1947, avec portraits, drapeaux et resume annuel

### Points Sensibles

- Chaque coureur doit garder la meme couleur pendant toute la race.
- L'annee doit rester centree dans son badge.
- Les graduations doivent passer derriere les barres.
- Lors d'un changement de rang, la barre qui monte doit passer visiblement au-dessus.

## Tour de France Through The Years Landscape

- Builder CSV: `scraper/cycling/build_tour_de_france_through_the_years_csv.py`
- Wrapper CSV: `scraper/build_tour_de_france_through_the_years_csv.py`
- Script canonique: `video_generator/cycling/generate_tour_de_france_through_the_years_moviepy.py`
- Wrapper: `video_generator/generate_tour_de_france_through_the_years_moviepy.py`
- Schema: `schemas/cycling/tour_de_france_through_the_years_v1.csv.md`
- CSV par defaut: `data/processed/cycling/tour_de_france/tour_de_france_through_the_years_1947_2025.csv`
- Preview PNG: `data/processed/cycling/tour_de_france/tour_de_france_through_the_years_preview.png`
- Sortie par defaut: `data/processed/cycling/tour_de_france/tour_de_france_through_the_years_1947_2025_4min_60fps.mp4`
- Format paysage `1920x1080`, duree par defaut `240s`, `60 fps`
- Contenu: timeline cartes annee par annee depuis 1947, avec vainqueur, podium et maillots.
- Assets maillots obligatoires: `data/raw/cycling/tour_de_france_jerseys/yellow.png`, `green.png`, `polka.png`.

### Points Sensibles

- Les maillots en bas des cartes doivent utiliser les PNG detoures fournis par l'utilisateur, pas une recreation vectorielle.
- Ne pas remettre de filtre jaune plein sur les photos; garder les portraits lisibles avec seulement le fade sombre bas.
- Valider la preview PNG avant tout rendu final long.
- Les noms sous les maillots doivent rester visibles; si les maillots sont agrandis, ajuster aussi la position du texte.
- Les assets locaux `data/raw/` et les rendus `data/processed/` sont ignores par git; documenter leur presence si le rendu en depend.

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

## World Population Race

- Builder canonique: `scraper/demography/build_world_population_timeseries_csv.py`
- Generateur canonique: `video_generator/demography/generate_world_population_race_moviepy.py`
- CSV par defaut: `data/processed/demography/world_population/world_population_1960_2024.csv`
- Sortie par defaut: `data/processed/demography/world_population/world_population_race_1960_2024_3min.mp4`
- Format paysage `1920x1080`, duree `180s`, `60 fps`, top 12.
- Source unique `1960-2024`: API officielle World Bank `SP.POP.TOTL`.

### Points Sensibles

- Exclure tous les agregats World Bank avant le classement.
- Chaque pays garde la meme couleur pendant toute la race.
- Les populations et les longueurs de barres doivent progresser lineairement entre deux annees.
- Les valeurs restent en millions avec une decimale, meme au-dessus d'un milliard.
- L'annee reste centree dans son badge.
- Les graduations restent derriere les barres.
- La barre qui monte doit etre dessinee au-dessus lors d'un changement de rang.
- Verifier une preview courte avant le rendu final de trois minutes.

## World Population Race Shorts

- Generateur canonique: `video_generator/demography/generate_world_population_race_shorts_moviepy.py`
- Wrapper: `video_generator/generate_world_population_race_shorts_moviepy.py`
- Sortie par defaut: `data/processed/demography/world_population/world_population_race_1960_2024_shorts.mp4`
- Format vertical `1080x1920`, duree `60s`, `60 fps`, top 12.
- Source unique: API officielle World Bank `SP.POP.TOTL`.

### Points Sensibles

- Garder les pays et valeurs lisibles meme quand les barres sont courtes.
- Conserver l'interpolation lineaire des populations et le format au dixieme de million.
- Garder les graduations derriere les barres et la barre montante au premier plan.
- L'annee doit rester centree dans son badge.

## France Female First Names Race

- Builder canonique: `scraper/demography/build_france_female_first_names_timeseries_csv.py`
- Wrapper builder: `scraper/build_france_female_first_names_timeseries_csv.py`
- Generateur canonique: `video_generator/demography/generate_france_female_first_names_race_moviepy.py`
- Wrapper video: `video_generator/generate_france_female_first_names_race_moviepy.py`
- CSV par defaut: `data/processed/demography/france_female_first_names/france_female_first_names_1900_2024.csv`
- Sortie par defaut: `data/processed/demography/france_female_first_names/france_female_first_names_race_1900_2024_3min.mp4`
- Miniature: `data/processed/demography/france_female_first_names/france_female_first_names_thumbnail_1900_2024.png`
- Format paysage `1920x1080`, duree `180s`, `60 fps`, top 12.
- Serie annuelle officielle Insee `1900-2024`, prenoms feminins, France entiere.

### Points Sensibles

- Les valeurs representent les naissances de l'annee et ne sont pas cumulatives.
- Exclure les prenoms rares regroupes sous un libelle technique.
- Garder une couleur fixe par prenom et interpoler lineairement les valeurs entre deux annees.
- Utiliser un axe dynamique afin de conserver des barres lisibles sur toute la periode.
- Garder l'annee centree, les graduations derriere les barres et la barre montante au premier plan.
- Ne pas afficher la source dans la video; conserver le footer neutre.
- Verifier une preview courte avant le rendu final de trois minutes.
- Conserver la miniature en `1280x720`, avec fond blanc casse et contraste fort pour la lecture mobile.

## France Female First Names Race Shorts

- Generateur canonique: `video_generator/demography/generate_france_female_first_names_race_shorts_moviepy.py`
- Wrapper video: `video_generator/generate_france_female_first_names_race_shorts_moviepy.py`
- CSV par defaut: `data/processed/demography/france_female_first_names/france_female_first_names_1900_2024.csv`
- Sortie par defaut: `data/processed/demography/france_female_first_names/france_female_first_names_race_1900_2024_shorts.mp4`
- Format vertical `1080x1920`, duree `100s`, `60 fps`, top 12.

### Points Sensibles

- Garder une intro plus lente pour laisser lire la premiere annee.
- Conserver l'axe dynamique, les graduations derriere les barres et la barre montante au premier plan.
- Garder l'annee centree dans son wrapper.
- Ne pas afficher la source dans la video; conserver le footer neutre.
- Verifier une preview courte avant le rendu final.

## France Male First Names Race

- Builder canonique: `scraper/demography/build_france_male_first_names_timeseries_csv.py`
- Wrapper builder: `scraper/build_france_male_first_names_timeseries_csv.py`
- Generateur canonique: `video_generator/demography/generate_france_male_first_names_race_moviepy.py`
- Wrapper video: `video_generator/generate_france_male_first_names_race_moviepy.py`
- CSV par defaut: `data/processed/demography/france_male_first_names/france_male_first_names_1900_2024.csv`
- Sortie par defaut: `data/processed/demography/france_male_first_names/france_male_first_names_race_1900_2024_3min.mp4`
- Format paysage `1920x1080`, duree `180s`, `60 fps`, top 12.
- Serie annuelle officielle Insee `1900-2024`, prenoms masculins, France entiere.

### Points Sensibles

- Les valeurs representent les naissances de l'annee et ne sont pas cumulatives.
- Exclure les prenoms rares regroupes sous un libelle technique.
- Garder une couleur fixe par prenom et interpoler lineairement les valeurs entre deux annees.
- Utiliser un axe dynamique afin de conserver des barres lisibles sur toute la periode.
- Garder l'annee centree, les graduations derriere les barres et la barre montante au premier plan.
- Ne pas afficher la source dans la video; conserver le footer neutre.
- Verifier une preview courte avant le rendu final de trois minutes.

## France Male First Names Race Shorts

- Generateur canonique: `video_generator/demography/generate_france_male_first_names_race_shorts_moviepy.py`
- Wrapper video: `video_generator/generate_france_male_first_names_race_shorts_moviepy.py`
- CSV par defaut: `data/processed/demography/france_male_first_names/france_male_first_names_1900_2024.csv`
- Sortie par defaut: `data/processed/demography/france_male_first_names/france_male_first_names_race_1900_2024_shorts.mp4`
- Format vertical `1080x1920`, duree `100s`, `60 fps`, top 12.

### Points Sensibles

- Garder une intro plus lente pour laisser lire la premiere annee.
- Conserver l'axe dynamique, les graduations derriere les barres et la barre montante au premier plan.
- Garder l'annee centree dans son wrapper.
- Ne pas afficher la source dans la video; conserver le footer neutre.
- Verifier une preview courte avant le rendu final.

## Browser Market Share Race Shorts

- Builder canonique: `scraper/technology/build_browser_market_share_timeseries_csv.py`
- Wrapper builder: `scraper/build_browser_market_share_timeseries_csv.py`
- Generateur canonique: `video_generator/technology/generate_browser_market_share_race_shorts_moviepy.py`
- Wrapper video: `video_generator/generate_browser_market_share_race_shorts_moviepy.py`
- CSV par defaut: `data/processed/technology/browser_market_share/browser_market_share_1995_2026.csv`
- Sortie par defaut: `data/processed/technology/browser_market_share/browser_market_share_race_1995_2026_shorts.mp4`
- Format vertical `1080x1920`, duree `60s`, `60 fps`, top 8.
- Serie mensuelle harmonisee `1995-2026`; Statcounter prend le relais en janvier 2009.

### Points Sensibles

- Ne pas afficher les sources dans la video; conserver un footer neutre.
- Garder un axe fixe `0-100%` pour comparer toutes les epoques.
- Ne pas faire apparaitre Chrome avant septembre 2008.
- Harmoniser le raccord 2008-2009 pour eviter une rupture visuelle de methodologie.
- Recalculer le leader sur les valeurs interpolees visibles, pas sur le seul mois cible.
- Garder la date centree, les graduations derriere les barres et la barre montante au premier plan.

## Browser Market Share Race

- Generateur canonique: `video_generator/technology/generate_browser_market_share_race_moviepy.py`
- Wrapper: `video_generator/generate_browser_market_share_race_moviepy.py`
- CSV par defaut: `data/processed/technology/browser_market_share/browser_market_share_1995_2026.csv`
- Sortie par defaut: `data/processed/technology/browser_market_share/browser_market_share_race_1995_2026_3min.mp4`
- Format paysage `1920x1080`, duree `180s`, `60 fps`, top 8.

### Points Sensibles

- Reutiliser exactement la serie harmonisee et les couleurs fixes du Short.
- Garder un axe fixe `0-100%` pendant toute la video.
- Ne pas afficher les sources; conserver le footer neutre.
- Afficher les pourcentages au dixieme de point.
- Garder la date centree, les graduations derriere les barres et la barre montante au premier plan.

## Social Media Market Share Race

- Builder canonique: `scraper/technology/build_social_media_market_share_timeseries_csv.py`
- Generateur canonique: `video_generator/technology/generate_social_media_market_share_race_moviepy.py`
- CSV par defaut: `data/processed/technology/social_media_market_share/social_media_market_share_2009_2026.csv`
- Sortie par defaut: `data/processed/technology/social_media_market_share/social_media_market_share_race_2009_2026_top10_3min.mp4`
- Format paysage `1920x1080`, duree `180s`, `60 fps`.
- Source: StatCounter Social Media Stats, parts de trafic / parts de marche, pas nombre d'utilisateurs.

### Points Sensibles

- Ne pas presenter ces donnees comme des utilisateurs actifs.
- Exclure `Other` de la video pour eviter une barre peu informative.
- Verifier le top 10 paysage: les entrants doivent rester dans la zone visible.
- Garder les logos/favicons stables dans `data/raw/technology/social_media_market_share/logos/`.

## Oil Consumption Race

- Generateur canonique: `video_generator/technology/generate_oil_consumption_race_moviepy.py`
- Wrapper: `video_generator/generate_oil_consumption_race_moviepy.py`
- CSV source par defaut: `C:\Users\leona\Downloads\oil-consumption-by-country.csv`
- Background par defaut: `C:\Users\leona\Downloads\ChatGPT Image 5 juil. 2026, 20_51_42.png`
- Sortie par defaut: `data/processed/technology/oil_consumption/oil_consumption_race_1965_2024_3min_uhd.mp4`
- Sortie finale utilisee: `data/processed/technology/oil_consumption/oil_consumption_race_1965_2024_2min.mp4`
- Miniature: `data/processed/technology/oil_consumption/oil_consumption_thumbnail_1280x720.png`
- Format paysage `1920x1080`, rendu final `120s`, `60 fps`, top 12.
- Unite: milliers de barils par jour.

### Points Sensibles

- Exclure tous les agregats OWID/continents; garder uniquement les pays ISO3.
- Garder les noms de pays dans les barres, avec drapeaux de taille fixe.
- Ne pas afficher une colonne `COUNTRY` separee quand `NAME_IN_BAR=True`.
- Garder une marge de `4%` apres le leader pour laisser respirer la valeur numerique.
- Conserver l'effet de depassement: `SNAP_TO_CURRENT_RANKS=False` pour le rendu final.
- `SNAP_TO_CURRENT_RANKS=True` supprime l'animation de croisement et ne doit servir qu'a diagnostiquer les entrants/sortants.

## Video Game Sales Publishers Race

- Builder canonique: `scraper/games/build_video_game_sales_publishers_timeseries_csv.py`
- Wrapper builder: `scraper/build_video_game_sales_publishers_timeseries_csv.py`
- Generateur paysage: `video_generator/games/generate_video_game_sales_publishers_race_moviepy.py`
- Generateur Shorts: `video_generator/games/generate_video_game_sales_publishers_race_shorts_moviepy.py`
- Miniature: `video_generator/games/generate_video_game_sales_publishers_thumbnail.py`
- CSV par defaut: `data/processed/video_game_sales/video_game_sales_publishers_1980_2017.csv`

### Points Sensibles

- Les valeurs sont des ventes cumulees en millions d'unites, pas des pourcentages.
- Le dataset s'arrete a 2017; ne pas etendre artificiellement sans nouvelle source.
- Verifier que les logos Electronic Arts et Konami restent lisibles si les assets changent.

## US Boy Names Race

- Generateur paysage: `video_generator/demography/generate_usa_male_names_race_moviepy.py`
- Miniature: `video_generator/demography/generate_usa_male_names_thumbnail.py`
- CSV source: `data/processed/demography/usa_male_names_top20_by_year_1880_2024.csv`
- CSV normalise: `data/processed/demography/usa_male_names/usa_male_names_1880_2025.csv`
- Sortie par defaut: `data/processed/demography/usa_male_names/usa_male_names_race_1880_2025_3min.mp4`

### Points Sensibles

- Le nom du CSV source indique `2024`, mais les donnees lues vont jusqu'a 2025.
- Les valeurs sont des naissances annuelles, pas un cumul.
- Le wrapper patch les libelles du moteur France first names en anglais; ne pas casser ces overrides.

## UN Population Projection Race

- Generateur canonique: `video_generator/demography/generate_un_population_projection_race_moviepy.py`
- CSV source: `data/processed/demography/population-with-un-projections.csv`
- CSV normalise: `data/processed/demography/un_population_projection/un_population_projection_2026_2100.csv`
- Sortie par defaut: `data/processed/demography/un_population_projection/un_population_projection_race_2026_2100_3min.mp4`
- Format paysage `1920x1080`, duree `180s`, `60 fps`, top 12.

### Points Sensibles

- Demarrer a `2026` pour raconter uniquement les projections futures.
- Utiliser la colonne `Population (Projected)` pour cette video; ne pas melanger avec l'historique 1950-2025.
- Exclure les agregats OWID/continents et garder seulement les pays avec code ISO3.
- Verifier les drapeaux des pays entrants tardifs, notamment `CD` pour DR Congo.

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
