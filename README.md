# youtube-videos

Pour les consignes operatives du projet et les conventions a suivre, voir `agents.md`.

Projet structure en 4 modules:

- `scraper/`: collecte, normalisation, export CSV.
- `video_generator/`: lecture du CSV et preparation des donnees video.
- `video_tools/`: utilitaires video comme l'extraction de snapshots.
- `history/`: timelines historiques, datasets, portraits et rendus video.

## Structure

- `data/raw/`: donnees brutes temporaires.
- `data/processed/`: CSV normalises consommes par la video.
- `schemas/`: contrat de colonnes CSV entre scraper et generateur video.
- `history/data/raw/`: donnees et portraits du module histoire.
- `history/data/processed/`: exports CSV, previews et videos du module histoire.
- `history/scripts/`: scripts de build/render du module histoire.

Regle d'organisation:

- A chaque creation de fichier, il faut penser a l'organisation globale du projet.
- Ne pas ajouter un fichier "au plus vite" dans un dossier racine si un sous-dossier metier existe deja ou doit etre cree.
- Ranger les fichiers par domaine (`cycling`, `tennis`, `football`, etc.), puis par competition ou usage si necessaire.
- Privilegier une structure stable des le depart pour eviter l'accumulation de fichiers disperses.
- Si un nouveau bloc fonctionnel apparait, creer les dossiers adaptes avant d'ajouter les fichiers.

## Module History

Le module `history/` est separe de la logique sport, tout en restant dans le meme repo.

Fichiers principaux:

- Dataset regnes: `history/data/raw/france_kings_reigns.csv`
- CSV annuel: `history/data/processed/france_kings_yearly_timeline.csv`
- Portraits: `history/data/raw/portraits/`
- Preview image: `history/data/processed/france_kings_timeline_preview.png`
- Preview video court: `history/data/processed/france_kings_timeline_preview_10s_15fps.mp4`
- Video finale canonique: `history/data/processed/france_kings_timeline_481_1870_300s_60fps_audio.mp4`
- La video de sortie ajoute une legende integree a la frise et garde les textes contenus dans la carte.
- Carte animee du territoire francais, basee sur des jalons fixes et une interpolation visuelle :
  `history/france_territory/scripts/generate_france_territory_video.py`

Commandes:

- `python history/scripts/build_france_kings_yearly_timeline.py`
- `python history/scripts/download_france_kings_portraits.py`
- `python history/scripts/render_france_kings_timeline_preview.py`
- `python history/scripts/generate_france_kings_timeline_video.py --duration 10 --fps 15 --output history/data/processed/france_kings_timeline_preview_10s_15fps.mp4`

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
8. Generer un Short Federer vs Nadal hyper optimise:
   - `python video_generator/generate_federer_vs_nadal_duel_shorts_moviepy.py`
9. Generer un Short Federer vs Nadal retro gaming:
   - `python video_generator/generate_federer_vs_nadal_retro_fight_shorts_moviepy.py`
10. Generer un Short Federer Nadal Djokovic Grand Slam par age:
   - `python video_generator/generate_federer_nadal_djokovic_age_race_shorts_moviepy.py`
11. Generer un Short Ronaldo vs Messi buts par age:
   - `python video_generator/football/generate_ronaldo_messi_goals_by_age_race_shorts_moviepy.py`
12. Generer un Short PSG vs Liverpool 2026 ultra anime:
   - `python video_generator/generate_psg_vs_liverpool_double_confrontation_shorts_moviepy.py`
13. Generer un Short SGA vs Jokic version Bird vs Magic:
   - `python video_generator/generate_sga_vs_jokic_career_shorts_moviepy.py`
14. Generer un Short Rookie of the Year Flagg vs Knueppel:
   - `python video_generator/generate_cooper_flagg_vs_kon_knueppel_roty_shorts_moviepy.py`
15. Generer un Short SGA vs Jokic vs Wemby:
   - `python video_generator/generate_sga_jokic_wemby_mvp_short_moviepy.py`
16. Generer un Short NBA bracket 2025 style TV:
   - `python video_generator/generate_nba_playoff_bracket_2025_moviepy.py`
17. Generer le CSV MVP Ladder NBA 2025-26 weekly:
   - `python scraper/basketball/build_nba_kia_mvp_ladder_weekly_csv.py`
18. Generer un Short NBA MVP Ladder weekly cumule:
   - `python video_generator/basketball/generate_nba_mvp_ladder_weekly_race_shorts_moviepy.py`
19. Generer le CSV LeBron vs Jordan vs Kobe points par age:
   - `python scraper/basketball/build_nba_points_by_age_csv.py`
20. Generer un Short LeBron vs Jordan vs Kobe points par age:
   - `python video_generator/basketball/generate_lebron_jordan_kobe_points_by_age_race_shorts_moviepy.py`
21. Extraire des snapshots ou toutes les frames d'une video YouTube:
   - `python video_tools/extract_youtube_short_snapshots.py "https://www.youtube.com/watch?v=VIDEO_ID" --interval 1`
   - `python video_tools/extract_youtube_short_snapshots.py "https://www.youtube.com/watch?v=VIDEO_ID" --all-frames`
   - `python video_tools/extract_youtube_short_snapshots.py "https://www.youtube.com/watch?v=VIDEO_ID" --all-frames --tail-seconds 40`
22. Generer un Short Federer vs Nadal H2H en scores:
   - `python video_generator/generate_federer_vs_nadal_h2h_scores_shorts_moviepy.py`
23. Generer un Short NBA titres par franchise en podium 2025:
   - `python video_generator/generate_nba_championship_podium_short_moviepy.py`
24. Generer une version Manim plus premium du podium NBA titres 2025:
   - Preview rapide: `python video_generator/generate_nba_titles_podium_2025_manim.py --render --quality l --duration 6 --width 540 --height 960 --fps 12 --output data/processed/basketball/nba_titles_franchise_podium_2025_manim_preview.mp4`
   - Final: `python video_generator/generate_nba_titles_podium_2025_manim.py --render --quality h --duration 80 --width 1080 --height 1920 --fps 30 --mix-audio`
25. Generer un Short Roland-Garros cards + timeline:
   - Preview rapide: `python video_generator/generate_roland_garros_titles_cards_timeline_shorts_moviepy.py --duration 8 --fps 30 --output data/processed/tennis/roland_garros_titles_cards_timeline_preview_8s_30fps.mp4`
   - Final: `python video_generator/generate_roland_garros_titles_cards_timeline_shorts_moviepy.py`
26. Generer un Short Roland-Garros ref-style cards:
   - Preview rapide: `python video_generator/generate_roland_garros_titles_cards_refstyle_shorts_moviepy.py --duration 8 --fps 30 --output data/processed/tennis/roland_garros_titles_cards_refstyle_preview_8s.mp4`
   - Final: `python video_generator/generate_roland_garros_titles_cards_refstyle_shorts_moviepy.py`
27. Generer un preview Roland-Garros cards + vrais points de match:
   - Preview Nadal: `python video_generator/generate_roland_garros_titles_cards_matchpoint_preview_moviepy.py --url "https://www.youtube.com/watch?v=Fkv_NJLsvAU" --start 9:08 --end 9:25 --focus-player Nadal`
   - Sortie par defaut: `data/processed/tennis/roland_garros_titles_cards_matchpoint_preview.mp4`
   - Montage sequence 5 -> 1, avec un ecran par rang.
   - Layout 1/3 carte en haut, 2/3 video en bas, sur toute la largeur.
   - La video de match est active uniquement sur Nadal; les autres ecrans gardent le bas noir.

## Snapshots et frames YouTube

Fichiers principaux:

- Script principal: `video_tools/extract_youtube_short_snapshots.py`
- Sortie par defaut: `data/processed/youtube_short_snapshots/<short_id>_<timestamp>/`

Usage:

- Short YouTube: `python video_tools/extract_youtube_short_snapshots.py "https://www.youtube.com/shorts/VIDEO_ID" --interval 1`
- Fichier local: `python video_tools/extract_youtube_short_snapshots.py data/processed/demo_short.mp4 --interval 1`
- Toutes les frames: `python video_tools/extract_youtube_short_snapshots.py "https://www.youtube.com/watch?v=VIDEO_ID" --all-frames`
- Les 40 dernieres secondes: `python video_tools/extract_youtube_short_snapshots.py "https://www.youtube.com/watch?v=VIDEO_ID" --all-frames --tail-seconds 40`

Resultat:

- PNG nommes `frame_00001.png`, `frame_00002.png`, etc.
- `manifest.json` avec les timestamps et les chemins de sortie
- `yt-dlp` est utilise pour telecharger une URL YouTube si besoin
- `ffprobe` est utilise pour estimer le pas de temps quand on extrait toutes les frames
- `ffmpeg -sseof` est utilise quand on limite l'extraction aux dernieres secondes

## Template NBA Playoff Bracket 2025

Fichiers principaux:

- Script principal: `video_generator/generate_nba_playoff_bracket_2025_moviepy.py`
- Sortie finale par defaut: `data/processed/basketball/nba_playoff_bracket_2025_style.mp4`
- Le moteur canonique est ce script racine. L'ancien doublon NBA bracket dans `video_generator/basketball/` ne doit pas etre recrree.
- Assets locaux attendus:
  - `data/raw/nba_logo.png`
  - `data/raw/nba_team_logos/orlando_magic.png`
  - `data/raw/nba_trophy_photo_alt.png`
  - `~/Downloads/bracket_lines_overlay.svg`

Comment est faite la video:

- Format vertical `1080x1920`
- Export encode avec un leger sharpen et une compression plus douce pour garder le texte net
- Les seeds et affichés suivent l'ordre reel du bracket 2025: 1/8, 4/5, 3/6, 2/7
- Bracket NBA 2025 en 3 paliers visibles: round 1, semis, conference finals
- Les lignes blanches sont visibles des le debut, avec le centre du bracket deja raccorde au titre
- Le titre utilise le vrai logo NBA en couleur
  - Le trophee central utilise la nouvelle image PNG du trophée sans fond visible, avec glow et matte
  - Les score badges sont jaune vif, plus grands, avec des chiffres agrandis pour rester lisibles sur mobile
  - Les logos gagnants glissent de leur seed vers leur place de bracket
  - Les equipes eliminees restent visibles en version assombrie pour conserver le contexte
  - Les seeds restent lisibles hors des logos
  - Les scores des finales de conference sont redessines au-dessus de la finale pour rester lisibles
  - La timeline est etiree pour que le dernier plan tienne seulement les 5 dernieres secondes
  - Le style visuel est inspire d'un fond split bleu / rouge avec halo central et ambience cine

## Nouveau template: NBA playoff wins without title ref-style cards

Fichiers principaux:

- Script principal: `video_generator/basketball/generate_nba_playoff_wins_without_title_refstyle_shorts_moviepy.py`
- Wrapper: `video_generator/generate_nba_playoff_wins_without_title_refstyle_shorts_moviepy.py`
- Sortie finale par defaut: `data/processed/basketball/nba_playoff_wins_without_title_refstyle_shorts.mp4`

Style:

- Format vertical `1080x1920`
- Cards premium en scroll horizontal linéaire
- Classement des joueurs NBA avec le plus de victoires en playoffs sans titre NBA
- Mise en avant du total de victoires, du nom du joueur et de l'equipe associee
- Le rendu garde le style ref-style des autres shorts cards, avec un header large et un bloc de contexte en bas

Commande:

- `python video_generator/generate_nba_playoff_wins_without_title_refstyle_shorts_moviepy.py`
- Audio de fond par defaut: `data/raw/audio/Midnight_Grip_20260402_0828.mp3`

Commandes:

- `python video_generator/generate_nba_playoff_bracket_2025_moviepy.py`

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

## Nouveau template: ATP Prize Money Leaders Race

Fichiers principaux:

- Builder CSV: `python scraper/tennis/build_atp_prize_money_leaders_csv.py`
- Schema CSV: `schemas/tennis/atp_prize_money_leaders_v1.csv.md`
- CSV par defaut: `data/processed/tennis/atp_prize_money_leaders_current.csv`
- Video finale: `python video_generator/generate_atp_prize_money_leaders_race_moviepy.py`
- Script canonique: `video_generator/tennis/generate_atp_prize_money_leaders_race_moviepy.py`
- Source officielle: `https://www.protennislive.com/posting/ramr/career_prize.pdf`

Assets locaux:

- Portraits joueurs: `data/raw/player_photos/`
- Drapeaux: `data/raw/flags/`
- Audio de fond: `data/raw/audio/audio.mp3`

Style:

- Format paysage `1920x1080`, duree par defaut `40s`, `60 fps`
- Classement ATP des gains en carriere, avec un scroll horizontal lent
- Fond ciel bleu, skyline sombre, colonnes 3D beige, portraits circulaires, drapeaux, et valeurs rouge fonce
- Le montage vise a coller au plus pres au style de la video de reference

Points sensibles:

- Garder la hierarchie visuelle `portrait / nom / montant` sur chaque colonne
- Conserver des proportions compactes pour rester proche du rendu original
- Ne pas remplacer le PDF ATP officiel par une source non verifiee sans mettre a jour le builder

## Nouveau template: Roland-Garros cards

Fichiers principaux:

- CSV: `data/processed/tennis/roland_garros_titles_top12_cards.csv`
- Builder CSV: `python scraper/build_roland_garros_titles_cards_csv.py`
- Video finale: `python video_generator/generate_roland_garros_titles_cards_shorts_moviepy.py`

Style:

- Format Shorts vertical avec cartes premium qui defilent horizontalement.
- Donnees par defaut: vainqueurs hommes de l'Open Era classes par nombre de titres.
- Chaque carte affiche le rang, le joueur, le pays, les annees gagnees et le premier/dernier titre.

## Nouveau template: Roland-Garros cards + timeline

Fichiers principaux:

- CSV: `data/processed/tennis/roland_garros_titles_top12_cards.csv`
- Builder CSV: `python scraper/build_roland_garros_titles_cards_csv.py`
- Video finale: `python video_generator/generate_roland_garros_titles_cards_timeline_shorts_moviepy.py`

Style:

- Format Shorts vertical avec cartes premium qui defilent horizontalement.
- Un panneau timeline en bas transforme le joueur central en chronologie lisible de ses titres.
- Les donnees par defaut restent celles des vainqueurs hommes de l'Open Era classes par nombre de titres.
- Chaque carte garde le rang, le joueur, le pays, les annees gagnees et le premier/dernier titre.

## Nouveau template: Roland-Garros ref-style cards

Fichiers principaux:

- CSV: `data/processed/tennis/roland_garros_titles_top12_cards.csv`
- Builder CSV: `python scraper/build_roland_garros_titles_cards_csv.py`
- Video finale: `python video_generator/generate_roland_garros_titles_cards_refstyle_shorts_moviepy.py`

Style:

- Format Shorts vertical inspire du debut de la video de reference.
- Cards larges, scroll horizontal lent, sans espace entre les cartes.
- Chaque card garde une structure en trois blocs: photo en haut, nom au milieu, bandeau statistiques en bas.
- Le rendu vise une lecture mobile tres claire, avec les photos conservees au bon ratio et un espace bas de frame volontairement plus respire.

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

## Tour de France Stage Wins Post-War Race

Fichiers principaux:

- CSV: `data/processed/cycling/tour_de_france/tour_de_france_stage_wins_postwar_1947_2025.csv`
- Builder CSV: `python scraper/cycling/build_tour_de_france_postwar_stage_wins_csv.py`
- Video finale: `python video_generator/cycling/generate_tour_de_france_stage_wins_race_moviepy.py`

Compatibilite:

- Le wrapper `python video_generator/generate_tour_de_france_stage_wins_race_moviepy.py` reste utilisable.

Comment est faite la video:

- Format paysage `1920x1080`, duree par defaut `240s`, `30 fps`.
- Classement cumule des victoires d'etape depuis 1947, en excluant les team time trials.
- Les prologues et les contre-la-montre individuels sont comptes comme des victoires d'etape.
- Le panneau de droite affiche les leaders de chaque edition annuelle.
- Les portraits et drapeaux locaux sont utilises si disponibles dans `data/raw/player_photos/` et `data/raw/flags/`.

Source de donnees:

- Les pages annuelles du Tour de France sur Wikipedia, de 1947 a 2025.

## Tour de France Stage Wins Race Shorts

Fichiers principaux:

- Script principal: `video_generator/cycling/generate_tour_de_france_stage_wins_race_shorts_moviepy.py`
- Wrapper: `video_generator/generate_tour_de_france_stage_wins_race_shorts_moviepy.py`
- Sortie finale: `data/processed/cycling/tour_de_france/tour_de_france_stage_wins_postwar_1947_2025_shorts.mp4`

Comment est faite la video:

- Format vertical `1080x1920`, duree par defaut `60s`, `60 fps`.
- Bar chart race du top 12 cumule depuis 1947.
- Portraits, drapeaux et couleurs fixes par coureur.
- Annee centree dans le header et resume des principaux vainqueurs de chaque edition.
- Les graduations restent derriere les barres et les changements de rang gardent la barre montante au-dessus.

Commande:

- `python video_generator/generate_tour_de_france_stage_wins_race_shorts_moviepy.py`

## Template Tour des Flandres Shorts

Fichiers principaux:

- CSV: `data/processed/cycling/tour_of_flanders_titles_top10_cards.csv`
- Builder CSV: `python scraper/cycling/build_tour_of_flanders_titles_cards_csv.py`
- Video finale: `python video_generator/cycling/generate_tour_of_flanders_titles_cards_shorts_moviepy.py`

Compatibilite:

- Le wrapper `python video_generator/generate_tour_of_flanders_titles_cards_shorts_moviepy.py` reste utilisable.

Comment est faite la video:

- Format Shorts vertical avec cards qui defilent horizontalement.
- On affiche les `10` plus gros vainqueurs du Tour des Flandres.
- Le tri se fait d'abord par `nombre de titres`, puis par `nombre de podiums` en cas d'egalite.
- La numerotation est reordonnee pour afficher `1` au meilleur rang.
- Chaque card affiche:
  - photo du coureur
  - drapeau
  - nombre de titres
  - nombre de podiums
  - annees de victoire
- Les portraits sont stockes localement dans `data/raw/` et la video finale est regeneree apres ajout/correction des photos manquantes.

Sorties:

- Video finale: `data/processed/cycling/tour_of_flanders_titles_top10_shorts.mp4`

Source de reference:

- Historique officiel hommes elite: `https://www.rondevanvlaanderen.be/en/race/men-elite/history`

## Template MVP Race Shorts

Fichiers principaux:

- Script principal: `video_generator/basketball/generate_mvp_race_shorts_moviepy.py`
- Wrapper: `video_generator/generate_mvp_race_shorts_moviepy.py`
- Sortie finale par defaut: `data/processed/basketball/mvp_race_shorts.mp4`
- Preview conseillee: `data/processed/basketball/mvp_race_shorts_preview.mp4`

Structure conseillee:

- `data/raw/mvp_race_assets/`
  - `jokic.png`
  - `sga.png`
  - `doncic.png`
  - autres PNG / JPG si tu adaptes le template
- `data/raw/audio/`
  - musique de fond
  - SFX optionnels `swoosh` et `hit`

Comment est faite la video:

- Format vertical `1080x1920`
- Duree par defaut `30s`
- `30 fps`
- Hook `2s`
- Intro candidats `4s`
- Comparaison de `4` stats `12s`
- Reveal score MVP `6s`
- Podium `4s`
- CTA final `2s`
- Fond dramatique avec glow, vignetting, particules et motion legere
- Fallback propre si images, fontes ou audio manquent

Stats d'exemple:

- `points per game`
- `assists per game`
- `rebounds per game`
- `team win %`
- `score MVP /100`

Requirements:

- `moviepy>=2.0`
- `Pillow>=10.0`
- `numpy>=1.24`

Commandes:

- Rendu standard:
  - `python video_generator/generate_mvp_race_shorts_moviepy.py`
- Avec SFX:
  - `python video_generator/generate_mvp_race_shorts_moviepy.py --swoosh-sfx data/raw/audio/swoosh.mp3 --hit-sfx data/raw/audio/hit.mp3`
- Preview rapide:
  - `python video_generator/generate_mvp_race_shorts_moviepy.py --output data/processed/basketball/mvp_race_shorts_preview.mp4 --music data/raw/audio/audio.mp3`

## Template NBA MVP Ladder Weekly Race Shorts

Fichiers principaux:

- Builder CSV: `scraper/basketball/build_nba_kia_mvp_ladder_weekly_csv.py`
- Script principal: `video_generator/basketball/generate_nba_mvp_ladder_weekly_race_shorts_moviepy.py`
- CSV par defaut: `data/processed/basketball/nba_kia_mvp_ladder_2025_26_weekly.csv`
- Sortie finale par defaut: `data/processed/basketball/nba_mvp_ladder_cumulative_race_shorts.mp4`
- Preview conseillee: `tmp_frames/nba_mvp_ladder_cumulative_race_preview.png`
- Assets locaux: `data/raw/mvp_race_assets/`
- Audio de fond par defaut: `data/raw/audio/Midnight_Grip_20260402_0828.mp3`

Comment est faite la video:

- Format vertical `1080x1920`
- Duree par defaut `40s`, `60 fps`
- Race hebdomadaire du top 5 de la Kia MVP Ladder NBA
- Score cumule par rang: `100`, `80`, `65`, `50`, `40`
- Les noms des joueurs ne sont pas affiches sur les barres pour coller au style age-race
- Les photos et barres sont recentrees dans le layout short
- Les changements de rang sont animes: la barre qui monte passe visiblement au-dessus

Commandes:

- Construire le CSV:
  - `python scraper/basketball/build_nba_kia_mvp_ladder_weekly_csv.py`
- Generer la video:
  - `python video_generator/basketball/generate_nba_mvp_ladder_weekly_race_shorts_moviepy.py`

## Template LeBron Jordan Kobe Points By Age Race Shorts

Fichiers principaux:

- Builder CSV: `scraper/basketball/build_nba_points_by_age_csv.py`
- Script principal: `video_generator/basketball/generate_lebron_jordan_kobe_points_by_age_race_shorts_moviepy.py`
- CSV par defaut: `data/processed/basketball/nba_points_by_age_lebron_jordan_kobe.csv`
- Sortie finale par defaut: `data/processed/basketball/lebron_jordan_kobe_points_by_age_race_shorts.mp4`
- Preview conseillee: `tmp_frames/lebron_jordan_kobe_points_by_age_race_preview.png`
- Assets locaux: `data/raw/nba_goat_assets/`
- Audio de fond par defaut: `data/raw/audio/Midnight_Grip_20260402_0828.mp3`

Comment est faite la video:

- Format vertical `1080x1920`
- Duree par defaut `40s`, `60 fps`
- Race LeBron vs Jordan vs Kobe sur les points NBA de saison reguliere cumules par age
- Donnees extraites depuis Basketball-Reference, table `totals_stats`
- Le script reutilise le moteur Federer/Nadal/Djokovic age race avec overrides NBA
- Axe age `18` a `41`, avec valeurs annuelles dans les ronds et cumul dans les barres
- Ligne horizontale continue jusqu'au bord droit, dessinee derriere les ronds pour rester toujours presente

Commandes:

- Construire le CSV:
  - `python scraper/basketball/build_nba_points_by_age_csv.py`
- Generer la video:
  - `python video_generator/basketball/generate_lebron_jordan_kobe_points_by_age_race_shorts_moviepy.py`

## Template NBA Playoff Bracket Shorts

Fichiers principaux:

- Script principal: `video_generator/basketball/generate_nba_playoff_bracket_shorts_moviepy.py`
- Wrapper: `video_generator/generate_nba_playoff_bracket_shorts_moviepy.py`
- Sortie finale par defaut: `data/processed/basketball/nba_playoff_bracket_shorts.mp4`

Comment est faite la video:

- Format vertical `1080x1920`
- Bracket stylise en 3 marches: round 1, semis, finals
- Les teams gagnantes glissent d'une colonne a l'autre jusqu'au titre
- Les equipes eliminees restent visibles en mode dim pour garder le contexte du bracket
- Le champion finit sur une carte doree au sommet de l'ecran

Commandes:

- `python video_generator/generate_nba_playoff_bracket_shorts_moviepy.py`

## Template MVP Race Shorts Manim

Fichiers principaux:

- Script principal: `video_generator/basketball/generate_mvp_race_shorts_manim.py`
- Wrapper: `video_generator/generate_mvp_race_shorts_manim.py`
- Requirements: `requirements-manim-mvp-race.txt`

Structure:

- `data/raw/mvp_race_assets/`
  - `jokic.png`
  - `sga.png`
  - `doncic.png`
- `data/raw/audio/`
  - musique de fond
  - SFX optionnels
- `data/processed/basketball/manim_mvp_race/`
  - rendus Manim

Comment est fait le template:

- `Scene` reutilisables:
  - `HookScene`
  - `Top3IntroScene`
  - `StatsBarsScene`
  - `ScoreRevealScene`
  - `PodiumScene`
  - `CTAScene`
  - `MVPRaceShort`
- Format vertical `1080x1920`
- Pacing Shorts avec mouvement ou intensification toutes les `1-2s`
- Fond dramatique, glow, cartes premium, reveal du leader et podium final
- Mix audio optionnel via MoviePy apres le render Manim

Installation:

- `pip install -r requirements-manim-mvp-race.txt`

Commandes:

- Afficher l'aide:
  - `python video_generator/generate_mvp_race_shorts_manim.py`
- Render de la scene principale:
  - `python video_generator/generate_mvp_race_shorts_manim.py --render --scene MVPRaceShort --quality h`
- Render + mix audio:
- `python video_generator/generate_mvp_race_shorts_manim.py --render --scene MVPRaceShort --quality h --mix-audio --audio data/raw/audio/audio.mp3`

## Template Federer vs Nadal Duel Shorts

Fichiers principaux:

- Script principal: `video_generator/tennis/generate_federer_vs_nadal_duel_shorts_moviepy.py`
- Wrapper: `video_generator/generate_federer_vs_nadal_duel_shorts_moviepy.py`
- Sortie finale par defaut: `data/processed/tennis/federer_vs_nadal_duel_shorts_optimized.mp4`

Caractere du template:

- Format vertical `1080x1920`
- Duree cible `11.5s`
- Hook ultra court, duel visuel immediat, puis 4 stats en mode battle
- Climax centre sur `Clay Titles 11 vs 63`
- Fin loop-friendly pour pousser la revision et le commentaire

Commande:

- `python video_generator/generate_federer_vs_nadal_duel_shorts_moviepy.py`

## Template Federer vs Nadal H2H Score Timeline Shorts

Fichiers principaux:

- Script principal: `video_generator/tennis/generate_federer_vs_nadal_h2h_scores_shorts_moviepy.py`
- Wrapper: `video_generator/generate_federer_vs_nadal_h2h_scores_shorts_moviepy.py`
- Sortie finale par defaut: `data/processed/tennis/federer_vs_nadal_h2h_scores_shorts.mp4`

Caractere du template:

- Format vertical `1080x1920`
- Duree cible `45s`
- Timeline verticale des matchs marquants Federer/Nadal
- Les trophées de l'exemple sont remplaces par des badges de score
- Scoreboard final de la rivalite: `16-24`
- Hook et outro pensés pour un Short mobile lisible et rapide

Commande:

- `python video_generator/generate_federer_vs_nadal_h2h_scores_shorts_moviepy.py`

## Template Federer vs Nadal Retro Fight Shorts

Fichiers principaux:

- Script principal: `video_generator/tennis/generate_federer_vs_nadal_retro_fight_shorts_moviepy.py`
- Wrapper: `video_generator/generate_federer_vs_nadal_retro_fight_shorts_moviepy.py`
- Sortie finale par defaut: `data/processed/tennis/federer_vs_nadal_retro_fight_shorts.mp4`

Caractere du template:

- Format vertical `1080x1920`
- Ambiance retro gaming, CRT, neon, arcade
- Fighters 3D-ish stylises Federer et Nadal
- Chaque stat declenche un coup, un impact et un renversement visuel
- Fin loop-friendly pour pousser la relecture

Commande:

- `python video_generator/generate_federer_vs_nadal_retro_fight_shorts_moviepy.py`

## Template Federer Nadal Djokovic Grand Slam Age Race Shorts

Fichiers principaux:

- Script principal: `video_generator/tennis/generate_federer_nadal_djokovic_age_race_shorts_moviepy.py`
- Wrapper: `video_generator/generate_federer_nadal_djokovic_age_race_shorts_moviepy.py`
- Sortie finale par defaut: `data/processed/tennis/federer_nadal_djokovic_age_race_shorts.mp4`

Caractere du template:

- Format vertical `1080x1920`
- Inspiré du poster de reference avec fond bleu-violet premium et layout sports broadcast
- Portraits stacks a gauche, barres animees a droite, axe des ages en haut
- Race par age sur les titres du Grand Chelem
- Totaux mis en avant: `20`, `22`, `24`
- Recap final des totaux en bas pour boucler le short

Commande:

- `python video_generator/generate_federer_nadal_djokovic_age_race_shorts_moviepy.py`

## World Population Race

Fichiers principaux:

- Builder CSV: `scraper/demography/build_world_population_timeseries_csv.py`
- Wrapper builder: `scraper/build_world_population_timeseries_csv.py`
- Generateur: `video_generator/demography/generate_world_population_race_moviepy.py`
- Wrapper video: `video_generator/generate_world_population_race_moviepy.py`
- CSV: `data/processed/demography/world_population/world_population_1960_2024.csv`
- Sortie: `data/processed/demography/world_population/world_population_race_1960_2024_3min.mp4`

Commandes:

- `python scraper/build_world_population_timeseries_csv.py`
- `python video_generator/generate_world_population_race_moviepy.py`

Caracteristiques:

- Format paysage `1920x1080`, duree par defaut `180s`, `60 fps`.
- Top 12 des pays par population totale entre 1960 et 2024.
- Donnees `1960-2024` depuis l'API officielle World Bank, indicateur `SP.POP.TOTL`.
- Les agregats regionaux et economiques World Bank sont exclus via les metadonnees pays.
- Couleur fixe par pays, drapeaux locaux et valeurs affichees au dixieme de million, meme au-dessus d'un milliard.
- Les populations et longueurs de barres evoluent lineairement entre deux annees.
- Annee centree dans son wrapper, graduations derriere les barres et barre montante au premier plan.

## Template Ronaldo Messi Goals By Age Race Shorts

Fichiers principaux:

- Script principal: `video_generator/football/generate_ronaldo_messi_goals_by_age_race_shorts_moviepy.py`
- Sortie finale par defaut: `data/processed/football/ronaldo_messi_goals_by_age_race_midnight.mp4`
- Preview conseillee: `tmp_frames/ronaldo_messi_goals_by_age_race_preview.png`
- Audio de fond par defaut: `data/raw/audio/Midnight_Grip_20260402_0828.mp3`

Caractere du template:

- Format vertical `1080x1920`
- Duree par defaut `40s`, `60 fps`
- Race Ronaldo vs Messi sur le nombre de buts cumules selon l'age
- Layout inspire du short de reference: fond bleu-violet, portraits a gauche, barres a droite
- Ligne horizontale continue jusqu'au bord droit, dessinee derriere les ronds pour rester toujours presente
- Ronds de gains annuels masques progressivement derriere la barre quand ils la touchent
- Animation de changement de rang: la barre qui monte passe entierement au-dessus

Commande:

- `python video_generator/football/generate_ronaldo_messi_goals_by_age_race_shorts_moviepy.py`
