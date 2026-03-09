# atp_vertical_timeline_v1

Colonnes obligatoires:

1. `year` (entier, ex: 1968)
2. `player_name` (nom affiche sur la carte)

Colonnes optionnelles:

1. `subtitle` (texte court sous le badge principal)
2. `image_path` (chemin relatif depuis la racine projet, ex: `data/raw/player_photos/rod_laver.jpg`)
3. `name_bg_color` (couleur hex, ex: `#f4df26`)
4. `card_bg_color` (couleur hex, ex: `#5f3518`)
5. `rank_label` (texte du badge, ex: `RG #1`)
6. `results` (liste de lignes separees par `|`)

Notes:

- Une ligne = une annee dans la timeline.
- Les annees sont triees automatiquement avant rendu video.
- Si `image_path` est vide ou introuvable, une carte de fallback est affichee.
