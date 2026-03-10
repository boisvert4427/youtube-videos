# paris_nice_timeline_postwar_v1

Colonnes obligatoires:

1. `year` (entier, ex: 1946)
2. `winner_name` (vainqueur du classement general)
3. `winner_team`
4. `winner_country` (code pays ou pays court)

Colonnes de media / style:

1. `image_path` (chemin relatif depuis la racine projet)
2. `card_bg_color` (couleur de fond de la carte)
3. `accent_color` (couleur d accent de la carte)
4. `badge_label` (texte du badge, ex: `PN #1`)

Classement general top 5:

1. `gc1_name`, `gc1_team`, `gc1_country`, `gc1_gap`
2. `gc2_name`, `gc2_team`, `gc2_country`, `gc2_gap`
3. `gc3_name`, `gc3_team`, `gc3_country`, `gc3_gap`
4. `gc4_name`, `gc4_team`, `gc4_country`, `gc4_gap`
5. `gc5_name`, `gc5_team`, `gc5_country`, `gc5_gap`

Classements annexes:

1. `points_name`, `points_team`, `points_country`
2. `mountains_name`, `mountains_team`, `mountains_country`

Colonne optionnelle:

1. `notes`

Notes:

- Une ligne = une edition de Paris-Nice.
- Hypothese de depart: periode apres-guerre = `1946` a aujourd hui.
- `gc1_*` doit en pratique correspondre au vainqueur et `gc1_gap` peut rester vide ou valoir `0:00`.
- Les ecarts (`gc*_gap`) doivent etre stockes en texte pour garder la mise en forme source (`0:34`, `+ 1:12`, etc.).
- Le template est pense pour une future card video affichant: vainqueur, top 5 general, maillot vert et maillot a pois.
