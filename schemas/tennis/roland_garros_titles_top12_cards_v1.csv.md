# roland_garros_titles_top12_cards_v1

Colonnes obligatoires:

1. `rank` (entier, `1` pour le plus grand vainqueur)
2. `player_name` (nom affiche sur la carte)
3. `country_code` (code pays ISO 3166-1 alpha-2 en minuscules, ex: `es`)
4. `titles` (nombre de titres a Roland-Garros)
5. `years_won` (liste des annees gagnees, separees par ` / `)
6. `first_title` (premiere annee gagnee)
7. `last_title` (derniere annee gagnee)
8. `badge_label` (badge du haut de carte, ex: `ROLAND-GARROS x14`)
9. `card_bg_color` (couleur de fond de la carte)
10. `accent_color` (couleur d'accent)

Notes:

- Le jeu de donnees par defaut couvre les vainqueurs hommes de l'Open Era.
- Les cartes sont triees pour afficher les outsiders d'abord et le recordman en dernier.
- Si aucune photo locale n'est trouvee, une carte de fallback avec initiales est affichee.

