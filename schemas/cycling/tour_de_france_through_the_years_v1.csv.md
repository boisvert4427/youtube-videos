# tour_de_france_through_the_years_v1

Chaque ligne correspond a une edition du Tour de France.

Colonnes:

1. `year` - annee de l'edition.
2. `winner_name` - vainqueur du classement general.
3. `winner_country` - code pays du vainqueur, idealement en `ISO3`.
4. `winner_team` - equipe du vainqueur.
5. `winner_time` - temps final du vainqueur.
6. `gc2_name` - deuxieme du classement general.
7. `gc2_country` - code pays du deuxieme.
8. `gc2_team` - equipe du deuxieme.
9. `gc2_gap` - ecart du deuxieme.
10. `gc3_name` - troisieme du classement general.
11. `gc3_country` - code pays du troisieme.
12. `gc3_team` - equipe du troisieme.
13. `gc3_gap` - ecart du troisieme.
14. `points_name` - vainqueur du classement par points.
15. `points_country` - code pays du vainqueur par points.
16. `points_team` - equipe du vainqueur par points.
17. `mountains_name` - vainqueur du classement de la montagne.
18. `mountains_country` - code pays du vainqueur de la montagne.
19. `mountains_team` - equipe du vainqueur de la montagne.
20. `badge_label` - libelle du badge central de la carte.
21. `card_bg_color` - couleur de fond de la carte.
22. `accent_color` - couleur d'accent de la carte.

Notes:

- Les classements annexes peuvent rester vides pour les editions ou le maillot n'existait pas encore.
- Le format est pense pour une timeline cartes en mode paysage, avec un vainqueur, un podium et les maillots en bas.
- Le generateur paysage associe ce CSV aux portraits locaux de
  `data/raw/player_photos/tour_de_france_through_the_years/` et aux drapeaux de
  `data/raw/flags/`.
- Les maillots affiches en bas de carte sont des assets PNG detoures depuis
  l'image de reference utilisateur, stockes dans
  `data/raw/cycling/tour_de_france_jerseys/`:
  `yellow.png`, `green.png`, `polka.png`.
- Si un asset maillot manque, le generateur contient un fallback dessine, mais
  le rendu valide doit utiliser les PNG detoures.
- Preview par defaut:
  `data/processed/cycling/tour_de_france/tour_de_france_through_the_years_preview.png`.
- Sortie video par defaut:
  `data/processed/cycling/tour_de_france/tour_de_france_through_the_years_1947_2025_4min_60fps.mp4`.
