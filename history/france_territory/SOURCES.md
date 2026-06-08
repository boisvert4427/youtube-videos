# Sources et méthode

Cette animation présente une synthèse pédagogique de l'évolution du territoire
français. Les surfaces historiques sont des reconstitutions schématiques
destinées à montrer les grandes mutations. Elles ne doivent pas être utilisées
comme des frontières cadastrales ou diplomatiques précises.

Le principe retenu est volontairement simple: des jalons datés fixes, puis une
interpolation visuelle entre deux cartes de référence. Quand la connaissance
historique est trop incertaine, on préfère un repère explicite à une frontière
inventée.

## Fond cartographique

- Natural Earth, `Admin 0 - Countries`, version 5.1.1 :
  https://www.naturalearthdata.com/downloads/10m-cultural-vectors/10m-admin-0-countries/
- Fichier simplifié utilisé pour le rendu :
  `data/raw/ne_50m_admin_0_countries.geojson`

Natural Earth fournit les côtes et les frontières contemporaines des pays
voisins. Les limites historiques sont stockées séparément dans
`data/raw/france_territory_periods.geojson`.

## Base chronologique

- Euratlas, `Georeferenced Historical Vector Data 100`, cartes vectorielles
  européennes datées à la fin de chaque siècle de l'an 1 à l'an 2000 :
  https://www.euratlas.net/shop/maps_gis/gis_100.html
- BnF / Gallica, carte de Cassini et sélections chronologiques des cartes de
  France :
  https://gallica.bnf.fr/selections/fr/html/cartes-de-france-acces-chronologique
- IGN, patrimoine cartographique et cartes anciennes dématérialisées :
  https://remonterletemps.ign.fr/
  https://www.data.gouv.fr/datasets/cartes-anciennes-dematerialisees

## Références historiques

- Bibliothèque nationale de France, sélection chronologique des cartes de
  France :
  https://gallica.bnf.fr/selections/fr/html/cartes-de-france-acces-chronologique
- IGN, carte de Cassini et histoire de la première couverture complète du
  royaume :
  https://macarte.ign.fr/carte/eIPAnb/La-carte-de-cassini
- FranceArchives, réunion de la Corse à la France :
  https://francearchives.gouv.fr/findingaid/026fa82fa7935f2c761cd7b43dd0a4b9b3583650
- Vie publique, rattachement de Nice en 1860 :
  https://www.vie-publique.fr/discours/182145-frederic-mitterrand-14052011-boulevard-francois-mitterrand-nice
- Chemins de mémoire, cession de l'Alsace-Lorraine en 1871 :
  https://www.cheminsdememoire.gouv.fr/fr/1871-03-01-par-576-voix-contre-107-lassemblee-nationale-accepte-les-conditions-de-paix-allemandes-d
- Chemins de mémoire, restitution de l'Alsace-Lorraine en 1919 :
  https://www.cheminsdememoire.gouv.fr/fr/1919-06-28-signature-versailles-dans-la-galerie-des-glaces-du-traite-de-paix-lalsace-lorraine-e
- Chemins de mémoire, ressources cartographiques sur la Révolution et le
  Premier Empire :
  https://www.cheminsdememoire.gouv.fr/fr/la-revolution-et-lempire

## Choix de représentation

- Avant 843, la vidéo parle de la Gaule et des royaumes francs, pas de la
  France au sens moderne.
- Pour 987, la surface colorée représente le domaine directement contrôlé par
  le roi, et non l'ensemble juridique du royaume.
- Pour 1360, elle représente principalement les espaces contrôlés par la
  couronne pendant la guerre de Cent Ans.
- Pour 1812, elle représente la France impériale et ses départements annexés,
  sans inclure tous les États alliés ou satellites de Napoléon.
- La carte principale reste centrée sur l'Europe occidentale. L'outre-mer est
  signalé dans la dernière étape par un cartouche dédié.
