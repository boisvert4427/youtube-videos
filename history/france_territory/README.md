# Évolution du territoire de la France

Animation Histovision en carte, de la Gaule romaine à la France
contemporaine.

## Méthode

Cette animation s'appuie sur des cartes et jalons datés fixes, puis
interpole seulement la transition visuelle entre deux repères. L'objectif est
de montrer une évolution lisible et fiable, sans prétendre à une frontière
historique continue exactement connue année par année.

## Contenu

- 15 étapes historiques
- carte d'Europe occidentale fondée sur Natural Earth
- territoires historiques vectoriels et schématiques
- transitions animées entre les périodes
- cartouches expliquant la nature exacte de chaque surface
- format YouTube `1920x1080`
- durée finale par défaut : `360` secondes
- fréquence finale par défaut : `60 fps`

## Prévisualisation

```powershell
python history/france_territory/scripts/generate_france_territory_video.py `
  --preview-year 1812 `
  --preview-output history/france_territory/data/processed/france_territory_preview_1812.png
```

## Vidéo complète

```powershell
python history/france_territory/scripts/generate_france_territory_video.py `
  --duration 360 `
  --fps 60 `
  --audio data/raw/audio/audio.mp3 `
  --output history/france_territory/data/processed/france_territory_2000_years_360s_60fps.mp4
```

La méthode et les références sont détaillées dans `SOURCES.md`.
