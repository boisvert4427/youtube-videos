# history

Sous-section historique du projet `youtube-videos`, dédiée aux timelines historiques.

Première timeline incluse :
- rois de France par année
- plage couverte : `481` à `1830`
- format de sortie : CSV annuel
- aperçu visuel : PNG premium en timeline horizontale
- sortie vidéo : MP4 timeline `france_kings_timeline_481_1830_300s_60fps_audio.mp4`
- dataset enrichi avec `3 faits marquants` par règne

## Structure

```text
history-timelines/
  data/
    raw/
      france_kings_reigns.csv
      portraits/
    processed/
      france_kings_yearly_timeline.csv
      france_kings_timeline_preview.png
      france_kings_timeline_481_1830_300s_60fps_audio.mp4
  scripts/
    build_france_kings_yearly_timeline.py
    download_france_kings_portraits.py
    render_france_kings_timeline_preview.py
    generate_france_kings_timeline_video.py
  requirements.txt
```

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Générer les données annuelles

```bash
python history/scripts/build_france_kings_yearly_timeline.py
```

## Enrichir les faits en français

```bash
python history/scripts/enrich_france_kings_facts.py
```

## Télécharger ou fabriquer les portraits

```bash
python history/scripts/download_france_kings_portraits.py
```

## Générer un aperçu visuel

```bash
python history/scripts/render_france_kings_timeline_preview.py
```

## Générer la vidéo

```bash
python history/scripts/generate_france_kings_timeline_video.py --duration 300 --fps 60 --audio data/raw/audio/audio.mp3 --output history/data/processed/france_kings_timeline_481_1830_300s_60fps_audio.mp4
```

La video de sortie reprend le style `HISTOVISION`, ajoute une legende integree a la frise, garde les blocs de texte contenus dans leur carte et s'arrete a `1830`.

## Hypothèse de travail

Cette version suit une succession canonique simplifiée de `Clovis Ier` à `Charles X` pour produire une timeline claire.

Les grandes ruptures de monarchie sont marquées comme `Sans roi`, et quelques portraits anciens introuvables sont remplacés par des visuels héraldiques premium.
