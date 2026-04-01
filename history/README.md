# history

Sous-section historique du projet `youtube-videos`, dédiée aux timelines historiques.

Première timeline incluse :
- rois de France par année
- plage couverte : `481` à `1848`
- format de sortie : CSV annuel
- aperçu visuel : PNG premium en timeline horizontale
- sortie vidéo : MP4 timeline
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
      france_kings_timeline_481_1848_60s.mp4
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
python history/scripts/generate_france_kings_timeline_video.py
```

## Hypothèse de travail

Cette version suit une succession canonique simplifiée de `Clovis Ier` à `Louis-Philippe Ier` pour produire une timeline claire.

Les grandes ruptures de monarchie sont marquées comme `Sans roi`, et quelques portraits anciens introuvables sont remplacés par des visuels héraldiques premium.
