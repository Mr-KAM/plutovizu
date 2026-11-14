# plutovizu

Bibliothèque Python de visualisation de données basée sur matplotlib, avec une API chaînable pour créer des graphiques à partir de DataFrames pandas.

## Installation

```bash
pip install plutovizu
```

Ou avec `uv` :

```bash
uv add plutovizu
```

**Dépendances :** matplotlib, pandas, numpy, seaborn (Python ≥ 3.12)

## Utilisation rapide

```python
from plutovizu import Chart
import pandas as pd

df = pd.DataFrame({'categorie': ['A', 'B', 'C'], 'ventes': [120, 85, 200]})

(Chart(df, figsize=(10, 6))
    .bar('categorie', 'ventes')
    .labels(xlabel='Catégorie', ylabel='Ventes')
    .grid()
    .show())
```

## Types de graphiques

| Méthode | Description |
| --- | --- |
| `line(x, y)` | Courbe (supporte plusieurs colonnes y) |
| `bar(x, y)` | Barres verticales ou horizontales |
| `scatter(x, y)` | Nuage de points |
| `histogram(column)` | Histogramme |
| `boxplot(columns)` | Boîte à moustaches |
| `pie(column)` | Camembert |
| `heatmap()` | Carte de chaleur (matrice de corrélation) |
| `area(x, y)` | Aire (empilée ou non) |
| `violin(columns)` | Violon |
| `ombrothermic(month, temp, precip)` | Diagramme ombrothermique (p=2T) |

## API — méthodes de personnalisation

```python
chart = Chart(df, figsize=(12, 8))

# Graphique
chart.line('date', ['ventes', 'profit'], marker='o', linewidth=2)

# Apparence
chart.labels(xlabel='Date', ylabel='Montant', fontsize=12)
chart.grid(show=True, alpha=0.3, linestyle='--')
chart.set_legend(show=True, location='upper right')
chart.set_limits(xlim=(0, 100), ylim=(0, 500))
chart.rotate_labels(x_rotation=45)
chart.font(family='Times New Roman', size=12, weight='normal')

# Couleurs
chart.set_colors('deep')                          # palette seaborn/matplotlib
chart.create_custom_palette('ma_palette', ['#FF5733', 'blue', 'green'])
chart.generate_gradient_palette('grad', 'blue', 'red', n_colors=8)
chart.preview_palette('Set1')

# Export
chart.tight_layout()
chart.save('graphique.png', dpi=300)
chart.show()
fig, ax = chart.get_figure()
```

## Polices

```python
from plutovizu import get_fonts

# Lister toutes les polices disponibles
get_fonts()

# Filtrer
get_fonts(pattern='Arial')

# Depuis une instance Chart
chart.get_available_fonts(pattern='Times')
Chart.print_available_fonts(limit=20)
```

## Diagramme ombrothermique

```python
import pandas as pd
from plutovizu import Chart

climate = pd.DataFrame({
    'Mois': ['Jan', 'Fév', 'Mar', 'Avr', 'Mai', 'Jun',
             'Jul', 'Aoû', 'Sep', 'Oct', 'Nov', 'Déc'],
    'Température (°C)': [24.1, 26.1, 26.6, 26.5, 26.1, 25.0,
                         23.9, 23.9, 24.3, 24.8, 24.9, 23.7],
    'Précipitations (mm)': [16, 23.9, 86.7, 116.7, 92.9, 147.8,
                            143.2, 215.8, 167.1, 109.4, 30.3, 33]
})

(Chart(climate, figsize=(14, 8))
    .ombrothermic('Mois', 'Température (°C)', 'Précipitations (mm)')
    .font('Times New Roman', 13)
    .grid(alpha=0.3)
    .tight_layout()
    .show())
```

## Palettes disponibles

```python
# Afficher toutes les palettes
Chart.print_available_palettes()

# Par catégorie : 'all', 'seaborn', 'matplotlib', 'custom'
Chart.print_available_palettes(category='seaborn', limit=10)
```

## Développement

```text
src/plutovizu/   code du package (chart.py : classe Chart, get_fonts)
tests/           tests pytest
docs/            documentation Quarto
examples/        notebook de démonstration
```

```bash
uv sync            # installe le package en mode éditable + dépendances dev
uv run pytest      # lance les tests
uv build           # construit sdist + wheel dans dist/
quarto render docs # génère la documentation
```
