# Notebooks de démonstration

Un notebook par thème, chacun indépendant (imports et données d'exemple inclus). Les données
sont fictives. La [documentation](https://mr-kam.github.io/plutovizu/) détaille chaque
graphique : usage, structure du DataFrame attendu et paramètres.

| Notebook | Contenu |
|:--|:--|
| [01_prise_en_main](01_prise_en_main.ipynb) | Courbes, barres, nuages de points : l'API chaînable |
| [02_catalogue_matplotlib](02_catalogue_matplotlib.ipynb) | Bande, frise, rose des vents, anneau, champs, 3D, ellipses |
| [03_climat_et_sols](03_climat_et_sols.ipynb) | Diagramme ombrothermique, climatogrammes, bilan hydrique, pédologie |
| [04_sciences_de_la_nature](04_sciences_de_la_nature.ipynb) | Essai agronomique : régression, Tukey, ACP, dendrogramme, textures |
| [05_ecologie](05_ecologie.ipynb) | GLM, courbes de réponse, ordination, diversité |
| [06_analyse_qualitative](06_analyse_qualitative.ipynb) | Nuage de mots, fréquences, Likert, matrice des codes |
| [07_map](07_map.ipynb) | Points, zones, flux, densité, isolignes, itinéraires (`pip install plutovizu[map]`) |
| [08_graphes_reseaux](08_graphes_reseaux.ipynb) | Réseaux, plus court chemin, arbre couvrant, flot maximal, chemin critique, tournées |
| [09_seaborn](09_seaborn.ipynb) | Graphiques seaborn avec le thème et le chaînage |
| [10_mise_en_forme](10_mise_en_forme.ipynb) | Thèmes, palettes, polices et export |

Les notebooks utilisent le noyau Jupyter `plutovizu` :

```bash
uv sync
uv run python -m ipykernel install --user --name plutovizu
```
