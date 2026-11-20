# Notebooks de démonstration

Un notebook par thème, chacun indépendant (imports et données d'exemple inclus). Les données
sont fictives. La [documentation](https://mr-kam.github.io/plutovizu/) détaille chaque
graphique : usage, structure du DataFrame attendu et paramètres.

| Notebook | Contenu |
|:--|:--|
| [01_prise_en_main](01_prise_en_main.ipynb) | Courbes, barres, nuages de points, classements (bump charts) : l'API chaînable |
| [02_catalogue_matplotlib](02_catalogue_matplotlib.ipynb) | Bande, frise, rose des vents, anneau, champs, 3D, ellipses |
| [03_climat_et_sols](03_climat_et_sols.ipynb) | Diagramme ombrothermique, climatogrammes, climagramme d'Emberger, écoclimatogramme, bilan hydrique, pédologie |
| [04_sciences_de_la_nature](04_sciences_de_la_nature.ipynb) | Essai agronomique : régression, Tukey, ACP, dendrogramme, textures |
| [05_ecologie](05_ecologie.ipynb) | GLM, courbes de réponse, ordination, diversité |
| [06_analyse_qualitative](06_analyse_qualitative.ipynb) | Nuage de mots, fréquences, Likert, matrice des codes |
| [07_map](07_map.ipynb) | Points, zones, flux, densité, isolignes, itinéraires (`pip install plutovizu[map]`) |
| [08_graphes_reseaux](08_graphes_reseaux.ipynb) | Réseaux, plus court chemin, arbre couvrant, flot maximal, chemin critique, tournées |
| [09_graphiques_complexes](09_graphiques_complexes.ipynb) | Pareto, carte de contrôle, Marimekko, Bland-Altman, raincloud, grilles seaborn |
| [10_mise_en_forme](10_mise_en_forme.ipynb) | Thèmes, palettes (dont pypalettes), polices (Fira Sans, Inter), annotations, coins arrondis, taille, résolution, traduction, export |
| [11_diagrammes_infographie](11_diagrammes_infographie.ipynb) | Sankey, cordes, Venn, logigramme, organigramme, séquence ; infographies (polices, fond, annotations, courbes étiquetées, histogramme log, l'IA en chiffres 2025-2026) ; données dans [datasets/](datasets/) |
| [12_cote_divoire](12_cote_divoire.ipynb) | Données réelles de Côte d'Ivoire : 60 ans de cacao, production par région (carte choroplèthe), chiffres cacao-café, financement agricole, pyramide des âges, carte des sites touristiques (geopandas) |

Les notebooks utilisent le noyau Jupyter `plutovizu` :

```bash
uv sync
uv run python -m ipykernel install --user --name plutovizu
```
