"""Palettes pypalettes (plus de 2 500 palettes : MetBrewer, MoMAColors, nationalparkcolors…)
utilisables partout dans plutovizu.

    chart.set_colors('Cross')                 # couleurs des séries
    chart.preview_palette('Hiroshige')
    Chart(df).contour('x', 'y', 'z', cmap=pypalette('Hiroshige', continuous=True))
    # une fois chargée, la palette est aussi utilisable par son nom : cmap='Hiroshige'
"""
import csv
import warnings
from functools import lru_cache
from importlib import resources
from typing import Dict, List, Optional

import matplotlib
from matplotlib.colors import Colormap


_KINDS = {'qualitative': 'qualitative', 'categorical': 'qualitative',
          'sequential': 'sequential', 'diverging': 'diverging'}


@lru_cache(maxsize=1)
def pypalettes_catalogue() -> Dict[str, str]:
    """{nom: type} des palettes pypalettes : 'qualitative', 'sequential', 'diverging' ou 'autre'."""
    try:
        with resources.files('pypalettes').joinpath('palettes.csv').open('r', encoding='utf-8') as f:
            return {row['name']: _KINDS.get(row['kind'].split('-')[0], 'autre')
                    for row in csv.DictReader(f)}
    except (ModuleNotFoundError, FileNotFoundError):  # ponytail: format interne de pypalettes
        return {}


def pypalette_colors(name: str, reverse: bool = False) -> Optional[List[str]]:
    """Couleurs (hex) d'une palette pypalettes, ou None si le nom est inconnu."""
    if name not in pypalettes_catalogue():
        return None
    from pypalettes import load_cmap
    return list(load_cmap(name, reverse=reverse).colors)


def pypalette(name: str, continuous: bool = False, reverse: bool = False) -> Colormap:
    """
    EN: A pypalettes palette as a matplotlib colormap — discrete (one color per class) or
        continuous (gradient, for cmap=...). It is also registered under its name, so
        cmap='Name' works afterwards in any chart.
    FR: Une palette pypalettes en colormap matplotlib — discrète (une couleur par classe) ou
        continue (dégradé, pour cmap=...). Elle est aussi enregistrée sous son nom :
        cmap='Nom' marche ensuite dans tous les graphiques.

    Example / Exemple:
        Chart(df).isopleth('date', 'profondeur', 'humidite', cmap=pypalette('Hokusai2', True))
    """
    from pypalettes import load_cmap
    if name not in pypalettes_catalogue():
        raise ValueError(f"Palette pypalettes inconnue : {name!r} (liste : Chart.get_available_palettes())")
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        cmap = load_cmap(name, cmap_type='continuous' if continuous else 'discrete', reverse=reverse)
    key = name + ('_r' if reverse else '')
    if key not in matplotlib.colormaps:
        matplotlib.colormaps.register(cmap.copy(), name=key)
    return cmap
