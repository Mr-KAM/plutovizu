"""Triangle des textures du sol USDA : 12 classes, limites en % de (sable, limon, argile).

    from plutovizu import texture_class
    texture_class(40, 40, 20)            # 'Loam'
    texture_class(df['sable'], df['limon'], df['argile'])   # une classe par ligne

Noms des classes : terminologie française usuelle de la classification USDA
(« loam » = terre franche).
"""
from typing import Dict, List, Tuple, Union

import numpy as np

# Sommets (sable, limon, argile) de chaque classe, pour le tracé du triangle
USDA_POLYGONS: Dict[str, List[Tuple[float, float, float]]] = {
    'Sable': [(100, 0, 0), (85, 15, 0), (90, 0, 10)],
    'Sable loameux': [(85, 15, 0), (70, 30, 0), (85, 0, 15), (90, 0, 10)],
    'Loam sableux': [(70, 30, 0), (50, 50, 0), (43, 50, 7), (52, 41, 7), (52, 28, 20),
                     (80, 0, 20), (85, 0, 15)],
    'Loam': [(43, 50, 7), (23, 50, 27), (45, 28, 27), (52, 28, 20), (52, 41, 7)],
    'Loam limoneux': [(50, 50, 0), (20, 80, 0), (8, 80, 12), (0, 88, 12), (0, 73, 27),
                      (23, 50, 27), (43, 50, 7)],
    'Limon': [(20, 80, 0), (0, 100, 0), (0, 88, 12), (8, 80, 12)],
    'Loam sablo-argileux': [(80, 0, 20), (52, 28, 20), (45, 28, 27), (45, 20, 35), (65, 0, 35)],
    'Loam argileux': [(45, 28, 27), (20, 53, 27), (20, 40, 40), (45, 15, 40)],
    'Loam limono-argileux': [(20, 53, 27), (0, 73, 27), (0, 60, 40), (20, 40, 40)],
    'Argile sableuse': [(65, 0, 35), (45, 20, 35), (45, 0, 55)],
    'Argile limoneuse': [(20, 40, 40), (0, 60, 40), (0, 40, 60)],
    'Argile': [(45, 15, 40), (20, 40, 40), (0, 40, 60), (0, 0, 100), (45, 0, 55)],
}


def _rules(sand, silt, clay) -> List[Tuple[str, np.ndarray]]:
    """Règles officielles du calculateur de texture USDA, dans l'ordre de priorité."""
    return [
        ('Sable', silt + 1.5 * clay < 15),
        ('Sable loameux', (silt + 1.5 * clay >= 15) & (silt + 2 * clay < 30)),
        ('Loam sableux', ((clay >= 7) & (clay < 20) & (sand > 52) & (silt + 2 * clay >= 30))
         | ((clay < 7) & (silt < 50) & (silt + 2 * clay >= 30))),
        ('Loam', (clay >= 7) & (clay < 27) & (silt >= 28) & (silt < 50) & (sand <= 52)),
        ('Loam limoneux', ((silt >= 50) & (clay >= 12) & (clay < 27))
         | ((silt >= 50) & (silt < 80) & (clay < 12))),
        ('Limon', (silt >= 80) & (clay < 12)),
        ('Loam sablo-argileux', (clay >= 20) & (clay < 35) & (silt < 28) & (sand > 45)),
        ('Loam argileux', (clay >= 27) & (clay < 40) & (sand > 20) & (sand <= 45)),
        ('Loam limono-argileux', (clay >= 27) & (clay < 40) & (sand <= 20)),
        ('Argile sableuse', (clay >= 35) & (sand > 45)),
        ('Argile limoneuse', (clay >= 40) & (silt >= 40)),
        ('Argile', (clay >= 40) & (sand <= 45) & (silt < 40)),
    ]


def texture_class(sand, silt, clay) -> Union[str, np.ndarray]:
    """Classe de texture USDA à partir des pourcentages (ou fractions) de sable,
    limon et argile. Chaque ligne est ramenée à 100 %.

    Retourne une chaîne pour des scalaires, un tableau de chaînes sinon.
    """
    s, si, c = (np.asarray(v, dtype=float) for v in (sand, silt, clay))
    total = s + si + c
    s, si, c = (100 * v / total for v in (s, si, c))
    out = np.full(np.shape(s), None, dtype=object)
    for name, mask in _rules(s, si, c):
        out = np.where((out == None) & mask, name, out)  # noqa: E711  (comparaison élément par élément)
    return out.item() if out.ndim == 0 else out
