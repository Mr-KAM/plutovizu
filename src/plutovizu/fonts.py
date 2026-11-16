"""Polices de plutovizu : Fira Sans (défaut) et Inter sont fournies avec le package et
enregistrées dans matplotlib dès l'import ; les autres polices Google Fonts sont
téléchargées (puis mises en cache) par pyfonts.

    from plutovizu import load_font
    titre = load_font('Inter', weight='bold')        # police fournie : pas de réseau
    manuscrite = load_font('Caveat')                 # Google Fonts, via pyfonts
    chart.title('Rendement', fontproperties=titre)

Les deux familles sont sous licence SIL Open Font License (fichiers OFL-*.txt).
"""
from pathlib import Path
from typing import List, Union

from matplotlib import font_manager as fm
from matplotlib.font_manager import FontProperties

FONTS_DIR = Path(__file__).parent / 'fonts'
BUNDLED_FONTS = ('Fira Sans', 'Inter')
DEFAULT_FONT = 'Fira Sans'

_WEIGHTS = {'light': 300, 'normal': 400, 'regular': 400, 'medium': 500,
            'semibold': 600, 'demibold': 600, 'bold': 700}
_NAMES = {300: 'Light', 400: 'Regular', 500: 'Medium', 600: 'SemiBold', 700: 'Bold'}


def register_fonts() -> List[str]:
    """Enregistre les polices fournies avec plutovizu dans matplotlib (appelée à l'import)."""
    files = sorted(FONTS_DIR.glob('*.ttf'))
    for f in files:
        fm.fontManager.addfont(str(f))
    return [f.name for f in files]


def _weight(weight: Union[int, str]) -> int:
    if isinstance(weight, str):
        if weight.lower() not in _WEIGHTS:
            raise ValueError(f'Graisse inconnue {weight!r} : {sorted(_WEIGHTS)} ou un nombre (300–700)')
        return _WEIGHTS[weight.lower()]
    return int(weight)


def load_font(family: str = DEFAULT_FONT, weight: Union[int, str] = 'regular',
              italic: bool = False) -> FontProperties:
    """
    EN: FontProperties for a font. Bundled fonts (Fira Sans, Inter) are read from the
        package (no network); any other Google Fonts family is downloaded and cached by
        pyfonts, then registered so its name also works in rcParams.
    FR: FontProperties d'une police. Les polices fournies (Fira Sans, Inter) sont lues dans
        le package (sans réseau) ; toute autre famille Google Fonts est téléchargée et mise
        en cache par pyfonts, puis enregistrée pour que son nom marche aussi dans rcParams.

    Example / Exemple:
        chart.title('Titre', fontproperties=load_font('Inter', 'semibold'))
    """
    import pyfonts
    w = _weight(weight)
    if family in BUNDLED_FONTS:
        nearest = min(_NAMES, key=lambda k: abs(k - w))
        style = _NAMES[nearest]
        if italic:  # italiques fournies : Regular et Bold
            style = 'BoldItalic' if nearest >= 600 else 'Italic'
        return pyfonts.load_font(str(FONTS_DIR / f"{family.replace(' ', '')}-{style}.ttf"))
    fp = pyfonts.load_google_font(family, weight=w, italic=italic)
    fm.fontManager.addfont(fp.get_file())
    return fp
