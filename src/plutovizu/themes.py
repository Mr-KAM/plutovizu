"""Thèmes prédéfinis pour plutovizu.

Un thème est un jeu de réglages matplotlib (rcParams) : fond, couleurs du texte,
palette, police, grille, épaisseurs… Comme ``seaborn.set_theme``, il s'applique
à tous les graphiques créés *après* l'appel.

    from plutovizu import Chart, set_theme, theme_context

    set_theme('pluto')                 # pour toute la session
    Chart(df, theme='neon')            # raccourci : set_theme('neon') puis Chart(df)
    with theme_context('paper'):       # seulement dans ce bloc
        Chart(df).line('x', 'y').show()
    set_theme(None)                    # retour au style matplotlib d'origine

Le thème 'editorial' est appliqué par défaut dès ``import plutovizu`` ;
``set_theme(None)`` revient au style matplotlib, ``set_theme()`` remet 'editorial'.

Si le paquet optionnel ``morethemes`` est installé (``pip install plutovizu[morethemes]``),
ses thèmes (wsj, economist, ft, retro, nord…) sont aussi disponibles par leur nom.
"""
import warnings
from contextlib import contextmanager
from typing import Dict, List, Optional, Union

import matplotlib as mpl
import seaborn as sns
from cycler import cycler

# Fira Sans (défaut) et Inter sont fournies avec plutovizu (voir fonts.py)
_SANS = ['Fira Sans', 'Inter', 'Segoe UI', 'Helvetica Neue', 'Arial', 'DejaVu Sans']
_SERIF = ['Georgia', 'Cambria', 'Times New Roman', 'DejaVu Serif']
_MONO = ['JetBrains Mono', 'Cascadia Mono', 'Consolas', 'DejaVu Sans Mono']


def _theme(bg: str, fg: str, muted: str, grid: str, palette: List[str],
           font: str = 'sans-serif', cmap: str = 'viridis', **extra) -> Dict:
    """Socle commun : fond uni, pas de cadre haut/droite, grille discrète,
    titre aligné à gauche, pas de graduations, légende sans cadre."""
    rc = {
        # Couleurs
        'figure.facecolor': bg, 'axes.facecolor': bg, 'savefig.facecolor': bg,
        'text.color': fg, 'axes.titlecolor': fg, 'axes.labelcolor': muted,
        'axes.edgecolor': grid, 'xtick.color': muted, 'ytick.color': muted,
        'grid.color': grid, 'axes.prop_cycle': cycler(color=palette), 'image.cmap': cmap,
        # Typo
        'font.family': font, 'font.sans-serif': _SANS, 'font.serif': _SERIF,
        'font.monospace': _MONO, 'font.size': 10,
        'axes.titlesize': 15, 'axes.titleweight': 'bold', 'axes.titlelocation': 'left',
        'axes.titlepad': 16, 'axes.labelsize': 10, 'axes.labelpad': 8,
        'xtick.labelsize': 9, 'ytick.labelsize': 9, 'legend.fontsize': 9,
        # Structure
        'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': 1.0,
        'axes.grid': True, 'axes.grid.axis': 'both', 'axes.axisbelow': True,
        'grid.linewidth': .8, 'grid.linestyle': '-', 'grid.alpha': 1.0,
        'xtick.major.size': 0, 'ytick.major.size': 0,
        'xtick.major.pad': 6, 'ytick.major.pad': 6,
        'legend.frameon': False,
        # Tracés
        'lines.linewidth': 2.2, 'lines.markersize': 6,
        # Export : 400 dpi par défaut (plt.savefig, fig.savefig, Chart.save)
        'savefig.dpi': 400,
        'patch.force_edgecolor': True, 'patch.edgecolor': bg, 'patch.linewidth': .8,
    }
    rc.update(extra)
    return rc


THEMES: Dict[str, Dict] = {
    # Couleurs du logo, assorti à la documentation : bleu nuit + violet + orange
    'pluto': _theme(
        bg='#14123A', fg='#F7F6FF', muted='#b3aedb', grid='#2B2670',
        palette=['#A79BFF', '#FF8A3D', '#5EEAD4', '#F472B6', '#FCD34D', '#60A5FA', '#E4E0FF'],
        cmap='magma'),
    # Presse : fond blanc, grille horizontale seule, aucun axe vertical
    'editorial': _theme(
        bg='#ffffff', fg='#111827', muted='#6b7280', grid='#e5e7eb',
        palette=['#2563eb', '#f97316', '#10b981', '#ef4444', '#8b5cf6', '#eab308', '#06b6d4'],
        cmap='Blues', **{'axes.grid.axis': 'y', 'axes.spines.left': False}),
    # Papier ancien : crème, typo serif, couleurs terre
    'paper': _theme(
        bg='#f7f3ea', fg='#2b2a26', muted='#6b675c', grid='#e4ddcc',
        palette=['#c2410c', '#1e3a5f', '#5f7f3a', '#b45309', '#7c3a5c', '#3f6f73'],
        font='serif', cmap='YlOrBr', **{'axes.titleweight': 'normal', 'axes.titlesize': 17}),
    # Froid et doux : gris bleuté, palette polaire
    'arctic': _theme(
        bg='#eceff4', fg='#2e3440', muted='#4c566a', grid='#d8dee9',
        palette=['#5e81ac', '#bf616a', '#a3be8c', '#d08770', '#b48ead', '#88c0d0', '#ebcb8b'],
        cmap='GnBu'),
    # Rétro-futuriste : violet nuit, couleurs fluo
    'neon': _theme(
        bg='#0f0a1e', fg='#f5f3ff', muted='#a49cc4', grid='#251d3d',
        palette=['#ff2bd6', '#00e5ff', '#faff00', '#7cff6b', '#ff7a00', '#b388ff'],
        cmap='plasma', **{'lines.linewidth': 2.6}),
    # Plan d'architecte : bleu, traits blancs fins, grille pointillée, police mono
    'blueprint': _theme(
        bg='#0b3d91', fg='#ffffff', muted='#c7d7f5', grid='#4a72b8',
        palette=['#ffffff', '#ffd166', '#7fdbff', '#ff9e9e', '#b9fbc0'],
        font='monospace', cmap='cool',
        **{'axes.spines.top': True, 'axes.spines.right': True, 'axes.edgecolor': '#c7d7f5',
           'grid.linestyle': ':', 'lines.linewidth': 1.6, 'axes.titleweight': 'normal'}),
    # Crépuscule : prune profond, oranges et roses chauds
    'sunset': _theme(
        bg='#1d1226', fg='#fdf2f8', muted='#c4a7c9', grid='#33223f',
        palette=['#ff7b54', '#ffd56b', '#e85d9a', '#9b5de5', '#ffb26b', '#f15bb5'],
        cmap='magma'),
    # Forêt : vert sous-bois, tons naturels
    'forest': _theme(
        bg='#0f1d17', fg='#ecf3ec', muted='#9fb5a5', grid='#1f3329',
        palette=['#8fd694', '#e9c46a', '#f4a261', '#76c7c0', '#e76f51', '#c9e4a7'],
        cmap='YlGn'),
    # Océan : bleu abyssal, cyan et corail
    'ocean': _theme(
        bg='#0a1929', fg='#e6f1ff', muted='#8aa4c2', grid='#16304d',
        palette=['#3dd6d0', '#5aa9ff', '#ffd166', '#ff6b9a', '#b39ddb', '#a0e7e5'],
        cmap='mako'),
    # Terminal rétro : fond noir, phosphore vert, police mono
    'terminal': _theme(
        bg='#050805', fg='#33ff66', muted='#2bbf55', grid='#0f2a14',
        palette=['#33ff66', '#ffb000', '#00e5ff', '#ff5f5f', '#d0ff5f'],
        font='monospace', cmap='Greens',
        **{'axes.titleweight': 'normal', 'lines.linewidth': 1.6}),
    # Encre : noir et blanc + un seul rouge d'accent, idéal pour l'impression
    'ink': _theme(
        bg='#ffffff', fg='#111111', muted='#555555', grid='#e6e6e6',
        palette=['#111111', '#d62828', '#7a7a7a', '#b0b0b0', '#4a4a4a'],
        cmap='Greys', **{'lines.linewidth': 1.8}),
    # Pastel : fond lavande très clair, couleurs douces mais lisibles
    'pastel': _theme(
        bg='#fbfaff', fg='#3b3552', muted='#7d7896', grid='#ece9f7',
        palette=['#7aa6f5', '#f28b8b', '#7ccf8a', '#f5b86b', '#a594f0', '#6cc9d6'],
        cmap='PuBuGn', **{'lines.linewidth': 2.8}),
    # Sahel : sable, terre cuite, ocre et indigo
    'sahel': _theme(
        bg='#f5ebdc', fg='#3b2a1a', muted='#7a6249', grid='#e6d6bf',
        palette=['#c1440e', '#335c67', '#e09f3e', '#9e2a2b', '#6a994e', '#540b0e'],
        cmap='YlOrBr'),
    # Accessible : palette Okabe-Ito, distinguable par les personnes daltoniennes
    'accessible': _theme(
        bg='#ffffff', fg='#1a1a1a', muted='#595959', grid='#e8e8e8',
        palette=['#0072B2', '#E69F00', '#009E73', '#D55E00', '#CC79A7', '#56B4E9', '#F0E442'],
        cmap='cividis'),
    # Présentation : aplats doux (sauge, corail, moutarde, ciel, lilas), titre centré non gras,
    # axes gauche/bas foncés, grille légère, police ronde géométrique (Raleway)
    'prezi': _theme(
        bg='#ffffff', fg='#3d3d3d', muted='#555555', grid='#dedede',
        palette=['#8ec3a7', '#de5253', '#f0ca6a', '#5eb6e4', '#ad92c8', '#f2a06d'],
        cmap='crest',
        **{'font.sans-serif': ['Raleway', 'Montserrat', 'Segoe UI', 'Helvetica Neue',
                               'Arial', 'DejaVu Sans'],
           'axes.titlelocation': 'center', 'axes.titleweight': 'normal', 'axes.titlesize': 17,
           'axes.titlepad': 18, 'axes.edgecolor': '#333333', 'axes.linewidth': 1.4,
           'axes.labelsize': 11, 'xtick.labelsize': 10, 'ytick.labelsize': 10,
           'legend.fontsize': 10, 'grid.linewidth': 0.9,
           'patch.edgecolor': '#ffffff', 'patch.linewidth': 1.5, 'lines.linewidth': 2.6}),
}

# Polices Google Fonts chargées (si pyfonts est installé) pour certains thèmes maison ;
# sans pyfonts ou sans réseau, la liste font.sans-serif du thème sert de repli, sans avertissement.
_GOOGLE_FONTS = {'prezi': 'Raleway'}

DESCRIPTIONS: Dict[str, str] = {
    'pluto': "Bleu nuit, violet et orange : les couleurs du logo et de la documentation.",
    'editorial': "Style presse : fond blanc, grille horizontale seule, aucun axe vertical.",
    'paper': "Papier crème, typographie serif et couleurs terre : un rendu imprimé.",
    'arctic': "Gris bleuté et palette polaire : doux et froid, idéal pour des rapports.",
    'neon': "Rétro-futuriste : violet nuit et couleurs fluo.",
    'blueprint': "Plan d'architecte : fond bleu, traits blancs fins, grille pointillée, police mono.",
    'sunset': "Crépuscule : prune profond, oranges et roses chauds.",
    'forest': "Sous-bois : vert profond et tons naturels.",
    'ocean': "Bleu abyssal, cyan et corail.",
    'terminal': "Terminal rétro : phosphore vert sur noir, police mono.",
    'ink': "Noir et blanc avec un seul rouge d'accent : parfait pour l'impression.",
    'pastel': "Fond lavande très clair, couleurs douces et traits épais.",
    'sahel': "Sable, terre cuite, ocre et indigo.",
    'accessible': "Palette Okabe-Ito, lisible par les personnes daltoniennes.",
    'prezi': "Style présentation : aplats doux, titre centré, police ronde (Raleway).",
}

# ── Thèmes du paquet optionnel morethemes (MIT, https://github.com/y-sunflower/morethemes)
# En cas de conflit de nom, le thème plutovizu est prioritaire.
# 'brand' est exclu : il exige un fichier _brand.yml.
try:
    import morethemes as _mt
    MORETHEMES: Dict[str, Dict] = {
        name: dict(_mt.get_rcparams(name))
        for name in _mt.ALL_THEMES if name != 'brand' and name not in THEMES
    }
    DESCRIPTIONS.update({name: f"{_mt.ALL_THEMES[name]['description']} (morethemes)"
                         for name in MORETHEMES})
except ImportError:
    MORETHEMES = {}

# ── Styles seaborn : les 5 styles officiels + la palette 'deep' de seaborn
_SNS_STYLES = {
    'darkgrid': "Fond gris clair et grille blanche : le style par défaut de seaborn.",
    'whitegrid': "Fond blanc et grille grise discrète.",
    'dark': "Fond gris clair, sans grille.",
    'white': "Fond blanc, sans grille.",
    'ticks': "Fond blanc, sans grille, avec graduations sur les axes.",
}
SEABORN: Dict[str, Dict] = {
    f'seaborn-{style}': {**dict(sns.axes_style(style)),
                         'axes.prop_cycle': cycler(color=sns.color_palette('deep'))}
    for style in _SNS_STYLES
}
DESCRIPTIONS.update({f'seaborn-{s}': f"{d} (seaborn)" for s, d in _SNS_STYLES.items()})

# Contextes seaborn : mettent le texte et les traits à l'échelle du support
CONTEXTS = ('paper', 'notebook', 'talk', 'poster')

# Valeurs matplotlib d'origine de toutes les clés touchées par un thème ou un contexte :
# on les restaure avant chaque changement pour qu'aucun réglage ne « fuie » d'un thème à l'autre.
_DEFAULTS = {k: mpl.rcParamsDefault[k]
             for k in set().union(*THEMES.values(), *MORETHEMES.values(), *SEABORN.values(),
                                  sns.plotting_context('notebook'))}

ThemeLike = Union[str, Dict, None]
_SOURCES = {'plutovizu': THEMES, 'morethemes': MORETHEMES, 'seaborn': SEABORN}


def list_themes(source: str = 'all') -> List[str]:
    """Retourne les noms des thèmes disponibles.

    Args:
        source: 'all' (défaut), 'plutovizu', 'morethemes' ou 'seaborn'.
    """
    if source == 'all':
        return sorted({name for group in _SOURCES.values() for name in group})
    if source not in _SOURCES:
        raise ValueError(f"source doit valoir 'all' ou {list(_SOURCES)}, pas '{source}'")
    return sorted(_SOURCES[source])


def _resolve(theme: ThemeLike, context: Optional[str] = None) -> Dict:
    if context is not None and context not in CONTEXTS:
        raise ValueError(f"Contexte inconnu '{context}'. Disponibles : {list(CONTEXTS)}")
    ctx = dict(sns.plotting_context(context)) if context else {}
    if theme is None:
        return ctx
    if isinstance(theme, dict):  # thème personnalisé : dict de rcParams
        return {**theme, **ctx}
    for group in _SOURCES.values():
        if theme in group:
            return {**group[theme], **ctx}
    hint = '' if MORETHEMES else " (plus de thèmes avec : pip install plutovizu[morethemes])"
    raise ValueError(f"Thème inconnu '{theme}'. Disponibles : {list_themes()}{hint}")


def _load_font(theme: ThemeLike) -> None:
    """Les thèmes morethemes utilisent une police Google Fonts, téléchargée (puis mise
    en cache) par pyfonts. Sans réseau : avertissement et police par défaut."""
    if not isinstance(theme, str):
        return
    if theme in _GOOGLE_FONTS:  # thème maison : police bonus, repli silencieux
        try:
            from pyfonts import load_google_font, set_default_font
            set_default_font(load_google_font(_GOOGLE_FONTS[theme]))
        except Exception:
            pass
        return
    if theme not in MORETHEMES:
        return
    family = MORETHEMES[theme].get('font.family')
    if not isinstance(family, str):
        return
    try:
        from pyfonts import load_google_font, set_default_font
        set_default_font(load_google_font(family))
    except Exception as exc:  # réseau absent, police retirée de Google Fonts…
        warnings.warn(f"Police « {family} » indisponible ({exc.__class__.__name__}) : "
                      "police par défaut utilisée.", stacklevel=3)
        mpl.rcParams['font.family'] = 'sans-serif'


def set_theme(theme: ThemeLike = 'editorial', context: Optional[str] = None) -> None:
    """Applique un thème à tous les graphiques créés ensuite.

    Args:
        theme: nom d'un thème (voir ``list_themes()``), dict de rcParams
            pour un thème personnalisé, ou ``None`` pour revenir au style matplotlib.
        context: échelle du texte et des traits selon le support :
            'paper', 'notebook', 'talk' ou 'poster' (contextes seaborn). Défaut : celle du thème.
    """
    rc = _resolve(theme, context)
    mpl.rcParams.update(_DEFAULTS)
    mpl.rcParams.update(rc)
    _load_font(theme)


@contextmanager
def theme_context(theme: ThemeLike, context: Optional[str] = None):
    """Context manager : applique un thème seulement à l'intérieur d'un bloc ``with``."""
    with mpl.rc_context({**_DEFAULTS, **_resolve(theme, context)}):
        _load_font(theme)
        yield


def get_theme(name: str) -> Dict:
    """Retourne une copie des réglages d'un thème, pour en dériver un thème perso.

    Exemple : ``set_theme({**get_theme('pluto'), 'axes.titlesize': 20})``
    """
    return dict(_resolve(name))
