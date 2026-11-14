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
"""
from typing import Dict, List, Union

import matplotlib as mpl
from cycler import cycler

_SANS = ['Inter', 'Segoe UI', 'Helvetica Neue', 'Arial', 'DejaVu Sans']
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
        'patch.force_edgecolor': True, 'patch.edgecolor': bg, 'patch.linewidth': .8,
    }
    rc.update(extra)
    return rc


THEMES: Dict[str, Dict] = {
    # Sombre, assorti à la documentation : quasi-noir + émeraude
    'pluto': _theme(
        bg='#0b0d0e', fg='#f4f4f5', muted='#a1a1aa', grid='#232629',
        palette=['#34d399', '#60a5fa', '#f472b6', '#fbbf24', '#a78bfa', '#f87171', '#2dd4bf'],
        cmap='mako'),
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
}

# Valeurs matplotlib d'origine de toutes les clés touchées par un thème :
# on les restaure avant chaque changement pour qu'aucun réglage ne « fuie » d'un thème à l'autre.
_DEFAULTS = {k: mpl.rcParamsDefault[k] for k in set().union(*THEMES.values())}

ThemeLike = Union[str, Dict, None]


def list_themes() -> List[str]:
    """Retourne les noms des thèmes disponibles."""
    return sorted(THEMES)


def _resolve(theme: ThemeLike) -> Dict:
    if theme is None:
        return {}
    if isinstance(theme, dict):  # thème personnalisé : dict de rcParams
        return theme
    try:
        return THEMES[theme]
    except KeyError:
        raise ValueError(f"Thème inconnu '{theme}'. Disponibles : {list_themes()}") from None


def set_theme(theme: ThemeLike = 'pluto') -> None:
    """Applique un thème à tous les graphiques créés ensuite.

    Args:
        theme: nom d'un thème (voir ``list_themes()``), dict de rcParams
            pour un thème personnalisé, ou ``None`` pour revenir au style matplotlib.
    """
    rc = _resolve(theme)
    mpl.rcParams.update(_DEFAULTS)
    mpl.rcParams.update(rc)


def theme_context(theme: ThemeLike):
    """Context manager : applique un thème seulement à l'intérieur d'un bloc ``with``."""
    return mpl.rc_context({**_DEFAULTS, **_resolve(theme)})


def get_theme(name: str) -> Dict:
    """Retourne une copie des réglages d'un thème, pour en dériver un thème perso.

    Exemple : ``set_theme({**get_theme('pluto'), 'axes.titlesize': 20})``
    """
    return dict(_resolve(name))
