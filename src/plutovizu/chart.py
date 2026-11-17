import warnings
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.colors as mcolors
import pandas as pd
import numpy as np
from typing import Optional, Union, List, Tuple, Dict, Any
import seaborn as sns

from .fonts import load_font
from .palettes import pypalette_colors, pypalettes_catalogue
from .themes import set_theme

FIGSIZE = (16 / 2.54, 10 / 2.54)  # taille par défaut : 16 × 10 cm (en pouces)


def cm(width: float, height: float) -> Tuple[float, float]:
    """Taille en centimètres → pouces, pour figsize : Chart(df, figsize=cm(20, 12))."""
    return width / 2.54, height / 2.54


def _opts(**kwargs) -> Dict[str, Any]:
    """Garde seulement les options fournies : les autres suivent le thème actif (rcParams)."""
    return {k: v for k, v in kwargs.items() if v is not None}


def _tukey_letters(samples: List[pd.Series], alpha: float = 0.05) -> List[str]:
    """Lettres de comparaison d'un test de Tukey HSD (compact letter display).

    Algorithme « insérer puis absorber » (Piepho, 2004) : deux groupes partagent une
    lettre s'ils ne diffèrent pas significativement. 'a' = plus forte moyenne.
    """
    from scipy.stats import tukey_hsd
    n = len(samples)
    if n < 2:
        return ['a'] * n
    if any(len(s) < 2 for s in samples):
        raise ValueError("Le test de Tukey demande au moins 2 observations par groupe")
    p = tukey_hsd(*samples).pvalue
    cols = [set(range(n))]
    for i in range(n):
        for j in range(i + 1, n):
            if p[i, j] >= alpha:
                continue
            split = []
            for col in cols:  # insérer : on sépare i et j dans chaque colonne qui les réunit
                split += [col - {i}, col - {j}] if {i, j} <= col else [col]
            cols = []
            for col in split:  # absorber : on retire les colonnes incluses dans une autre
                if not any(col <= kept for kept in cols):
                    cols = [kept for kept in cols if not kept < col] + [col]
    rank = {g: r for r, g in enumerate(np.argsort([-np.mean(s) for s in samples]))}
    cols.sort(key=lambda col: sorted(rank[g] for g in col))
    letters = 'abcdefghijklmnopqrstuvwxyz'
    return [''.join(letters[k] for k, col in enumerate(cols) if g in col) for g in range(n)]


def get_fonts(pattern: str = None, sort: bool = True) -> List[str]:
    """
    EN: Return the list of all font names available on the system.
        Optionally filter by a substring pattern and sort alphabetically.
    FR: Retourne la liste de toutes les polices disponibles sur le système.
        Filtrage optionnel par sous-chaîne, tri alphabétique optionnel.

    Args / Paramètres:
        pattern (str, optional):
            EN: Case-insensitive substring to filter font names (e.g. 'Arial').
            FR: Sous-chaîne (insensible à la casse) pour filtrer les polices.
        sort (bool):
            EN: Whether to sort the result alphabetically. Default True.
            FR: Trier le résultat alphabétiquement. Défaut True.

    Returns / Retourne:
        List[str]:
            EN: Sorted (or unsorted) list of font family names.
            FR: Liste (triée ou non) des noms de familles de polices.

    Example / Exemple:
        get_fonts()                        # toutes les polices / all fonts
        get_fonts(pattern='Times')         # polices contenant 'Times'
        get_fonts(pattern='Arial', sort=False)
    """
    fonts = [f.name for f in fm.fontManager.ttflist]
    fonts = list(set(fonts))
    if pattern:
        fonts = [font for font in fonts if pattern.lower() in font.lower()]
    if sort:
        fonts.sort()
    return fonts



def _round_path(path, r: float, min_angle: float = 25.0):
    """Arrondit les coins vifs (> min_angle degrés) d'un chemin fermé, en pixels.

    Chaque coin est remplacé par une courbe de Bézier quadratique entre les points situés
    à r le long du contour de part et d'autre (r réduit si les coins sont proches).
    Retourne None s'il n'y a aucun coin à arrondir.
    """
    from matplotlib.path import Path
    verts, codes, changed = [], [], False
    for P in path.to_polygons(closed_only=True):
        if len(P) > 1 and np.allclose(P[0], P[-1]):
            P = P[:-1]
        P = P[np.r_[True, np.any(np.abs(np.diff(P, axis=0)) > 1e-9, axis=1)]]
        n = len(P)
        if n < 3:
            continue
        nxt = np.roll(P, -1, axis=0) - P
        seg = np.linalg.norm(nxt, axis=1)
        prev = np.roll(nxt, 1, axis=0)
        cosang = (prev * nxt).sum(1) / np.maximum(np.roll(seg, 1) * seg, 1e-12)
        sharp = np.where(np.degrees(np.arccos(np.clip(cosang, -1, 1))) > min_angle)[0]
        s = np.r_[0, np.cumsum(seg)[:-1]]
        L = seg.sum()
        if len(sharp) == 0 or L <= 0:
            verts += [*P, P[0]]
            codes += [Path.MOVETO] + [Path.LINETO] * (n - 1) + [Path.CLOSEPOLY]
            continue
        ss = s[sharp]
        gaps = np.diff(np.r_[ss, ss[0] + L])
        rr = min(r, 0.45 * gaps.min())

        def at(t):
            t %= L
            k = min(np.searchsorted(s, t, side='right') - 1, n - 1)
            return P[k] + nxt[k] * (t - s[k]) / max(seg[k], 1e-12)

        start = ss[0] + gaps[0] / 2                      # entre deux coins : hors arrondi
        sharp_set = set(sharp.tolist())
        items = []
        for j in range(n):
            d = (s[j] - start) % L
            if j in sharp_set:
                items += [(d - rr, Path.LINETO, at(s[j] - rr)), (d, Path.CURVE3, P[j]),
                          (d + rr, Path.CURVE3, at(s[j] + rr))]
            elif not any(abs(((s[j] - c + L / 2) % L) - L / 2) < rr for c in ss):
                items.append((d, Path.LINETO, P[j]))
        items.sort(key=lambda it: it[0])
        verts += [at(start)] + [v for _, _, v in items] + [at(start)]
        codes += [Path.MOVETO] + [c for _, c, _ in items] + [Path.CLOSEPOLY]
        changed = True
    return Path(np.array(verts), codes) if changed else None


class Chart:
    """
    EN: Fluent chart builder that wraps matplotlib for pandas DataFrames.
        Every method returns ``self``, enabling method chaining::

            Chart(df).line('date', 'sales').grid().set_legend().show()

        Supported chart types:
            line, bar, scatter, histogram, boxplot, pie, heatmap, area,
            violin, lollipop, ombrothermic, radar, waterfall, bump,
            ridgeline, treemap, hexbin, parallel_coordinates, dumbbell,
            streamgraph, gantt, calendar_heatmap, sankey.

    FR: Constructeur de graphiques fluide qui encapsule matplotlib pour
        les DataFrames pandas. Chaque méthode retourne ``self`` pour
        permettre le chaînage::

            Chart(df).line('date', 'ventes').grid().set_legend().show()

        Types de graphiques disponibles :
            line, bar, scatter, histogram, boxplot, pie, heatmap, area,
            violin, lollipop, ombrothermic, radar, waterfall, bump,
            ridgeline, treemap, hexbin, parallel_coordinates, dumbbell,
            streamgraph, gantt, calendar_heatmap, sankey.

    Attributes / Attributs:
        data (pd.DataFrame):
            EN: The DataFrame passed at construction.
            FR: Le DataFrame fourni à la construction.
        fig (plt.Figure):
            EN: The underlying matplotlib Figure.
            FR: La Figure matplotlib sous-jacente.
        ax (plt.Axes):
            EN: The primary matplotlib Axes.
            FR: L'Axes matplotlib principal.
        chart_type (str):
            EN: Name of the last chart type drawn.
            FR: Nom du dernier type de graphique tracé.
        custom_palettes (dict):
            EN: Registry of user-defined color palettes.
            FR: Registre des palettes de couleurs définies par l'utilisateur.
        current_palette:
            EN: Currently active color palette.
            FR: Palette de couleurs actuellement active.
    """

    # Coins arrondis appliqués par défaut à tous les graphiques (points) ;
    # Chart.default_round = 0 les désactive pour toute la session, .rounded(0) pour un graphique.
    default_round: float = 8.0
    # Graphiques dont les formes doivent rester jointives : pas d'arrondi par défaut
    # (.rounded(r) reste possible explicitement).
    _NO_DEFAULT_ROUND = frozenset({'soil_texture', 'ternary', 'calendar_heatmap', 'sankey', 'mesh',
                                   'soil_profile', 'climate_stripes', 'seaborn.jointplot',
                                   'exchange_complex'})

    def __init__(self, data: pd.DataFrame, figsize: Tuple[float, float] = FIGSIZE,
                 theme: Optional[str] = None):
        """
        EN: Initialise a Chart from a pandas DataFrame.
        FR: Initialise un Chart à partir d'un DataFrame pandas.

        Args / Paramètres:
            data (pd.DataFrame):
                EN: The dataset to visualise.
                FR: Le jeu de données à visualiser.
            figsize (Tuple[float, float]):
                EN: Figure width and height in inches. Default 16 × 10 cm (FIGSIZE);
                    cm(16, 10) gives a size in centimetres.
                FR: Largeur et hauteur de la figure en pouces. Défaut 16 × 10 cm (FIGSIZE) ;
                    cm(16, 10) donne une taille en centimètres.
            theme (str, optional):
                EN: Theme name (see list_themes()). Shortcut for set_theme(theme):
                    like seaborn.set_theme, it also applies to the following charts.
                FR: Nom d'un thème (voir list_themes()). Raccourci de set_theme(theme) :
                    comme seaborn.set_theme, il s'applique aussi aux graphiques suivants.

        Returns / Retourne:
            Chart: EN: The Chart instance (for chaining). / FR: L'instance Chart (pour le chaînage).

        Example / Exemple:
            chart = Chart(df)
            chart = Chart(df, figsize=(14, 8), theme='pluto')
        """
        if theme is not None:
            set_theme(theme)
        self.data = data
        self.fig, self.ax = plt.subplots(figsize=figsize)
        self.chart_type = None
        self.custom_palettes = {}
        self.current_palette = None
        self.sns_grid = None  # grille seaborn (pairplot, jointplot, clustermap…) le cas échéant
        self._round_radius = Chart.default_round   # coins arrondis par défaut (voir rounded)
        self._round_explicit = False

    # ──────────────────────────────────────── Appearance / Apparence

    def title(self, title: str, fontsize: Optional[int] = None, fontweight: Optional[str] = None,
              color: Optional[str] = None, pad: Optional[float] = None,
              loc: Optional[str] = None, **kwargs) -> 'Chart':
        """
        EN: Set the chart title. Options left to None follow the active theme.
        FR: Définit le titre du graphique. Les options laissées à None suivent le thème actif.

        Args / Paramètres:
            title (str):
                EN: Title text.
                FR: Texte du titre.
            fontsize (int, optional):
                EN: Font size. Default: theme.
                FR: Taille de la police. Défaut : thème.
            fontweight (str, optional):
                EN: Font weight ('normal', 'bold', 'light'…). Default: theme.
                FR: Graisse de la police. Défaut : thème.
            color (str, optional):
                EN: Title color. Default: theme.
                FR: Couleur du titre. Défaut : thème.
            pad (float, optional):
                EN: Padding above the axes in points. Default: theme.
                FR: Espace au-dessus des axes en points. Défaut : thème.
            loc (str, optional):
                EN: Alignment: 'left', 'center' or 'right'. Default: theme.
                FR: Alignement : 'left', 'center' ou 'right'. Défaut : thème.
            **kwargs:
                EN: Other matplotlib Text options, e.g. fontproperties=load_font('Inter', 'bold').
                FR: Autres options de texte matplotlib, ex. fontproperties=load_font('Inter', 'bold').

        Returns / Retourne:
            Chart: self

        Example / Exemple:
            chart.title('Ventes annuelles', fontsize=18, color='steelblue')
            chart.title('Ventes', fontproperties=load_font('Inter', 'semibold'))
        """
        if 'fontproperties' in kwargs:  # police choisie : garder la taille du titre du thème
            fp = fm.FontProperties._from_any(kwargs['fontproperties']).copy()
            fp.set_size(fontsize or plt.rcParams['axes.titlesize'])
            kwargs['fontproperties'] = fp
        if self.sns_grid is not None or getattr(self, '_panels', False):  # plusieurs panneaux : titre général
            self.fig.suptitle(title, y=1.02, **_opts(fontsize=fontsize, fontweight=fontweight,
                                                     color=color), **kwargs)
            return self
        self.ax.set_title(title, **_opts(fontsize=fontsize, fontweight=fontweight,
                                         color=color, pad=pad, loc=loc), **kwargs)
        return self

    def labels(self, xlabel: str = None, ylabel: str = None,
               fontsize: Optional[int] = None, color: Optional[str] = None) -> 'Chart':
        """
        EN: Set the x-axis and/or y-axis labels.
        FR: Définit les étiquettes des axes x et/ou y.

        Args / Paramètres:
            xlabel (str, optional):
                EN: Label for the x-axis.
                FR: Étiquette de l'axe x.
            ylabel (str, optional):
                EN: Label for the y-axis.
                FR: Étiquette de l'axe y.
            fontsize (int, optional):
                EN: Font size for both labels. Default: theme.
                FR: Taille de la police pour les deux étiquettes. Défaut : thème.
            color (str, optional):
                EN: Text color. Default: theme.
                FR: Couleur du texte. Défaut : thème.

        Returns / Retourne:
            Chart: self

        Example / Exemple:
            chart.labels(xlabel='Date', ylabel='Ventes (€)', fontsize=13)
        """
        opts = _opts(fontsize=fontsize, color=color)
        if self.sns_grid is not None and hasattr(self.sns_grid, 'set_axis_labels'):
            self.sns_grid.set_axis_labels(xlabel or '', ylabel or '', **opts)
            return self
        if xlabel:
            self.ax.set_xlabel(xlabel, **opts)
        if ylabel:
            self.ax.set_ylabel(ylabel, **opts)
        return self

    def grid(self, show: bool = True, alpha: Optional[float] = None,
             linestyle: Optional[str] = None, linewidth: Optional[float] = None) -> 'Chart':
        """
        EN: Toggle and style the axes grid.
        FR: Active/désactive et stylise la grille des axes.

        Args / Paramètres:
            show (bool):
                EN: Display the grid. Default True.
                FR: Afficher la grille. Défaut True.
            alpha (float, optional):
                EN: Grid opacity, 0–1. Default: theme.
                FR: Opacité de la grille, 0–1. Défaut : thème.
            linestyle (str, optional):
                EN: Line style ('-', '--', ':', '-.'). Default: theme.
                FR: Style des traits ('-', '--', ':', '-.'). Défaut : thème.
            linewidth (float, optional):
                EN: Line width in points. Default: theme.
                FR: Épaisseur du trait en points. Défaut : thème.

        Returns / Retourne:
            Chart: self

        Example / Exemple:
            chart.grid(alpha=0.4, linestyle='--')
        """
        opts = _opts(alpha=alpha, linestyle=linestyle, linewidth=linewidth)
        if show:
            # axis du thème : 'editorial' n'affiche que la grille horizontale
            self.ax.grid(True, axis=plt.rcParams['axes.grid.axis'], **opts)
        else:
            self.ax.grid(False)
        return self

    def set_legend(self, show: bool = True, location: str = 'best',
                   fontsize: Optional[int] = None, frameon: Optional[bool] = None) -> 'Chart':
        """
        EN: Show or hide the chart legend.
        FR: Affiche ou masque la légende du graphique.

        Args / Paramètres:
            show (bool):
                EN: Display the legend. Default True.
                FR: Afficher la légende. Défaut True.
            location (str):
                EN: Legend position ('best', 'upper right', 'lower left'…).
                FR: Position de la légende ('best', 'upper right', 'lower left'…).
            fontsize (int, optional):
                EN: Legend text size. Default: theme.
                FR: Taille du texte de la légende. Défaut : thème.
            frameon (bool, optional):
                EN: Draw a box around the legend. Default: theme.
                FR: Dessiner un cadre autour de la légende. Défaut : thème.

        Returns / Retourne:
            Chart: self

        Example / Exemple:
            chart.set_legend(location='upper left', frameon=False)
        """
        if show:
            self.ax.legend(loc=location, fontsize=fontsize, frameon=frameon)
        else:
            self.ax.legend().set_visible(False)
        return self

    def set_limits(self, xlim: Tuple[float, float] = None,
                   ylim: Tuple[float, float] = None) -> 'Chart':
        """
        EN: Set the visible range of the x and/or y axes.
        FR: Définit la plage visible des axes x et/ou y.

        Args / Paramètres:
            xlim (Tuple[float, float], optional):
                EN: (min, max) for the x-axis.
                FR: (min, max) pour l'axe x.
            ylim (Tuple[float, float], optional):
                EN: (min, max) for the y-axis.
                FR: (min, max) pour l'axe y.

        Returns / Retourne:
            Chart: self

        Example / Exemple:
            chart.set_limits(xlim=(2010, 2024), ylim=(0, 1000))
        """
        if xlim:
            self.ax.set_xlim(xlim)
        if ylim:
            self.ax.set_ylim(ylim)
        return self

    def rotate_labels(self, x_rotation: float = 0, y_rotation: float = 0) -> 'Chart':
        """
        EN: Rotate the tick labels on the x and/or y axis.
        FR: Fait pivoter les étiquettes des axes x et/ou y.

        Args / Paramètres:
            x_rotation (float):
                EN: Rotation angle in degrees for x-axis tick labels. Default 0.
                FR: Angle de rotation en degrés pour l'axe x. Défaut 0.
            y_rotation (float):
                EN: Rotation angle in degrees for y-axis tick labels. Default 0.
                FR: Angle de rotation en degrés pour l'axe y. Défaut 0.

        Returns / Retourne:
            Chart: self

        Example / Exemple:
            chart.rotate_labels(x_rotation=45)
        """
        if x_rotation:
            self.ax.tick_params(axis='x', rotation=x_rotation)
        if y_rotation:
            self.ax.tick_params(axis='y', rotation=y_rotation)
        return self

    def font(self, family: str = 'sans-serif', size: int = 10,
             weight: str = 'normal') -> 'Chart':
        """
        EN: Set the global font used across the entire chart.
            Falls back to 'sans-serif' if the requested family is not installed.
        FR: Définit la police globale utilisée sur tout le graphique.
            Revient à 'sans-serif' si la famille demandée n'est pas installée.

        Args / Paramètres:
            family (str):
                EN: Font family name or generic ('sans-serif', 'serif', 'monospace'…).
                    Use get_available_fonts() to list installed fonts.
                FR: Nom de la famille de polices ou générique ('sans-serif', 'serif'…).
                    Utiliser get_available_fonts() pour lister les polices installées.
            size (int):
                EN: Base font size in points. Default 10.
                FR: Taille de base de la police en points. Défaut 10.
            weight (str):
                EN: Font weight ('normal', 'bold', 'light'…). Default 'normal'.
                FR: Graisse de la police. Défaut 'normal'.

        Returns / Retourne:
            Chart: self

        Example / Exemple:
            chart.font('Times New Roman', size=13, weight='bold')
        """
        available_fonts = self.get_available_fonts()
        if family not in ['sans-serif', 'serif', 'monospace', 'fantasy', 'cursive'] \
                and family not in available_fonts:
            try:  # pas installée : Google Fonts via pyfonts (téléchargée une fois, en cache)
                load_font(family, weight=weight)
            except Exception:
                warnings.warn(f"Police '{family}' introuvable (ni installée, ni sur Google Fonts) : "
                              "'sans-serif' utilisée.", stacklevel=2)
                family = 'sans-serif'
        plt.rcParams.update({
            'font.family': family,
            'font.size': size,
            'font.weight': weight
        })
        return self

    # ──────────────────────────────────────── Color / Couleur

    def set_colors(self, colors: Union[str, List[str]] = None) -> 'Chart':
        """
        EN: Apply a color palette to subsequent chart elements.
            Accepts a seaborn palette name, a matplotlib colormap name, one of the
            2,700 pypalettes palettes (e.g. 'Cross', 'Hiroshige'), a custom palette name
            (created with create_custom_palette), or a list of color strings.
        FR: Applique une palette de couleurs aux éléments du graphique.
            Accepte un nom de palette seaborn, un colormap matplotlib, une des 2 700
            palettes pypalettes (ex. 'Cross', 'Hiroshige'), un nom de palette
            personnalisée (créée avec create_custom_palette), ou une liste de couleurs.

        Args / Paramètres:
            colors (Union[str, List[str]], optional):
                EN: Palette name string or explicit list of colors.
                FR: Nom de la palette ou liste explicite de couleurs.

        Returns / Retourne:
            Chart: self

        Example / Exemple:
            chart.set_colors('deep')
            chart.set_colors('Cross')                     # pypalettes
            chart.set_colors(['#e63946', '#457b9d', '#1d3557'])
        """
        if colors:
            if isinstance(colors, str):
                if colors in self.custom_palettes:
                    colors = self.custom_palettes[colors]
                else:  # palette seaborn ou colormap matplotlib, sinon pypalettes
                    try:
                        colors = sns.color_palette(colors)
                    except Exception:
                        found = pypalette_colors(colors)
                        if found is None:
                            raise ValueError(f"Palette '{colors}' inconnue (seaborn, matplotlib, "
                                             "pypalettes ou personnalisée)") from None
                        colors = found
            self._apply_palette(colors)
        return self

    def _apply_palette(self, colors) -> None:
        """Applique une palette à CE graphique seulement (pas aux réglages globaux)."""
        self.current_palette = [mcolors.to_hex(c) for c in colors]
        self.ax.set_prop_cycle(color=self.current_palette)

    def create_custom_palette(self, name: str, colors: List[str],
                              save_palette: bool = True) -> 'Chart':
        """
        EN: Define and optionally register a named custom color palette.
            Invalid color strings are skipped with a warning;
            raises ValueError if none is valid.
        FR: Définit et enregistre optionnellement une palette de couleurs nommée.
            Les couleurs invalides sont ignorées avec un avertissement ;
            lève ValueError si aucune n'est valide.

        Args / Paramètres:
            name (str):
                EN: Identifier for the palette (used in set_colors).
                FR: Identifiant de la palette (utilisé dans set_colors).
            colors (List[str]):
                EN: List of color values — hex codes, CSS names, RGB tuples…
                FR: Liste de valeurs de couleur — codes hex, noms CSS, tuples RGB…
            save_palette (bool):
                EN: Register the palette so it can be reused by name. Default True.
                FR: Enregistre la palette pour la réutiliser par son nom. Défaut True.

        Returns / Retourne:
            Chart: self

        Example / Exemple:
            chart.create_custom_palette('brand', ['#003049', '#d62828', '#f77f00'])
            chart.set_colors('brand')
        """
        valid_colors = []
        for color in colors:
            try:
                mcolors.to_rgba(color)
                valid_colors.append(color)
            except ValueError:
                warnings.warn(f"Invalid color '{color}' ignored", stacklevel=2)
        if not valid_colors:
            raise ValueError("No valid colors provided")
        if save_palette:
            self.custom_palettes[name] = valid_colors
        self._apply_palette(valid_colors)
        return self

    def get_available_palettes(self, include_custom: bool = True,
                               include_seaborn: bool = True,
                               include_matplotlib: bool = True,
                               include_pypalettes: bool = True) -> Dict[str, List[str]]:
        """
        EN: Return a catalogue of all available color palettes grouped by category.
        FR: Retourne un catalogue de toutes les palettes disponibles, groupées par catégorie.

        Args / Paramètres:
            include_custom (bool):
                EN: Include user-defined palettes. Default True.
                FR: Inclure les palettes définies par l'utilisateur. Défaut True.
            include_seaborn (bool):
                EN: Include seaborn built-in palettes. Default True.
                FR: Inclure les palettes intégrées de seaborn. Défaut True.
            include_matplotlib (bool):
                EN: Include matplotlib colormaps. Default True.
                FR: Inclure les colormaps matplotlib. Défaut True.
            include_pypalettes (bool):
                EN: Include the pypalettes palettes (qualitative, sequential, diverging, other).
                FR: Inclure les palettes pypalettes (qualitatives, séquentielles, divergentes, autres).

        Returns / Retourne:
            Dict[str, List[str]]:
                EN: Keys are category names; values are lists of palette/colormap names.
                FR: Les clés sont les noms de catégories; les valeurs sont des listes de noms.

        Example / Exemple:
            palettes = chart.get_available_palettes(include_matplotlib=False)
            print(palettes['seaborn_qualitative'])
        """
        palettes = {
            'custom': [],
            'seaborn_qualitative': [],
            'seaborn_sequential': [],
            'seaborn_diverging': [],
            'matplotlib_sequential': [],
            'matplotlib_diverging': [],
            'matplotlib_cyclic': [],
            'matplotlib_qualitative': []
        }
        if include_custom:
            palettes['custom'] = list(self.custom_palettes.keys())
        if include_seaborn:
            palettes['seaborn_qualitative'] = [
                'deep', 'muted', 'bright', 'pastel', 'dark', 'colorblind',
                'Set1', 'Set2', 'Set3', 'Paired', 'tab10', 'tab20'
            ]
            palettes['seaborn_sequential'] = [
                'Blues', 'BuGn', 'BuPu', 'GnBu', 'Greens', 'Greys', 'Oranges',
                'OrRd', 'PuBu', 'PuBuGn', 'PuRd', 'Purples', 'RdPu', 'Reds',
                'YlGn', 'YlGnBu', 'YlOrBr', 'YlOrRd', 'rocket', 'mako', 'flare', 'crest'
            ]
            palettes['seaborn_diverging'] = [
                'BrBG', 'PiYG', 'PRGn', 'PuOr', 'RdBu', 'RdGy', 'RdYlBu',
                'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic', 'icefire', 'vlag'
            ]
        if include_matplotlib:
            palettes['matplotlib_sequential'] = [
                'viridis', 'plasma', 'inferno', 'magma', 'cividis',
                'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
                'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
                'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn'
            ]
            palettes['matplotlib_diverging'] = [
                'PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu',
                'RdYlBu', 'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic'
            ]
            palettes['matplotlib_cyclic'] = ['twilight', 'twilight_shifted', 'hsv']
            palettes['matplotlib_qualitative'] = [
                'Pastel1', 'Pastel2', 'Paired', 'Accent', 'Dark2',
                'Set1', 'Set2', 'Set3', 'tab10', 'tab20', 'tab20b', 'tab20c'
            ]
        if include_pypalettes:
            for name, kind in sorted(pypalettes_catalogue().items()):
                palettes.setdefault(f'pypalettes_{kind}', []).append(name)
        return palettes

    @staticmethod
    def print_available_palettes(category: str = 'all', limit: int = None) -> None:
        """
        EN: Print the catalogue of available palettes to the console.
        FR: Affiche le catalogue des palettes disponibles dans la console.

        Args / Paramètres:
            category (str):
                EN: Filter output: 'all', 'seaborn', 'matplotlib', 'pypalettes', 'custom',
                    or any specific sub-category key. Default 'all'.
                FR: Filtre la sortie : 'all', 'seaborn', 'matplotlib', 'pypalettes', 'custom',
                    ou toute clé de sous-catégorie. Défaut 'all'.
            limit (int, optional):
                EN: Maximum palettes to show per category.
                FR: Nombre maximum de palettes à afficher par catégorie.

        Returns / Retourne:
            None

        Example / Exemple:
            Chart.print_available_palettes(category='seaborn', limit=5)
        """
        temp_chart = Chart(pd.DataFrame({'x': [1], 'y': [1]}))
        palettes = temp_chart.get_available_palettes()
        if category == 'all':
            categories_to_show = list(palettes.keys())
        elif category == 'seaborn':
            categories_to_show = [k for k in palettes if k.startswith('seaborn')]
        elif category == 'matplotlib':
            categories_to_show = [k for k in palettes if k.startswith('matplotlib')]
        elif category == 'pypalettes':
            categories_to_show = [k for k in palettes if k.startswith('pypalettes')]
        elif category in palettes:
            categories_to_show = [category]
        else:
            raise ValueError(f"Unknown category '{category}'. Available: "
                             f"{['all', 'seaborn', 'matplotlib', 'pypalettes', *palettes]}")
        print("Available Color Palettes:")
        print("=" * 50)
        for cat in categories_to_show:
            if palettes[cat]:
                print(f"\n{cat.replace('_', ' ').title()}:")
                print("-" * 30)
                palette_list = palettes[cat][:limit] if limit else palettes[cat]
                for i, p in enumerate(palette_list, 1):
                    print(f"{i:3d}. {p}")
                if limit and len(palettes[cat]) > limit:
                    print(f"    ... and {len(palettes[cat]) - limit} more")

    def preview_palette(self, palette_name: str, n_colors: int = 8) -> 'Chart':
        """
        EN: Render a horizontal color-bar preview of the given palette.
            Supports custom palettes, seaborn palettes, matplotlib colormaps and pypalettes.
        FR: Affiche un aperçu horizontal de la palette sous forme de barre de couleurs.
            Supporte les palettes personnalisées, seaborn, les colormaps matplotlib et pypalettes.

        Args / Paramètres:
            palette_name (str):
                EN: Name of the palette or colormap to preview.
                FR: Nom de la palette ou du colormap à prévisualiser.
            n_colors (int):
                EN: Number of color swatches to show. Default 8.
                FR: Nombre d'échantillons de couleur à afficher. Défaut 8.

        Returns / Retourne:
            Chart: self

        Example / Exemple:
            chart.preview_palette('viridis', n_colors=10)
            chart.preview_palette('brand')   # palette personnalisée / custom
        """
        if palette_name in self.custom_palettes:
            colors = self.custom_palettes[palette_name][:n_colors]
        else:
            try:
                colors = sns.color_palette(palette_name, n_colors)
            except Exception:
                try:
                    cmap = matplotlib.colormaps[palette_name]
                    colors = [cmap(i / max(n_colors - 1, 1)) for i in range(n_colors)]
                except KeyError:
                    colors = pypalette_colors(palette_name)
                    if colors is None:
                        raise ValueError(f"Palette '{palette_name}' not found") from None
                    colors = colors[:n_colors]
        self.ax.clear()
        for i, color in enumerate(colors):
            self.ax.barh(0, 1, left=i, color=color, edgecolor='white', linewidth=0.5)
        self.ax.set_xlim(0, len(colors))
        self.ax.set_ylim(-0.5, 0.5)
        self.ax.set_yticks([])
        self.ax.set_xticks(range(len(colors)))
        self.ax.set_xticklabels([f"C{i+1}" for i in range(len(colors))])
        self.ax.set_title(f"Palette Preview: {palette_name}")
        for i, color in enumerate(colors):
            color_text = color if isinstance(color, str) else mcolors.to_hex(color)
            self.ax.text(i + 0.5, 0, color_text, ha='center', va='center',
                         rotation=90, fontsize=8, color='white', weight='bold')
        return self

    def generate_gradient_palette(self, name: str, start_color: str, end_color: str,
                                  n_colors: int = 10, save_palette: bool = True) -> 'Chart':
        """
        EN: Generate a smooth linear gradient palette between two colors
            and optionally register it under the given name.
        FR: Génère une palette en dégradé linéaire entre deux couleurs
            et l'enregistre optionnellement sous le nom donné.

        Args / Paramètres:
            name (str):
                EN: Identifier for the palette.
                FR: Identifiant de la palette.
            start_color (str):
                EN: Starting color (any matplotlib-compatible color string).
                FR: Couleur de départ (toute chaîne de couleur compatible matplotlib).
            end_color (str):
                EN: Ending color.
                FR: Couleur d'arrivée.
            n_colors (int):
                EN: Number of steps in the gradient. Default 10.
                FR: Nombre d'étapes dans le dégradé. Défaut 10.
            save_palette (bool):
                EN: Register the palette for later reuse. Default True.
                FR: Enregistre la palette pour une réutilisation ultérieure. Défaut True.

        Returns / Retourne:
            Chart: self

        Example / Exemple:
            chart.generate_gradient_palette('ocean', '#003049', '#fcbf49', n_colors=8)
            chart.set_colors('ocean')
        """
        start_rgb = mcolors.to_rgba(start_color)  # lève ValueError si couleur invalide
        end_rgb = mcolors.to_rgba(end_color)
        colors = []
        for i in range(n_colors):
            ratio = i / max(n_colors - 1, 1)
            r = start_rgb[0] * (1 - ratio) + end_rgb[0] * ratio
            g = start_rgb[1] * (1 - ratio) + end_rgb[1] * ratio
            b = start_rgb[2] * (1 - ratio) + end_rgb[2] * ratio
            a = start_rgb[3] * (1 - ratio) + end_rgb[3] * ratio
            colors.append(mcolors.to_hex((r, g, b, a)))
        if save_palette:
            self.custom_palettes[name] = colors
        self._apply_palette(colors)
        return self

    # ──────────────────────────────────────── Fonts / Polices

    def get_available_fonts(self, pattern: str = None, sort: bool = True) -> List[str]:
        """
        EN: Return all font families installed on the system (instance method version).
            Identical to the module-level get_fonts() function.
        FR: Retourne toutes les familles de polices installées sur le système (méthode d'instance).
            Identique à la fonction de module get_fonts().

        Args / Paramètres:
            pattern (str, optional):
                EN: Case-insensitive substring filter.
                FR: Filtre par sous-chaîne, insensible à la casse.
            sort (bool):
                EN: Sort alphabetically. Default True.
                FR: Trier alphabétiquement. Défaut True.

        Returns / Retourne:
            List[str]: EN: Font family names. / FR: Noms des familles de polices.

        Example / Exemple:
            chart.get_available_fonts(pattern='Calibri')
        """
        fonts = [f.name for f in fm.fontManager.ttflist]
        fonts = list(set(fonts))
        if pattern:
            fonts = [font for font in fonts if pattern.lower() in font.lower()]
        if sort:
            fonts.sort()
        return fonts

    @staticmethod
    def print_available_fonts(pattern: str = None, limit: int = None) -> None:
        """
        EN: Print the list of installed fonts to the console, with optional filtering.
        FR: Affiche la liste des polices installées dans la console, avec filtrage optionnel.

        Args / Paramètres:
            pattern (str, optional):
                EN: Case-insensitive substring filter (e.g. 'Arial').
                FR: Filtre par sous-chaîne, insensible à la casse (ex. 'Arial').
            limit (int, optional):
                EN: Maximum number of fonts to display.
                FR: Nombre maximum de polices à afficher.

        Returns / Retourne:
            None

        Example / Exemple:
            Chart.print_available_fonts(pattern='Times', limit=10)
        """
        temp_chart = Chart(pd.DataFrame({'x': [1], 'y': [1]}))
        fonts = temp_chart.get_available_fonts(pattern)
        if limit:
            fonts = fonts[:limit]
        print(f"Available fonts{f' (filtered by \"{pattern}\")' if pattern else ''}: {len(fonts)} found")
        print("-" * 50)
        for i, font in enumerate(fonts, 1):
            print(f"{i:3d}. {font}")
        full = temp_chart.get_available_fonts(pattern)
        if limit and len(full) > limit:
            print(f"\n... and {len(full) - limit} more fonts")

    # ──────────────────────────────────────── Chart types / Types de graphiques

    def line(self, x: str, y: Union[str, List[str]],
             marker: str = 'o', linewidth: float = 2,
             markersize: float = 6, alpha: float = 1.0) -> 'Chart':
        """
        EN: Draw one or more line series against a common x-axis.
            Multiple y columns produce multiple lines, each labelled automatically.
        FR: Trace une ou plusieurs séries de lignes sur un axe x commun.
            Plusieurs colonnes y produisent plusieurs lignes, chacune étiquetée automatiquement.

        Args / Paramètres:
            x (str):
                EN: Column for the x-axis (dates, categories, numeric).
                FR: Colonne pour l'axe x (dates, catégories, numérique).
            y (Union[str, List[str]]):
                EN: One column name or a list of column names for the y-axis.
                FR: Un nom de colonne ou une liste de noms pour l'axe y.
            marker (str):
                EN: Marker symbol ('o', 's', '^', 'None'…). Default 'o'.
                FR: Symbole de marqueur ('o', 's', '^', 'None'…). Défaut 'o'.
            linewidth (float):
                EN: Stroke width. Default 2.
                FR: Épaisseur du trait. Défaut 2.
            markersize (float):
                EN: Marker diameter. Default 6.
                FR: Diamètre du marqueur. Défaut 6.
            alpha (float):
                EN: Opacity, 0–1. Default 1.0.
                FR: Opacité, 0–1. Défaut 1.0.

        Returns / Retourne:
            Chart: self

        Example / Exemple:
            chart.line('date', ['ventes', 'profit'], marker='s', linewidth=1.5)
        """
        self.chart_type = 'line'
        if isinstance(y, str):
            y = [y]
        for col in y:
            self.ax.plot(self.data[x], self.data[col], marker=marker,
                         linewidth=linewidth, markersize=markersize,
                         alpha=alpha, label=col)
        self._concise_dates(x)
        return self

    def bar(self, x: str, y: Union[str, List[str]], horizontal: bool = False,
            width: float = 0.8, alpha: float = 0.8,
            color: Optional[str] = None, stacked: bool = False,
            show_values: bool = False, value_format: str = '{:,.0f}') -> 'Chart':
        """
        EN: Draw a bar chart, vertical (default) or horizontal.
        FR: Trace un graphique en barres, vertical (défaut) ou horizontal.

        Args / Paramètres:
            x (str):
                EN: Column for bar labels (vertical) or values (horizontal).
                FR: Colonne des étiquettes (vertical) ou des valeurs (horizontal).
            y (str):
                EN: Column for bar heights (vertical) or labels (horizontal).
                FR: Colonne des hauteurs (vertical) ou des étiquettes (horizontal).
            horizontal (bool):
                EN: Draw bars horizontally (barh). Default False.
                FR: Tracer les barres horizontalement (barh). Défaut False.
            width (float):
                EN: Bar width fraction, 0–1. Default 0.8.
                FR: Largeur des barres en fraction, 0–1. Défaut 0.8.
            alpha (float):
                EN: Opacity, 0–1. Default 0.8.
                FR: Opacité, 0–1. Défaut 0.8.
            color (str, optional):
                EN: Bar fill color.
                FR: Couleur de remplissage des barres.

        Returns / Retourne:
            Chart: self

            stacked (bool):
                EN: With several y columns: stack them instead of grouping side by side.
                FR: Avec plusieurs colonnes y : les empiler au lieu de les grouper.
            show_values (bool):
                EN: Write each value on its bar. / FR: Écrit la valeur sur chaque barre.
            value_format (str):
                EN/FR: Format des valeurs affichées. Défaut '{:,.0f}'.

        Example / Exemple:
            chart.bar('categorie', 'ventes', color='steelblue', show_values=True)
            chart.bar('categorie', 'ventes', horizontal=True)
            chart.bar('annee', ['usa', 'chine', 'japon'], stacked=True)   # barres empilées
            chart.bar('mois', ['usa', 'chine'])                           # barres groupées
        """
        self.chart_type = 'bar'
        ys = [y] if isinstance(y, str) else list(y)
        draw = self.ax.barh if horizontal else self.ax.bar
        if len(ys) == 1:  # une série : positions d'origine (catégories, nombres, dates)
            bars = draw(self.data[x], self.data[ys[0]], width, alpha=alpha,
                        color=color or self._palette()[0])
            if show_values:
                self.ax.bar_label(bars, fmt=value_format, padding=3, fontsize=9)
            return self
        pos = np.arange(len(self.data))
        bottom = np.zeros(len(self.data))
        for j, (col, c) in enumerate(zip(ys, self._palette() * 10)):
            vals = self.data[col].to_numpy(dtype=float)
            if stacked:
                offs, w = pos, width
                extra = {'left' if horizontal else 'bottom': bottom.copy()}
            else:
                w = width / len(ys)
                offs, extra = pos - width / 2 + w * (j + 0.5), {}
            bars = draw(offs, vals, w, alpha=alpha, color=c, label=col, **extra)
            if show_values:
                self.ax.bar_label(bars, fmt=value_format, fontsize=8,
                                  label_type='center' if stacked else 'edge',
                                  padding=0 if stacked else 2)
            bottom += np.nan_to_num(vals)
        (self.ax.set_yticks if horizontal else self.ax.set_xticks)(
            pos, self.data[x].astype(str).tolist())
        self.ax.legend()
        return self

    def scatter(self, x: str, y: str, size: Union[str, float] = 50,
                color: Optional[str] = None, alpha: float = 0.7,
                marker: str = 'o') -> 'Chart':
        """
        EN: Draw a scatter plot with optional bubble sizing and color encoding.
        FR: Trace un nuage de points avec dimensionnement et encodage couleur optionnels.

        Args / Paramètres:
            x (str):
                EN: Column for x-axis positions.
                FR: Colonne pour les positions sur l'axe x.
            y (str):
                EN: Column for y-axis positions.
                FR: Colonne pour les positions sur l'axe y.
            size (Union[str, float]):
                EN: Column name to size bubbles by, or a fixed numeric size. Default 50.
                FR: Nom de colonne pour dimensionner les bulles, ou taille numérique fixe. Défaut 50.
            color (str, optional):
                EN: Column name for color encoding, or a fixed color string.
                FR: Nom de colonne pour l'encodage couleur, ou chaîne de couleur fixe.
            alpha (float):
                EN: Opacity, 0–1. Default 0.7.
                FR: Opacité, 0–1. Défaut 0.7.
            marker (str):
                EN: Marker symbol. Default 'o'.
                FR: Symbole de marqueur. Défaut 'o'.

        Returns / Retourne:
            Chart: self

        Example / Exemple:
            chart.scatter('temperature', 'visiteurs', size='ventes', alpha=0.6)
        """
        self.chart_type = 'scatter'
        sizes = self.data[size] if isinstance(size, str) else size
        colors = self.data[color] if isinstance(color, str) else color
        self.ax.scatter(self.data[x], self.data[y], s=sizes, c=colors,
                        alpha=alpha, marker=marker)
        return self

    def histogram(self, column: str, bins: Union[int, str] = 30,
                  alpha: float = 0.7, color: str = None,
                  density: bool = False, edgecolor: Optional[str] = None) -> 'Chart':
        """
        EN: Draw a histogram of the distribution of a single column.
        FR: Trace un histogramme de la distribution d'une seule colonne.

        Args / Paramètres:
            column (str):
                EN: Column whose values are binned.
                FR: Colonne dont les valeurs sont regroupées en classes.
            bins (Union[int, str]):
                EN: Number of bins or binning strategy ('auto', 'fd', 'scott'…). Default 30.
                FR: Nombre de classes ou stratégie de regroupement. Défaut 30.
            alpha (float):
                EN: Opacity. Default 0.7.
                FR: Opacité. Défaut 0.7.
            color (str, optional):
                EN: Bar fill color.
                FR: Couleur de remplissage.
            density (bool):
                EN: Normalize so the area sums to 1. Default False.
                FR: Normalise pour que l'aire totale vaille 1. Défaut False.
            edgecolor (str):
                EN: Bar border color. Default: theme.
                FR: Couleur des bordures de barres. Défaut : thème.

        Returns / Retourne:
            Chart: self

        Example / Exemple:
            chart.histogram('salaire', bins=50, density=True, color='steelblue')
        """
        self.chart_type = 'histogram'
        self.ax.hist(self.data[column], bins=bins, alpha=alpha, color=color,
                     density=density, edgecolor=edgecolor)
        return self

    def boxplot(self, columns: Union[str, List[str]],
                vert: bool = True, patch_artist: bool = True) -> 'Chart':
        """
        EN: Draw a box-and-whisker plot for one or more numeric columns.
            Shows median, IQR, and outliers.
        FR: Trace un diagramme en boîte pour une ou plusieurs colonnes numériques.
            Affiche la médiane, l'IQR et les valeurs aberrantes.

        Args / Paramètres:
            columns (Union[str, List[str]]):
                EN: Column name or list of column names to plot.
                FR: Nom de colonne ou liste de noms de colonnes à tracer.
            vert (bool):
                EN: Vertical orientation. Default True.
                FR: Orientation verticale. Défaut True.
            patch_artist (bool):
                EN: Fill boxes with color. Default True.
                FR: Remplir les boîtes avec une couleur. Défaut True.

        Returns / Retourne:
            Chart: self

        Example / Exemple:
            chart.boxplot(['ventes', 'profit', 'coût'], vert=False)
        """
        self.chart_type = 'boxplot'
        if isinstance(columns, str):
            columns = [columns]
        data_to_plot = [self.data[col].dropna() for col in columns]
        orientation = 'vertical' if vert else 'horizontal'
        self.ax.boxplot(data_to_plot, tick_labels=columns,
                        orientation=orientation, patch_artist=patch_artist)
        return self

    def pie(self, column: str, labels: Optional[str] = None,
            autopct: str = '%1.1f%%', startangle: float = 90,
            explode: Optional[List[float]] = None) -> 'Chart':
        """
        EN: Draw a pie chart showing proportional composition.
        FR: Trace un graphique circulaire (camembert) montrant la composition proportionnelle.

        Args / Paramètres:
            column (str):
                EN: Column with numeric slice sizes.
                FR: Colonne contenant les tailles de secteurs (numériques).
            labels (str, optional):
                EN: Column to use as slice labels. Falls back to DataFrame index.
                FR: Colonne utilisée comme étiquettes. Utilise l'index par défaut.
            autopct (str):
                EN: Printf-style format string for percentage labels. Default '%1.1f%%'.
                FR: Format de chaîne de style printf pour les pourcentages. Défaut '%1.1f%%'.
            startangle (float):
                EN: Angle in degrees for the first slice. Default 90 (12 o'clock).
                FR: Angle en degrés pour le premier secteur. Défaut 90 (12 heures).
            explode (List[float], optional):
                EN: Fractional offset to pull each slice outward (0 = no offset).
                FR: Décalage fractionnaire pour extraire chaque secteur (0 = pas de décalage).

        Returns / Retourne:
            Chart: self

        Example / Exemple:
            chart.pie('part', labels='region', explode=[0.05, 0, 0, 0])
        """
        self.chart_type = 'pie'
        labels_data = self.data[labels] if labels else self.data.index
        self.ax.pie(self.data[column], labels=labels_data, autopct=autopct,
                    startangle=startangle, explode=explode)
        self.ax.axis('equal')
        return self

    def heatmap(self, annot: bool = True, cmap: Optional[str] = None,
                fmt: str = '.2f', cbar: bool = True) -> 'Chart':
        """
        EN: Draw a correlation heatmap from all numeric columns in the DataFrame.
            Cells are coloured by Pearson correlation coefficient (–1 to +1).
        FR: Trace une carte de chaleur des corrélations à partir de toutes les colonnes
            numériques du DataFrame. Les cellules sont colorées selon le coefficient de
            corrélation de Pearson (–1 à +1).

        Args / Paramètres:
            annot (bool):
                EN: Annotate each cell with its numeric value. Default True.
                FR: Annoter chaque cellule avec sa valeur numérique. Défaut True.
            cmap (str):
                EN: Colormap name. Default 'viridis'.
                FR: Nom du colormap. Défaut 'viridis'.
            fmt (str):
                EN: Python format string for cell values. Default '.2f'.
                FR: Chaîne de format pour les valeurs des cellules. Défaut '.2f'.
            cbar (bool):
                EN: Show the colorbar scale. Default True.
                FR: Afficher l'échelle de couleur. Défaut True.

        Returns / Retourne:
            Chart: self

        Example / Exemple:
            chart.heatmap(cmap='coolwarm', fmt='.1f')
        """
        self.chart_type = 'heatmap'
        corr = self.data.select_dtypes(include=[np.number]).corr()
        # Défaut : échelle divergente tirée du thème (2e couleur ← fond → 1re couleur), centrée sur 0
        im = self.ax.imshow(corr, cmap=cmap or self._diverging_cmap(), vmin=-1, vmax=1,
                            aspect='auto')
        if annot:
            for i in range(len(corr)):
                for j in range(len(corr)):
                    value = corr.iloc[i, j]
                    self.ax.text(j, i, format(value, fmt), ha='center', va='center',
                                 fontsize=9, color=self._ink_on(im.cmap(im.norm(value))))
        self.ax.set_xticks(range(len(corr)), corr.columns, rotation=30, ha='right')
        self.ax.set_yticks(range(len(corr)), corr.columns)
        self.ax.grid(False)
        if cbar:
            self._colorbar(im, 'Corrélation (r)')
        return self

    def area(self, x: str, y: Union[str, List[str]],
             alpha: float = 0.7, stacked: bool = False) -> 'Chart':
        """
        EN: Draw an area chart — optionally stacked.
            Stacked mode uses ax.stackplot; non-stacked fills each series independently.
        FR: Trace un graphique en aire — optionnellement empilé.
            Le mode empilé utilise ax.stackplot; sinon chaque série est remplie indépendamment.

        Args / Paramètres:
            x (str):
                EN: Column for the x-axis.
                FR: Colonne pour l'axe x.
            y (Union[str, List[str]]):
                EN: One or more columns to fill.
                FR: Une ou plusieurs colonnes à remplir.
            alpha (float):
                EN: Fill opacity. Default 0.7.
                FR: Opacité du remplissage. Défaut 0.7.
            stacked (bool):
                EN: Stack areas cumulatively. Default False.
                FR: Empiler les aires cumulativement. Défaut False.

        Returns / Retourne:
            Chart: self

        Example / Exemple:
            chart.area('année', ['rock', 'pop', 'jazz'], stacked=True)
        """
        self.chart_type = 'area'
        if isinstance(y, str):
            y = [y]
        if stacked:
            self.ax.stackplot(self.data[x], *[self.data[col] for col in y],
                              labels=y, alpha=alpha)
        else:
            for col in y:
                self.ax.fill_between(self.data[x], self.data[col], alpha=alpha, label=col)
        self._concise_dates(x)
        return self

    def violin(self, columns: Union[str, List[str]]) -> 'Chart':
        """
        EN: Draw violin plots showing the KDE-estimated distribution of one or more columns.
            Wider sections indicate higher data density.
        FR: Trace des graphiques en violon montrant la distribution estimée par KDE
            d'une ou plusieurs colonnes. Les sections plus larges indiquent une densité plus élevée.

        Args / Paramètres:
            columns (Union[str, List[str]]):
                EN: Column name or list of column names.
                FR: Nom de colonne ou liste de noms de colonnes.

        Returns / Retourne:
            Chart: self

        Example / Exemple:
            chart.violin(['notes_maths', 'notes_sciences', 'notes_arts'])
        """
        self.chart_type = 'violin'
        if isinstance(columns, str):
            columns = [columns]
        data_to_plot = [self.data[col].dropna() for col in columns]
        self.ax.violinplot(data_to_plot, positions=range(1, len(columns) + 1))
        self.ax.set_xticks(range(1, len(columns) + 1))
        self.ax.set_xticklabels(columns)
        return self

    def lollipop(self, x: str, y: str, group: str,
                 colors: Dict[str, str], order: Optional[List[str]] = None,
                 shift_step: float = 0.2) -> 'Chart':
        """
        EN: Draw a grouped lollipop (bubble + stem) chart.
            Aggregates y (sum) per (x, group) pair, then draws a sized bubble
            and a dashed vertical stem for each group, offset horizontally
            so groups side-by-side don't overlap.
        FR: Trace un graphique lollipop groupé (bulle + tige).
            Agrège y (somme) par paire (x, groupe), puis trace une bulle dimensionnée
            et une tige verticale pointillée par groupe, décalées horizontalement
            pour que les groupes adjacents ne se chevauchent pas.

        Args / Paramètres:
            x (str):
                EN: Column for the x-axis (e.g. year).
                FR: Colonne pour l'axe x (ex. année).
            y (str):
                EN: Column to aggregate by sum (used for bubble height and size).
                FR: Colonne à agréger par somme (utilisée pour la hauteur et la taille de la bulle).
            group (str):
                EN: Column defining the groups (one bubble series per value).
                FR: Colonne définissant les groupes (une série de bulles par valeur).
            colors (Dict[str, str]):
                EN: Mapping {group_value: color_string}.
                FR: Correspondance {valeur_groupe: chaîne_couleur}.
            order (List[str], optional):
                EN: Display order of group values.
                FR: Ordre d'affichage des valeurs de groupe.
            shift_step (float):
                EN: Horizontal offset between consecutive groups. Default 0.2.
                FR: Décalage horizontal entre groupes consécutifs. Défaut 0.2.

        Returns / Retourne:
            Chart: self

        Example / Exemple:
            couleurs = {'hurricane': '#fcbf49', 'tropical storm': '#d62828'}
            chart.lollipop('year', 'n', 'status', colors=couleurs,
                           order=['tropical storm', 'hurricane'])
        """
        self.chart_type = 'lollipop'
        x_vals = np.sort(self.data[x].unique())
        groups = [g for g in order if g in self.data[group].unique()] \
            if order else list(self.data[group].unique())
        shift_start = -(len(groups) - 1) * shift_step / 2
        for i, g in enumerate(groups):
            offset = shift_start + i * shift_step
            counts = [self.data[(self.data[x] == xv) & (self.data[group] == g)][y].sum()
                      for xv in x_vals]
            counts = np.array(counts)
            color = colors.get(g)
            self.ax.scatter(x_vals + offset, counts, s=counts, color=color, label=g)
            self.ax.vlines(x_vals + offset, ymin=0, ymax=counts,
                           linestyle='--', linewidth=1, color=color)
        return self

    def ombrothermic(self, month_col: str, temperature_col: str, precipitation_col: str,
                     temp_color: str = 'tab:red', precip_color: str = 'tab:blue',
                     dry_period_color: str = 'red', dry_period_alpha: float = 0.2,
                     show_threshold_line: bool = True, temp_scale_factor: float = 2.0,
                     kind: str = 'bars', hatch: str = '///',
                     humid_fill: bool = True) -> 'Chart':
        """
        EN: Draw an ombrothermic (Bagnouls-Gaussen) diagram showing monthly temperature
            and precipitation on dual axes with the p = 2T scale convention.
            Dry periods (precipitation < 2 × temperature) are highlighted.
        FR: Trace un diagramme ombrothermique (Bagnouls et Gaussen) montrant la température
            mensuelle et les précipitations sur deux axes avec la convention d'échelle p = 2T.
            Les périodes sèches (précipitations < 2 × température) sont mises en évidence.

        Args / Paramètres:
            month_col (str):
                EN: Column with month labels (x-axis).
                FR: Colonne avec les étiquettes de mois (axe x).
            temperature_col (str):
                EN: Column with mean monthly temperature (°C).
                FR: Colonne avec la température mensuelle moyenne (°C).
            precipitation_col (str):
                EN: Column with total monthly precipitation (mm).
                FR: Colonne avec les précipitations mensuelles totales (mm).
            temp_color (str):
                EN: Color for the temperature line. Default 'tab:red'.
                FR: Couleur de la courbe de température. Défaut 'tab:red'.
            precip_color (str):
                EN: Color for the precipitation bars. Default 'tab:blue'.
                FR: Couleur des barres de précipitations. Défaut 'tab:blue'.
            dry_period_color (str):
                EN: Highlight color for dry-period zones. Default 'red'.
                FR: Couleur de surbrillance des zones sèches. Défaut 'red'.
            dry_period_alpha (float):
                EN: Opacity of dry-period zones. Default 0.2.
                FR: Opacité des zones sèches. Défaut 0.2.
            show_threshold_line (bool):
                EN: Draw the p = 2T reference line. Default True.
                FR: Afficher la ligne de référence p = 2T. Défaut True.
            temp_scale_factor (float):
                EN: Precipitation-to-temperature scale ratio (default 2.0 for p = 2T).
                FR: Rapport d'échelle précipitations/température (défaut 2.0 pour p = 2T).

        Returns / Retourne:
            Chart: self

            kind (str):
                EN: 'bars' (precipitation bars, default) or 'lines' (two curves, dry periods
                    hatched between the curves where P < factor × T).
                FR: 'bars' (barres de précipitations, défaut) ou 'lines' (deux courbes,
                    périodes sèches hachurées entre les courbes là où P < facteur × T).
            hatch (str):
                EN/FR: Motif des hachures (kind='lines') : '///', '\\\\\\\\', 'xx', '...'.
            humid_fill (bool):
                EN/FR: kind='lines' : colore légèrement la période humide. Défaut True.

        Example / Exemple:
            chart.ombrothermic('Mois', 'Température (°C)', 'Précipitations (mm)')
            chart.ombrothermic('Mois', 'Température (°C)', 'Précipitations (mm)', kind='lines')
        """
        if kind == 'lines':
            return self._ombrothermic_lines(month_col, temperature_col, precipitation_col,
                                            temp_color, precip_color, dry_period_color,
                                            temp_scale_factor, hatch, humid_fill)
        if kind != 'bars':
            raise ValueError("kind doit valoir 'bars' ou 'lines'")
        self.chart_type = 'ombrothermic'
        self.ax.clear()
        self.ax.plot(self.data[month_col], self.data[temperature_col],
                     color=temp_color, marker='o', linewidth=2,
                     label='Température', markersize=6)
        self.ax.set_ylabel(
            f'Température ({temperature_col.split("(")[-1].replace(")", "")})',
            color=temp_color)
        self.ax.tick_params(axis='y', labelcolor=temp_color)
        ax2 = self.ax.twinx()
        # Échelle de Gaussen : axe P = facteur × axe T, les deux partant de 0
        # (ou du minimum si températures négatives) → la courbe T coïncide avec le seuil.
        temps = self.data[temperature_col]
        t_low = min(0, temps.min())
        p_high = max(self.data[precipitation_col].max(), temp_scale_factor * temps.max()) * 1.1
        self.ax.set_ylim(t_low, p_high / temp_scale_factor)
        ax2.set_ylim(t_low * temp_scale_factor, p_high)
        # Courbe de température au-dessus des barres de précipitations
        self.ax.set_zorder(ax2.get_zorder() + 1)
        self.ax.patch.set_visible(False)
        ax2.bar(self.data[month_col], self.data[precipitation_col],
                color=precip_color, alpha=0.6, label='Précipitations')
        ax2.set_ylabel(
            f'Précipitations ({precipitation_col.split("(")[-1].replace(")", "")})',
            color=precip_color)
        ax2.tick_params(axis='y', labelcolor=precip_color)
        x_pos = range(len(self.data[month_col]))
        temp_threshold = [temp_scale_factor * t for t in self.data[temperature_col]]
        dry_periods = []
        for i in range(len(self.data)):
            precip = self.data[precipitation_col].iloc[i]
            threshold = temp_threshold[i]
            if precip < threshold:
                dry_periods.append(i)
                ax2.fill_between([i - 0.4, i + 0.4],
                                 [precip, precip], [threshold, threshold],
                                 color=dry_period_color, alpha=dry_period_alpha,
                                 label='Période sèche' if len(dry_periods) == 1 else '')
        if show_threshold_line:
            ax2.plot(x_pos, temp_threshold, color='red', linestyle='--',
                     alpha=0.7, linewidth=1, label='p=2T (seuil de sécheresse)')
        self.ax.set_xticks(range(len(self.data[month_col])))
        self.ax.set_xticklabels(self.data[month_col], rotation=45, ha='right')
        lines1, labels1 = self.ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        self.ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        self.ax2 = ax2
        return self

    def _ombrothermic_lines(self, month_col, temperature_col, precipitation_col, temp_color,
                            precip_color, dry_color, factor, hatch, humid_fill) -> 'Chart':
        """Diagramme ombrothermique en deux courbes, périodes sèches hachurées (P < facteur × T)."""
        self.chart_type = 'ombrothermic'
        d = self.data
        n = len(d)
        pos = np.arange(n)
        T = d[temperature_col].to_numpy(float)
        P = d[precipitation_col].to_numpy(float)
        Ps = P / factor                                  # pluie sur l'échelle des températures
        # Interpolation fine : les hachures s'arrêtent exactement aux croisements des courbes
        xf = np.linspace(0, n - 1, 600)
        Tf, Pf = np.interp(xf, pos, T), np.interp(xf, pos, Ps)
        self.ax.fill_between(xf, Pf, Tf, where=Tf > Pf, interpolate=True, facecolor='none',
                             edgecolor=dry_color, hatch=hatch, linewidth=0, label='Période sèche')
        if humid_fill:
            self.ax.fill_between(xf, Tf, Pf, where=Pf >= Tf, interpolate=True, color=precip_color,
                                 alpha=0.12, linewidth=0, label='Période humide')
        self.ax.plot(pos, T, color=temp_color, marker='o', markersize=5, linewidth=2.2,
                     label='Température')
        self.ax.plot(pos, Ps, color=precip_color, marker='s', markersize=5, linewidth=2.2,
                     label=f'Précipitations (P = {factor:g}T)')
        low = min(0.0, T.min(), Ps.min())
        high = max(T.max(), Ps.max()) * 1.12
        self.ax.set_ylim(low, high)
        self.ax.set_xlim(-0.3, n - 0.7)
        self.ax.set_xticks(pos, d[month_col].astype(str).tolist())
        self.ax.set_ylabel(f'Température ({temperature_col.split("(")[-1].replace(")", "")})',
                           color=temp_color)
        ax2 = self.ax.twinx()                            # même courbe lue en mm : facteur × °C
        ax2.set_ylim(low * factor, high * factor)
        ax2.set_ylabel(f'Précipitations ({precipitation_col.split("(")[-1].replace(")", "")})',
                       color=precip_color)
        ax2.grid(False)
        ax2.spines['right'].set_visible(True)
        self.ax.legend(loc='upper left', fontsize=8)
        self.ax2 = ax2
        self.dry_months = d.loc[P < factor * T, month_col].tolist()
        return self

    def print_ombrothermic_analysis(self, month_col: str, temperature_col: str,
                                    precipitation_col: str,
                                    temp_scale_factor: float = 2.0) -> 'Chart':
        """
        EN: Print a textual month-by-month analysis of dry vs. wet periods
            based on the p = 2T criterion, followed by a summary count.
        FR: Affiche une analyse textuelle mois par mois des périodes sèches et humides
            selon le critère p = 2T, suivie d'un résumé du nombre de mois.

        Args / Paramètres:
            month_col (str):
                EN: Column with month labels.
                FR: Colonne avec les étiquettes de mois.
            temperature_col (str):
                EN: Column with mean monthly temperature (°C).
                FR: Colonne avec la température mensuelle moyenne (°C).
            precipitation_col (str):
                EN: Column with total monthly precipitation (mm).
                FR: Colonne avec les précipitations mensuelles totales (mm).
            temp_scale_factor (float):
                EN: Scale factor for the dryness threshold (default 2.0 → p < 2T = dry).
                FR: Facteur d'échelle pour le seuil de sécheresse (défaut 2.0 → p < 2T = sec).

        Returns / Retourne:
            Chart: self  (EN: allows chaining. / FR: permet le chaînage.)

        Example / Exemple:
            (Chart(df)
                .ombrothermic('Mois', 'T (°C)', 'P (mm)')
                .print_ombrothermic_analysis('Mois', 'T (°C)', 'P (mm)')
                .tight_layout()
                .show())
        """
        print("Analyse des périodes sèches et humides:")
        print("=" * 40)
        for i in range(len(self.data)):
            month = self.data[month_col].iloc[i]
            temp = self.data[temperature_col].iloc[i]
            precip = self.data[precipitation_col].iloc[i]
            threshold = temp_scale_factor * temp
            if precip < threshold:
                print(f"{month}: PÉRIODE SÈCHE - Précip: {precip:.1f}mm < Seuil: {threshold:.1f}mm")
            else:
                print(f"{month}: Période humide - Précip: {precip:.1f}mm ≥ Seuil: {threshold:.1f}mm")
        dry_months = sum(
            1 for i in range(len(self.data))
            if self.data[precipitation_col].iloc[i]
            < temp_scale_factor * self.data[temperature_col].iloc[i]
        )
        print(f"\nRésumé: {dry_months} mois secs sur {len(self.data)} mois")
        return self

    def radar(self, columns: List[str], group: Optional[str] = None,
              alpha: float = 0.25) -> 'Chart':
        """
        EN: Draw a spider/radar chart to compare multivariate profiles.
            Each column becomes one spoke; values are plotted on a polar axis.
            When a group column is provided, one filled polygon is drawn per group
            using the mean of each metric within that group.
        FR: Trace un graphique radar (toile d'araignée) pour comparer des profils multivariés.
            Chaque colonne devient un rayon; les valeurs sont tracées sur un axe polaire.
            Quand une colonne de groupe est fournie, un polygone rempli est tracé par groupe
            en utilisant la moyenne de chaque métrique au sein du groupe.

        Args / Paramètres:
            columns (List[str]):
                EN: Numeric columns to use as radar spokes.
                FR: Colonnes numériques utilisées comme rayons du radar.
            group (str, optional):
                EN: Column for grouping rows into separate polygons.
                FR: Colonne pour regrouper les lignes en polygones séparés.
            alpha (float):
                EN: Fill opacity. Default 0.25.
                FR: Opacité du remplissage. Défaut 0.25.

        Returns / Retourne:
            Chart: self

        Example / Exemple:
            chart.radar(['vitesse', 'endurance', 'force', 'agilité'], group='equipe')
        """
        self.chart_type = 'radar'
        self.fig.clear()
        self.ax = self.fig.add_subplot(111, polar=True)
        n = len(columns)
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
        angles += angles[:1]
        groups = self.data[group].unique() if group else [None]
        for g in groups:
            subset = self.data[self.data[group] == g] if g is not None else self.data
            values = subset[columns].mean().tolist()
            values += values[:1]
            self.ax.plot(angles, values, linewidth=2,
                         label=str(g) if g is not None else None)
            self.ax.fill(angles, values, alpha=alpha)
        self.ax.set_xticks(angles[:-1])
        self.ax.set_xticklabels(columns)
        return self

    def waterfall(self, x: str, y: str,
                  positive_color: Optional[str] = None,
                  negative_color: Optional[str] = None) -> 'Chart':
        """
        EN: Draw a waterfall chart showing the cumulative effect of sequential values.
            Bars are stacked from the running total: positive values extend upward,
            negative values extend downward. A zero baseline is drawn for reference.
        FR: Trace un graphique en cascade montrant l'effet cumulatif de valeurs séquentielles.
            Les barres s'empilent à partir du total courant : les valeurs positives s'étendent
            vers le haut, les valeurs négatives vers le bas. Une ligne de base zéro est tracée.

        Args / Paramètres:
            x (str):
                EN: Column with category or period labels.
                FR: Colonne avec les étiquettes de catégories ou de périodes.
            y (str):
                EN: Column with incremental values (positive or negative).
                FR: Colonne avec les valeurs incrémentielles (positives ou négatives).
            positive_color (str):
                EN: Bar color for positive increments. Default 'steelblue'.
                FR: Couleur des barres pour les incréments positifs. Défaut 'steelblue'.
            negative_color (str):
                EN: Bar color for negative increments. Default 'salmon'.
                FR: Couleur des barres pour les incréments négatifs. Défaut 'salmon'.

        Returns / Retourne:
            Chart: self

        Example / Exemple:
            chart.waterfall('mois', 'delta_revenu', positive_color='green',
                            negative_color='crimson')
        """
        self.chart_type = 'waterfall'
        palette = self._palette()  # défaut : 1re couleur du thème = hausse, 2e = baisse
        positive_color = positive_color or palette[0]
        negative_color = negative_color or palette[1 % len(palette)]
        values = self.data[y].tolist()
        running = 0
        bottoms, heights, colors = [], [], []
        for v in values:
            bottoms.append(running if v >= 0 else running + v)
            heights.append(abs(v))
            colors.append(positive_color if v >= 0 else negative_color)
            running += v
        self.ax.bar(self.data[x].tolist(), heights, bottom=bottoms, color=colors)
        self.ax.axhline(0, color=plt.rcParams['axes.labelcolor'], linewidth=0.8)
        return self

    def bump(self, x: str, y: str, group: str, linewidth: float = 2.5, markersize: float = 8,
             values: bool = False, curve_force: float = 0.5,
             highlight: Optional[List] = None, end_labels: bool = True,
             ordinal: bool = False) -> 'Chart':
        """
        EN: Bump chart (bumplot): evolution of rankings over time with smooth Bézier curves;
            rank 1 at the top. `y` holds ranks (1 = best), or raw values with values=True
            (highest value = rank 1, computed at each x). `highlight` emphasises some groups
            (others greyed), `end_labels` writes the names at the end of the lines.
            Ranks in `chart.ranks` (one row per x, one column per group).
        FR: Graphique de classement (bump chart, avec bumplot) : évolution des rangs dans le
            temps en courbes de Bézier lissées ; rang 1 en haut. `y` contient des rangs
            (1 = premier), ou des valeurs brutes avec values=True (plus forte valeur = rang 1,
            calculé à chaque x). `highlight` met en avant certains groupes (les autres en
            gris), `end_labels` écrit les noms au bout des lignes. Rangs dans `chart.ranks`
            (une ligne par x, une colonne par groupe).

        Args / Paramètres:
            x (str): EN/FR: Colonne des périodes (années, saisons, journées…).
            y (str): EN/FR: Colonne des rangs (1 = premier), ou des valeurs avec values=True.
            group (str): EN/FR: Colonne des éléments classés (une ligne par élément).
            linewidth (float): EN/FR: Épaisseur des courbes.
            markersize (float): EN/FR: Diamètre des points (points).
            values (bool): EN/FR: `y` contient des valeurs : le rang est calculé à chaque x (plus forte = 1).
            curve_force (float): EN/FR: Courbure des liaisons, de 0 (segments droits) à 1 (paliers plats et transitions raides) ; 0.5 = sigmoïde douce.
            highlight (list, optional): EN/FR: Groupes mis en couleur ; les autres passent en gris clair.
            end_labels (bool): EN/FR: Écrit le nom de chaque groupe au bout de sa ligne (à la place de la légende).
            ordinal (bool): EN/FR: Rangs écrits « 1er, 2e, 3e… » sur l'axe vertical.

        Returns / Retourne:
            Chart: self

        Example / Exemple:
            chart.bump('saison', 'classement', 'equipe')
            chart.bump('annee', 'production', 'pays', values=True, highlight=['Sénégal'])
        """
        from bumplot import bumplot
        if not 0 <= curve_force <= 1:
            raise ValueError('curve_force doit être compris entre 0 et 1 (au-delà, les courbes font des boucles)')
        self.chart_type = 'bump'
        wide = self.data.pivot_table(index=x, columns=group, values=y, aggfunc='first', sort=True)
        if wide.isna().any().any():
            missing = wide.columns[wide.isna().any()].tolist()
            raise ValueError(f'Valeur manquante pour certaines périodes : {missing} '
                             '(un bump chart demande une valeur par groupe et par période)')
        names = [str(c) for c in wide.columns]
        wide.columns = names
        # bumplot classe par valeurs décroissantes : pour des rangs (1 = premier), on inverse
        table = (wide if values else -wide).reset_index()
        palette = self._palette()
        muted = mcolors.to_hex(mcolors.to_rgba(plt.rcParams['axes.labelcolor'], 0.35), keep_alpha=True)
        if highlight is not None:
            unknown = set(map(str, highlight)) - set(names)
            if unknown:
                raise ValueError(f'highlight : groupes inconnus {sorted(unknown)}')
            focus = [n for n in names if n in set(map(str, highlight))]
            colors = [palette[focus.index(n) % len(palette)] if n in focus else muted for n in names]
            columns = [(n, {'zorder': 3 if n in focus else 2}) for n in names]
        else:
            focus, colors, columns = names, [palette[k % len(palette)] for k in range(len(names))], names
        _, artists = bumplot(x, columns, table, curve_force=curve_force, colors=colors, ax=self.ax,
                             plot_kwargs={'linewidth': linewidth},
                             scatter_kwargs={'s': markersize ** 2, 'zorder': 4,
                                             'edgecolor': plt.rcParams['axes.facecolor'],
                                             'linewidth': 1.2})
        self.ranks = wide.rank(axis=1, ascending=False, method='first').astype(int) if values \
            else wide.astype(int)
        if ordinal:
            ticks = self.ax.get_yticks()
            self.ax.set_yticks(ticks, [f'{int(t)}{"er" if int(t) == 1 else "e"}' for t in ticks])
        self.ax.grid(False)
        if end_labels:
            ink = plt.rcParams['axes.labelcolor']
            for name, color in zip(names, colors):
                px, py = artists[name][1].get_offsets()[-1]
                self.ax.annotate(name, (px, py), xytext=(10, 0), textcoords='offset points',
                                 va='center', fontsize=9, color=color if name in focus else ink,
                                 fontweight='bold' if highlight is not None and name in focus
                                 else 'normal', annotation_clip=False)
            self.ax.margins(x=0.04)
            for side in ('top', 'right'):
                self.ax.spines[side].set_visible(False)
        else:
            self.ax.legend(title=group, loc='center left', bbox_to_anchor=(1.01, 0.5))
        return self

    def ridgeline(self, value_col: str, group_col: str,
                  overlap: float = 1.5, alpha: float = 0.7) -> 'Chart':
        """
        EN: Draw a ridgeline (Joy) plot — overlapping KDE distributions, one per group.
            Groups are stacked vertically with a configurable overlap. Requires scipy.
        FR: Trace un graphique à crêtes (Joy plot) — distributions KDE superposées, une par groupe.
            Les groupes sont empilés verticalement avec un chevauchement configurable. Nécessite scipy.

        Args / Paramètres:
            value_col (str):
                EN: Numeric column whose distribution is estimated per group.
                FR: Colonne numérique dont la distribution est estimée par groupe.
            group_col (str):
                EN: Categorical column defining groups (one ridge per unique value).
                FR: Colonne catégorielle définissant les groupes (une crête par valeur unique).
            overlap (float):
                EN: Height of the tallest ridge, in rows (1 = touches the row above).
                    Densities share one scale, whatever the units of value_col. Default 1.5.
                FR: Hauteur de la plus haute crête, en lignes (1 = touche la ligne du dessus).
                    Toutes les densités partagent une échelle, quelles que soient les unités. Défaut 1.5.
            alpha (float):
                EN: Fill opacity. Default 0.7.
                FR: Opacité du remplissage. Défaut 0.7.

        Returns / Retourne:
            Chart: self

        Example / Exemple:
            chart.ridgeline('salaire', 'secteur', overlap=2.0, alpha=0.6)
        """
        from scipy.stats import gaussian_kde
        self.chart_type = 'ridgeline'
        groups = list(self.data[group_col].dropna().unique())
        values = self.data[value_col].dropna()
        pad = 0.05 * (values.max() - values.min())
        x_range = np.linspace(values.min() - pad, values.max() + pad, 300)
        densities = {}
        for g in groups:
            vals = self.data.loc[self.data[group_col] == g, value_col].dropna().values
            if len(vals) >= 2 and np.ptp(vals) > 0:
                densities[g] = gaussian_kde(vals)(x_range)
        if not densities:
            raise ValueError("ridgeline demande au moins un groupe d'au moins 2 valeurs distinctes")
        # Densités brutes en unités de value_col (ex. 4e-5 par € pour des salaires) :
        # même échelle pour toutes, la plus haute crête montant à `overlap` lignes.
        scale = overlap / max(d.max() for d in densities.values())
        palette = self._palette()
        edge = plt.rcParams['axes.facecolor']
        n = len(groups)
        for i, g in enumerate(groups):
            if g not in densities:
                continue
            offset = n - 1 - i                      # premier groupe en haut
            curve = densities[g] * scale + offset
            z = 2 * i                               # chaque crête passe devant celle du dessus
            self.ax.fill_between(x_range, offset, curve, alpha=alpha, linewidth=0,
                                 color=palette[i % len(palette)], label=str(g), zorder=z)
            self.ax.plot(x_range, curve, color=edge, linewidth=1, zorder=z + 1)
        self.ax.set_yticks(np.arange(n)[::-1], [str(g) for g in groups])
        self.ax.set_ylim(-0.2, n - 1 + overlap + 0.2)
        self.ax.set_xlabel(value_col)
        self.ax.grid(False, axis='y')
        return self

    def treemap(self, values: str, labels: Optional[str] = None,
                alpha: float = 0.9, show_values: bool = True,
                value_format: str = '{:,.0f}') -> 'Chart':
        """
        EN: Draw a treemap — nested rectangles sized by a numeric column.
            Requires the ``squarify`` package (installed automatically with uv).
        FR: Trace une carte arborescente (treemap) — rectangles imbriqués dimensionnés
            par une colonne numérique. Nécessite le paquet ``squarify``.

        Args / Paramètres:
            values (str):
                EN: Column with positive numeric sizes (determines rectangle area).
                FR: Colonne avec des tailles numériques positives (détermine l'aire du rectangle).
            labels (str, optional):
                EN: Column for text labels inside each rectangle.
                FR: Colonne pour les étiquettes textuelles à l'intérieur de chaque rectangle.
            alpha (float):
                EN: Fill opacity. Default 0.8.
                FR: Opacité du remplissage. Défaut 0.8.

        Returns / Retourne:
            Chart: self

        Example / Exemple:
            chart.treemap('chiffre_affaires', labels='departement')
        """
        try:
            import squarify
        except ImportError:
            raise ImportError("treemap requires squarify: uv add squarify")
        self.chart_type = 'treemap'
        d = self.data[self.data[values] > 0].sort_values(values, ascending=False)
        names = d[labels].astype(str).tolist() if labels else [''] * len(d)
        W, H = 100.0, 100.0
        rects = squarify.squarify(squarify.normalize_sizes(d[values].tolist(), W, H), 0, 0, W, H)
        palette = self._palette()
        # Couleurs du thème, séparateurs de la couleur du fond, étiquette en haut à gauche
        for k, (r, name, v) in enumerate(zip(rects, names, d[values])):
            face = mcolors.to_rgba(palette[k % len(palette)], alpha)
            self.ax.add_patch(plt.Rectangle((r['x'], r['y']), r['dx'], r['dy'], facecolor=face,
                                            edgecolor=plt.rcParams['axes.facecolor'], linewidth=2))
            if min(r['dx'], r['dy']) < 6:  # bloc trop petit pour une étiquette lisible
                continue
            lum = 0.2126 * face[0] + 0.7152 * face[1] + 0.0722 * face[2]
            ink = '#111111' if lum > 0.62 else 'white'
            size = max(7, min(16, np.sqrt(r['dx'] * r['dy']) / 3.2))
            text = name + (f'\n{value_format.format(v)}' if show_values else '')
            self.ax.text(r['x'] + 1.2, r['y'] + r['dy'] - 1.2, text, ha='left', va='top',
                         color=ink, fontsize=size, linespacing=1.25, wrap=True)
        self.ax.set_xlim(0, W)
        self.ax.set_ylim(0, H)
        self.ax.axis('off')
        return self

    def hexbin(self, x: str, y: str, gridsize: int = 30,
               cmap: Optional[str] = None, mincnt: int = 1) -> 'Chart':
        """
        EN: Draw a hexagonal binning plot — the recommended scatter replacement
            when the dataset is too large for individual points to be readable.
            Each hexagon is coloured by the number of data points it contains.
        FR: Trace un graphique à pavage hexagonal — remplaçant recommandé du nuage de points
            quand le jeu de données est trop grand pour que les points individuels soient lisibles.
            Chaque hexagone est coloré par le nombre de points de données qu'il contient.

        Args / Paramètres:
            x (str):
                EN: Column for x-axis values.
                FR: Colonne pour les valeurs de l'axe x.
            y (str):
                EN: Column for y-axis values.
                FR: Colonne pour les valeurs de l'axe y.
            gridsize (int):
                EN: Number of hexagons along the x-direction. Default 30.
                FR: Nombre d'hexagones dans la direction x. Défaut 30.
            cmap (str):
                EN: Colormap for count encoding. Default 'YlOrRd'.
                FR: Colormap pour l'encodage du compte. Défaut 'YlOrRd'.
            mincnt (int):
                EN: Minimum count to render a hexagon (hides empty bins). Default 1.
                FR: Compte minimum pour afficher un hexagone (masque les cases vides). Défaut 1.

        Returns / Retourne:
            Chart: self

        Example / Exemple:
            chart.hexbin('age', 'revenu', gridsize=50, cmap='Blues')
        """
        self.chart_type = 'hexbin'
        hb = self.ax.hexbin(self.data[x], self.data[y], gridsize=gridsize,
                            cmap=cmap or plt.rcParams['image.cmap'], mincnt=mincnt,
                            linewidths=0.2, edgecolors=plt.rcParams['axes.facecolor'])
        self._colorbar(hb, 'Effectif')
        self.ax.set_xlabel(x)
        self.ax.set_ylabel(y)
        return self

    def parallel_coordinates(self, columns: List[str],
                              group: Optional[str] = None,
                              alpha: float = 0.4) -> 'Chart':
        """
        EN: Draw a parallel coordinates plot — a multivariate visualisation where
            each column becomes a vertical normalised axis and each row is a polyline.
            Lines are coloured by group when a group column is supplied.
        FR: Trace un graphique à coordonnées parallèles — une visualisation multivariée
            où chaque colonne devient un axe vertical normalisé et chaque ligne est une polyligne.
            Les lignes sont colorées par groupe si une colonne de groupe est fournie.

        Args / Paramètres:
            columns (List[str]):
                EN: Numeric columns to use as parallel axes (each normalised to [0, 1]).
                FR: Colonnes numériques utilisées comme axes parallèles (normalisées vers [0, 1]).
            group (str, optional):
                EN: Column for colour-coding lines by category.
                FR: Colonne pour colorier les lignes par catégorie.
            alpha (float):
                EN: Line opacity (lower = less cluttered with many rows). Default 0.4.
                FR: Opacité des lignes (plus bas = moins encombré avec beaucoup de lignes). Défaut 0.4.

        Returns / Retourne:
            Chart: self

        Example / Exemple:
            chart.parallel_coordinates(['vitesse', 'puissance', 'économie'],
                                        group='constructeur', alpha=0.3)
        """
        self.chart_type = 'parallel_coordinates'
        n = len(columns)
        norm = self.data[columns].copy().astype(float)
        for col in columns:
            lo, hi = norm[col].min(), norm[col].max()
            if hi > lo:
                norm[col] = (norm[col] - lo) / (hi - lo)
        for i in range(n):
            self.ax.axvline(i, color='gray', linewidth=0.8, zorder=0)
        groups = self.data[group].unique() if group else [None]
        palette = plt.cm.tab10(np.linspace(0, 1, len(groups)))
        color_map = dict(zip(groups, palette))
        for _, row in norm.iterrows():
            g = self.data.loc[row.name, group] if group else None
            self.ax.plot(range(n), row[columns].values,
                         color=color_map[g], alpha=alpha, linewidth=0.8)
        if group:
            for g, c in color_map.items():
                self.ax.plot([], [], color=c, label=str(g))
        self.ax.set_xticks(range(n))
        self.ax.set_xticklabels(columns)
        self.ax.set_ylim(-0.05, 1.05)
        self.ax.set_ylabel('Normalized value')
        return self

    def dumbbell(self, y: str, x_start: str, x_end: str,
                 color_start: Optional[str] = None, color_end: Optional[str] = None,
                 line_color: Optional[str] = None, markersize: float = 80,
                 linewidth: float = 2.0) -> 'Chart':
        """
        EN: Draw a dumbbell (connected dot) chart — ideal for before/after or A vs B
            comparisons within the same category. Each row produces one horizontal
            dumbbell: a line connecting two dots of different colours.
        FR: Trace un graphique en haltère (points reliés) — idéal pour les comparaisons
            avant/après ou A vs B dans une même catégorie. Chaque ligne produit un
            haltère horizontal : un trait reliant deux points de couleurs différentes.

        Args / Paramètres:
            y (str):
                EN: Column with category labels (one row per dumbbell).
                FR: Colonne avec les étiquettes de catégories (une ligne par haltère).
            x_start (str):
                EN: Column for the left (start / before) value.
                FR: Colonne pour la valeur gauche (départ / avant).
            x_end (str):
                EN: Column for the right (end / after) value.
                FR: Colonne pour la valeur droite (arrivée / après).
            color_start (str):
                EN: Dot color for the start value. Default 'steelblue'.
                FR: Couleur du point pour la valeur de départ. Défaut 'steelblue'.
            color_end (str):
                EN: Dot color for the end value. Default 'salmon'.
                FR: Couleur du point pour la valeur d'arrivée. Défaut 'salmon'.
            line_color (str):
                EN: Color of the connecting segment. Default '#aaaaaa'.
                FR: Couleur du segment de connexion. Défaut '#aaaaaa'.
            markersize (float):
                EN: Dot area in points². Default 80.
                FR: Aire du point en points². Défaut 80.
            linewidth (float):
                EN: Width of the connecting segment. Default 2.0.
                FR: Épaisseur du segment de connexion. Défaut 2.0.

        Returns / Retourne:
            Chart: self

        Example / Exemple:
            chart.dumbbell('pays', 'pib_2020', 'pib_2024',
                           color_start='#457b9d', color_end='#e63946')
        """
        self.chart_type = 'dumbbell'
        palette = self._palette()  # défaut : couleurs du thème, trait gris discret
        color_start = color_start or palette[0]
        color_end = color_end or palette[1 % len(palette)]
        line_color = line_color or mcolors.to_rgba(plt.rcParams['axes.labelcolor'], 0.45)
        for _, row in self.data.iterrows():
            self.ax.plot([row[x_start], row[x_end]], [row[y], row[y]],
                         color=line_color, linewidth=linewidth, zorder=1)
        self.ax.scatter(self.data[x_start], self.data[y],
                        color=color_start, s=markersize, zorder=2, label=x_start)
        self.ax.scatter(self.data[x_end], self.data[y],
                        color=color_end, s=markersize, zorder=2, label=x_end)
        return self

    def streamgraph(self, x: str, y: Union[str, List[str]],
                    alpha: float = 0.85) -> 'Chart':
        """
        EN: Draw a streamgraph — a stacked area chart with a centred (wiggle) baseline,
            producing an organic, flowing appearance. Well suited for showing how the
            composition of a total evolves over time without a fixed zero reference.
        FR: Trace un streamgraph — graphique à aires empilées avec une ligne de base centrée
            (wiggle), produisant un aspect organique et fluide. Bien adapté pour montrer
            comment la composition d'un total évolue dans le temps sans référence zéro fixe.

        Args / Paramètres:
            x (str):
                EN: Column for the time / period axis.
                FR: Colonne pour l'axe temporel / de période.
            y (Union[str, List[str]]):
                EN: One or more columns to stack as streams.
                FR: Une ou plusieurs colonnes à empiler comme flux.
            alpha (float):
                EN: Fill opacity. Default 0.85.
                FR: Opacité du remplissage. Défaut 0.85.

        Returns / Retourne:
            Chart: self

        Example / Exemple:
            chart.streamgraph('annee', ['rock', 'pop', 'jazz', 'classique'])
        """
        self.chart_type = 'streamgraph'
        if isinstance(y, str):
            y = [y]
        self.ax.stackplot(self.data[x], *[self.data[col] for col in y],
                          labels=y, alpha=alpha, baseline='wiggle')
        return self

    def gantt(self, task: str, start: str, end: str,
              group: Optional[str] = None, alpha: float = 0.8) -> 'Chart':
        """
        EN: Draw a Gantt chart for project or task scheduling.
            Bars are horizontal, positioned by date offset from the earliest start date.
            An optional group column colours bars by team, phase, or category.
        FR: Trace un diagramme de Gantt pour la planification de projets ou de tâches.
            Les barres sont horizontales, positionnées par décalage de date depuis la date
            de début la plus ancienne. Une colonne de groupe optionnelle colorie les barres
            par équipe, phase ou catégorie.

        Args / Paramètres:
            task (str):
                EN: Column with task names (used as y-axis labels).
                FR: Colonne avec les noms des tâches (utilisée comme étiquettes de l'axe y).
            start (str):
                EN: Column with start dates (any format parseable by pd.to_datetime).
                FR: Colonne avec les dates de début (tout format analysable par pd.to_datetime).
            end (str):
                EN: Column with end dates.
                FR: Colonne avec les dates de fin.
            group (str, optional):
                EN: Column for colour-coding bars by category (e.g. team, phase).
                FR: Colonne pour colorier les barres par catégorie (ex. équipe, phase).
            alpha (float):
                EN: Bar opacity. Default 0.8.
                FR: Opacité des barres. Défaut 0.8.

        Returns / Retourne:
            Chart: self

        Example / Exemple:
            chart.gantt('tache', 'debut', 'fin', group='equipe')
        """
        self.chart_type = 'gantt'
        starts = pd.to_datetime(self.data[start])
        ends = pd.to_datetime(self.data[end])
        origin = starts.min()
        offsets = [(s - origin).days for s in starts]
        durations = [(e - s).days for s, e in zip(starts, ends)]
        tasks = self.data[task].tolist()
        if group:
            groups = self.data[group].unique()
            palette = plt.cm.tab10(np.linspace(0, 1, len(groups)))
            color_map = dict(zip(groups, palette))
            colors = [color_map[g] for g in self.data[group]]
            for g, c in color_map.items():
                self.ax.barh([], [], color=c, label=str(g))
        else:
            colors = plt.cm.tab10(np.linspace(0, 1, len(tasks)))
        self.ax.barh(tasks, durations, left=offsets, color=colors, alpha=alpha)
        self.ax.set_xlabel(f'Jours depuis le {origin.date()}')
        self.ax.invert_yaxis()
        return self

    def calendar_heatmap(self, date_col: str, value_col: str,
                         cmap: str = 'YlGn') -> 'Chart':
        """
        EN: Draw a GitHub-style calendar heatmap — one coloured cell per day,
            arranged as a grid of weeks (columns) × days of the week (rows).
            Missing days are filled with 0.
        FR: Trace une carte de chaleur calendaire style GitHub — une cellule colorée par jour,
            disposée en grille semaines (colonnes) × jours de la semaine (lignes).
            Les jours manquants sont remplis avec 0.

        Args / Paramètres:
            date_col (str):
                EN: Column with dates (any format parseable by pd.to_datetime).
                FR: Colonne avec les dates (tout format analysable par pd.to_datetime).
            value_col (str):
                EN: Numeric column driving cell colour intensity.
                FR: Colonne numérique déterminant l'intensité de couleur des cellules.
            cmap (str):
                EN: Colormap name. Default 'YlGn' (GitHub palette).
                FR: Nom du colormap. Défaut 'YlGn' (palette GitHub).

        Returns / Retourne:
            Chart: self

        Example / Exemple:
            chart.calendar_heatmap('date', 'commits', cmap='Greens')
        """
        self.chart_type = 'calendar_heatmap'
        df = self.data[[date_col, value_col]].copy()
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.set_index(date_col).sort_index()
        full_range = pd.date_range(df.index.min(), df.index.max())
        values = df.reindex(full_range, fill_value=0)[value_col].values
        weeks = (full_range - full_range[0]).days // 7
        days = full_range.dayofweek
        n_weeks = int(weeks.max()) + 1
        grid = np.full((7, n_weeks), np.nan)
        for w, d, v in zip(weeks, days, values):
            grid[d, w] = v
        self.ax.pcolor(grid, cmap=cmap, edgecolors='white', linewidths=0.5)
        self.ax.set_yticks(np.arange(7) + 0.5)
        self.ax.set_yticklabels(['Lun', 'Mar', 'Mer', 'Jeu', 'Ven', 'Sam', 'Dim'])
        self.ax.set_aspect('equal')
        self.ax.invert_yaxis()
        self.ax.set_xlabel('Semaine')
        return self

    def sankey(self, flows: str, labels: str,
               orientations: Optional[str] = None,
               unit: str = '', scale: float = 0.01) -> 'Chart':
        """
        EN: Draw a Sankey diagram to visualise flows between categories.
            Uses matplotlib's built-in ``matplotlib.sankey.Sankey`` class.
            Each row of the DataFrame is one flow segment:
            positive values are inputs (entering), negative values are outputs (leaving).
        FR: Trace un diagramme de Sankey pour visualiser les flux entre catégories.
            Utilise la classe intégrée ``matplotlib.sankey.Sankey`` de matplotlib.
            Chaque ligne du DataFrame est un segment de flux :
            les valeurs positives sont des entrées, les valeurs négatives des sorties.

        Args / Paramètres:
            flows (str):
                EN: Column with flow magnitudes (positive = input, negative = output).
                FR: Colonne avec les amplitudes de flux (positif = entrée, négatif = sortie).
            labels (str):
                EN: Column with label for each flow segment.
                FR: Colonne avec l'étiquette de chaque segment de flux.
            orientations (str, optional):
                EN: Column with integer orientations: 0 = right, 1 = up, –1 = down.
                FR: Colonne avec les orientations entières : 0 = droite, 1 = haut, –1 = bas.
            unit (str):
                EN: Unit string appended to each numeric label (e.g. ' MW').
                FR: Chaîne d'unité ajoutée à chaque étiquette numérique (ex. ' MW').
            scale (float):
                EN: Global scale factor; smaller value = wider arrows. Default 0.01.
                FR: Facteur d'échelle global ; valeur plus petite = flèches plus larges. Défaut 0.01.

        Returns / Retourne:
            Chart: self

        Example / Exemple:
            chart.sankey('flux', 'etiquette', orientations='direction', unit=' GWh')
        """
        from matplotlib.sankey import Sankey
        self.chart_type = 'sankey'
        self.ax.axis('off')
        orient_vals = (self.data[orientations].tolist()
                       if orientations else [0] * len(self.data))
        Sankey(ax=self.ax, unit=unit, scale=scale).add(
            flows=self.data[flows].tolist(),
            labels=self.data[labels].tolist(),
            orientations=orient_vals,
        ).finish()
        return self

    # ──────────────────────────────────────── Sciences & data analysis / Sciences & analyse

    def _palette(self) -> List[str]:
        """Couleurs de ce graphique : la palette choisie (set_colors…), sinon celles du thème."""
        return (getattr(self, 'current_palette', None)
                or plt.rcParams['axes.prop_cycle'].by_key()['color'])

    def _diverging_cmap(self) -> mcolors.Colormap:
        """Échelle divergente tirée du thème : 2e couleur ← couleur du fond → 1re couleur."""
        p = self._palette()
        return mcolors.LinearSegmentedColormap.from_list(
            'plutovizu_divergent', [p[1 % len(p)], plt.rcParams['axes.facecolor'], p[0]])

    @staticmethod
    def _ink_on(color) -> str:
        """Couleur de texte lisible (presque noir ou blanc) sur un fond de couleur donnée."""
        r, g, b, _ = mcolors.to_rgba(color)
        return '#111111' if 0.2126 * r + 0.7152 * g + 0.0722 * b > 0.55 else 'white'

    @staticmethod
    def _error(values: pd.Series, error: str) -> float:
        """Erreur d'une série : 'se' (erreur type), 'sd' (écart-type) ou 'ci95'."""
        n = values.count()
        sd = values.std(ddof=1) if n > 1 else 0.0
        if error == 'sd':
            return sd
        se = sd / np.sqrt(n) if n else 0.0
        if error == 'se':
            return se
        if error == 'ci95':
            from scipy import stats
            return stats.t.ppf(0.975, n - 1) * se if n > 1 else 0.0
        raise ValueError(f"error doit valoir 'se', 'sd' ou 'ci95', pas '{error}'")

    def dendrogram(self, labels: Optional[str] = None, columns: Optional[List[str]] = None,
                   method: str = 'ward', metric: str = 'euclidean',
                   color_threshold: Optional[float] = None, standardize: bool = True,
                   orientation: str = 'top', leaf_rotation: float = 90) -> 'Chart':
        """
        EN: Hierarchical clustering dendrogram (scipy). Branches below
            color_threshold share a color; a dashed line marks the cut.
        FR: Dendrogramme de classification hiérarchique (scipy). Les branches sous
            color_threshold partagent une couleur ; une ligne pointillée marque la coupe.

        Args / Paramètres:
            labels (str, optional): EN: Column naming the leaves. / FR: Colonne nommant les feuilles.
            columns (List[str], optional): EN: Numeric columns used (default: all).
                / FR: Colonnes numériques utilisées (défaut : toutes).
            method (str): EN/FR: 'ward', 'average', 'complete', 'single'…
            metric (str): EN/FR: Distance ('euclidean', 'cityblock', 'correlation'…).
            color_threshold (float, optional): EN: Cut height. / FR: Hauteur de coupe.
            standardize (bool): EN: Z-score columns first (mixed units). Default True.
                / FR: Centre-réduit les colonnes (unités différentes). Défaut True.
            orientation (str): 'top', 'bottom', 'left', 'right'.
            leaf_rotation (float): EN/FR: Rotation des étiquettes des feuilles.

        Example / Exemple:
            chart.dendrogram(labels='variete', color_threshold=4)
        """
        from scipy.cluster.hierarchy import dendrogram, linkage, set_link_color_palette
        self.chart_type = 'dendrogram'
        cols = columns or [c for c in self.data.select_dtypes('number').columns if c != labels]
        X = self.data[cols].to_numpy(dtype=float)
        if standardize:
            X = (X - X.mean(axis=0)) / X.std(axis=0, ddof=1)
        leaf_labels = (self.data[labels] if labels else self.data.index).astype(str).tolist()
        Z = linkage(X, method=method, metric=metric)
        set_link_color_palette([mcolors.to_hex(c) for c in self._palette()])
        try:
            dendrogram(Z, labels=leaf_labels, ax=self.ax, orientation=orientation,
                       color_threshold=color_threshold,
                       above_threshold_color=plt.rcParams['axes.labelcolor'],
                       leaf_rotation=leaf_rotation)
        finally:
            set_link_color_palette(None)  # palette scipy globale : on la restaure
        self.ax.grid(False)
        vertical = orientation in ('top', 'bottom')
        (self.ax.set_ylabel if vertical else self.ax.set_xlabel)('Distance')
        if color_threshold is not None:
            line = self.ax.axhline if vertical else self.ax.axvline
            line(color_threshold, color=plt.rcParams['axes.labelcolor'], linestyle='--', linewidth=1)
        self.linkage = Z
        return self

    def regression(self, x: str, y: str, order: int = 1, ci: bool = True,
                   show_equation: bool = True, color: Optional[str] = None,
                   alpha: float = 0.7) -> 'Chart':
        """
        EN: Scatter plot with a polynomial fit, its 95% confidence band and the
            equation, R² (and p-value for a linear fit). Results in self.regression_stats.
        FR: Nuage de points avec ajustement polynomial, bande de confiance à 95 %,
            équation, R² (et p-value si linéaire). Résultats dans self.regression_stats.

        Args / Paramètres:
            x, y (str): EN/FR: Colonnes explicative et réponse.
            order (int): EN: Polynomial degree (1 = linear, 2 = quadratic…).
                / FR: Degré du polynôme (1 = linéaire, 2 = quadratique…).
            ci (bool): EN/FR: Bande de confiance à 95 %. Défaut True.
            show_equation (bool): EN/FR: Affiche équation et R². Défaut True.

        Example / Exemple:
            chart.regression('azote', 'rendement', order=2)   # courbe de réponse
        """
        from scipy import stats
        self.chart_type = 'regression'
        d = self.data[[x, y]].dropna()
        xv, yv = d[x].to_numpy(float), d[y].to_numpy(float)
        n = len(xv)
        if n <= order + 1:
            raise ValueError(f"Il faut plus de {order + 1} points pour un polynôme de degré {order}")
        coefs = np.polyfit(xv, yv, order)
        residuals = yv - np.polyval(coefs, xv)
        r2 = 1 - (residuals ** 2).sum() / ((yv - yv.mean()) ** 2).sum()
        p = stats.linregress(xv, yv).pvalue if order == 1 else None
        c = color or self._palette()[0]
        self.ax.scatter(xv, yv, color=c, alpha=alpha, label=y)
        xs = np.linspace(xv.min(), xv.max(), 200)
        fit = np.polyval(coefs, xs)
        self.ax.plot(xs, fit, color=c, label='Ajustement')
        if ci:
            # Bande de confiance de la moyenne : se = sqrt(v' (X'X)^-1 v) * s
            Xd = np.vander(xv, order + 1)
            s2 = (residuals ** 2).sum() / (n - order - 1)
            cov = s2 * np.linalg.pinv(Xd.T @ Xd)
            V = np.vander(xs, order + 1)
            half = stats.t.ppf(0.975, n - order - 1) * np.sqrt(np.einsum('ij,jk,ik->i', V, cov, V))
            self.ax.fill_between(xs, fit - half, fit + half, color=c, alpha=0.18, linewidth=0)
        if show_equation:
            terms = []
            for power, coef in zip(range(order, -1, -1), coefs):
                var = '' if power == 0 else ('x' if power == 1 else f'x$^{power}$')
                terms.append(f'{coef:+.3g}{var}')
            eq = 'y = ' + ' '.join(terms).lstrip('+')
            text = f'{eq}\n$R^2$ = {r2:.3f}' + (f'   p = {p:.2g}' if p is not None else '')
            self.ax.text(0.02, 0.97, text, transform=self.ax.transAxes, va='top', fontsize=9,
                         bbox=dict(boxstyle='round,pad=0.4', fc=plt.rcParams['axes.facecolor'],
                                   ec=plt.rcParams['grid.color'], alpha=0.9))
        self.ax.set_xlabel(x)
        self.ax.set_ylabel(y)
        self.regression_stats = {'coefs': coefs, 'r2': r2, 'p': p, 'n': n}
        return self

    def pca_biplot(self, columns: Optional[List[str]] = None, group: Optional[str] = None,
                   scale: bool = True, arrows: bool = True, alpha: float = 0.8) -> 'Chart':
        """
        EN: Principal component analysis biplot: individuals on PC1/PC2 (colored by
            group) and variable loadings as arrows. Results in self.pca.
        FR: Biplot d'analyse en composantes principales : individus sur CP1/CP2
            (colorés par groupe) et variables en flèches. Résultats dans self.pca.

        Args / Paramètres:
            columns (List[str], optional): EN/FR: Variables numériques (défaut : toutes).
            group (str, optional): EN/FR: Colonne de groupe (couleurs + légende).
            scale (bool): EN: Standardize variables (normed PCA). Default True.
                / FR: Centre-réduit les variables (ACP normée). Défaut True.
            arrows (bool): EN/FR: Affiche les flèches des variables. Défaut True.

        Example / Exemple:
            chart.pca_biplot(group='espece')
        """
        self.chart_type = 'pca_biplot'
        cols = columns or [c for c in self.data.select_dtypes('number').columns if c != group]
        if len(cols) < 2:
            raise ValueError("L'ACP demande au moins deux variables numériques")
        d = self.data.dropna(subset=cols)
        X = d[cols].to_numpy(float)
        X = X - X.mean(axis=0)
        if scale:
            X = X / X.std(axis=0, ddof=1)
        U, S, Vt = np.linalg.svd(X, full_matrices=False)
        scores = U * S
        explained = S ** 2 / (S ** 2).sum()
        loadings = Vt.T * S / np.sqrt(len(X) - 1)  # corrélations variable/composante si normée
        groups = d[group] if group else pd.Series('', index=d.index)
        for color, (name, idx) in zip(self._palette() * 10, groups.groupby(groups).groups.items()):
            pos = d.index.get_indexer(idx)
            self.ax.scatter(scores[pos, 0], scores[pos, 1], color=color, alpha=alpha,
                            label=name if group else None)
        if arrows:
            k = 0.85 * np.abs(scores[:, :2]).max() / np.abs(loadings[:, :2]).max()
            ink = plt.rcParams['text.color']
            tips = loadings[:, :2] * k
            for name, (lx, ly) in zip(cols, tips):
                self.ax.annotate('', xy=(lx, ly), xytext=(0, 0),
                                 arrowprops=dict(arrowstyle='->', color=ink, lw=1.2))
                self.ax.text(lx * 1.08, ly * 1.08, name, color=ink, fontsize=9,
                             ha='center', va='center')
            # annotate() n'agrandit pas les axes : on y inclut les pointes et étiquettes
            self.ax.update_datalim(tips * 1.2)
            self.ax.autoscale_view()
        for line in (self.ax.axhline, self.ax.axvline):
            line(0, color=plt.rcParams['grid.color'], linewidth=1, zorder=0)
        self.ax.set_xlabel(f'CP1 ({explained[0]:.1%})')
        self.ax.set_ylabel(f'CP2 ({explained[1]:.1%})')
        if group:
            self.ax.legend()
        pcs = [f'CP{i + 1}' for i in range(len(S))]
        self.pca = {'explained_variance_ratio': explained,
                    'scores': pd.DataFrame(scores, index=d.index, columns=pcs),
                    'loadings': pd.DataFrame(loadings, index=cols, columns=pcs)}
        return self

    def mean_bar(self, x: str, y: str, hue: Optional[str] = None, error: str = 'se',
                 points: bool = False, letters: bool = False, alpha_test: float = 0.05,
                 capsize: float = 4, alpha: float = 0.85) -> 'Chart':
        """
        EN: Bar chart of group means with error bars — the classic display of
            field-trial treatment means. Optional second factor (hue), raw points and
            Tukey HSD compact letters (same letter = not significantly different).
        FR: Barres des moyennes par groupe avec barres d'erreur — l'affichage classique
            des moyennes de traitements d'un essai. Second facteur (hue), points bruts et
            lettres du test de Tukey (même lettre = pas de différence significative).

        Args / Paramètres:
            x (str): EN/FR: Facteur (traitement, variété…).
            y (str): EN/FR: Variable réponse.
            hue (str, optional): EN/FR: Second facteur → barres groupées.
            error (str): 'se' (erreur type, défaut), 'sd' (écart-type) ou 'ci95'.
            points (bool): EN/FR: Superpose les observations individuelles.
            letters (bool): EN: Tukey HSD letters — between x levels, or between hue
                groups within each x level when hue is given.
                FR: Lettres de Tukey — entre niveaux de x, ou entre groupes hue à
                l'intérieur de chaque niveau de x si hue est fourni.
            alpha_test (float): EN/FR: Seuil de significativité du test. Défaut 0.05.

        Example / Exemple:
            chart.mean_bar('traitement', 'rendement', letters=True)
        """
        self.chart_type = 'mean_bar'
        # valeurs Python natives (0 plutôt que np.int64(0)) : clés lisibles dans tukey_letters
        native = lambda v: v.item() if isinstance(v, np.generic) else v  # noqa: E731
        levels = [native(v) for v in pd.unique(self.data[x])]
        hues = [native(v) for v in pd.unique(self.data[hue])] if hue else [None]
        width = 0.8 / len(hues)
        pos = np.arange(len(levels))
        tops = {}  # (niveau x, groupe hue) -> (abscisse, haut de la barre d'erreur)
        for j, (h, color) in enumerate(zip(hues, self._palette() * 10)):
            sub = self.data if h is None else self.data[self.data[hue] == h]
            g = sub.groupby(x)[y]
            means = g.mean().reindex(levels)
            errs = [self._error(sub.loc[sub[x] == lv, y], error) for lv in levels]
            offs = pos - 0.4 + width * (j + 0.5)
            self.ax.bar(offs, means, width=width * 0.92, yerr=errs, capsize=capsize,
                        color=color, alpha=alpha, label=None if h is None else str(h),
                        error_kw={'ecolor': plt.rcParams['axes.labelcolor'], 'elinewidth': 1})
            for k, lv in enumerate(levels):
                tops[(lv, h)] = (offs[k], means[lv] + errs[k])
            if points:
                rng = np.random.default_rng(0)
                for k, lv in enumerate(levels):
                    vals = sub.loc[sub[x] == lv, y].dropna()
                    jitter = rng.uniform(-width * 0.25, width * 0.25, len(vals))
                    self.ax.scatter(offs[k] + jitter, vals, s=12, zorder=3,
                                    color=plt.rcParams['text.color'], alpha=0.6)
                    if len(vals):
                        tops[(lv, h)] = (tops[(lv, h)][0], max(tops[(lv, h)][1], vals.max()))
        if letters:
            # Familles comparées : les niveaux de x, ou les groupes hue dans chaque niveau de x
            families = ([[(lv, None) for lv in levels]] if hue is None
                        else [[(lv, h) for h in hues] for lv in levels])
            self.tukey_letters = {}
            span = np.nanmax([t for _, t in tops.values()]) or 1
            for family in families:
                samples = [self.data.loc[(self.data[x] == lv) &
                                         (True if h is None else self.data[hue] == h), y].dropna()
                           for lv, h in family]
                labels = _tukey_letters(samples, alpha_test)
                for key, lab in zip(family, labels):
                    self.tukey_letters[key[0] if hue is None else key] = lab
                    xpos, top = tops[key]
                    self.ax.text(xpos, top + 0.04 * span, lab, ha='center', va='bottom',
                                 fontsize=9, fontweight='bold', color=plt.rcParams['text.color'])
            self.ax.margins(y=0.08)
        self.ax.set_xticks(pos)
        self.ax.set_xticklabels([str(v) for v in levels])
        self.ax.set_xlabel(x)
        self.ax.set_ylabel(f'{y} (moyenne ± {error.upper()})')
        if hue:
            self.ax.legend(title=hue)
        return self

    def interaction(self, x: str, trace: str, response: str, error: Optional[str] = 'se',
                    marker: str = 'o') -> 'Chart':
        """
        EN: Interaction plot: mean response along factor x, one line per level of
            trace. Non-parallel lines suggest an interaction (two-way ANOVA).
        FR: Graphique d'interaction : réponse moyenne selon le facteur x, une ligne par
            niveau de trace. Des lignes non parallèles suggèrent une interaction (ANOVA à 2 facteurs).

        Args / Paramètres:
            x (str): EN/FR: Facteur en abscisse (ex. dose d'azote).
            trace (str): EN/FR: Facteur tracé en lignes (ex. variété).
            response (str): EN/FR: Variable réponse (ex. rendement).
            error (str, optional): 'se', 'sd', 'ci95' ou None (sans barres).

        Example / Exemple:
            chart.interaction('dose', 'variete', 'rendement')
        """
        self.chart_type = 'interaction'
        levels = list(pd.unique(self.data[x]))
        pos = np.arange(len(levels))
        for t, color in zip(pd.unique(self.data[trace]), self._palette() * 10):
            sub = self.data[self.data[trace] == t]
            means = sub.groupby(x)[response].mean().reindex(levels)
            errs = ([self._error(sub.loc[sub[x] == lv, response], error) for lv in levels]
                    if error else None)
            self.ax.errorbar(pos, means, yerr=errs, color=color, marker=marker,
                             capsize=3, label=str(t))
        self.ax.set_xticks(pos)
        self.ax.set_xticklabels([str(v) for v in levels])
        self.ax.set_xlabel(x)
        self.ax.set_ylabel(f'{response} (moyenne)')
        self.ax.legend(title=trace)
        return self

    def qqplot(self, column: str, color: Optional[str] = None) -> 'Chart':
        """
        EN: Normal quantile-quantile plot: points close to the line indicate
            normally distributed data (e.g. ANOVA residuals).
        FR: Diagramme quantile-quantile normal : des points proches de la droite
            indiquent une distribution normale (ex. résidus d'ANOVA).

        Example / Exemple:
            chart.qqplot('residus')
        """
        from scipy import stats
        self.chart_type = 'qqplot'
        (osm, osr), (slope, intercept, r) = stats.probplot(self.data[column].dropna(), dist='norm')
        c = color or self._palette()[0]
        self.ax.scatter(osm, osr, color=c, alpha=0.8)
        line_x = np.array([osm.min(), osm.max()])
        self.ax.plot(line_x, slope * line_x + intercept, color=plt.rcParams['axes.labelcolor'],
                     linestyle='--', linewidth=1.2)
        self.ax.set_xlabel('Quantiles théoriques (loi normale)')
        self.ax.set_ylabel(f'Quantiles observés — {column}')
        self.ax.text(0.02, 0.97, f'r = {r:.3f}', transform=self.ax.transAxes, va='top', fontsize=9)
        return self

    def ternary(self, a: str, b: str, c: str, group: Optional[str] = None,
                grid_step: float = 0.1, alpha: float = 0.85, size: float = 40) -> 'Chart':
        """
        EN: Ternary diagram for three-part compositions summing to a constant
            (e.g. soil texture sand / silt / clay). Values may be fractions or percents.
            Corners: a bottom-left, b bottom-right, c top.
        FR: Diagramme ternaire pour des compositions en trois parts de somme constante
            (ex. texture du sol sable / limon / argile). Fractions ou pourcentages.
            Sommets : a en bas à gauche, b en bas à droite, c en haut.

        Example / Exemple:
            chart.ternary('sable', 'limon', 'argile', group='parcelle')
        """
        self.chart_type = 'ternary'
        h = np.sqrt(3) / 2

        def xy(fa, fb, fc):
            return fb + fc / 2, fc * h

        ink = plt.rcParams['axes.labelcolor']
        grid_color = plt.rcParams['grid.color']
        self.ax.plot([0, 1, 0.5, 0], [0, 0, h, 0], color=ink, linewidth=1.2)
        for f in np.arange(grid_step, 1, grid_step):
            for p1, p2 in (((1 - f, 0, f), (0, 1 - f, f)),     # c constant
                           ((f, 1 - f, 0), (f, 0, 1 - f)),     # a constant
                           ((1 - f, f, 0), (0, f, 1 - f))):    # b constant
                (x1, y1), (x2, y2) = xy(*p1), xy(*p2)
                self.ax.plot([x1, x2], [y1, y2], color=grid_color, linewidth=0.7, zorder=0)
            # Convention du triangle USDA : c lu sur le bord gauche (lignes horizontales),
            # b sur le bord droit, a sur le bord inférieur.
            pct = f'{f * 100:.0f}'
            cx, cy = xy(1 - f, 0, f)                       # graduations de c (gauche)
            self.ax.text(cx - 0.03, cy, pct, ha='right', va='center', fontsize=8, color=ink)
            bx, by = xy(0, f, 1 - f)                       # graduations de b (droite)
            self.ax.text(bx + 0.03, by, pct, ha='left', va='center', fontsize=8, color=ink)
            ax_, ay_ = xy(f, 1 - f, 0)                     # graduations de a (bas)
            self.ax.text(ax_, ay_ - 0.035, pct, ha='center', va='top', fontsize=8, color=ink)
        for (px, py), name, ha, va in ((xy(1, 0, 0), a, 'right', 'top'),
                                       (xy(0, 1, 0), b, 'left', 'top'),
                                       (xy(0, 0, 1), c, 'center', 'bottom')):
            self.ax.text(px + {'right': -0.04, 'left': 0.04, 'center': 0}[ha],
                         py + (0.04 if va == 'bottom' else -0.07), name,
                         ha=ha, va=va, fontweight='bold', color=plt.rcParams['text.color'])
        comp = self.data[[a, b, c]].to_numpy(float)
        comp = comp / comp.sum(axis=1, keepdims=True)
        px, py = xy(comp[:, 0], comp[:, 1], comp[:, 2])
        groups = self.data[group] if group else pd.Series('', index=self.data.index)
        for color, (name, idx) in zip(self._palette() * 10, groups.groupby(groups).groups.items()):
            pos = self.data.index.get_indexer(idx)
            self.ax.scatter(px[pos], py[pos], s=size, color=color, alpha=alpha, zorder=3,
                            label=name if group else None)
        self.ax.set_aspect('equal')
        self.ax.set_xlim(-0.12, 1.12)
        self.ax.set_ylim(-0.12, h + 0.08)
        self.ax.axis('off')
        if group:
            self.ax.legend(loc='upper right')
        return self

    def soil_texture(self, sand: str, silt: str, clay: str, group: Optional[str] = None,
                     show_classes: bool = True, fill: bool = True, **kwargs) -> 'Chart':
        """
        EN: USDA soil texture triangle: the 12 texture classes (drawn and named) with
            the samples on top. The class of each sample is in self.texture_classes.
        FR: Triangle des textures USDA : les 12 classes (tracées et nommées) et les
            échantillons par-dessus. Classe de chaque échantillon dans self.texture_classes.

        Args / Paramètres:
            sand, silt, clay (str): EN/FR: Colonnes sable, limon, argile (% ou fractions).
            group (str, optional): EN/FR: Colonne de groupe (couleurs + légende).
            show_classes (bool): EN/FR: Trace les classes. Défaut True.
            fill (bool): EN/FR: Colore légèrement chaque classe. Défaut True.
            **kwargs: EN/FR: Transmis à ternary() (alpha, size, grid_step).

        Example / Exemple:
            chart.soil_texture('sable', 'limon', 'argile', group='parcelle')
        """
        from .soil import USDA_POLYGONS, texture_class
        self.ternary(sand, silt, clay, group=group, **kwargs)
        self.chart_type = 'soil_texture'
        if show_classes:
            h = np.sqrt(3) / 2
            ink = plt.rcParams['axes.labelcolor']
            for color, (name, verts) in zip(self._palette() * 10, USDA_POLYGONS.items()):
                pts = np.array([(b / 100 + c / 200, c / 100 * h) for _, b, c in verts])
                self.ax.add_patch(plt.Polygon(
                    pts, closed=True, edgecolor=ink, linewidth=0.9, zorder=1,
                    facecolor=mcolors.to_rgba(color, 0.10) if fill else 'none'))
                cx, cy = pts.mean(axis=0)
                self.ax.text(cx, cy, name.replace(' ', '\n', 1), ha='center', va='center',
                             fontsize=6.5, color=ink, zorder=2, linespacing=1.0)
        self.texture_classes = pd.Series(
            texture_class(self.data[sand], self.data[silt], self.data[clay]),
            index=self.data.index, name='texture')
        return self

    def circle_packing(self, values: str, labels: Optional[str] = None,
                       group: Optional[Union[str, List[str]]] = None,
                       show_labels: bool = True, show_values: bool = False,
                       alpha: float = 0.85, min_label_radius: float = 0.05) -> 'Chart':
        """
        EN: Circle packing (circlify): one circle per row, area proportional to the
            value. With group (one column or a list = several levels), circles are
            nested inside their group's circle. Circles are in self.circles.
        FR: Cercles imbriqués (circlify) : un cercle par ligne, d'aire proportionnelle
            à la valeur. Avec group (une colonne ou une liste = plusieurs niveaux), les
            cercles sont regroupés dans le cercle de leur groupe. Cercles dans self.circles.

        Args / Paramètres:
            values (str): EN/FR: Colonne numérique (valeurs positives).
            labels (str, optional): EN/FR: Colonne des étiquettes (défaut : l'index).
            group (str | List[str], optional): EN/FR: Niveau(x) de regroupement.
            show_labels (bool): EN/FR: Affiche les étiquettes. Défaut True.
            show_values (bool): EN/FR: Ajoute la valeur sous l'étiquette. Défaut False.
            min_label_radius (float): EN: Smaller circles get no label (radius, 0–1).
                FR: Pas d'étiquette sous ce rayon (0–1). Défaut 0.05.

        Example / Exemple:
            chart.circle_packing('surface', labels='culture', group='famille')
        """
        import circlify
        self.chart_type = 'circle_packing'
        levels = [group] if isinstance(group, str) else list(group or [])
        d = self.data.dropna(subset=[values])
        if (d[values] < 0).any():
            raise ValueError("circle_packing demande des valeurs positives (l'aire d'un cercle)")
        names = (d[labels] if labels else d.index.to_series()).astype(str)
        palette = self._palette()
        meta = {}  # id unique -> étiquette et couleur (circlify n'accepte que id/datum/children)

        def build(frame, depth, path, color):
            if depth == len(levels):
                nodes = []
                for k, (i, v) in enumerate(frame[values].items()):
                    if v > 0:
                        uid = f'{path}/{i}'
                        meta[uid] = {'label': names[i],
                                     'color': color or palette[k % len(palette)]}
                        nodes.append({'id': uid, 'datum': float(v)})
                return nodes
            nodes = []
            for k, (key, sub) in enumerate(frame.groupby(levels[depth], sort=False)):
                c = palette[k % len(palette)] if depth == 0 else color
                children = build(sub, depth + 1, f'{path}/{key}', c)
                if children:
                    uid = f'{path}/{key}'
                    meta[uid] = {'label': str(key), 'color': c}
                    nodes.append({'id': uid, 'datum': sum(ch['datum'] for ch in children),
                                  'children': children})
            return nodes

        self.circles = circlify.circlify(build(d, 0, '', None), show_enclosure=False)
        leaf_level = len(levels) + 1
        bg = plt.rcParams['axes.facecolor']
        for circle in self.circles:
            x, y, r = circle.x, circle.y, circle.r
            info = {**meta[circle.ex['id']], 'datum': circle.ex['datum']}
            circle.ex.update(meta[circle.ex['id']])  # étiquette et couleur lisibles dans self.circles
            if circle.level < leaf_level:   # cercle de groupe : contour + fond léger
                self.ax.add_patch(plt.Circle((x, y), r, facecolor=mcolors.to_rgba(info['color'], 0.10),
                                             edgecolor=info['color'], linewidth=1.5))
                if show_labels:  # étiquette posée sur le bord haut du cercle de groupe
                    self.ax.text(x, y + r, info['label'], ha='center', va='center', zorder=5,
                                 fontsize=10 if circle.level == 1 else 8.5, fontweight='bold',
                                 color=plt.rcParams['text.color'],
                                 bbox=dict(boxstyle='round,pad=0.25', fc=bg,
                                           ec=info['color'], lw=1))
                continue
            face = mcolors.to_rgba(info['color'], alpha)
            self.ax.add_patch(plt.Circle((x, y), r, facecolor=face, edgecolor=bg, linewidth=1))
            if show_labels and r >= min_label_radius:
                lum = 0.2126 * face[0] + 0.7152 * face[1] + 0.0722 * face[2]  # texte lisible
                text = info['label'] + (f"\n{info['datum']:,.4g}" if show_values else '')
                self.ax.text(x, y, text, ha='center', va='center', linespacing=1.1,
                             fontsize=max(6.5, min(12, r * 45)),
                             color='#111111' if lum > 0.55 else 'white')
        self.ax.set_xlim(-1.03, 1.03)
        self.ax.set_ylim(-1.03, 1.03)
        self.ax.set_aspect('equal')
        self.ax.axis('off')
        return self

    def clustermap(self, columns: Optional[List[str]] = None, labels: Optional[str] = None,
                   standardize: bool = True, cmap: Optional[str] = None,
                   method: str = 'ward', figsize: Optional[Tuple[float, float]] = None,
                   **kwargs) -> 'Chart':
        """
        EN: Heatmap with hierarchical clustering dendrograms on rows and columns
            (seaborn.clustermap). Creates its own figure.
        FR: Carte de chaleur avec dendrogrammes de classification sur les lignes et les
            colonnes (seaborn.clustermap). Crée sa propre figure.

        Args / Paramètres:
            columns (List[str], optional): EN/FR: Variables numériques (défaut : toutes).
            labels (str, optional): EN/FR: Colonne nommant les lignes.
            standardize (bool): EN: Z-score each column (mixed units). Default True.
                / FR: Centre-réduit chaque colonne (unités différentes). Défaut True.
            cmap (str, optional): EN/FR: Défaut 'vlag' si standardisé, sinon celle du thème.
            method (str): EN/FR: Méthode de liaison ('ward', 'average'…).

        Example / Exemple:
            chart.clustermap(labels='variete')
        """
        cols = columns or [c for c in self.data.select_dtypes('number').columns if c != labels]
        mat = (self.data.set_index(labels) if labels else self.data)[cols]
        return self.seaborn('clustermap', data=mat, method=method,
                            z_score=1 if standardize else None,
                            cmap=cmap or ('vlag' if standardize else None),
                            center=0 if standardize else None,
                            figsize=figsize or self.fig.get_size_inches(), **kwargs)

    # ──────────────────────────────────────── Matplotlib plot types / Types matplotlib

    def _replace_axes(self, **kwargs):
        """Remplace l'axe courant (projection polaire, 3D…)."""
        self.fig.delaxes(self.ax)
        self.ax = self.fig.add_subplot(111, **kwargs)
        if kwargs.get('projection') == '3d':  # panneaux 3D à la couleur du thème
            for axis in (self.ax.xaxis, self.ax.yaxis, self.ax.zaxis):
                axis.set_pane_color(mcolors.to_rgba(plt.rcParams['axes.facecolor']))
        return self.ax

    def _as_grid(self, x: str, y: str, *z: str):
        """Si (x, y) forment une grille régulière complète, retourne (xs, ys, [Z…]) avec
        Z de forme (len(ys), len(xs)) ; sinon None (points irréguliers)."""
        d = self.data[[x, y, *z]].dropna()
        xs, ys = np.sort(d[x].unique()), np.sort(d[y].unique())
        if len(xs) * len(ys) != len(d) or d.duplicated([x, y]).any() or min(len(xs), len(ys)) < 2:
            return None
        return xs, ys, [d.pivot(index=y, columns=x, values=col).reindex(index=ys, columns=xs)
                        .to_numpy(float) for col in z]

    def _legend_below(self, inches: float = 0.5) -> Dict[str, Any]:
        """Options de légende centrée sous l'axe, à `inches` pouces du bas du graphique
        (sous les graduations et le titre de l'axe), quelle que soit la taille de la figure."""
        import matplotlib.transforms as mtransforms
        below = self.ax.transAxes + mtransforms.ScaledTranslation(0, -inches, self.fig.dpi_scale_trans)
        return dict(loc='upper center', bbox_to_anchor=(0.5, 0), bbox_transform=below)

    def _concise_dates(self, column: str) -> None:
        """Dates en abscisse : graduations compactes (« juin », « 15 »…) au lieu de dates complètes."""
        if pd.api.types.is_datetime64_any_dtype(self.data[column]):
            import matplotlib.dates as mdates
            locator = mdates.AutoDateLocator()
            self.ax.xaxis.set_major_locator(locator)
            self.ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))

    def _colorbar(self, mappable, label: str):
        cbar = self.fig.colorbar(mappable, ax=self.ax)
        cbar.set_label(label)
        cbar.outline.set_visible(False)
        return cbar

    def stem(self, x: str, y: str, baseline: float = 0.0, color: Optional[str] = None) -> 'Chart':
        """
        EN: Stem plot: a vertical line from the baseline to each value (matplotlib stem).
        FR: Diagramme en bâtons : un trait de la ligne de base à chaque valeur (stem).

        Example / Exemple:
            chart.stem('jour', 'pluie')
        """
        self.chart_type = 'stem'
        c = color or self._palette()[0]
        markers, stems, base = self.ax.stem(self.data[x], self.data[y], bottom=baseline)
        plt.setp(markers, color=c)
        plt.setp(stems, color=c, linewidth=1.5)
        plt.setp(base, color=plt.rcParams['axes.labelcolor'], linewidth=1)
        self.ax.set_xlabel(x)
        self._concise_dates(x)
        self.ax.set_ylabel(y)
        return self

    def step(self, x: str, y: Union[str, List[str]], where: str = 'mid',
             fill: bool = False, alpha: float = 0.2) -> 'Chart':
        """
        EN: Step (stairs) curve — values constant between two points (matplotlib step).
        FR: Courbe en escalier — valeur constante entre deux points (step / stairs).

        Args / Paramètres:
            where (str): 'pre', 'post' ou 'mid' (défaut) : position de la marche.
            fill (bool): EN/FR: Remplit sous la courbe.

        Example / Exemple:
            chart.step('date', 'stock', where='post', fill=True)
        """
        self.chart_type = 'step'
        ys = [y] if isinstance(y, str) else list(y)
        for col, c in zip(ys, self._palette() * 10):
            self.ax.step(self.data[x], self.data[col], where=where, color=c, label=col)
            if fill:
                self.ax.fill_between(self.data[x], self.data[col], step=where, color=c,
                                     alpha=alpha, linewidth=0)
        self.ax.set_xlabel(x)
        self._concise_dates(x)
        if len(ys) > 1:
            self.ax.legend()
        else:
            self.ax.set_ylabel(ys[0])
        return self

    def band(self, x: str, low: str, high: str, y: Optional[str] = None,
             color: Optional[str] = None, alpha: float = 0.25,
             label: Optional[str] = None) -> 'Chart':
        """
        EN: Shaded band between two columns (min–max, confidence interval…), with an
            optional central curve (matplotlib fill_between).
        FR: Bande colorée entre deux colonnes (min–max, intervalle de confiance…), avec
            une courbe centrale optionnelle (fill_between).

        Example / Exemple:
            chart.band('jour', 'tmin', 'tmax', y='tmoy')
        """
        self.chart_type = 'band'
        c = color or self._palette()[0]
        self.ax.fill_between(self.data[x], self.data[low], self.data[high], color=c,
                             alpha=alpha, linewidth=0, label=label or f'{low} – {high}')
        if y:
            self.ax.plot(self.data[x], self.data[y], color=c, label=y)
        self.ax.set_xlabel(x)
        self._concise_dates(x)
        self.ax.legend()
        return self

    def errorbar(self, x: str, y: str, yerr: Optional[Union[str, float]] = None,
                 xerr: Optional[Union[str, float]] = None, group: Optional[str] = None,
                 capsize: float = 3, marker: str = 'o', line: bool = False) -> 'Chart':
        """
        EN: Points with error bars; yerr / xerr are columns or constants (matplotlib errorbar).
        FR: Points avec barres d'erreur ; yerr / xerr sont des colonnes ou des constantes.

        Args / Paramètres:
            group (str, optional): EN/FR: Une série (couleur) par groupe.
            line (bool): EN/FR: Relie les points. Défaut False.

        Example / Exemple:
            chart.errorbar('dose', 'moyenne', yerr='ecart_type', group='variete', line=True)
        """
        self.chart_type = 'errorbar'
        groups = self.data.groupby(group, sort=False) if group else [(None, self.data)]
        for (name, sub), c in zip(groups, self._palette() * 10):
            err = {k: (sub[v] if isinstance(v, str) else v)
                   for k, v in (('yerr', yerr), ('xerr', xerr)) if v is not None}
            self.ax.errorbar(sub[x], sub[y], capsize=capsize, marker=marker, color=c,
                             linestyle='-' if line else 'none', label=name, **err)
        self.ax.set_xlabel(x)
        self.ax.set_ylabel(y)
        if group:
            self.ax.legend(title=group)
        return self

    def eventplot(self, time: str, group: Optional[str] = None,
                  linelength: float = 0.7, linewidth: float = 1.5) -> 'Chart':
        """
        EN: Event plot: one tick per event, one row per group (observation dates,
            phenological stages, arrivals…). Works with numbers or dates.
        FR: Diagramme d'événements : un trait par événement, une ligne par groupe (dates
            d'observation, stades phénologiques, arrivées…). Nombres ou dates.

        Example / Exemple:
            chart.eventplot('date_floraison', group='variete')
        """
        import matplotlib.dates as mdates
        self.chart_type = 'eventplot'
        groups = list(self.data.groupby(group, sort=False)) if group else [('', self.data)]
        is_date = pd.api.types.is_datetime64_any_dtype(self.data[time])
        positions = [mdates.date2num(sub[time].dropna()) if is_date else sub[time].dropna()
                     for _, sub in groups]
        self.ax.eventplot(positions, colors=(self._palette() * 10)[:len(groups)],
                          lineoffsets=np.arange(len(groups)), linelengths=linelength,
                          linewidths=linewidth)
        if is_date:
            self.ax.xaxis_date()
            self._concise_dates(time)
        self.ax.set_yticks(np.arange(len(groups)), [str(n) for n, _ in groups])
        self.ax.set_xlabel(time)
        self.ax.grid(False, axis='y')
        return self

    def hist2d(self, x: str, y: str, bins: Union[int, Tuple[int, int]] = 30,
               cmap: Optional[str] = None, colorbar: bool = True) -> 'Chart':
        """
        EN: Two-dimensional histogram: counts in rectangular bins (matplotlib hist2d).
        FR: Histogramme 2D : effectifs dans une grille de rectangles (hist2d).

        Example / Exemple:
            chart.hist2d('temperature', 'humidite', bins=25)
        """
        self.chart_type = 'hist2d'
        d = self.data[[x, y]].dropna()
        *_, image = self.ax.hist2d(d[x], d[y], bins=bins, cmin=1,
                                   cmap=cmap or plt.rcParams['image.cmap'])
        if colorbar:
            self._colorbar(image, 'Effectif')
        self.ax.set_xlabel(x)
        self.ax.set_ylabel(y)
        self.ax.grid(False)
        return self

    def bihistogram(self, column: str, group: str, bins: Union[int, str] = 20,
                    horizontal: bool = False, alpha: float = 0.85) -> 'Chart':
        """
        EN: Back-to-back histograms of two groups sharing the same bins (one upward, one
            downward). horizontal=True gives a population-pyramid layout.
        FR: Histogrammes dos à dos de deux groupes aux mêmes classes (l'un vers le haut,
            l'autre vers le bas). horizontal=True donne une pyramide des âges.

        Example / Exemple:
            chart.bihistogram('age', 'sexe', bins=range(0, 90, 5), horizontal=True)
        """
        self.chart_type = 'bihistogram'
        levels = list(pd.unique(self.data[group].dropna()))
        if len(levels) != 2:
            raise ValueError(f"bihistogram compare exactement 2 groupes ; '{group}' en a {len(levels)}")
        values = self.data[column].dropna()
        edges = np.histogram_bin_edges(values, bins=bins)
        widths = np.diff(edges)
        for sign, level, c in zip((1, -1), levels, self._palette()):
            counts, _ = np.histogram(self.data.loc[self.data[group] == level, column].dropna(), edges)
            if horizontal:
                self.ax.barh(edges[:-1], sign * counts, height=widths, align='edge',
                             color=c, alpha=alpha, label=str(level))
            else:
                self.ax.bar(edges[:-1], sign * counts, width=widths, align='edge',
                            color=c, alpha=alpha, label=str(level))
        zero = self.ax.axvline if horizontal else self.ax.axhline
        zero(0, color=plt.rcParams['axes.labelcolor'], linewidth=1)
        count_axis = self.ax.xaxis if horizontal else self.ax.yaxis
        count_axis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{abs(v):.0f}'))
        (self.ax.set_ylabel if horizontal else self.ax.set_xlabel)(column)
        (self.ax.set_xlabel if horizontal else self.ax.set_ylabel)('Effectif')
        self.ax.legend()
        return self

    def contour(self, x: str, y: str, z: str, filled: bool = True, levels: int = 10,
                cmap: Optional[str] = None, show_points: bool = False,
                show_mesh: bool = False, labels: bool = True,
                colorbar: bool = True) -> 'Chart':
        """
        EN: Contour map of z over (x, y). Regular grid → contour/contourf; scattered
            points (field samples) → triangulation (tricontour/tricontourf), with the
            sample points and triangle mesh (triplot) on demand.
        FR: Carte en courbes de niveau de z selon (x, y). Grille régulière → contour /
            contourf ; points dispersés (prélèvements) → triangulation (tricontour /
            tricontourf), avec points et maillage (triplot) sur demande.

        Example / Exemple:
            chart.contour('x', 'y', 'ph', show_points=True)       # carte du pH d'une parcelle
        """
        import matplotlib.tri as mtri
        self.chart_type = 'contour'
        cmap = cmap or plt.rcParams['image.cmap']
        ink = plt.rcParams['axes.labelcolor']
        grid = self._as_grid(x, y, z)
        d = self.data[[x, y, z]].dropna()
        if grid:
            xs, ys, (Z,) = grid
            args = (xs, ys, Z)
            fill_fn, line_fn = self.ax.contourf, self.ax.contour
        else:
            tri = mtri.Triangulation(d[x], d[y])
            args = (tri, d[z])
            fill_fn, line_fn = self.ax.tricontourf, self.ax.tricontour
            if show_mesh:
                self.ax.triplot(tri, color=plt.rcParams['grid.color'], linewidth=0.6)
        if filled:
            mappable = fill_fn(*args, levels=levels, cmap=cmap)
            lines = line_fn(*args, levels=mappable.levels, colors=ink, linewidths=0.5, alpha=0.6)
        else:
            mappable = lines = line_fn(*args, levels=levels, cmap=cmap)
        if labels:
            self.ax.clabel(lines, fontsize=8, fmt='%.3g')
        if show_points:
            self.ax.scatter(d[x], d[y], s=10, color=ink, zorder=3)
        if colorbar:
            self._colorbar(mappable, z)
        self.ax.set_xlabel(x)
        self.ax.set_ylabel(y)
        self.ax.grid(False)
        return self

    def mesh(self, x: str, y: str, z: str, cmap: Optional[str] = None,
             colorbar: bool = True, show_points: bool = False) -> 'Chart':
        """
        EN: Colored cells of z over (x, y): pcolormesh on a regular grid, tripcolor
            (colored triangles) on scattered points.
        FR: Cellules colorées de z selon (x, y) : pcolormesh sur une grille régulière,
            tripcolor (triangles colorés) sur des points dispersés.

        Example / Exemple:
            chart.mesh('colonne', 'rang', 'rendement')   # rendement par micro-parcelle
        """
        self.chart_type = 'mesh'
        cmap = cmap or plt.rcParams['image.cmap']
        grid = self._as_grid(x, y, z)
        d = self.data[[x, y, z]].dropna()
        if grid:
            xs, ys, (Z,) = grid
            mappable = self.ax.pcolormesh(xs, ys, Z, cmap=cmap, shading='nearest',
                                          edgecolors=plt.rcParams['axes.facecolor'], linewidth=0.5)
        else:
            mappable = self.ax.tripcolor(d[x], d[y], d[z], cmap=cmap, shading='gouraud')
        if show_points:
            self.ax.scatter(d[x], d[y], s=8, color=plt.rcParams['axes.labelcolor'], zorder=3)
        if colorbar:
            self._colorbar(mappable, z)
        self.ax.set_xlabel(x)
        self.ax.set_ylabel(y)
        self.ax.grid(False)
        return self

    def vector_field(self, x: str, y: str, u: str, v: str, kind: str = 'quiver',
                     cmap: Optional[str] = None, density: float = 1.2,
                     colorbar: bool = True) -> 'Chart':
        """
        EN: Vector field (wind, water flow…) colored by magnitude: 'quiver' (arrows),
            'barbs' (wind barbs) or 'stream' (streamlines, regular grid only).
        FR: Champ de vecteurs (vent, écoulement…) coloré par l'intensité : 'quiver'
            (flèches), 'barbs' (barbules de vent) ou 'stream' (lignes de courant, grille
            régulière seulement).

        Args / Paramètres:
            u, v (str): EN/FR: Composantes horizontale et verticale du vecteur.
            kind (str): 'quiver' (défaut), 'barbs' ou 'stream'.

        Example / Exemple:
            chart.vector_field('lon', 'lat', 'u_vent', 'v_vent', kind='stream')
        """
        self.chart_type = 'vector_field'
        cmap = cmap or plt.rcParams['image.cmap']
        d = self.data[[x, y, u, v]].dropna()
        magnitude = np.hypot(d[u], d[v])
        if kind == 'quiver':
            mappable = self.ax.quiver(d[x], d[y], d[u], d[v], magnitude, cmap=cmap)
        elif kind == 'barbs':
            mappable = self.ax.barbs(d[x], d[y], d[u], d[v], magnitude, cmap=cmap, length=6)
        elif kind == 'stream':
            grid = self._as_grid(x, y, u, v)
            if grid is None:
                raise ValueError("kind='stream' demande une grille régulière complète de (x, y)")
            xs, ys, (U, V) = grid
            mappable = self.ax.streamplot(xs, ys, U, V, color=np.hypot(U, V), cmap=cmap,
                                          density=density, linewidth=1.2).lines
        else:
            raise ValueError("kind doit valoir 'quiver', 'barbs' ou 'stream'")
        if colorbar:
            self._colorbar(mappable, 'Intensité')
        self.ax.set_xlabel(x)
        self.ax.set_ylabel(y)
        self.ax.set_aspect('equal', adjustable='datalim')
        return self

    def scatter3d(self, x: str, y: str, z: str, group: Optional[str] = None,
                  size: float = 30, alpha: float = 0.8) -> 'Chart':
        """
        EN: 3D scatter plot, colored by group. / FR: Nuage de points 3D, coloré par groupe.

        Example / Exemple:
            chart.scatter3d('azote', 'phosphore', 'rendement', group='variete')
        """
        self.chart_type = 'scatter3d'
        ax = self._replace_axes(projection='3d')
        groups = self.data.groupby(group, sort=False) if group else [(None, self.data)]
        for (name, sub), c in zip(groups, self._palette() * 10):
            ax.scatter(sub[x], sub[y], sub[z], s=size, color=c, alpha=alpha, label=name)
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.set_zlabel(z)
        if group:
            ax.legend(title=group)
        return self

    def line3d(self, x: str, y: str, z: str, group: Optional[str] = None) -> 'Chart':
        """
        EN: 3D curve (trajectory), one per group. / FR: Courbe 3D (trajectoire), une par groupe.

        Example / Exemple:
            chart.line3d('lon', 'lat', 'altitude', group='balise')
        """
        self.chart_type = 'line3d'
        ax = self._replace_axes(projection='3d')
        groups = self.data.groupby(group, sort=False) if group else [(None, self.data)]
        for (name, sub), c in zip(groups, self._palette() * 10):
            ax.plot(sub[x], sub[y], sub[z], color=c, label=name)
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.set_zlabel(z)
        if group:
            ax.legend(title=group)
        return self

    def surface(self, x: str, y: str, z: str, kind: str = 'surface',
                cmap: Optional[str] = None, colorbar: bool = True) -> 'Chart':
        """
        EN: 3D surface of z over (x, y): 'surface' or 'wireframe' on a regular grid,
            'trisurf' (triangulated) on scattered points — chosen automatically.
        FR: Surface 3D de z selon (x, y) : 'surface' ou 'wireframe' sur une grille
            régulière, 'trisurf' (triangulée) sur des points dispersés — choix automatique.

        Example / Exemple:
            chart.surface('azote', 'eau', 'rendement')            # surface de réponse
            chart.surface('azote', 'eau', 'rendement', kind='wireframe')
        """
        self.chart_type = 'surface'
        ax = self._replace_axes(projection='3d')
        cmap = cmap or plt.rcParams['image.cmap']
        grid = self._as_grid(x, y, z) if kind != 'trisurf' else None
        mappable = None
        if grid:
            xs, ys, (Z,) = grid
            X, Y = np.meshgrid(xs, ys)
            if kind == 'wireframe':
                ax.plot_wireframe(X, Y, Z, color=self._palette()[0], linewidth=0.8)
            elif kind == 'surface':
                mappable = ax.plot_surface(X, Y, Z, cmap=cmap, edgecolor='none', alpha=0.95)
            else:
                raise ValueError("kind doit valoir 'surface', 'wireframe' ou 'trisurf'")
        else:
            d = self.data[[x, y, z]].dropna()
            mappable = ax.plot_trisurf(d[x], d[y], d[z], cmap=cmap, edgecolor='none')
        if colorbar and mappable is not None:
            cbar = self.fig.colorbar(mappable, ax=ax, shrink=0.6, pad=0.1)
            cbar.set_label(z)
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.set_zlabel(z)
        return self

    def bar3d(self, x: str, y: str, z: str, alpha: float = 0.9) -> 'Chart':
        """
        EN: 3D bars: one bar of height z per (x, y) pair; x and y can be categories.
        FR: Barres 3D : une barre de hauteur z par couple (x, y) ; x et y peuvent être
            des catégories.

        Example / Exemple:
            chart.bar3d('variete', 'site', 'rendement')
        """
        self.chart_type = 'bar3d'
        ax = self._replace_axes(projection='3d')
        d = self.data[[x, y, z]].dropna()
        xcat, ycat = pd.Categorical(d[x]), pd.Categorical(d[y])
        palette = self._palette()
        colors = [palette[k % len(palette)] for k in xcat.codes]
        ax.bar3d(xcat.codes - 0.3, ycat.codes - 0.3, 0, 0.6, 0.6, d[z], color=colors,
                 alpha=alpha, shade=True)
        ax.set_xticks(range(len(xcat.categories)), [str(c) for c in xcat.categories])
        ax.set_yticks(range(len(ycat.categories)), [str(c) for c in ycat.categories])
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.set_zlabel(z)
        return self

    def timeline(self, date: str, label: str, levels: Optional[List[float]] = None,
                 color: Optional[str] = None) -> 'Chart':
        """
        EN: Timeline: events placed on a date axis, each with a stem and its label
            (alternating above and below to avoid overlaps).
        FR: Frise chronologique : événements sur un axe de dates, chacun avec une tige
            et son libellé (alternés dessus/dessous pour éviter les chevauchements).

        Example / Exemple:
            chart.timeline('date', 'etape')     # calendrier cultural, jalons de projet
        """
        self.chart_type = 'timeline'
        d = self.data[[date, label]].dropna().sort_values(date)
        heights = levels or [3, -3, 2, -2, 1, -1]
        lv = np.resize(np.asarray(heights, dtype=float), len(d))
        c = color or self._palette()[0]
        ink = plt.rcParams['text.color']
        self.ax.vlines(d[date], 0, lv, color=c, linewidth=1.2)
        self.ax.axhline(0, color=plt.rcParams['axes.labelcolor'], linewidth=1.5, zorder=1)
        self.ax.plot(d[date], np.zeros(len(d)), 'o', color=c, markerfacecolor=plt.rcParams['axes.facecolor'],
                     markeredgewidth=1.5, zorder=3)
        for when, text, h in zip(d[date], d[label], lv):
            self.ax.annotate(str(text), xy=(when, h), xytext=(0, 4 if h > 0 else -4),
                             textcoords='offset points', ha='center',
                             va='bottom' if h > 0 else 'top', fontsize=9, color=ink)
        self.ax.yaxis.set_visible(False)
        for side in ('left', 'top', 'right'):
            self.ax.spines[side].set_visible(False)
        self.ax.grid(False)
        self.ax.margins(y=0.25)
        self._concise_dates(date)
        return self

    def confidence_ellipse(self, x: str, y: str, group: Optional[str] = None,
                           level: float = 0.95, show_points: bool = True,
                           fill: bool = True, alpha: float = 0.15) -> 'Chart':
        """
        EN: Scatter plot with one confidence ellipse per group (bivariate normal,
            covering `level` of the population: 95 % by default).
        FR: Nuage de points avec une ellipse de confiance par groupe (loi normale
            bivariée, couvrant `level` de la population : 95 % par défaut).

        Example / Exemple:
            chart.confidence_ellipse('longueur', 'largeur', group='espece')
        """
        from matplotlib.patches import Ellipse
        from scipy.stats import chi2
        self.chart_type = 'confidence_ellipse'
        scale = np.sqrt(chi2.ppf(level, 2))
        groups = self.data.groupby(group, sort=False) if group else [(None, self.data)]
        for (name, sub), c in zip(groups, self._palette() * 10):
            pts = sub[[x, y]].dropna().to_numpy(float)
            if show_points:
                self.ax.scatter(pts[:, 0], pts[:, 1], s=18, color=c, alpha=0.8, label=name)
            if len(pts) < 3:
                continue
            vals, vecs = np.linalg.eigh(np.cov(pts, rowvar=False))
            angle = np.degrees(np.arctan2(vecs[1, -1], vecs[0, -1]))  # axe principal
            w, h = 2 * scale * np.sqrt(vals[::-1])
            self.ax.add_patch(Ellipse(pts.mean(axis=0), w, h, angle=angle, edgecolor=c,
                                      facecolor=mcolors.to_rgba(c, alpha if fill else 0),
                                      linewidth=1.5))
        self.ax.autoscale_view()
        self.ax.set_xlabel(x)
        self.ax.set_ylabel(y)
        if group:
            self.ax.legend(title=group)
        return self

    def donut(self, values: str, labels: Optional[str] = None, group: Optional[str] = None,
              hole: float = 0.55, center_text: Optional[str] = None,
              show_values: bool = True) -> 'Chart':
        """
        EN: Donut chart. With group: nested rings — groups inside, their items outside
            in shades of the group color (matplotlib "nested pie").
        FR: Graphique en anneau. Avec group : anneaux imbriqués — groupes à l'intérieur,
            leurs éléments à l'extérieur dans des nuances de la couleur du groupe.

        Example / Exemple:
            chart.donut('surface', labels='culture', group='famille', center_text='1 865 ha')
        """
        self.chart_type = 'donut'
        bg = plt.rcParams['axes.facecolor']
        palette = self._palette()
        d = self.data[self.data[values] > 0]
        names = d[labels].astype(str).tolist() if labels else None
        width = 1 - hole
        pct = (lambda p: f'{p:.0f} %' if p >= 4 else '') if show_values else None
        if group is None:
            self.ax.pie(d[values], labels=names, colors=palette * 10, startangle=90,
                        counterclock=False, autopct=pct, pctdistance=1 - width / 2,
                        wedgeprops=dict(width=width, edgecolor=bg, linewidth=2))
        else:
            order = list(pd.unique(d[group]))
            d = d.assign(_g=pd.Categorical(d[group], order)).sort_values('_g', kind='stable')
            totals = d.groupby('_g', observed=True)[values].sum()
            inner = [palette[k % len(palette)] for k in range(len(order))]
            outer = []
            for k, g in enumerate(order):
                n = (d[group] == g).sum()
                outer += [mcolors.to_hex(np.array(mcolors.to_rgb(inner[k])) * (1 - t) + t)
                          for t in np.linspace(0.0, 0.55, n)]  # nuances vers le blanc
            ring = width / 2
            self.ax.pie(d[values], labels=d[labels].astype(str) if labels else None,
                        colors=outer, startangle=90, counterclock=False, radius=1,
                        wedgeprops=dict(width=ring, edgecolor=bg, linewidth=1.5))
            self.ax.pie(totals, labels=None, colors=inner, startangle=90, counterclock=False,
                        radius=1 - ring, autopct=pct, pctdistance=1 - ring / 2 / (1 - ring),
                        wedgeprops=dict(width=ring, edgecolor=bg, linewidth=1.5))
            self.ax.legend([plt.Rectangle((0, 0), 1, 1, color=c) for c in inner], order,
                           loc='center left', bbox_to_anchor=(1, 0.5), title=group)
        if center_text:
            self.ax.text(0, 0, center_text, ha='center', va='center', fontsize=14,
                         fontweight='bold', color=plt.rcParams['text.color'])
        self.ax.set_aspect('equal')
        return self

    def polar(self, theta: str, r: str, kind: str = 'bar', group: Optional[str] = None,
              degrees: bool = True, compass: bool = False, alpha: float = 0.85) -> 'Chart':
        """
        EN: Polar chart: 'bar' (rose / wind rose, stacked by group), 'line' or 'scatter'.
            theta can be categories (evenly spaced) or angles in degrees.
        FR: Graphique polaire : 'bar' (rose / rose des vents, empilée par groupe), 'line'
            ou 'scatter'. theta : catégories (réparties sur le cercle) ou angles en degrés.

        Args / Paramètres:
            compass (bool): EN/FR: 0° au nord, sens horaire (directions de vent).

        Example / Exemple:
            chart.polar('direction', 'frequence', group='vitesse', compass=True)  # rose des vents
            chart.polar('mois', 'pluie')                                         # pluie mensuelle
        """
        self.chart_type = 'polar'
        ax = self._replace_axes(projection='polar')
        if compass:
            ax.set_theta_zero_location('N')
            ax.set_theta_direction(-1)
        categorical = not pd.api.types.is_numeric_dtype(self.data[theta])
        cats = list(pd.unique(self.data[theta])) if categorical else None

        def angles(s):
            if categorical:
                return 2 * np.pi * np.array([cats.index(v) for v in s]) / len(cats)
            return np.radians(s) if degrees else np.asarray(s, float)

        n_dirs = len(cats) if categorical else max(self.data[theta].nunique(), 1)
        width = 2 * np.pi / n_dirs * 0.9
        groups = list(self.data.groupby(group, sort=False)) if group else [(None, self.data)]
        bottom = pd.Series(0.0, index=pd.unique(self.data[theta]))
        for (name, sub), c in zip(groups, self._palette() * 10):
            sub = sub.groupby(theta, sort=False)[r].sum()
            th = angles(sub.index)
            if kind == 'bar':
                base = bottom.reindex(sub.index).fillna(0).to_numpy()
                ax.bar(th, sub.to_numpy(), width=width, bottom=base, color=c, alpha=alpha,
                       edgecolor=plt.rcParams['axes.facecolor'], linewidth=1, label=name)
                bottom = bottom.add(sub, fill_value=0)
            elif kind in ('line', 'scatter'):
                order = np.argsort(th)
                tt, rr = th[order], sub.to_numpy()[order]
                if kind == 'line':
                    ax.plot(np.append(tt, tt[0]), np.append(rr, rr[0]), color=c, label=name)
                else:
                    ax.scatter(tt, rr, color=c, alpha=alpha, label=name)
            else:
                raise ValueError("kind doit valoir 'bar', 'line' ou 'scatter'")
        if categorical:
            ax.set_xticks(2 * np.pi * np.arange(len(cats)) / len(cats), [str(v) for v in cats])
        if group:
            ax.legend(title=group, loc='upper left', bbox_to_anchor=(1.05, 1.0))
        return self

    def pivot_heatmap(self, x: str, y: str, value: str, aggfunc: str = 'mean',
                      annot: bool = True, fmt: str = '.1f', cmap: Optional[str] = None) -> 'Chart':
        """
        EN: Annotated heatmap of a value aggregated by two categories (pivot table),
            e.g. mean yield by variety × site. Table in self.pivot_table.
        FR: Carte de chaleur annotée d'une valeur agrégée selon deux catégories
            (tableau croisé), ex. rendement moyen par variété × site. Tableau dans self.pivot_table.

        Example / Exemple:
            chart.pivot_heatmap('site', 'variete', 'rendement')
        """
        self.chart_type = 'pivot_heatmap'
        table = pd.pivot_table(self.data, index=y, columns=x, values=value, aggfunc=aggfunc)
        sns.heatmap(table, ax=self.ax, annot=annot, fmt=fmt, cmap=cmap or plt.rcParams['image.cmap'],
                    linewidths=1, linecolor=plt.rcParams['axes.facecolor'],
                    cbar_kws={'label': f'{value} ({aggfunc})'})
        self.ax.set_xlabel(x)
        self.ax.set_ylabel(y)
        self.pivot_table = table
        return self

    def radial_bar(self, values: str, labels: Optional[str] = None, max_value: float = 100,
                   width: float = 0.12, gap: float = 0.04, show_values: bool = True) -> 'Chart':
        """
        EN: Radial bars: one concentric arc per row, its length = value / max_value of a
            full turn (progress of several indicators).
        FR: Barres radiales : un arc concentrique par ligne, de longueur valeur / max_value
            d'un tour complet (avancement de plusieurs indicateurs).

        Example / Exemple:
            chart.radial_bar('avancement', labels='activite')     # en % par défaut
        """
        from matplotlib.patches import Wedge
        self.chart_type = 'radial_bar'
        track = plt.rcParams['grid.color']
        names = (self.data[labels].astype(str) if labels else self.data.index.astype(str)).tolist()
        handles = []
        for k, (name, v, c) in enumerate(zip(names, self.data[values], self._palette() * 10)):
            radius = 1 - k * (width + gap)
            if radius - width <= 0:
                raise ValueError("Trop de barres pour ce rayon : réduire width ou gap")
            frac = float(np.clip(v / max_value, 0, 1))
            self.ax.add_patch(Wedge((0, 0), radius, 0, 360, width=width, color=track, alpha=0.6))
            handles.append(self.ax.add_patch(Wedge((0, 0), radius, 90 - 360 * frac, 90,
                                                   width=width, color=c)))
        legend = [f'{n} : {v:g}' + (' %' if max_value == 100 else '') if show_values else n
                  for n, v in zip(names, self.data[values])]
        self.ax.legend(handles, legend, loc='upper center', bbox_to_anchor=(0.5, 0.0),
                       ncol=min(len(names), 3), frameon=False)
        self.ax.set_xlim(-1.05, 1.05)
        self.ax.set_ylim(-1.05, 1.05)
        self.ax.set_aspect('equal')
        self.ax.axis('off')
        return self

    def progress_bar(self, value: Union[str, float], total: float = 100,
                     labels: Optional[str] = None, color: Optional[str] = None,
                     height: float = 0.5) -> 'Chart':
        """
        EN: Progress bar(s): a filled bar over a light track. value is a number (one bar)
            or a column (one bar per row, named by labels).
        FR: Barre(s) de progression : une barre pleine sur un rail clair. value est un
            nombre (une barre) ou une colonne (une barre par ligne, nommée par labels).

        Example / Exemple:
            Chart(df).progress_bar(80, total=100)                       # 80 % collectés
            chart.progress_bar('realise', total=100, labels='objectif')
        """
        self.chart_type = 'progress_bar'
        if isinstance(value, str):
            vals = self.data[value].to_numpy(float)
            names = (self.data[labels].astype(str) if labels else self.data.index.astype(str)).tolist()
        else:
            vals, names = np.array([float(value)]), ['']
        pos = np.arange(len(vals))[::-1]
        palette = self._palette()
        self.ax.barh(pos, [total] * len(vals), height=height, color=plt.rcParams['grid.color'],
                     alpha=0.7)
        fill = [color or palette[k % len(palette)] for k in range(len(vals))]
        self.ax.barh(pos, np.clip(vals, 0, total), height=height, color=fill)
        for p, v in zip(pos, vals):
            self.ax.text(total * 1.01, p, f'{100 * v / total:.0f} %', va='center', fontsize=11,
                         color=plt.rcParams['text.color'])
        self.ax.set_yticks(pos, names)
        self.ax.set_xlim(0, total * 1.12)
        self.ax.xaxis.set_visible(False)
        self.ax.grid(False)
        for side in ('top', 'right', 'bottom'):
            self.ax.spines[side].set_visible(False)
        if names == ['']:
            self.ax.spines['left'].set_visible(False)
        return self

    # ──────────────────────────────────────── Climate & soil / Climat & sols
    # Couleurs conventionnelles par défaut (température rouge, pluie bleue) : elles portent
    # le sens du graphique ; chacune reste réglable.

    def walter_lieth(self, month: str, temp: str, precip: str, tmin: Optional[str] = None,
                     station: Optional[str] = None, temp_color: str = '#d1495b',
                     precip_color: str = '#2e86de') -> 'Chart':
        """
        EN: Walter–Lieth climate diagram: 10 °C ↔ 20 mm, precipitation scale compressed
            1:10 above 100 mm (solid fill); humid season shaded blue, arid season dotted
            red; frost months (tmin < 0) marked below the axis. Summary in self.climate_summary.
        FR: Climatogramme de Walter-Lieth : 10 °C ↔ 20 mm, échelle des pluies comprimée au
            1/10 au-delà de 100 mm (aplat) ; saison humide en bleu, saison sèche pointillée
            en rouge ; mois de gel (tmin < 0) sous l'axe. Résumé dans self.climate_summary.

        Example / Exemple:
            chart.walter_lieth('mois', 'temperature', 'pluie', station='Station A — 350 m')
        """
        self.chart_type = 'walter_lieth'
        d = self.data
        n = len(d)
        pos = np.arange(n)
        T = d[temp].to_numpy(float)
        P = d[precip].to_numpy(float)
        to_y = lambda p: np.where(p <= 100, p / 2, 50 + (p - 100) / 20)  # noqa: E731
        Py = to_y(P)
        xf = np.linspace(0, n - 1, 400)
        Tf, Pf = np.interp(xf, pos, T), np.interp(xf, pos, Py)
        self.ax.fill_between(xf, Tf, np.minimum(Pf, 50), where=Pf >= Tf, interpolate=True,
                             color=precip_color, alpha=0.25, linewidth=0, label='Saison humide')
        self.ax.fill_between(xf, Pf, Tf, where=Tf > Pf, interpolate=True, facecolor='none',
                             edgecolor=temp_color, hatch='...', linewidth=0, label='Saison sèche')
        self.ax.fill_between(xf, 50, Pf, where=Pf > 50, interpolate=True, color=precip_color,
                             alpha=0.85, linewidth=0, label='Pluie > 100 mm (1/10)')
        self.ax.plot(pos, T, color=temp_color, linewidth=2, label='Température')
        self.ax.plot(pos, Py, color=precip_color, linewidth=2, label='Précipitations')
        low = min(0.0, np.floor(T.min() / 10) * 10)
        high = max(50.0, np.ceil(max(T.max(), Py.max()) / 10) * 10)
        self.ax.set_ylim(low, high)
        self.ax.set_xlim(0, n - 1)
        self.ax.set_xticks(pos, d[month].astype(str).tolist())
        self.ax.set_ylabel('Température (°C)', color=temp_color)
        ax2 = self.ax.twinx()
        ax2.set_ylim(low, high)
        ticks = [y for y in np.arange(0, high + 1, 10) if y >= 0]
        labels = [f'{y * 2:.0f}' if y <= 50 else f'{100 + (y - 50) * 20:.0f}' for y in ticks]
        ax2.set_yticks(ticks, labels)
        ax2.set_ylabel('Précipitations (mm)', color=precip_color)
        ax2.grid(False)
        ax2.spines['right'].set_visible(True)
        if tmin is not None:
            frost = d[tmin].to_numpy(float) < 0
            for k in np.flatnonzero(frost):
                self.ax.axvspan(k - 0.5, k + 0.5, ymin=0, ymax=0.03, color='#5dade2')
        self.ax.text(0.99, 1.02, f'{T.mean():.1f} °C   {P.sum():.0f} mm', transform=self.ax.transAxes,
                     ha='right', va='bottom', fontsize=10, color=plt.rcParams['text.color'])
        if station:
            self.ax.text(0.01, 1.02, station, transform=self.ax.transAxes, ha='left', va='bottom',
                         fontsize=10, color=plt.rcParams['text.color'])
        self.ax.legend(loc='upper left', fontsize=8)
        self.ax2 = ax2
        self.climate_summary = {'t_moyenne': float(T.mean()), 'p_totale': float(P.sum()),
                                'mois_secs': d.loc[2 * T > P, month].tolist()}
        return self

    def climogram(self, precip: str, temp: str, month: Optional[str] = None,
                  group: Optional[str] = None, labels: bool = True) -> 'Chart':
        """
        EN: Climograph (hythergraph): monthly temperature against precipitation, joined into
            a closed loop — one loop per station to compare climates.
        FR: Climogramme (hythergramme) : température mensuelle selon les précipitations,
            points reliés en boucle fermée — une boucle par station pour comparer des climats.

        Example / Exemple:
            chart.climogram('pluie', 'temperature', month='mois', group='station')
        """
        self.chart_type = 'climogram'
        groups = self.data.groupby(group, sort=False) if group else [(None, self.data)]
        for (name, sub), c in zip(groups, self._palette() * 10):
            x = np.append(sub[precip].to_numpy(float), sub[precip].iloc[0])
            y = np.append(sub[temp].to_numpy(float), sub[temp].iloc[0])
            self.ax.plot(x, y, marker='o', markersize=5, color=c, label=name)
            if labels and month:
                for xm, ym, m in zip(sub[precip], sub[temp], sub[month]):
                    self.ax.annotate(str(m), (xm, ym), xytext=(4, 4), textcoords='offset points',
                                     fontsize=7, color=c)
        self.ax.set_xlabel('Précipitations (mm)')
        self.ax.set_ylabel('Température (°C)')
        if group:
            self.ax.legend(title=group)
        return self

    def ecoclimatogram(self, humidity: str, temp: str, month: Optional[str] = None,
                       group: Optional[str] = None, optimum: Optional[Tuple] = None,
                       tolerance: Optional[Tuple] = None, labels: bool = True,
                       zone_color: str = '#2e9e5b') -> 'Chart':
        """
        EN: Ecoclimatogram (Bodenheimer): monthly mean temperature against relative humidity,
            joined into a closed loop (one per station or year), compared with the optimum
            zone of a species (pest, locust, crop…) and optionally its wider tolerance zone.
            Zones are (humidity min, humidity max, temperature min, temperature max).
            Month-by-month position in `chart.ecoclimate` ('optimale', 'favorable',
            'défavorable').
        FR: Écoclimatogramme (Bodenheimer) : température moyenne mensuelle selon l'humidité
            relative, points reliés en boucle fermée (une par station ou par année), comparés
            à la zone optimale d'une espèce (ravageur, criquet, culture…) et éventuellement à
            sa zone de tolérance, plus large. Zones : (humidité min, humidité max,
            température min, température max). Position de chaque mois dans
            `chart.ecoclimate` ('optimale', 'favorable', 'défavorable').

        Args / Paramètres:
            humidity (str): EN/FR: Colonne de l'humidité relative moyenne mensuelle (%).
            temp (str): EN/FR: Colonne de la température moyenne mensuelle (°C).
            month (str, optional): EN/FR: Colonne des mois, écrits à côté des points.
            group (str, optional): EN/FR: Colonne des stations (ou années) : une boucle par groupe.
            optimum (tuple, optional): EN/FR: Zone optimale (hum_min, hum_max, t_min, t_max).
            tolerance (tuple, optional): EN/FR: Zone de tolérance, plus large (même format) : mois « favorables ».
            labels (bool): EN/FR: Écrit les mois à côté des points (avec `month`).
            zone_color (str): EN/FR: Couleur des zones optimale et de tolérance (distincte des courbes).

        Example / Exemple:
            chart.ecoclimatogram('humidite', 'temperature', month='mois',
                                 optimum=(60, 80, 24, 30), tolerance=(40, 90, 18, 34))
        """
        self.chart_type = 'ecoclimatogram'
        ink = plt.rcParams['axes.labelcolor']
        p = self._palette()
        for zone, alpha, name in ((tolerance, 0.10, 'Zone de tolérance'), (optimum, 0.22, 'Zone optimale')):
            if zone is None:
                continue
            if len(zone) != 4 or zone[0] >= zone[1] or zone[2] >= zone[3]:
                raise ValueError(f'{name} : (hum_min, hum_max, t_min, t_max) avec min < max attendu')
            h0, h1, t0, t1 = zone
            self.ax.add_patch(plt.Rectangle((h0, t0), h1 - h0, t1 - t0, facecolor=zone_color,
                                            alpha=alpha, edgecolor=zone_color, lw=1.2,
                                            ls='-' if zone is optimum else '--', zorder=0,
                                            label=name))

        def inside(zone, h, t):
            return zone is not None and zone[0] <= h <= zone[1] and zone[2] <= t <= zone[3]

        rows = []
        groups = self.data.groupby(group, sort=False) if group else [(None, self.data)]
        for (name, sub), c in zip(groups, p * 10):
            h, t = sub[humidity].to_numpy(float), sub[temp].to_numpy(float)
            self.ax.plot(np.append(h, h[0]), np.append(t, t[0]), color=c, lw=1.8,
                         marker='o', markersize=5, label=name, zorder=3)
            months = sub[month].astype(str).tolist() if month else [str(k + 1) for k in range(len(sub))]
            for hm, tm, mo in zip(h, t, months):
                where = ('optimale' if inside(optimum, hm, tm)
                         else 'favorable' if inside(tolerance, hm, tm) else 'défavorable')
                rows.append({'groupe': name, 'mois': mo, humidity: hm, temp: tm, 'zone': where})
                if labels and month:
                    self.ax.annotate(mo, (hm, tm), xytext=(4, 4), textcoords='offset points',
                                     fontsize=7, color=c, path_effects=self._halo(), zorder=4)
        self.ecoclimate = pd.DataFrame(rows)
        if not group:
            self.ecoclimate = self.ecoclimate.drop(columns='groupe')
        self.ax.set_xlabel('Humidité relative (%)')
        self.ax.set_ylabel('Température (°C)')
        if group or optimum is not None or tolerance is not None:
            self.ax.legend(loc='best', fontsize=8)
        return self

    # Étages et variantes du climagramme d'Emberger (échelle de Q2 la plus citée en
    # bioclimatologie méditerranéenne ; les limites exactes varient selon les auteurs).
    EMBERGER_STAGES = ('Saharien', 'Aride', 'Semi-aride', 'Subhumide', 'Humide', 'Perhumide')
    EMBERGER_LIMITS = (10, 45, 70, 110, 150)
    EMBERGER_VARIANTS = ('très froid', 'froid', 'frais', 'tempéré', 'chaud', 'très chaud')
    EMBERGER_M = (-3, 0, 3, 7, 10)

    def emberger(self, precip: str, tmax: str, tmin: str, station: Optional[str] = None,
                 monthly: bool = False, formula: str = 'emberger',
                 limits: Optional[List] = None, labels: bool = True,
                 log: bool = False) -> 'Chart':
        """
        EN: Emberger climagram: pluviothermic quotient Q2 against m (mean minimum of the
            coldest month), with the bioclimatic stages (Saharan to per-humid) and the winter
            variants (very cold to very hot). Q2 = 2000·P / (M² − m²), temperatures in kelvin
            (formula='emberger'), or Stewart's Q2 = 3.43·P / (M − m) (formula='stewart').
            One row per station (P annual, M, m), or monthly data with monthly=True
            (P = sum, M = highest monthly max, m = lowest monthly min).
            Stage limits: Q2 = 10, 45, 70, 110, 150 by default; `limits` accepts numbers or
            (a, b) pairs for oblique lines Q2 = a + b·m. Table in `chart.emberger_table`.
        FR: Climagramme d'Emberger : quotient pluviothermique Q2 selon m (minimum moyen du
            mois le plus froid), avec les étages bioclimatiques (saharien à perhumide) et les
            variantes thermiques (hiver très froid à très chaud). Q2 = 2000·P / (M² − m²),
            températures en kelvin (formula='emberger'), ou Q2 = 3,43·P / (M − m) de Stewart
            (formula='stewart'). Une ligne par station (P annuel, M, m), ou données mensuelles
            avec monthly=True (P = somme, M = plus fort maximum mensuel, m = plus faible
            minimum mensuel). Limites des étages : Q2 = 10, 45, 70, 110, 150 par défaut ;
            `limits` accepte des nombres ou des couples (a, b) pour des droites obliques
            Q2 = a + b·m. Tableau dans `chart.emberger_table`.

        Args / Paramètres:
            precip (str): EN/FR: Colonne des précipitations (mm) : annuelles, ou mensuelles avec monthly=True.
            tmax (str): EN/FR: Colonne M (°C) : moyenne des maxima du mois le plus chaud, ou maxima mensuels.
            tmin (str): EN/FR: Colonne m (°C) : moyenne des minima du mois le plus froid, ou minima mensuels.
            station (str, optional): EN/FR: Colonne du nom des stations (une ligne, ou 12 avec monthly=True).
            monthly (bool): EN/FR: Données mensuelles : P, M et m sont calculés par station.
            formula (str): EN/FR: 'emberger' (2000·P / (M² − m²), en kelvin) ou 'stewart' (3,43·P / (M − m)).
            limits (list, optional): EN/FR: 5 limites entre étages : nombres (Q2 constant) ou couples (a, b) pour Q2 = a + b·m.
            labels (bool): EN/FR: Écrit le nom des stations à côté des points.
            log (bool): EN/FR: Axe de Q2 en échelle logarithmique.

        Example / Exemple:
            chart.emberger('pluie', 'M', 'm', station='station')
            chart.emberger('pluie', 'tmax', 'tmin', station='station', monthly=True)
        """
        if formula not in ('emberger', 'stewart'):
            raise ValueError("formula doit valoir 'emberger' ou 'stewart'")
        limits = list(limits) if limits is not None else list(self.EMBERGER_LIMITS)
        if len(limits) != len(self.EMBERGER_STAGES) - 1:
            raise ValueError(f'limits : {len(self.EMBERGER_STAGES) - 1} limites attendues '
                             f'(entre {", ".join(self.EMBERGER_STAGES)})')
        self.chart_type = 'emberger'
        d = self.data
        if monthly:
            if not station:
                d = d.assign(_station='Station')
            key = station or '_station'
            d = d.groupby(key, sort=False).agg(P=(precip, 'sum'), M=(tmax, 'max'),
                                               m=(tmin, 'min')).reset_index()
            names = d[key].astype(str).tolist()
        else:
            d = pd.DataFrame({'P': d[precip], 'M': d[tmax], 'm': d[tmin]}).reset_index(drop=True)
            names = (self.data[station].astype(str).tolist() if station
                     else [str(k + 1) for k in range(len(d))])
        P, M, m = (d[c].to_numpy(float) for c in ('P', 'M', 'm'))
        if np.any(M <= m):
            raise ValueError('M (maximum du mois le plus chaud) doit dépasser m (minimum du mois le plus froid)')
        q2 = 2000 * P / ((M + 273.15) ** 2 - (m + 273.15) ** 2) if formula == 'emberger' \
            else 3.43 * P / (M - m)

        def limit(k, x):  # limite k entre deux étages, à la température m = x
            lim = limits[k]
            return np.full_like(np.asarray(x, float), lim, dtype=float) if np.isscalar(lim) \
                else lim[0] + lim[1] * np.asarray(x, float)

        stage = np.sum([q2 >= limit(k, m) for k in range(len(limits))], axis=0)
        variant = np.digitize(m, self.EMBERGER_M)
        self.emberger_table = pd.DataFrame({
            'station': names, 'P': P, 'M': M, 'm': m, 'Q2': q2.round(1),
            'etage': [self.EMBERGER_STAGES[k] for k in stage],
            'variante': [f'hiver {self.EMBERGER_VARIANTS[k]}' for k in variant]})
        self.emberger_table['bioclimat'] = (self.emberger_table['etage'] + ' à '
                                            + self.emberger_table['variante'])

        # ── Fond : étages (bandes colorées) et variantes (verticales)
        x0, x1 = min(-6.0, m.min() - 1.5), max(14.0, m.max() + 1.5)
        ytop = max(200.0, q2.max() * 1.2)
        ybot = 1.0 if log else 0.0
        xs = np.linspace(x0, x1, 50)
        bounds = [np.full_like(xs, ybot)] + [limit(k, xs) for k in range(len(limits))] \
            + [np.full_like(xs, ytop)]
        shades = matplotlib.colormaps['BrBG'](np.linspace(0.28, 0.78, len(self.EMBERGER_STAGES)))
        ink = plt.rcParams['axes.labelcolor']
        for k, name in enumerate(self.EMBERGER_STAGES):
            lo, hi = np.clip(bounds[k], ybot, ytop), np.clip(bounds[k + 1], ybot, ytop)
            self.ax.fill_between(xs, lo, hi, color=shades[k], alpha=0.35, linewidth=0, zorder=0)
            mid = np.sqrt(lo[-1] * hi[-1]) if log else (lo[-1] + hi[-1]) / 2
            if hi[-1] > lo[-1]:
                self.ax.text(x1 - 0.2, mid, name, ha='right', va='center', fontsize=8,
                             color=ink, fontweight='bold', alpha=0.8, zorder=1)
        for k in range(len(limits)):
            self.ax.plot(xs, limit(k, xs), color=ink, lw=0.7, alpha=0.6, zorder=1)
        for t in self.EMBERGER_M:
            self.ax.axvline(t, color=ink, lw=0.6, ls='--', alpha=0.5, zorder=1)
        edges = [x0, *self.EMBERGER_M, x1]
        for (a, b), name in zip(zip(edges, edges[1:]), self.EMBERGER_VARIANTS):
            self.ax.text((a + b) / 2, 1.01, name, transform=self.ax.get_xaxis_transform(),
                         ha='center', va='bottom', fontsize=7, color=ink)
        # ── Stations
        colors = self._palette()
        for k, (x, y, name) in enumerate(zip(m, q2, names)):
            self.ax.scatter(x, y, s=55, color=colors[k % len(colors)], zorder=3,
                            edgecolors=plt.rcParams['axes.facecolor'], linewidths=1.2)
            if labels:
                self.ax.annotate(name, (x, y), xytext=(6, 4), textcoords='offset points',
                                 fontsize=8, path_effects=self._halo(), zorder=4)
        if log:
            self.ax.set_yscale('log')
            self.ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f'{v:g}'))
        self.ax.set_xlim(x0, x1)
        self.ax.set_ylim(ybot, ytop)
        self.ax.grid(False)
        self.ax.set_xlabel('m : minimum moyen du mois le plus froid (°C)')
        self.ax.set_ylabel('Q2' + (' (Stewart)' if formula == 'stewart' else ''))
        return self

    def water_balance(self, month: str, precip: str, etp: str, show_bars: bool = True,
                      surplus_color: str = '#2e86de', deficit_color: str = '#e67e22') -> 'Chart':
        """
        EN: Simplified water balance: precipitation against potential evapotranspiration
            (PET), surplus (P > PET) and deficit (P < PET) shaded. Table in self.water_table.
        FR: Bilan hydrique simplifié : précipitations face à l'évapotranspiration potentielle
            (ETP), excédent (P > ETP) et déficit (P < ETP) colorés. Tableau dans self.water_table.

        Example / Exemple:
            chart.water_balance('mois', 'pluie', 'etp')
        """
        self.chart_type = 'water_balance'
        d = self.data
        n = len(d)
        pos = np.arange(n)
        P, E = d[precip].to_numpy(float), d[etp].to_numpy(float)
        if show_bars:
            self.ax.bar(pos, P, width=0.6, color=surplus_color, alpha=0.2, label='Précipitations')
        xf = np.linspace(0, n - 1, 400)
        Pf, Ef = np.interp(xf, pos, P), np.interp(xf, pos, E)
        self.ax.fill_between(xf, Ef, Pf, where=Pf >= Ef, interpolate=True, color=surplus_color,
                             alpha=0.35, linewidth=0, label='Excédent')
        self.ax.fill_between(xf, Pf, Ef, where=Ef > Pf, interpolate=True, color=deficit_color,
                             alpha=0.35, linewidth=0, label='Déficit')
        self.ax.plot(pos, P, color=surplus_color, marker='o', markersize=4, label='P')
        self.ax.plot(pos, E, color=deficit_color, marker='o', markersize=4, label='ETP')
        self.ax.set_xticks(pos, d[month].astype(str).tolist())
        self.ax.set_ylabel('mm')
        self.ax.legend(fontsize=8, ncol=3)
        bilan = P - E
        self.water_table = pd.DataFrame({month: d[month].to_numpy(), 'P': P, 'ETP': E,
                                         'bilan': bilan})
        self.water_summary = {'excedent': float(bilan[bilan > 0].sum()),
                              'deficit': float(-bilan[bilan < 0].sum())}
        return self

    def degree_days(self, date: str, tmin: str, tmax: str, base: float = 10.0,
                    cap: Optional[float] = None, stages: Optional[Dict[str, float]] = None,
                    color: Optional[str] = None) -> 'Chart':
        """
        EN: Cumulative growing degree days, GDD = max(0, (Tmax + Tmin) / 2 − base), with
            Tmax capped at `cap`; `stages` {name: GDD} marks the date each stage is reached.
            Daily series in self.gdd, stage dates in self.stage_dates.
        FR: Degrés-jours de croissance cumulés, DJ = max(0, (Tmax + Tmin) / 2 − base), Tmax
            plafonnée à `cap` ; `stages` {nom: DJ} marque la date d'atteinte de chaque stade.
            Série journalière dans self.gdd, dates des stades dans self.stage_dates.

        Example / Exemple:
            chart.degree_days('jour', 'tmin', 'tmax', base=10, stages={'Floraison': 900})
        """
        self.chart_type = 'degree_days'
        d = self.data[[date, tmin, tmax]].dropna().sort_values(date)
        hi = d[tmax].clip(upper=cap) if cap is not None else d[tmax]
        daily = ((hi + d[tmin]) / 2 - base).clip(lower=0)
        cumul = daily.cumsum()
        c = color or self._palette()[0]
        self.ax.plot(d[date], cumul, color=c, linewidth=2.2)
        self.ax.fill_between(d[date], 0, cumul, color=c, alpha=0.12, linewidth=0)
        self.stage_dates = {}
        ink = plt.rcParams['text.color']
        for name, threshold in (stages or {}).items():
            reached = d.loc[cumul >= threshold, date]
            when = reached.iloc[0] if len(reached) else None
            self.stage_dates[name] = when
            self.ax.axhline(threshold, color=plt.rcParams['axes.labelcolor'], linewidth=0.8,
                            linestyle='--', alpha=0.6)
            if when is not None:
                self.ax.scatter([when], [threshold], color=c, zorder=3, s=30)
                label = f'{name}\n{when:%d/%m}' if hasattr(when, 'strftime') else f'{name}\n{when}'
                self.ax.annotate(label, (when, threshold), xytext=(-6, 6), textcoords='offset points',
                                 ha='right', va='bottom', fontsize=8, color=ink)
        self.ax.set_ylabel(f'Degrés-jours cumulés (base {base:g} °C)')
        self.ax.set_xlabel(date)
        self._concise_dates(date)
        self.gdd = pd.DataFrame({date: d[date].to_numpy(), 'dj': daily.to_numpy(),
                                 'dj_cumules': cumul.to_numpy()})
        return self

    def wind_rose(self, direction: str, speed: str, sectors: int = 16,
                  speed_bins: Optional[List[float]] = None, unit: str = 'm/s',
                  cmap: Optional[str] = None) -> 'Chart':
        """
        EN: Wind rose from raw observations: directions (degrees, 0 = north) grouped into
            `sectors`, speeds into classes, bars stacked in % of observations. Table in
            self.wind_table.
        FR: Rose des vents à partir des observations brutes : directions (degrés, 0 = nord)
            regroupées en `sectors` secteurs, vitesses en classes, barres empilées en % des
            observations. Tableau dans self.wind_table.

        Example / Exemple:
            chart.wind_rose('direction', 'vitesse', speed_bins=[0, 2, 4, 6, 10])
        """
        self.chart_type = 'wind_rose'
        d = self.data[[direction, speed]].dropna()
        width = 360 / sectors
        sector = (((d[direction] % 360) + width / 2) // width % sectors).astype(int)
        bins = list(speed_bins) if speed_bins else [0, 2, 4, 6, 8]
        edges = bins + [np.inf]
        names = [f'{a:g}–{b:g}' for a, b in zip(bins[:-1], bins[1:])] + [f'≥ {bins[-1]:g}']
        classes = pd.cut(d[speed], edges, right=False, labels=names)
        table = (pd.crosstab(sector, classes, normalize='all') * 100).reindex(
            index=range(sectors), columns=names, fill_value=0)
        ax = self._replace_axes(projection='polar')
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)
        theta = np.radians(np.arange(sectors) * width)
        colors = plt.get_cmap(cmap or plt.rcParams['image.cmap'])(np.linspace(0.25, 0.95, len(names)))
        bottom = np.zeros(sectors)
        for name, c in zip(names, colors):
            ax.bar(theta, table[name], width=np.radians(width) * 0.92, bottom=bottom, color=c,
                   edgecolor=plt.rcParams['axes.facecolor'], linewidth=0.6, label=name)
            bottom += table[name].to_numpy()
        ax.set_xticks(np.radians(np.arange(0, 360, 45)), ['N', 'NE', 'E', 'SE', 'S', 'SO', 'O', 'NO'])
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.0f} %'))
        ax.legend(title=f'Vitesse ({unit})', loc='upper left', bbox_to_anchor=(1.08, 1.0), fontsize=8)
        self.wind_table = table
        return self

    def anomaly(self, x: str, value: str, baseline: Optional[Union[float, Tuple]] = None,
                window: Optional[int] = None, pos_color: str = '#d6604d',
                neg_color: str = '#4393c3') -> 'Chart':
        """
        EN: Anomaly bars: departure of each value from a reference — the mean (default),
            a number, or the mean over a period (start, end) of x. Optional rolling mean.
        FR: Barres d'anomalies : écart de chaque valeur à une référence — la moyenne (défaut),
            un nombre, ou la moyenne sur une période (début, fin) de x. Moyenne glissante en option.

        Example / Exemple:
            chart.anomaly('annee', 'temperature', baseline=(1961, 1990), window=10)
        """
        self.chart_type = 'anomaly'
        d = self.data[[x, value]].dropna()
        if baseline is None:
            ref = d[value].mean()
        elif isinstance(baseline, tuple):
            ref = d.loc[d[x].between(*baseline), value].mean()
        else:
            ref = float(baseline)
        an = d[value] - ref
        self.ax.bar(d[x], an, color=np.where(an >= 0, pos_color, neg_color), width=0.8)
        self.ax.axhline(0, color=plt.rcParams['axes.labelcolor'], linewidth=1)
        if window:
            self.ax.plot(d[x], an.rolling(window, center=True, min_periods=1).mean(),
                         color=plt.rcParams['text.color'], linewidth=2,
                         label=f'Moyenne glissante ({window})')
            self.ax.legend()
        self.ax.set_ylabel(f'Écart à la référence ({value})')
        self.ax.set_xlabel(x)
        self.anomalies = pd.Series(an.to_numpy(), index=d[x].to_numpy(), name='anomalie')
        self.anomaly_reference = float(ref)
        return self

    def climate_stripes(self, x: str, value: str, cmap: str = 'RdBu_r',
                        show_years: bool = True) -> 'Chart':
        """
        EN: Climate stripes: one vertical stripe per period (year), colored by the departure
            from the mean on a diverging scale (cold blue → warm red).
        FR: Bandes climatiques : une bande verticale par période (année), colorée selon
            l'écart à la moyenne sur une échelle divergente (bleu froid → rouge chaud).

        Example / Exemple:
            chart.climate_stripes('annee', 'temperature')
        """
        self.chart_type = 'climate_stripes'
        d = self.data[[x, value]].dropna()
        an = (d[value] - d[value].mean()).to_numpy()
        lim = np.abs(an).max() or 1
        colors = plt.get_cmap(cmap)(mcolors.Normalize(-lim, lim)(an))
        pos = np.arange(len(d))
        self.ax.bar(pos, 1, width=1.0, color=colors, linewidth=0)
        self.ax.set_xlim(-0.5, len(d) - 0.5)
        self.ax.set_ylim(0, 1)
        self.ax.yaxis.set_visible(False)
        for side in self.ax.spines.values():
            side.set_visible(False)
        self.ax.grid(False)
        if show_years:
            step = max(1, len(d) // 8)
            self.ax.set_xticks(pos[::step], d[x].iloc[::step].astype(str).tolist())
        else:
            self.ax.xaxis.set_visible(False)
        return self

    def soil_profile(self, top: str, bottom: str, horizon: str, profile: Optional[str] = None,
                     color: Optional[str] = None, width: float = 0.6,
                     show_labels: bool = True) -> 'Chart':
        """
        EN: Soil profile diagram: horizons stacked by depth (cm, increasing downward), one
            column per profile. `color` is an optional column of colors (e.g. from Munsell
            notations converted to hex); otherwise each horizon name gets a theme color.
        FR: Profil de sol : horizons empilés selon la profondeur (cm, vers le bas), une colonne
            par profil. `color` : colonne de couleurs optionnelle (ex. notations Munsell
            converties en hex) ; sinon chaque horizon reçoit une couleur du thème.

        Example / Exemple:
            chart.soil_profile('haut', 'bas', 'horizon', profile='fosse')
        """
        self.chart_type = 'soil_profile'
        groups = list(self.data.groupby(profile, sort=False)) if profile else [('', self.data)]
        palette = self._palette()
        names = list(pd.unique(self.data[horizon]))
        default = {h: palette[k % len(palette)] for k, h in enumerate(names)}
        ink = plt.rcParams['axes.labelcolor']
        for k, (_, sub) in enumerate(groups):
            for _, row in sub.iterrows():
                face = row[color] if color else default[row[horizon]]
                self.ax.add_patch(plt.Rectangle((k - width / 2, row[top]), width,
                                                row[bottom] - row[top], facecolor=face,
                                                edgecolor=ink, linewidth=0.8))
                if show_labels and row[bottom] - row[top] > 0:
                    self.ax.text(k, (row[top] + row[bottom]) / 2, str(row[horizon]),
                                 ha='center', va='center', fontsize=9, color=self._ink_on(face))
        depth = self.data[bottom].max()
        self.ax.set_ylim(depth * 1.03, 0)
        self.ax.set_xlim(-0.6, len(groups) - 0.4)
        self.ax.set_xticks(range(len(groups)), [str(n) for n, _ in groups])
        self.ax.set_ylabel('Profondeur (cm)')
        self.ax.grid(False, axis='x')
        return self

    def depth_profile(self, depth: str, value: str, group: Optional[str] = None,
                      marker: str = 'o') -> 'Chart':
        """
        EN: A variable along depth (soil moisture, organic matter, temperature…): depth on
            a downward vertical axis, the variable read on top. One curve per group.
        FR: Une variable selon la profondeur (humidité, matière organique, température…) :
            profondeur sur un axe vertical vers le bas, variable lue en haut. Une courbe par groupe.

        Example / Exemple:
            chart.depth_profile('profondeur', 'humidite', group='date')
        """
        self.chart_type = 'depth_profile'
        groups = self.data.groupby(group, sort=False) if group else [(None, self.data)]
        for (name, sub), c in zip(groups, self._palette() * 10):
            sub = sub.sort_values(depth)
            self.ax.plot(sub[value], sub[depth], marker=marker, color=c, label=name)
        self.ax.invert_yaxis()
        self.ax.xaxis.set_label_position('top')
        self.ax.xaxis.tick_top()
        self.ax.set_xlabel(value)
        self.ax.set_ylabel(depth)
        if group:
            self.ax.legend(title=group)
        return self

    @staticmethod
    def _van_genuchten(h, theta_r, theta_s, alpha, n):
        """Teneur en eau θ(h) de van Genuchten (h : succion en cm, m = 1 - 1/n)."""
        return theta_r + (theta_s - theta_r) / (1 + (alpha * h) ** n) ** (1 - 1 / n)

    def retention_curve(self, suction: str, water: str, group: Optional[str] = None,
                        unit: str = 'pF', fit: bool = True, field_capacity: float = 2.5,
                        wilting_point: float = 4.2) -> 'Chart':
        """
        EN: Soil water retention curve (pF curve): water content against suction (pF), with
            field capacity, wilting point and available water between them. Optional van
            Genuchten fit (scipy). `unit`: 'pF', 'kPa' or 'cm' (suction head).
            Results in `chart.retention` (one row per group).
        FR: Courbe de rétention d'eau (courbe pF) : teneur en eau selon la succion (pF), avec
            capacité au champ, point de flétrissement et réserve utile entre les deux.
            Ajustement de van Genuchten optionnel (scipy). `unit` : 'pF', 'kPa' ou 'cm'.
            Résultats dans `chart.retention` (une ligne par groupe).

        Example / Exemple:
            chart.retention_curve('pF', 'theta', group='horizon')
        """
        from scipy.optimize import curve_fit
        if unit not in ('pF', 'kPa', 'cm'):
            raise ValueError("unit doit valoir 'pF', 'kPa' ou 'cm'")
        self.chart_type = 'retention_curve'
        groups = list(self.data.groupby(group, sort=False)) if group else [(None, self.data)]
        rows = []
        for (name, sub), c in zip(groups, self._palette() * 10):
            s = sub[suction].to_numpy(float)
            pf = s if unit == 'pF' else np.log10(s * (10.197 if unit == 'kPa' else 1.0))
            theta = sub[water].to_numpy(float)
            order = np.argsort(pf)
            pf, theta = pf[order], theta[order]
            self.ax.plot(theta, pf, 'o', color=c, label=None if fit else name)
            row = {'groupe': name}
            params = None
            if fit and len(pf) >= 4:
                try:
                    params, _ = curve_fit(self._van_genuchten, 10 ** pf, theta,
                                          p0=[theta.min() * 0.5, theta.max(), 0.02, 1.5],
                                          bounds=([0, 0, 1e-5, 1.01], [theta.min(), 1.5 * theta.max() + 1e-9, 10, 10]),
                                          maxfev=20000)
                except RuntimeError:
                    warnings.warn(f"Ajustement de van Genuchten impossible ({name}) : interpolation utilisée")
            if params is not None:
                grid = np.linspace(min(pf.min(), 0), max(pf.max(), wilting_point), 200)
                self.ax.plot(self._van_genuchten(10 ** grid, *params), grid, color=c, label=name)
                fc, wp = (self._van_genuchten(10 ** p, *params) for p in (field_capacity, wilting_point))
                row.update(theta_r=params[0], theta_s=params[1], alpha=params[2], n=params[3])
            else:
                if fit and len(pf) < 4:
                    warnings.warn(f"Moins de 4 points ({name}) : pas d'ajustement, interpolation utilisée")
                self.ax.plot(theta, pf, color=c, alpha=0.6)
                fc, wp = (np.interp(p, pf, theta) for p in (field_capacity, wilting_point))
            row.update(capacite_champ=fc, point_fletrissement=wp, reserve_utile=fc - wp)
            rows.append(row)
        self.retention = pd.DataFrame(rows)
        if not group:
            self.retention = self.retention.drop(columns='groupe')
        ink = plt.rcParams['axes.labelcolor']
        for p, txt in ((field_capacity, f'Capacité au champ (pF {field_capacity:g})'),
                       (wilting_point, f'Point de flétrissement (pF {wilting_point:g})')):
            self.ax.axhline(p, color=ink, ls='--', lw=0.8, alpha=0.7)
            self.ax.annotate(txt, (1, p), xycoords=('axes fraction', 'data'), ha='right',
                             va='bottom', fontsize=8, color=ink)
        if not group:  # une seule courbe : la réserve utile est un intervalle lisible
            r = self.retention.iloc[0]
            self.ax.axvspan(r['point_fletrissement'], r['capacite_champ'],
                            color=self._palette()[0], alpha=0.12, label='Réserve utile')
            self.ax.legend()
        else:
            self.ax.legend(title=group)
        self.ax.set_xlabel(water)
        self.ax.set_ylabel('Succion (pF)')
        return self

    def grain_size_curve(self, diameter: str, passing: str, group: Optional[str] = None,
                         limits: Tuple[float, ...] = (0.002, 0.05, 2.0),
                         names: Tuple[str, ...] = ('Argile', 'Limon', 'Sable', 'Graviers')) -> 'Chart':
        """
        EN: Cumulative grain-size distribution: % passing (finer than) against particle
            diameter (mm, log scale), with clay / silt / sand / gravel limits. D10, D30, D60,
            uniformity Cu = D60/D10 and curvature Cc = D30²/(D10·D60) in `chart.grain_size`.
        FR: Courbe granulométrique cumulative : % de passants (plus fins) selon le diamètre
            (mm, échelle log), avec les limites argile / limon / sable / graviers. D10, D30, D60,
            uniformité Cu = D60/D10 et courbure Cc = D30²/(D10·D60) dans `chart.grain_size`.

        Example / Exemple:
            chart.grain_size_curve('diametre', 'passant', group='echantillon')
        """
        self.chart_type = 'grain_size_curve'
        groups = list(self.data.groupby(group, sort=False)) if group else [(None, self.data)]
        rows = []
        for (name, sub), c in zip(groups, self._palette() * 10):
            sub = sub.sort_values(diameter)
            d, p = sub[diameter].to_numpy(float), sub[passing].to_numpy(float)
            self.ax.plot(d, p, marker='o', ms=4, color=c, label=name)
            # Dxx : interpolation en log(diamètre) ; hors des mesures → NaN
            dx = {k: (10 ** np.interp(k, p, np.log10(d)) if p.min() <= k <= p.max() else np.nan)
                  for k in (10, 30, 60)}
            rows.append({'groupe': name, 'D10': dx[10], 'D30': dx[30], 'D60': dx[60],
                         'Cu': dx[60] / dx[10], 'Cc': dx[30] ** 2 / (dx[10] * dx[60])})
        self.grain_size = pd.DataFrame(rows)
        if not group:
            self.grain_size = self.grain_size.drop(columns='groupe')
        self.ax.set_xscale('log')
        self.ax.set_ylim(0, 100)
        ink = plt.rcParams['axes.labelcolor']
        lo, hi = self.ax.get_xlim()
        bounds = [lo, *limits, hi]
        for b in limits:
            self.ax.axvline(b, color=ink, lw=0.8, ls='--', alpha=0.6)
        for a, b, txt in zip(bounds, bounds[1:], names):
            if b > lo and a < hi:
                mid = np.sqrt(max(a, lo) * min(b, hi))
                self.ax.text(mid, 1.01, txt, transform=self.ax.get_xaxis_transform(),
                             ha='center', va='bottom', fontsize=9, color=ink)
        self.ax.set_xlim(lo, hi)
        self.ax.set_xlabel('Diamètre des particules (mm)')
        self.ax.set_ylabel('Passants cumulés (%)')
        if group:  # en bas à droite : zone vide d'une courbe cumulative
            self.ax.legend(title=group, loc='lower right')
        return self

    def exchange_complex(self, horizon: str, cations: List[str], acidity: Optional[str] = None,
                         cec: Optional[str] = None, show_saturation: bool = True) -> 'Chart':
        """
        EN: Exchange complex per horizon: exchangeable base cations (Ca, Mg, K, Na… in
            cmol+/kg) stacked as horizontal bars, optional exchangeable acidity (hatched) and
            CEC marker. Base saturation V = S/T (S = sum of bases, T = CEC, or S + acidity)
            is written at the end of each bar and stored in `chart.base_saturation`.
        FR: Complexe d'échange par horizon : cations basiques échangeables (Ca, Mg, K, Na… en
            cmol+/kg) en barres horizontales empilées, acidité échangeable (hachurée) et CEC
            optionnelles. Le taux de saturation V = S/T (S = somme des bases, T = CEC, ou
            S + acidité) est écrit au bout de chaque barre et stocké dans `chart.base_saturation`.

        Example / Exemple:
            chart.exchange_complex('horizon', ['Ca', 'Mg', 'K', 'Na'], acidity='H_Al', cec='CEC')
        """
        self.chart_type = 'exchange_complex'
        d = self.data
        y = np.arange(len(d))
        left = np.zeros(len(d))
        for col, c in zip(cations, self._palette() * 10):
            self.ax.barh(y, d[col], left=left, color=c, label=col)
            left += d[col].to_numpy(float)
        s = left.copy()
        if acidity:
            self.ax.barh(y, d[acidity], left=left, facecolor='none', hatch='///',
                         edgecolor=plt.rcParams['axes.labelcolor'], lw=0.6, label=acidity)
            left += d[acidity].to_numpy(float)
        t = d[cec].to_numpy(float) if cec else (left if acidity else None)
        if cec:
            self.ax.scatter(t, y, marker='|', s=500, lw=2.5, zorder=3,
                            color=plt.rcParams['axes.labelcolor'], label=f'{cec} (T)')
        if t is not None:
            self.base_saturation = pd.Series(100 * s / t, index=d[horizon].to_numpy(), name='V (%)')
            if show_saturation:
                for yi, xi, v in zip(y, np.maximum(left, t), self.base_saturation):
                    self.ax.text(xi, yi, f'  V = {v:.0f} %', va='center', fontsize=9,
                                 color=plt.rcParams['axes.labelcolor'])
                self.ax.set_xlim(0, np.nanmax(np.maximum(left, t)) * 1.18)
        self.ax.set_yticks(y, d[horizon].astype(str).tolist())
        self.ax.invert_yaxis()  # horizon de surface en haut
        self.ax.set_xlabel('cmol+/kg')
        self.ax.grid(False, axis='y')
        handles, labels = self.ax.get_legend_handles_labels()
        if cec:  # la CEC (premier artiste ajouté en scatter) en fin de légende
            k = labels.index(f'{cec} (T)')
            handles.append(handles.pop(k))
            labels.append(labels.pop(k))
        self.ax.legend(handles, labels, ncol=min(len(labels), 6), frameon=False,
                       **self._legend_below())
        return self

    def infiltration_curve(self, time: str, cumulative: str, model: str = 'philip',
                           show_rate: bool = True) -> 'Chart':
        """
        EN: Infiltration test: cumulative infiltration against time with a fitted model —
            'philip' I = S·√t + A·t, or 'horton' I = fc·t + (f0 − fc)(1 − e^(−kt))/k — and the
            infiltration rate on a right-hand axis. Parameters and final rate (A or fc, in
            units of `cumulative` per unit of `time`) in `chart.infiltration`.
        FR: Essai d'infiltration : infiltration cumulée selon le temps avec un modèle ajusté —
            'philip' I = S·√t + A·t, ou 'horton' I = fc·t + (f0 − fc)(1 − e^(−kt))/k — et la
            vitesse d'infiltration sur un axe de droite. Paramètres et vitesse finale (A ou fc,
            en unités de `cumulative` par unité de `time`) dans `chart.infiltration`.

        Example / Exemple:
            chart.infiltration_curve('minutes', 'infiltration_mm', model='horton')
        """
        if model not in ('philip', 'horton'):
            raise ValueError("model doit valoir 'philip' ou 'horton'")
        self.chart_type = 'infiltration_curve'
        d = self.data.sort_values(time)
        t, cum = d[time].to_numpy(float), d[cumulative].to_numpy(float)
        grid = np.linspace(0, t.max(), 300)
        p = self._palette()
        if model == 'philip':  # linéaire en (√t, t) : moindres carrés directs
            (S, A), *_ = np.linalg.lstsq(np.column_stack([np.sqrt(t), t]), cum, rcond=None)
            fit_cum = S * np.sqrt(grid) + A * grid
            with np.errstate(divide='ignore'):
                fit_rate = S / (2 * np.sqrt(grid)) + A
            self.infiltration = {'model': 'philip', 'S': float(S), 'A': float(A), 'final_rate': float(A)}
        else:
            from scipy.optimize import curve_fit

            def horton(tt, f0, fc, k):
                return fc * tt + (f0 - fc) * (1 - np.exp(-k * tt)) / k
            slope = np.gradient(cum, t)
            (f0, fc, k), _ = curve_fit(horton, t, cum, p0=[slope[0], slope[-1], 1 / t.mean()],
                                       bounds=([0, 0, 1e-6], [np.inf, np.inf, np.inf]), maxfev=20000)
            fit_cum = horton(grid, f0, fc, k)
            fit_rate = fc + (f0 - fc) * np.exp(-k * grid)
            self.infiltration = {'model': 'horton', 'f0': float(f0), 'fc': float(fc), 'k': float(k), 'final_rate': float(fc)}
        self.ax.plot(t, cum, 'o', color=p[0], label='Cumul mesuré')
        self.ax.plot(grid, fit_cum, color=p[0], label=f'Modèle de {model.capitalize()}')
        self.ax.set_xlabel(time)
        self.ax.set_ylabel(cumulative)
        self.ax.set_xlim(0, t.max() * 1.02)
        self.ax.set_ylim(0, None)
        handles, labels = self.ax.get_legend_handles_labels()
        if show_rate:
            c2 = p[1 % len(p)]
            self.ax2 = self.ax.twinx()
            self.ax2.grid(False)
            mid = (t[1:] + t[:-1]) / 2
            self.ax2.plot(mid, np.diff(cum) / np.diff(t), 's', ms=4, color=c2, alpha=0.7,
                          label='Vitesse mesurée')
            self.ax2.plot(grid[1:], fit_rate[1:], color=c2, ls='--', label='Vitesse modélisée')
            self.ax2.axhline(self.infiltration['final_rate'], color=c2, lw=0.8, ls=':')
            self.ax2.set_ylim(0, np.diff(cum)[0] / np.diff(t)[0] * 1.3)
            self.ax2.set_ylabel(f'Vitesse ({cumulative} / {time})')
            h2, l2 = self.ax2.get_legend_handles_labels()
            handles, labels = handles + h2, labels + l2
        self.ax.legend(handles, labels, loc='center right')
        return self

    def isopleth(self, time: str, depth: str, value: str, levels: int = 10,
                 cmap: Optional[str] = None, contour_labels: bool = True) -> 'Chart':
        """
        EN: Isopleth diagram: a soil variable (moisture, temperature…) in the depth × time
            plane, filled contours with depth increasing downward. Long-format data (one row
            per date and depth); dates are allowed on the time axis.
        FR: Diagramme des isoplèthes : une variable du sol (humidité, température…) dans le
            plan profondeur × temps, en courbes de niveau remplies, profondeur vers le bas.
            Données en format long (une ligne par date et profondeur) ; dates acceptées.

        Example / Exemple:
            chart.isopleth('date', 'profondeur', 'humidite')
        """
        import matplotlib.dates as mdates
        self.chart_type = 'isopleth'
        grid = self.data.pivot_table(index=depth, columns=time, values=value).sort_index()
        x = grid.columns
        is_date = pd.api.types.is_datetime64_any_dtype(x)
        xs = mdates.date2num(x.to_pydatetime()) if is_date else x.to_numpy(float)
        z = grid.to_numpy(float)
        cs = self.ax.contourf(xs, grid.index.to_numpy(float), z, levels=levels, cmap=cmap)
        if contour_labels:
            lines = self.ax.contour(xs, grid.index.to_numpy(float), z, levels=cs.levels,
                                    colors=plt.rcParams['axes.labelcolor'], linewidths=0.5, alpha=0.6)
            self.ax.clabel(lines, fontsize=7, fmt='%g')
        if is_date:
            self.ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=3, maxticks=7))
            self.ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(self.ax.xaxis.get_major_locator()))
        self.ax.invert_yaxis()
        self.ax.grid(False)
        self.fig.colorbar(cs, ax=self.ax, label=value)
        self.ax.set_xlabel(time)
        self.ax.set_ylabel(depth)
        return self

    def soil_properties(self, depth: str, columns: List[str], bottom: Optional[str] = None,
                        group: Optional[str] = None, marker: str = 'o') -> 'Chart':
        """
        EN: Several soil properties side by side (pH, organic matter, clay, CEC…), one panel
            per property, sharing a downward depth axis. With `bottom`, `depth` is the top of
            each horizon and values are drawn as steps over the horizon thickness.
            Panels in `chart.axes`.
        FR: Plusieurs propriétés du sol côte à côte (pH, matière organique, argile, CEC…), un
            panneau par propriété, sur un axe de profondeur commun orienté vers le bas. Avec
            `bottom`, `depth` est le haut de chaque horizon et les valeurs sont tracées en
            paliers sur l'épaisseur de l'horizon. Panneaux dans `chart.axes`.

        Example / Exemple:
            chart.soil_properties('haut', ['pH', 'MO', 'argile'], bottom='bas', group='fosse')
        """
        self.chart_type = 'soil_properties'
        spec = self.ax.get_subplotspec()
        self.ax.remove()
        sub = spec.subgridspec(1, len(columns), wspace=0.08)
        first = self.fig.add_subplot(sub[0, 0])
        self.axes = [first] + [self.fig.add_subplot(sub[0, k], sharey=first)
                               for k in range(1, len(columns))]
        self.ax = first
        self._panels = True
        groups = list(self.data.groupby(group, sort=False)) if group else [(None, self.data)]
        for ax, col in zip(self.axes, columns):
            for (name, g), c in zip(groups, self._palette() * 10):
                g = g.sort_values(depth)
                if bottom:  # paliers : (v, haut) → (v, bas) pour chaque horizon
                    ys = g[[depth, bottom]].to_numpy(float).ravel()
                    xs = np.repeat(g[col].to_numpy(float), 2)
                    ax.plot(xs, ys, color=c, label=name)
                else:
                    ax.plot(g[col], g[depth], marker=marker, color=c, label=name)
            ax.xaxis.set_label_position('top')
            ax.xaxis.tick_top()
            ax.set_xlabel(col)
            if ax is not first:
                ax.tick_params(labelleft=False)
        first.set_ylim(self.data[bottom or depth].max() * 1.03, 0)
        first.set_ylabel('Profondeur (cm)')
        if group:
            self.axes[-1].legend(title=group)
        return self

    # ──────────────────────────────────────── Graphs & networks / Graphes & réseaux

    def _graph_frame(self, source: str, target: str, weight: Optional[str], layout):
        """Nœuds, indices des arêtes et positions (layout) du réseau de self.data."""
        from . import graph as g
        nodes, i, j = g._index(self.data, source, target)
        return nodes, i, j, g.layout(nodes, i, j, layout)

    def _draw_graph(self, nodes, pos, i, j, directed=False, node_size='degree', labels=True,
                    widths=None, colors=None, styles=None, edge_text=None, node_colors=None):
        """Dessine un réseau : arêtes (flèches si orienté), nœuds dimensionnés, étiquettes."""
        ink = plt.rcParams['axes.labelcolor']
        p = self._palette()
        deg = np.bincount(np.concatenate([i, j]), minlength=len(nodes))
        if node_size == 'degree':
            sizes = 120 + 420 * deg / max(deg.max(), 1)
        else:
            sizes = np.full(len(nodes), float(node_size))
        radius = np.sqrt(sizes) / 2                     # rayon des nœuds en points
        pairs = set(zip(i, j))
        n_e = len(i)
        widths = np.full(n_e, 1.4) if widths is None else widths
        colors = [ink] * n_e if colors is None else colors
        styles = ['-'] * n_e if styles is None else styles
        for k, (a, b) in enumerate(zip(i, j)):
            curved = directed and (b, a) in pairs       # deux sens : arcs séparés
            self.ax.annotate('', xy=pos[nodes[b]], xytext=pos[nodes[a]], zorder=1,
                             arrowprops=dict(arrowstyle='-|>' if directed else '-',
                                             color=colors[k], lw=widths[k], ls=styles[k],
                                             shrinkA=radius[a], shrinkB=radius[b] + 1,
                                             mutation_scale=10 + 2 * widths[k],
                                             connectionstyle=f"arc3,rad={0.15 if curved else 0}"))
            if edge_text is not None and edge_text[k] != '':
                (xa, ya), (xb, yb) = pos[nodes[a]], pos[nodes[b]]
                off = 0.08 if curved else 0.0
                self.ax.text((xa + xb) / 2 + off * (yb - ya), (ya + yb) / 2 - off * (xb - xa),
                             edge_text[k], ha='center', va='center', fontsize=8, color=ink,
                             path_effects=self._halo(), zorder=2)
        xy = np.array([pos[n] for n in nodes])
        faces = node_colors or [p[0]] * len(nodes)
        self.ax.scatter(xy[:, 0], xy[:, 1], s=sizes, c=faces, zorder=3,
                        edgecolors=plt.rcParams['axes.facecolor'], linewidths=1.5)
        if labels:  # sous le nœud, avec un halo : lisible quelle que soit la longueur du nom
            for (x, y), n, r in zip(xy, nodes, radius):
                self.ax.annotate(str(n), (x, y), xytext=(0, -r - 2), textcoords='offset points',
                                 ha='center', va='top', fontsize=8, zorder=4,
                                 path_effects=self._halo())
        self.ax.margins(0.15)
        self.ax.set_aspect('equal', adjustable='datalim')
        self.ax.set_axis_off()
        self.graph = {'nodes': nodes, 'positions': pos,
                      'degree': pd.Series(deg, index=nodes, name='degre')}

    @staticmethod
    def _fmt(v) -> str:
        return f'{v:g}' if isinstance(v, (int, float, np.number)) else str(v)

    def network(self, source: str, target: str, weight: Optional[str] = None,
                directed: bool = False, layout: Union[str, Dict] = 'spring',
                node_size: Union[str, float] = 'degree', labels: bool = True,
                edge_labels: bool = False, max_width: float = 6.0) -> 'Chart':
        """
        EN: Network graph from an edge list (one row per link): nodes sized by degree, edge
            width proportional to `weight`, arrows if `directed`. `layout`: 'spring',
            'circular' or {node: (x, y)} (e.g. real coordinates). Degrees in `chart.graph`.
        FR: Graphe de réseau à partir d'une liste d'arêtes (une ligne par lien) : nœuds
            dimensionnés selon leur degré, épaisseur des liens proportionnelle à `weight`,
            flèches si `directed`. `layout` : 'spring', 'circular' ou {nœud: (x, y)}
            (ex. coordonnées réelles). Degrés dans `chart.graph`.

        Example / Exemple:
            chart.network('de', 'vers', weight='echanges', directed=True)
        """
        self.chart_type = 'network'
        nodes, i, j, pos = self._graph_frame(source, target, weight, layout)
        w = self.data[weight].to_numpy(float) if weight else None
        widths = 0.8 + (max_width - 0.8) * w / w.max() if weight else None
        text = [self._fmt(v) for v in self.data[weight]] if (weight and edge_labels) else None
        self._draw_graph(nodes, pos, i, j, directed, node_size, labels, widths=widths,
                         edge_text=text)
        return self

    def shortest_path(self, source: str, target: str, start, end, weight: Optional[str] = None,
                      directed: bool = False, layout: Union[str, Dict] = 'spring',
                      edge_labels: bool = True, color: Optional[str] = None) -> 'Chart':
        """
        EN: Shortest path (Dijkstra) between two nodes, highlighted on the network; the other
            links are faded. Path in `chart.path`, length in `chart.path_length`.
        FR: Plus court chemin (Dijkstra) entre deux nœuds, mis en évidence sur le réseau ; les
            autres liens sont estompés. Chemin dans `chart.path`, longueur dans `chart.path_length`.

        Example / Exemple:
            chart.shortest_path('de', 'vers', 'Dakar', 'Kédougou', weight='km')
        """
        from . import graph as g
        self.chart_type = 'shortest_path'
        self.path, self.path_length = g.shortest_path(self.data, source, target, start, end,
                                                      weight, directed)
        nodes, i, j, pos = self._graph_frame(source, target, weight, layout)
        p = self._palette()
        hi = color or p[1 % len(p)]
        steps = set(zip(self.path, self.path[1:]))
        on = [(nodes[a], nodes[b]) in steps or (not directed and (nodes[b], nodes[a]) in steps)
              for a, b in zip(i, j)]
        ink = plt.rcParams['axes.labelcolor']
        faded = mcolors.to_hex(mcolors.to_rgba(ink, 0.35), keep_alpha=True)
        text = [self._fmt(v) for v in self.data[weight]] if (weight and edge_labels) else None
        node_colors = [hi if n in self.path else p[0] for n in nodes]
        self._draw_graph(nodes, pos, i, j, directed, 'degree', True,
                         widths=np.where(on, 4.0, 1.0), colors=[hi if o else faded for o in on],
                         edge_text=text, node_colors=node_colors)
        self.ax.plot([], [], color=hi, lw=4, label=f'Plus court chemin : {self.path_length:g}')
        self.ax.legend(loc='best')
        return self

    def spanning_tree(self, source: str, target: str, weight: Optional[str] = None,
                      layout: Union[str, Dict] = 'spring', edge_labels: bool = True,
                      color: Optional[str] = None) -> 'Chart':
        """
        EN: Minimum spanning tree: the cheapest set of links connecting all nodes (network
            design: pipes, cables, tracks). Tree links bold, others dashed. Tree in
            `chart.tree`, total weight in `chart.tree_weight`.
        FR: Arbre couvrant minimal : l'ensemble de liens le moins coûteux qui relie tous les
            nœuds (conception de réseaux : canalisations, câbles, pistes). Liens de l'arbre en
            gras, autres en pointillés. Arbre dans `chart.tree`, poids total dans `chart.tree_weight`.

        Example / Exemple:
            chart.spanning_tree('de', 'vers', weight='cout')
        """
        from . import graph as g
        self.chart_type = 'spanning_tree'
        self.tree, self.tree_weight = g.spanning_tree(self.data, source, target, weight)
        nodes, i, j, pos = self._graph_frame(source, target, weight, layout)
        hi = color or self._palette()[1 % len(self._palette())]
        links = set(zip(self.tree[source], self.tree[target]))
        on = [(nodes[a], nodes[b]) in links or (nodes[b], nodes[a]) in links for a, b in zip(i, j)]
        ink = plt.rcParams['axes.labelcolor']
        text = [self._fmt(v) for v in self.data[weight]] if (weight and edge_labels) else None
        self._draw_graph(nodes, pos, i, j, False, 'degree', True,
                         widths=np.where(on, 3.5, 1.0), colors=[hi if o else ink for o in on],
                         styles=['-' if o else ':' for o in on], edge_text=text)
        self.ax.plot([], [], color=hi, lw=3.5, label=f'Arbre couvrant : {self.tree_weight:g}')
        self.ax.plot([], [], color=ink, lw=1, ls=':', label='Liens non retenus')
        self.ax.legend(loc='best')
        return self

    def max_flow(self, source: str, target: str, capacity: str, start, end,
                 layout: Union[str, Dict] = 'spring', max_width: float = 7.0,
                 color: Optional[str] = None) -> 'Chart':
        """
        EN: Maximum flow from `start` to `end` in a directed network with capacities (water,
            traffic, supply). Arrow width ∝ flow, label flow/capacity; saturated links (flow =
            capacity, the bottlenecks) in the accent color. Value in `chart.flow_value`,
            flows per link in `chart.flow`.
        FR: Flot maximal de `start` vers `end` dans un réseau orienté à capacités (eau, trafic,
            approvisionnement). Épaisseur ∝ flux, étiquette flux/capacité ; liens saturés
            (flux = capacité, les goulots d'étranglement) en couleur d'accent. Valeur dans
            `chart.flow_value`, flux par lien dans `chart.flow`.

        Example / Exemple:
            chart.max_flow('de', 'vers', 'capacite', 'Forage', 'Village')
        """
        from . import graph as g
        self.chart_type = 'max_flow'
        self.flow_value, self.flow = g.max_flow(self.data, source, target, capacity, start, end)
        nodes, i, j, pos = self._graph_frame(source, target, capacity, layout)
        p = self._palette()
        hi = color or p[1 % len(p)]
        f, c = self.flow['flux'].to_numpy(float), self.data[capacity].to_numpy(float)
        ink = plt.rcParams['axes.labelcolor']
        full = np.isclose(f, c) & (f > 0)
        colors = [hi if s else (p[0] if v > 0 else mcolors.to_hex(mcolors.to_rgba(ink, .35), keep_alpha=True))
                  for s, v in zip(full, f)]
        self._draw_graph(nodes, pos, i, j, True, 'degree', True,
                         widths=0.8 + (max_width - 0.8) * f / max(f.max(), 1e-12), colors=colors,
                         edge_text=[f'{self._fmt(a)}/{self._fmt(b)}' for a, b in zip(np.round(f, 3), c)],
                         node_colors=[hi if n in (start, end) else p[0] for n in nodes])
        self.ax.plot([], [], color=p[0], lw=3, label='Flux (flux/capacité)')
        self.ax.plot([], [], color=hi, lw=3, label='Lien saturé')
        self.ax.set_title(f'Flot maximal : {self.flow_value:g}', loc='right', fontsize=9)
        self.ax.legend(loc='best')
        return self

    def critical_path(self, task: str, duration: str, predecessors: str,
                      show_float: bool = True, color: Optional[str] = None) -> 'Chart':
        """
        EN: Critical path (CPM / PERT) as a Gantt chart: each task from its earliest start,
            total float hatched up to its latest finish, critical tasks (zero float) in the
            accent color, dependency arrows. Schedule in `chart.schedule`, duration in
            `chart.project_duration`.
        FR: Chemin critique (CPM / PERT) en diagramme de Gantt : chaque tâche à sa date de
            début au plus tôt, marge totale hachurée jusqu'à sa fin au plus tard, tâches
            critiques (marge nulle) en couleur d'accent, flèches de dépendance. Planning dans
            `chart.schedule`, durée dans `chart.project_duration`.

        Example / Exemple:
            chart.critical_path('tache', 'duree', 'avant')
        """
        from . import graph as g
        self.chart_type = 'critical_path'
        self.schedule, self.project_duration = g.critical_path(self.data, task, duration, predecessors)
        s = self.schedule
        p = self._palette()
        hi = color or p[1 % len(p)]
        ink = plt.rcParams['axes.labelcolor']
        y = {n: k for k, n in enumerate(s.index)}
        self.ax.barh(list(y.values()), s['duree'], left=s['debut_tot'], height=0.55,
                     color=[hi if c else p[0] for c in s['critique']], zorder=2)
        if show_float:
            m = s[s['marge'] > 1e-9]
            self.ax.barh([y[n] for n in m.index], m['marge'], left=m['fin_tot'], height=0.55,
                         facecolor='none', edgecolor=ink, hatch='///', lw=0.6, zorder=2)
        for n, row in s.iterrows():  # dépendance : flèche verticale depuis la fin du prédécesseur
            for q in row['predecesseurs']:
                xq, sign = s.at[q, 'fin_tot'], np.sign(y[n] - y[q])
                self.ax.annotate('', xy=(xq, y[n] - 0.3 * sign), xytext=(xq, y[q] + 0.3 * sign),
                                 zorder=1, arrowprops=dict(arrowstyle='-|>', color=ink, lw=0.7,
                                                           alpha=0.6, shrinkA=0, shrinkB=0))
                if row['debut_tot'] > xq + 1e-9:     # attente avant le début de la tâche
                    self.ax.plot([xq, row['debut_tot']], [y[n] - 0.3 * sign] * 2, color=ink,
                                 lw=0.7, ls=':', alpha=0.6, zorder=1)
        self.ax.set_yticks(list(y.values()), list(y))
        self.ax.invert_yaxis()
        self.ax.axvline(self.project_duration, color=ink, ls='--', lw=0.8)
        self.ax.set_xlabel(f'Temps ({duration})')
        self.ax.grid(False, axis='y')
        handles = [plt.Rectangle((0, 0), 1, 1, color=hi, label='Tâche critique'),
                   plt.Rectangle((0, 0), 1, 1, color=p[0], label='Tâche non critique')]
        if show_float:
            handles.append(plt.Rectangle((0, 0), 1, 1, facecolor='none', edgecolor=ink,
                                         hatch='///', label='Marge totale'))
        self.ax.legend(handles=handles, ncol=3, frameon=False, **self._legend_below())
        self.ax.set_xlim(0, self.project_duration * 1.04)
        return self

    def tsp_route(self, x: str, y: str, labels: Optional[str] = None, start=None,
                  closed: bool = True, color: Optional[str] = None) -> 'Chart':
        """
        EN: Traveling salesman tour: shortest round trip visiting every point once (nearest
            neighbour + 2-opt heuristic, Euclidean distance). Visit numbers on the points.
            Order in `chart.tour` (labels or row indices), length in `chart.tour_length`;
            column `ordre` added to `chart.tour_table` (usable with route_map).
        FR: Tournée du voyageur de commerce : le plus court circuit passant une fois par
            chaque point (heuristique plus proche voisin + 2-opt, distance euclidienne).
            Numéros de passage sur les points. Ordre dans `chart.tour` (étiquettes ou indices),
            longueur dans `chart.tour_length` ; colonne `ordre` dans `chart.tour_table`
            (réutilisable avec route_map).

        Example / Exemple:
            chart.tsp_route('x', 'y', labels='client', start='Dépôt')
        """
        from . import graph as g
        self.chart_type = 'tsp_route'
        d = self.data.reset_index(drop=True)
        names = d[labels].tolist() if labels else list(d.index)
        s0 = names.index(start) if start is not None else 0
        order, self.tour_length = g.tsp(d[[x, y]].to_numpy(float), s0, closed)
        self.tour = [names[k] for k in order]
        self.tour_table = d.iloc[order].assign(ordre=range(1, len(order) + 1))
        p = self._palette()
        c = color or p[0]
        pts = d[[x, y]].to_numpy(float)[order + ([order[0]] if closed else [])]
        for (xa, ya), (xb, yb) in zip(pts, pts[1:]):
            self.ax.annotate('', xy=(xb, yb), xytext=(xa, ya), zorder=1,
                             arrowprops=dict(arrowstyle='-|>', color=c, lw=1.8,
                                             shrinkA=7, shrinkB=7, mutation_scale=12))
        self.ax.scatter(d[x], d[y], s=60, color=c, zorder=2,
                        edgecolors=plt.rcParams['axes.facecolor'])
        self.ax.scatter(*pts[0], s=160, marker='s', color=p[1 % len(p)], zorder=3, label='Départ')
        for rank, k in enumerate(order, 1):
            txt = f'{rank}. {names[k]}' if labels else str(rank)
            self.ax.annotate(txt, (d.at[k, x], d.at[k, y]), xytext=(6, 6), textcoords='offset points',
                             fontsize=8, path_effects=self._halo(), zorder=4)
        self.ax.plot([], [], color=c, lw=1.8, label=f'Tournée : {self.tour_length:.4g}')
        self.ax.set_xlabel(x)
        self.ax.set_ylabel(y)
        self.ax.legend(loc='best')
        return self

    def arc_diagram(self, source: str, target: str, weight: Optional[str] = None,
                    order: Optional[List] = None, max_width: float = 8.0,
                    labels: bool = True) -> 'Chart':
        """
        EN: Arc diagram: nodes aligned on a line, one arc per link, width ∝ `weight`. Links
            going right are drawn above, links going left below — the direction of movements
            (migrations, transfers between sites) reads at a glance. `order`: node order.
        FR: Diagramme en arcs : nœuds alignés sur une ligne, un arc par lien, épaisseur ∝
            `weight`. Les liens vers la droite passent au-dessus, ceux vers la gauche en
            dessous — le sens des mouvements (migrations, transferts entre sites) se lit d'un
            coup d'œil. `order` : ordre des nœuds.

        Example / Exemple:
            chart.arc_diagram('depart', 'arrivee', weight='effectif')
        """
        from matplotlib.patches import Arc
        from . import graph as g
        self.chart_type = 'arc_diagram'
        nodes, i, j = g._index(self.data, source, target)
        if order is not None:
            missing = set(nodes) - set(order)
            if missing:
                raise ValueError(f'order ne contient pas : {sorted(map(str, missing))}')
            nodes = [n for n in order if n in set(nodes)]
            idx = {n: k for k, n in enumerate(nodes)}
            i = self.data[source].map(idx).to_numpy()
            j = self.data[target].map(idx).to_numpy()
        w = self.data[weight].to_numpy(float) if weight else np.ones(len(i))
        widths = 0.8 + (max_width - 0.8) * w / w.max()
        p = self._palette()
        top = 0.0
        for a, b, lw in zip(i, j, widths):
            span = abs(b - a)
            up = b > a
            self.ax.add_patch(Arc(((a + b) / 2, 0), span, span, theta1=0 if up else 180,
                                  theta2=180 if up else 360, lw=lw, alpha=0.7,
                                  color=p[0] if up else p[1 % len(p)]))
            top = max(top, span / 2)
        deg = np.bincount(np.concatenate([i, j]), minlength=len(nodes))
        self.ax.scatter(range(len(nodes)), np.zeros(len(nodes)), s=40 + 160 * deg / deg.max(),
                        color=plt.rcParams['axes.labelcolor'], zorder=3)
        if labels:
            for k, n in enumerate(nodes):
                self.ax.annotate(str(n), (k, 0), xytext=(0, -9), textcoords='offset points',
                                 ha='center', va='top', fontsize=8, path_effects=self._halo(),
                                 zorder=4)
        self.ax.set_xlim(-0.7, len(nodes) - 0.3)
        self.ax.set_ylim(-top * 1.1 - 0.5, top * 1.1 + 0.5)
        self.ax.set_aspect('equal')
        self.ax.set_axis_off()
        self.ax.plot([], [], color=p[0], lw=3, label='Vers la droite (au-dessus)')
        self.ax.plot([], [], color=p[1 % len(p)], lw=3, label='Vers la gauche (en dessous)')
        self.ax.legend(loc='upper left', frameon=False)
        return self

    # ──────────────────────────────────────── Map / Cartes

    def _halo(self):
        """Contour de la couleur du fond autour d'un texte (lisible sur un fond de carte)."""
        import matplotlib.patheffects as pe
        return [pe.withStroke(linewidth=3, foreground=plt.rcParams['axes.facecolor'])]

    def _geo_frame(self, lons, lats, pad: float = 0.08) -> None:
        """Cadre en longitude/latitude avec marge, rapport d'aspect 1/cos(latitude) et
        graduations en degrés (17° O, 14° N). Mémorise le repère pour les éléments de carte."""
        lons, lats = np.asarray(lons, float), np.asarray(lats, float)
        dx = max(np.ptp(lons), 0.05) * pad
        dy = max(np.ptp(lats), 0.05) * pad
        self.ax.set_xlim(lons.min() - dx, lons.max() + dx)
        self.ax.set_ylim(lats.min() - dy, lats.max() + dy)
        self.ax.set_aspect(1 / np.cos(np.radians(lats.mean())))
        self.ax.xaxis.set_major_locator(plt.MaxNLocator(5))   # libellés longs : peu de graduations
        self.ax.yaxis.set_major_locator(plt.MaxNLocator(6))
        self.ax.xaxis.set_major_formatter(
            plt.FuncFormatter(lambda v, _: f"{abs(v):.4g}° {'E' if v >= 0 else 'O'}"))
        self.ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda v, _: f"{abs(v):.4g}° {'N' if v >= 0 else 'S'}"))
        self.ax.set_xlabel('Longitude')
        self.ax.set_ylabel('Latitude')
        self._geo = {'units': 'deg', 'lat0': float(lats.mean()), 'crs': 'EPSG:4326'}

    def _require_geo(self, what: str) -> dict:
        geo = getattr(self, '_geo', None)
        if geo is None:
            raise ValueError(f"{what} s'utilise après une carte (map_points, choropleth, "
                             "density_map, field_map, route_map, flow_map)")
        return geo

    def _basemap(self, crs: str = 'EPSG:4326', source=None, ax=None,
                 attribution: bool = True) -> None:
        """Fond de carte (contextily) ; sans contextily ou sans réseau : avertissement, pas d'erreur.

        Défaut : Esri World Gray Canvas (thème clair) ou Esri World Imagery (thème sombre),
        utilisables sans clé d'API. `source` : nom de fournisseur xyzservices
        ('Esri.WorldTopoMap', 'OpenTopoMap'…) ou objet fournisseur (ex. avec votre clé).
        """
        try:
            import contextily as ctx
        except ImportError:
            warnings.warn("Fond de carte : installer l'option map (pip install plutovizu[map])",
                          stacklevel=3)
            return
        if isinstance(source, str):
            source = ctx.providers.query_name(source)
        dark = sum(mcolors.to_rgb(plt.rcParams['axes.facecolor'])) / 3 < 0.5
        provider = source or (ctx.providers.Esri.WorldImagery if dark
                              else ctx.providers.Esri.WorldGrayCanvas)
        ax = ax or self.ax
        if ax is self.ax and getattr(self, '_geo', None) is not None:
            self._geo['source'] = provider          # réutilisé par inset_map
        try:
            ctx.add_basemap(ax, crs=crs, source=provider, attribution_size=6,
                            attribution=None if attribution else False)
            ax.grid(False)
        except Exception as exc:  # réseau absent, serveur de tuiles indisponible…
            warnings.warn(f"Fond de carte indisponible ({exc.__class__.__name__}) : carte sans fond",
                          stacklevel=3)

    def map_points(self, lon: str, lat: str, value: Optional[str] = None,
                   size: Optional[str] = None, group: Optional[str] = None,
                   labels: Optional[str] = None, basemap: bool = True,
                   cmap: Optional[str] = None, sizes: Tuple[float, float] = (30, 500),
                   alpha: float = 0.85, basemap_source=None,
                   size_legend: bool = True) -> 'Chart':
        """
        EN: Points on a map (longitude/latitude, WGS84): colored by a value (colorbar) or a
            group (legend), sized by a column (proportional symbols, with a size legend),
            labeled, over a web basemap (contextily, optional).
        FR: Points sur une carte (longitude/latitude, WGS84) : colorés par une valeur (barre
            de couleur) ou un groupe (légende), dimensionnés par une colonne (symboles
            proportionnels, avec légende des tailles), nommés, sur un fond de carte web
            (contextily, optionnel).

        Example / Exemple:
            chart.map_points('lon', 'lat', value='pluie', labels='station')
        """
        self.chart_type = 'map_points'
        d = self.data.dropna(subset=[lon, lat])
        to_area = None
        if size:
            s = d[size].astype(float)
            lo, span = s.min(), (s.max() - s.min()) or 1
            to_area = lambda v: sizes[0] + (v - lo) / span * (sizes[1] - sizes[0])  # noqa: E731
            area = to_area(s)
        else:
            area = pd.Series(sizes[0] * 2, index=d.index)
        edge = plt.rcParams['axes.facecolor']
        if group:
            for (name, sub), c in zip(d.groupby(group, sort=False), self._palette() * 10):
                self.ax.scatter(sub[lon], sub[lat], s=area[sub.index], color=c, alpha=alpha,
                                edgecolor=edge, linewidth=0.8, label=name, zorder=3)
            self.ax.legend(title=group)
        elif value:
            sc = self.ax.scatter(d[lon], d[lat], c=d[value], s=area, alpha=alpha, zorder=3,
                                 cmap=cmap or plt.rcParams['image.cmap'], edgecolor=edge,
                                 linewidth=0.8)
            self._colorbar(sc, value)
        else:
            self.ax.scatter(d[lon], d[lat], s=area, color=self._palette()[0], alpha=alpha,
                            edgecolor=edge, linewidth=0.8, zorder=3)
        if labels:
            for x, y, t in zip(d[lon], d[lat], d[labels]):
                self.ax.annotate(str(t), (x, y), xytext=(5, 5), textcoords='offset points',
                                 fontsize=8, color=plt.rcParams['text.color'],
                                 path_effects=self._halo(), zorder=4)
        if size and size_legend:
            self._size_legend(d[size].astype(float), to_area, title=size)
        self._geo_frame(d[lon], d[lat])
        if basemap:
            self._basemap(source=basemap_source)
        return self

    def _size_legend(self, values: pd.Series, to_area, title: str, loc: str = 'upper left') -> None:
        """Légende des symboles proportionnels : 3 valeurs arrondies (min, médiane, max)."""
        nice = lambda v: float(f'{v:.2g}')  # noqa: E731
        refs = sorted({nice(values.min()), nice(values.median()), nice(values.max())})
        ink = plt.rcParams['text.color']
        handles = [self.ax.scatter([], [], s=max(to_area(v), 4), facecolor='none', edgecolor=ink,
                                   linewidth=1) for v in refs]
        previous = self.ax.get_legend()
        self.ax.legend(handles, [f'{v:,.4g}' for v in refs], title=title, loc=loc,
                       labelspacing=1.3, borderpad=0.9, handletextpad=1.2, frameon=True,
                       framealpha=0.85, facecolor=plt.rcParams['axes.facecolor'])
        if previous is not None:
            self.ax.add_artist(previous)

    @staticmethod
    def _class_breaks(values, k: int, scheme: str) -> np.ndarray:
        """Bornes de k classes : 'quantiles', 'equal' (intervalles égaux) ou 'natural'
        (seuils naturels de Jenks, programmation dynamique de Fisher)."""
        x = np.sort(np.asarray(values, float)[~np.isnan(np.asarray(values, float))])
        if scheme == 'quantiles':
            edges = np.quantile(x, np.linspace(0, 1, k + 1))
        elif scheme == 'equal':
            edges = np.linspace(x[0], x[-1], k + 1)
        elif scheme == 'natural':
            if len(x) > 1000:  # échantillon régulier : même forme, calcul rapide
                x = np.quantile(x, np.linspace(0, 1, 1000))
            n, k = len(x), min(k, len(np.unique(x)))
            s1 = np.concatenate([[0.0], np.cumsum(x)])
            s2 = np.concatenate([[0.0], np.cumsum(x ** 2)])
            D = np.full((k + 1, n), np.inf)
            B = np.zeros((k + 1, n), dtype=int)
            j = np.arange(n)
            D[1] = s2[j + 1] - s1[j + 1] ** 2 / (j + 1)        # variance intra-classe x[0..j]
            for c in range(2, k + 1):
                for jj in range(c - 1, n):
                    i = np.arange(c - 1, jj + 1)                 # début de la dernière classe
                    m = jj - i + 1
                    cost = (s2[jj + 1] - s2[i]) - (s1[jj + 1] - s1[i]) ** 2 / m
                    total = D[c - 1, i - 1] + cost
                    b = int(np.argmin(total))
                    D[c, jj], B[c, jj] = total[b], i[b]
            edges, jj = [x[-1]], n - 1
            for c in range(k, 1, -1):
                i = B[c, jj]
                edges.append(x[i - 1])
                jj = i - 1
            edges = np.array(sorted(edges + [x[0]]))
        else:
            raise ValueError("scheme doit valoir 'quantiles', 'equal' ou 'natural'")
        return np.unique(edges)

    def choropleth(self, column: str, labels: Optional[str] = None, cmap: Optional[str] = None,
                   categorical: Optional[bool] = None, legend: bool = True,
                   basemap: bool = False, edgecolor: Optional[str] = None,
                   basemap_source=None, scheme: Optional[str] = None, k: int = 5) -> 'Chart':
        """
        EN: Choropleth map: each zone (region, plot…) of a GeoDataFrame colored by a column
            — continuous (colorbar) or categorical (legend). Needs geopandas
            (pip install plutovizu[map]); read zones with geopandas.read_file(...).
        FR: Carte choroplèthe : chaque zone (région, parcelle…) d'un GeoDataFrame colorée
            selon une colonne — continue (barre de couleur) ou catégorielle (légende).
            Nécessite geopandas (pip install plutovizu[map]) ; lire les zones avec
            geopandas.read_file(...).

        Args / Paramètres:
            scheme (str, optional): EN: Discrete classes for a numeric column: 'quantiles',
                'equal' (equal intervals) or 'natural' (Jenks natural breaks); class bounds in
                self.classes. / FR: Classes de couleurs pour une colonne numérique :
                'quantiles', 'equal' (intervalles égaux) ou 'natural' (seuils naturels de
                Jenks) ; bornes dans self.classes.
            k (int): EN/FR: Nombre de classes. Défaut 5.

        Example / Exemple:
            regions = geopandas.read_file('regions.geojson')
            Chart(regions).choropleth('rendement', labels='nom')
            Chart(regions).choropleth('rendement', scheme='natural', k=4)
        """
        if not hasattr(self.data, 'geometry'):
            raise ValueError("choropleth demande un GeoDataFrame : geopandas.read_file('zones.geojson') "
                             "(pip install plutovizu[map])")
        self.chart_type = 'choropleth'
        gdf = self.data
        if categorical is None:
            categorical = not pd.api.types.is_numeric_dtype(gdf[column])
        common = dict(ax=self.ax, column=column, legend=legend, linewidth=0.8,
                      edgecolor=edgecolor or plt.rcParams['axes.facecolor'])
        if categorical:
            k = gdf[column].nunique()
            palette = (self._palette() * 10)[:k]
            gdf.plot(categorical=True, cmap=mcolors.ListedColormap(palette),
                     legend_kwds={'title': column, 'loc': 'upper left', 'bbox_to_anchor': (1, 1)},
                     **common)
        else:
            extra = {}
            if scheme:
                self.classes = self._class_breaks(gdf[column], k, scheme)
                extra['norm'] = mcolors.BoundaryNorm(self.classes, ncolors=256)
            gdf.plot(cmap=cmap or plt.rcParams['image.cmap'],
                     legend_kwds={'label': column, 'shrink': 0.7}, **common, **extra)
        if labels:
            for geom, t in zip(gdf.geometry, gdf[labels]):
                p = geom.representative_point()
                self.ax.annotate(str(t), (p.x, p.y), ha='center', va='center', fontsize=8,
                                 color=plt.rcParams['text.color'], path_effects=self._halo())
        x0, y0, x1, y1 = gdf.total_bounds
        geographic = gdf.crs is None or gdf.crs.is_geographic
        self._geo = {'units': 'deg' if geographic else 'm', 'lat0': (y0 + y1) / 2,
                     'crs': gdf.crs.to_string() if gdf.crs else 'EPSG:4326'}
        self.ax.set_axis_off()
        if basemap:
            self._basemap(crs=gdf.crs.to_string() if gdf.crs else 'EPSG:4326',
                          source=basemap_source)
        return self

    def flow_map(self, lon_from: str, lat_from: str, lon_to: str, lat_to: str,
                 weight: Optional[str] = None, label_from: Optional[str] = None,
                 label_to: Optional[str] = None, basemap: bool = True,
                 curvature: float = 0.25, color: Optional[str] = None,
                 max_width: float = 8, basemap_source=None) -> 'Chart':
        """
        EN: Flow map: one curved arrow per row from origin to destination (lon/lat), width
            proportional to `weight` — trade, transport, migrations.
        FR: Carte de flux : une flèche courbe par ligne de l'origine vers la destination
            (lon/lat), d'épaisseur proportionnelle à `weight` — échanges, transport, migrations.

        Example / Exemple:
            chart.flow_map('lon_dep', 'lat_dep', 'lon_arr', 'lat_arr', weight='tonnes',
                           label_from='depart', label_to='arrivee')
        """
        from matplotlib.patches import FancyArrowPatch
        self.chart_type = 'flow_map'
        d = self.data.dropna(subset=[lon_from, lat_from, lon_to, lat_to])
        if weight:
            w = d[weight].astype(float)
            span = (w.max() - w.min()) or 1
            widths = 1 + (w - w.min()) / span * (max_width - 1)
        else:
            widths = pd.Series(2.0, index=d.index)
        c = color or self._palette()[0]
        for i, row in d.iterrows():
            self.ax.add_patch(FancyArrowPatch(
                (row[lon_from], row[lat_from]), (row[lon_to], row[lat_to]),
                connectionstyle=f'arc3,rad={curvature}', arrowstyle='-|>',
                mutation_scale=8 + widths[i] * 1.5, linewidth=widths[i], color=c, alpha=0.75,
                zorder=3))
        ends = pd.concat([d[[lon_from, lat_from]].set_axis(['lon', 'lat'], axis=1)
                          .assign(nom=d[label_from] if label_from else None),
                          d[[lon_to, lat_to]].set_axis(['lon', 'lat'], axis=1)
                          .assign(nom=d[label_to] if label_to else None)]).drop_duplicates(['lon', 'lat'])
        self.ax.scatter(ends['lon'], ends['lat'], s=30, color=plt.rcParams['text.color'], zorder=4)
        if label_from or label_to:
            for _, e in ends.dropna(subset=['nom']).iterrows():
                self.ax.annotate(str(e['nom']), (e['lon'], e['lat']), xytext=(5, 5),
                                 textcoords='offset points', fontsize=8,
                                 color=plt.rcParams['text.color'], path_effects=self._halo(), zorder=5)
        self._geo_frame(ends['lon'], ends['lat'], pad=0.15)
        if basemap:
            self._basemap(source=basemap_source)
        return self

    def density_map(self, lon: str, lat: str, weight: Optional[str] = None,
                    bandwidth: Optional[float] = None, levels: int = 8,
                    cmap: Optional[str] = None, alpha: float = 0.6, show_points: bool = False,
                    colorbar: bool = True, basemap: bool = True, basemap_source=None) -> 'Chart':
        """
        EN: Density (hot-spot) map: kernel density of the points, optionally weighted,
            drawn as filled contours of relative density (0–1) over the basemap.
        FR: Carte de densité (zones chaudes) : densité de noyau des points, éventuellement
            pondérée, en aplats de densité relative (0–1) sur le fond de carte.

        Args / Paramètres:
            weight (str, optional): EN/FR: Poids de chaque point (ex. effectif, surface).
            bandwidth (float, optional): EN/FR: Lissage (méthode de scipy) ; défaut automatique.

        Example / Exemple:
            chart.density_map('lon', 'lat', weight='cas')     # foyers d'un ravageur
        """
        from scipy.stats import gaussian_kde
        self.chart_type = 'density_map'
        d = self.data.dropna(subset=[lon, lat])
        self._geo_frame(d[lon], d[lat], pad=0.15)
        (x0, x1), (y0, y1) = self.ax.get_xlim(), self.ax.get_ylim()
        X, Y = np.meshgrid(np.linspace(x0, x1, 200), np.linspace(y0, y1, 200))
        kde = gaussian_kde(np.vstack([d[lon], d[lat]]),
                           weights=d[weight].to_numpy(float) if weight else None,
                           bw_method=bandwidth)
        Z = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)
        Z = Z / Z.max()
        filled = self.ax.contourf(X, Y, Z, levels=np.linspace(0.08, 1, levels + 1),
                                  cmap=cmap or plt.rcParams['image.cmap'], alpha=alpha, zorder=2)
        if show_points:
            self.ax.scatter(d[lon], d[lat], s=6, color=plt.rcParams['text.color'], alpha=0.6, zorder=3)
        if colorbar:
            self._colorbar(filled, 'Densité relative')
        self.ax.set_xlim(x0, x1)
        self.ax.set_ylim(y0, y1)
        self.density_grid = (X, Y, Z)
        if basemap:
            self._basemap(source=basemap_source)
        return self

    def field_map(self, lon: str, lat: str, value: str, levels: int = 10,
                  cmap: Optional[str] = None, alpha: float = 0.65, isolines: bool = True,
                  show_points: bool = True, labels: Optional[str] = None, colorbar: bool = True,
                  basemap: bool = True, basemap_source=None) -> 'Chart':
        """
        EN: Interpolated field map from point measurements (stations): linear interpolation
            on a triangulation of the points, filled classes and labeled isolines (e.g.
            isohyets), over the basemap. Values exist only inside the hull of the points.
        FR: Carte d'une grandeur interpolée à partir de mesures ponctuelles (stations) :
            interpolation linéaire sur une triangulation des points, aplats et isolignes
            annotées (ex. isohyètes), sur le fond de carte. Valeurs seulement à l'intérieur
            de l'enveloppe des points.

        Example / Exemple:
            chart.field_map('lon', 'lat', 'pluie', labels='station')     # isohyètes
        """
        import matplotlib.tri as mtri
        self.chart_type = 'field_map'
        d = self.data.dropna(subset=[lon, lat, value])
        tri = mtri.Triangulation(d[lon], d[lat])
        self._geo_frame(d[lon], d[lat])
        ink = plt.rcParams['text.color']
        filled = self.ax.tricontourf(tri, d[value], levels=levels, alpha=alpha, zorder=2,
                                     cmap=cmap or plt.rcParams['image.cmap'])
        if isolines:
            lines = self.ax.tricontour(tri, d[value], levels=filled.levels, colors=ink,
                                       linewidths=0.6, alpha=0.8, zorder=3)
            self.ax.clabel(lines, fontsize=7, fmt='%.4g')
        if show_points:
            self.ax.scatter(d[lon], d[lat], s=14, color=ink, zorder=4)
        if labels:
            for x, y, t in zip(d[lon], d[lat], d[labels]):
                self.ax.annotate(str(t), (x, y), xytext=(4, 4), textcoords='offset points',
                                 fontsize=7, color=ink, path_effects=self._halo(), zorder=5)
        if colorbar:
            self._colorbar(filled, value)
        if basemap:
            self._basemap(source=basemap_source)
        return self

    def route_map(self, lon: str, lat: str, group: Optional[str] = None,
                  order: Optional[str] = None, labels: Optional[str] = None,
                  arrows: bool = True, linewidth: float = 2.2, basemap: bool = True,
                  basemap_source=None) -> 'Chart':
        """
        EN: Routes / tracks: points joined in order (one line per group — herd, vehicle,
            GPS track), start ● and end ■ marked, arrows showing the direction.
        FR: Itinéraires / trajets : points reliés dans l'ordre (une ligne par groupe —
            troupeau, véhicule, trace GPS), départ ● et arrivée ■ marqués, flèches de sens.

        Args / Paramètres:
            order (str, optional): EN/FR: Colonne d'ordre (date, étape) ; défaut : ordre des lignes.
            labels (str, optional): EN/FR: Nom de chaque étape.

        Example / Exemple:
            chart.route_map('lon', 'lat', group='troupeau', order='date', labels='etape')
        """
        self.chart_type = 'route_map'
        d = self.data.dropna(subset=[lon, lat])
        bg, ink = plt.rcParams['axes.facecolor'], plt.rcParams['text.color']
        groups = d.groupby(group, sort=False) if group else [(None, d)]
        for (name, sub), c in zip(groups, self._palette() * 10):
            sub = sub.sort_values(order) if order else sub
            x, y = sub[lon].to_numpy(float), sub[lat].to_numpy(float)
            self.ax.plot(x, y, color=c, linewidth=linewidth, label=name, zorder=3,
                         solid_capstyle='round', solid_joinstyle='round')
            self.ax.scatter(x[:1], y[:1], s=45, color=c, edgecolor=bg, linewidth=1, zorder=4)
            self.ax.scatter(x[-1:], y[-1:], s=55, marker='s', color=c, edgecolor=bg,
                            linewidth=1, zorder=4)
            if arrows:
                for k in range(len(x) - 1):   # tête de flèche au milieu de chaque tronçon
                    xm, ym = (x[k] + x[k + 1]) / 2, (y[k] + y[k + 1]) / 2
                    self.ax.annotate('', xy=(xm + 0.01 * (x[k + 1] - x[k]), ym + 0.01 * (y[k + 1] - y[k])),
                                     xytext=(xm, ym), zorder=4,
                                     arrowprops=dict(arrowstyle='-|>', color=c, lw=0,
                                                     mutation_scale=14))
            if labels:
                for xi, yi, t in zip(x, y, sub[labels]):
                    self.ax.annotate(str(t), (xi, yi), xytext=(5, 5), textcoords='offset points',
                                     fontsize=7.5, color=ink, path_effects=self._halo(), zorder=5)
        if group:
            self.ax.legend(title=group)
        self._geo_frame(d[lon], d[lat], pad=0.12)
        if basemap:
            self._basemap(source=basemap_source)
        return self

    # ── Éléments de carte : à chaîner après une carte

    _CORNERS = {'upper left': (0.06, 0.84), 'upper right': (0.94, 0.84),
                'lower left': (0.06, 0.08), 'lower right': (0.94, 0.08)}

    def north_arrow(self, loc: str = 'upper right', size: float = 0.08) -> 'Chart':
        """
        EN: North arrow in a corner of the map ('upper right', 'upper left', 'lower right',
            'lower left'). Assumes north is up (true for all plutovizu maps).
        FR: Flèche du nord dans un coin de la carte ('upper right', 'upper left',
            'lower right', 'lower left'). Suppose le nord en haut (vrai pour les cartes plutovizu).

        Example / Exemple:
            chart.map_points('lon', 'lat').north_arrow().scale_bar()
        """
        if loc not in self._CORNERS:
            raise ValueError(f"loc doit valoir {list(self._CORNERS)}")
        x, y = self._CORNERS[loc]
        if 'upper' in loc:
            y -= size
        ink = plt.rcParams['text.color']
        self.ax.annotate('', xy=(x, y + size), xytext=(x, y), xycoords='axes fraction',
                         arrowprops=dict(arrowstyle='-|>,head_length=0.8,head_width=0.45',
                                         color=ink, lw=2.2), zorder=7)
        self.ax.text(x, y + size + 0.015, 'N', transform=self.ax.transAxes, ha='center',
                     va='bottom', fontsize=12, fontweight='bold', color=ink,
                     path_effects=self._halo(), zorder=7)
        return self

    def scale_bar(self, length: Optional[float] = None, loc: str = 'lower left') -> 'Chart':
        """
        EN: Scale bar in km (m below 1 km). Length chosen automatically (1, 2, 5 × 10ⁿ, about
            a fifth of the map width) unless given. Correct in degrees (corrected for the
            latitude) and in projected metric coordinates.
        FR: Barre d'échelle en km (m sous 1 km). Longueur choisie automatiquement (1, 2, 5 × 10ⁿ,
            environ un cinquième de la largeur) sauf si précisée. Juste en degrés (corrigée de
            la latitude) comme en coordonnées projetées métriques.

        Example / Exemple:
            chart.choropleth('rendement').scale_bar(length=0.2)     # 200 m
        """
        from matplotlib.patches import Rectangle
        geo = self._require_geo('scale_bar')
        (x0, x1), (y0, y1) = self.ax.get_xlim(), self.ax.get_ylim()
        km_per_unit = (111.32 * np.cos(np.radians(geo['lat0'])) if geo['units'] == 'deg' else 0.001)
        if length is None:
            target = (x1 - x0) * km_per_unit / 5
            mag = 10 ** np.floor(np.log10(target))
            length = max(m for m in (1, 2, 5, 10) if m * mag <= target) * mag
        bar = length / km_per_unit
        w, h = x1 - x0, y1 - y0
        xs = x0 + 0.05 * w if 'left' in loc else x1 - 0.05 * w - bar
        ys = y0 + 0.06 * h if 'lower' in loc else y1 - 0.1 * h
        ink, bg = plt.rcParams['text.color'], plt.rcParams['axes.facecolor']
        for k in range(2):
            self.ax.add_patch(Rectangle((xs + k * bar / 2, ys), bar / 2, 0.014 * h, zorder=7,
                                        facecolor=ink if k == 0 else bg, edgecolor=ink, lw=0.8))
        label = f'{length:g} km' if length >= 1 else f'{length * 1000:g} m'
        for xt, txt in ((xs, '0'), (xs + bar, label)):
            self.ax.text(xt, ys + 0.03 * h, txt, ha='center', va='bottom', fontsize=8, color=ink,
                         path_effects=self._halo(), zorder=7)
        self.scale_length_km = float(length)
        return self

    def inset_map(self, zoom_out: float = 6.0, loc: str = 'lower right', size: float = 0.3,
                  basemap: bool = True, basemap_source=None,
                  color: Optional[str] = None) -> 'Chart':
        """
        EN: Locator inset: a small map zoomed out `zoom_out` times around the map, with a
            rectangle showing the mapped area.
        FR: Carte de situation en médaillon : une petite carte `zoom_out` fois plus large
            autour de la carte, avec un rectangle montrant la zone cartographiée.

        Example / Exemple:
            chart.map_points('lon', 'lat').inset_map(zoom_out=8, loc='lower right')
        """
        from matplotlib.patches import Rectangle
        geo = self._require_geo('inset_map')
        (x0, x1), (y0, y1) = self.ax.get_xlim(), self.ax.get_ylim()
        cx, cy, w, h = (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0
        pad = 0.02
        boxes = {'upper left': [pad, 1 - size - pad, size, size],
                 'upper right': [1 - size - pad, 1 - size - pad, size, size],
                 'lower left': [pad, pad, size, size],
                 'lower right': [1 - size - pad, pad, size, size]}
        if loc not in boxes:
            raise ValueError(f"loc doit valoir {list(boxes)}")
        ins = self.ax.inset_axes(boxes[loc])
        ex0, ex1 = cx - w * zoom_out / 2, cx + w * zoom_out / 2
        ey0, ey1 = cy - h * zoom_out / 2, cy + h * zoom_out / 2
        if geo['units'] == 'deg':
            ex0, ex1, ey0, ey1 = max(ex0, -180), min(ex1, 180), max(ey0, -85), min(ey1, 85)
            ins.set_aspect(1 / np.cos(np.radians(cy)))
        else:
            ins.set_aspect('equal')
        ins.set_xlim(ex0, ex1)
        ins.set_ylim(ey0, ey1)
        ins.set_xticks([])
        ins.set_yticks([])
        ins.set_facecolor(plt.rcParams['axes.facecolor'])
        for side in ins.spines.values():
            side.set_visible(True)
            side.set_color(plt.rcParams['text.color'])
            side.set_linewidth(1)
        ins.add_patch(Rectangle((x0, y0), w, h, fill=False, linewidth=1.6, zorder=5,
                                edgecolor=color or self._palette()[1 % len(self._palette())]))
        if basemap:
            self._basemap(crs=geo['crs'], source=basemap_source or geo.get('source'),
                          ax=ins, attribution=False)
        self.inset_ax = ins
        return self

    # ──────────────────────────────────────── Ecology / Écologie
    # Calculs dans plutovizu.ecology (GLM, lissage, ordination, diversité) ; ici le tracé.

    @staticmethod
    def _plain_log_axis(axis) -> None:
        """Axe logarithmique en nombres simples (0.5, 1, 2, 5, 10…) au lieu de 5×10⁻¹."""
        axis.set_major_locator(matplotlib.ticker.LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
        axis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:g}'))
        axis.set_minor_formatter(matplotlib.ticker.NullFormatter())

    def glm_curve(self, x: str, y: str, family: str = 'binomial', degree: int = 1,
                  ci: bool = True, jitter: float = 0.03, color: Optional[str] = None,
                  show_stats: bool = True) -> 'Chart':
        """
        EN: GLM response curve with its 95 % confidence band: 'binomial' (presence /
            absence → sigmoid probability) or 'poisson' (counts). degree=2 fits a bell-shaped
            (unimodal) response and marks its optimum. Results in self.glm.
        FR: Courbe de réponse d'un GLM avec sa bande de confiance à 95 % : 'binomial'
            (présence / absence → probabilité en sigmoïde) ou 'poisson' (comptages).
            degree=2 ajuste une réponse en cloche (unimodale) et marque son optimum.
            Résultats dans self.glm.

        Example / Exemple:
            chart.glm_curve('nitrates', 'presence')                    # régression logistique
            chart.glm_curve('ph', 'presence', degree=2)                # niche : optimum de pH
            chart.glm_curve('conductivite', 'abondance', family='poisson')
        """
        from .ecology import _inv_link, glm_fit
        self.chart_type = 'glm_curve'
        d = self.data[[x, y]].dropna()
        xv, yv = d[x].to_numpy(float), d[y].to_numpy(float)
        m, s = xv.mean(), xv.std() or 1.0
        design = lambda v: np.vander((np.asarray(v, float) - m) / s, degree + 1, increasing=True)  # noqa: E731
        fit = glm_fit(design(xv), yv, family)
        c = color or self._palette()[0]
        jit = np.random.default_rng(0).uniform(-jitter, jitter, len(yv)) if family == 'binomial' else 0
        self.ax.scatter(xv, yv + jit, s=18, color=plt.rcParams['text.color'], alpha=0.35,
                        linewidth=0, label='Observations')
        xs = np.linspace(xv.min(), xv.max(), 300)
        G = design(xs)
        eta = G @ fit['coefs']
        self.ax.plot(xs, _inv_link(eta, family), color=c, linewidth=2.4, label='Modèle')
        if ci:
            se = np.sqrt(np.einsum('ij,jk,ik->i', G, fit['cov'], G))
            self.ax.fill_between(xs, _inv_link(eta - 1.96 * se, family),
                                 _inv_link(eta + 1.96 * se, family), color=c, alpha=0.2,
                                 linewidth=0, label='IC 95 %')
        fit['optimum'] = None
        if degree == 2 and fit['coefs'][2] < 0:
            opt = m + s * (-fit['coefs'][1] / (2 * fit['coefs'][2]))
            if xv.min() <= opt <= xv.max():
                fit['optimum'] = float(opt)
                self.ax.axvline(opt, color=c, linestyle='--', linewidth=1)
                self.ax.annotate(f'Optimum ≈ {opt:.3g}', (opt, 0.86),
                                 xycoords=('data', 'axes fraction'), xytext=(5, 0),
                                 textcoords='offset points', fontsize=8.5, color=c,
                                 path_effects=self._halo())
        if degree == 1:   # coefficients sur l'échelle d'origine de x
            fit['coefs_original'] = np.array([fit['coefs'][0] - fit['coefs'][1] * m / s,
                                              fit['coefs'][1] / s])
        if show_stats:
            self.ax.text(0.02, 0.97, f"D² = {fit['d2']:.2f}   p = {fit['p_value']:.2g}\n"
                         f"AIC = {fit['aic']:.1f}   n = {fit['n']}", transform=self.ax.transAxes,
                         va='top', fontsize=8.5, bbox=dict(boxstyle='round,pad=0.35',
                         fc=plt.rcParams['axes.facecolor'], ec=plt.rcParams['grid.color'], alpha=0.9))
        self.ax.set_xlabel(x)
        self.ax.set_ylabel(f'Probabilité de {y}' if family == 'binomial' else y)
        self.ax.legend(loc='center right' if family == 'binomial' else 'upper left', fontsize=8)
        self.glm = fit
        return self

    def response_curve(self, x: str, y: str, frac: float = 0.5, ci: bool = True,
                       n_boot: int = 200, color: Optional[str] = None,
                       show_optimum: bool = True, random_state: int = 0) -> 'Chart':
        """
        EN: Smoothed response of a species (or any variable) along a gradient — local
            regression (LOWESS, GAM-like) with a bootstrap 95 % band; the ecological
            optimum (maximum of the curve) is marked. Results in self.response.
        FR: Réponse lissée d'une espèce (ou de toute variable) le long d'un gradient —
            régression locale (LOWESS, dans l'esprit d'un GAM) avec bande à 95 % par
            bootstrap ; l'optimum écologique (maximum de la courbe) est marqué.
            Résultats dans self.response.

        Args / Paramètres:
            frac (float): EN/FR: Part des points utilisée pour chaque estimation locale
                (plus grand = plus lisse). Défaut 0.5.

        Example / Exemple:
            chart.response_curve('altitude', 'abondance')
        """
        from .ecology import lowess
        self.chart_type = 'response_curve'
        d = self.data[[x, y]].dropna()
        xv, yv = d[x].to_numpy(float), d[y].to_numpy(float)
        xs = np.linspace(xv.min(), xv.max(), 200)
        smooth = lowess(xv, yv, xs, frac)
        c = color or self._palette()[0]
        self.ax.scatter(xv, yv, s=18, color=plt.rcParams['text.color'], alpha=0.35, linewidth=0)
        if ci:
            rng = np.random.default_rng(random_state)
            boots = np.array([lowess(xv[i], yv[i], xs, frac)
                              for i in (rng.integers(0, len(xv), len(xv)) for _ in range(n_boot))])
            lo, hi = np.nanpercentile(boots, [2.5, 97.5], axis=0)
            self.ax.fill_between(xs, lo, hi, color=c, alpha=0.2, linewidth=0, label='IC 95 % (bootstrap)')
        self.ax.plot(xs, smooth, color=c, linewidth=2.4, label='Réponse lissée')
        opt = float(xs[np.argmax(smooth)])
        if show_optimum:
            self.ax.axvline(opt, color=c, linestyle='--', linewidth=1)
            self.ax.annotate(f'Optimum ≈ {opt:.3g}', (opt, smooth.max()), xytext=(6, 6),
                             textcoords='offset points', fontsize=8, color=c)
        self.ax.set_xlabel(x)
        self.ax.set_ylabel(y)
        self.ax.legend(fontsize=8)
        self.response = {'x': xs, 'smooth': smooth, 'optimum': opt, 'max': float(smooth.max())}
        return self

    def count_distribution(self, column: str, max_count: Optional[int] = None,
                           color: Optional[str] = None) -> 'Chart':
        """
        EN: Observed count frequencies against those expected under a Poisson law and a
            negative binomial law (method of moments): reveals excess zeros and
            overdispersion (variance > mean). Statistics in self.count_stats.
        FR: Fréquences des comptages observés face à celles attendues sous une loi de
            Poisson et une loi binomiale négative (méthode des moments) : révèle l'excès de
            zéros et la surdispersion (variance > moyenne). Statistiques dans self.count_stats.

        Example / Exemple:
            chart.count_distribution('nb_individus')
        """
        from scipy import stats
        self.chart_type = 'count_distribution'
        v = self.data[column].dropna().to_numpy()
        if (v < 0).any() or not np.allclose(v, np.round(v)):
            raise ValueError("count_distribution attend des comptages entiers positifs ou nuls")
        v = v.astype(int)
        mean, var = v.mean(), v.var(ddof=1)
        top = max_count if max_count is not None else int(np.percentile(v, 99)) + 1
        k = np.arange(top + 1)
        observed = np.array([(v == i).sum() for i in k]) / len(v)
        poisson = stats.poisson.pmf(k, mean)
        palette = self._palette()
        self.ax.bar(k, observed, color=color or palette[0], alpha=0.75, label='Observé')
        self.ax.plot(k, poisson, 'o-', color=palette[1 % len(palette)], markersize=4,
                     label=f'Poisson (moyenne {mean:.2g})')
        negbin = None
        if var > mean:
            r_, p_ = mean ** 2 / (var - mean), mean / var
            negbin = stats.nbinom.pmf(k, r_, p_)
            self.ax.plot(k, negbin, 's--', color=palette[2 % len(palette)], markersize=4,
                         label='Binomiale négative')
        self.ax.set_xlabel(column)
        self.ax.set_ylabel('Fréquence')
        self.ax.legend(fontsize=8)
        self.count_stats = {'moyenne': float(mean), 'variance': float(var),
                            'dispersion': float(var / mean) if mean > 0 else np.nan,
                            'zeros_observes': float(observed[0]),
                            'zeros_poisson': float(poisson[0]),
                            'zeros_binneg': float(negbin[0]) if negbin is not None else None}
        st = self.count_stats
        self.ax.text(0.98, 0.97, f"Variance / moyenne = {st['dispersion']:.2f}\n"
                     f"Zéros : {st['zeros_observes']:.0%} observés, {st['zeros_poisson']:.0%} attendus (Poisson)",
                     transform=self.ax.transAxes, ha='right', va='top', fontsize=8.5,
                     bbox=dict(boxstyle='round,pad=0.35', fc=plt.rcParams['axes.facecolor'],
                               ec=plt.rcParams['grid.color'], alpha=0.9))
        return self

    def forest_plot(self, label: str, estimate: str, lower: Optional[str] = None,
                    upper: Optional[str] = None, se: Optional[str] = None,
                    group: Optional[str] = None, ref: float = 0.0, log: bool = False,
                    sort: bool = False) -> 'Chart':
        """
        EN: Forest plot of estimates with their intervals (model coefficients, random
            effects "caterpillar", odds ratios, Bayesian credible intervals…). Give lower and
            upper, or se (95 % interval = estimate ± 1.96 se). Intervals that exclude `ref`
            are highlighted. log=True for ratios (ref=1).
        FR: Forest plot d'estimations avec leurs intervalles (coefficients d'un modèle,
            effets aléatoires en « chenille », odds ratios, intervalles crédibles bayésiens…).
            Donner lower et upper, ou se (intervalle à 95 % = estimation ± 1,96 se). Les
            intervalles qui excluent `ref` sont mis en évidence. log=True pour des rapports (ref=1).

        Example / Exemple:
            chart.forest_plot('variable', 'odds_ratio', 'ic_bas', 'ic_haut', ref=1, log=True)
        """
        self.chart_type = 'forest_plot'
        d = self.data.copy()
        if lower is None or upper is None:
            if se is None:
                raise ValueError("Donner lower et upper, ou se")
            d['_lo'], d['_hi'] = d[estimate] - 1.96 * d[se], d[estimate] + 1.96 * d[se]
            lower, upper = '_lo', '_hi'
        if sort:
            d = d.sort_values(estimate)
        d = d.iloc[::-1]                           # premier élément en haut
        pos = np.arange(len(d))
        significant = (d[lower] > ref) | (d[upper] < ref)
        muted = mcolors.to_rgba(plt.rcParams['axes.labelcolor'], 0.55)
        palette = self._palette()
        if group:
            levels = list(pd.unique(self.data[group]))
            colors = [palette[levels.index(g) % len(palette)] for g in d[group]]
        else:
            colors = [palette[0] if s else muted for s in significant]
        self.ax.hlines(pos, d[lower], d[upper], colors=colors, linewidth=2)
        self.ax.scatter(d[estimate], pos, color=colors, s=45, zorder=3, edgecolor=plt.rcParams['axes.facecolor'])
        self.ax.axvline(ref, color=plt.rcParams['axes.labelcolor'], linestyle='--', linewidth=1)
        if log:
            self.ax.set_xscale('log')
            self._plain_log_axis(self.ax.xaxis)
        self.ax.set_yticks(pos, d[label].astype(str).tolist())
        self.ax.set_xlabel(estimate)
        self.ax.grid(False, axis='y')
        if group:
            self.ax.legend([plt.Line2D([], [], color=palette[i % len(palette)], marker='o', linewidth=2)
                            for i in range(len(levels))], levels, title=group, fontsize=8)
        self.forest_table = d.iloc[::-1].assign(significatif=significant.iloc[::-1].to_numpy())
        return self

    def ordination(self, species: List[str], env: Optional[List[str]] = None, method: str = 'ca',
                   labels: Optional[str] = None, group: Optional[str] = None,
                   show_species: bool = True, axes: Tuple[int, int] = (1, 2),
                   scale: bool = False) -> 'Chart':
        """
        EN: Community ordination triplot: sites (points), species (+) and environmental
            variables (arrows). 'ca' (correspondence analysis), 'cca' (canonical
            correspondence, unimodal responses), 'pca', 'rda' (redundancy, linear
            responses). Axis labels give the % of total inertia. Results in self.ord.
        FR: Triplot d'ordination de communautés : sites (points), espèces (+) et variables
            environnementales (flèches). 'ca' (AFC), 'cca' (analyse canonique des
            correspondances, réponses unimodales), 'pca', 'rda' (analyse de redondance,
            réponses linéaires). Les axes indiquent le % d'inertie totale. Résultats dans self.ord.

        Args / Paramètres:
            species (List[str]): EN/FR: Colonnes d'abondance (une par espèce, une ligne par site).
            env (List[str], optional): EN/FR: Variables environnementales (obligatoires pour cca, rda).
            labels / group (str, optional): EN/FR: Nom des sites / groupe (couleurs).
            scale (bool): EN/FR: pca/rda : centre-réduit les espèces. Défaut False.

        Example / Exemple:
            chart.ordination(especes, env=['humidite', 'ph'], method='cca', group='zone')
        """
        from .ecology import ordinate
        self.chart_type = 'ordination'
        res = ordinate(self.data[species].to_numpy(float),
                       self.data[env].to_numpy(float) if env else None, method, scale)
        a, b = axes[0] - 1, axes[1] - 1
        if res['sites'].shape[1] <= b:
            raise ValueError("Pas assez d'axes : ajouter des variables (env) ou des espèces")
        S = res['sites'][:, [a, b]]
        palette = self._palette()
        edge = plt.rcParams['axes.facecolor']
        if group:
            for (name, idx), c in zip(self.data.groupby(group, sort=False).groups.items(), palette * 10):
                pos = self.data.index.get_indexer(idx)
                self.ax.scatter(S[pos, 0], S[pos, 1], s=40, color=c, edgecolor=edge, label=name, zorder=3)
            self.ax.legend(title=group, fontsize=8)
        else:
            self.ax.scatter(S[:, 0], S[:, 1], s=40, color=palette[0], edgecolor=edge, zorder=3)
        if labels:
            for (xx, yy), t in zip(S, self.data[labels]):
                self.ax.annotate(str(t), (xx, yy), xytext=(3, 3), textcoords='offset points', fontsize=7,
                                 color=plt.rcParams['axes.labelcolor'])
        reach = np.abs(S).max() or 1
        if show_species:
            sp = res['species'][:, [a, b]]
            sp = sp * (0.9 * reach / (np.abs(sp).max() or 1))
            col = palette[1 % len(palette)]
            self.ax.scatter(sp[:, 0], sp[:, 1], marker='+', s=60, color=col, zorder=4)
            for (xx, yy), t in zip(sp, species):
                self.ax.annotate(t, (xx, yy), xytext=(3, -3), textcoords='offset points',
                                 fontsize=7.5, fontstyle='italic', color=col, va='top', zorder=4)
        if res['env'] is not None:
            ev = res['env'][:, [a, b]] * 0.95 * reach
            ink = plt.rcParams['text.color']
            for (xx, yy), t in zip(ev, env):
                self.ax.annotate('', xy=(xx, yy), xytext=(0, 0),
                                 arrowprops=dict(arrowstyle='->', color=ink, lw=1.4), zorder=5)
                self.ax.annotate(t, (xx * 1.08, yy * 1.08), ha='center', va='center', fontsize=8.5,
                                 fontweight='bold', color=ink, path_effects=self._halo(), zorder=5)
        for line in (self.ax.axhline, self.ax.axvline):
            line(0, color=plt.rcParams['grid.color'], linewidth=1, zorder=0)
        name = method.upper()
        ex = res['explained']
        self.ax.set_xlabel(f'{name}{axes[0]} ({ex[a]:.1%})')
        self.ax.set_ylabel(f'{name}{axes[1]} ({ex[b]:.1%})')
        if method in ('cca', 'rda'):
            self.ax.text(0.02, 0.02, f"Inertie contrainte : {res['constrained']:.1%}",
                         transform=self.ax.transAxes, fontsize=8.5, va='bottom',
                         color=plt.rcParams['text.color'])
        self.ax.margins(0.12)
        self.ord = res
        return self

    def rank_abundance(self, species: List[str], labels: Optional[str] = None,
                       pooled: bool = False, log: bool = True, marker: str = 'o') -> 'Chart':
        """
        EN: Rank-abundance (Whittaker) curve: relative abundance of each species against its
            rank, one curve per site (or all sites pooled). A steep curve = a few dominant
            species; a flat curve = an even community.
        FR: Courbe rang-abondance (diagramme de Whittaker) : abondance relative de chaque
            espèce selon son rang, une courbe par site (ou sites regroupés). Courbe raide =
            quelques espèces dominantes ; courbe plate = communauté équilibrée.

        Example / Exemple:
            chart.rank_abundance(especes, labels='site')
        """
        self.chart_type = 'rank_abundance'
        A = self.data[species]
        rows = [('Tous les sites', A.sum())] if pooled else \
            [((self.data[labels].iloc[i] if labels else A.index[i]), A.iloc[i]) for i in range(len(A))]
        for (name, ab), c in zip(rows, self._palette() * 10):
            ab = ab[ab > 0].sort_values(ascending=False)
            rel = 100 * ab / ab.sum()
            self.ax.plot(np.arange(1, len(rel) + 1), rel.to_numpy(), marker=marker, markersize=4,
                         color=c, label=str(name))
        if log:
            self.ax.set_yscale('log')
            self._plain_log_axis(self.ax.yaxis)
        self.ax.set_xlabel("Rang de l'espèce")
        self.ax.set_ylabel('Abondance relative (%)')
        if len(rows) <= 12:
            self.ax.legend(fontsize=8)
        return self

    def accumulation(self, species: List[str], kind: str = 'sites', labels: Optional[str] = None,
                     n_perm: int = 100, color: Optional[str] = None) -> 'Chart':
        """
        EN: Species accumulation: kind='sites' — richness as sites are added (random order,
            mean ± sd over permutations) with the Chao2 estimate of total richness;
            kind='individuals' — rarefaction curve of each site (expected richness for n
            individuals). A curve reaching a plateau means sampling is sufficient.
        FR: Accumulation des espèces : kind='sites' — richesse quand on ajoute des sites (ordre
            aléatoire, moyenne ± écart-type sur des permutations) avec l'estimateur Chao2 de la
            richesse totale ; kind='individuals' — courbe de raréfaction de chaque site
            (richesse attendue pour n individus). Un plateau = échantillonnage suffisant.

        Example / Exemple:
            chart.accumulation(especes)                              # effort d'échantillonnage
            chart.accumulation(especes, kind='individuals', labels='site')
        """
        from .ecology import accumulation as accumulate, rarefaction
        self.chart_type = 'accumulation'
        A = self.data[species]
        palette = self._palette()
        if kind == 'sites':
            mean, sd, info = accumulate(A.to_numpy(), n_perm=n_perm)
            n = np.arange(1, len(mean) + 1)
            c = color or palette[0]
            self.ax.fill_between(n, mean - sd, mean + sd, color=c, alpha=0.2, linewidth=0, label='± écart-type')
            self.ax.plot(n, mean, color=c, marker='o', markersize=4, label='Richesse cumulée')
            self.ax.axhline(info['chao2'], color=plt.rcParams['axes.labelcolor'], linestyle='--', linewidth=1,
                            label=f"Chao2 ≈ {info['chao2']:.1f} espèces")
            self.ax.set_xlabel('Nombre de sites')
            self.accumulation_stats = info
        elif kind == 'individuals':
            for i, c in zip(range(len(A)), palette * 10):
                counts = A.iloc[i].to_numpy(float)
                N = int(counts.sum())
                sizes = np.unique(np.linspace(1, N, min(N, 60)).astype(int))
                self.ax.plot(sizes, rarefaction(counts, sizes), color=c,
                             label=str(self.data[labels].iloc[i] if labels else A.index[i]))
            self.ax.set_xlabel("Nombre d'individus")
        else:
            raise ValueError("kind doit valoir 'sites' ou 'individuals'")
        self.ax.set_ylabel("Nombre d'espèces")
        if kind == 'sites' or len(A) <= 12:
            self.ax.legend(fontsize=8)
        return self

    def diversity(self, species: List[str], index: str = 'shannon', labels: Optional[str] = None,
                  group: Optional[str] = None, letters: bool = False,
                  color: Optional[str] = None) -> 'Chart':
        """
        EN: Diversity index per site — 'richesse', 'shannon', 'simpson', 'inv_simpson' or
            'pielou' — as bars; with group, mean ± SE per group (Tukey letters optional).
            All indices for all sites in self.diversity_table.
        FR: Indice de diversité par site — 'richesse', 'shannon', 'simpson', 'inv_simpson' ou
            'pielou' — en barres ; avec group, moyenne ± erreur type par groupe (lettres de
            Tukey en option). Tous les indices de tous les sites dans self.diversity_table.

        Example / Exemple:
            chart.diversity(especes, index='shannon', group='zone', letters=True)
        """
        from .ecology import diversity_indices
        self.chart_type = 'diversity'
        table = diversity_indices(self.data[species])
        if index not in table.columns:
            raise ValueError(f"index doit valoir {list(table.columns)}")
        if labels:
            table.insert(0, labels, self.data[labels].to_numpy())
        if group:
            table[group] = self.data[group].to_numpy()
            original, self.data = self.data, table
            try:
                self.mean_bar(group, index, letters=letters)
            finally:
                self.data = original
        else:
            names = table[labels].astype(str) if labels else table.index.astype(str)
            self.ax.bar(names, table[index], color=color or self._palette()[0], alpha=0.85)
            self.ax.set_xlabel(labels or 'site')
            self.ax.tick_params(axis='x', rotation=45)
        nice = {'richesse': 'Richesse spécifique', 'shannon': "Indice de Shannon H'",
                'simpson': 'Indice de Simpson (1 − D)', 'inv_simpson': 'Inverse de Simpson (1/D)',
                'pielou': 'Équitabilité de Pielou J'}
        self.ax.set_ylabel(nice[index])
        self.diversity_table = table
        return self

    # ──────────────────────────────────────── Qualitative analysis / Analyse qualitative

    def word_cloud(self, text: Optional[str] = None, words: Optional[str] = None,
                   weights: Optional[str] = None, max_words: int = 150,
                   language: Optional[str] = 'fr', stopwords: Optional[List[str]] = None,
                   min_length: int = 3, ngram: int = 1, shape: str = 'rectangle',
                   colormap: Optional[str] = None, random_state: int = 42) -> 'Chart':
        """
        EN: Word cloud (wordcloud package). Either a text column (open-ended answers,
            interviews…) whose words are counted, or a words column with its weights.
            Colors and font follow the active theme. Frequencies in self.word_frequencies.
        FR: Nuage de mots (paquet wordcloud). Soit une colonne de texte (réponses
            ouvertes, entretiens…) dont on compte les mots, soit une colonne de mots et
            leurs poids. Couleurs et police suivent le thème. Fréquences dans self.word_frequencies.

        Args / Paramètres:
            text (str, optional): EN/FR: Colonne de texte libre.
            words, weights (str, optional): EN/FR: Colonnes mot / poids (fréquences déjà calculées).
            max_words (int): EN/FR: Nombre maximal de mots affichés. Défaut 150.
            language (str): EN/FR: Mots vides ignorés : 'fr' (défaut), 'en', 'fr+en' ou None.
            stopwords (List[str], optional): EN/FR: Mots supplémentaires à ignorer.
            min_length (int): EN/FR: Longueur minimale d'un mot. Défaut 3.
            ngram (int): EN/FR: 2 = expressions de deux mots, etc.
            shape (str): 'rectangle' (défaut) ou 'circle'.
            colormap (str, optional): EN/FR: Palette continue ; défaut : couleurs du thème.

        Example / Exemple:
            chart.word_cloud('reponse', stopwords=['agriculteur'])
            chart.word_cloud(words='mot', weights='frequence', shape='circle')
        """
        from wordcloud import WordCloud
        from .text import word_counts
        self.chart_type = 'word_cloud'
        if words and weights:
            freqs = self.data.groupby(words)[weights].sum().astype(float).to_dict()
        elif text:
            freqs = dict(word_counts(self.data[text], language=language, stopwords=stopwords,
                                     min_length=min_length, ngram=ngram))
        else:
            raise ValueError("Indiquer text= (texte libre) ou words= et weights= (fréquences)")
        if not freqs:
            raise ValueError("Aucun mot à afficher après filtrage des mots vides")
        w_in, h_in = self.fig.get_size_inches() * self.ax.get_position().size
        width, height = int(w_in * 100), int(h_in * 100)
        mask = None
        if shape == 'circle':
            side = min(width, height)
            yy, xx = np.ogrid[:side, :side]
            outside = (xx - side / 2) ** 2 + (yy - side / 2) ** 2 > (side / 2 - 2) ** 2
            mask = (outside * 255).astype(np.uint8)
            width = height = side
        elif shape != 'rectangle':
            raise ValueError("shape doit valoir 'rectangle' ou 'circle'")
        palette = [mcolors.to_hex(c) for c in self._palette()]  # PIL attend des couleurs hex
        rng = np.random.default_rng(random_state)
        color_func = None if colormap else (lambda *a, **k: palette[rng.integers(len(palette))])
        font = fm.findfont(fm.FontProperties(family=plt.rcParams['font.family']))
        cloud = WordCloud(width=width, height=height, max_words=max_words, mask=mask,
                          background_color=mcolors.to_hex(plt.rcParams['axes.facecolor']),
                          mode='RGBA',
                          colormap=colormap, color_func=color_func, font_path=font,
                          prefer_horizontal=0.9, random_state=random_state, scale=2,
                          relative_scaling=0.5).generate_from_frequencies(freqs)
        self.ax.imshow(cloud, interpolation='bilinear')
        self.ax.axis('off')
        self.word_frequencies = (pd.Series(freqs, name='frequence')
                                 .sort_values(ascending=False).head(max_words))
        return self

    def word_frequency(self, text: str, top: int = 20, ngram: int = 1,
                       language: Optional[str] = 'fr', stopwords: Optional[List[str]] = None,
                       min_length: int = 3, color: Optional[str] = None,
                       show_values: bool = True) -> 'Chart':
        """
        EN: Horizontal bars of the most frequent words (or n-word expressions) in a text
            column — the quantitative companion of the word cloud.
        FR: Barres horizontales des mots (ou expressions de n mots) les plus fréquents
            d'une colonne de texte — le complément chiffré du nuage de mots.

        Example / Exemple:
            chart.word_frequency('reponse', top=15, ngram=2)
        """
        from .text import word_counts
        self.chart_type = 'word_frequency'
        counts = word_counts(self.data[text], language=language, stopwords=stopwords,
                             min_length=min_length, ngram=ngram).most_common(top)
        if not counts:
            raise ValueError("Aucun mot à afficher après filtrage des mots vides")
        words, values = zip(*counts[::-1])  # le plus fréquent en haut
        bars = self.ax.barh(words, values, color=color or self._palette()[0])
        if show_values:
            self.ax.bar_label(bars, padding=3, fontsize=9, color=plt.rcParams['text.color'])
        self.ax.set_xlabel('Occurrences')
        self.ax.grid(False, axis='y')
        self.word_frequencies = pd.Series(dict(counts), name='frequence')
        return self

    def likert(self, columns: List[str], order: List, labels: Optional[Dict[str, str]] = None,
               cmap: str = 'RdYlGn', sort: bool = True, show_values: bool = True,
               min_label: float = 5) -> 'Chart':
        """
        EN: Diverging stacked bars for Likert-scale survey items: one bar per question,
            centered on the neutral answer; disagreement left, agreement right (in %).
            Percentages in self.likert_table.
        FR: Barres empilées divergentes pour des questions en échelle de Likert : une
            barre par question, centrée sur la réponse neutre ; désaccord à gauche,
            accord à droite (en %). Pourcentages dans self.likert_table.

        Args / Paramètres:
            columns (List[str]): EN/FR: Une colonne par question.
            order (List): EN/FR: Modalités du plus négatif au plus positif.
            labels (dict, optional): EN/FR: Libellés des questions {colonne: texte}.
            cmap (str): EN/FR: Palette divergente (rouge → vert par défaut).
            sort (bool): EN/FR: Trie les questions par accord net. Défaut True.
            min_label (float): EN/FR: Pas d'étiquette sous ce pourcentage. Défaut 5.

        Example / Exemple:
            echelle = ['Pas du tout d’accord', 'Pas d’accord', 'Neutre', 'D’accord', 'Tout à fait d’accord']
            chart.likert(['q1', 'q2', 'q3'], order=echelle)
        """
        self.chart_type = 'likert'
        table = pd.DataFrame({c: self.data[c].value_counts(normalize=True).reindex(order)
                              for c in columns}).T.fillna(0) * 100
        unknown = set(self.data[columns].stack().dropna()) - set(order)
        if unknown:
            raise ValueError(f"Modalités absentes de order : {sorted(map(str, unknown))}")
        k = len(order)
        neg = list(order[:k // 2])
        neutral = [order[k // 2]] if k % 2 else []
        if sort:
            pos = list(order[(k + 1) // 2:])
            table = table.loc[(table[pos].sum(axis=1) - table[neg].sum(axis=1)).sort_values().index]
        colors = plt.get_cmap(cmap)(np.linspace(0.08, 0.92, k))
        left = -(table[neg].sum(axis=1) + (table[neutral[0]] / 2 if neutral else 0)).to_numpy()
        names = [(labels or {}).get(q, q) for q in table.index]
        for level, color in zip(order, colors):
            widths = table[level].to_numpy()
            bars = self.ax.barh(names, widths, left=left, color=color, label=str(level),
                                edgecolor=plt.rcParams['axes.facecolor'], linewidth=1)
            if show_values:
                lum = 0.2126 * color[0] + 0.7152 * color[1] + 0.0722 * color[2]
                self.ax.bar_label(bars, labels=[f'{w:.0f}' if w >= min_label else '' for w in widths],
                                  label_type='center', fontsize=8,
                                  color='#111111' if lum > 0.55 else 'white')
            left = left + widths
        self.ax.axvline(0, color=plt.rcParams['axes.labelcolor'], linewidth=1)
        lim = max(abs(self.ax.get_xlim()[0]), abs(self.ax.get_xlim()[1]))
        self.ax.set_xlim(-lim, lim)
        self.ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{abs(v):.0f} %'))
        self.ax.grid(False, axis='y')
        self.ax.legend(ncol=min(k, 5), loc='upper center', bbox_to_anchor=(0.5, -0.12),
                       frameon=False, fontsize=8)
        self.likert_table = table
        return self

    def code_matrix(self, case: str, code: str, normalize: bool = False,
                    annot: bool = True, cmap: Optional[str] = None) -> 'Chart':
        """
        EN: Code matrix for qualitative coding (cases × codes): how often each code
            (theme) appears in each case (interview, document, group). Input: one row
            per coded segment. normalize=True gives % per case. Table in self.code_table.
        FR: Matrice des codes pour le codage qualitatif (cas × codes) : nombre de fois où
            chaque code (thème) apparaît dans chaque cas (entretien, document, groupe).
            Entrée : une ligne par segment codé. normalize=True donne des % par cas.
            Tableau dans self.code_table.

        Example / Exemple:
            chart.code_matrix('entretien', 'theme')
        """
        self.chart_type = 'code_matrix'
        table = pd.crosstab(self.data[case], self.data[code])
        table = table[table.sum().sort_values(ascending=False).index]  # codes les plus cités d'abord
        if normalize:
            table = table.div(table.sum(axis=1), axis=0) * 100
        sns.heatmap(table, ax=self.ax, annot=annot, fmt='.0f', cmap=cmap or plt.rcParams['image.cmap'],
                    linewidths=1, linecolor=plt.rcParams['axes.facecolor'],
                    cbar_kws={'label': '% du cas' if normalize else 'Occurrences'})
        self.ax.set_xlabel(code)
        self.ax.set_ylabel(case)
        self.ax.tick_params(axis='x', rotation=30)
        for t in self.ax.get_xticklabels():
            t.set_ha('right')
        self.code_table = table
        return self

    # ──────────────────────────────────────── Seaborn

    # Fonctions seaborn qui créent leur propre figure (une grille de panneaux)
    _SNS_FIGURE_LEVEL = {'relplot', 'displot', 'catplot', 'lmplot',
                         'jointplot', 'pairplot', 'clustermap'}

    def seaborn(self, name: str, /, **kwargs) -> 'Chart':
        """
        EN: Draw any seaborn chart with this Chart's data, theme and chaining.
            Axes-level functions draw on the chart; figure-level ones (pairplot,
            jointplot, relplot, displot, catplot, lmplot, clustermap) replace the
            figure — the seaborn grid is then in self.sns_grid.
        FR: Trace n'importe quel graphique seaborn avec les données, le thème et le
            chaînage de ce Chart. Les fonctions « axes » dessinent sur le graphique ;
            les fonctions « figure » (pairplot, jointplot, relplot, displot, catplot,
            lmplot, clustermap) remplacent la figure — la grille seaborn est dans self.sns_grid.

        Args / Paramètres:
            name (str): EN/FR: Nom de la fonction seaborn ('violinplot', 'kdeplot'…).
            **kwargs: EN/FR: Arguments de la fonction seaborn (x, y, hue, palette…).

        Example / Exemple:
            Chart(df).seaborn('violinplot', x='variete', y='rendement', inner='quart')
            Chart(df).seaborn('relplot', x='dose', y='rendement', col='site', kind='line')
        """
        func = getattr(sns, name, None)
        if name.startswith('_') or not callable(func):
            raise ValueError(f"'{name}' n'est pas une fonction de tracé seaborn")
        self.chart_type = f'seaborn.{name}'
        kwargs.setdefault('data', self.data)
        if name in self._SNS_FIGURE_LEVEL:
            if name != 'clustermap':
                kwargs.pop('figsize', None)  # ces fonctions utilisent height/aspect
            plt.close(self.fig)
            self.sns_grid = func(**kwargs)
            self.fig = self.sns_grid.figure
            self.ax = (getattr(self.sns_grid, 'ax_heatmap', None) or getattr(self.sns_grid, 'ax_joint', None)
                       or self.sns_grid.axes.flat[0])
        else:
            func(ax=self.ax, **kwargs)
        return self

    def kde(self, x: str, hue: Optional[str] = None, fill: bool = True, **kwargs) -> 'Chart':
        """EN: Kernel density estimate (seaborn.kdeplot). / FR: Densité lissée (kdeplot)."""
        return self.seaborn('kdeplot', x=x, hue=hue, fill=fill, **kwargs)

    def ecdf(self, x: str, hue: Optional[str] = None, **kwargs) -> 'Chart':
        """EN: Empirical cumulative distribution (seaborn.ecdfplot).
        FR: Fonction de répartition empirique (ecdfplot)."""
        return self.seaborn('ecdfplot', x=x, hue=hue, **kwargs)

    def strip(self, x: str, y: str, hue: Optional[str] = None, **kwargs) -> 'Chart':
        """EN: Jittered points per category (seaborn.stripplot).
        FR: Points dispersés par catégorie (stripplot)."""
        return self.seaborn('stripplot', x=x, y=y, hue=hue, **kwargs)

    def swarm(self, x: str, y: str, hue: Optional[str] = None, **kwargs) -> 'Chart':
        """EN: Non-overlapping points per category (seaborn.swarmplot).
        FR: Points sans chevauchement par catégorie (swarmplot)."""
        return self.seaborn('swarmplot', x=x, y=y, hue=hue, **kwargs)

    def boxen(self, x: str, y: str, hue: Optional[str] = None, **kwargs) -> 'Chart':
        """EN: Letter-value plot for large samples (seaborn.boxenplot).
        FR: Boîte à moustaches étendue, pour grands échantillons (boxenplot)."""
        return self.seaborn('boxenplot', x=x, y=y, hue=hue, **kwargs)

    def pointplot(self, x: str, y: str, hue: Optional[str] = None, **kwargs) -> 'Chart':
        """EN: Means with confidence intervals joined by lines (seaborn.pointplot).
        FR: Moyennes et intervalles de confiance reliés (pointplot)."""
        return self.seaborn('pointplot', x=x, y=y, hue=hue, **kwargs)

    def countplot(self, x: str, hue: Optional[str] = None, **kwargs) -> 'Chart':
        """EN: Count of observations per category (seaborn.countplot).
        FR: Effectifs par catégorie (countplot)."""
        return self.seaborn('countplot', x=x, hue=hue, **kwargs)

    def residplot(self, x: str, y: str, order: int = 1, **kwargs) -> 'Chart':
        """EN: Residuals of a polynomial regression (seaborn.residplot).
        FR: Résidus d'une régression polynomiale (residplot)."""
        return self.seaborn('residplot', x=x, y=y, order=order, **kwargs)

    def pairplot(self, vars: Optional[List[str]] = None, hue: Optional[str] = None,
                 corner: bool = False, **kwargs) -> 'Chart':
        """EN: Matrix of pairwise scatter plots and distributions (seaborn.pairplot).
        FR: Matrice de nuages de points deux à deux et distributions (pairplot)."""
        return self.seaborn('pairplot', vars=vars, hue=hue, corner=corner, **kwargs)

    def jointplot(self, x: str, y: str, hue: Optional[str] = None, kind: str = 'scatter',
                  **kwargs) -> 'Chart':
        """EN: Bivariate plot with marginal distributions (seaborn.jointplot).
        FR: Graphique bivarié avec distributions marginales (jointplot)."""
        return self.seaborn('jointplot', x=x, y=y, hue=hue, kind=kind, **kwargs)

    # ──────────────────────────────────────── Utilities / Utilitaires

    def add_annotation(self, x: float, y: float, text: str,
                       fontsize: int = 10, color: Optional[str] = None,
                       arrow: bool = False, arrowprops: Optional[dict] = None) -> 'Chart':
        """
        EN: Add a text annotation (optionally with an arrow) at coordinates (x, y).
        FR: Ajoute une annotation textuelle (optionnellement avec une flèche) aux coordonnées (x, y).

        Args / Paramètres:
            x (float):
                EN: x-coordinate in data space.
                FR: Coordonnée x dans l'espace des données.
            y (float):
                EN: y-coordinate in data space.
                FR: Coordonnée y dans l'espace des données.
            text (str):
                EN: Annotation text.
                FR: Texte de l'annotation.
            fontsize (int):
                EN: Font size. Default 10.
                FR: Taille de la police. Défaut 10.
            color (str):
                EN: Text color. Default 'black'.
                FR: Couleur du texte. Défaut 'black'.
            arrow (bool):
                EN: Draw an arrow pointing to (x, y). Default False.
                FR: Dessiner une flèche pointant vers (x, y). Défaut False.
            arrowprops (dict, optional):
                EN: Matplotlib arrowprops dict (overrides default arrow style).
                FR: Dictionnaire arrowprops matplotlib (remplace le style de flèche par défaut).

        Returns / Retourne:
            Chart: self

        Example / Exemple:
            chart.add_annotation(2020, 500, 'Pic COVID', arrow=True, color='red')
        """
        color = color or plt.rcParams['text.color']
        if arrow and arrowprops is None:
            arrowprops = dict(arrowstyle='->', color=color)
        self.ax.annotate(text, xy=(x, y), fontsize=fontsize, color=color,
                         arrowprops=arrowprops if arrow else None)
        return self

    def annotation(self, text: str, xy: Tuple[float, float], xytext: Tuple[float, float],
                   radius: float = 0.3, color: Optional[str] = None,
                   arrow_color: Optional[str] = None, fontsize: Optional[float] = None,
                   fontweight: Optional[str] = None, width: float = 1.2,
                   head_width: float = 3.5, head_length: float = 6.0, coords: str = 'data',
                   **text_kwargs) -> 'Chart':
        """
        EN: Annotation with a curved arrow (drawarrow): the text is written at `xytext` and
            an arrow goes from the text to the annotated point `xy`. The text is aligned
            automatically so the arrow starts at its edge.
        FR: Annotation avec une flèche courbe (drawarrow) : le texte est écrit en `xytext` et
            une flèche part du texte vers le point annoté `xy`. Le texte est aligné
            automatiquement pour que la flèche parte de son bord.

        Args / Paramètres:
            text (str): EN/FR: Texte de l'annotation (retours à la ligne avec \\n).
            xy (tuple): EN/FR: Point annoté (pointe de la flèche).
            xytext (tuple): EN/FR: Position du texte (départ de la flèche).
            radius (float): EN/FR: Courbure de la flèche (0 = droite, négatif = courbe dans l'autre sens).
            color (str, optional): EN/FR: Couleur du texte ; défaut : couleur du texte du thème.
            arrow_color (str, optional): EN/FR: Couleur de la flèche ; défaut : celle du texte.
            fontsize (float, optional): EN/FR: Taille du texte ; défaut : thème.
            fontweight (str, optional): EN/FR: Graisse du texte ('bold', 'semibold'…).
            width (float): EN/FR: Épaisseur du trait de la flèche.
            head_width (float): EN/FR: Largeur de la pointe.
            head_length (float): EN/FR: Longueur de la pointe.
            coords (str): EN/FR: Repère des positions : 'data' (données) ou 'axes' (0–1 dans le graphique).
            **text_kwargs: EN/FR: Options du texte matplotlib (fontproperties, ha, va, bbox…).

        Example / Exemple:
            chart.annotation('Record\\nde 2024', xy=(2024, 5.1), xytext=(2019, 5.6), radius=-0.3)
        """
        from drawarrow import ax_arrow
        if coords not in ('data', 'axes'):
            raise ValueError("coords doit valoir 'data' ou 'axes'")
        tr = self.ax.transData if coords == 'data' else self.ax.transAxes
        if coords == 'data':  # dates, catégories… → nombres matplotlib
            xy, xytext = [(self.ax.convert_xunits(px), self.ax.convert_yunits(py))
                          for px, py in (xy, xytext)]
        self.ax.get_xlim()  # limites à jour avant de passer en pixels
        (x0, y0), (x1, y1) = tr.transform([xytext, xy])
        dx, dy = x1 - x0, y1 - y0
        if abs(dx) >= abs(dy):     # flèche plutôt horizontale : le texte finit (ou commence) au départ
            ha, va = ('right' if dx > 0 else 'left'), 'center'
        else:                      # plutôt verticale : texte au-dessus ou en dessous du départ
            ha, va = 'center', ('top' if dy > 0 else 'bottom')
        color = color or plt.rcParams['text.color']
        self.ax.text(*xytext, text, transform=tr, color=color, fontsize=fontsize,
                     fontweight=fontweight, ha=text_kwargs.pop('ha', ha),
                     va=text_kwargs.pop('va', va), **text_kwargs)
        ax_arrow(xytext, xy, ax=self.ax, radius=radius, color=arrow_color or color,
                 width=width, head_width=head_width, head_length=head_length,
                 transform=tr, shrinkA=4, shrinkB=2)
        return self

    def arrow(self, tail: Tuple[float, float], head: Tuple[float, float], radius: float = 0.3,
              color: Optional[str] = None, width: float = 1.5, head_width: float = 4.0,
              head_length: float = 7.0, double_headed: bool = False,
              inflection: Optional[Tuple[float, float]] = None, coords: str = 'data',
              **kwargs) -> 'Chart':
        """
        EN: Draw a (curved) arrow with drawarrow from `tail` to `head`, in data, axes (0–1)
            or figure (0–1) coordinates.
        FR: Trace une flèche (courbe) avec drawarrow de `tail` à `head`, en coordonnées des
            données, du graphique (0–1) ou de la figure (0–1).

        Args / Paramètres:
            tail (tuple): EN/FR: Point de départ de la flèche.
            head (tuple): EN/FR: Point d'arrivée (pointe).
            radius (float): EN/FR: Courbure (0 = droite ; le signe change le sens de la courbe).
            color (str, optional): EN/FR: Couleur ; défaut : couleur du texte du thème.
            width (float): EN/FR: Épaisseur du trait.
            head_width (float): EN/FR: Largeur de la pointe.
            head_length (float): EN/FR: Longueur de la pointe.
            double_headed (bool): EN/FR: Pointe aux deux extrémités.
            inflection (tuple, optional): EN/FR: Point de passage intermédiaire (flèche en S).
            coords (str): EN/FR: 'data', 'axes' (0–1 dans le graphique) ou 'figure' (0–1 dans la figure).
            **kwargs: EN/FR: Options de drawarrow / FancyArrowPatch (alpha, linestyle, zorder…).

        Example / Exemple:
            chart.arrow((0.1, 0.9), (0.3, 0.6), coords='axes', radius=-0.4)
        """
        from drawarrow import ax_arrow, fig_arrow
        if coords not in ('data', 'axes', 'figure'):
            raise ValueError("coords doit valoir 'data', 'axes' ou 'figure'")
        opts = dict(radius=radius, color=color or plt.rcParams['text.color'], width=width,
                    head_width=head_width, head_length=head_length,
                    double_headed=double_headed, inflection_position=inflection, **kwargs)
        if coords == 'figure':
            fig_arrow(tail, head, fig=self.fig, **opts)
        else:
            if coords == 'data':  # dates, catégories… → nombres matplotlib
                tail, head = [(self.ax.convert_xunits(px), self.ax.convert_yunits(py))
                              for px, py in (tail, head)]
            ax_arrow(tail, head, ax=self.ax, transform=self.ax.transData if coords == 'data'
                     else self.ax.transAxes, **opts)
        return self

    def rounded(self, radius: float = 8.0) -> 'Chart':
        """
        EN: Smooth (rounded) corners on the chart's shapes: bars, histogram and treemap
            rectangles, pie, donut and radial-bar sectors, polygons. `radius` is in points
            (clamped so thin shapes stay intact). Applied when the chart is shown or saved,
            once the final layout is known, so it can be called anywhere in the chain.
        FR: Coins adoucis (arrondis) sur les formes du graphique : barres, rectangles
            d'histogramme et de treemap, secteurs de camembert, d'anneau et de barres
            radiales, polygones, et (plus légèrement) pastilles de légende. `radius` est en
            points (réduit pour les formes fines).
            Appliqué à l'affichage ou à l'enregistrement, une fois la mise en page connue :
            on peut donc l'appeler n'importe où dans la chaîne.
            Par défaut, tous les graphiques ont des coins arrondis de 8 points
            (Chart.default_round) ; rounded(0) garde des coins vifs.

        Args / Paramètres:
            radius (float): EN/FR: Rayon des coins en points (1 point = 1/72 de pouce) ; 0 = coins vifs.

        Example / Exemple:
            Chart(df).bar('region', 'ventes').rounded(8).show()
            Chart(df).donut('surface', labels='culture').rounded(5).show()
            Chart(df).bar('region', 'ventes').rounded(0).show()     # coins vifs
        """
        if radius < 0:
            raise ValueError('radius doit être positif')
        self._round_radius = radius
        self._round_explicit = True
        return self

    def _apply_rounding(self) -> None:
        """Remplace les formes anguleuses par leur version aux coins arrondis (une seule fois)."""
        from matplotlib.patches import FancyArrowPatch, PathPatch, Polygon, Rectangle, Wedge
        radius = getattr(self, '_round_radius', None)
        if not radius:
            return
        if not getattr(self, '_round_explicit', False) and self.chart_type in self._NO_DEFAULT_ROUND:
            return
        self._round_radius = None
        r_px = radius * self.fig.dpi / 72
        for ax in self.fig.axes:
            ax.get_xlim()  # limites à jour
            for patch in list(ax.patches):
                if isinstance(patch, FancyArrowPatch) or not isinstance(
                        patch, (Rectangle, Wedge, Polygon, PathPatch)):
                    continue
                if not patch.get_fill() or patch.get_facecolor()[3] == 0:
                    continue  # tracé sans remplissage (courbe de bump chart…) : on n'y touche pas
                # repère des données (et non celui de la forme, non inversible si largeur ou hauteur = 0)
                tr = patch.get_data_transform()
                data_path = patch.get_patch_transform().transform_path(patch.get_path())
                new_path = _round_path(tr.transform_path(data_path), r_px)
                if new_path is None:
                    continue
                try:
                    new = PathPatch(tr.inverted().transform_path(new_path))
                except np.linalg.LinAlgError:
                    continue
                new.update_from(patch)
                new.set_transform(tr)
                new.set_zorder(patch.get_zorder())
                new.set_label(patch.get_label())
                new.set_clip_on(patch.get_clip_on())
                clip = patch.get_clip_path()
                patch.remove()
                ax.add_patch(new)
                new.set_clip_path(clip)
        self._round_legends(radius)

    def _round_legends(self, radius: float) -> None:
        """Arrondit légèrement les pastilles rectangulaires des légendes (unités : points)."""
        from matplotlib.offsetbox import DrawingArea
        from matplotlib.patches import FancyBboxPatch, Rectangle
        legends = [ax.get_legend() for ax in self.fig.axes] + list(self.fig.legends)
        for leg in filter(None, legends):
            for da in leg.findobj(DrawingArea):
                for k, h in enumerate(da._children):
                    if type(h) is not Rectangle:
                        continue
                    w, hh = h.get_width(), h.get_height()
                    rs = min(radius / 3, 0.35 * min(w, hh))  # ponytail: « légèrement » = 1/3 du rayon
                    new = FancyBboxPatch(h.get_xy(), w, hh,
                                         boxstyle=f'round,pad=0,rounding_size={rs}')
                    new.update_from(h)
                    new.set_transform(h.get_data_transform())  # sans la transformation propre au rectangle
                    new.set_figure(self.fig)
                    da._children[k] = new
                    if h in leg.legend_handles:
                        leg.legend_handles[leg.legend_handles.index(h)] = new

    def set_style(self, style: str = 'default') -> 'Chart':
        """
        EN: Apply a matplotlib style sheet to the figure.
            Must be called before drawing chart elements to take full effect.
        FR: Applique une feuille de style matplotlib à la figure.
            Doit être appelée avant de tracer les éléments pour un effet complet.

        Args / Paramètres:
            style (str):
                EN: Style name (e.g. 'ggplot', 'seaborn-v0_8', 'dark_background').
                    Use plt.style.available to list all installed styles.
                FR: Nom du style (ex. 'ggplot', 'seaborn-v0_8', 'dark_background').
                    Utiliser plt.style.available pour lister tous les styles installés.

        Returns / Retourne:
            Chart: self

        Example / Exemple:
            chart.set_style('ggplot')
        """
        plt.style.use(style)
        # La figure a été créée avant le style : on la reconstruit (vide) pour qu'il s'applique
        size = self.fig.get_size_inches()
        plt.close(self.fig)
        self.fig, self.ax = plt.subplots(figsize=size)
        return self

    def tight_layout(self) -> 'Chart':
        """
        EN: Automatically adjust subplot parameters to prevent labels from being clipped.
            Equivalent to calling plt.tight_layout() on this figure.
        FR: Ajuste automatiquement les paramètres de sous-graphique pour éviter
            que les étiquettes soient coupées. Équivalent à plt.tight_layout() sur cette figure.

        Returns / Retourne:
            Chart: self

        Example / Exemple:
            chart.line('date', 'ventes').labels('Date', 'Ventes').tight_layout().show()
        """
        if self.sns_grid is not None:
            # clustermap gère sa propre mise en page ; les autres grilles seaborn ont la leur
            if hasattr(self.sns_grid, 'tight_layout') and not hasattr(self.sns_grid, 'ax_heatmap'):
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    self.sns_grid.tight_layout()
            return self
        self.fig.tight_layout()
        return self

    def save(self, filename: str, dpi: int = 400, bbox_inches: str = 'tight',
             format: Optional[str] = None) -> 'Chart':
        """
        EN: Save the figure to a file. Returns self for continued chaining.
        FR: Enregistre la figure dans un fichier. Retourne self pour continuer le chaînage.

        Args / Paramètres:
            filename (str):
                EN: Output file path (extension inferred unless format is given).
                FR: Chemin du fichier de sortie (extension déduite sauf si format est précisé).
            dpi (int):
                EN: Resolution in dots per inch. Default 400.
                FR: Résolution en points par pouce. Défaut 400.
            bbox_inches (str):
                EN: Bounding box for saving ('tight' removes surrounding whitespace).
                FR: Boîte englobante pour l'enregistrement ('tight' supprime l'espace vide).
            format (str):
                EN: Output format ('png', 'pdf', 'svg', 'eps'…). Default: from the file
                    extension (png without extension).
                FR: Format de sortie ('png', 'pdf', 'svg', 'eps'…). Défaut : déduit de
                    l'extension du fichier (png sans extension).

        Returns / Retourne:
            Chart: self

        Example / Exemple:
            chart.line('date', 'ventes').tight_layout().save('rapport.pdf', format='pdf')
        """
        self._apply_rounding()
        self.fig.savefig(filename, dpi=dpi, bbox_inches=bbox_inches, format=format)
        return self

    def show(self) -> None:
        """
        EN: Display this chart's figure (notebook or interactive window), then close it
            to free memory. Call save() before show() if you need both.
        FR: Affiche la figure de ce graphique (notebook ou fenêtre interactive), puis la
            ferme pour libérer la mémoire. Appeler save() avant show() si besoin des deux.

        Returns / Retourne:
            None

        Example / Exemple:
            chart.line('date', 'ventes').tight_layout().show()
        """
        self._apply_rounding()
        if 'inline' in matplotlib.get_backend():
            from IPython.display import display  # backend inline ⇒ IPython présent
            display(self.fig)
        else:
            # ponytail: en script (fenêtre GUI), plt.show() affiche aussi les autres
            # figures encore ouvertes ; les fermer (show/plt.close) si ça gêne.
            plt.show()
        plt.close(self.fig)

    def get_figure(self) -> Tuple[plt.Figure, plt.Axes]:
        """
        EN: Return the underlying matplotlib Figure and primary Axes objects.
            Use these for advanced customisation beyond the Chart API
            (e.g. adding a second subplot, custom patches, seaborn layer…).
        FR: Retourne les objets matplotlib Figure et Axes principaux sous-jacents.
            Utilisez-les pour une personnalisation avancée au-delà de l'API Chart
            (ex. ajouter un second sous-graphique, des patches personnalisés, une couche seaborn…).

        Returns / Retourne:
            Tuple[plt.Figure, plt.Axes]:
                EN: (fig, ax) — the matplotlib Figure and its primary Axes.
                FR: (fig, ax) — la Figure matplotlib et son Axes principal.

        Example / Exemple:
            fig, ax = chart.line('date', 'ventes').get_figure()
            ax.axvspan('2020-01-01', '2021-01-01', alpha=0.1, color='red')
        """
        return self.fig, self.ax
