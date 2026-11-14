import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.colors as mcolors
import pandas as pd
import numpy as np
from typing import Optional, Union, List, Tuple, Dict, Any
import seaborn as sns


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

    def __init__(self, data: pd.DataFrame, figsize: Tuple[int, int] = (10, 6)):
        """
        EN: Initialise a Chart from a pandas DataFrame.
        FR: Initialise un Chart à partir d'un DataFrame pandas.

        Args / Paramètres:
            data (pd.DataFrame):
                EN: The dataset to visualise.
                FR: Le jeu de données à visualiser.
            figsize (Tuple[int, int]):
                EN: Figure width and height in inches. Default (10, 6).
                FR: Largeur et hauteur de la figure en pouces. Défaut (10, 6).

        Returns / Retourne:
            Chart: EN: The Chart instance (for chaining). / FR: L'instance Chart (pour le chaînage).

        Example / Exemple:
            chart = Chart(df)
            chart = Chart(df, figsize=(14, 8))
        """
        self.data = data
        self.fig, self.ax = plt.subplots(figsize=figsize)
        self.chart_type = None
        self.custom_palettes = {}
        self.current_palette = None

    # ──────────────────────────────────────── Appearance / Apparence

    def title(self, title: str, fontsize: int = 14, fontweight: str = 'bold',
              color: str = 'black', pad: float = 20) -> 'Chart':
        """
        EN: Set the chart title.
        FR: Définit le titre du graphique.

        Args / Paramètres:
            title (str):
                EN: Title text.
                FR: Texte du titre.
            fontsize (int):
                EN: Font size. Default 14.
                FR: Taille de la police. Défaut 14.
            fontweight (str):
                EN: Font weight ('normal', 'bold', 'light'…). Default 'bold'.
                FR: Graisse de la police. Défaut 'bold'.
            color (str):
                EN: Title color. Default 'black'.
                FR: Couleur du titre. Défaut 'black'.
            pad (float):
                EN: Padding above the axes in points. Default 20.
                FR: Espace au-dessus des axes en points. Défaut 20.

        Returns / Retourne:
            Chart: self

        Example / Exemple:
            chart.title('Ventes annuelles', fontsize=18, color='steelblue')
        """
        self.ax.set_title(title, fontsize=fontsize, fontweight=fontweight,
                          color=color, pad=pad)
        return self

    def labels(self, xlabel: str = None, ylabel: str = None,
               fontsize: int = 12, color: str = 'black') -> 'Chart':
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
            fontsize (int):
                EN: Font size for both labels. Default 12.
                FR: Taille de la police pour les deux étiquettes. Défaut 12.
            color (str):
                EN: Text color. Default 'black'.
                FR: Couleur du texte. Défaut 'black'.

        Returns / Retourne:
            Chart: self

        Example / Exemple:
            chart.labels(xlabel='Date', ylabel='Ventes (€)', fontsize=13)
        """
        if xlabel:
            self.ax.set_xlabel(xlabel, fontsize=fontsize, color=color)
        if ylabel:
            self.ax.set_ylabel(ylabel, fontsize=fontsize, color=color)
        return self

    def grid(self, show: bool = True, alpha: float = 0.3,
             linestyle: str = '-', linewidth: float = 0.5) -> 'Chart':
        """
        EN: Toggle and style the axes grid.
        FR: Active/désactive et stylise la grille des axes.

        Args / Paramètres:
            show (bool):
                EN: Display the grid. Default True.
                FR: Afficher la grille. Défaut True.
            alpha (float):
                EN: Grid opacity, 0–1. Default 0.3.
                FR: Opacité de la grille, 0–1. Défaut 0.3.
            linestyle (str):
                EN: Line style ('-', '--', ':', '-.'). Default '-'.
                FR: Style des traits ('-', '--', ':', '-.'). Défaut '-'.
            linewidth (float):
                EN: Line width in points. Default 0.5.
                FR: Épaisseur du trait en points. Défaut 0.5.

        Returns / Retourne:
            Chart: self

        Example / Exemple:
            chart.grid(alpha=0.4, linestyle='--')
        """
        self.ax.grid(show, alpha=alpha, linestyle=linestyle, linewidth=linewidth)
        return self

    def set_legend(self, show: bool = True, location: str = 'best',
                   fontsize: int = 10, frameon: bool = True) -> 'Chart':
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
            fontsize (int):
                EN: Legend text size. Default 10.
                FR: Taille du texte de la légende. Défaut 10.
            frameon (bool):
                EN: Draw a box around the legend. Default True.
                FR: Dessiner un cadre autour de la légende. Défaut True.

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
            print(f"Warning: Font '{family}' not found. Using 'sans-serif'.")
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
            Accepts a seaborn palette name, a matplotlib style name,
            a custom palette name (created with create_custom_palette),
            or a list of color strings.
        FR: Applique une palette de couleurs aux éléments du graphique.
            Accepte un nom de palette seaborn, un style matplotlib,
            un nom de palette personnalisée (créée avec create_custom_palette),
            ou une liste de couleurs.

        Args / Paramètres:
            colors (Union[str, List[str]], optional):
                EN: Palette name string or explicit list of colors.
                FR: Nom de la palette ou liste explicite de couleurs.

        Returns / Retourne:
            Chart: self

        Example / Exemple:
            chart.set_colors('deep')
            chart.set_colors(['#e63946', '#457b9d', '#1d3557'])
        """
        if colors:
            if isinstance(colors, str):
                if colors in self.custom_palettes:
                    self.current_palette = self.custom_palettes[colors]
                    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=self.current_palette)
                else:
                    try:
                        plt.style.use(colors)
                    except Exception:
                        try:
                            sns.set_palette(colors)
                            self.current_palette = sns.color_palette(colors)
                        except Exception:
                            print(f"Warning: Color palette '{colors}' not recognized")
            else:
                self.current_palette = colors
                plt.rcParams['axes.prop_cycle'] = plt.cycler(color=colors)
        return self

    def create_custom_palette(self, name: str, colors: List[str],
                              save_palette: bool = True) -> 'Chart':
        """
        EN: Define and optionally register a named custom color palette.
            Invalid color strings are silently skipped.
        FR: Définit et enregistre optionnellement une palette de couleurs nommée.
            Les chaînes de couleur invalides sont ignorées silencieusement.

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
                print(f"Warning: Invalid color '{color}' ignored")
        if not valid_colors:
            print("Error: No valid colors provided")
            return self
        if save_palette:
            self.custom_palettes[name] = valid_colors
        self.current_palette = valid_colors
        plt.rcParams['axes.prop_cycle'] = plt.cycler(color=valid_colors)
        print(f"Custom palette '{name}' created with {len(valid_colors)} colors")
        return self

    def get_available_palettes(self, include_custom: bool = True,
                               include_seaborn: bool = True,
                               include_matplotlib: bool = True) -> Dict[str, List[str]]:
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
        return palettes

    @staticmethod
    def print_available_palettes(category: str = 'all', limit: int = None) -> None:
        """
        EN: Print the catalogue of available palettes to the console.
        FR: Affiche le catalogue des palettes disponibles dans la console.

        Args / Paramètres:
            category (str):
                EN: Filter output: 'all', 'seaborn', 'matplotlib', 'custom',
                    or any specific sub-category key. Default 'all'.
                FR: Filtre la sortie : 'all', 'seaborn', 'matplotlib', 'custom',
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
        elif category in palettes:
            categories_to_show = [category]
        else:
            print(f"Unknown category: {category}")
            print("Available categories:", list(palettes.keys()))
            return
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
            Supports custom palettes, seaborn palettes, and matplotlib colormaps.
        FR: Affiche un aperçu horizontal de la palette sous forme de barre de couleurs.
            Supporte les palettes personnalisées, seaborn et les colormaps matplotlib.

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
        self.ax.clear()
        colors = []
        if palette_name in self.custom_palettes:
            colors = self.custom_palettes[palette_name][:n_colors]
        else:
            try:
                colors = sns.color_palette(palette_name, n_colors)
            except Exception:
                try:
                    cmap = plt.cm.get_cmap(palette_name)
                    colors = [cmap(i / (n_colors - 1)) for i in range(n_colors)]
                except Exception:
                    print(f"Palette '{palette_name}' not found")
                    return self
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
        try:
            start_rgb = mcolors.to_rgba(start_color)
            end_rgb = mcolors.to_rgba(end_color)
            colors = []
            for i in range(n_colors):
                ratio = i / (n_colors - 1)
                r = start_rgb[0] * (1 - ratio) + end_rgb[0] * ratio
                g = start_rgb[1] * (1 - ratio) + end_rgb[1] * ratio
                b = start_rgb[2] * (1 - ratio) + end_rgb[2] * ratio
                a = start_rgb[3] * (1 - ratio) + end_rgb[3] * ratio
                colors.append(mcolors.to_hex((r, g, b, a)))
            if save_palette:
                self.custom_palettes[name] = colors
            self.current_palette = colors
            plt.rcParams['axes.prop_cycle'] = plt.cycler(color=colors)
            print(f"Gradient palette '{name}' created: {start_color} → {end_color}")
            return self
        except ValueError as e:
            print(f"Error creating gradient: {e}")
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
        return self

    def bar(self, x: str, y: str, horizontal: bool = False,
            width: float = 0.8, alpha: float = 0.8,
            color: Optional[str] = None) -> 'Chart':
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

        Example / Exemple:
            chart.bar('categorie', 'ventes', color='steelblue')
            chart.bar('ventes', 'categorie', horizontal=True)
        """
        self.chart_type = 'bar'
        if horizontal:
            self.ax.barh(self.data[x], self.data[y], height=width, alpha=alpha, color=color)
        else:
            self.ax.bar(self.data[x], self.data[y], width=width, alpha=alpha, color=color)
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
                  density: bool = False, edgecolor: str = 'black') -> 'Chart':
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
                EN: Bar border color. Default 'black'.
                FR: Couleur des bordures de barres. Défaut 'black'.

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

    def heatmap(self, annot: bool = True, cmap: str = 'viridis',
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
        numeric_data = self.data.select_dtypes(include=[np.number])
        im = self.ax.imshow(numeric_data.corr(), cmap=cmap, aspect='auto')
        if annot:
            for i in range(len(numeric_data.columns)):
                for j in range(len(numeric_data.columns)):
                    self.ax.text(j, i, format(numeric_data.corr().iloc[i, j], fmt),
                                 ha='center', va='center', color='white')
        self.ax.set_xticks(range(len(numeric_data.columns)))
        self.ax.set_yticks(range(len(numeric_data.columns)))
        self.ax.set_xticklabels(numeric_data.columns)
        self.ax.set_yticklabels(numeric_data.columns)
        if cbar:
            self.fig.colorbar(im, ax=self.ax)
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
                     show_threshold_line: bool = True, temp_scale_factor: float = 2.0) -> 'Chart':
        """
        EN: Draw an ombrothermic (Walter-Lieth) diagram showing monthly temperature
            and precipitation on dual axes with the p = 2T scale convention.
            Dry periods (precipitation < 2 × temperature) are highlighted.
        FR: Trace un diagramme ombrothermique (Walter-Lieth) montrant la température
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

        Example / Exemple:
            chart.ombrothermic('Mois', 'Température (°C)', 'Précipitations (mm)')
        """
        self.chart_type = 'ombrothermic'
        self.ax.clear()
        self.ax.plot(self.data[month_col], self.data[temperature_col],
                     color=temp_color, marker='o', linewidth=2,
                     label='Température', markersize=6)
        self.ax.set_ylabel(
            f'Température ({temperature_col.split("(")[-1].replace(")", "")})',
            color=temp_color)
        self.ax.tick_params(axis='y', labelcolor=temp_color)
        temp_min = self.data[temperature_col].min() / temp_scale_factor
        temp_max = self.data[temperature_col].max() / temp_scale_factor
        self.ax.set_ylim(temp_min, temp_max)
        ax2 = self.ax.twinx()
        ax2.bar(self.data[month_col], self.data[precipitation_col],
                color=precip_color, alpha=0.6, label='Précipitations')
        ax2.set_ylabel(
            f'Précipitations ({precipitation_col.split("(")[-1].replace(")", "")})',
            color=precip_color)
        ax2.tick_params(axis='y', labelcolor=precip_color)
        precip_min = self.data[precipitation_col].min() - 2
        precip_max = self.data[precipitation_col].max() + 2
        ax2.set_ylim(precip_min, precip_max)
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
                  positive_color: str = 'steelblue',
                  negative_color: str = 'salmon') -> 'Chart':
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
        values = self.data[y].tolist()
        running = 0
        bottoms, heights, colors = [], [], []
        for v in values:
            bottoms.append(running if v >= 0 else running + v)
            heights.append(abs(v))
            colors.append(positive_color if v >= 0 else negative_color)
            running += v
        self.ax.bar(self.data[x].tolist(), heights, bottom=bottoms, color=colors)
        self.ax.axhline(0, color='black', linewidth=0.8)
        return self

    def bump(self, x: str, y: str, group: str,
             linewidth: float = 2.5, markersize: float = 8) -> 'Chart':
        """
        EN: Draw a bump chart to visualise ranking changes over time.
            The y-axis is inverted so rank 1 appears at the top.
            Each group (team, country, product…) gets one line.
        FR: Trace un graphique de classement (bump chart) pour visualiser l'évolution
            des rangs dans le temps. L'axe y est inversé pour que le rang 1 apparaisse en haut.
            Chaque groupe (équipe, pays, produit…) reçoit une ligne.

        Args / Paramètres:
            x (str):
                EN: Column for time periods or seasons.
                FR: Colonne pour les périodes ou saisons.
            y (str):
                EN: Column with rank values (integer, lower = better).
                FR: Colonne avec les valeurs de rang (entier, plus bas = meilleur).
            group (str):
                EN: Column identifying each ranked entity.
                FR: Colonne identifiant chaque entité classée.
            linewidth (float):
                EN: Line stroke width. Default 2.5.
                FR: Épaisseur du trait. Défaut 2.5.
            markersize (float):
                EN: Marker diameter at each data point. Default 8.
                FR: Diamètre du marqueur à chaque point de données. Défaut 8.

        Returns / Retourne:
            Chart: self

        Example / Exemple:
            chart.bump('saison', 'classement', 'equipe')
        """
        self.chart_type = 'bump'
        for g, gdf in self.data.groupby(group):
            gdf = gdf.sort_values(x)
            self.ax.plot(gdf[x], gdf[y], marker='o', linewidth=linewidth,
                         markersize=markersize, label=str(g))
        self.ax.invert_yaxis()
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
                EN: Vertical spacing between ridge baselines.
                    Higher values increase overlap. Default 1.5.
                FR: Espacement vertical entre les lignes de base des crêtes.
                    Des valeurs plus élevées augmentent le chevauchement. Défaut 1.5.
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
        groups = list(self.data[group_col].unique())
        x_range = np.linspace(self.data[value_col].min(),
                              self.data[value_col].max(), 300)
        for i, g in enumerate(groups):
            vals = self.data[self.data[group_col] == g][value_col].dropna().values
            if len(vals) < 2:
                continue
            density = gaussian_kde(vals)(x_range)
            offset = i * overlap
            self.ax.fill_between(x_range, offset, density + offset,
                                 alpha=alpha, label=str(g))
            self.ax.plot(x_range, density + offset, color='white', linewidth=0.8)
        self.ax.set_yticks(np.arange(len(groups)) * overlap)
        self.ax.set_yticklabels(groups)
        return self

    def treemap(self, values: str, labels: Optional[str] = None,
                alpha: float = 0.8) -> 'Chart':
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
        sizes = self.data[values].tolist()
        label_data = self.data[labels].tolist() if labels else None
        colors = plt.colormaps.get_cmap('tab20')(np.linspace(0, 1, len(sizes)))
        squarify.plot(sizes=sizes, label=label_data, color=colors, ax=self.ax, alpha=alpha)
        self.ax.axis('off')
        return self

    def hexbin(self, x: str, y: str, gridsize: int = 30,
               cmap: str = 'YlOrRd', mincnt: int = 1) -> 'Chart':
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
        hb = self.ax.hexbin(self.data[x], self.data[y],
                            gridsize=gridsize, cmap=cmap, mincnt=mincnt)
        self.fig.colorbar(hb, ax=self.ax, label='count')
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
                 color_start: str = 'steelblue', color_end: str = 'salmon',
                 line_color: str = '#aaaaaa', markersize: float = 80,
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

    # ──────────────────────────────────────── Utilities / Utilitaires

    def add_annotation(self, x: float, y: float, text: str,
                       fontsize: int = 10, color: str = 'black',
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
        if arrow and arrowprops is None:
            arrowprops = dict(arrowstyle='->', color=color)
        self.ax.annotate(text, xy=(x, y), fontsize=fontsize, color=color,
                         arrowprops=arrowprops if arrow else None)
        return self

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
        self.fig.tight_layout()
        return self

    def save(self, filename: str, dpi: int = 300, bbox_inches: str = 'tight',
             format: str = 'png') -> 'Chart':
        """
        EN: Save the figure to a file. Returns self for continued chaining.
        FR: Enregistre la figure dans un fichier. Retourne self pour continuer le chaînage.

        Args / Paramètres:
            filename (str):
                EN: Output file path (extension inferred unless format is given).
                FR: Chemin du fichier de sortie (extension déduite sauf si format est précisé).
            dpi (int):
                EN: Resolution in dots per inch. Default 300.
                FR: Résolution en points par pouce. Défaut 300.
            bbox_inches (str):
                EN: Bounding box for saving ('tight' removes surrounding whitespace).
                FR: Boîte englobante pour l'enregistrement ('tight' supprime l'espace vide).
            format (str):
                EN: Output format ('png', 'pdf', 'svg', 'eps'…). Default 'png'.
                FR: Format de sortie ('png', 'pdf', 'svg', 'eps'…). Défaut 'png'.

        Returns / Retourne:
            Chart: self

        Example / Exemple:
            chart.line('date', 'ventes').tight_layout().save('rapport.pdf', format='pdf')
        """
        self.fig.savefig(filename, dpi=dpi, bbox_inches=bbox_inches, format=format)
        return self

    def show(self) -> None:
        """
        EN: Display the figure in the active matplotlib backend (interactive window or notebook).
            Calls plt.show() — does not return self as no further chaining is expected.
        FR: Affiche la figure dans le backend matplotlib actif (fenêtre interactive ou notebook).
            Appelle plt.show() — ne retourne pas self car aucun chaînage ultérieur n'est attendu.

        Returns / Retourne:
            None

        Example / Exemple:
            chart.line('date', 'ventes').tight_layout().show()
        """
        plt.show()

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
