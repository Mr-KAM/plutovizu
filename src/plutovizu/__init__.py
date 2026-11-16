"""plutovizu — API chaînable autour de matplotlib pour les DataFrames pandas.

Le thème 'editorial' et la police Fira Sans (fournie, comme Inter) sont appliqués dès
l'import ; la taille par défaut d'un graphique est 16 × 10 cm. ``set_theme(None)`` revient au style
matplotlib, ``set_theme('nom')`` en choisit un autre.
"""
from .chart import FIGSIZE, Chart, cm, get_fonts
from .fonts import BUNDLED_FONTS, DEFAULT_FONT, load_font, register_fonts
from .ecology import diversity_indices, glm_fit, ordinate
from .graph import critical_path, max_flow, shortest_path, spanning_tree, tsp
from .soil import USDA_POLYGONS, texture_class
from .text import FRENCH_STOPWORDS, word_counts
from .themes import (CONTEXTS, DESCRIPTIONS, MORETHEMES, SEABORN, THEMES, get_theme,
                     list_themes, set_theme, theme_context)

DEFAULT_THEME = 'editorial'
register_fonts()  # Fira Sans et Inter, fournies avec le package
set_theme(DEFAULT_THEME)

__all__ = ["Chart", "FIGSIZE", "cm", "get_fonts", "load_font", "register_fonts",
           "BUNDLED_FONTS", "DEFAULT_FONT", "texture_class", "USDA_POLYGONS",
           "diversity_indices", "glm_fit", "ordinate",
           "critical_path", "max_flow", "shortest_path", "spanning_tree", "tsp",
           "word_counts", "FRENCH_STOPWORDS", "DEFAULT_THEME",
           "CONTEXTS", "DESCRIPTIONS", "MORETHEMES", "SEABORN", "THEMES",
           "get_theme", "list_themes", "set_theme", "theme_context"]
__version__ = "0.1.0"
