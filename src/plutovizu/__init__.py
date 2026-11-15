"""plutovizu — API chaînable autour de matplotlib pour les DataFrames pandas.

Le thème 'editorial' est appliqué dès l'import ; ``set_theme(None)`` revient au style
matplotlib, ``set_theme('nom')`` en choisit un autre.
"""
from .chart import Chart, get_fonts
from .ecology import diversity_indices, glm_fit, ordinate
from .graph import critical_path, max_flow, shortest_path, spanning_tree, tsp
from .soil import USDA_POLYGONS, texture_class
from .text import FRENCH_STOPWORDS, word_counts
from .themes import (CONTEXTS, DESCRIPTIONS, MORETHEMES, SEABORN, THEMES, get_theme,
                     list_themes, set_theme, theme_context)

DEFAULT_THEME = 'editorial'
set_theme(DEFAULT_THEME)

__all__ = ["Chart", "get_fonts", "texture_class", "USDA_POLYGONS",
           "diversity_indices", "glm_fit", "ordinate",
           "critical_path", "max_flow", "shortest_path", "spanning_tree", "tsp",
           "word_counts", "FRENCH_STOPWORDS", "DEFAULT_THEME",
           "CONTEXTS", "DESCRIPTIONS", "MORETHEMES", "SEABORN", "THEMES",
           "get_theme", "list_themes", "set_theme", "theme_context"]
__version__ = "0.1.0"
