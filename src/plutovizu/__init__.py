"""plutovizu — API chaînable autour de matplotlib pour les DataFrames pandas.

Le thème 'editorial' est appliqué dès l'import ; ``set_theme(None)`` revient au style
matplotlib, ``set_theme('nom')`` en choisit un autre.
"""
from .chart import Chart, get_fonts
from .soil import USDA_POLYGONS, texture_class
from .text import FRENCH_STOPWORDS, word_counts
from .themes import (CONTEXTS, DESCRIPTIONS, MORETHEMES, SEABORN, THEMES, get_theme,
                     list_themes, set_theme, theme_context)

DEFAULT_THEME = 'editorial'
set_theme(DEFAULT_THEME)

__all__ = ["Chart", "get_fonts", "texture_class", "USDA_POLYGONS",
           "word_counts", "FRENCH_STOPWORDS", "DEFAULT_THEME",
           "CONTEXTS", "DESCRIPTIONS", "MORETHEMES", "SEABORN", "THEMES",
           "get_theme", "list_themes", "set_theme", "theme_context"]
__version__ = "0.1.0"
