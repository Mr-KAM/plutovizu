"""plutovizu — API chaînable autour de matplotlib pour les DataFrames pandas."""
from .chart import Chart, get_fonts
from .soil import USDA_POLYGONS, texture_class
from .themes import (CONTEXTS, DESCRIPTIONS, MORETHEMES, SEABORN, THEMES, get_theme,
                     list_themes, set_theme, theme_context)

__all__ = ["Chart", "get_fonts", "texture_class", "USDA_POLYGONS",
           "CONTEXTS", "DESCRIPTIONS", "MORETHEMES", "SEABORN", "THEMES",
           "get_theme", "list_themes", "set_theme", "theme_context"]
__version__ = "0.1.0"
