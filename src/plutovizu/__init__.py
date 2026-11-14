"""plutovizu — API chaînable autour de matplotlib pour les DataFrames pandas."""
from .chart import Chart, get_fonts
from .themes import THEMES, get_theme, list_themes, set_theme, theme_context

__all__ = ["Chart", "get_fonts", "THEMES", "get_theme", "list_themes",
           "set_theme", "theme_context"]
__version__ = "0.1.0"
