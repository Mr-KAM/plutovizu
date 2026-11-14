import matplotlib
matplotlib.use('Agg')  # pas de fenêtre GUI pendant les tests

import os
import tempfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pytest

from plutovizu import Chart, get_fonts, list_themes, set_theme, theme_context


@pytest.fixture(autouse=True)
def close_figures():
    with matplotlib.rc_context():  # aucun thème ne fuit d'un test à l'autre
        yield
    plt.close('all')


@pytest.fixture
def df():
    np.random.seed(0)
    return pd.DataFrame({
        'x': range(10),
        'a': np.random.randn(10),
        'b': np.random.randn(10),
        'cat': list('AABBCCDDEE'),
    })


@pytest.fixture
def climate():
    return pd.DataFrame({
        'mois': ['Jan', 'Fév', 'Mar', 'Avr', 'Mai', 'Jun',
                 'Jul', 'Aoû', 'Sep', 'Oct', 'Nov', 'Déc'],
        'temp': [24.1, 26.1, 26.6, 26.5, 26.1, 25.0,
                 23.9, 23.9, 24.3, 24.8, 24.9, 23.7],
        'precip': [16, 23.9, 86.7, 116.7, 92.9, 147.8,
                   143.2, 215.8, 167.1, 109.4, 30.3, 33],
    })


# ── get_fonts ────────────────────────────────────────────────────────────────

def test_get_fonts_returns_list():
    fonts = get_fonts()
    assert isinstance(fonts, list)
    assert len(fonts) > 0

def test_get_fonts_sorted():
    fonts = get_fonts()
    assert fonts == sorted(fonts)

def test_get_fonts_pattern():
    fonts = get_fonts(pattern='arial')
    assert all('arial' in f.lower() for f in fonts)


# ── Chart.__init__ ───────────────────────────────────────────────────────────

def test_chart_init_attributes(df):
    c = Chart(df)
    assert c.data is df
    assert c.chart_type is None
    assert c.custom_palettes == {}
    assert c.current_palette is None
    assert c.fig is not None
    assert c.ax is not None


# ── chaînabilité ─────────────────────────────────────────────────────────────

def test_line_returns_self(df):
    c = Chart(df)
    assert c.line('x', 'a') is c

def test_bar_returns_self(df):
    c = Chart(df)
    assert c.bar('cat', 'a') is c

def test_scatter_returns_self(df):
    c = Chart(df)
    assert c.scatter('x', 'a') is c

def test_histogram_returns_self(df):
    c = Chart(df)
    assert c.histogram('a') is c

def test_boxplot_returns_self(df):
    c = Chart(df)
    assert c.boxplot(['a', 'b']) is c

def test_area_returns_self(df):
    c = Chart(df)
    assert c.area('x', ['a', 'b']) is c

def test_violin_returns_self(df):
    c = Chart(df)
    assert c.violin(['a', 'b']) is c

def test_heatmap_returns_self(df):
    c = Chart(df)
    assert c.heatmap() is c

def test_title_returns_self(df):
    c = Chart(df)
    assert c.title('Mon titre') is c

def test_labels_returns_self(df):
    c = Chart(df)
    assert c.labels(xlabel='X', ylabel='Y') is c

def test_grid_returns_self(df):
    c = Chart(df)
    assert c.grid() is c

def test_tight_layout_returns_self(df):
    c = Chart(df)
    assert c.tight_layout() is c

def test_set_style_returns_self(df):
    c = Chart(df)
    assert c.set_style('default') is c

def test_set_colors_seaborn_returns_self(df):
    c = Chart(df)
    assert c.set_colors('deep') is c

def test_set_colors_list_returns_self(df):
    c = Chart(df)
    assert c.set_colors(['red', 'blue', 'green']) is c


# ── palettes personnalisées ──────────────────────────────────────────────────

def test_create_custom_palette_stores(df):
    c = Chart(df)
    c.create_custom_palette('test', ['#FF0000', '#00FF00', '#0000FF'])
    assert 'test' in c.custom_palettes
    assert c.custom_palettes['test'] == ['#FF0000', '#00FF00', '#0000FF']

def test_create_custom_palette_invalid_color_ignored(df):
    c = Chart(df)
    with pytest.warns(UserWarning, match='not_a_color_xyz'):
        c.create_custom_palette('p', ['red', 'not_a_color_xyz'])
    assert c.custom_palettes['p'] == ['red']

def test_create_custom_palette_no_valid_color_raises(df):
    with pytest.warns(UserWarning), pytest.raises(ValueError):
        Chart(df).create_custom_palette('p', ['not_a_color_xyz'])

def test_unknown_palette_raises(df):
    with pytest.raises(ValueError):
        Chart(df).set_colors('palette_inexistante')
    with pytest.raises(ValueError):
        Chart(df).preview_palette('palette_inexistante')
    with pytest.raises(ValueError):
        Chart.print_available_palettes(category='inexistante')

def test_gradient_invalid_color_raises(df):
    with pytest.raises(ValueError):
        Chart(df).generate_gradient_palette('g', 'pas_une_couleur', 'red')

def test_font_missing_warns(df):
    with pytest.warns(UserWarning, match='not found'):
        Chart(df).font('Police Qui N Existe Pas')

def test_generate_gradient_palette(df):
    c = Chart(df)
    c.generate_gradient_palette('grad', 'blue', 'red', n_colors=5)
    assert 'grad' in c.custom_palettes
    assert len(c.custom_palettes['grad']) == 5

def test_get_available_palettes_keys(df):
    palettes = Chart(df).get_available_palettes()
    assert 'seaborn_qualitative' in palettes
    assert 'matplotlib_sequential' in palettes
    assert 'custom' in palettes


# ── pie ──────────────────────────────────────────────────────────────────────

def test_pie_returns_self():
    d = pd.DataFrame({'val': [10, 20, 30], 'label': ['A', 'B', 'C']})
    c = Chart(d)
    assert c.pie('val', labels='label') is c


# ── ombrothermic ─────────────────────────────────────────────────────────────

def test_ombrothermic_returns_self(climate):
    c = Chart(climate)
    assert c.ombrothermic('mois', 'temp', 'precip') is c

def test_ombrothermic_creates_twin_axis(climate):
    c = Chart(climate)
    c.ombrothermic('mois', 'temp', 'precip')
    assert hasattr(c, 'ax2')

def test_print_ombrothermic_analysis_returns_self(climate):
    c = Chart(climate)
    assert c.print_ombrothermic_analysis('mois', 'temp', 'precip') is c


# ── save ─────────────────────────────────────────────────────────────────────

def test_save_writes_file_and_returns_self(df, tmp_path):
    path = str(tmp_path / 'out.png')
    c = Chart(df)
    c.line('x', 'a')
    result = c.save(path)
    assert result is c
    assert os.path.exists(path)
    assert os.path.getsize(path) > 0


# ── get_figure ───────────────────────────────────────────────────────────────

@pytest.mark.filterwarnings('ignore:FigureCanvasAgg is non-interactive')
def test_show_closes_only_its_figure(df):
    other = Chart(df)
    c = Chart(df).line('x', 'a')
    c.show()
    assert not plt.fignum_exists(c.fig.number)
    assert plt.fignum_exists(other.fig.number)


def test_get_figure(df):
    c = Chart(df)
    fig, ax = c.get_figure()
    assert fig is c.fig
    assert ax is c.ax


# ── thèmes ───────────────────────────────────────────────────────────────────

@pytest.mark.parametrize('name', list_themes())
def test_every_theme_renders(df, name):
    set_theme(name)
    c = (Chart(df).line('x', ['a', 'b']).title('t').labels('x', 'y')
         .grid().set_legend())
    c.fig.canvas.draw()  # valide police, cmap, etc. au rendu
    assert matplotlib.colors.same_color(c.fig.get_facecolor(),
                                        plt.rcParams['figure.facecolor'])

def test_theme_applies_to_title_and_background(df):
    c = Chart(df, theme='pluto').line('x', 'a').title('Titre')
    assert matplotlib.colors.to_hex(c.fig.get_facecolor()) == '#0b0d0e'
    assert matplotlib.colors.to_hex(c.ax.title.get_color()) == '#f4f4f5'

def test_switching_theme_resets_previous_settings():
    set_theme('editorial')             # grille horizontale seule
    set_theme('pluto')
    assert plt.rcParams['axes.grid.axis'] == 'both'
    set_theme(None)
    assert plt.rcParams['axes.facecolor'] == 'white'

def test_theme_context_is_scoped():
    before = plt.rcParams['axes.facecolor']
    with theme_context('neon'):
        assert plt.rcParams['axes.facecolor'] == '#0f0a1e'
    assert plt.rcParams['axes.facecolor'] == before

def test_unknown_theme_raises():
    with pytest.raises(ValueError, match='Disponibles'):
        set_theme('inexistant')
