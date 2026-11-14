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

@pytest.mark.parametrize('name', list_themes('plutovizu'))
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

def test_ombrothermic_gaussen_scale(climate):
    c = Chart(climate).ombrothermic('mois', 'temp', 'precip')
    t_lo, t_hi = c.ax.get_ylim()
    p_lo, p_hi = c.ax2.get_ylim()
    assert t_lo <= climate['temp'].min() and climate['temp'].max() <= t_hi  # courbe visible
    assert p_hi == pytest.approx(2 * t_hi) and p_lo == pytest.approx(2 * t_lo)  # P = 2T

def test_every_theme_has_description():
    from plutovizu import DESCRIPTIONS
    assert set(DESCRIPTIONS) == set(list_themes())


# ── morethemes (paquet optionnel) ───────────────────────────────────────────

from plutovizu import MORETHEMES
import plutovizu.themes as themes_mod

needs_mt = pytest.mark.skipif(not MORETHEMES, reason='morethemes non installé')


@needs_mt
@pytest.mark.parametrize('name', sorted(MORETHEMES))
def test_every_morethemes_theme_renders(df, name, monkeypatch):
    monkeypatch.setattr(themes_mod, '_load_font', lambda theme: None)  # pas de réseau
    with theme_context(name):
        c = Chart(df).line('x', ['a', 'b']).title('t').grid().set_legend()
        c.fig.canvas.draw()

@needs_mt
def test_morethemes_listed_and_resolvable():
    assert 'wsj' in list_themes('morethemes') and 'wsj' in list_themes()
    assert 'brand' not in list_themes()
    assert set(list_themes('plutovizu')).isdisjoint(list_themes('morethemes'))

@needs_mt
def test_morethemes_font_offline_falls_back(monkeypatch):
    import pyfonts
    def offline(*a, **k):
        raise ConnectionError('pas de réseau')
    monkeypatch.setattr(pyfonts, 'load_google_font', offline)
    with pytest.warns(UserWarning, match='indisponible'):
        set_theme('wsj')
    assert plt.rcParams['font.family'] == ['sans-serif']


# ── sciences & analyse ──────────────────────────────────────────────────────

@pytest.fixture
def essai():
    """Essai agronomique : 3 variétés × 4 doses d'azote × 3 répétitions."""
    rng = np.random.default_rng(1)
    rows = []
    for v, gain in [('V1', 0.0), ('V2', 0.6), ('V3', 1.2)]:
        for dose in [0, 40, 80, 120]:
            for rep in range(3):
                rows.append({'variete': v, 'dose': dose, 'rep': rep,
                             'rendement': 3 + gain + 0.02 * dose - 0.0001 * dose ** 2
                                          + rng.normal(0, 0.2)})
    return pd.DataFrame(rows)

def test_dendrogram_leaves_and_threshold(df):
    c = Chart(df).dendrogram(labels='cat', columns=['a', 'b'], color_threshold=1.5)
    assert c.linkage.shape == (len(df) - 1, 4)
    assert len(c.ax.get_xticklabels()) == len(df)

def test_regression_perfect_line():
    d = pd.DataFrame({'x': np.arange(10.0), 'y': 2 * np.arange(10.0) + 1})
    c = Chart(d).regression('x', 'y')
    assert c.regression_stats['r2'] == pytest.approx(1.0)
    assert c.regression_stats['coefs'] == pytest.approx([2, 1])

def test_regression_quadratic_has_no_pvalue(essai):
    c = Chart(essai).regression('dose', 'rendement', order=2)
    assert c.regression_stats['p'] is None and 0 < c.regression_stats['r2'] <= 1

def test_regression_too_few_points_raises():
    with pytest.raises(ValueError):
        Chart(pd.DataFrame({'x': [1, 2], 'y': [1, 2]})).regression('x', 'y', order=2)

def test_pca_biplot(essai):
    c = Chart(essai).pca_biplot(columns=['dose', 'rep', 'rendement'], group='variete')
    assert c.pca['explained_variance_ratio'].sum() == pytest.approx(1.0)
    assert list(c.pca['loadings'].index) == ['dose', 'rep', 'rendement']
    assert len(c.ax.get_legend().get_texts()) == 3

def test_mean_bar_heights_are_means(essai):
    c = Chart(essai).mean_bar('variete', 'rendement', error='ci95')
    heights = [p.get_height() for p in c.ax.patches]
    expected = essai.groupby('variete')['rendement'].mean().reindex(['V1', 'V2', 'V3'])
    assert heights == pytest.approx(expected.tolist())

def test_mean_bar_with_hue_groups(essai):
    c = Chart(essai).mean_bar('dose', 'rendement', hue='variete', points=True)
    assert len(c.ax.patches) == 4 * 3

def test_mean_bar_bad_error_raises(essai):
    with pytest.raises(ValueError):
        Chart(essai).mean_bar('variete', 'rendement', error='iqr')

def test_interaction_one_line_per_level(essai):
    c = Chart(essai).interaction('dose', 'variete', 'rendement')
    assert len(c.ax.get_legend().get_texts()) == 3

def test_qqplot(df):
    c = Chart(df).qqplot('a')
    assert len(c.ax.collections[0].get_offsets()) == len(df)

def test_ternary_points_inside_triangle():
    sol = pd.DataFrame({'sable': [70, 20, 40], 'limon': [20, 60, 40], 'argile': [10, 20, 20]})
    c = Chart(sol).ternary('sable', 'limon', 'argile')
    pts = c.ax.collections[-1].get_offsets()
    h = np.sqrt(3) / 2
    assert ((pts[:, 1] >= 0) & (pts[:, 1] <= h) & (pts[:, 0] >= 0) & (pts[:, 0] <= 1)).all()

def test_pca_arrows_inside_axes():
    rng = np.random.default_rng(0)
    d = pd.DataFrame(rng.normal(size=(20, 4)), columns=list('abcd'))
    d['d'] *= 0.01  # variable quasi indépendante → flèche longue sur un autre axe
    c = Chart(d).pca_biplot()
    (x0, x1), (y0, y1) = c.ax.get_xlim(), c.ax.get_ylim()
    for t in c.ax.texts:
        x, y = t.get_position()
        assert x0 <= x <= x1 and y0 <= y <= y1
