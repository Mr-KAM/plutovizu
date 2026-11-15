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


# ── texture du sol, Tukey, seaborn ──────────────────────────────────────────

from plutovizu import texture_class, USDA_POLYGONS, SEABORN, CONTEXTS
from plutovizu.chart import _tukey_letters

@pytest.mark.parametrize('sand, silt, clay, expected', [
    (92, 5, 3, 'Sable'), (80, 15, 5, 'Sable loameux'), (65, 25, 10, 'Loam sableux'),
    (40, 40, 20, 'Loam'), (20, 65, 15, 'Loam limoneux'), (5, 90, 5, 'Limon'),
    (60, 15, 25, 'Loam sablo-argileux'), (30, 35, 35, 'Loam argileux'),
    (10, 55, 35, 'Loam limono-argileux'), (50, 5, 45, 'Argile sableuse'),
    (5, 50, 45, 'Argile limoneuse'), (20, 20, 60, 'Argile'),
    (100, 0, 0, 'Sable'), (0, 0, 100, 'Argile'),   # sommets
    (0.4, 0.4, 0.2, 'Loam'),                        # fractions
])
def test_texture_class(sand, silt, clay, expected):
    assert texture_class(sand, silt, clay) == expected

def test_texture_class_every_point_classified():
    g = np.array([(s, si, 100 - s - si) for s in range(0, 101, 2) for si in range(0, 101 - s, 2)])
    assert (texture_class(g[:, 0], g[:, 1], g[:, 2]) != None).all()  # noqa: E711

def test_usda_polygons_sum_to_100():
    assert len(USDA_POLYGONS) == 12
    assert all(sum(v) == 100 for verts in USDA_POLYGONS.values() for v in verts)

def test_soil_texture_chart():
    sol = pd.DataFrame({'sable': [92, 40, 20], 'limon': [5, 40, 20], 'argile': [3, 20, 60]})
    c = Chart(sol).soil_texture('sable', 'limon', 'argile')
    assert c.texture_classes.tolist() == ['Sable', 'Loam', 'Argile']
    assert len(c.ax.patches) == 12

def test_tukey_letters_separated_and_identical():
    rng = np.random.default_rng(0)
    far = [pd.Series(rng.normal(m, 0.1, 8)) for m in (10, 5, 0)]
    assert _tukey_letters(far) == ['a', 'b', 'c']
    same = [pd.Series(rng.normal(0, 1, 8)) for _ in range(3)]
    assert _tukey_letters(same) == ['a', 'a', 'a']

def test_tukey_letters_overlap():
    # A > C nettement (écart 3), B à mi-chemin (écart 1.5 non significatif) → a, ab, b
    dev = np.array([-2.0, 2.0] * 5)  # mêmes écarts dans chaque groupe : écart-type ≈ 2
    g = [pd.Series(m + dev) for m in (3.0, 1.5, 0.0)]
    assert _tukey_letters(g) == ['a', 'ab', 'b']

def test_mean_bar_letters(essai):
    c = Chart(essai).mean_bar('variete', 'rendement', letters=True)
    assert set(c.tukey_letters) == {'V1', 'V2', 'V3'}
    c2 = Chart(essai).mean_bar('dose', 'rendement', hue='variete', letters=True)
    assert len(c2.tukey_letters) == 4 * 3

@pytest.mark.parametrize('method, kwargs', [
    ('kde', dict(x='rendement', hue='variete')), ('ecdf', dict(x='rendement')),
    ('strip', dict(x='variete', y='rendement')), ('swarm', dict(x='variete', y='rendement')),
    ('boxen', dict(x='variete', y='rendement')), ('pointplot', dict(x='dose', y='rendement')),
    ('countplot', dict(x='variete')), ('residplot', dict(x='dose', y='rendement')),
])
def test_seaborn_axes_level(essai, method, kwargs):
    c = getattr(Chart(essai), method)(**kwargs)
    assert c.sns_grid is None and c.ax.has_data()

@pytest.mark.parametrize('build', [
    lambda d: Chart(d).pairplot(vars=['dose', 'rendement'], hue='variete'),
    lambda d: Chart(d).jointplot('dose', 'rendement'),
    lambda d: Chart(d).clustermap(columns=['dose', 'rep', 'rendement']),
    lambda d: Chart(d).seaborn('relplot', x='dose', y='rendement', col='variete', kind='line'),
])
def test_seaborn_figure_level(essai, build):
    c = build(essai).title('Titre').labels('x', 'y').tight_layout()
    assert c.sns_grid is not None and c.fig is c.sns_grid.figure
    assert c.fig._suptitle.get_text() == 'Titre'

def test_seaborn_unknown_raises(df):
    with pytest.raises(ValueError):
        Chart(df).seaborn('pas_une_fonction')

def test_seaborn_themes_and_context():
    assert set(list_themes('seaborn')) == set(SEABORN) and len(SEABORN) == 5
    set_theme('seaborn-darkgrid', context='poster')
    assert plt.rcParams['axes.facecolor'] == '#EAEAF2'
    assert plt.rcParams['font.size'] > 15
    set_theme('pluto')
    assert plt.rcParams['font.size'] == 10   # le contexte poster ne fuit pas
    with pytest.raises(ValueError):
        set_theme('pluto', context='cinema')


# ── circle packing ──────────────────────────────────────────────────────────

@pytest.fixture
def cultures():
    return pd.DataFrame({
        'famille': ['Céréales', 'Céréales', 'Céréales', 'Légumineuses', 'Légumineuses', 'Tubercules'],
        'culture': ['Riz', 'Maïs', 'Mil', 'Niébé', 'Arachide', 'Manioc'],
        'surface': [40, 25, 15, 12, 20, 8],
    })

def test_circle_packing_flat_areas_proportional(cultures):
    c = Chart(cultures).circle_packing('surface', labels='culture')
    assert len(c.ax.patches) == len(cultures)
    r = {circ.ex['label']: circ.r for circ in c.circles}
    assert (r['Riz'] / r['Manioc']) ** 2 == pytest.approx(40 / 8, rel=1e-6)

def test_circle_packing_grouped(cultures):
    c = Chart(cultures).circle_packing('surface', labels='culture', group='famille', show_values=True)
    assert len(c.ax.patches) == len(cultures) + 3          # 6 cultures + 3 familles
    levels = sorted({circ.level for circ in c.circles})
    assert levels == [1, 2]

def test_circle_packing_two_levels(cultures):
    cultures['cycle'] = ['annuel'] * 5 + ['pluriannuel']
    c = Chart(cultures).circle_packing('surface', labels='culture', group=['cycle', 'famille'])
    assert sorted({circ.level for circ in c.circles}) == [1, 2, 3]

def test_circle_packing_negative_raises(cultures):
    cultures.loc[0, 'surface'] = -1
    with pytest.raises(ValueError):
        Chart(cultures).circle_packing('surface')


# ── analyse qualitative ─────────────────────────────────────────────────────

from plutovizu import word_counts

@pytest.fixture
def avis():
    return pd.DataFrame({
        'reponse': ["Le prix de l'engrais est trop élevé",
                    "Manque d'eau et prix de l'engrais",
                    "L'accès à l'eau reste difficile, le prix aussi",
                    "Semences améliorées trop chères"],
        'q1': ['D’accord', 'Tout à fait', 'Neutre', 'D’accord'],
        'q2': ['Pas du tout', 'Pas d’accord', 'Neutre', 'D’accord'],
    })

ECHELLE = ['Pas du tout', 'Pas d’accord', 'Neutre', 'D’accord', 'Tout à fait']

def test_word_counts_stopwords_and_apostrophes(avis):
    c = word_counts(avis['reponse'])
    assert c['prix'] == 3 and c['eau'] == 2 and c['engrais'] == 2
    assert 'le' not in c and 'est' not in c and "l'eau" not in c

def test_word_counts_ngrams_do_not_cross_stopwords():
    c = word_counts(["prix de l'engrais élevé, prix engrais élevé"], ngram=2)
    assert c['prix engrais'] == 1 and c['engrais élevé'] == 2

def test_word_counts_bad_language():
    with pytest.raises(ValueError):
        word_counts(['texte'], language='de')

def test_word_cloud_from_text_and_weights(avis):
    c = Chart(avis).word_cloud('reponse', shape='circle')
    assert c.word_frequencies.index[0] == 'prix'
    freq = pd.DataFrame({'mot': ['riz', 'mil'], 'n': [10, 3]})
    c2 = Chart(freq).word_cloud(words='mot', weights='n')
    assert c2.word_frequencies.to_dict() == {'riz': 10.0, 'mil': 3.0}

def test_word_cloud_needs_input(avis):
    with pytest.raises(ValueError):
        Chart(avis).word_cloud()

def test_word_frequency(avis):
    c = Chart(avis).word_frequency('reponse', top=3)
    assert c.word_frequencies.index[0] == 'prix' and len(c.ax.patches) == 3

def test_likert_centered_on_neutral(avis):
    c = Chart(avis).likert(['q1', 'q2'], order=ECHELLE)
    t = c.likert_table
    assert t.loc['q1'].sum() == pytest.approx(100)
    # chaque barre commence à -(désaccord + neutre/2)
    starts = {}
    for p in c.ax.patches:
        y = round(p.get_y() + p.get_height() / 2)
        starts[y] = min(starts.get(y, 0), p.get_x())
    expected = sorted(-(t[ECHELLE[:2]].sum(axis=1) + t['Neutre'] / 2))
    assert sorted(starts.values()) == pytest.approx(expected)

def test_likert_unknown_level_raises(avis):
    with pytest.raises(ValueError, match='order'):
        Chart(avis).likert(['q1'], order=ECHELLE[:3])

def test_code_matrix_counts():
    seg = pd.DataFrame({'entretien': ['E1', 'E1', 'E1', 'E2', 'E2'],
                        'theme': ['eau', 'eau', 'prix', 'prix', 'crédit']})
    c = Chart(seg).code_matrix('entretien', 'theme')
    assert c.code_table.loc['E1', 'eau'] == 2 and c.code_table.loc['E2', 'eau'] == 0
    assert list(c.code_table.columns)[0] in ('eau', 'prix')    # codes les plus cités d'abord
    pct = Chart(seg).code_matrix('entretien', 'theme', normalize=True).code_table
    assert pct.sum(axis=1).tolist() == pytest.approx([100, 100])

def test_prezi_theme():
    assert 'prezi' in list_themes('plutovizu')
    with theme_context('prezi'):
        assert plt.rcParams['axes.titlelocation'] == 'center'
        assert plt.rcParams['axes.titleweight'] == 'normal'


# ── types matplotlib ────────────────────────────────────────────────────────

@pytest.fixture
def grille():
    xs, ys = np.meshgrid(np.linspace(0, 4, 9), np.linspace(0, 3, 7))
    return pd.DataFrame({'x': xs.ravel(), 'y': ys.ravel(),
                         'z': np.sin(xs.ravel()) + np.cos(ys.ravel()),
                         'u': -ys.ravel() + 1.5, 'v': xs.ravel() - 2})

@pytest.fixture
def points():
    rng = np.random.default_rng(2)
    x, y = rng.uniform(0, 100, 40), rng.uniform(0, 60, 40)
    return pd.DataFrame({'x': x, 'y': y, 'ph': 5.5 + x / 100 + rng.normal(0, .1, 40),
                         'u': rng.normal(size=40), 'v': rng.normal(size=40)})

def test_bar_grouped_and_stacked():
    d = pd.DataFrame({'an': ['2019', '2020'], 'a': [1, 2], 'b': [3, 4]})
    grouped = Chart(d).bar('an', ['a', 'b'], show_values=True)
    assert len(grouped.ax.patches) == 4 and len(grouped.ax.get_legend().get_texts()) == 2
    stacked = Chart(d).bar('an', ['a', 'b'], stacked=True)
    tops = sorted(p.get_y() + p.get_height() for p in stacked.ax.patches)
    assert tops[-2:] == pytest.approx([4, 6])        # empilement a + b

def test_stem_step_band_errorbar(df):
    Chart(df).stem('x', 'a')
    Chart(df).step('x', ['a', 'b'], fill=True)
    d = df.assign(lo=df['a'] - 1, hi=df['a'] + 1)
    Chart(d).band('x', 'lo', 'hi', y='a')
    c = Chart(df).errorbar('x', 'a', yerr=0.2, group='cat', line=True)
    assert len(c.ax.get_legend().get_texts()) == df['cat'].nunique()

def test_eventplot_dates():
    d = pd.DataFrame({'date': pd.to_datetime(['2024-03-01', '2024-04-15', '2024-05-02']),
                      'variete': ['A', 'B', 'A']})
    c = Chart(d).eventplot('date', group='variete')
    assert [t.get_text() for t in c.ax.get_yticklabels()] == ['A', 'B']

def test_hist2d_and_bihistogram(df):
    Chart(df).hist2d('a', 'b', bins=5)
    d = pd.DataFrame({'age': [10, 12, 30, 35, 36, 50], 'sexe': ['F', 'H', 'F', 'H', 'H', 'F']})
    c = Chart(d).bihistogram('age', 'sexe', bins=[0, 20, 40, 60], horizontal=True)
    widths = [p.get_width() for p in c.ax.patches]
    assert sorted(widths) == [-2, -1, 0, 1, 1, 1]       # H vers la gauche, F vers la droite
    with pytest.raises(ValueError):
        Chart(df).bihistogram('a', 'cat')                # plus de deux groupes

def test_as_grid_detection(grille, points):
    assert Chart(grille)._as_grid('x', 'y', 'z')[2][0].shape == (7, 9)
    assert Chart(points)._as_grid('x', 'y', 'ph') is None

def test_contour_mesh_grid_and_scattered(grille, points):
    Chart(grille).contour('x', 'y', 'z')
    Chart(grille).mesh('x', 'y', 'z')
    Chart(points).contour('x', 'y', 'ph', show_points=True, show_mesh=True)
    Chart(points).mesh('x', 'y', 'ph', show_points=True)

def test_vector_field(grille, points):
    for kind in ('quiver', 'barbs', 'stream'):
        Chart(grille).vector_field('x', 'y', 'u', 'v', kind=kind)
    with pytest.raises(ValueError, match='grille'):
        Chart(points).vector_field('x', 'y', 'u', 'v', kind='stream')

def test_3d_charts(grille, points):
    assert Chart(points).scatter3d('x', 'y', 'ph').ax.name == '3d'
    Chart(points).line3d('x', 'y', 'ph')
    for kind in ('surface', 'wireframe'):
        Chart(grille).surface('x', 'y', 'z', kind=kind)
    Chart(points).surface('x', 'y', 'ph')                     # trisurf automatique
    d = pd.DataFrame({'v': ['A', 'A', 'B'], 's': ['s1', 's2', 's1'], 'r': [3, 4, 5]})
    c = Chart(d).bar3d('v', 's', 'r')
    assert len(c.fig.axes) == 1                               # l'ancien axe 2D est retiré

def test_timeline():
    d = pd.DataFrame({'date': pd.to_datetime(['2024-06-01', '2024-07-15', '2024-10-20']),
                      'etape': ['Semis', 'Désherbage', 'Récolte']})
    c = Chart(d).timeline('date', 'etape')
    assert [t.get_text() for t in c.ax.texts] == ['Semis', 'Désherbage', 'Récolte']

def test_confidence_ellipse_axes():
    rng = np.random.default_rng(0)
    d = pd.DataFrame({'x': rng.normal(0, 2, 4000), 'y': rng.normal(0, 1, 4000)})
    c = Chart(d).confidence_ellipse('x', 'y', level=0.95)
    e = c.ax.patches[0]
    from scipy.stats import chi2
    k = 2 * np.sqrt(chi2.ppf(0.95, 2))
    assert max(e.width, e.height) == pytest.approx(k * 2, rel=0.05)   # écart-type 2
    assert min(e.width, e.height) == pytest.approx(k * 1, rel=0.05)   # écart-type 1

def test_donut_simple_and_nested():
    d = pd.DataFrame({'g': ['A', 'A', 'B'], 'n': ['a1', 'a2', 'b1'], 'v': [2, 3, 5]})
    simple = Chart(d).donut('v', labels='n', center_text='10')
    assert len(simple.ax.patches) == 3
    nested = Chart(d).donut('v', labels='n', group='g')
    assert len(nested.ax.patches) == 3 + 2                    # éléments + groupes

def test_polar_kinds():
    d = pd.DataFrame({'dir': ['N', 'E', 'S', 'O'] * 2, 'f': [5, 3, 2, 4, 1, 2, 1, 1],
                      'vit': ['faible'] * 4 + ['fort'] * 4})
    c = Chart(d).polar('dir', 'f', group='vit', compass=True)
    assert c.ax.name == 'polar' and len(c.ax.patches) == 8
    Chart(d).polar('dir', 'f', kind='line')
    deg = pd.DataFrame({'angle': [0, 90, 180, 270], 'r': [1, 2, 3, 4]})
    Chart(deg).polar('angle', 'r', kind='scatter')

def test_pivot_heatmap(essai):
    c = Chart(essai).pivot_heatmap('dose', 'variete', 'rendement')
    assert c.pivot_table.shape == (3, 4)

def test_radial_and_progress_bars():
    d = pd.DataFrame({'act': ['Pub', 'Lettre'], 'pct': [75, 25]})
    c = Chart(d).radial_bar('pct', labels='act')
    filled = [p for p in c.ax.patches if p.theta2 == 90]   # les rails vont de 0 à 360
    assert sorted(round(90 - p.theta1) for p in filled) == [90, 270]   # 25 % et 75 % d'un tour
    Chart(d).progress_bar(80)
    Chart(d).progress_bar('pct', labels='act')


# ── thème par défaut et couleurs qui suivent le thème ────────────────────────

def test_import_applies_editorial_by_default():
    import subprocess, sys
    code = ("import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt; "
            "import plutovizu; print(plt.rcParams['axes.grid.axis'], "
            "plt.rcParams['axes.spines.left'], plutovizu.DEFAULT_THEME)")
    out = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, check=True)
    assert out.stdout.split() == ['y', 'False', 'editorial']   # réglages propres à editorial

def test_set_theme_without_argument_is_editorial():
    set_theme('neon')
    set_theme()
    assert plt.rcParams['axes.grid.axis'] == 'y'

def _hex(c):
    return matplotlib.colors.to_hex(c)

@pytest.mark.parametrize('theme', ['editorial', 'neon', 'sahel'])
def test_waterfall_and_dumbbell_follow_theme(theme):
    with theme_context(theme):
        palette = [_hex(c) for c in plt.rcParams['axes.prop_cycle'].by_key()['color']]
        w = Chart(pd.DataFrame({'p': ['a', 'b'], 'v': [5, -2]})).waterfall('p', 'v')
        assert [_hex(p.get_facecolor()) for p in w.ax.patches] == palette[:2]
        d = Chart(pd.DataFrame({'k': ['x'], 'a': [1], 'b': [2]})).dumbbell('k', 'a', 'b')
        assert [_hex(col.get_facecolor()[0]) for col in d.ax.collections] == palette[:2]

@pytest.mark.parametrize('theme', ['editorial', 'pluto'])
def test_hexbin_and_heatmap_follow_theme(df, theme):
    with theme_context(theme):
        h = Chart(df).hexbin('a', 'b', gridsize=5)
        assert h.ax.collections[0].get_cmap().name == plt.rcParams['image.cmap']
        m = Chart(df).heatmap()
        im = m.ax.images[0]
        first = _hex(plt.rcParams['axes.prop_cycle'].by_key()['color'][0])
        assert _hex(im.cmap(1.0)) == first                              # r = +1 → 1re couleur
        fond = matplotlib.colors.to_rgba(plt.rcParams['axes.facecolor'])
        assert np.allclose(im.cmap(0.5), fond, atol=0.02)  # r = 0 → fond (256 paliers)
        assert im.norm.vmin == -1 and im.norm.vmax == 1
        # texte lisible : sombre sur cellule claire, blanc sur cellule foncée
        for t in m.ax.texts:
            r = float(t.get_text())
            assert t.get_color() == Chart._ink_on(im.cmap(im.norm(r)))


# ── palettes appliquées au graphique courant ────────────────────────────────

def test_set_colors_applies_to_this_chart_only(df):
    before = plt.rcParams['axes.prop_cycle']
    c = Chart(df).set_colors(['red', 'blue']).line('x', ['a', 'b'])
    assert [_hex(l.get_color()) for l in c.ax.lines] == ['#ff0000', '#0000ff']
    assert plt.rcParams['axes.prop_cycle'] == before            # aucun effet global
    nxt = Chart(df).line('x', ['a', 'b'])
    assert _hex(nxt.ax.lines[0].get_color()) != '#ff0000'

def test_custom_and_gradient_palettes_apply_now(df):
    c = Chart(df).create_custom_palette('p', ['green', 'orange']).line('x', ['a', 'b'])
    assert [_hex(l.get_color()) for l in c.ax.lines] == [_hex('green'), _hex('orange')]
    g = Chart(df).generate_gradient_palette('g', 'black', 'white', n_colors=2).line('x', ['a', 'b'])
    assert [_hex(l.get_color()) for l in g.ax.lines] == ['#000000', '#ffffff']

def test_named_palette_feeds_new_methods(essai):
    c = Chart(essai).set_colors('Set2').mean_bar('variete', 'rendement')
    import seaborn as sns
    assert _hex(c.ax.patches[0].get_facecolor()) == _hex(sns.color_palette('Set2')[0])


# ── ridgeline : visible quelles que soient les unités ───────────────────────

@pytest.mark.parametrize('scale', [1e-3, 1, 50_000])
def test_ridgeline_visible_whatever_the_units(scale):
    rng = np.random.default_rng(3)
    d = pd.DataFrame({'v': np.concatenate([rng.normal(m, 0.2, 150) for m in (1, 2, 3)]) * scale,
                      'g': np.repeat(['A', 'B', 'C'], 150)})
    c = Chart(d).ridgeline('v', 'g', overlap=1.5)
    heights = []
    for poly in c.ax.collections:                      # une surface remplie par groupe
        ys = poly.get_paths()[0].vertices[:, 1]
        heights.append(ys.max() - ys.min())
    assert len(heights) == 3
    assert max(heights) == pytest.approx(1.5, rel=0.01)   # la plus haute crête = overlap lignes
    assert min(heights) > 0.5                             # toutes nettement visibles
    pos = {t.get_text(): y for t, y in zip(c.ax.get_yticklabels(), c.ax.get_yticks())}
    assert pos['A'] > pos['B'] > pos['C']              # premier groupe en haut

def test_ridgeline_needs_data():
    with pytest.raises(ValueError):
        Chart(pd.DataFrame({'v': [1.0], 'g': ['A']})).ridgeline('v', 'g')


# ── climat & sols ───────────────────────────────────────────────────────────

MOIS = ['J', 'F', 'M', 'A', 'M2', 'J2', 'J3', 'A2', 'S', 'O', 'N', 'D']


@pytest.fixture
def station():
    return pd.DataFrame({'mois': MOIS,
                         'temp': [24, 26, 28, 30, 31, 30, 28, 27, 27, 28, 27, 25],
                         'pluie': [0, 0, 0, 5, 30, 100, 300, 300, 150, 40, 2, 0],
                         'etp': [140, 150, 180, 190, 200, 170, 140, 120, 120, 140, 140, 135],
                         'tmin': [-1, 2, 5, 8, 10, 12, 12, 12, 10, 6, 2, -2]})


def test_walter_lieth_scale_and_summary(station):
    c = Chart(station).walter_lieth('mois', 'temp', 'pluie', tmin='tmin', station='Station A')
    precip_line = c.ax.lines[1].get_ydata()
    assert precip_line[5] == pytest.approx(50)          # 100 mm → ligne des 50 °C
    assert precip_line[6] == pytest.approx(60)          # 300 mm → 50 + 200/20
    labels = {t.get_text() for t in c.ax2.get_yticklabels()}
    assert {'100', '300'} <= labels
    assert c.climate_summary['p_totale'] == pytest.approx(927)
    assert 'J' in c.climate_summary['mois_secs'] and 'J3' not in c.climate_summary['mois_secs']


def test_climogram_closed_loop(station):
    d = pd.concat([station.assign(st='A'), station.assign(st='B', temp=station['temp'] - 5)])
    c = Chart(d).climogram('pluie', 'temp', month='mois', group='st')
    assert len(c.ax.lines) == 2 and len(c.ax.lines[0].get_xdata()) == 13   # boucle fermée


def test_water_balance(station):
    c = Chart(station).water_balance('mois', 'pluie', 'etp')
    assert c.water_table['bilan'].tolist() == (station['pluie'] - station['etp']).tolist()
    assert c.water_summary['excedent'] == pytest.approx(160 + 180 + 30)   # juil., août, sept.


def test_degree_days_and_stages():
    d = pd.DataFrame({'jour': pd.date_range('2024-06-01', periods=10),
                      'tmin': [10.0] * 10, 'tmax': [30.0] * 10})
    c = Chart(d).degree_days('jour', 'tmin', 'tmax', base=10, stages={'Levée': 50})
    assert c.gdd['dj'].tolist() == [10.0] * 10
    assert c.stage_dates['Levée'] == pd.Timestamp('2024-06-05')
    capped = Chart(d).degree_days('jour', 'tmin', 'tmax', base=10, cap=20)
    assert capped.gdd['dj'].iloc[0] == 5.0                      # (20 + 10) / 2 − 10


def test_wind_rose_sectors_and_total():
    d = pd.DataFrame({'dir': [355, 5, 90, 180, 270, 92], 'v': [1, 3, 5, 7, 9, 1]})
    c = Chart(d).wind_rose('dir', 'v', sectors=4)
    t = c.wind_table
    assert t.to_numpy().sum() == pytest.approx(100)
    assert t.loc[0].sum() == pytest.approx(100 * 2 / 6)         # 355° et 5° → nord
    assert c.ax.name == 'polar'


def test_anomaly_baselines():
    d = pd.DataFrame({'an': range(2000, 2010), 't': np.arange(10.0)})
    c = Chart(d).anomaly('an', 't', window=3)
    assert c.anomalies.mean() == pytest.approx(0)
    ref = Chart(d).anomaly('an', 't', baseline=(2000, 2004))
    assert ref.anomaly_reference == pytest.approx(2.0)


def test_climate_stripes():
    d = pd.DataFrame({'an': range(1990, 2020), 't': np.linspace(14, 15.5, 30)})
    c = Chart(d).climate_stripes('an', 't')
    faces = [p.get_facecolor() for p in c.ax.patches]
    assert len(faces) == 30
    assert faces[0][2] > faces[0][0] and faces[-1][0] > faces[-1][2]   # bleu → rouge


def test_soil_and_depth_profiles():
    h = pd.DataFrame({'fosse': ['P1'] * 3 + ['P2'] * 2, 'horizon': ['A', 'B', 'C', 'A', 'Bt'],
                      'haut': [0, 20, 60, 0, 30], 'bas': [20, 60, 120, 30, 90]})
    c = Chart(h).soil_profile('haut', 'bas', 'horizon', profile='fosse')
    assert len(c.ax.patches) == 5
    assert c.ax.get_ylim()[0] > c.ax.get_ylim()[1]               # profondeur vers le bas
    p = pd.DataFrame({'z': [0, 10, 30, 60] * 2, 'hum': [30, 25, 20, 18, 15, 14, 16, 17],
                      'date': ['juin'] * 4 + ['août'] * 4})
    dp = Chart(p).depth_profile('z', 'hum', group='date')
    assert dp.ax.yaxis_inverted()


# ── map ─────────────────────────────────────────────────────────────────────

@pytest.fixture
def villes():
    return pd.DataFrame({'ville': ['Dakar', 'Thiès', 'Kaolack', 'Tambacounda'],
                         'lon': [-17.44, -16.93, -16.07, -13.67],
                         'lat': [14.69, 14.79, 14.15, 13.77],
                         'pluie': [400, 500, 650, 850],
                         'zone': ['côte', 'centre', 'centre', 'est']})


def test_map_points_without_basemap(villes):
    c = Chart(villes).map_points('lon', 'lat', value='pluie', size='pluie', labels='ville',
                                 basemap=False)
    assert c.ax.get_aspect() == pytest.approx(1 / np.cos(np.radians(villes['lat'].mean())))
    assert [t.get_text() for t in c.ax.texts] == villes['ville'].tolist()
    g = Chart(villes).map_points('lon', 'lat', group='zone', basemap=False)
    assert len(g.ax.get_legend().get_texts()) == 3


def test_basemap_failure_only_warns(villes, monkeypatch):
    contextily = pytest.importorskip('contextily')

    def offline(*a, **k):
        raise ConnectionError('pas de réseau')
    monkeypatch.setattr(contextily, 'add_basemap', offline)
    with pytest.warns(UserWarning, match='Fond de carte indisponible'):
        Chart(villes).map_points('lon', 'lat')


def test_choropleth_numeric_categorical_and_error(villes):
    gpd = pytest.importorskip('geopandas')
    from shapely.geometry import box
    parcelles = gpd.GeoDataFrame({'id': ['P1', 'P2', 'P3', 'P4'], 'rdt': [3.1, 4.2, 5.0, 2.4],
                                  'culture': ['riz', 'mil', 'riz', 'maïs']},
                                 geometry=[box(i, 0, i + 1, 1) for i in range(4)],
                                 crs='EPSG:32628')
    c = Chart(parcelles).choropleth('rdt', labels='id')
    assert [t.get_text() for t in c.ax.texts] == ['P1', 'P2', 'P3', 'P4']
    cat = Chart(parcelles).choropleth('culture')
    assert len(cat.ax.get_legend().get_texts()) == 3
    with pytest.raises(ValueError, match='GeoDataFrame'):
        Chart(villes).choropleth('pluie')


def test_flow_map(villes):
    flux = pd.DataFrame({'de': ['Kaolack', 'Tambacounda'], 'vers': ['Dakar', 'Dakar'],
                         'lon1': [-16.07, -13.67], 'lat1': [14.15, 13.77],
                         'lon2': [-17.44, -17.44], 'lat2': [14.69, 14.69], 't': [1200, 400]})
    c = Chart(flux).flow_map('lon1', 'lat1', 'lon2', 'lat2', weight='t', label_from='de',
                             label_to='vers', basemap=False)
    widths = sorted(p.get_linewidth() for p in c.ax.patches)
    assert len(widths) == 2 and widths[-1] > widths[0]
    assert sorted(t.get_text() for t in c.ax.texts) == ['Dakar', 'Kaolack', 'Tambacounda']


# ── map : nouveaux types et éléments de carte ───────────────────────────────

def test_geo_ticks_in_degrees(villes):
    c = Chart(villes).map_points('lon', 'lat', basemap=False)
    fmt = c.ax.xaxis.get_major_formatter()
    assert fmt(-17, 0) == '17° O' and c.ax.yaxis.get_major_formatter()(14, 0) == '14° N'


def test_size_legend_with_group_legend(villes):
    c = Chart(villes).map_points('lon', 'lat', group='zone', size='pluie', basemap=False)
    legends = [a for a in c.ax.get_children() if isinstance(a, matplotlib.legend.Legend)]
    titles = sorted(lg.get_title().get_text() for lg in legends)
    assert titles == ['pluie', 'zone']                       # les deux légendes coexistent


def test_class_breaks_schemes():
    v = np.array([1, 2, 3, 10, 11, 12, 50, 51, 52], float)
    nat = Chart._class_breaks(v, 3, 'natural')
    assert nat[0] == 1 and nat[-1] == 52
    assert 3 <= nat[1] < 10 and 12 <= nat[2] < 50            # coupures entre les groupes
    assert Chart._class_breaks(v, 4, 'equal') == pytest.approx([1, 13.75, 26.5, 39.25, 52])
    assert len(Chart._class_breaks(v, 3, 'quantiles')) == 4
    with pytest.raises(ValueError):
        Chart._class_breaks(v, 3, 'jenks')


def test_choropleth_scheme_classes():
    gpd = pytest.importorskip('geopandas')
    from shapely.geometry import box
    g = gpd.GeoDataFrame({'v': [1, 2, 3, 10, 11, 12, 50, 51, 52]},
                         geometry=[box(i, 0, i + 1, 1) for i in range(9)], crs='EPSG:32628')
    c = Chart(g).choropleth('v', scheme='natural', k=3)
    assert len(c.classes) == 4
    colors = {tuple(np.round(fc, 3)) for fc in c.ax.collections[0].get_facecolor()}
    assert len(colors) == 3                                  # 3 classes → 3 couleurs


def test_density_map_peak_on_cluster():
    rng = np.random.default_rng(0)
    d = pd.DataFrame({'lon': np.r_[rng.normal(-16, .05, 300), rng.uniform(-17, -13, 30)],
                      'lat': np.r_[rng.normal(14, .05, 300), rng.uniform(12.5, 16, 30)]})
    c = Chart(d).density_map('lon', 'lat', basemap=False)
    X, Y, Z = c.density_grid
    i = np.unravel_index(Z.argmax(), Z.shape)
    assert abs(X[i] + 16) < 0.2 and abs(Y[i] - 14) < 0.2 and Z.max() == pytest.approx(1)


def test_field_map(villes):
    c = Chart(villes).field_map('lon', 'lat', 'pluie', labels='ville', basemap=False)
    assert c.ax.collections and [t.get_text() for t in c.ax.texts if t.get_text() in set(villes['ville'])]


def test_route_map_order_and_markers():
    d = pd.DataFrame({'lon': [-15.0, -14.0, -16.0], 'lat': [15.0, 14.0, 16.0], 'etape': [2, 3, 1]})
    c = Chart(d).route_map('lon', 'lat', order='etape', labels='etape', basemap=False)
    line = c.ax.lines[0]
    assert list(line.get_xdata()) == [-16.0, -15.0, -14.0]   # suit l'ordre des étapes


def test_scale_bar_length_in_degrees(villes):
    c = Chart(villes).map_points('lon', 'lat', basemap=False).scale_bar(length=100)
    bar = [p for p in c.ax.patches if isinstance(p, matplotlib.patches.Rectangle)]
    width_deg = sum(p.get_width() for p in bar)
    lat0 = villes['lat'].mean()
    assert width_deg * 111.32 * np.cos(np.radians(lat0)) == pytest.approx(100, rel=1e-6)
    auto = Chart(villes).map_points('lon', 'lat', basemap=False).scale_bar()
    assert auto.scale_length_km in {10, 20, 50, 100, 200}


def test_scale_bar_in_meters_and_requires_map(df):
    gpd = pytest.importorskip('geopandas')
    from shapely.geometry import box
    g = gpd.GeoDataFrame({'v': [1, 2]}, geometry=[box(0, 0, 500, 300), box(500, 0, 1000, 300)],
                         crs='EPSG:32628')
    c = Chart(g).choropleth('v').scale_bar()
    assert c.scale_length_km in {0.1, 0.2}                    # ~ 1/5 de 1 km de large
    with pytest.raises(ValueError, match='après une carte'):
        Chart(df).scale_bar()


def test_north_arrow_and_inset(villes):
    c = (Chart(villes).map_points('lon', 'lat', basemap=False)
         .north_arrow(loc='upper left').inset_map(zoom_out=5, basemap=False))
    assert any(t.get_text() == 'N' for t in c.ax.texts)
    (x0, x1) = c.ax.get_xlim()
    (i0, i1) = c.inset_ax.get_xlim()
    assert (i1 - i0) == pytest.approx((x1 - x0) * 5, rel=1e-6)
    with pytest.raises(ValueError):
        c.north_arrow(loc='centre')


# ── écologie ────────────────────────────────────────────────────────────────

from plutovizu import diversity_indices, glm_fit, ordinate
from plutovizu.ecology import lowess, rarefaction, accumulation as accumulate


@pytest.fixture
def communaute():
    """20 sites le long d'un gradient d'humidité, 8 espèces à réponse en cloche."""
    rng = np.random.default_rng(4)
    hum = np.linspace(10, 90, 20)
    optima = np.linspace(15, 85, 8)
    ab = rng.poisson(30 * np.exp(-((hum[:, None] - optima[None, :]) / 15) ** 2)) + (rng.random((20, 8)) < .1)
    d = pd.DataFrame(ab, columns=[f'sp{i}' for i in range(8)])
    d['humidite'] = hum
    d['ph'] = 5 + rng.normal(0, .3, 20) + hum / 100
    d['site'] = [f'S{i:02d}' for i in range(20)]
    d['zone'] = np.where(hum < 50, 'sèche', 'humide')
    return d


SP = [f'sp{i}' for i in range(8)]


def test_glm_fit_recovers_coefficients():
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 8, 20000)
    X = np.c_[np.ones_like(x), x]
    y = (rng.random(len(x)) < 1 / (1 + np.exp(-(-3 + 0.8 * x)))).astype(float)
    assert glm_fit(X, y, 'binomial')['coefs'] == pytest.approx([-3, 0.8], abs=0.1)
    yp = rng.poisson(np.exp(0.5 + 0.3 * x))
    assert glm_fit(X, yp, 'poisson')['coefs'] == pytest.approx([0.5, 0.3], abs=0.02)
    with pytest.raises(ValueError):
        glm_fit(X, yp, 'binomial')                       # réponse non 0/1


def test_glm_curve_logistic_and_niche_optimum():
    rng = np.random.default_rng(1)
    x = rng.uniform(4, 9, 600)
    p = 1 / (1 + np.exp(-(-2 + 3.5 * (x - 6.5) - 2.0 * (x - 6.5) ** 2)))
    d = pd.DataFrame({'ph': x, 'presence': (rng.random(600) < p).astype(int)})
    c = Chart(d).glm_curve('ph', 'presence')
    assert 0 < c.glm['d2'] < 1 and len(c.glm['coefs_original']) == 2
    niche = Chart(d).glm_curve('ph', 'presence', degree=2)
    assert niche.glm['optimum'] == pytest.approx(6.5 + 3.5 / 4, abs=0.25)   # sommet de la parabole


def test_glm_curve_poisson():
    rng = np.random.default_rng(2)
    x = rng.uniform(0, 5, 300)
    d = pd.DataFrame({'cond': x, 'abondance': rng.poisson(np.exp(0.2 + 0.4 * x))})
    c = Chart(d).glm_curve('cond', 'abondance', family='poisson')
    assert c.glm['coefs_original'] == pytest.approx([0.2, 0.4], abs=0.15)


def test_lowess_and_response_optimum():
    x = np.linspace(0, 10, 200)
    y = np.exp(-(x - 6.5) ** 2 / 4)
    assert x[np.argmax(lowess(x, y, x, 0.3))] == pytest.approx(6.5, abs=0.1)
    rng = np.random.default_rng(3)
    d = pd.DataFrame({'alt': x, 'ab': 20 * y + rng.normal(0, 1, 200)})
    c = Chart(d).response_curve('alt', 'ab', n_boot=30)
    assert c.response['optimum'] == pytest.approx(6.5, abs=0.5)


def test_count_distribution_detects_zero_inflation():
    rng = np.random.default_rng(5)
    v = np.where(rng.random(500) < 0.4, 0, rng.poisson(4, 500))
    c = Chart(pd.DataFrame({'n': v})).count_distribution('n')
    st = c.count_stats
    assert st['zeros_observes'] > 3 * st['zeros_poisson'] and st['dispersion'] > 1.5
    with pytest.raises(ValueError):
        Chart(pd.DataFrame({'n': [1.5, 2]})).count_distribution('n')


def test_forest_plot_significance_and_se():
    d = pd.DataFrame({'var': ['a', 'b', 'c'], 'est': [0.8, 0.1, -0.5], 'se': [0.2, 0.2, 0.2]})
    c = Chart(d).forest_plot('var', 'est', se='se')
    assert c.forest_table['significatif'].tolist() == [True, False, True]
    assert [t.get_text() for t in c.ax.get_yticklabels()] == ['c', 'b', 'a']   # 'a' en haut
    o = pd.DataFrame({'v': ['x'], 'or': [2.0], 'lo': [1.2], 'hi': [3.1]})
    assert Chart(o).forest_plot('v', 'or', 'lo', 'hi', ref=1, log=True).ax.get_xscale() == 'log'
    with pytest.raises(ValueError):
        Chart(d).forest_plot('var', 'est')


def test_ordinate_invariants(communaute):
    from scipy.stats import chi2_contingency
    Y = communaute[SP].to_numpy(float)
    ca = ordinate(Y, method='ca')
    assert ca['eigenvalues'].sum() == pytest.approx(chi2_contingency(Y, correction=False)[0] / Y.sum())
    X = communaute[['humidite', 'ph']].to_numpy()
    rda = ordinate(Y, X, 'rda')
    Yc = Y - Y.mean(0)
    Xc = (X - X.mean(0)) / X.std(0)
    fit = Xc @ np.linalg.lstsq(Xc, Yc, rcond=None)[0]
    assert rda['constrained'] == pytest.approx((fit ** 2).sum() / (Yc ** 2).sum())
    cca = ordinate(Y, X, 'cca')
    assert 0 < cca['constrained'] < 1 and cca['env'].shape[0] == 2
    assert abs(cca['env'][0, 0]) > 0.8               # l'humidité structure le 1er axe
    with pytest.raises(ValueError):
        ordinate(Y, method='rda')                     # rda sans variables


def test_ordination_chart(communaute):
    c = Chart(communaute).ordination(SP, env=['humidite', 'ph'], method='cca', labels='site', group='zone')
    assert c.ax.get_xlabel().startswith('CCA1 (')
    assert 'humidite' in [t.get_text() for t in c.ax.texts]
    Chart(communaute).ordination(SP, method='ca')
    Chart(communaute).ordination(SP, method='pca', scale=True)


def test_diversity_indices_known_values():
    t = diversity_indices(pd.DataFrame([[5, 5, 5, 5], [10, 0, 0, 0]]))
    assert t.loc[0, 'shannon'] == pytest.approx(np.log(4)) and t.loc[0, 'simpson'] == pytest.approx(0.75)
    assert t.loc[0, 'pielou'] == pytest.approx(1) and t.loc[1, 'shannon'] == 0 and t.loc[1, 'richesse'] == 1


def test_rarefaction_and_accumulation_endpoints():
    counts = np.array([10, 5, 3, 1, 1])
    r = rarefaction(counts, [1, counts.sum()])
    assert r == pytest.approx([1, 5])
    rng = np.random.default_rng(6)
    mean, sd, info = accumulate(rng.random((12, 20)) < .3)
    assert mean[-1] == info['s_obs'] and np.all(np.diff(mean) >= 0) and info['chao2'] >= info['s_obs']


def test_community_charts(communaute):
    Chart(communaute).rank_abundance(SP, labels='site')
    Chart(communaute).rank_abundance(SP, pooled=True)
    a = Chart(communaute).accumulation(SP)
    assert a.accumulation_stats['s_obs'] == 8
    Chart(communaute.head(4)).accumulation(SP, kind='individuals', labels='site')
    d = Chart(communaute).diversity(SP, labels='site')
    assert list(d.diversity_table.columns[:2]) == ['site', 'richesse']
    g = Chart(communaute).diversity(SP, group='zone', letters=True)
    assert set(g.tukey_letters) == {'sèche', 'humide'}
    with pytest.raises(ValueError):
        Chart(communaute).diversity(SP, index='chao')


# ── diagramme ombrothermique en courbes, périodes sèches hachurées ──────────

def test_ombrothermic_lines_hatched_dry_periods(station):
    c = Chart(station).ombrothermic('mois', 'temp', 'pluie', kind='lines')
    expected = station.loc[station['pluie'] < 2 * station['temp'], 'mois'].tolist()
    assert c.dry_months == expected
    hatched = [col for col in c.ax.collections if col.get_hatch()]
    assert hatched and hatched[0].get_label() == 'Période sèche'
    precip_line = c.ax.lines[1].get_ydata()
    assert precip_line == pytest.approx(station['pluie'] / 2)          # échelle P = 2T
    assert c.ax2.get_ylim()[1] == pytest.approx(2 * c.ax.get_ylim()[1])  # axe droit en mm


def test_ombrothermic_kind_validation(station):
    with pytest.raises(ValueError):
        Chart(station).ombrothermic('mois', 'temp', 'pluie', kind='aires')
    Chart(station).ombrothermic('mois', 'temp', 'pluie')                 # défaut : barres


# ──────────────────────────────── Pédologie

def test_retention_curve_recovers_van_genuchten():
    pf = np.array([0, 1, 1.5, 2, 2.5, 3, 3.5, 4.2])
    theta = Chart._van_genuchten(10 ** pf, .08, .45, .02, 1.45)
    c = Chart(pd.DataFrame({'pF': pf, 'theta': theta})).retention_curve('pF', 'theta')
    r = c.retention.iloc[0]
    assert r['alpha'] == pytest.approx(.02, rel=1e-3) and r['n'] == pytest.approx(1.45, rel=1e-3)
    assert r['capacite_champ'] == pytest.approx(Chart._van_genuchten(10 ** 2.5, .08, .45, .02, 1.45))
    assert r['reserve_utile'] == pytest.approx(r['capacite_champ'] - r['point_fletrissement'])
    # kPa → pF : 10 kPa ≈ 102 cm ≈ pF 2,01 ; sans ajustement → interpolation
    kpa = Chart(pd.DataFrame({'s': [1, 10, 100, 1500], 'w': [.4, .3, .2, .1]})).retention_curve(
        's', 'w', unit='kPa', fit=False)
    assert kpa.retention.iloc[0]['point_fletrissement'] == pytest.approx(.1, abs=.01)
    with pytest.raises(ValueError):
        Chart(pd.DataFrame({'s': [1], 'w': [1]})).retention_curve('s', 'w', unit='bar')


def test_grain_size_curve_d_values():
    d = pd.DataFrame({'d': [0.001, 0.01, 0.1, 1.0], 'p': [0, 10, 60, 100]})
    g = Chart(d).grain_size_curve('d', 'p').grain_size.iloc[0]
    assert g['D10'] == pytest.approx(0.01) and g['D60'] == pytest.approx(0.1)
    assert g['Cu'] == pytest.approx(10)
    assert np.isnan(Chart(d.iloc[2:]).grain_size_curve('d', 'p').grain_size.iloc[0]['D10'])


def test_exchange_complex_base_saturation():
    d = pd.DataFrame({'h': ['A', 'B'], 'Ca': [6, 3], 'Mg': [2, 1], 'acid': [2, 4], 'T': [10, 10]})
    c = Chart(d).exchange_complex('h', ['Ca', 'Mg'], acidity='acid', cec='T')
    assert c.base_saturation.tolist() == pytest.approx([80, 40])
    c = Chart(d).exchange_complex('h', ['Ca', 'Mg'], acidity='acid')    # T = S + acidité
    assert c.base_saturation.tolist() == pytest.approx([80, 50])


@pytest.mark.parametrize('model', ['philip', 'horton'])
def test_infiltration_curve_fits(model):
    t = np.array([1, 2, 5, 10, 20, 30, 60, 90, 120], float)
    cum = 6 * np.sqrt(t) + .4 * t if model == 'philip' else .8 * t + (4 - .8) * (1 - np.exp(-.15 * t)) / .15
    c = Chart(pd.DataFrame({'t': t, 'I': cum})).infiltration_curve('t', 'I', model=model)
    assert c.infiltration['final_rate'] == pytest.approx(.4 if model == 'philip' else .8, rel=1e-3)
    with pytest.raises(ValueError):
        Chart(pd.DataFrame({'t': t, 'I': cum})).infiltration_curve('t', 'I', model='kostiakov')


def test_isopleth_dates_and_depth_down():
    dates = pd.date_range('2025-01-01', periods=6, freq='MS')
    d = pd.DataFrame([(dt, z, z + k) for k, dt in enumerate(dates) for z in (10, 30, 60)],
                     columns=['date', 'prof', 'hum'])
    c = Chart(d).isopleth('date', 'prof', 'hum')
    lo, hi = c.ax.get_ylim()
    assert lo > hi and len(c.fig.axes) == 2                            # profondeur vers le bas + barre de couleur


def test_soil_properties_panels_and_steps():
    d = pd.DataFrame({'haut': [0, 20], 'bas': [20, 50], 'pH': [6, 7], 'MO': [3, 1]})
    c = Chart(d).soil_properties('haut', ['pH', 'MO'], bottom='bas').title('Profil')
    assert len(c.axes) == 2 and c.ax is c.axes[0]
    assert c.axes[0].lines[0].get_ydata().tolist() == [0, 20, 20, 50]   # paliers par horizon
    assert c.axes[0].get_ylim()[0] > 50 and c.fig._suptitle.get_text() == 'Profil'


# ──────────────────────────────── Graphes & réseaux

@pytest.fixture
def pistes():
    return pd.DataFrame([('A', 'B', 4), ('A', 'C', 1), ('C', 'B', 2), ('B', 'D', 5), ('C', 'D', 8)],
                        columns=['de', 'vers', 'km'])


def test_shortest_path_dijkstra(pistes):
    c = Chart(pistes).shortest_path('de', 'vers', 'A', 'D', weight='km')
    assert c.path == ['A', 'C', 'B', 'D'] and c.path_length == 8
    # orienté : D → A impossible
    from plutovizu import shortest_path
    with pytest.raises(ValueError):
        shortest_path(pistes, 'de', 'vers', 'D', 'A', weight='km', directed=True)
    with pytest.raises(ValueError):
        shortest_path(pistes, 'de', 'vers', 'A', 'Z')


def test_spanning_tree_total(pistes):
    c = Chart(pistes).spanning_tree('de', 'vers', weight='km')
    assert c.tree_weight == 8 and len(c.tree) == 3                   # 1 + 2 + 5


def test_max_flow_value_and_conservation():
    net = pd.DataFrame([('s', 'a', 3), ('s', 'b', 2), ('a', 'b', 1), ('a', 't', 2), ('b', 't', 3)],
                       columns=['de', 'vers', 'cap'])
    c = Chart(net).max_flow('de', 'vers', 'cap', 's', 't')
    assert c.flow_value == 5
    f = c.flow
    assert (f['flux'] <= f['cap'] + 1e-9).all()
    inflow_b = f.loc[f['vers'] == 'b', 'flux'].sum()
    assert inflow_b == pytest.approx(f.loc[f['de'] == 'b', 'flux'].sum())   # conservation
    dec = Chart(net.assign(cap=net['cap'] / 2)).max_flow('de', 'vers', 'cap', 's', 't')
    assert dec.flow_value == pytest.approx(2.5)                            # capacités décimales


def test_critical_path_schedule():
    t = pd.DataFrame({'t': ['A', 'B', 'C', 'D'], 'd': [3, 2, 4, 1], 'p': ['', 'A', 'A', 'B; C']})
    c = Chart(t).critical_path('t', 'd', 'p')
    s = c.schedule
    assert c.project_duration == 8
    assert s['critique'].to_dict() == {'A': True, 'B': False, 'C': True, 'D': True}
    assert s.at['B', 'marge'] == 2
    with pytest.raises(ValueError):
        Chart(pd.DataFrame({'t': ['A', 'B'], 'd': [1, 1], 'p': ['B', 'A']})).critical_path('t', 'd', 'p')


def test_tsp_route_square():
    pts = pd.DataFrame({'x': [0, 1, 0, 1], 'y': [0, 1, 1, 0], 'n': list('abcd')})
    c = Chart(pts).tsp_route('x', 'y', labels='n', start='a')
    assert c.tour_length == pytest.approx(4) and c.tour[0] == 'a'
    assert c.tour_table['ordre'].tolist() == [1, 2, 3, 4]


def test_network_and_arc_diagram(pistes):
    c = Chart(pistes).network('de', 'vers', weight='km', layout='circular')
    assert c.graph['degree'].to_dict() == {'A': 2, 'B': 3, 'C': 3, 'D': 2}
    pos = {'A': (0, 0), 'B': (1, 0), 'C': (0, 1), 'D': (1, 1)}
    assert Chart(pistes).network('de', 'vers', layout=pos).graph['positions'] == pos
    a = Chart(pistes).arc_diagram('de', 'vers', weight='km', order=list('DCBA'))
    assert len(a.ax.patches) == len(pistes)
    with pytest.raises(ValueError):
        Chart(pistes).arc_diagram('de', 'vers', order=['A'])
