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

def test_font_missing_warns(df, monkeypatch):
    def introuvable(*a, **k):  # pas de réseau dans les tests : Google Fonts « échoue »
        raise RuntimeError('absente')
    monkeypatch.setattr('plutovizu.chart.load_font', introuvable)
    with pytest.warns(UserWarning, match='introuvable'):
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
    assert matplotlib.colors.to_hex(c.fig.get_facecolor()) == '#14123a'   # bleu nuit du logo
    assert matplotlib.colors.to_hex(c.ax.title.get_color()) == '#f7f6ff'

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

def test_chart_theme_is_local(df):
    set_theme('editorial')
    before = plt.rcParams['axes.facecolor']
    Chart(df, theme='neon').font(size=20)
    assert plt.rcParams['axes.facecolor'] == '#0f0a1e'     # actif pendant la chaîne
    c = Chart(df)                                            # graphique suivant : thème de fond
    assert matplotlib.colors.same_color(c.fig.get_facecolor(), plt.rcParams['figure.facecolor'])
    assert plt.rcParams['axes.facecolor'] == before and plt.rcParams['font.size'] != 20
    Chart(df, theme='neon').show()                           # show() rétablit aussi
    assert plt.rcParams['axes.facecolor'] == before
    Chart(df, theme='neon'); set_theme('pluto'); Chart(df)  # set_theme explicite gagne
    assert matplotlib.colors.to_hex(plt.rcParams['axes.facecolor']) != before

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


def test_map_points_geodataframe_boundary_and_labels_without_overlap():
    gpd = pytest.importorskip('geopandas')
    from shapely.geometry import Point, box
    sites = gpd.GeoDataFrame({'nom': [f'Site numéro {i}' for i in range(12)], 'type': ['a', 'b'] * 6},
                             geometry=[Point(-4 + i * 0.01, 5.3) for i in range(12)], crs='EPSG:4326')
    pays = gpd.GeoDataFrame(geometry=[box(-8, 4, -2, 10)], crs='EPSG:4326')
    c = Chart(sites).map_points(group='type', labels='nom', boundary=pays, basemap=False,
                                colors={'a': '#ff0000'})
    assert c.ax.get_xlim()[0] < -8 and c.ax.get_ylim()[1] > 10           # cadré sur le pays
    r = c.fig.canvas.get_renderer()
    boxes = [t.get_window_extent(r) for t in c.ax.texts]
    assert len(boxes) + len(c.hidden_labels) == 12 and c.hidden_labels   # serrés : certains masqués
    assert not any(a.overlaps(b) for i, a in enumerate(boxes) for b in boxes[i + 1:])
    assert matplotlib.colors.same_color(c.ax.collections[1].get_facecolor()[0][:3], (1, 0, 0))
    with pytest.raises(ValueError, match='GeoDataFrame'):
        Chart(pd.DataFrame({'x': [1]})).map_points()


def test_choropleth_draws_missing_zones():
    gpd = pytest.importorskip('geopandas')
    from shapely.geometry import box
    z = gpd.GeoDataFrame({'v': [1.0, np.nan, 3.0]}, geometry=[box(i, 0, i + 1, 1) for i in range(3)])
    c = Chart(z).choropleth('v')
    assert sum(len(col.get_paths()) for col in c.ax.collections
               if hasattr(col, 'get_paths')) >= 3                          # la zone NaN est dessinée


def test_pyramid():
    pop = pd.DataFrame({'age': ['0-4', '5-9', '10+'], 'h': [5, 4, 2], 'f': [6, 4, 1]})
    c = Chart(pop).pyramid('age', 'h', 'f', show_values=True)
    widths = [p.get_width() for p in c.ax.patches]
    assert widths == [-5, -4, -2, 6, 4, 1]
    assert c.ax.xaxis.get_major_formatter()(-4, 0) == '4'
    assert [t.get_text() for t in c.ax.get_yticklabels()] == ['0-4', '5-9', '10+']
    assert {'h', 'f'} <= {t.get_text() for t in c.ax.texts}


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


# ──────────────────────────────── Polices, taille, annotations, coins arrondis

def test_bundled_fonts_and_defaults(df):
    from matplotlib import font_manager as fm
    import plutovizu
    assert {f.name for f in fm.fontManager.ttflist} >= {'Fira Sans', 'Inter'}
    assert fm.findfont(fm.FontProperties(family=['sans-serif'])).endswith('FiraSans-Regular.ttf')
    assert plutovizu.load_font('Inter', 'semibold').get_file().endswith('Inter-SemiBold.ttf')
    assert plutovizu.load_font('Fira Sans', 'bold', italic=True).get_file().endswith('FiraSans-BoldItalic.ttf')
    assert Chart(df).fig.get_size_inches() * 2.54 == pytest.approx([16, 10])
    assert plutovizu.cm(20, 12) == pytest.approx((20 / 2.54, 12 / 2.54))


def test_annotation_and_arrow(df):
    from matplotlib.patches import FancyArrowPatch
    c = (Chart(df).line('x', 'a')
         .annotation('Pic', xy=(5, 0.5), xytext=(2, 1.5))
         .arrow((0.1, 0.9), (0.3, 0.6), coords='axes')
         .arrow((0.1, 0.1), (0.2, 0.2), coords='figure'))
    assert [t.get_text() for t in c.ax.texts] == ['Pic']
    assert c.ax.texts[0].get_ha() == 'right'          # flèche vers la droite : texte qui finit au départ
    assert sum(isinstance(p, FancyArrowPatch) for p in c.ax.patches) == 2
    assert any(isinstance(p, FancyArrowPatch) for p in c.fig.patches)
    with pytest.raises(ValueError):
        Chart(df).arrow((0, 0), (1, 1), coords='pixels')


def test_rounded_corners_bars_and_pie(df, tmp_path):
    from matplotlib.patches import PathPatch, Rectangle
    c = Chart(df).bar('cat', 'b').rounded(8)
    n = len([p for p in c.ax.patches if isinstance(p, Rectangle)])
    c.save(tmp_path / 'b.png')
    paths = [p for p in c.ax.patches if isinstance(p, PathPatch)]
    assert len(paths) == n and not any(isinstance(p, Rectangle) for p in c.ax.patches)
    assert any(code == 3 for p in paths for code in p.get_path().codes)   # courbes (CURVE3)
    # les coins restent dans la barre d'origine (pas de débordement)
    parts = pd.DataFrame({'v': [40, 30, 30]})
    pie = Chart(parts).pie('v').rounded(6)
    pie.save(tmp_path / 'p.png')
    assert all(isinstance(p, PathPatch) for p in pie.ax.patches)
    with pytest.raises(ValueError):
        Chart(df).rounded(-1)


def test_round_path_keeps_shape():
    from matplotlib.path import Path
    from plutovizu.chart import _round_path
    square = Path([(0, 0), (100, 0), (100, 50), (0, 50), (0, 0)], closed=True)
    r = _round_path(square, 10)
    v = r.vertices
    assert v[:, 0].min() >= -1e-9 and v[:, 0].max() <= 100 + 1e-9 and v[:, 1].max() <= 50 + 1e-9
    assert (r.codes == Path.CURVE3).sum() == 8                    # 4 coins × (contrôle + fin)
    assert not r.contains_point((0.5, 0.5))                       # coin effectivement arrondi
    assert r.contains_point((50, 25))
    assert _round_path(Path([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)], closed=True), 0.4) is not None


# ──────────────────────────────── Bump chart (bumplot)

@pytest.fixture
def classement():
    perms = [[1, 2, 3], [2, 1, 3], [3, 1, 2]]
    return pd.DataFrame([(s, e, r) for s, p in zip(['J1', 'J2', 'J3'], perms) for e, r in zip('ABC', p)],
                        columns=['journee', 'equipe', 'rang'])


def test_bump_ranks_keep_order(classement):
    from matplotlib.patches import PathPatch
    c = Chart(classement).bump('journee', 'rang', 'equipe')
    assert c.ranks.loc['J3'].to_dict() == {'A': 3, 'B': 1, 'C': 2}
    assert sum(isinstance(p, PathPatch) for p in c.ax.patches) == 3          # une courbe par équipe
    pts = {col.get_label(): col.get_offsets()[:, 1].tolist() for col in c.ax.collections}
    assert pts['A'] == [1, 2, 3]                                             # rang 1 conservé (pas inversé)
    assert c.ax.yaxis_inverted()
    assert [t.get_text() for t in c.ax.texts] == ['A', 'B', 'C']             # noms en bout de ligne


def test_bump_values_highlight_ordinal(classement):
    vals = classement.assign(score=lambda d: 10 - d['rang'])                 # plus fort = 1er
    c = Chart(vals).bump('journee', 'score', 'equipe', values=True, highlight=['B'], ordinal=True)
    assert c.ranks.loc['J2'].to_dict() == {'A': 2, 'B': 1, 'C': 3}
    assert [t.get_text() for t in c.ax.get_yticklabels()][:2] == ['1er', '2e']
    with pytest.raises(ValueError):
        Chart(vals).bump('journee', 'score', 'equipe', values=True, highlight=['Z'])
    with pytest.raises(ValueError):
        Chart(classement).bump('journee', 'rang', 'equipe', curve_force=2)
    with pytest.raises(ValueError):
        Chart(classement.iloc[:-1]).bump('journee', 'rang', 'equipe')       # période incomplète


def test_rounded_leaves_bump_curves(classement, tmp_path):
    c = Chart(classement).bump('journee', 'rang', 'equipe').rounded(6)
    before = [p.get_path().vertices.copy() for p in c.ax.patches]
    c.save(tmp_path / 'b.png')
    assert all(np.allclose(a, p.get_path().vertices) for a, p in zip(before, c.ax.patches))


def test_save_defaults_400dpi_and_format_from_extension(df, tmp_path):
    from PIL import Image
    import matplotlib as mpl
    assert mpl.rcParams['savefig.dpi'] == 400                     # thème : plt.savefig aussi
    Chart(df).line('x', 'a').save(tmp_path / 'g.png', bbox_inches=None)
    with Image.open(tmp_path / 'g.png') as im:
        assert im.size == (int(16 / 2.54 * 400), int(10 / 2.54 * 400))   # 16 × 10 cm à 400 dpi
    Chart(df).line('x', 'a').save(tmp_path / 'g.pdf')
    assert (tmp_path / 'g.pdf').read_bytes()[:4] == b'%PDF'


def test_default_rounding_8_points(df, tmp_path):
    from matplotlib.patches import PathPatch, Rectangle
    assert Chart.default_round == 8
    c = Chart(df).bar('cat', 'b')
    c.save(tmp_path / 'a.png')                                    # arrondi par défaut
    assert all(isinstance(p, PathPatch) for p in c.ax.patches)
    sharp = Chart(df).bar('cat', 'b').rounded(0)                   # coins vifs pour ce graphique
    sharp.save(tmp_path / 'b.png')
    assert all(isinstance(p, Rectangle) for p in sharp.ax.patches)
    prof = pd.DataFrame({'haut': [0, 20], 'bas': [20, 50], 'horizon': ['A', 'B']})
    sp = Chart(prof).soil_profile('haut', 'bas', 'horizon')        # formes jointives : pas d'arrondi
    sp.save(tmp_path / 'c.png')
    assert all(isinstance(p, Rectangle) for p in sp.ax.patches)
    try:
        Chart.default_round = 0                                    # désactivé pour la session
        off = Chart(df).bar('cat', 'b')
        off.save(tmp_path / 'd.png')
        assert all(isinstance(p, Rectangle) for p in off.ax.patches)
    finally:
        Chart.default_round = 8


# ──────────────────────────────── Climagramme d'Emberger, pypalettes

@pytest.fixture
def stations_emb():
    return pd.DataFrame({'station': ['A', 'B', 'C'], 'P': [650, 290, 45],
                         'M': [30.5, 34.5, 41.0], 'm': [8.0, -0.5, 5.5]})


def test_emberger_q2_and_classes(stations_emb):
    c = Chart(stations_emb).emberger('P', 'M', 'm', station='station')
    t = c.emberger_table.set_index('station')
    q2 = 2000 * 650 / ((30.5 + 273.15) ** 2 - (8 + 273.15) ** 2)
    assert t.at['A', 'Q2'] == pytest.approx(q2, abs=0.05)
    assert t['etage'].tolist() == ['Subhumide', 'Aride', 'Saharien']
    assert t['variante'].tolist() == ['hiver chaud', 'hiver froid', 'hiver tempéré']
    assert t.at['B', 'bioclimat'] == 'Aride à hiver froid'
    s = Chart(stations_emb).emberger('P', 'M', 'm', formula='stewart').emberger_table
    assert s['Q2'].iloc[0] == pytest.approx(3.43 * 650 / (30.5 - 8), abs=0.05)
    # limite oblique : Q2 = 0 + 20·m → A (m = 8, seuil 160) passe sous la 1re limite
    o = Chart(stations_emb).emberger('P', 'M', 'm', limits=[(0, 20), 200, 300, 400, 500]).emberger_table
    assert o['etage'].iloc[0] == 'Saharien'
    with pytest.raises(ValueError):
        Chart(stations_emb).emberger('P', 'M', 'm', limits=[10, 20])
    with pytest.raises(ValueError):
        Chart(stations_emb).emberger('P', 'm', 'M')                       # M < m


def test_emberger_monthly_aggregation():
    months = pd.DataFrame({'st': ['X'] * 12, 'p': [10] * 12,
                           'tx': np.linspace(15, 32, 12), 'tn': np.linspace(2, 18, 12)})
    t = Chart(months).emberger('p', 'tx', 'tn', station='st', monthly=True).emberger_table
    assert t[['P', 'M', 'm']].iloc[0].tolist() == pytest.approx([120, 32, 2])


def test_pypalettes_integration(df):
    import matplotlib
    from plutovizu import pypalette
    c = Chart(df).set_colors('Cross')
    assert c.current_palette[0] == '#c969a1'
    pals = c.get_available_palettes()
    assert 'Cross' in pals['pypalettes_qualitative'] and len(pals['pypalettes_sequential']) > 50
    c.preview_palette('Hiroshige')
    cmap = pypalette('Hiroshige', continuous=True)
    assert 'Hiroshige' in matplotlib.colormaps and cmap.N > 9
    with pytest.raises(ValueError):
        Chart(df).set_colors('PaletteQuiNExistePas')
    with pytest.raises(ValueError):
        pypalette('PaletteQuiNExistePas')


def test_ecoclimatogram_zones():
    d = pd.DataFrame({'mois': ['J', 'F', 'M', 'A'], 'hr': [70, 50, 30, 85], 't': [26, 22, 33, 20]})
    c = Chart(d).ecoclimatogram('hr', 't', month='mois', optimum=(60, 80, 24, 30),
                                tolerance=(40, 90, 18, 34))
    assert c.ecoclimate['zone'].tolist() == ['optimale', 'favorable', 'défavorable', 'favorable']
    line = c.ax.lines[0]
    assert line.get_xdata()[-1] == 70 and len(line.get_xdata()) == 5            # boucle fermée
    with pytest.raises(ValueError):
        Chart(d).ecoclimatogram('hr', 't', optimum=(80, 60, 24, 30))             # min > max


def test_legend_handles_rounded():
    from matplotlib.patches import FancyBboxPatch, Rectangle
    df = pd.DataFrame({'r': ['A', 'B'], 'x': [3, 5], 'y': [4, 1]})
    c = Chart(df).bar('r', ['x', 'y'])
    c._apply_rounding()
    assert all(isinstance(h, FancyBboxPatch) for h in c.ax.get_legend().legend_handles)
    c0 = Chart(df).bar('r', ['x', 'y']).rounded(0)
    c0._apply_rounding()
    assert all(type(h) is Rectangle for h in c0.ax.get_legend().legend_handles)


# ── Graphiques complexes

def test_pareto_vital_few_and_cumul():
    df = pd.DataFrame({'cause': list('ABCDE'), 'n': [50, 25, 15, 6, 4]})
    c = Chart(df).pareto('cause', 'n')
    t = c.pareto_table
    assert t['cumul'].tolist() == [50, 75, 90, 96, 100]
    assert t['classe'].tolist() == ['essentielle'] * 3 + ['secondaire'] * 2  # A+B+C franchit 80 %
    counted = Chart(pd.DataFrame({'e': list('aaab')})).pareto('e').pareto_table
    assert counted['valeur'].tolist() == [3, 1]


def test_control_chart_limits_flag_outliers():
    v = [10.0, 10.2, 9.9, 10.1, 9.8, 10.0] * 3 + [13.0, 10.1]
    c = Chart(pd.DataFrame({'lot': range(20), 'y': v})).control_chart('lot', 'y')
    lim = c.control_limits
    mr = np.abs(np.diff(v)).mean() / 1.128
    assert lim['lsc'] == pytest.approx(np.mean(v) + 3 * mr)
    assert c.control_table['hors_limites'].tolist() == [False] * 18 + [True, False]


def test_marimekko_widths_and_bland_altman():
    df = pd.DataFrame({'r': ['N', 'N', 'S'], 'c': ['a', 'b', 'a'], 'v': [30, 10, 60]})
    c = Chart(df).marimekko('r', 'c', 'v')
    assert c.marimekko_table.loc['S', 'a'] == 60
    m = pd.DataFrame({'a': [10, 12, 14, 16.0], 'b': [9, 11.5, 13, 15.5]})
    ag = Chart(m).bland_altman('a', 'b').agreement
    assert ag['biais'] == pytest.approx(0.75)
    assert ag['n'] == 4


def test_raincloud_and_seaborn_grids():
    df = pd.DataFrame({'g': ['a'] * 10 + ['b'] * 10, 'v': np.arange(20.0), 'x': np.tile(np.arange(10.0), 2)})
    Chart(df).raincloud('g', 'v')
    for c in (Chart(df).relplot('x', 'v', col='g'), Chart(df).displot('v', col='g'),
              Chart(df).catplot('g', 'v', kind='box'), Chart(df).lmplot('x', 'v', hue='g')):
        assert c.sns_grid is not None


def test_ombrothermic_and_gantt_follow_palette():
    import matplotlib.colors as mc
    pal = ['#2a9d8f', '#e76f51', '#264653']
    d = pd.DataFrame({'m': list('JFMAMJJASOND'), 't': [8, 9, 12, 15, 19, 24, 27, 27, 23, 18, 12, 9],
                      'p': [70, 60, 55, 45, 30, 12, 4, 8, 35, 70, 85, 80]})
    c = Chart(d).set_colors(pal).ombrothermic('m', 't', 'p')
    assert mc.to_hex(c.ax.lines[0].get_color()) == pal[1]                 # température
    assert mc.to_hex(c.ax2.patches[0].get_facecolor()) == pal[0]          # précipitations
    assert len(c.ax2.patches) == 12                                       # mois répétés : 12 barres
    dry = [col for col in c.ax2.collections if col.get_hatch()]
    assert len(dry) == 5 and mc.to_hex(dry[0].get_edgecolor()[0]) == pal[1]   # mai → sept. hachurés
    c = Chart(d).ombrothermic('m', 't', 'p').set_colors(pal)                # set_colors APRÈS
    assert mc.to_hex(c.ax.lines[0].get_color()) == pal[1]
    assert mc.to_hex(c.ax2.patches[0].get_facecolor()) == pal[0]
    assert c.ax2.patches[0].get_facecolor()[3] == pytest.approx(0.6)      # transparence gardée
    dry = [col for col in c.ax2.collections if col.get_hatch()]
    assert mc.to_hex(dry[0].get_edgecolor()[0]) == pal[1]
    assert mc.to_hex(c.ax.yaxis.label.get_color()) == pal[1]
    g = pd.DataFrame({'t': ['a', 'b', 'c'], 'd': ['2024-01-01'] * 3, 'f': ['2024-01-05'] * 3,
                      'e': ['x', 'y', 'x']})
    c = Chart(g).set_colors(pal).gantt('t', 'd', 'f', group='e')
    leg = c.ax.legend()
    assert [mc.to_hex(h.get_facecolor()) for h in leg.legend_handles] == pal[:2]
    assert [mc.to_hex(p.get_facecolor()) for p in c.ax.patches] == [pal[0], pal[0], pal[1]]


def test_water_balance_legend_vertical():
    d = pd.DataFrame({'m': list('abcdef'), 'p': [0, 50, 200, 150, 10, 0], 'e': [100] * 6})
    leg = Chart(d).water_balance('m', 'p', 'e').ax.get_legend()
    assert leg._ncols == 1
    assert [t.get_text() for t in leg.texts] == ['Précipitations (P)', 'Évapotranspiration (ETP)',
                                                  'Excédent (P > ETP)', 'Déficit (P < ETP)']


def test_translate_builtin_and_back():
    d = pd.DataFrame({'m': list('abcdef'), 'p': [0, 50, 200, 150, 10, 0], 'e': [100] * 6})
    c = Chart(d).water_balance('m', 'p', 'e').title('Mon titre').translate('en')
    texts = lambda: [t.get_text() for t in c.ax.get_legend().texts]   # noqa: E731
    assert texts() == ['Precipitation (P)', 'Evapotranspiration (PET)', 'Surplus (P > PET)',
                       'Deficit (P < PET)']
    assert c.ax.get_title(loc='left') == 'Mon titre'                     # vos textes : inchangés en builtin
    c.translate('es')
    assert texts()[0] == 'Precipitación (P)'
    c.translate('ar')
    assert texts()[2].startswith('الفائض') and 'DejaVu Sans' in c.ax.get_legend().texts[0].get_fontfamily()
    c.translate('fr')
    assert texts()[0] == 'Précipitations (P)'
    with pytest.raises(ValueError):
        c.translate('de')                                       # pas de dictionnaire intégré
    # Libellés ambigus traduits seulement pour leur graphique ; graduations comprises
    days = pd.DataFrame({'d': pd.date_range('2024-01-01', periods=30), 'v': range(30)})
    c = Chart(days).calendar_heatmap('d', 'v').translate('en')
    c.fig.draw_without_rendering()
    assert [t.get_text() for t in c.ax.get_yticklabels()][:2] == ['Mon', 'Tue']
    c = Chart(pd.DataFrame({'x': ['Mar', 'Avr'], 'y': [1, 2]})).bar('x', 'y').translate('en')
    c.fig.draw_without_rendering()
    assert c.ax.get_xticklabels()[0].get_text() == 'Mar'       # mars, pas mardi


def test_translate_online(monkeypatch):
    from plutovizu import translations
    calls = []
    monkeypatch.setattr(translations, 'online',
                        lambda s, lang, src: calls.append(s) or f'<{lang}:{s}>')
    d = pd.DataFrame({'m': list('abcdef'), 'p': [0, 50, 200, 150, 10, 0], 'e': [100] * 6})
    c = Chart(d).water_balance('m', 'p', 'e').title('Bilan hydrique').translate('de', from_='online')
    assert c.ax.get_title(loc='left') == '<de:Bilan hydrique>'
    assert 'Bilan hydrique' in calls and '200' not in calls     # nombres : pas d'appel
    c = Chart(d).water_balance('m', 'p', 'e').translate('en', from_='online')
    assert c.ax.get_legend().texts[0].get_text() == 'Precipitation (P)'   # dictionnaire d'abord


def test_seaborn_grids_follow_figsize_and_dpi():
    df = pd.DataFrame({'g': ['a'] * 10 + ['b'] * 10, 'v': np.arange(20.0), 'x': np.tile(np.arange(10.0), 2)})
    c = Chart(df, figsize=(9, 3), dpi=150).relplot('x', 'v', col='g', height=5, aspect=2)
    assert tuple(c.fig.get_size_inches()) == (9, 3) and c.fig.dpi == 150     # figsize > height/aspect
    assert tuple(Chart(df).catplot('g', 'v', figsize=(5, 4)).fig.get_size_inches()) == (5, 4)
    assert tuple(Chart(df, figsize=(6, 6)).pairplot(['v', 'x']).fig.get_size_inches()) == (6, 6)
    assert tuple(Chart(df).displot('v', height=2, aspect=1).fig.get_size_inches()) == (2, 2)
    assert tuple(Chart(df, figsize=(7, 5)).clustermap(['v', 'x']).fig.get_size_inches()) == (7, 5)
    with tempfile.TemporaryDirectory() as d:
        Chart(df, figsize=(2, 1), dpi=50).line('x', 'v').save(os.path.join(d, 'a.png'), bbox_inches=None)
        assert plt.imread(os.path.join(d, 'a.png')).shape[:2] == (50, 100)   # save() reprend dpi


def test_legend_moved_outside_axes():
    df = pd.DataFrame({'x': np.arange(10), 'a': np.arange(10.0), 'b': np.arange(10.0)[::-1]})
    for tight in (False, True):
        c = Chart(df).line('x', ['a', 'b'])
        c.ax.legend(loc='center')                       # en plein milieu du graphique
        if tight:
            c.tight_layout()
        c._legends_outside()
        r = c.fig._get_renderer()
        leg = c.ax.get_legend().get_window_extent(r).padded(-1)
        assert not any(leg.overlaps(a.get_window_extent(r)) for a in c.fig.axes)
        assert leg.x1 <= c.fig.bbox.x1                    # et visible en entier


# ── Diagrammes et infographie

def _links_df():
    return pd.DataFrame([('A', 'B', 30), ('A', 'C', 20), ('B', 'D', 25), ('C', 'D', 15),
                         ('B', 'E', 5)], columns=['s', 't', 'v'])


def test_sankey_multilevel_and_legacy():
    c = Chart(_links_df()).sankey('s', 't', 'v')
    assert len(c.sankey_table) == 5 and c.sankey_table['v'].sum() == 95
    assert len(c.ax.patches) == 5 + 5                       # 5 rubans + 5 nœuds
    old = pd.DataFrame({'flux': [100, -60, -40], 'etiquette': ['in', 'a', 'b']})
    Chart(old).sankey('flux', 'etiquette')                  # ancienne forme matplotlib


def test_chord_matrix():
    c = Chart(_links_df()).chord('s', 't', 'v')
    assert c.chord_table.loc['A', 'B'] == 30 and c.chord_table.values.sum() == 95


def test_histogram_log_hue_smooth():
    d = pd.DataFrame({'p': [1, 10, 10, 100, 1000, 10000] * 5, 'g': list('aabbcc') * 5})
    c = Chart(d).histogram('p', bins=4, log=True, hue='g', order=['c', 'b', 'a'], smooth=True)
    assert c.ax.get_xscale() == 'log'
    tops = [p.get_y() + p.get_height() for p in c.ax.patches[-4:]]
    assert sum(tops) == 30 and len(c.ax.lines) == 1              # empilées : total en haut
    assert c.ax.get_legend_handles_labels()[1] == ['c', 'b', 'a']


def test_line_direct_labels_and_tick_format():
    d = pd.DataFrame({'x': range(10), 'a': range(10), 'b': [9.0] * 9 + [9.2]})
    c = Chart(d).line('x', ['a', 'b'], direct_labels=True)
    assert {t.get_text() for t in c.ax.texts} == {'a', 'b'}
    c = Chart(d).line('x', ['a', 'b'], direct_labels='end', value_format='{:.1f} %')
    ends = {t.get_text(): t.xy[1] for t in c.ax.texts}
    assert set(ends) == {'9.0 %', '9.2 %'} and abs(ends['9.2 %'] - ends['9.0 %']) > 0.2  # écartées
    c.tick_format(y='{:.0f} %')
    assert c.ax.yaxis.get_major_formatter()(5, 0) == '5 %'
    c = Chart(d).line('x', 'b', direct_labels='end', value_format=lambda v: f'{v:.1f}'.replace('.', ','))
    assert c.ax.texts[-1].get_text() == '9,2'


def test_equal_aspect_charts_fill_figsize():
    """Chord, Venn, waffle : l'axe remplit la figure au lieu d'un carré (figsize respecté)."""
    df = pd.DataFrame({'a': [True, False], 'b': [True, True], 'g': ['x', 'y'], 'v': [30, 12]})
    for c in (Chart(_links_df(), figsize=(8, 4)).chord('s', 't', 'v'),
              Chart(df, figsize=(8, 4)).venn(['a', 'b']),
              Chart(df, figsize=(8, 4)).waffle('g', 'v')):
        c.fig.canvas.draw()
        assert c.ax.get_position().width > 0.7          # carré : ~0.39


def test_venn_counts_bool_and_items():
    df = pd.DataFrame({'a': [True, True, False, True], 'b': [True, False, True, False],
                       'c': [False, False, False, True]})
    t = Chart(df).venn(['a', 'b', 'c']).venn_table.set_index('zone')['n']
    assert t['a'] == 1 and t['a ∩ b'] == 1 and t['b'] == 1 and t['a ∩ c'] == 1 and t.sum() == 4
    items = pd.DataFrame({'x': ['p', 'q', 'r'], 'y': ['q', 'r', None]})
    t = Chart(items).venn(['x', 'y']).venn_table.set_index('zone')['n']
    assert (t['x'], t['y'], t['x ∩ y']) == (1, 0, 2)
    with pytest.raises(ValueError):
        Chart(df).venn(['a'])


def test_flowchart_layers_back_edges_and_labels():
    df = pd.DataFrame([('Début', 'Test ?', None), ('Test ?', 'Fin', 'oui'),
                       ('Test ?', 'Début', 'non')], columns=['a', 'b', 'l'])
    c = Chart(df).flowchart('a', 'b', label='l')
    lay = c.flow_layout.set_index('noeud')['niveau']
    assert (lay['Début'], lay['Test ?'], lay['Fin']) == (0, 1, 2)   # la boucle ne compte pas
    c.fig.draw_without_rendering()
    texts = [t.get_text() for t in c.ax.texts]
    assert {'oui', 'non'} <= set(texts)
    diamond = next(t for t in c.ax.texts if t.get_text() == 'Test ?')
    assert diamond.get_bbox_patch() is not None
    lr = Chart(df).flowchart('a', 'b', direction='LR').flow_layout.set_index('noeud')
    assert lr.loc['Fin', 'x'] == 2
    with pytest.raises(ValueError):
        Chart(df).flowchart('a', 'b', direction='XY')


def test_sequence_diagram_draws_messages():
    df = pd.DataFrame([('P', 'C', 'livre', False), ('C', 'C', 'calcule', False),
                       ('C', 'P', 'paie', True)], columns=['s', 'r', 'm', 'd'])
    c = Chart(df).sequence_diagram('s', 'r', 'm', dashed='d', numbered=True)
    assert {'1. livre', '2. calcule', '3. paie'} <= {t.get_text() for t in c.ax.texts}


def test_waffle_single_and_by_x():
    df = pd.DataFrame({'g': ['a', 'b'], 'v': [30, 12]})
    c = Chart(df).waffle('g', 'v')
    assert c.waffle_table['cases'].tolist() == [30, 12]
    assert sum(len(col.get_paths()) for col in c.ax.collections) == 42
    df = pd.DataFrame({'annee': [1, 1, 2, 2], 'g': ['a', 'b', 'a', 'b'], 'v': [50, 20, 70, 0]})
    c = Chart(df).waffle('g', 'v', x='annee', unit=5, columns=5, direct_labels=True)
    t = c.waffle_table.set_index(['annee', 'g'])['cases']
    assert (t[(1, 'a')], t[(1, 'b')], t[(2, 'a')], t[(2, 'b')]) == (10, 4, 14, 0)
    assert '70' in {x.get_text() for x in c.ax.texts}                # total de l'année 2


def test_stem_colormap_and_hlines():
    import matplotlib.colors as mc
    df = pd.DataFrame({'x': range(5), 'y': [-0.4, -0.1, 0.2, 0.5, 0.9]})
    c = Chart(df).stem('x', 'y', cmap=['#000080', '#ff0000'], hlines=[-0.4, 0.4, 0.8])
    pts = c.ax.collections[-1].get_facecolors()
    assert mc.to_hex(pts[0]) == '#000080' and mc.to_hex(pts[-1]) == '#ff0000'
    assert list(c.ax.get_yticks()) == [-0.4, 0.0, 0.4, 0.8]


def test_infographic_helpers():
    import matplotlib.colors as mc
    df = pd.DataFrame({'x': list('abc'), 'y': [1, 2, 3]})
    c = (Chart(df).bar('x', 'y').title('Titre').subtitle('Sous-titre').caption('Source')
         .background('#f6eee3').labels('', ''))
    assert c.ax.get_xlabel() == '' and mc.to_hex(c.fig.get_facecolor()) == '#f6eee3'
    c.fig.draw_without_rendering()
    r = c.fig.canvas.get_renderer()
    title = c.ax.title if c.ax.get_title() else c.ax._left_title
    sub = next(t for t in c.ax.texts if t.get_text() == 'Sous-titre')
    cap = next(t for t in c.ax.texts if t.get_text() == 'Source')
    assert title.get_window_extent(r).y0 >= sub.get_window_extent(r).y1 - 1   # titre au-dessus
    assert cap.get_window_extent(r).y1 <= c.ax.xaxis.get_tightbbox(r).y0 + 1  # source dessous
    with tempfile.TemporaryDirectory() as d:
        c.save(os.path.join(d, 'f.png'), dpi=20, bbox_inches=None)
        px = plt.imread(os.path.join(d, 'f.png'))[1, 1, :3]
        assert mc.to_hex(px) == '#f6eee3'                                     # fond gardé
    c = Chart(pd.DataFrame({'x': [1, 2], 'a': [1, 2], 'b': [2, 1]})).line('x', ['a', 'b'])
    c.set_legend(location='below')
    c.fig.draw_without_rendering()
    assert c.ax.get_legend().get_window_extent().y1 <= c.ax.get_window_extent().y0


def test_font_applies_to_existing_texts():
    c = Chart(pd.DataFrame({'x': [1, 2], 'y': [1, 2]})).line('x', 'y').title('T')
    c.font('Inter')
    assert 'Inter' in c.ax.xaxis.label.get_fontfamily()
    c.fig.draw_without_rendering()
    assert 'Inter' in c.ax.get_xticklabels()[0].get_fontfamily()


def test_choropleth_histogram():
    gpd = pytest.importorskip('geopandas')
    from shapely.geometry import box
    g = gpd.GeoDataFrame({'v': [1.0, 2.0, 3.0, 4.0], 'n': ['a', '', None, 'd']},
                         geometry=[box(i, 0, i + 1, 1) for i in range(4)])
    c = Chart(g).choropleth('v', labels='n', histogram=True, bins=4)
    assert len(c.ax_hist.patches) == 4 and len(c.fig.axes) == 1            # encart, pas de barre de couleur
    assert {t.get_text() for t in c.ax.texts} == {'a', 'd'}                 # vides ignorées
