"""Catalogue des graphiques de plutovizu → vignettes + cartes de la galerie.

    uv run python docs/_vignettes.py              # toutes les vignettes
    uv run python docs/_vignettes.py pyramid      # seulement celles nommées (galerie.yml complet)

Génère docs/images/vignettes/<nom>.png (thème editorial) et docs/galerie.yml, lu par
docs/gallery.qmd (listing Quarto en grille). À relancer quand un graphique est ajouté.
"""
import os
import pathlib
import warnings

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from plutovizu import Chart, theme_context  # noqa: E402

warnings.filterwarnings('ignore')

DOCS = pathlib.Path(__file__).parent
os.chdir(DOCS / 'charts')                       # _donnees.py et chemins relatifs
exec(open('_donnees.py', encoding='utf-8').read())

# Petites données propres aux vignettes
serie = pd.DataFrame({'mois': range(1, 13),
                      'riz': [12, 15, 18, 22, 28, 35, 40, 38, 30, 25, 20, 16],
                      'mais': [10, 11, 12, 14, 16, 18, 20, 19, 17, 15, 13, 12],
                      'mil': [2, 4, 6, 8, 12, 17, 20, 19, 13, 10, 7, 4]})
rangs = pd.DataFrame([(a, v, r) for a in range(2019, 2025)
                      for v, r in zip(['Riz', 'Maïs', 'Mil', 'Sorgho'],
                                      np.roll([1, 2, 3, 4], a % 3))],
                     columns=['annee', 'culture', 'rang'])
taches = pd.DataFrame({'tache': ['Labour', 'Semis', 'Entretien', 'Récolte'],
                       'debut': pd.to_datetime(['2024-06-01', '2024-06-20', '2024-07-10', '2024-10-01']),
                       'fin': pd.to_datetime(['2024-06-20', '2024-07-05', '2024-09-20', '2024-10-30'])})
bilan = pd.DataFrame({'poste': ['Recettes', 'Semences', 'Engrais', 'Main-d’œuvre', 'Aides'],
                      'montant': [900, -150, -220, -180, 120]})
flux = pd.DataFrame({'flux': [100, -60, -25, -15], 'etiquette': ['Récolte', 'Vente', 'Stock', 'Pertes'],
                     'orientation': [0, 0, 1, -1]})
avant_apres = pd.DataFrame({'variete': ['Sahel-108', 'IR-841', 'Nerica-4'],
                            'avant': [3.1, 3.8, 4.2], 'apres': [3.9, 4.5, 5.6]})
climat = pd.DataFrame({'mois': ['J', 'F', 'M', 'A', 'M ', 'J ', 'J  ', 'A ', 'S', 'O', 'N', 'D'],
                       'temp': [23.1, 23.5, 24.2, 24.8, 26.5, 28.2, 28.5, 28.3, 28.0, 27.6, 25.6, 23.7],
                       'precip': [3, 1, 0, 0, 0, 16, 88, 235, 155, 60, 5, 5]})
moyennes = essai.groupby(['variete', 'dose'], as_index=False)['rendement'].agg(
    moyenne='mean', ecart_type='std')
traits = ['hauteur', 'jours_floraison', 'talles', 'pmg', 'rendement']
t = np.linspace(0, 6 * np.pi, 200)
vol = pd.DataFrame({'x': np.cos(t) * t, 'y': np.sin(t) * t, 'z': 20 + 2 * t})

C = lambda d, w=4, h=3: Chart(d, figsize=(w, h))  # noqa: E731

import geopandas as gpd  # noqa: E402
from shapely.geometry import box  # noqa: E402

_r = np.random.default_rng(1)
parcelles = gpd.GeoDataFrame({'rendement': 3 + _r.normal(0, .6, 20)},
                             geometry=[box(x * 110, y * 80, x * 110 + 100, y * 80 + 70)
                                       for y in range(4) for x in range(5)], crs='EPSG:32628')
_dakar = villes.set_index('ville').loc['Dakar']
flux_vignette = villes[villes['ville'].isin(['Saint-Louis', 'Kaolack', 'Tambacounda', 'Kolda', 'Ziguinchor'])]     .assign(lon2=_dakar['lon'], lat2=_dakar['lat'], t=[400, 1200, 300, 650, 500])

EVO, DIS, COM, REL = 'Évolution', 'Distribution', 'Comparaison', 'Relation'
MUL, CPO, CHA, D3 = 'Multivarié', 'Composition', 'Champs', '3D'
SCI, QUA, CPX, CAR, ECO = 'Climat & sols', 'Qualitatif', 'Graphiques complexes', 'Map', 'Écologie'
GRA, DIA = 'Graphes & réseaux', 'Diagrammes'
PAGE = {EVO: 'evolution', DIS: 'distribution', COM: 'comparaison', REL: 'relation',
        MUL: 'multivarie', CPO: 'composition', CHA: 'champs', D3: '3d', SCI: 'sciences',
        QUA: 'qualitatif', CPX: 'complexes', CAR: 'map', ECO: 'ecologie', GRA: 'graphes',
        DIA: 'diagrammes'}

# (méthode, titre de la carte, catégorie, usage, vignette)
CATALOGUE = [
    ('line', 'Courbe', EVO, "Évolution d'une ou plusieurs séries",
     lambda: C(serie).line('mois', ['riz', 'mais', 'mil'], markersize=3)),
    ('area', 'Aire empilée', EVO, "Évolution de parts d'un total",
     lambda: C(serie).area('mois', ['riz', 'mais', 'mil'], stacked=True)),
    ('streamgraph', 'Flux temporel', EVO, 'Composition qui évolue dans le temps',
     lambda: C(serie).streamgraph('mois', ['riz', 'mais', 'mil'])),
    ('step', 'Escalier', EVO, 'Valeur constante entre deux mesures',
     lambda: C(meteo.assign(c=meteo['pluie'].cumsum())).step('jour', 'c', where='post', fill=True)),
    ('stem', 'Bâtons', EVO, 'Événements ponctuels (pluies…)',
     lambda: C(meteo[meteo['pluie'] > 0]).stem('jour', 'pluie')),
    ('band', 'Bande', EVO, 'Plage min–max ou intervalle autour d’une courbe',
     lambda: C(meteo).band('jour', 'tmin', 'tmax', y='tmoy')),
    ('bump', 'Classement', EVO, 'Évolution des rangs',
     lambda: C(rangs).bump('annee', 'rang', 'culture', highlight=['Riz', 'Mil'])),
    ('timeline', 'Frise chronologique', EVO, 'Jalons et calendrier cultural',
     lambda: C(calendrier, 5, 3).timeline('date', 'etape')),
    ('eventplot', 'Événements', EVO, 'Dates d’observation par groupe',
     lambda: C(floraison).eventplot('date', group='variete')),
    ('calendar_heatmap', 'Calendrier', EVO, 'Valeurs jour par jour',
     lambda: C(meteo, 5, 2.5).calendar_heatmap('jour', 'pluie')),
    ('gantt', 'Gantt', EVO, 'Planning de tâches',
     lambda: C(taches).gantt('tache', 'debut', 'fin')),

    ('histogram', 'Histogramme', DIS, "Forme d'une distribution",
     lambda: C(essai).histogram('rendement', bins=15)),
    ('bihistogram', 'Histogrammes dos à dos', DIS, 'Deux groupes, pyramide des âges',
     lambda: C(population).bihistogram('age', 'sexe', bins=range(0, 95, 5), horizontal=True)),
    ('boxplot', 'Boîte à moustaches', DIS, 'Médiane, quartiles, extrêmes',
     lambda: C(caracteres).boxplot(['talles', 'pmg'])),
    ('violin', 'Violon', DIS, 'Densité complète par groupe',
     lambda: C(caracteres).violin(['talles', 'pmg'])),
    ('ridgeline', 'Crêtes', DIS, 'Beaucoup de distributions superposées',
     lambda: C(essai).ridgeline('rendement', 'dose')),
    ('kde', 'Densité', DIS, 'Distributions continues lissées',
     lambda: C(essai).kde('rendement', hue='variete', legend=False)),
    ('ecdf', 'Répartition empirique', DIS, 'Distribution sans choix de classes',
     lambda: C(essai).ecdf('rendement', hue='variete', legend=False)),
    ('strip', 'Points par catégorie', DIS, 'Toutes les observations (strip, swarm)',
     lambda: C(essai).swarm('variete', 'rendement', size=3)),
    ('boxen', 'Boîte étendue', DIS, 'Grands échantillons et queues',
     lambda: C(essai).boxen('dose', 'rendement')),
    ('qqplot', 'Quantile-quantile', DIS, 'Vérifier la normalité',
     lambda: C(essai.assign(r=essai['rendement'] - essai.groupby(['variete', 'dose'])['rendement']
                            .transform('mean'))).qqplot('r')),
    ('hist2d', 'Histogramme 2D', DIS, 'Densité d’un grand nuage',
     lambda: C(pd.DataFrame({'a': rng.normal(size=3000), 'b': rng.normal(size=3000)})
               .assign(b=lambda d: d['a'] * .6 + d['b'] * .8)).hist2d('a', 'b', bins=25, colorbar=False)),

    ('bar', 'Barres', COM, 'Simples, groupées ou empilées',
     lambda: C(serie.head(5)).bar('mois', ['riz', 'mais', 'mil'], stacked=True)),
    ('mean_bar', 'Moyennes ± erreur', COM, 'Traitements, lettres de Tukey',
     lambda: C(essai).mean_bar('variete', 'rendement', letters=True)),
    ('errorbar', "Barres d'erreur", COM, 'Points ± erreur, par groupe',
     lambda: C(moyennes).errorbar('dose', 'moyenne', yerr='ecart_type', group='variete', line=True)),
    ('interaction', 'Interaction', COM, 'Deux facteurs (ANOVA)',
     lambda: C(essai).interaction('dose', 'variete', 'rendement')),
    ('pointplot', 'Moyennes reliées', COM, 'Moyennes et IC par catégorie',
     lambda: C(essai).pointplot('dose', 'rendement', hue='variete', legend=False)),
    ('countplot', 'Effectifs', COM, 'Nombre d’observations par catégorie',
     lambda: C(sols).countplot('zone')),
    ('lollipop', 'Lollipop', COM, 'Barres allégées, groupées',
     lambda: C(moyennes[moyennes['dose'] > 0]).lollipop(
         'dose', 'moyenne', 'variete',
         colors=dict(zip(moyennes['variete'].unique(), C(serie)._palette())))),
    ('dumbbell', 'Haltère', COM, 'Avant / après',
     lambda: C(avant_apres).dumbbell('variete', 'avant', 'apres')),
    ('pyramid', 'Pyramide des âges', COM, 'Deux séries dos à dos par classe',
     lambda: C(pd.DataFrame({'age': ['0-19', '20-39', '40-59', '60+'], 'h': [9, 7, 4, 2],
                             'f': [8.5, 7, 4.5, 2.5]}), 3.4, 3).pyramid('age', 'h', 'f')),
    ('radar', 'Radar', COM, 'Profils multicritères',
     lambda: C(caracteres.groupby('cycle', as_index=False)[traits].mean()
               .assign(**{c: lambda d, c=c: d[c] / d[c].max() for c in traits}), 3.4, 3.4)
     .radar(traits, group='cycle')),
    ('parallel_coordinates', 'Coordonnées parallèles', COM, 'Individus sur plusieurs variables',
     lambda: C(caracteres).parallel_coordinates(traits, group='cycle')),
    ('polar', 'Polaire', COM, 'Rose des vents, cycles',
     lambda: C(rose, 3.4, 3.4).polar('direction', 'frequence', group='vitesse', compass=True)),

    ('scatter', 'Nuage de points', REL, 'Relation entre deux variables',
     lambda: C(caracteres).scatter('jours_floraison', 'hauteur')),
    ('regression', 'Régression', REL, 'Ajustement, équation, R²',
     lambda: C(essai).regression('dose', 'rendement', order=2, show_equation=False)),
    ('residplot', 'Résidus', REL, 'Vérifier un ajustement',
     lambda: C(essai).residplot('dose', 'rendement')),
    ('confidence_ellipse', 'Ellipses de confiance', REL, 'Dispersion de groupes',
     lambda: C(caracteres).confidence_ellipse('jours_floraison', 'hauteur', group='cycle')),
    ('hexbin', 'Pavage hexagonal', REL, 'Nuage très dense',
     lambda: C(pd.DataFrame({'a': rng.normal(size=3000), 'b': rng.normal(size=3000)})).hexbin('a', 'b', gridsize=18)),
    ('heatmap', 'Corrélations', REL, 'Matrice de corrélation',
     lambda: C(caracteres[traits], 4, 3.2).heatmap(annot=False)),
    ('pivot_heatmap', 'Tableau croisé coloré', REL, 'Moyenne par deux catégories',
     lambda: C(essai).pivot_heatmap('dose', 'variete', 'rendement', annot=False)),

    ('pca_biplot', 'ACP (biplot)', MUL, 'Individus et variables sur 2 axes',
     lambda: C(caracteres).pca_biplot(group='cycle')),
    ('dendrogram', 'Dendrogramme', MUL, 'Classification hiérarchique',
     lambda: C(caracteres.head(16)).dendrogram(labels='variete', color_threshold=5)),

    ('pie', 'Camembert', CPO, "Parts d'un total",
     lambda: C(cultures.groupby('famille', as_index=False)['surface'].sum()).pie('surface', labels='famille')),
    ('donut', 'Anneau', CPO, 'Parts, anneaux imbriqués',
     lambda: C(cultures).donut('surface', show_values=False, center_text='1 865')),
    ('treemap', 'Carte arborescente', CPO, 'Parts, beaucoup de catégories',
     lambda: C(cultures).treemap('surface', labels='culture', show_values=False)),
    ('circle_packing', 'Cercles imbriqués', CPO, 'Parts regroupées par niveaux',
     lambda: C(cultures, 3.4, 3.4).circle_packing('surface', group='famille', show_labels=False)),
    ('waterfall', 'Cascade', CPO, 'Contributions successives',
     lambda: C(bilan).waterfall('poste', 'montant')),
    ('ternary', 'Ternaire', CPO, 'Compositions en 3 parts',
     lambda: C(sols, 3.6, 3.2).ternary('sable', 'limon', 'argile', group='zone')),
    ('waffle', 'Gaufre', CPO, 'Parts à compter, carré par carré',
     lambda: C(cultures_ha, 3.4, 3.4).waffle('culture', 'ha')),
    ('radial_bar', 'Barres radiales', CPO, 'Avancement de plusieurs indicateurs',
     lambda: C(avancement, 3.4, 3.4).radial_bar('pct', labels='activite')),
    ('progress_bar', 'Barre de progression', CPO, 'Objectif atteint',
     lambda: C(avancement, 4, 2).progress_bar('pct', labels='activite')),

    ('contour', 'Courbes de niveau', CHA, 'Courbes sur grille ou prélèvements',
     lambda: C(champ).contour('x', 'y', 'ph', colorbar=False, labels=False, show_points=True)),
    ('mesh', 'Grille colorée', CHA, 'Valeur par cellule (pcolormesh)',
     lambda: C(reponse).mesh('azote', 'eau', 'rendement', colorbar=False)),
    ('vector_field', 'Champ de vecteurs', CHA, 'Vent, écoulement (flèches, lignes)',
     lambda: C(vent).vector_field('lon', 'lat', 'u', 'v', kind='stream', colorbar=False)),

    ('scatter3d', 'Nuage 3D', D3, 'Trois variables',
     lambda: C(caracteres).scatter3d('hauteur', 'jours_floraison', 'rendement', group='cycle')),
    ('line3d', 'Courbe 3D', D3, 'Trajectoires',
     lambda: C(vol).line3d('x', 'y', 'z')),
    ('surface', 'Surface 3D', D3, 'Surface de réponse',
     lambda: C(reponse).surface('azote', 'eau', 'rendement', colorbar=False)),
    ('bar3d', 'Barres 3D', D3, 'Valeur par couple de catégories',
     lambda: C(moyennes).bar3d('variete', 'dose', 'moyenne')),

    ('ombrothermic', 'Ombrothermique', SCI, 'Mois secs et humides (p = 2T)',
     lambda: C(climat).ombrothermic('mois', 'temp', 'precip')),
    ('soil_texture', 'Textures USDA', SCI, 'Classer des sols',
     lambda: C(sols, 3.6, 3.2).soil_texture('sable', 'limon', 'argile', show_classes=True)),

    ('walter_lieth', 'Walter-Lieth', SCI, 'Climatogramme de référence',
     lambda: C(station_b).walter_lieth('mois', 'temperature', 'pluie')),
    ('climogram', 'Climogramme', SCI, 'Comparer des climats',
     lambda: C(stations_climat).climogram('pluie', 'temperature', group='station')),
    ('emberger', 'Climagramme d\'Emberger', SCI, 'Étage bioclimatique, Q2',
     lambda: C(stations_emberger).emberger('pluie', 'M', 'm', station='station', labels=False)),
    ('ecoclimatogram', 'Écoclimatogramme', SCI, 'Mois favorables à une espèce',
     lambda: C(ecoclimat).ecoclimatogram('humidite', 'temperature', group='station',
                                         optimum=(60, 80, 24, 30), tolerance=(40, 90, 18, 34))),
    ('water_balance', 'Bilan hydrique', SCI, 'Excédent et déficit P – ETP',
     lambda: C(station_b).water_balance('mois', 'pluie', 'etp')),
    ('degree_days', 'Degrés-jours', SCI, 'Somme des températures, stades',
     lambda: C(meteo).degree_days('jour', 'tmin', 'tmax', stages={'Floraison': 1250})),
    ('wind_rose', 'Rose des vents', SCI, 'Direction et vitesse du vent',
     lambda: C(vent_obs, 3.4, 3.4).wind_rose('direction', 'vitesse')),
    ('anomaly', 'Anomalies', SCI, 'Écart à une normale',
     lambda: C(annees).anomaly('annee', 'temperature', baseline=(1961, 1990))),
    ('climate_stripes', 'Bandes climatiques', SCI, 'Une couleur par année',
     lambda: C(annees, 4, 2).climate_stripes('annee', 'temperature', show_years=False)),
    ('soil_profile', 'Profil de sol', SCI, 'Horizons selon la profondeur',
     lambda: C(horizons).soil_profile('haut', 'bas', 'horizon', profile='fosse', color='couleur')),
    ('depth_profile', 'Selon la profondeur', SCI, 'Humidité, matière organique…',
     lambda: C(profil_humidite, 3.2, 3).depth_profile('profondeur', 'humidite', group='date')),
    ('soil_properties', 'Profil multi-variables', SCI, 'pH, MO, argile… par horizon',
     lambda: C(proprietes).soil_properties('haut', ['pH', 'MO (%)', 'Argile (%)'], bottom='bas', group='fosse').tight_layout()),
    ('retention_curve', 'Courbe pF', SCI, 'Rétention d\'eau, réserve utile',
     lambda: C(retention).retention_curve('pF', 'theta', group='horizon')),
    ('grain_size_curve', 'Courbe granulométrique', SCI, 'D10, D60, uniformité',
     lambda: C(granulo).grain_size_curve('diametre_mm', 'passant', group='echantillon')),
    ('exchange_complex', 'Complexe d\'échange', SCI, 'Cations, CEC, saturation',
     lambda: C(complexe).exchange_complex('horizon', ['Ca', 'Mg', 'K', 'Na'], acidity='H_Al', cec='CEC')),
    ('infiltration_curve', 'Infiltration', SCI, 'Modèles de Philip et Horton',
     lambda: C(infiltration).infiltration_curve('minutes', 'infiltration_mm').tight_layout()),
    ('isopleth', 'Isoplèthes', SCI, 'Profondeur × temps',
     lambda: C(humidite_sol).isopleth('date', 'profondeur', 'humidite', cmap='Blues', contour_labels=False).tight_layout()),

    ('map_points', 'Points sur carte', CAR, 'Stations, parcelles géolocalisées',
     lambda: C(villes, 3.6, 3).map_points('lon', 'lat', value='pluie', size='pluie')),
    ('choropleth', 'Carte de zones', CAR, 'Régions, parcelles colorées',
     lambda: C(parcelles).choropleth('rendement', legend=False)),
    ('flow_map', 'Carte de flux', CAR, 'Échanges entre lieux',
     lambda: C(flux_vignette, 3.6, 3).flow_map('lon', 'lat', 'lon2', 'lat2', weight='t')),
    ('density_map', 'Carte de densité', CAR, 'Zones de concentration de points',
     lambda: C(signalements, 3.6, 3).density_map('lon', 'lat', colorbar=False)),
    ('field_map', 'Carte interpolée', CAR, 'Isolignes à partir de stations',
     lambda: C(villes, 3.6, 3).field_map('lon', 'lat', 'pluie', colorbar=False, show_points=False)),
    ('route_map', 'Itinéraires', CAR, 'Trajets, transhumance, traces GPS',
     lambda: C(transhumance, 3.2, 3.2).route_map('lon', 'lat', order='jour')),

    ('glm_curve', 'Courbe de GLM', ECO, 'Présence/absence, comptages',
     lambda: C(grenouille).glm_curve('nitrates', 'presence', show_stats=False)),
    ('response_curve', 'Réponse lissée', ECO, 'Optimum le long d’un gradient',
     lambda: C(communaute).response_curve('humidite', 'Typha', frac=0.4, n_boot=50)),
    ('count_distribution', 'Excès de zéros', ECO, 'Comptages : Poisson ou binomiale négative',
     lambda: C(comptages).count_distribution('individus')),
    ('forest_plot', 'Forest plot', ECO, 'Effets estimés et intervalles',
     lambda: C(effets).forest_plot('variable', 'odds_ratio', 'ic_bas', 'ic_haut', ref=1, log=True)),
    ('ordination', 'Ordination (CCA)', ECO, 'AFC, CCA, ACP, RDA des communautés',
     lambda: C(communaute, 3.6, 3.2).ordination(especes, env=['humidite', 'ph'], method='cca',
                                                group='zone')),
    ('rank_abundance', 'Rang-abondance', ECO, 'Dominance et équitabilité',
     lambda: C(communaute.groupby('zone', sort=False)[especes].sum().reset_index())
     .rank_abundance(especes, labels='zone')),
    ('accumulation', 'Accumulation', ECO, 'Effort d’échantillonnage, Chao2',
     lambda: C(communaute).accumulation(especes, n_perm=50)),
    ('diversity', 'Diversité', ECO, 'Shannon, Simpson, Pielou par site',
     lambda: C(communaute).diversity(especes, group='zone')),

    ('word_cloud', 'Nuage de mots', QUA, "Thèmes dominants d'un texte",
     lambda: C(enquete).word_cloud('reponse', max_words=60)),
    ('word_frequency', 'Fréquence des mots', QUA, 'Mots les plus cités',
     lambda: C(enquete).word_frequency('reponse', top=8, show_values=False)),
    ('likert', 'Échelle de Likert', QUA, "Opinions d'enquête",
     lambda: C(opinions, 4.5, 3).likert(list(opinions.columns), order=echelle, show_values=False)),
    ('code_matrix', 'Matrice des codes', QUA, "Codage d'entretiens",
     lambda: C(segments).code_matrix('entretien', 'code', annot=False)),

    ('network', 'Réseau', GRA, 'Liens entre acteurs, échanges',
     lambda: C(echanges).network('de', 'vers', weight='tonnes', directed=True)),
    ('shortest_path', 'Plus court chemin', GRA, 'Itinéraire optimal (Dijkstra)',
     lambda: C(pistes).shortest_path('de', 'vers', 'Dépôt', 'Tivaouane', weight='km', edge_labels=False)),
    ('spanning_tree', 'Arbre couvrant minimal', GRA, 'Réseau le moins coûteux',
     lambda: C(pistes).spanning_tree('de', 'vers', weight='km', edge_labels=False)),
    ('max_flow', 'Flot maximal', GRA, 'Capacités, goulots',
     lambda: C(adduction).max_flow('de', 'vers', 'capacite', 'Forage', 'Marché')),
    ('critical_path', 'Chemin critique', GRA, 'Planning PERT / CPM',
     lambda: C(projet).critical_path('tache', 'duree', 'avant').tight_layout()),
    ('tsp_route', 'Tournée optimale', GRA, 'Voyageur de commerce',
     lambda: C(collecte).tsp_route('x', 'y', labels='ferme', start='Laiterie')),
    ('arc_diagram', 'Diagramme en arcs', GRA, 'Mouvements entre lieux',
     lambda: C(transferts).arc_diagram('depart', 'arrivee', weight='tetes')),

    ('pareto', 'Pareto', CPX, 'Les causes qui pèsent le plus (80/20)',
     lambda: C(pertes).pareto('cause', 'tonnes', show_values=False)),
    ('control_chart', 'Carte de contrôle', CPX, 'Stabilité d\'un procédé, dérives',
     lambda: C(lots).control_chart('lot', 'humidite')),
    ('marimekko', 'Marimekko', CPX, 'Deux niveaux de composition',
     lambda: C(assolement).marimekko('region', 'culture', 'surface', show_values=False)),
    ('bland_altman', 'Bland-Altman', CPX, 'Accord entre deux méthodes de mesure',
     lambda: C(mesures).bland_altman('gravimetrie', 'sonde')),
    ('raincloud', 'Nuage de pluie', CPX, 'Densité, boîte et observations',
     lambda: C(essai).raincloud('variete', 'rendement')),
    ('relplot', 'Grille de relations', CPX, 'Courbes ou nuages par panneau',
     lambda: Chart(essai).relplot('dose', 'rendement', hue='variete', kind='line', height=3, aspect=1.33, legend=False)),
    ('displot', 'Grille de distributions', CPX, 'Histogrammes et densités par groupe',
     lambda: Chart(essai).displot('rendement', hue='variete', kind='kde', fill=True, height=3, aspect=1.33, legend=False)),
    ('catplot', 'Grille par catégorie', CPX, 'Boîtes, violons, points par panneau',
     lambda: Chart(essai).catplot('variete', 'rendement', kind='box', height=3, aspect=1.33)),
    ('lmplot', 'Grille de régressions', CPX, 'Régressions par groupe',
     lambda: Chart(essai).lmplot('dose', 'rendement', hue='variete', order=2, height=3, aspect=1.33, legend=False)),
    ('jointplot', 'Nuage + marges', CPX, 'Relation et distributions marginales',
     lambda: Chart(caracteres).jointplot('jours_floraison', 'hauteur', height=3)),
    ('pairplot', 'Matrice de nuages', CPX, 'Toutes les paires de variables',
     lambda: Chart(caracteres).pairplot(vars=traits[:3], hue='cycle', height=1.1, corner=True)),
    ('clustermap', 'Clustermap', CPX, 'Lignes et colonnes classées',
     lambda: Chart(caracteres.head(15), figsize=(4, 4)).clustermap(labels='variete', cbar_pos=None)),
    ('seaborn', 'Tout seaborn', CPX, "N'importe quel graphique seaborn",
     lambda: C(essai).seaborn('violinplot', x='variete', y='rendement', hue='variete', legend=False)),
    ('sankey', 'Sankey', DIA, 'Flux entre étapes, sur plusieurs niveaux',
     lambda: C(devenir).sankey('de', 'vers', 'tonnes', show_values=False)),
    ('chord', 'Cordes', DIA, 'Échanges croisés entre acteurs',
     lambda: C(echanges_regions, 3.4, 3.4).chord('de', 'vers', 'kt')),
    ('venn', 'Venn', DIA, 'Recouvrements entre ensembles',
     lambda: C(menages, 3.4, 3.4).venn(['Maraîchage', 'Élevage', 'Pêche'])),
    ('flowchart', 'Logigramme', DIA, 'Étapes, décisions, organigrammes',
     lambda: C(controle, 3.2, 4.2).flowchart('de', 'vers', label='si', fontsize=6)),
    ('sequence_diagram', 'Séquence', DIA, 'Messages entre acteurs dans le temps',
     lambda: C(paiement, 4.6, 3).sequence_diagram('de', 'vers', 'message', dashed='reponse', fontsize=6)),
]


def build(only=()):
    out = DOCS / 'images' / 'vignettes'
    out.mkdir(parents=True, exist_ok=True)
    items = []
    for name, title, cat, usage, make in CATALOGUE:
        items.append({'title': title, 'description': f'{name}() — {usage}',
                      'path': f'charts/{PAGE[cat]}.html#g-{name}',
                      'image': f'images/vignettes/{name}.png', 'categories': [cat]})
        if only and name not in only:
            continue
        with theme_context('editorial'):
            chart = make()
            for ax in chart.fig.axes:  # tous les panneaux (soil_properties…)
                if ax.get_legend() is not None and name not in ('polar', 'scatter3d'):
                    ax.get_legend().remove()
            chart.save(out / f'{name}.png', dpi=80, bbox_inches=None)   # applique l'arrondi par défaut
        plt.close('all')
    lines = []
    for it in items:
        lines += [f"- title: \"{it['title']}\"", f"  description: \"{it['description']}\"",
                  f"  path: {it['path']}", f"  image: {it['image']}",
                  f"  categories: [\"{it['categories'][0]}\"]"]
    (DOCS / 'galerie.yml').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f'{len(items)} vignettes → {out}')


if __name__ == '__main__':
    import sys
    build(sys.argv[1:])
