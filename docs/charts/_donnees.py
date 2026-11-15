"""Données d'exemple partagées par les pages de graphiques de la documentation.

Chargé en tête de chaque page (``exec(open('_donnees.py').read())``) et par
``docs/_vignettes.py``. Applique le thème par défaut de la documentation : editorial.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plutovizu import Chart, set_theme, theme_context

set_theme('editorial')

# ── Essai agronomique : 3 variétés de riz × 5 doses d'azote × 4 blocs
rng = np.random.default_rng(3)
_lignes = []
for _variete, _gain in [('Sahel-108', 0.0), ('IR-841', 0.7), ('Nerica-4', 1.3)]:
    for _dose in [0, 30, 60, 90, 120]:
        for _bloc in range(1, 5):
            _lignes.append({
                'variete': _variete, 'dose': _dose, 'bloc': _bloc,
                'rendement': (2.5 + _gain + 0.035 * _dose - 0.00018 * _dose ** 2
                              + (0.004 * _dose if _variete == 'Nerica-4' else 0)
                              + rng.normal(0, 0.25)),
            })
essai = pd.DataFrame(_lignes)

# ── Caractères de 30 variétés (ACP, classification)
rng = np.random.default_rng(7)
_n = 30
_precocite = rng.normal(size=_n)
caracteres = pd.DataFrame({
    'variete': [f'V{i:02d}' for i in range(_n)],
    'cycle': np.where(_precocite > 0, 'tardif', 'précoce'),
    'hauteur': 95 + 12 * _precocite + rng.normal(0, 5, _n),
    'jours_floraison': 85 + 10 * _precocite + rng.normal(0, 3, _n),
    'talles': 12 - 2 * _precocite + rng.normal(0, 1, _n),
    'pmg': 25 + rng.normal(0, 2.5, _n),
    'rendement': 4.5 + 0.5 * _precocite + rng.normal(0, 0.4, _n),
})

# ── Texture des sols de 40 parcelles
rng = np.random.default_rng(11)
sols = pd.DataFrame({'zone': rng.choice(['bas-fond', 'plateau'], 40)})
sols['sable'] = np.where(sols['zone'] == 'plateau', rng.uniform(55, 85, 40), rng.uniform(10, 45, 40))
sols['argile'] = rng.uniform(5, 100 - sols['sable'] - 5)
sols['limon'] = 100 - sols['sable'] - sols['argile']

# ── Surfaces cultivées par famille
cultures = pd.DataFrame({
    'famille': ['Céréales'] * 4 + ['Légumineuses'] * 3 + ['Tubercules'] * 3 + ['Maraîchage'] * 3,
    'culture': ['Riz', 'Maïs', 'Mil', 'Sorgho', 'Niébé', 'Arachide', 'Soja',
                'Manioc', 'Igname', 'Patate douce', 'Tomate', 'Oignon', 'Gombo'],
    'surface': [420, 310, 180, 150, 140, 260, 60, 230, 120, 70, 45, 55, 25],
})

# ── Météo journalière d'une saison (90 jours)
rng = np.random.default_rng(21)
_jours = pd.date_range('2024-06-01', periods=90, freq='D')
_t = np.arange(90)
meteo = pd.DataFrame({
    'jour': _jours,
    'tmoy': 28 - 3 * np.sin(_t / 90 * np.pi) + rng.normal(0, 0.6, 90),
    'pluie': np.where(rng.random(90) < 0.35, rng.gamma(2, 9, 90), 0).round(1),
})
meteo['tmin'] = meteo['tmoy'] - rng.uniform(4, 6, 90)
meteo['tmax'] = meteo['tmoy'] + rng.uniform(4, 7, 90)

# ── Prélèvements de sol dispersés dans une parcelle (carte du pH)
rng = np.random.default_rng(4)
_x, _y = rng.uniform(0, 200, 60), rng.uniform(0, 120, 60)
champ = pd.DataFrame({'x': _x.round(1), 'y': _y.round(1),
                      'ph': (5.4 + 0.006 * _x + 0.4 * np.exp(-((_x - 140) ** 2 + (_y - 40) ** 2) / 1500)
                             + rng.normal(0, 0.05, 60)).round(2)})

# ── Surface de réponse sur une grille régulière (azote × irrigation)
_az, _eau = np.meshgrid(np.linspace(0, 150, 16), np.linspace(200, 800, 13))
reponse = pd.DataFrame({'azote': _az.ravel(), 'eau': _eau.ravel()})
reponse['rendement'] = (2 + 0.03 * reponse['azote'] - 0.00012 * reponse['azote'] ** 2
                        + 0.006 * reponse['eau'] - 0.0000045 * reponse['eau'] ** 2)

# ── Vent sur une grille régulière (tourbillon autour d'une dépression)
_lon, _lat = np.meshgrid(np.linspace(-4, 4, 17), np.linspace(-3, 3, 13))
vent = pd.DataFrame({'lon': _lon.ravel(), 'lat': _lat.ravel()})
vent['u'] = -vent['lat'] * 2 + 3
vent['v'] = vent['lon'] * 2

# ── Rose des vents : fréquence par direction et classe de vitesse
_dirs = ['N', 'NE', 'E', 'SE', 'S', 'SO', 'O', 'NO']
rose = pd.DataFrame([(d, v, f) for d, base in zip(_dirs, [4, 6, 9, 5, 3, 7, 12, 8])
                     for v, f in zip(['< 2 m/s', '2–5 m/s', '> 5 m/s'], [base, base * 0.7, base * 0.3])],
                    columns=['direction', 'vitesse', 'frequence'])

# ── Calendrier cultural
calendrier = pd.DataFrame({
    'date': pd.to_datetime(['2024-06-10', '2024-06-25', '2024-07-15', '2024-08-05',
                            '2024-08-28', '2024-09-20', '2024-10-25']),
    'etape': ['Labour', 'Semis', 'Désherbage', 'Fertilisation', 'Épiaison', 'Traitement', 'Récolte'],
})

# ── Dates de floraison observées par variété
rng = np.random.default_rng(8)
floraison = pd.DataFrame([
    {'variete': v, 'date': pd.Timestamp('2024-08-01') + pd.Timedelta(days=int(d))}
    for v, centre in [('Sahel-108', 10), ('IR-841', 20), ('Nerica-4', 30)]
    for d in rng.normal(centre, 4, 15)])

# ── Pyramide des âges d'un village
rng = np.random.default_rng(9)
population = pd.DataFrame({
    'age': np.concatenate([rng.gamma(2.2, 13, 400), rng.gamma(2.4, 13, 380)]).clip(0, 90),
    'sexe': ['Femmes'] * 400 + ['Hommes'] * 380,
})

# ── Avancement d'un projet
avancement = pd.DataFrame({'activite': ['Formation', 'Distribution des semences', 'Suivi des parcelles'],
                           'pct': [85, 60, 35]})

# ── Enquête qualitative auprès de producteurs
rng = np.random.default_rng(5)
temoignages = [
    "Le prix de l'engrais est trop élevé cette année",
    "Manque d'eau pour l'irrigation en saison sèche",
    "Accès au crédit difficile pour acheter les semences",
    "Les semences améliorées donnent un meilleur rendement",
    "Le prix des semences et de l'engrais augmente encore",
    "L'eau manque et les pompes tombent souvent en panne",
    "Formation utile sur les techniques d'irrigation goutte à goutte",
    "Le crédit agricole arrive trop tard pour la campagne",
    "Rendement en hausse grâce aux semences améliorées",
    "Difficile de vendre la récolte à bon prix au marché",
    "Les routes empêchent d'écouler la récolte au marché",
    "Le prix de vente du riz ne couvre pas le prix de l'engrais",
]
enquete = pd.DataFrame({'reponse': rng.choice(temoignages, 60)})
echelle = ['Pas du tout d’accord', 'Pas d’accord', 'Neutre', 'D’accord', 'Tout à fait d’accord']
opinions = pd.DataFrame({q: rng.choice(echelle, 80, p=p) for q, p in {
    'La formation était utile': [.05, .10, .15, .40, .30],
    'Les semences sont accessibles': [.25, .30, .20, .15, .10],
    'Le crédit arrive à temps': [.35, .30, .15, .15, .05],
    'Je recommande le projet': [.05, .05, .20, .35, .35],
}.items()})
segments = pd.DataFrame({
    'entretien': rng.choice([f'Entretien {i}' for i in range(1, 7)], 120),
    'code': rng.choice(['Eau', 'Prix', 'Crédit', 'Semences', 'Formation', 'Marché'], 120,
                       p=[.25, .2, .15, .15, .1, .15]),
})

# ── Climat de trois stations FICTIVES (valeurs illustratives, types sahélien → guinéen)
_mois = ['Jan', 'Fév', 'Mar', 'Avr', 'Mai', 'Juin', 'Juil', 'Août', 'Sep', 'Oct', 'Nov', 'Déc']
stations_climat = pd.DataFrame({
    'station': np.repeat(['Station A (sahélienne)', 'Station B (soudanienne)', 'Station C (guinéenne)'], 12),
    'mois': _mois * 3,
    'temperature': [22, 25, 29, 32, 34, 33, 30, 28, 29, 30, 27, 23,
                    25, 27, 30, 32, 31, 28, 26, 26, 26, 27, 26, 25,
                    26, 27, 27, 27, 26, 25, 24, 24, 25, 25, 26, 26],
    'pluie': [0, 0, 0, 1, 8, 35, 110, 170, 95, 15, 1, 0,
              0, 1, 4, 20, 85, 150, 230, 290, 220, 85, 10, 1,
              8, 15, 45, 110, 190, 240, 280, 330, 300, 220, 90, 20],
    'tmin': [14, 16, 20, 24, 26, 26, 24, 23, 23, 22, 18, 15,
             17, 19, 22, 25, 24, 22, 21, 21, 21, 21, 19, 17,
             20, 21, 22, 22, 22, 21, 21, 21, 21, 21, 21, 20],
    'etp': [140, 160, 210, 230, 240, 210, 170, 140, 145, 160, 145, 135,
            150, 165, 200, 210, 185, 150, 125, 115, 120, 140, 145, 145,
            120, 130, 150, 150, 140, 125, 110, 105, 110, 120, 120, 115],
})
station_b = stations_climat[stations_climat['station'].str.startswith('Station B')].reset_index(drop=True)

# ── Observations de vent brutes (direction en degrés, vitesse en m/s)
rng = np.random.default_rng(12)
vent_obs = pd.DataFrame({
    'direction': np.concatenate([rng.normal(270, 25, 500), rng.normal(45, 30, 250),
                                 rng.uniform(0, 360, 150)]) % 360,
    'vitesse': np.concatenate([rng.gamma(3, 1.4, 500), rng.gamma(2, 1.2, 250), rng.gamma(1.5, 1, 150)]),
})

# ── Température annuelle d'une station fictive (tendance + variabilité)
rng = np.random.default_rng(13)
annees = pd.DataFrame({'annee': np.arange(1961, 2025)})
annees['temperature'] = 26.8 + 0.022 * (annees['annee'] - 1961) + rng.normal(0, 0.25, len(annees))
annees['pluie'] = 620 + rng.normal(0, 110, len(annees))

# ── Horizons de trois fosses pédologiques (couleurs approchées des notations Munsell)
horizons = pd.DataFrame({
    'fosse': ['Plateau'] * 4 + ['Versant'] * 3 + ['Bas-fond'] * 4,
    'horizon': ['A', 'Bt', 'Bc', 'C', 'A', 'Bw', 'C', 'Ag', 'Bg1', 'Bg2', 'Cg'],
    'haut': [0, 18, 55, 110, 0, 25, 70, 0, 20, 60, 105],
    'bas': [18, 55, 110, 160, 25, 70, 140, 20, 60, 105, 150],
    'couleur': ['#6b4f3a', '#a0522d', '#b5651d', '#d2b48c', '#7a5c44', '#b08a5e', '#d8c3a5',
                '#5b5446', '#7d7f6b', '#8e9585', '#a9ae9f'],
})

# ── Humidité du sol selon la profondeur, à deux dates
profil_humidite = pd.DataFrame({
    'profondeur': [0, 10, 20, 40, 60, 80, 100] * 2,
    'humidite': [32, 30, 28, 25, 23, 22, 22, 9, 11, 14, 17, 19, 20, 21],
    'date': ['Fin de saison des pluies'] * 7 + ['Fin de saison sèche'] * 7,
})

# ── Villes du Sénégal : coordonnées réelles, valeurs ILLUSTRATIVES
villes = pd.DataFrame({
    'ville': ['Dakar', 'Thiès', 'Saint-Louis', 'Louga', 'Diourbel', 'Kaolack', 'Fatick',
              'Tambacounda', 'Kolda', 'Ziguinchor', 'Kédougou', 'Matam'],
    'lon': [-17.44, -16.93, -16.49, -16.22, -16.23, -16.07, -16.41, -13.67, -14.94, -16.27, -12.18, -13.26],
    'lat': [14.69, 14.79, 16.03, 15.62, 14.66, 14.15, 14.34, 13.77, 12.89, 12.58, 12.56, 15.66],
    'pluie': [380, 480, 280, 330, 520, 640, 620, 820, 1050, 1350, 1200, 400],
    'zone': ['Ouest', 'Ouest', 'Nord', 'Nord', 'Centre', 'Centre', 'Centre', 'Est', 'Sud', 'Sud', 'Sud', 'Nord'],
})

# ── Signalements d'un ravageur autour de trois foyers FICTIFS
rng = np.random.default_rng(2)
_foyers = [(-16.07, 14.15, 120), (-15.6, 13.6, 60), (-16.2, 12.6, 80)]
signalements = pd.DataFrame(np.vstack([np.c_[rng.normal(x, .18, n), rng.normal(y, .15, n)]
                                       for x, y, n in _foyers]), columns=['lon', 'lat'])

# ── Trajet de transhumance FICTIF (localités réelles, dates illustratives)
transhumance = pd.DataFrame({
    'etape': ['Podor', 'Linguère', 'Dahra', 'Kaffrine', 'Koungheul', 'Tambacounda'],
    'lon': [-14.96, -15.12, -15.48, -15.55, -14.80, -13.67],
    'lat': [16.65, 15.39, 15.35, 14.10, 13.98, 13.77],
    'jour': [0, 9, 12, 25, 31, 44],
})

# ── Écologie (données FICTIVES) ─────────────────────────────────────────────
rng = np.random.default_rng(7)
_nit = rng.uniform(0, 60, 250)
grenouille = pd.DataFrame({'nitrates': _nit.round(1),
                           'presence': (rng.random(250) < 1 / (1 + np.exp(-(2.2 - 0.09 * _nit)))).astype(int)})
_ph = rng.uniform(4, 9, 400)
niche = pd.DataFrame({'ph': _ph.round(2),
                      'presence': (rng.random(400) < 1 / (1 + np.exp(-(1.5 - 1.6 * (_ph - 6.6) ** 2)))).astype(int)})
# Communauté : 30 placettes × 10 espèces de zone humide, le long d'un gradient d'humidité
_hum = np.sort(rng.uniform(5, 95, 30))
especes = ['Carex', 'Juncus', 'Typha', 'Phragmites', 'Salix', 'Alnus', 'Iris', 'Mentha', 'Lythrum', 'Lemna']
communaute = pd.DataFrame(rng.poisson(25 * np.exp(-((_hum[:, None] - np.linspace(10, 90, 10)[None, :]) / 14) ** 2)),
                          columns=especes)
communaute['humidite'] = _hum.round(1)
communaute['ph'] = (6 + _hum / 60 + rng.normal(0, .25, 30)).round(2)
communaute['placette'] = [f'P{i + 1:02d}' for i in range(30)]
communaute['zone'] = pd.cut(_hum, [0, 35, 65, 100], labels=['sèche', 'intermédiaire', 'humide']).astype(str)
# Comptages d'individus avec excès de zéros
comptages = pd.DataFrame({'individus': np.where(rng.random(400) < .35, 0, rng.negative_binomial(3, .4, 400))})
# Résultats d'un modèle (odds ratios et IC 95 %, illustratifs)
effets = pd.DataFrame({'variable': ['Nitrates', 'Conductivité', 'Couvert végétal', 'Profondeur', 'Distance à la route'],
                       'odds_ratio': [0.62, 0.85, 1.9, 1.15, 1.02],
                       'ic_bas': [0.48, 0.66, 1.3, 0.92, 0.95], 'ic_haut': [0.80, 1.09, 2.8, 1.43, 1.10]})

# ── Pédologie (données FICTIVES) ────────────────────────────────────────────
_rp = np.random.default_rng(3)

# Courbe pF : 2 horizons, van Genuchten + bruit
_pf = np.array([0, 1, 1.5, 2, 2.5, 3, 3.5, 4.2])
def _vg(h, tr, ts, a, n):
    return tr + (ts - tr) / (1 + (a * h) ** n) ** (1 - 1 / n)
retention = pd.concat([
    pd.DataFrame({'horizon': nom, 'pF': _pf,
                  'theta': _vg(10 ** _pf, *p) + _rp.normal(0, .004, _pf.size)})
    for nom, p in [('A (limoneux)', (.08, .45, .02, 1.45)), ('Bt (argileux)', (.15, .50, .008, 1.25))]
], ignore_index=True)

# Granulométrie cumulative : 3 échantillons
_d = np.array([0.001, 0.002, 0.005, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5])
def _cum(d50, s):
    from scipy.stats import norm
    return 100 * norm.cdf((np.log10(_d) - np.log10(d50)) / s)
granulo = pd.concat([pd.DataFrame({'echantillon': n, 'diametre_mm': _d, 'passant': _cum(m, s)})
                     for n, m, s in [('Sable de dune', .25, .25), ('Limon', .03, .6), ('Argile', .004, .7)]],
                    ignore_index=True)

# Complexe d'échange
complexe = pd.DataFrame({'horizon': ['A', 'E', 'Bt', 'C'],
                         'Ca': [9.5, 4.1, 12.3, 14.0], 'Mg': [2.1, 1.0, 3.4, 3.9],
                         'K': [0.6, 0.3, 0.5, 0.4], 'Na': [0.1, 0.1, 0.3, 0.4],
                         'H_Al': [1.8, 2.6, 1.2, 0.3]})
complexe['CEC'] = complexe[['Ca', 'Mg', 'K', 'Na', 'H_Al']].sum(axis=1) + [0.4, 0.2, 0.5, 0.1]

# Infiltration (double anneau) : Philip S=6 mm/min½, A=0.4 mm/min
_t = np.array([1, 2, 5, 10, 15, 20, 30, 45, 60, 90, 120])
infiltration = pd.DataFrame({'minutes': _t,
                             'infiltration_mm': 6 * np.sqrt(_t) + .4 * _t + _rp.normal(0, .6, _t.size)})

# Isoplèthes : humidité selon date × profondeur
_dates = pd.date_range('2025-01-01', periods=24, freq='SMS')
_prof = np.arange(10, 110, 10)
_jour = (_dates.dayofyear.to_numpy() / 365)[None, :]
humidite_sol = pd.DataFrame(
    [(dt, z, 30 - 12 * np.sin(2 * np.pi * (j - .15 - z / 400)) * np.exp(-z / 120))
     for dt, j in zip(_dates, _jour[0]) for z in _prof],
    columns=['date', 'profondeur', 'humidite'])

# Propriétés par horizon, 2 fosses
proprietes = pd.DataFrame({
    'fosse': ['Fosse 1'] * 4 + ['Fosse 2'] * 4,
    'haut': [0, 20, 45, 80] * 2, 'bas': [20, 45, 80, 120] * 2,
    'pH': [6.2, 5.8, 6.4, 7.1, 5.4, 5.1, 5.6, 6.0],
    'MO (%)': [3.8, 1.2, 0.6, 0.2, 5.1, 2.0, 0.8, 0.3],
    'Argile (%)': [18, 15, 34, 28, 12, 11, 25, 22],
    'CEC (cmol+/kg)': [14, 8, 19, 17, 11, 7, 14, 12]})

# ── Graphes & réseaux (données FICTIVES) ────────────────────────────────────
# Pistes entre villages (km) — réseau non orienté
pistes = pd.DataFrame([
    ('Dépôt', 'Keur Massar', 12), ('Dépôt', 'Ndiaye', 18), ('Keur Massar', 'Ndiaye', 7),
    ('Keur Massar', 'Sangalkam', 15), ('Ndiaye', 'Bayakh', 11), ('Sangalkam', 'Bayakh', 9),
    ('Sangalkam', 'Kayar', 21), ('Bayakh', 'Kayar', 14), ('Bayakh', 'Tivaouane', 26),
    ('Kayar', 'Tivaouane', 17), ('Ndiaye', 'Pout', 20), ('Pout', 'Tivaouane', 24)],
    columns=['de', 'vers', 'km'])

# Réseau d'adduction d'eau (m³/h) — orienté, capacités
adduction = pd.DataFrame([
    ('Forage', 'Château A', 60), ('Forage', 'Château B', 45), ('Château A', 'Quartier 1', 30),
    ('Château A', 'Château B', 15), ('Château B', 'Quartier 2', 40), ('Quartier 1', 'Marché', 25),
    ('Quartier 2', 'Marché', 35), ('Château A', 'Quartier 2', 10)],
    columns=['de', 'vers', 'capacite'])

# Projet d'aménagement hydro-agricole (semaines)
projet = pd.DataFrame({
    'tache': ['Étude', 'Financement', 'Défrichage', 'Canaux', 'Pompe', 'Parcelles', 'Formation', 'Mise en eau'],
    'duree': [4, 6, 3, 8, 5, 4, 3, 2],
    'avant': ['', 'Étude', 'Étude', 'Défrichage, Financement', 'Financement', 'Défrichage',
              'Financement', 'Canaux, Pompe, Parcelles']})

# Tournée de collecte de lait (km, coordonnées locales)
collecte = pd.DataFrame({
    'ferme': ['Laiterie', 'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9'],
    'x': [0, 4, 9, 12, 7, 2, -3, -6, -4, 6],
    'y': [0, 5, 3, -2, -6, -8, -5, 1, 6, 10]})

# Mouvements de troupeaux entre zones (têtes)
transferts = pd.DataFrame([
    ('Ferlo', 'Saloum', 420), ('Ferlo', 'Niayes', 150), ('Fleuve', 'Ferlo', 300),
    ('Saloum', 'Casamance', 120), ('Niayes', 'Ferlo', 60), ('Saloum', 'Ferlo', 90),
    ('Fleuve', 'Saloum', 80), ('Casamance', 'Saloum', 40)],
    columns=['depart', 'arrivee', 'tetes'])

# Échanges entre coopératives — réseau orienté pondéré
echanges = pd.DataFrame([
    ('Coop A', 'Coop B', 12), ('Coop B', 'Coop C', 5), ('Coop A', 'Coop C', 8),
    ('Coop C', 'Coop D', 9), ('Coop D', 'Coop A', 3), ('Coop B', 'Coop E', 7),
    ('Coop E', 'Coop D', 4), ('Coop C', 'Coop B', 2), ('Coop F', 'Coop A', 6), ('Coop F', 'Coop E', 3)],
    columns=['de', 'vers', 'tonnes'])
