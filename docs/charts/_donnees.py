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
