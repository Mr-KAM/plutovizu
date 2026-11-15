"""Fiches des graphiques : description, cas d'usage, données attendues et paramètres.

Le contenu rédigé est dans fiches/<page>.yml ; les paramètres (noms, valeurs par défaut)
sont lus dans la signature de la méthode, donc toujours à jour. Description d'un paramètre :
fiche YAML > docstring (ligne « FR : ») > COMMUNS. Dans une page :

    ```{python}
    #| echo: false
    #| output: asis
    fiche('line')
    ```
"""
import inspect
import pathlib
import re

import yaml

from plutovizu import Chart

_DIR = pathlib.Path('.')  # ponytail: pages et vignettes s'exécutent depuis docs/charts
FICHES = {}
for _f in sorted((_DIR / 'fiches').glob('*.yml')):
    FICHES.update(yaml.safe_load(_f.read_text(encoding='utf-8')) or {})

COMMUNS = {
    'x': "Colonne de l'axe horizontal.",
    'y': "Colonne de l'axe vertical.",
    'z': 'Colonne de la troisième dimension (hauteur, valeur).',
    'group': 'Colonne de regroupement : une couleur (et une entrée de légende) par groupe.',
    'hue': 'Colonne de regroupement : une couleur par modalité.',
    'alpha': 'Opacité, de 0 (transparent) à 1 (opaque).',
    'color': 'Couleur unique (nom, hexadécimal). `None` : couleur du thème.',
    'cmap': 'Palette de couleurs continue (nom matplotlib). `None` : celle du thème.',
    'colorbar': "Affiche l'échelle de couleurs.",
    'labels': 'Colonne des étiquettes (noms affichés).',
    'marker': 'Symbole des points (`o`, `s`, `^`, `None`…).',
    'linewidth': 'Épaisseur des lignes.',
    'markersize': 'Taille des points.',
    'show_values': 'Écrit les valeurs sur le graphique.',
    'value_format': 'Format des valeurs affichées (`{:,.0f}`, `{:.1%}`…).',
    'kwargs': 'Options supplémentaires transmises à la fonction sous-jacente (seaborn ou matplotlib).',
    'basemap': 'Ajoute un fond de carte en ligne (option `map`, connexion Internet).',
    'basemap_source': 'Fournisseur de tuiles contextily ; `None` : Esri WorldGrayCanvas.',
    'lon': 'Colonne des longitudes (degrés décimaux, WGS84).',
    'lat': 'Colonne des latitudes (degrés décimaux, WGS84).',
    'show_points': 'Affiche aussi les points de mesure.',
    'levels': 'Nombre de niveaux (ou liste des seuils) des courbes de niveau.',
    'columns': 'Liste des colonnes numériques à représenter.',
    'species': "Liste des colonnes d'espèces (abondances, une colonne par espèce).",
    'random_state': 'Graine du hasard, pour un résultat reproductible.',
    'edgecolor': 'Couleur des contours. `None` : thème.',
    'fill': 'Remplit les surfaces.',
}

TYPES = {bool: 'booléen', int: 'entier', float: 'nombre', str: 'texte', tuple: 'tuple', list: 'liste'}


def _docstring_fr(method):
    """{paramètre: description FR} lus dans la section « Args / Paramètres » du docstring.

    Deux formats : « nom (type): » suivi de lignes « EN: » / « FR: », ou tout sur une ligne
    « nom (type): EN/FR: texte » (plusieurs noms possibles : « words, weights (str): »).
    """
    doc = inspect.getdoc(method) or ''
    out, current, lang = {}, [], None
    for line in doc.splitlines():
        m = re.match(r'\s{4}(\*{0,2}\w+(?:,\s*\w+)*)\s*\(.*?\):\s*(.*)$', line)
        if m:
            current = [n.strip().lstrip('*') for n in m.group(1).split(',')]
            rest = '' if re.match(r'EN\s*:', m.group(2)) else \
                re.sub(r'^(EN/FR|FR)\s*:\s*', '', m.group(2)).strip()
            lang = 'FR' if rest else None
            for n in current:
                if rest:
                    out[n] = rest
            continue
        m = re.match(r'\s+(EN|FR)\s*:\s*(.*)', line)
        if m and current:
            lang = m.group(1)
            if lang == 'FR':
                for n in current:
                    out[n] = m.group(2).strip()
            continue
        if current and lang == 'FR' and line.startswith(' ' * 12) and line.strip():
            for n in current:                       # suite de la description FR
                out[n] += ' ' + line.strip()
        elif not line.startswith(' '):
            current, lang = [], None
    # la colonne « Défaut » du tableau donne déjà la valeur par défaut
    return {k: re.sub(r'\s*Défaut\b.*$', '', v).strip() or v for k, v in out.items()}


def _cell(text):
    return str(text).replace('|', '\\|').replace('\n', ' ')


def _default(p):
    if p.kind is p.VAR_KEYWORD:
        return '—'
    if p.default is inspect.Parameter.empty:
        return '**obligatoire**'
    return f'`{p.default!r}`'


def describe(name):
    """(paramètre, défaut, description) pour chaque paramètre ; description None si absente."""
    method = getattr(Chart, name)
    own = (FICHES.get(name) or {}).get('params') or {}
    doc = _docstring_fr(method)
    rows = []
    for p in inspect.signature(method).parameters.values():
        if p.name == 'self':
            continue
        label = f'`**{p.name}`' if p.kind is p.VAR_KEYWORD else f'`{p.name}`'
        rows.append((label, _default(p), own.get(p.name) or doc.get(p.name) or COMMUNS.get(p.name)))
    return rows


def fiche(name):
    """Affiche la fiche Markdown d'un graphique (cellule `output: asis`)."""
    f = FICHES.get(name)
    if f is None:
        raise KeyError(f'Pas de fiche pour {name!r} (docs/charts/fiches/*.yml)')
    out = [f"{f['desc'].strip()}", '',
           f"**Quand l'utiliser.** {f['usage'].strip()}", '']
    if f.get('data'):
        kind = 'GeoDataFrame' if f.get('geo') else 'DataFrame'
        out += [f"**Données attendues** : un {kind}, {f['rows'].strip()}.", '',
                '| Colonne (exemple) | Type | Rôle | Exemple de valeur |',
                '|:--|:--|:--|:--|']
        out += [f'| `{_cell(c)}` | {_cell(t)} | {_cell(r)} | {_cell(v)} |' for c, t, r, v in f['data']]
        out.append('')
    elif f.get('rows'):
        out += [f"**Données attendues** : {f['rows'].strip()}", '']
    out += ['**Paramètres**', '', '| Paramètre | Défaut | Description |', '|:--|:--|:--|']
    out += [f'| {p} | {_cell(d)} | {_cell(t or "")} |' for p, d, t in describe(name)]
    if f.get('returns'):
        out += ['', f"**Résultats calculés** : {f['returns'].strip()}"]
    out += ['', '**Exemple**', '']
    print('\n'.join(out))
