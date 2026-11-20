"""Diagrammes : Sankey, chord, Venn, logigrammes / organigrammes (flowchart) et diagrammes de
séquence. Méthodes de Chart (classe mixin)."""
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.path import Path
from matplotlib.patches import Circle, PathPatch, Rectangle, Wedge

_M, _L, _C3, _C4, _CL = Path.MOVETO, Path.LINETO, Path.CURVE3, Path.CURVE4, Path.CLOSEPOLY


def _mix(color, background, t: float) -> str:
    """Mélange une couleur avec le fond : t = 0 → la couleur, t = 1 → le fond."""
    c, b = np.array(mcolors.to_rgb(color)), np.array(mcolors.to_rgb(background))
    return mcolors.to_hex(c + (b - c) * t)


def _appearance(*cols) -> List[str]:
    """Nœuds dans l'ordre d'apparition, ligne par ligne."""
    return list(dict.fromkeys(np.ravel(np.column_stack([np.asarray(c, dtype=str) for c in cols]))))


def _links(data: pd.DataFrame, source: str, target: str, value: Optional[str]) -> pd.DataFrame:
    """Liens agrégés (s, t, v) ; sans `value`, chaque ligne compte pour 1."""
    v = data[value].astype(float).to_numpy() if value else np.ones(len(data))
    links = pd.DataFrame({'s': data[source].astype(str).to_numpy(),
                          't': data[target].astype(str).to_numpy(), 'v': v})
    links = links.groupby(['s', 't'], sort=False, as_index=False)['v'].sum()
    return links[links['v'] > 0].reset_index(drop=True)


def _layers(nodes: List[str], edges: List[tuple]):
    """Rang de chaque nœud (plus long chemin depuis les racines) ; les arcs qui ferment un
    cycle (retours) sont ignorés pour le calcul et renvoyés à part."""
    succ = {n: [] for n in nodes}
    for a, b in edges:
        succ[a].append(b)
    state, back = {}, set()
    for root in nodes:                        # parcours en profondeur itératif
        if root in state:
            continue
        stack = [(root, iter(succ[root]))]
        state[root] = 1
        while stack:
            n, it = stack[-1]
            m = next(it, None)
            if m is None:
                state[n] = 2
                stack.pop()
            elif state.get(m) == 1:
                back.add((n, m))
            elif m not in state:
                state[m] = 1
                stack.append((m, iter(succ[m])))
    layer = dict.fromkeys(nodes, 0)
    forward = [(a, b) for a, b in edges if (a, b) not in back and a != b]
    for _ in range(len(nodes)):
        changed = False
        for a, b in forward:
            if layer[b] < layer[a] + 1:
                layer[b], changed = layer[a] + 1, True
        if not changed:
            break
    return layer, back


def _fit(ax, x0: float, x1: float, y0: float, y1: float, aspect='equal') -> None:
    """Montre au moins le cadre (x0, x1, y0, y1) à proportions fixes en remplissant tout l'axe :
    x ou y est élargi selon figsize (adjustable='box' rétrécirait l'axe, figsize ignoré)."""
    ax.update_datalim([(x0, y0), (x1, y1)])
    ax.margins(0)
    ax.autoscale_view()
    ax.set_aspect(aspect, adjustable='datalim')


class _Diamond:
    """Forme de losange pour les boîtes de texte (décisions d'un logigramme)."""

    def __init__(self, pad: float = 0.3):
        self.pad = pad

    def __call__(self, x0, y0, width, height, mutation_size):
        p = self.pad * mutation_size
        cx, cy = x0 + width / 2, y0 + height / 2
        w, h = width / 2 + p, height / 2 + p
        return Path([(cx - 2 * w, cy), (cx, cy + 2 * h), (cx + 2 * w, cy), (cx, cy - 2 * h),
                     (cx - 2 * w, cy)], closed=True)


_SHAPES = {
    'box': 'round,pad=0.55,rounding_size=0.25',
    'square': 'square,pad=0.55',
    'stadium': 'round,pad=0.55,rounding_size=0.95',
    'circle': 'circle,pad=0.35',
}


class DiagramsMixin:
    """Sankey, chord, Venn, flowchart, sequence_diagram (mélangés dans Chart)."""

    # ──────────────────────────────────────── Sankey

    def sankey(self, source: str, target: Optional[str] = None, value: Optional[str] = None,
               order: Optional[List[str]] = None, colors: Optional[Dict[str, str]] = None,
               node_width: float = 0.025, gap: float = 0.04, alpha: float = 0.4,
               show_values: bool = True, unit: str = '', **legacy) -> 'DiagramsMixin':
        """
        EN: Sankey diagram: flows between nodes on several levels, as curved ribbons whose
            width is proportional to `value`. One row per link (source → target); columns
            are deduced from the links. Table of links in self.sankey_table.
            The old form sankey('flows', 'labels', orientations=…) (matplotlib.sankey)
            still works.
        FR: Diagramme de Sankey : flux entre nœuds sur plusieurs niveaux, en rubans courbes
            d'épaisseur proportionnelle à `value`. Une ligne par lien (source → cible) ;
            les colonnes sont déduites des liens. Tableau des liens dans self.sankey_table.
            L'ancienne forme sankey('flux', 'etiquette', orientations=…) (matplotlib.sankey)
            reste acceptée.

        Args / Paramètres:
            source, target (str): EN/FR: Colonnes de départ et d'arrivée de chaque flux.
            value (str, optional): EN/FR: Colonne des quantités ; sans elle, chaque ligne compte 1.
            order (list, optional): EN/FR: Ordre vertical des nœuds (haut → bas) ; défaut :
                rangés pour limiter les croisements.
            colors (dict, optional): EN/FR: {nœud: couleur} ; défaut : palette du thème.
            node_width (float): EN/FR: Largeur des nœuds (fraction de la largeur). Défaut 0.025.
            gap (float): EN/FR: Espace entre nœuds d'une colonne (fraction). Défaut 0.04.
            alpha (float): EN/FR: Opacité des rubans. Défaut 0.4.
            show_values (bool): EN/FR: Affiche la quantité sous le nom de chaque nœud.
            unit (str): EN/FR: Unité ajoutée aux valeurs (' t', ' k€'…).

        Example / Exemple:
            Chart(flux).sankey('origine', 'destination', 'tonnes', unit=' t')
        """
        if target is None or pd.api.types.is_numeric_dtype(self.data[source]):
            return self._sankey_matplotlib(source, target, **legacy)
        self.chart_type = 'sankey'
        links = _links(self.data, source, target, value)
        nodes = _appearance(links['s'], links['t'])
        layer, _ = _layers(nodes, list(zip(links['s'], links['t'])))
        last = max(layer.values()) or 1
        for n in set(nodes) - set(links['s']):          # sorties alignées à droite
            layer[n] = last
        size = {n: max(links.loc[links['t'] == n, 'v'].sum(), links.loc[links['s'] == n, 'v'].sum())
                for n in nodes}
        seq = ([n for n in order if n in nodes] + [n for n in nodes if n not in order]) if order else nodes
        cols = [[n for n in seq if layer[n] == k] for k in range(last + 1)]
        scale = min((1 - gap * (len(c) - 1)) / sum(size[n] for n in c) for c in cols if c)
        top = {}
        for k, col in enumerate(cols):
            if k and not order:                          # sous leurs sources : moins de croisements
                def center(n):
                    inc = links[(links['t'] == n) & links['s'].isin(top)]
                    if inc.empty:
                        return 0.5
                    return np.average([top[s] - size[s] * scale / 2 for s in inc['s']], weights=inc['v'])
                col.sort(key=center, reverse=True)
            y = 0.5 + (sum(size[n] for n in col) * scale + gap * (len(col) - 1)) / 2
            for n in col:
                top[n] = y
                y -= size[n] * scale + gap
        x = {n: layer[n] / last * (1 - node_width) for n in nodes}
        palette = self._palette()
        color = {n: (colors or {}).get(n, palette[i % len(palette)]) for i, n in enumerate(nodes)}

        h = links['v'].to_numpy() * scale
        ys, yt = np.zeros(len(links)), np.zeros(len(links))
        cur = dict(top)
        for i in sorted(range(len(links)), key=lambda i: -top[links['t'][i]]):
            ys[i], cur[links['s'][i]] = cur[links['s'][i]], cur[links['s'][i]] - h[i]
        cur = dict(top)
        for i in sorted(range(len(links)), key=lambda i: -top[links['s'][i]]):
            yt[i], cur[links['t'][i]] = cur[links['t'][i]], cur[links['t'][i]] - h[i]
        for i, (s, t) in enumerate(zip(links['s'], links['t'])):
            xa, xb = x[s] + node_width, x[t]
            xm = (xa + xb) / 2
            verts = [(xa, ys[i]), (xm, ys[i]), (xm, yt[i]), (xb, yt[i]), (xb, yt[i] - h[i]),
                     (xm, yt[i] - h[i]), (xm, ys[i] - h[i]), (xa, ys[i] - h[i]), (xa, ys[i])]
            codes = [_M, _C4, _C4, _C4, _L, _C4, _C4, _C4, _CL]
            self.ax.add_patch(PathPatch(Path(verts, codes), facecolor=color[s], alpha=alpha,
                                        edgecolor='none', zorder=1))
        for n in nodes:
            hn = size[n] * scale
            self.ax.add_patch(Rectangle((x[n], top[n] - hn), node_width, hn, facecolor=color[n],
                                        edgecolor='none', zorder=2))
            text = n + (f'\n{size[n]:,.4g}{unit}'.replace(',', ' ') if show_values else '')
            # Étiquettes vers l'intérieur : rien ne dépasse du cadre, quelle que soit la taille
            if layer[n] == last:
                xt, ha = x[n] - 0.008, 'right'
            else:
                xt, ha = x[n] + node_width + 0.008, 'left'
            self.ax.text(xt, top[n] - hn / 2, text, ha=ha, va='center', zorder=3,
                         fontsize=plt.rcParams['font.size'] * 0.9, path_effects=self._halo())
        self.ax.set_xlim(-0.01, 1.01)
        self.ax.set_ylim(-0.01, 1.01)
        self.ax.set_axis_off()
        self.sankey_table = links.rename(columns={'s': source, 't': target, 'v': value or 'n'})
        return self

    def _sankey_matplotlib(self, flows: str, labels: str, orientations: Optional[str] = None,
                           unit: str = '', scale: float = 0.01) -> 'DiagramsMixin':
        """Ancienne forme : un seul nœud, flux entrants (+) et sortants (−), matplotlib.sankey."""
        from matplotlib.sankey import Sankey
        self.chart_type = 'sankey'
        self.ax.axis('off')
        orient = self.data[orientations].tolist() if orientations else [0] * len(self.data)
        Sankey(ax=self.ax, unit=unit, scale=scale).add(
            flows=self.data[flows].tolist(), labels=self.data[labels].tolist(),
            orientations=orient).finish()
        return self

    # ──────────────────────────────────────── Chord

    def chord(self, source: str, target: str, value: Optional[str] = None,
              order: Optional[List[str]] = None, colors: Optional[Dict[str, str]] = None,
              gap: float = 3.0, alpha: float = 0.65, labels: bool = True) -> 'DiagramsMixin':
        """
        EN: Chord diagram: nodes on a circle (arc length ∝ total flow), links as ribbons
            through the centre, coloured like their source. Matrix in self.chord_table.
        FR: Diagramme en cordes : nœuds sur un cercle (longueur d'arc ∝ flux total), liens
            en rubans passant par le centre, de la couleur de leur source. Matrice dans
            self.chord_table.

        Args / Paramètres:
            source, target (str): EN/FR: Colonnes des deux extrémités de chaque lien.
            value (str, optional): EN/FR: Quantité du lien ; sans elle, chaque ligne compte 1.
            order (list, optional): EN/FR: Ordre des nœuds autour du cercle.
            gap (float): EN/FR: Espace entre deux nœuds, en degrés. Défaut 3.
            alpha (float): EN/FR: Opacité des rubans. Défaut 0.65.

        Example / Exemple:
            Chart(echanges).chord('de', 'vers', 'tonnes')
        """
        self.chart_type = 'chord'
        links = _links(self.data, source, target, value)
        found = _appearance(links['s'], links['t'])
        nodes = ([n for n in order if n in found] + [n for n in found if n not in order]) if order else found
        idx = {n: i for i, n in enumerate(nodes)}
        total = {n: links.loc[links['s'] == n, 'v'].sum() + links.loc[links['t'] == n, 'v'].sum()
                 for n in nodes}
        k = (360 - gap * len(nodes)) / sum(total.values())   # degrés par unité
        palette = self._palette()
        color = {n: (colors or {}).get(n, palette[i % len(palette)]) for i, n in enumerate(nodes)}
        start, a = {}, 90.0
        for n in nodes:                                       # sens horaire depuis le haut
            start[n] = a
            a -= total[n] * k + gap
        # Extrémités de chaque lien sur l'arc de son nœud, rangées selon l'autre nœud
        ends = {n: [] for n in nodes}
        for i, (s, t) in enumerate(zip(links['s'], links['t'])):
            ends[s].append((i, 's', t))
            ends[t].append((i, 't', s))
        span = {}
        for n in nodes:
            cur = start[n]
            for i, side, other in sorted(ends[n], key=lambda e: (idx[e[2]] - idx[n]) % len(nodes),
                                         reverse=True):
                w = links['v'][i] * k
                span[(i, side)] = (cur, cur - w)
                cur -= w

        def arc(a0, a1, r=1.0):
            t = np.radians(np.linspace(a0, a1, max(int(abs(a1 - a0)) + 2, 3)))
            return list(zip(r * np.cos(t), r * np.sin(t)))

        r_in = 0.98
        for i, (s, t) in enumerate(zip(links['s'], links['t'])):
            (a0, a1), (b0, b1) = span[(i, 's')], span[(i, 't')]
            pa, pb = arc(a0, a1, r_in), arc(b0, b1, r_in)
            verts = pa + [(0, 0), pb[0]] + pb[1:] + [(0, 0), pa[0], pa[0]]
            codes = [_M] + [_L] * (len(pa) - 1) + [_C3, _C3] + [_L] * (len(pb) - 1) + [_C3, _C3, _CL]
            self.ax.add_patch(PathPatch(Path(verts, codes), facecolor=color[s], alpha=alpha,
                                        edgecolor='none', zorder=1))
        for n in nodes:
            a0, a1 = start[n] - total[n] * k, start[n]
            self.ax.add_patch(Wedge((0, 0), 1.08, a0, a1, width=0.08, facecolor=color[n],
                                    edgecolor='none', zorder=2))
            if labels:
                mid = (a0 + a1) / 2
                rad = np.radians(mid)
                left = np.cos(rad) < 0
                self.ax.text(1.13 * np.cos(rad), 1.13 * np.sin(rad), n, rotation=mid + (180 if left else 0),
                             rotation_mode='anchor', ha='right' if left else 'left', va='center')
        _fit(self.ax, -1.4, 1.4, -1.4, 1.4)    # place pour les noms autour du cercle
        self.ax.set_axis_off()
        self.chord_table = (links.pivot_table(index='s', columns='t', values='v', aggfunc='sum',
                                              fill_value=0)
                            .rename_axis(index=source, columns=target))
        return self

    # ──────────────────────────────────────── Venn

    def venn(self, sets: List[str], labels: Optional[List[str]] = None,
             colors: Optional[List[str]] = None, alpha: float = 0.35,
             show_values: bool = True) -> 'DiagramsMixin':
        """
        EN: Venn diagram of 2 or 3 sets, with the number of items in each region. Each set is
            a column: boolean (True = the row belongs to the set) or a list of items (empty
            cells ignored). Region counts in self.venn_table.
        FR: Diagramme de Venn de 2 ou 3 ensembles, avec le nombre d'éléments de chaque zone.
            Chaque ensemble est une colonne : booléenne (True = la ligne appartient à
            l'ensemble) ou liste d'éléments (cases vides ignorées). Effectifs dans
            self.venn_table.

        Args / Paramètres:
            sets (list): EN/FR: 2 ou 3 colonnes.
            labels (list, optional): EN/FR: Noms affichés (défaut : noms des colonnes).
            colors (list, optional): EN/FR: Couleurs des cercles (défaut : palette).

        Example / Exemple:
            Chart(enquete).venn(['maraichage', 'elevage', 'peche'])
        """
        if len(sets) not in (2, 3):
            raise ValueError('venn accepte 2 ou 3 ensembles')
        self.chart_type = 'venn'
        items = []
        for c in sets:
            col = self.data[c]
            items.append(set(self.data.index[col.fillna(False).astype(bool)])
                         if pd.api.types.is_bool_dtype(col) else set(col.dropna()))
        names = labels or list(sets)
        palette = colors or self._palette()
        if len(sets) == 2:
            centers = [(-0.55, 0.0), (0.55, 0.0)]
            spots = {(1, 0): (-1.0, 0.0), (0, 1): (1.0, 0.0), (1, 1): (0.0, 0.0)}
            name_xy = [(-0.75, 1.12), (0.75, 1.12)]
        else:
            centers = [(-0.55, 0.32), (0.55, 0.32), (0.0, -0.62)]
            spots = {(1, 0, 0): (-1.02, 0.55), (0, 1, 0): (1.02, 0.55), (0, 0, 1): (0.0, -1.2),
                     (1, 1, 0): (0.0, 0.85), (1, 0, 1): (-0.62, -0.38), (0, 1, 1): (0.62, -0.38),
                     (1, 1, 1): (0.0, 0.02)}
            name_xy = [(-0.95, 1.47), (0.95, 1.47), (0.0, -1.78)]
        for (cx, cy), c in zip(centers, palette):
            self.ax.add_patch(Circle((cx, cy), 1.0, facecolor=c, alpha=alpha, edgecolor='none'))
            self.ax.add_patch(Circle((cx, cy), 1.0, facecolor='none', edgecolor=c, linewidth=1.5))
        rows = []
        everything = set().union(*items)
        for mask, (tx, ty) in spots.items():
            inside = set(everything)
            for m, s in zip(mask, items):
                inside = inside & s if m else inside - s
            region = ' ∩ '.join(n for m, n in zip(mask, names) if m)
            rows.append({'zone': region, 'n': len(inside)})
            if show_values:
                self.ax.text(tx, ty, f'{len(inside)}', ha='center', va='center',
                             fontsize=plt.rcParams['font.size'] * 1.2, fontweight='bold')
        for (tx, ty), n, s, c in zip(name_xy, names, items, palette):
            self.ax.text(tx, ty, f'{n} ({len(s)})', ha='center', va='center', color=c,
                         fontweight='bold')
        lim = 1.7 if len(sets) == 2 else 2.0
        dy = 0.35 if len(sets) == 2 else 0
        _fit(self.ax, -lim, lim, -lim + dy, lim - dy)
        self.ax.set_axis_off()
        self.venn_table = pd.DataFrame(rows)
        return self

    # ──────────────────────────────────────── Logigrammes, organigrammes

    def flowchart(self, source: str, target: str, label: Optional[str] = None,
                  direction: str = 'TB', shapes: Optional[Dict[str, str]] = None,
                  colors: Optional[Dict[str, str]] = None,
                  fontsize: Optional[float] = None) -> 'DiagramsMixin':
        """
        EN: Flowchart / org chart, like a Mermaid "flowchart": one row per arrow
            (source → target), nodes placed automatically in levels. A node whose name ends
            with '?' is drawn as a decision (diamond). Layout in self.flow_layout.
        FR: Logigramme / organigramme, comme un « flowchart » Mermaid : une ligne par flèche
            (source → cible), nœuds placés automatiquement par niveaux. Un nœud dont le nom
            finit par « ? » est une décision (losange). Disposition dans self.flow_layout.

        Args / Paramètres:
            source, target (str): EN/FR: Colonnes de départ et d'arrivée des flèches.
            label (str, optional): EN/FR: Colonne du texte écrit sur les flèches ('oui', 'non'…).
            direction (str): EN/FR: 'TB' (de haut en bas, défaut) ou 'LR' (de gauche à droite).
            shapes (dict, optional): EN/FR: {nœud: forme} parmi 'box' (défaut), 'stadium'
                (début / fin), 'circle', 'diamond' (décision), 'square'.
            colors (dict, optional): EN/FR: {nœud: couleur}.
            fontsize (float, optional): EN/FR: Taille du texte des nœuds.

        Example / Exemple:
            Chart(etapes).flowchart('de', 'vers', label='condition', shapes={'Début': 'stadium'})
        """
        if direction.upper() not in ('TB', 'TD', 'LR'):
            raise ValueError("direction doit valoir 'TB' ou 'LR'")
        self.chart_type = 'flowchart'
        d = self.data
        edges = list(zip(d[source].astype(str), d[target].astype(str)))
        nodes = _appearance(d[source], d[target])
        layer, back = _layers(nodes, edges)
        n_layers = max(layer.values()) + 1
        pos = {}
        for k in range(n_layers):                  # ordre dans un niveau : sous les parents
            row = [n for n in nodes if layer[n] == k]
            if k:
                def bary(n):
                    p = [pos[a] for a, b in edges if b == n and a in pos and layer[a] < k]
                    return np.mean(p) if p else 1e9
                row.sort(key=bary)
            for i, n in enumerate(row):
                pos[n] = i - (len(row) - 1) / 2
        lr = direction.upper() == 'LR'
        xy = {n: ((layer[n], -pos[n]) if lr else (pos[n], -layer[n])) for n in nodes}

        palette = self._palette()
        bg = plt.rcParams['axes.facecolor']
        if not mcolors.is_color_like(bg) or bg == 'none':
            bg = 'white'
        fs = fontsize or plt.rcParams['font.size']
        kind_color = {'diamond': palette[1 % len(palette)], 'stadium': palette[2 % len(palette)],
                      'circle': palette[2 % len(palette)]}
        texts = {}
        for n in nodes:
            shape = (shapes or {}).get(n, 'diamond' if n.rstrip().endswith('?') else 'box')
            c = (colors or {}).get(n, kind_color.get(shape, palette[0]))
            style = _Diamond(0.3) if shape == 'diamond' else _SHAPES.get(shape, _SHAPES['box'])
            texts[n] = self.ax.text(*xy[n], n, ha='center', va='center', fontsize=fs, zorder=3,
                                    color=plt.rcParams['text.color'],
                                    bbox=dict(boxstyle=style, facecolor=_mix(c, bg, 0.82),
                                              edgecolor=c, linewidth=1.3))
        ink = plt.rcParams['axes.labelcolor']
        labels = d[label].tolist() if label else [None] * len(edges)
        # Retours : coude sur le côté (trait parallèle au sens du logigramme)
        back_style = f"bar,angle={0 if lr else 90},fraction=0.15"
        for (a, b), lab in zip(edges, labels):
            curved = (a, b) in back or a == b or layer[a] == layer[b]
            arrow = self.ax.annotate('', xy=xy[b], xytext=xy[a], zorder=4,
                                     arrowprops=dict(arrowstyle='-|>,head_length=0.5,head_width=0.25',
                                                     color=ink, lw=1.2, shrinkA=1, shrinkB=1,
                                                     patchA=texts[a].get_bbox_patch(),
                                                     patchB=texts[b].get_bbox_patch(),
                                                     connectionstyle=(back_style if curved else 'arc3')))
            if lab is not None and str(lab) not in ('', 'nan'):
                # Au milieu de la flèche ; courbe : au sommet, du côté où elle se bombe
                (xa, ya), (xb, yb) = xy[a], xy[b]
                nx, ny = (yb - ya, xa - xb) if curved else (0, 0)
                norm = np.hypot(nx, ny) or 1
                self.ax.annotate(str(lab), xy=(0.5 + 0.5 * nx / norm, 0.5 + 0.5 * ny / norm),
                                 xycoords=arrow.arrow_patch, ha='center', va='center',
                                 fontsize=fs * 0.85, zorder=5, color=ink,
                                 bbox=dict(boxstyle='round,pad=0.2', facecolor=bg, edgecolor='none'))
        xs, ys = zip(*xy.values())
        self.ax.set_xlim(min(xs) - 0.7, max(xs) + 0.7)
        self.ax.set_ylim(min(ys) - 0.6, max(ys) + 0.6)
        self.ax.set_axis_off()
        self.flow_layout = pd.DataFrame({'noeud': nodes, 'niveau': [layer[n] for n in nodes],
                                         'x': [xy[n][0] for n in nodes], 'y': [xy[n][1] for n in nodes]})
        return self

    # ──────────────────────────────────────── Diagramme de séquence

    def sequence_diagram(self, sender: str, receiver: str, message: str,
                         participants: Optional[List[str]] = None, dashed: Optional[str] = None,
                         numbered: bool = False, fontsize: Optional[float] = None) -> 'DiagramsMixin':
        """
        EN: Sequence diagram (like Mermaid "sequenceDiagram"): participants as columns with
            lifelines, one arrow per row in time order, top to bottom.
        FR: Diagramme de séquence (comme un « sequenceDiagram » Mermaid) : participants en
            colonnes avec leur ligne de vie, une flèche par ligne, dans l'ordre, de haut en bas.

        Args / Paramètres:
            sender, receiver, message (str): EN/FR: Colonnes émetteur, destinataire, message.
            participants (list, optional): EN/FR: Ordre des colonnes (défaut : ordre d'apparition).
            dashed (str, optional): EN/FR: Colonne booléenne : True = flèche en pointillés (réponse).
            numbered (bool): EN/FR: Numérote les messages.

        Example / Exemple:
            Chart(echanges).sequence_diagram('de', 'vers', 'message', dashed='reponse')
        """
        self.chart_type = 'sequence_diagram'
        d = self.data
        parts = participants or _appearance(d[sender], d[receiver])
        X = {p: i for i, p in enumerate(parts)}
        n = len(d)
        palette = self._palette()
        bg = plt.rcParams['axes.facecolor']
        if not mcolors.is_color_like(bg) or bg == 'none':
            bg = 'white'
        ink = plt.rcParams['axes.labelcolor']
        fs = fontsize or plt.rcParams['font.size']
        for p, x in X.items():
            c = palette[x % len(palette)]
            self.ax.plot([x, x], [0, -(n + 1)], color=c, lw=1, ls=(0, (4, 3)), alpha=0.7, zorder=1)
            for y in (0, -(n + 1)):
                self.ax.text(x, y, p, ha='center', va='center', fontsize=fs, zorder=3,
                             fontweight='bold',
                             bbox=dict(boxstyle=_SHAPES['box'], facecolor=_mix(c, bg, 0.82),
                                       edgecolor=c, linewidth=1.3))
        dash = d[dashed].fillna(False).astype(bool).tolist() if dashed else [False] * n
        for i, (s, r, m, dsh) in enumerate(zip(d[sender].astype(str), d[receiver].astype(str),
                                               d[message].astype(str), dash)):
            y = -(i + 1)
            text = f'{i + 1}. {m}' if numbered else m
            style = dict(arrowstyle='-|>,head_length=0.6,head_width=0.3', color=ink, lw=1.3,
                         linestyle=(0, (4, 3)) if dsh else '-', shrinkA=0, shrinkB=0)
            if s == r:                                   # message à soi-même : boucle à droite
                x = X[s]
                self.ax.plot([x, x + 0.22, x + 0.22], [y + 0.18, y + 0.18, y - 0.18], color=ink,
                             lw=1.3, ls=style['linestyle'], zorder=4, solid_capstyle='butt')
                self.ax.annotate('', xy=(x, y - 0.18), xytext=(x + 0.22, y - 0.18), zorder=4,
                                 arrowprops=style)
                self.ax.text(x + 0.27, y, text, ha='left', va='center', fontsize=fs * 0.9)
            else:
                self.ax.annotate('', xy=(X[r], y), xytext=(X[s], y), zorder=4, arrowprops=style)
                self.ax.text((X[s] + X[r]) / 2, y + 0.08, text, ha='center', va='bottom',
                             fontsize=fs * 0.9, zorder=5)
        self.ax.set_xlim(-0.6, len(parts) - 0.4)
        self.ax.set_ylim(-(n + 1.6), 0.6)
        self.ax.set_axis_off()
        return self
