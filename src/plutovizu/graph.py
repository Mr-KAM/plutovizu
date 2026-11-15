"""Graphes et réseaux : plus court chemin, arbre couvrant minimal, flot maximal, chemin
critique (PERT/CPM), tournée du voyageur de commerce et disposition des nœuds.

Un graphe est une liste d'arêtes (DataFrame : une ligne par arête, colonnes source,
cible et éventuellement poids). Calculs : scipy.sparse.csgraph (aucune dépendance en plus).
"""
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import csgraph, csr_matrix


def _index(edges: pd.DataFrame, source: str, target: str):
    """Liste des nœuds (ordre d'apparition) et indices source / cible de chaque arête."""
    nodes = list(pd.unique(pd.concat([edges[source], edges[target]], ignore_index=True)))
    idx = {n: k for k, n in enumerate(nodes)}
    return nodes, edges[source].map(idx).to_numpy(), edges[target].map(idx).to_numpy()


def _weights(edges: pd.DataFrame, weight: Optional[str]) -> np.ndarray:
    w = edges[weight].to_numpy(float) if weight else np.ones(len(edges))
    if (w <= 0).any() or np.isnan(w).any():
        raise ValueError(f'Les poids ({weight}) doivent être strictement positifs')
    return w


def _matrix(n, i, j, w, directed: bool, how: str = 'min') -> csr_matrix:
    """Matrice d'adjacence creuse ; arêtes en double : plus petit poids (ou somme)."""
    d = pd.DataFrame({'i': i, 'j': j, 'w': w})
    if not directed:
        d = pd.concat([d, d.rename(columns={'i': 'j', 'j': 'i'})], ignore_index=True)
    d = d[d['i'] != d['j']].groupby(['i', 'j'], as_index=False)['w'].agg(how)
    return csr_matrix((d['w'], (d['i'], d['j'])), shape=(n, n))


def shortest_path(edges: pd.DataFrame, source: str, target: str, start, end,
                  weight: Optional[str] = None, directed: bool = False) -> Tuple[list, float]:
    """
    EN: Shortest path (Dijkstra) from `start` to `end`. Returns (list of nodes, length).
    FR: Plus court chemin (Dijkstra) de `start` à `end`. Retourne (liste des nœuds, longueur).

    Example / Exemple:
        chemin, longueur = shortest_path(routes, 'de', 'vers', 'Dakar', 'Kédougou', weight='km')
    """
    nodes, i, j = _index(edges, source, target)
    for n in (start, end):
        if n not in nodes:
            raise ValueError(f'Nœud inconnu : {n!r}')
    s, e = nodes.index(start), nodes.index(end)
    dist, pred = csgraph.dijkstra(_matrix(len(nodes), i, j, _weights(edges, weight), directed),
                                  directed=True, indices=s, return_predecessors=True)
    if np.isinf(dist[e]):
        raise ValueError(f'Aucun chemin de {start!r} à {end!r}')
    path = [e]
    while path[-1] != s:
        path.append(pred[path[-1]])
    return [nodes[k] for k in reversed(path)], float(dist[e])


def spanning_tree(edges: pd.DataFrame, source: str, target: str,
                  weight: Optional[str] = None) -> Tuple[pd.DataFrame, float]:
    """
    EN: Minimum spanning tree (forest if the graph is disconnected): the cheapest set of
        edges connecting every node. Returns (tree edges, total weight).
    FR: Arbre couvrant minimal (forêt si le graphe n'est pas connexe) : l'ensemble d'arêtes
        le moins coûteux reliant tous les nœuds. Retourne (arêtes de l'arbre, poids total).

    Example / Exemple:
        arbre, cout = spanning_tree(liaisons, 'de', 'vers', weight='cout')
    """
    nodes, i, j = _index(edges, source, target)
    tree = csgraph.minimum_spanning_tree(_matrix(len(nodes), i, j, _weights(edges, weight), False)).tocoo()
    out = pd.DataFrame({source: [nodes[k] for k in tree.row], target: [nodes[k] for k in tree.col],
                        weight or 'poids': tree.data})
    return out, float(tree.data.sum())


def max_flow(edges: pd.DataFrame, source: str, target: str, capacity: str,
             start, end) -> Tuple[float, pd.DataFrame]:
    """
    EN: Maximum flow from `start` to `end` in a directed network with edge capacities.
        Returns (flow value, edges with a `flux` column).
    FR: Flot maximal de `start` vers `end` dans un réseau orienté à capacités.
        Retourne (valeur du flot, arêtes avec une colonne `flux`).

    Example / Exemple:
        valeur, flux = max_flow(reseau, 'de', 'vers', 'capacite', 'Source', 'Puits')
    """
    nodes, i, j = _index(edges, source, target)
    cap = _weights(edges, capacity)
    # scipy veut des capacités entières : facteur 1000 pour les décimales (précision 0,001)
    scale = 1 if np.allclose(cap, np.round(cap)) else 1000
    m = _matrix(len(nodes), i, j, np.round(cap * scale), True, how='sum').astype(np.int64)
    res = csgraph.maximum_flow(m, nodes.index(start), nodes.index(end))
    flow = res.flow.toarray()
    pair_cap = m.toarray()
    out = edges.copy()
    # arêtes en double (même sens) : flux réparti au prorata des capacités
    out['flux'] = [max(flow[a, b], 0) / pair_cap[a, b] * c for a, b, c in zip(i, j, cap)]
    return res.flow_value / scale, out


def _predecessors(value) -> List[str]:
    if isinstance(value, (list, tuple, set)):
        return [str(v) for v in value]
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    return [p for p in (s.strip() for s in re.split(r'[,;]', str(value))) if p]


def critical_path(tasks: pd.DataFrame, task: str, duration: str,
                  predecessors: str) -> Tuple[pd.DataFrame, float]:
    """
    EN: Critical path method (CPM / PERT): earliest and latest start and finish, total float
        and critical tasks (zero float). `predecessors`: column of lists or strings 'A, B'.
        Returns (schedule indexed by task, project duration).
    FR: Méthode du chemin critique (CPM / PERT) : dates de début et de fin au plus tôt et au
        plus tard, marge totale, tâches critiques (marge nulle). `predecessors` : colonne de
        listes ou de textes 'A, B'. Retourne (planning indexé par tâche, durée du projet).

    Example / Exemple:
        planning, duree = critical_path(taches, 'tache', 'duree', 'avant')
    """
    names = tasks[task].astype(str).tolist()
    dur = dict(zip(names, tasks[duration].astype(float)))
    pred = {n: _predecessors(p) for n, p in zip(names, tasks[predecessors])}
    for n, ps in pred.items():
        unknown = set(ps) - set(names)
        if unknown:
            raise ValueError(f'Prédécesseur(s) inconnu(s) de {n!r} : {sorted(unknown)}')
    order, done = [], set()
    while len(order) < len(names):                       # tri topologique (Kahn)
        ready = [n for n in names if n not in done and set(pred[n]) <= done]
        if not ready:
            raise ValueError('Dépendances circulaires entre les tâches')
        order += ready
        done |= set(ready)
    es, ef = {}, {}
    for n in order:
        es[n] = max((ef[p] for p in pred[n]), default=0.0)
        ef[n] = es[n] + dur[n]
    total = max(ef.values())
    succ = {n: [m for m in names if n in pred[m]] for n in names}
    ls, lf = {}, {}
    for n in reversed(order):
        lf[n] = min((ls[m] for m in succ[n]), default=total)
        ls[n] = lf[n] - dur[n]
    out = pd.DataFrame({'duree': dur, 'debut_tot': es, 'fin_tot': ef, 'debut_tard': ls,
                        'fin_tard': lf}).loc[order]
    out['marge'] = out['debut_tard'] - out['debut_tot']
    out['critique'] = np.isclose(out['marge'], 0)
    out['predecesseurs'] = [pred[n] for n in order]
    return out, float(total)


def tsp(xy: np.ndarray, start: int = 0, closed: bool = True) -> Tuple[List[int], float]:
    """
    EN: Traveling salesman tour: nearest-neighbour start improved by 2-opt (heuristic, near
        optimal). Euclidean distances. Returns (visit order as indices, length).
    FR: Tournée du voyageur de commerce : plus proche voisin amélioré par 2-opt (heuristique,
        proche de l'optimum). Distances euclidiennes. Retourne (ordre de visite, longueur).

    Example / Exemple:
        ordre, longueur = tsp(points[['x', 'y']].to_numpy())
    """
    # ponytail: 2-opt en O(n²) par passe, bien jusqu'à quelques centaines de points
    xy = np.asarray(xy, float)
    n = len(xy)
    d = np.linalg.norm(xy[:, None] - xy[None], axis=2)
    tour, left = [start], set(range(n)) - {start}
    while left:
        nxt = min(left, key=lambda k: d[tour[-1], k])
        tour.append(nxt)
        left.remove(nxt)
    improved = True
    while improved:
        improved = False
        for a in range(1, n - 1):
            for b in range(a + 1, n if closed else n - 1):
                c, e = tour[b], tour[(b + 1) % n]
                if d[tour[a - 1], tour[a]] + d[c, e] > d[tour[a - 1], c] + d[tour[a], e] + 1e-12:
                    tour[a:b + 1] = tour[a:b + 1][::-1]
                    improved = True
    length = sum(d[tour[k], tour[k + 1]] for k in range(n - 1)) + (d[tour[-1], tour[0]] if closed else 0)
    return tour, float(length)


def layout(nodes: list, i, j, kind='spring', seed: int = 0) -> Dict:
    """
    EN: Node positions: 'spring' (force-directed, Fruchterman-Reingold), 'circular', or a
        dict {node: (x, y)} (e.g. real coordinates) returned as is.
    FR: Positions des nœuds : 'spring' (forces, Fruchterman-Reingold), 'circular', ou un
        dictionnaire {nœud: (x, y)} (ex. coordonnées réelles) renvoyé tel quel.
    """
    if isinstance(kind, dict):
        missing = set(nodes) - set(kind)
        if missing:
            raise ValueError(f'Positions manquantes pour : {sorted(map(str, missing))}')
        return {n: tuple(kind[n]) for n in nodes}
    n = len(nodes)
    angle = 2 * np.pi * np.arange(n) / max(n, 1)
    pos = np.column_stack([np.cos(angle), np.sin(angle)])
    if kind == 'spring' and n > 2:
        rng = np.random.default_rng(seed)
        pos = pos + rng.normal(0, 0.05, pos.shape)
        k, temp = 1 / np.sqrt(n), 0.1
        for _ in range(300):
            delta = pos[:, None] - pos[None]
            dist = np.maximum(np.linalg.norm(delta, axis=2), 1e-3)
            disp = (delta / dist[..., None] * (k * k / dist)[..., None]).sum(axis=1)   # répulsion
            dv = pos[i] - pos[j]
            dl = np.maximum(np.linalg.norm(dv, axis=1), 1e-3)[:, None]
            pull = dv / dl * dl ** 2 / k                                            # attraction
            np.add.at(disp, i, -pull)
            np.add.at(disp, j, pull)
            step = np.maximum(np.linalg.norm(disp, axis=1), 1e-9)[:, None]
            pos += disp / step * np.minimum(step, temp)
            temp *= 0.985
    elif kind not in ('spring', 'circular'):
        raise ValueError("layout doit valoir 'spring', 'circular' ou un dictionnaire {nœud: (x, y)}")
    return {nd: tuple(p) for nd, p in zip(nodes, pos)}
