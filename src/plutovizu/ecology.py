"""Calculs pour l'écologie : GLM, lissage, ordination (AFC, CCA, RDA), diversité.

Utilisables seuls (sans graphique) :

    from plutovizu.ecology import glm_fit, ordinate, diversity_indices
    diversity_indices(df[especes])          # richesse, Shannon, Simpson… par site

Les méthodes de Chart (glm_curve, response_curve, ordination, diversity…) s'appuient
dessus. Tout repose sur numpy et scipy.
"""
import warnings
from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import gammaln

# ── Modèles linéaires généralisés (IRLS) ────────────────────────────────────

FAMILIES = ('binomial', 'poisson')


def _inv_link(eta, family):
    if family == 'binomial':
        return 1 / (1 + np.exp(-np.clip(eta, -30, 30)))
    return np.exp(np.clip(eta, -30, 30))


def glm_fit(X: np.ndarray, y: np.ndarray, family: str = 'binomial',
            max_iter: int = 100, tol: float = 1e-9) -> Dict:
    """Ajuste un GLM à lien canonique (logit pour 'binomial', log pour 'poisson') par
    moindres carrés pondérés itératifs. X doit contenir la colonne de l'intercept.

    Retourne coefficients, matrice de covariance, déviances, AIC, D² (part de déviance
    expliquée) et la p-value du test du rapport de vraisemblance contre le modèle nul.
    """
    if family not in FAMILIES:
        raise ValueError(f"family doit valoir {FAMILIES}, pas '{family}'")
    X, y = np.asarray(X, float), np.asarray(y, float)
    if family == 'binomial' and not np.isin(y, (0, 1)).all():
        raise ValueError("family='binomial' attend une réponse 0/1 (absence/présence)")
    if family == 'poisson' and (y < 0).any():
        raise ValueError("family='poisson' attend des comptages positifs ou nuls")
    mu = (y + 0.5) / 2 if family == 'binomial' else y + 0.5
    eta = np.log(mu / (1 - mu)) if family == 'binomial' else np.log(mu)
    beta = np.zeros(X.shape[1])
    converged = False
    for _ in range(max_iter):
        mu = _inv_link(eta, family)
        var = np.clip(mu * (1 - mu), 1e-10, None) if family == 'binomial' else np.clip(mu, 1e-10, None)
        z = eta + (y - mu) / var                    # lien canonique : dμ/dη = var(μ)
        XtW = X.T * var
        new = np.linalg.solve(XtW @ X, XtW @ z)
        eta = X @ new
        if np.max(np.abs(new - beta)) < tol:
            beta, converged = new, True
            break
        beta = new
    if not converged:
        warnings.warn("Le GLM n'a pas convergé (séparation parfaite ou données trop rares ?)",
                      stacklevel=3)
    mu = _inv_link(X @ beta, family)
    var = np.clip(mu * (1 - mu), 1e-10, None) if family == 'binomial' else np.clip(mu, 1e-10, None)
    cov = np.linalg.inv((X.T * var) @ X)

    def deviance(m):
        if family == 'binomial':
            m = np.clip(m, 1e-12, 1 - 1e-12)
            return -2 * np.sum(y * np.log(m) + (1 - y) * np.log(1 - m))
        with np.errstate(divide='ignore', invalid='ignore'):
            term = np.where(y > 0, y * np.log(y / m), 0.0)
        return 2 * np.sum(term - (y - m))

    def loglik(m):
        if family == 'binomial':
            m = np.clip(m, 1e-12, 1 - 1e-12)
            return np.sum(y * np.log(m) + (1 - y) * np.log(1 - m))
        return np.sum(y * np.log(np.clip(m, 1e-300, None)) - m - gammaln(y + 1))

    dev, null_dev = deviance(mu), deviance(np.full_like(y, y.mean()))
    df = X.shape[1] - 1
    return {'family': family, 'coefs': beta, 'cov': cov, 'se': np.sqrt(np.diag(cov)),
            'deviance': dev, 'null_deviance': null_dev,
            'd2': 1 - dev / null_dev if null_dev > 0 else np.nan,
            'aic': -2 * loglik(mu) + 2 * X.shape[1],
            'p_value': stats.chi2.sf(null_dev - dev, df) if df > 0 else np.nan,
            'converged': converged, 'n': len(y)}


# ── Lissage LOWESS (régression locale pondérée) ─────────────────────────────

def lowess(x: np.ndarray, y: np.ndarray, x_eval: np.ndarray, frac: float = 0.5) -> np.ndarray:
    """Régression linéaire locale (noyau tricube sur les `frac` × n plus proches voisins)."""
    x, y, x_eval = (np.asarray(a, float) for a in (x, y, x_eval))
    k = int(np.clip(np.ceil(frac * len(x)), 3, len(x)))
    D = np.abs(x[None, :] - x_eval[:, None])
    h = np.partition(D, k - 1, axis=1)[:, k - 1][:, None]
    W = np.clip(1 - (D / np.where(h > 0, h, 1e-12)) ** 3, 0, None) ** 3
    sw = W.sum(axis=1, keepdims=True)
    xm, ym = (W @ x)[:, None] / sw, (W @ y)[:, None] / sw
    sxx = (W * (x[None, :] - xm) ** 2).sum(axis=1)
    sxy = (W * (x[None, :] - xm) * (y[None, :] - ym)).sum(axis=1)
    slope = np.where(sxx > 1e-12, sxy / np.where(sxx > 1e-12, sxx, 1), 0.0)
    return ym[:, 0] + slope * (x_eval - xm[:, 0])


# ── Ordination : AFC, CCA, ACP, RDA ─────────────────────────────────────────

ORDINATIONS = ('ca', 'cca', 'pca', 'rda')


def ordinate(Y: np.ndarray, X: Optional[np.ndarray] = None, method: str = 'ca',
             scale: bool = False) -> Dict:
    """Ordination d'un tableau sites × espèces Y, éventuellement contrainte par des
    variables environnementales X (sites × variables).

    - 'ca'  : analyse factorielle des correspondances (AFC) ;
    - 'cca' : analyse canonique des correspondances (réponses unimodales) ;
    - 'pca' : ACP du tableau d'espèces ;
    - 'rda' : analyse de redondance (réponses linéaires).

    Échelle symétrique (sites et espèces multipliés par la racine des valeurs propres).
    Retourne valeurs propres, part d'inertie de chaque axe, part contrainte, scores des
    sites, des espèces et corrélations des variables avec les axes (flèches).
    """
    if method not in ORDINATIONS:
        raise ValueError(f"method doit valoir {ORDINATIONS}, pas '{method}'")
    if method in ('cca', 'rda') and X is None:
        raise ValueError(f"method='{method}' demande des variables environnementales (env=...)")
    Y = np.asarray(Y, float)
    n = Y.shape[0]
    Xs = None
    if method in ('ca', 'cca'):
        if (Y < 0).any():
            raise ValueError("L'AFC/CCA demande des abondances positives ou nulles")
        P = Y / Y.sum()
        r, c = P.sum(axis=1), P.sum(axis=0)
        if (r == 0).any() or (c == 0).any():
            raise ValueError("Un site ou une espèce a une abondance totale nulle : à retirer")
        Q = (P - np.outer(r, c)) / np.sqrt(np.outer(r, c))
        total = float((Q ** 2).sum())
        M = Q
        if method == 'cca':
            X = np.asarray(X, float)
            xm = r @ X
            sd = np.sqrt(r @ (X - xm) ** 2)
            Xs = (X - xm) / np.where(sd > 0, sd, 1)
            Xw = np.sqrt(r)[:, None] * Xs
            M = Xw @ np.linalg.lstsq(Xw, Q, rcond=None)[0]
        U, S, Vt = np.linalg.svd(M, full_matrices=False)
        keep = S > 1e-10
        U, S, Vt = U[:, keep], S[keep], Vt[keep]
        sites = U / np.sqrt(r)[:, None] * np.sqrt(S)
        species = Vt.T / np.sqrt(c)[:, None] * np.sqrt(S)
        weights = r
    else:
        Yc = Y - Y.mean(axis=0)
        if scale:
            Yc = Yc / np.where(Yc.std(axis=0, ddof=1) > 0, Yc.std(axis=0, ddof=1), 1)
        Yc = Yc / np.sqrt(n - 1)
        total = float((Yc ** 2).sum())
        M = Yc
        if method == 'rda':
            X = np.asarray(X, float)
            Xs = (X - X.mean(axis=0)) / np.where(X.std(axis=0) > 0, X.std(axis=0), 1)
            M = Xs @ np.linalg.lstsq(Xs, Yc, rcond=None)[0]
        U, S, Vt = np.linalg.svd(M, full_matrices=False)
        keep = S > 1e-10
        U, S, Vt = U[:, keep], S[keep], Vt[keep]
        sites = U * np.sqrt(S) * np.sqrt(n - 1)
        species = Vt.T * np.sqrt(S)
        weights = np.full(n, 1 / n)
    eig = S ** 2
    env = None
    if Xs is not None:   # corrélations (pondérées) variables ↔ scores des sites : les flèches
        w = weights / weights.sum()
        a = sites - w @ sites
        b = Xs - w @ Xs
        cov = (b * w[:, None]).T @ a
        env = cov / np.sqrt(np.outer(w @ b ** 2, w @ a ** 2))
    return {'method': method, 'eigenvalues': eig, 'explained': eig / total,
            'total_inertia': total, 'constrained': float(eig.sum() / total),
            'sites': sites, 'species': species, 'env': env}


# ── Diversité ───────────────────────────────────────────────────────────────

def diversity_indices(table: pd.DataFrame) -> pd.DataFrame:
    """Indices par site (lignes) d'un tableau sites × espèces d'abondances :
    richesse S, Shannon H' (logarithme népérien), Simpson 1 − Σp², inverse de Simpson
    1/Σp² et équitabilité de Pielou J = H'/ln S."""
    A = table.to_numpy(float)
    if (A < 0).any():
        raise ValueError("Les abondances doivent être positives ou nulles")
    tot = A.sum(axis=1, keepdims=True)
    p = np.divide(A, tot, out=np.zeros_like(A), where=tot > 0)
    with np.errstate(divide='ignore', invalid='ignore'):
        H = -np.nansum(np.where(p > 0, p * np.log(p), 0.0), axis=1)
    H = H + 0.0          # −0.0 → 0.0 pour un site à une seule espèce
    S = (A > 0).sum(axis=1)
    D = (p ** 2).sum(axis=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        J = np.where(S > 1, H / np.log(np.where(S > 1, S, 2)), np.nan)
        inv = np.where(D > 0, 1 / np.where(D > 0, D, 1), np.nan)
    return pd.DataFrame({'richesse': S, 'shannon': H, 'simpson': 1 - D,
                         'inv_simpson': inv, 'pielou': J}, index=table.index)


def rarefaction(counts: np.ndarray, sizes: np.ndarray) -> np.ndarray:
    """Richesse attendue E[S_n] dans un sous-échantillon de n individus (Hurlbert) :
    Σ_i [1 − C(N − N_i, n) / C(N, n)]."""
    counts = np.asarray(counts, float)
    counts = counts[counts > 0]
    N = counts.sum()
    out = []
    for n in np.asarray(sizes, float):
        if n > N:
            out.append(np.nan)
            continue
        log_c = gammaln(N - counts + 1) - gammaln(N - counts - n + 1) \
            - (gammaln(N + 1) - gammaln(N - n + 1))
        ratio = np.where(N - counts >= n, np.exp(np.where(N - counts >= n, log_c, 0)), 0.0)
        out.append(float(np.sum(1 - ratio)))
    return np.array(out)


def accumulation(presence: np.ndarray, n_perm: int = 100, random_state: int = 0):
    """Courbe d'accumulation des espèces (sites ajoutés dans un ordre aléatoire) :
    richesse cumulée moyenne et écart-type sur `n_perm` permutations, et estimateur
    Chao2 de la richesse totale."""
    PA = np.asarray(presence) > 0
    rng = np.random.default_rng(random_state)
    # pour chaque permutation : espèces déjà vues après 1, 2, … sites
    curves = np.array([np.logical_or.accumulate(PA[rng.permutation(len(PA))], axis=0).sum(axis=1)
                       for _ in range(n_perm)])
    freq = PA.sum(axis=0)
    s_obs = int((freq > 0).sum())
    q1, q2 = int((freq == 1).sum()), int((freq == 2).sum())
    m = len(PA)
    chao2 = s_obs + ((m - 1) / m) * (q1 ** 2 / (2 * q2) if q2 > 0 else q1 * (q1 - 1) / 2)
    return curves.mean(axis=0), curves.std(axis=0), {'s_obs': s_obs, 'chao2': float(chao2),
                                                      'uniques': q1, 'duplicates': q2}
