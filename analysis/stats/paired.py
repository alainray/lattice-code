"""Tests pareados sobre diferencias por (dataset, semilla). Sin scipy."""
import itertools
import numpy as np


def _ranks(v):
    """Rangos con promedio en los empates."""
    v = np.asarray(v, float)
    o = np.argsort(v, kind="mergesort")
    r = np.empty(len(v))
    i = 0
    while i < len(v):
        j = i
        while j + 1 < len(v) and v[o[j + 1]] == v[o[i]]:
            j += 1
        r[o[i:j + 1]] = (i + j) / 2.0 + 1
        i = j + 1
    return r


def wilcoxon(d):
    """Signed-rank de Wilcoxon, dos colas. Exacto hasta n=20, normal despues.

    Se descartan los ceros (Wilcoxon clasico) y se corrige la varianza por
    empates en |d|.
    """
    d = np.asarray([x for x in d if x != 0], float)
    n = len(d)
    if n == 0:
        return float("nan"), float("nan")
    r = _ranks(np.abs(d))
    W = float(r[d > 0].sum())
    if n <= 20:
        # enumeracion exacta de los 2^n signos posibles bajo H0
        hits = 0
        count = 0
        obs = abs(W - n * (n + 1) / 4.0)
        for signs in itertools.product([0, 1], repeat=n):
            w = float(sum(ri for ri, s in zip(r, signs) if s))
            count += 1
            if abs(w - n * (n + 1) / 4.0) >= obs - 1e-12:
                hits += 1
        return W, hits / count
    mu = n * (n + 1) / 4.0
    _, cnt = np.unique(r, return_counts=True)
    var = n * (n + 1) * (2 * n + 1) / 24.0 - (cnt ** 3 - cnt).sum() / 48.0
    z = (abs(W - mu) - 0.5) / np.sqrt(var)
    from math import erf, sqrt
    return W, 2 * (1 - 0.5 * (1 + erf(z / sqrt(2))))


def sign_flip(d, n_perm=20000, seed=0):
    """Test de permutacion por cambio de signo sobre la media (dos colas)."""
    d = np.asarray(d, float)
    rng = np.random.default_rng(seed)
    obs = abs(d.mean())
    flips = rng.choice([-1.0, 1.0], size=(n_perm, len(d)))
    null = np.abs((flips * d).mean(axis=1))
    return (1 + (null >= obs - 1e-12).sum()) / (n_perm + 1)


def boot_ci(d, groups=None, n_boot=20000, seed=0, alpha=0.05):
    """IC percentil de la media. Con groups remuestrea datasets (cluster),
    que es el nivel al que se quiere generalizar; sin groups, pares sueltos."""
    d = np.asarray(d, float)
    rng = np.random.default_rng(seed)
    if groups is None:
        idx = rng.integers(0, len(d), size=(n_boot, len(d)))
        means = d[idx].mean(axis=1)
    else:
        groups = np.asarray(groups)
        uniq = np.unique(groups)
        by = [d[groups == g] for g in uniq]
        means = np.empty(n_boot)
        for b in range(n_boot):
            pick = rng.integers(0, len(uniq), size=len(uniq))
            vals = np.concatenate([by[i][rng.integers(0, len(by[i]), size=len(by[i]))] for i in pick])
            means[b] = vals.mean()
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def holm(pvals):
    """Correccion de Holm-Bonferroni; devuelve los p ajustados en el orden dado."""
    idx = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    run = 0.0
    for k, i in enumerate(idx):
        run = max(run, (m - k) * pvals[i])
        adj[i] = min(run, 1.0)
    return adj


def _betacf(a, b, x, itmax=200, eps=3e-16):
    """Fraccion continua de la beta incompleta (Numerical Recipes)."""
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    if abs(d) < 1e-300:
        d = 1e-300
    d = 1.0 / d
    h = d
    for m in range(1, itmax + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-300:
            d = 1e-300
        c = 1.0 + aa / c
        if abs(c) < 1e-300:
            c = 1e-300
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-300:
            d = 1e-300
        c = 1.0 + aa / c
        if abs(c) < 1e-300:
            c = 1e-300
        d = 1.0 / d
        de = d * c
        h *= de
        if abs(de - 1.0) < eps:
            break
    return h


def _betai(a, b, x):
    """Beta incompleta regularizada I_x(a, b)."""
    from math import exp, lgamma
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    bt = exp(lgamma(a + b) - lgamma(a) - lgamma(b) + a * np.log(x) + b * np.log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def ttest_rel(d):
    """t de Student pareado (una muestra sobre las diferencias), dos colas."""
    d = np.asarray(d, float)
    n = len(d)
    sd = d.std(ddof=1)
    if sd == 0:
        return float("inf") if d.mean() else 0.0, 0.0 if d.mean() else 1.0, n - 1
    t = d.mean() / (sd / np.sqrt(n))
    df = n - 1
    p = _betai(df / 2.0, 0.5, df / (df + t * t))
    return float(t), float(p), df
