"""rho_(I) por j, y su relacion con el accuracy de test."""
import csv, glob, json, os
from collections import defaultdict
import numpy as np

def spearmanr(a, b):
    """Spearman sin scipy: Pearson sobre los rangos (promediando empates)."""
    def rank(v):
        v = np.asarray(v, dtype=float)
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
    ra, rb = rank(a), rank(b)
    n = len(ra)
    rs = float(np.corrcoef(ra, rb)[0, 1])
    # p aproximado por el estadistico t habitual
    if n > 2 and abs(rs) < 1:
        t = rs * np.sqrt((n - 2) / (1 - rs ** 2))
        from math import erf, sqrt
        p = 2 * (1 - 0.5 * (1 + erf(abs(t) / sqrt(2))))
    else:
        p = float("nan")
    return rs, p


RD = os.path.expanduser("~/results/rho")
FAMS = [("resnet18", "ResNet-18"), ("ain", "AIN"), ("ed", "ED")]
METHODS = ["Base", "+ CRM", "+ LATTICE", "+ CRM + LATTICE"]
DS = ["cars3d", "dsprites", "mpi3d", "clevr", "iraven", "shapes3d"]

key_of = {r["run_dir"].rstrip("/"): (r["family"], r["method"], r["dataset"], r["seed"])
          for r in csv.DictReader(open(os.path.join(RD, "runs.csv"))) if r["run_dir"]}

def test_acc(run_dir):
    p = os.path.join(run_dir, "checkpoints", "results.json")
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    # CRM decide con B*; las filas sin CRM traen la clave sin prefijo
    for k in ("final_test_crm_acc", "final_test_acc", "test_acc"):
        if k in d:
            return float(d[k])
    return None

rows = []
for f in sorted(glob.glob(os.path.join(RD, "rho_*.csv"))):
    for r in csv.DictReader(open(f)):
        k = key_of.get(r["run_dir"].rstrip("/"))
        if not k or not r["rho_all"]:
            continue
        rows.append(dict(fam=k[0], met=k[1], ds=k[2], seed=k[3], rho=float(r["rho_all"]),
                         js={j: float(r[f"rho_j{j}"]) for j in range(2, 7) if r[f"rho_j{j}"]},
                         acc=test_acc(r["run_dir"])))

print("\nrho_(I) por numero de valores dificiles j (media sobre datasets y semillas)\n")
print(f"{'':12s}{'metodo':<18s}" + "".join(f"{'j='+str(j):>9s}" for j in range(2, 7)))
for fam, name in FAMS:
    for met in METHODS:
        per = defaultdict(list)
        for r in rows:
            if r["fam"] == fam and r["met"] == met:
                for j, v in r["js"].items():
                    per[j].append(v)
        line = f"{name:<12s}{met:<18s}"
        for j in range(2, 7):
            line += f"{np.mean(per[j]):>9.3f}" if per[j] else f"{'--':>9s}"
        print(line)

print("\nEfecto de cada componente sobre rho (delta medio por dataset x semilla emparejado)\n")
idx = {(r["fam"], r["met"], r["ds"], r["seed"]): r["rho"] for r in rows}
for fam, name in FAMS:
    for a, b in [("Base", "+ LATTICE"), ("Base", "+ CRM"), ("+ CRM", "+ CRM + LATTICE"),
                 ("+ LATTICE", "+ CRM + LATTICE")]:
        d = [idx[(fam, b, ds, s)] - idx[(fam, a, ds, s)]
             for ds in DS for s in map(str, range(1, 6))
             if (fam, a, ds, s) in idx and (fam, b, ds, s) in idx]
        if d:
            d = np.array(d)
            print(f"{name:<12s}{a:<16s}->{b:<18s} delta={d.mean():+.3f} ± {d.std(ddof=1):.3f}"
                  f"   baja en {int((d < 0).sum())}/{len(d)}")

print("\nCorrelacion de Spearman entre rho_(I) y el accuracy de test\n")
print("  dentro de cada dataset, sobre los 12 metodos x 5 semillas:")
allr = []
for ds in DS:
    x = [(r["rho"], r["acc"]) for r in rows if r["ds"] == ds and r["acc"] is not None]
    if len(x) > 3:
        rho_s, p = spearmanr([a for a, _ in x], [b for _, b in x])
        allr.append(rho_s)
        print(f"    {ds:10s} n={len(x):3d}  spearman={rho_s:+.3f}  p={p:.1e}")
print(f"    {'media':10s}          spearman={np.mean(allr):+.3f}")
x = [(r["rho"], r["acc"]) for r in rows if r["acc"] is not None]
print(f"  agrupando todo (n={len(x)}): spearman={spearmanr([a for a,_ in x],[b for _,b in x])[0]:+.3f}")
