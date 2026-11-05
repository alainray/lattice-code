"""Spearman entre metricas de representacion en VAL y el accuracy de TEST,
desglosado por familia de arquitectura. Se correlaciona val contra test: usar el
valor de test de la metrica contra el accuracy de test seria circular."""
import csv, glob, json, os
from collections import defaultdict
import numpy as np

RD = os.path.expanduser("~/results/rho")
FAMS = [("resnet18", "ResNet-18"), ("ain", "AIN"), ("ed", "ED")]
DS = ["cars3d", "dsprites", "mpi3d", "clevr", "iraven", "shapes3d"]
METRICS = [("pscore_mean", "P-Score"), ("topsim", "TopSim"), ("twonn_id", "TwoNN ID"),
           ("n_components_90pct", "NC-90"), ("sv_auc", "SV-AUC"), ("hoyer_sparsity", "Hoyer")]

def spearman(a, b):
    def rank(v):
        v = np.asarray(v, float); o = np.argsort(v, kind="mergesort"); r = np.empty(len(v)); i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and v[o[j + 1]] == v[o[i]]:
                j += 1
            r[o[i:j + 1]] = (i + j) / 2.0 + 1; i = j + 1
        return r
    if len(a) < 4:
        return np.nan
    return float(np.corrcoef(rank(a), rank(b))[0, 1])

key_of = {r["run_dir"].rstrip("/"): (r["family"], r["method"], r["dataset"], r["seed"])
          for r in csv.DictReader(open(os.path.join(RD, "runs.csv"))) if r["run_dir"]}
rho_of = {}
for f in glob.glob(os.path.join(RD, "rho_*.csv")):
    for r in csv.DictReader(open(f)):
        if r["rho_all"]:
            rho_of[r["run_dir"].rstrip("/")] = float(r["rho_all"])

data = defaultdict(list)   # (fam, ds) -> [(metricas, acc)]
missing = defaultdict(int)
for rd, (fam, met, ds, seed) in key_of.items():
    p = os.path.join(rd, "checkpoints", "results.json")
    if not os.path.exists(p):
        continue
    d = json.load(open(p))
    acc = next((d[k] for k in ("final_test_crm_acc", "final_test_acc", "test_acc") if k in d), None)
    if acc is None:
        continue
    row = {}
    for m, _ in METRICS:
        v = d.get(f"val_4cases_{m}")
        if v is None:
            missing[m] += 1
        row[m] = v
    row["rho"] = rho_of.get(rd)
    data[(fam, ds)].append((row, float(acc)))

print("Spearman de la metrica en VAL (val_4cases) contra el accuracy de TEST.")
print("Celda = media de las correlaciones calculadas dentro de cada dataset (6),")
print("cada una sobre los 4 metodos x 5 semillas de esa familia (n = 20).\n")
names = ["P-Score", "TopSim", "TwoNN ID", "NC-90", "SV-AUC", "Hoyer", "rho_(I) [test]"]
print(f"{'metrica':<16s}" + "".join(f"{n:>14s}" for _, n in FAMS) + f"{'las 3 juntas':>14s}")
for (m, label), disp in zip(METRICS + [("rho", "rho_(I) [test]")], names):
    line = f"{disp:<16s}"
    for fam, _ in FAMS:
        rs = []
        for ds in DS:
            pairs = [(r[m], a) for r, a in data[(fam, ds)] if r.get(m) is not None]
            if len(pairs) >= 4:
                rs.append(spearman([x for x, _ in pairs], [y for _, y in pairs]))
        line += f"{np.mean(rs):>+14.3f}" if rs else f"{'--':>14s}"
    rs = []
    for fam, _ in FAMS:
        for ds in DS:
            pairs = [(r[m], a) for r, a in data[(fam, ds)] if r.get(m) is not None]
            if len(pairs) >= 4:
                rs.append(spearman([x for x, _ in pairs], [y for _, y in pairs]))
    line += f"{np.mean(rs):>+14.3f}"
    print(line)

print("\nn por celda (corridas con la metrica disponible):")
for fam, name in FAMS:
    tot = {m: sum(1 for ds in DS for r, _ in data[(fam, ds)] if r.get(m) is not None) for m, _ in METRICS}
    nr = sum(1 for ds in DS for r, _ in data[(fam, ds)] if r.get("rho") is not None)
    print(f"  {name:<12s} metricas de val: {tot['pscore_mean']}/120   rho: {nr}/120")
