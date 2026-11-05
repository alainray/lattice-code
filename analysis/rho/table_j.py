import csv, glob, os
from collections import defaultdict
import numpy as np
RD = os.path.expanduser("~/results/rho")
FAMS = [("resnet18", "ResNet-18"), ("ain", "AIN"), ("ed", "ED")]
METHODS = ["Base", "+ CRM", "+ LATTICE", "+ CRM + LATTICE"]
key_of = {r["run_dir"].rstrip("/"): (r["family"], r["method"], r["dataset"], r["seed"])
          for r in csv.DictReader(open(os.path.join(RD, "runs.csv"))) if r["run_dir"]}
cell = defaultdict(list)      # (fam, met, j) -> valores
cols_ds = defaultdict(set)    # j -> datasets que aportan
for f in glob.glob(os.path.join(RD, "rho_*.csv")):
    for r in csv.DictReader(open(f)):
        k = key_of.get(r["run_dir"].rstrip("/"))
        if not k or not r["rho_all"]:
            continue
        fam, met, ds, _ = k
        cell[(fam, met, "todas")].append(float(r["rho_all"]))
        for j in range(2, 7):
            if r[f"rho_j{j}"]:
                cell[(fam, met, j)].append(float(r[f"rho_j{j}"]))
                cols_ds[j].add(ds)
COLS = [2, 3, 4, 5, 6, "todas"]
def cf(v, sd):
    return f"{np.mean(v):.3f} ± {np.std(v, ddof=1):.3f}" if sd and len(v) > 1 else f"{np.mean(v):.3f}"
for sd in (False, True):
    print("\n" + ("MEDIA ± SD" if sd else "MEDIA") + " de rho_(I) por j\n")
    w = 14 if not sd else 20
    print(f"{'modelo':<26s}" + "".join(f"{('j='+str(c) if c!='todas' else 'todas'):>{w}s}" for c in COLS))
    for fam, name in FAMS:
        for met in METHODS:
            lbl = name if met == "Base" else f"{name} {met}"
            line = f"{lbl:<26s}"
            for c in COLS:
                v = cell[(fam, met, c)]
                line += f"{(cf(v, sd) if v else '--'):>{w}s}"
            print(line)
print("\nn (corridas que aportan a cada celda) y datasets por columna\n")
for c in COLS:
    ns = sorted({len(cell[(f, m, c)]) for f, _ in FAMS for m in METHODS if cell[(f, m, c)]})
    ds = ", ".join(sorted(cols_ds[c])) if c != "todas" else "los 6"
    print(f"  j={str(c):<6s} n={ns}   datasets: {ds}")
