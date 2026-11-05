"""Agrega los rho_(I) por corrida en la tabla 4x3 y en la tabla por dataset."""
import csv, glob, os, sys
from collections import defaultdict
import numpy as np

RHO_DIR = os.path.expanduser("~/results/rho")
LIST = sys.argv[1] if len(sys.argv) > 1 else os.path.join(RHO_DIR, "runs.csv")
FAMS = [("resnet18", "ResNet-18"), ("ain", "AIN"), ("ed", "ED")]
METHODS = ["Base", "+ CRM", "+ LATTICE", "+ CRM + LATTICE"]
DS = ["cars3d", "dsprites", "mpi3d", "clevr", "iraven", "shapes3d"]

key_of = {}
for r in csv.DictReader(open(LIST)):
    if r["run_dir"]:
        key_of[r["run_dir"].rstrip("/")] = (r["family"], r["method"], r["dataset"], r["seed"])

vals, extra = defaultdict(list), defaultdict(list)
for f in sorted(glob.glob(os.path.join(RHO_DIR, "rho_*.csv"))):
    for r in csv.DictReader(open(f)):
        k = key_of.get(r["run_dir"].rstrip("/"))
        if k is None or not r["rho_all"] or r["rho_all"] == "nan":
            continue
        fam, met, ds, seed = k
        vals[(fam, met)].append((ds, float(r["rho_all"])))
        extra[(fam, met)].append((int(r["n_U"]), float(r["coverage_mean"]), int(r["sampled"])))

def fmt(v):
    if not v:
        return "--"
    a = np.array([x[1] for x in v])
    return f"{a.mean():.3f} ± {a.std(ddof=1):.3f}" if len(a) > 1 else f"{a.mean():.3f}"

w = max(len(m) for m in METHODS) + 2
print("\nrho_(I) — media ± sd sobre 6 datasets x 5 semillas (menor = mas composicional)\n")
print(" " * w + "".join(f"{n:>22s}" for _, n in FAMS))
for met in METHODS:
    line = f"{met:<{w}s}"
    for fam, _ in FAMS:
        line += f"{fmt(vals[(fam, met)]):>22s}"
    n = [len(vals[(fam, met)]) for fam, _ in FAMS]
    print(line + f"   n={n}")

print("\nPor dataset (media sobre 5 semillas)\n")
hdr = f"{'':<18s}{'':<18s}" + "".join(f"{d:>10s}" for d in DS)
print(hdr)
for fam, name in FAMS:
    for met in METHODS:
        per = defaultdict(list)
        for ds, v in vals[(fam, met)]:
            per[ds].append(v)
        row = f"{name:<18s}{met:<18s}"
        for d in DS:
            row += f"{np.mean(per[d]):>10.3f}" if per[d] else f"{'--':>10s}"
        print(row)

print("\nCobertura de anclas y tamano de U\n")
for fam, name in FAMS:
    for met in METHODS:
        e = extra[(fam, met)]
        if not e:
            continue
        nu = np.array([x[0] for x in e]); cv = np.array([x[1] for x in e])
        print(f"{name:<12s}{met:<18s} n_U medio={nu.mean():8.0f}  cobertura media={cv.mean():.3f}"
              f"  con muestreo de anclas: {sum(x[2] for x in e)}/{len(e)}")
