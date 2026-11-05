"""Tests pareados sobre P-Score (val y test) y rho_(I).

P-Score se lee de las mismas corridas que la Tabla 1 (crm/ y metrics/, no las
re-corridas _rep). rho_(I) viene de rho_<ds>.csv, que se calculo sobre el
conjunto con checkpoints; las filas donde ambos conjuntos difieren van marcadas.
"""
import csv, glob, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paired import wilcoxon, sign_flip, boot_ci, holm

OUT = os.path.expanduser("~/results")
RD = os.path.join(OUT, "rho")
DS = ["cars3d", "dsprites", "mpi3d", "clevr", "iraven", "shapes3d"]
SEEDS = [str(i) for i in range(1, 6)]
FAMS = ["ResNet-18", "AIN", "ED"]
METHODS = ["", " + CRM", " + LATTICE", " + LATTICE + CRM"]
# fila -> (experimento, modelo), anclado a lo que reproduce las tablas publicadas
DIRS = {
 "ResNet-18": ("metrics", "resnet18"), "ResNet-18 + CRM": ("crm", "crm_resnet18"),
 "ResNet-18 + LATTICE": ("metrics", "lattice_resnet18_alg_l_1.5"),
 "ResNet-18 + LATTICE + CRM": ("crm", "crm_resnet18_mixer_l15"),
 "AIN": ("metrics", "split_lin"), "AIN + CRM": ("crm", "crm_split_resnet"),
 "AIN + LATTICE": ("metrics", "lattice_ain_alg_l_1.5"),
 "AIN + LATTICE + CRM": ("crm", "crm_split_resnet_mixer_l15"),
 "ED": ("metrics", "split_4_lin"), "ED + CRM": ("crm", "crm_split_resnet_s4"),
 "ED + LATTICE": ("metrics", "split_resnet_mixer_alg_s4"),
 "ED + LATTICE + CRM": ("crm", "crm_split_resnet_mixer_s4_l15")}

def pscore(key):
    v = {}
    for row, (exp, model) in DIRS.items():
        for ds in DS:
            for p in glob.glob(f"{OUT}/{exp}/{ds}/general_composition_1/{model}/*/*/checkpoints/results.json"):
                d = json.load(open(p))
                if key in d:
                    v[(row, ds, p.split("/")[-3])] = float(d[key])
    return v

def rho():
    key_of = {r["run_dir"].rstrip("/"): (r["family"], r["method"], r["dataset"], r["seed"])
              for r in csv.DictReader(open(os.path.join(RD, "runs.csv"))) if r["run_dir"]}
    lbl = {("resnet18", "Base"): "ResNet-18", ("ain", "Base"): "AIN", ("ed", "Base"): "ED"}
    v = {}
    for f in glob.glob(os.path.join(RD, "rho_*.csv")):
        for r in csv.DictReader(open(f)):
            k = key_of.get(r["run_dir"].rstrip("/"))
            if not k or not r["rho_all"]:
                continue
            fam, met, ds, seed = k
            name = {"resnet18": "ResNet-18", "ain": "AIN", "ed": "ED"}[fam]
            suf = {"Base": "", "+ CRM": " + CRM", "+ LATTICE": " + LATTICE",
                   "+ CRM + LATTICE": " + LATTICE + CRM"}[met]
            v[(name + suf, ds, seed)] = float(r["rho_all"])
    return v

CONTRASTS = [("", " + LATTICE"), ("", " + CRM"), ("", " + LATTICE + CRM"),
             (" + CRM", " + LATTICE + CRM"), (" + LATTICE", " + LATTICE + CRM")]

def run(vals, title, better):
    print(f"\n{'=' * 108}\n{title}   (mejor = {better})\n{'=' * 108}")
    print(f"{'familia':<11s}{'contraste':<34s}{'n':>4s}{'media':>9s}{'IC95% (cluster)':>20s}"
          f"{'sube':>8s}{'Wilcoxon':>11s}{'Holm':>9s}")
    for fam in FAMS:
        rows, ps = [], []
        for a, b in CONTRASTS:
            d, g = [], []
            for ds in DS:
                for s in SEEDS:
                    ka, kb = (fam + a, ds, s), (fam + b, ds, s)
                    if ka in vals and kb in vals:
                        d.append(vals[kb] - vals[ka]); g.append(ds)
            if not d:
                continue
            d, g = np.array(d), np.array(g)
            _, pw = wilcoxon(d)
            lo, hi = boot_ci(d, groups=g, seed=0)
            rows.append(((a.strip() or "Base") + " -> " + b.strip(), len(d), d.mean(), (lo, hi),
                         int((d > 0).sum()), pw))
            ps.append(pw)
        for (name, n, m, (lo, hi), w, pw), pa in zip(rows, holm(ps)):
            star = "*" if pa < 0.05 else " "
            print(f"{fam:<11s}{name:<34s}{n:>4d}{m:>+9.3f}{f'[{lo:+.3f}, {hi:+.3f}]':>20s}"
                  f"{f'{w}/{n}':>8s}{pw:>11.4f}{pa:>8.4f}{star}")

if __name__ == "__main__":
    run(pscore("val_4cases_pscore_mean"), "P-Score en validacion (val_4cases)", "mayor")
    run(pscore("testing_pscore_mean"), "P-Score en test", "mayor")
    run(rho(), "rho_(I)  [OJO: conjunto de corridas _rep, no el de la Tabla 1]", "menor")
