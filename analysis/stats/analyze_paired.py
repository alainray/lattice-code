"""Tests pareados sobre la tabla de accuracy de test (Tabla 1)."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paired import wilcoxon, sign_flip, boot_ci, holm

HERE = os.path.dirname(os.path.abspath(__file__))
acc = {tuple(k.split("|")): v for k, v in json.load(open(os.path.join(HERE, "acc_paper.json"))).items()}
DS = ["cars3d", "dsprites", "mpi3d", "clevr", "iraven", "shapes3d"]
SEEDS = [str(i) for i in range(1, 6)]

CONTRASTS = [("", " + LATTICE"), ("", " + CRM"), ("", " + CRM_loss"),
             ("", " + LATTICE + CRM"), ("", " + LATTICE + CRM_loss"),
             (" + CRM", " + LATTICE + CRM"), (" + CRM_loss", " + LATTICE + CRM_loss"),
             (" + LATTICE", " + LATTICE + CRM"), (" + LATTICE", " + LATTICE + CRM_loss")]

def pairs(fam, a, b):
    d, g = [], []
    for ds in DS:
        for s in SEEDS:
            ka, kb = (fam + a, ds, s), (fam + b, ds, s)
            if ka in acc and kb in acc:
                d.append(acc[kb] - acc[ka])
                g.append(ds)
    return np.array(d), np.array(g)

for fam in ("ResNet-18", "AIN", "ED"):
    print(f"\n{'=' * 118}\n{fam}: diferencias emparejadas por (dataset, semilla), en puntos de accuracy\n{'=' * 118}")
    print(f"{'contraste':<42s}{'n':>4s}{'media':>9s}{'IC95% (cluster)':>20s}{'gana':>8s}"
          f"{'Wilcoxon':>11s}{'sign-flip':>11s}{'Holm':>9s}")
    rows, ps = [], []
    for a, b in CONTRASTS:
        d, g = pairs(fam, a, b)
        if len(d) == 0:
            continue
        _, pw = wilcoxon(d)
        pf = sign_flip(d, seed=0)
        lo, hi = boot_ci(d, groups=g, seed=0)
        rows.append((f"{(fam + a) if a else fam}  ->  {fam + b}", len(d), d.mean(),
                     (lo, hi), int((d > 0).sum()), pw, pf))
        ps.append(pw)
    adj = holm(ps)
    for (name, n, m, (lo, hi), w, pw, pf), pa in zip(rows, adj):
        star = "*" if pa < 0.05 else " "
        short = name.replace(fam, "").replace("  ->  ", " -> ").strip() or name
        print(f"{short:<42s}{n:>4d}{m:>+9.2f}{f'[{lo:+.2f}, {hi:+.2f}]':>20s}{f'{w}/{n}':>8s}"
              f"{pw:>11.4f}{pf:>11.4f}{pa:>8.4f}{star}")

print(f"\n{'=' * 118}\nVersion conservadora: un par por dataset (media de las 5 semillas), n = 6, Wilcoxon exacto\n"
      f"El minimo p alcanzable con n = 6 a dos colas es 0.0312.\n{'=' * 118}")
print(f"{'familia':<11s}{'contraste':<42s}{'media':>9s}{'gana':>7s}{'Wilcoxon':>11s}{'sign-flip':>11s}")
for fam in ("ResNet-18", "AIN", "ED"):
    for a, b in CONTRASTS:
        d, g = pairs(fam, a, b)
        if len(d) == 0:
            continue
        per_ds = np.array([d[g == ds].mean() for ds in DS if (g == ds).any()])
        _, pw = wilcoxon(per_ds)
        pf = sign_flip(per_ds, seed=0)
        short = f"{a.strip() or 'Base'} -> {b.strip()}"
        print(f"{fam:<11s}{short:<42s}{per_ds.mean():>+9.2f}{f'{int((per_ds > 0).sum())}/{len(per_ds)}':>7s}"
              f"{pw:>11.4f}{pf:>11.4f}")
