"""Encuentra que corridas reproducen cada celda de la Tabla 1 del paper.

No se asume el directorio: se busca, entre todos los candidatos, el que da la
media publicada. Asi los tests pareados operan sobre las mismas corridas que la
tabla, y no sobre las re-corridas con metricas por epoca (_rep / crm_rep), que
son entrenamientos distintos del mismo config.
"""
import glob, json, os, sys
import numpy as np

OUT = os.path.expanduser("~/results")
DS = ["cars3d", "dsprites", "mpi3d", "clevr", "iraven", "shapes3d"]
PUB = {  # fila -> medias publicadas, en el orden de DS
 "ResNet-18":                     [33.12, 21.30, 43.65, 20.40, 11.49, 84.78],
 "ResNet-18 + CRM":               [52.95, 10.76, 30.48, 64.03, 12.54, 32.06],
 "ResNet-18 + CRM_loss":          [51.84, 16.55, 55.65, 56.40, 19.22, 79.64],
 "ResNet-18 + LATTICE":           [53.99, 22.78, 43.57, 37.89, 23.44, 87.82],
 "ResNet-18 + LATTICE + CRM":     [73.04, 11.12, 59.06, 59.95, 27.87, 75.49],
 "ResNet-18 + LATTICE + CRM_loss":[59.36, 24.67, 54.75, 60.09, 32.23, 92.07],
 "AIN":                           [46.82, 55.76, 59.55, 47.24, 59.20, 84.07],
 "AIN + CRM":                     [57.08, 27.40, 53.26, 69.58, 49.33, 51.20],
 "AIN + CRM_loss":                [61.62, 54.06, 66.71, 67.54, 68.65, 91.45],
 "AIN + LATTICE":                 [56.72, 67.25, 57.92, 60.50, 76.49, 92.89],
 "AIN + LATTICE + CRM":           [71.14, 49.29, 69.61, 68.70, 68.82, 80.55],
 "AIN + LATTICE + CRM_loss":      [60.90, 63.26, 68.69, 60.22, 80.63, 92.70],
 "ED":                            [49.22, 69.99, 68.92, 50.33, 70.49, 90.17],
 "ED + CRM":                      [61.67, 28.68, 54.86, 67.43, 62.09, 49.84],
 "ED + CRM_loss":                 [64.83, 59.49, 71.34, 69.41, 73.78, 91.17],
 "ED + LATTICE":                  [38.00, 64.51, 64.20, 58.52, 77.02, 89.90],
 "ED + LATTICE + CRM":            [71.88, 58.83, 72.30, 67.50, 70.68, 83.49],
 "ED + LATTICE + CRM_loss":       [61.79, 63.24, 78.59, 59.10, 78.77, 91.21]}
KEY = lambda row: ("final_test_crm_acc" if row.endswith("+ CRM") else
                   "final_test_baseline_acc" if row.endswith("CRM_loss") else "test_acc")

def seeds_of(d, key):
    out = {}
    for p in sorted(glob.glob(d + "/*/*/checkpoints/results.json")):
        seed = p.split("/")[-3]
        j = json.load(open(p))
        if key in j:
            v = float(j[key])
            out[seed] = v * (100 if v <= 1 else 1)
    return out

matched, acc, report = {}, {}, []
for row, pubs in PUB.items():
    key = KEY(row)
    # 1) todos los candidatos que reproducen la media publicada, por dataset
    cands = {}
    for ds, pub in zip(DS, pubs):
        hits = {}
        for d in sorted(glob.glob(f"{OUT}/*/{ds}/general_composition_1/*")):
            sd = seeds_of(d, key)
            if len(sd) < 3:
                continue
            if abs(np.mean(list(sd.values())) - pub) < 0.005:
                parts = d.replace(OUT + "/", "").split("/")
                hits[(parts[0], parts[-1])] = (d, sd)
        cands[ds] = hits
    # 2) una misma combinacion (experimento, modelo) para los 6 datasets: si dos
    #    directorios distintos dan la misma media en un dataset, el desempate es
    #    el que explica el resto de la fila, no el orden del glob.
    votes = {}
    for ds in DS:
        for k in cands[ds]:
            votes[k] = votes.get(k, 0) + 1
    pick = max(votes, key=lambda k: votes[k]) if votes else None
    for ds, pub in zip(DS, pubs):
        hit = cands[ds].get(pick)
        if hit is None:
            report.append(f"SIN CALZE  {row:<32s}{ds:<10s} publicado {pub}  "
                          f"(candidatos: {sorted(cands[ds])})")
            continue
        matched[(row, ds)] = hit[0].replace(OUT + "/", "")
        for seed, v in hit[1].items():
            acc[(row, ds, seed)] = v

json.dump({f"{r}|{d}": v for (r, d), v in matched.items()},
          open(os.path.join(os.path.dirname(__file__), "matched_dirs.json"), "w"), indent=1)
json.dump({f"{r}|{d}|{s}": v for (r, d, s), v in acc.items()},
          open(os.path.join(os.path.dirname(__file__), "acc_paper.json"), "w"), indent=1)
print(f"celdas calzadas: {len(matched)}/{len(PUB) * len(DS)}")
for line in report:
    print(" ", line)
print("\ndirectorio usado por fila (si es el mismo en los 6 datasets):")
for row in PUB:
    dirs = {matched.get((row, ds), "?").split("/")[0] + "/" + matched.get((row, ds), "?").split("/")[-1]
            for ds in DS if (row, ds) in matched}
    print(f"  {row:<32s}{sorted(dirs)}")
