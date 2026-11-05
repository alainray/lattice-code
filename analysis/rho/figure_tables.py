"""Imagen de las tablas de rho_(I) por j, para pegar en un correo."""
import csv, glob, os
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RD = os.path.expanduser("~/results/rho")
FAMS = [("resnet18", "ResNet-18"), ("ain", "AIN"), ("ed", "ED")]
METHODS = ["Base", "+ CRM", "+ LATTICE", "+ CRM + LATTICE"]
key_of = {r["run_dir"].rstrip("/"): (r["family"], r["method"], r["dataset"], r["seed"])
          for r in csv.DictReader(open(os.path.join(RD, "runs.csv"))) if r["run_dir"]}
rows = []
for f in glob.glob(os.path.join(RD, "rho_*.csv")):
    for r in csv.DictReader(open(f)):
        k = key_of.get(r["run_dir"].rstrip("/"))
        if k and r["rho_all"]:
            rows.append((k, r))

def collect(ds=None):
    cell = defaultdict(list)
    for (fam, met, d, _), r in rows:
        if ds and d != ds:
            continue
        cell[(fam, met, "todas")].append(float(r["rho_all"]))
        for j in range(2, 7):
            if r[f"rho_j{j}"]:
                cell[(fam, met, j)].append(float(r[f"rho_j{j}"]))
    return cell

def label(name, met):
    return name if met == "Base" else f"{name} {met}"

def build(cell, cols, sd=True):
    text, empty = [], []
    for fam, name in FAMS:
        for met in METHODS:
            row = []
            for c in cols:
                v = cell[(fam, met, c)]
                if not v:
                    row.append("--")
                elif sd and len(v) > 1:
                    row.append(f"{np.mean(v):.3f} ± {np.std(v, ddof=1):.3f}")
                else:
                    row.append(f"{np.mean(v):.3f}")
            text.append(row)
            empty.append(all(x == "--" for x in row))
    return text, empty

ROWLAB = [label(n, m) for _, n in FAMS for m in METHODS]
FAMOF = [f for f, _ in FAMS for _ in METHODS]
TINT = {"resnet18": "#f4f1ea", "ain": "#eef2f7", "ed": "#eef5f0"}

def draw(ax, cell, cols, title, sub, sd=True):
    text, empty = build(cell, cols, sd)
    head = [("j = " + str(c)) if c != "todas" else "todas" for c in cols]
    keep = [i for i, e in enumerate(empty) if not e]
    tbl = ax.table(cellText=[text[i] for i in keep], rowLabels=[ROWLAB[i] for i in keep],
                   colLabels=head, cellLoc="center", rowLoc="left", loc="upper center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10.5)
    tbl.scale(1, 1.35)
    ncol = len(cols)
    for (r, c), cellobj in tbl.get_celld().items():
        cellobj.set_edgecolor("#d8d5cd")
        cellobj.set_linewidth(0.6)
        if r == 0:
            cellobj.set_facecolor("#e8e5dd")
            cellobj.set_text_props(weight="medium" if False else "bold", color="#2c2c2a")
        else:
            fam = FAMOF[keep[r - 1]]
            cellobj.set_facecolor(TINT[fam])
            if c == -1:
                cellobj.set_text_props(color="#2c2c2a")
        if c == -1:
            cellobj.set_width(0.30)
        elif ncol == 6:
            cellobj.set_width(0.148)
        else:
            cellobj.set_width(0.125)
    ax.axis("off")
    ax.set_title(title, loc="left", fontsize=13, pad=26, color="#2c2c2a")
    ax.text(0, 1.045, sub, transform=ax.transAxes, fontsize=9.5, color="#5f5e5a", va="bottom")

fig = plt.figure(figsize=(14.5, 10.2), dpi=190, facecolor="white")
fig.suptitle(r"$\rho_{(I)}$: brecha de imaginación normalizada, por número de valores difíciles $j$",
             fontsize=16, x=0.045, ha="left", y=0.975, color="#2c2c2a")
fig.text(0.045, 0.941, "Menor es mejor. 0 = la configuración nueva es exactamente la superposición de sus partes; "
                       "1 = imaginarla no ayuda más que el promedio del soporte; >1 = la analogía lineal aleja de lo real.",
         fontsize=10.5, color="#5f5e5a")

ax1 = fig.add_axes([0.045, 0.600, 0.93, 0.25])
draw(ax1, collect(), [2, 3, 4, 5, 6, "todas"],
     "Los 6 datasets — media ± sd",
     "Cuidado: las columnas no promedian los mismos datasets. j=2 y j=3 son los seis (n=30); j=4 son cuatro (n=20); "
     "j=5 son mpi3d y shapes3d (n=10); j=6 es sólo mpi3d (n=5).")

ax2 = fig.add_axes([0.045, 0.155, 0.93, 0.25])
draw(ax2, collect("mpi3d"), [2, 3, 4, 5, 6, "todas"],
     "Sólo mpi3d — media ± sd sobre 5 semillas",
     "El único dataset con j = 2…6, así que aquí las columnas sí son comparables entre sí.")

fig.text(0.045, 0.022,
         "360 modelos: 12 métodos × 6 datasets × 5 semillas, todos con n=30 por celda.\n"
         "Representación: salida del encoder (extract_representation), antes del readout y del mixer. "
         "Cobertura de anclas 1.000 en las 360 corridas; bajar K de 256 a 128 mueve ρ un 0.01% como máximo.",
         fontsize=9, color="#5f5e5a")

out = os.path.join(RD, "rho_tabla_por_j.png")
fig.savefig(out, facecolor="white", bbox_inches="tight", pad_inches=0.35)
print(out)
