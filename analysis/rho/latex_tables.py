"""LaTeX de las tablas de rho_(I), con el formato de las tablas del paper."""
import csv, glob, os
from collections import defaultdict
import numpy as np
RD = os.path.expanduser("~/results/rho")
FAMS = [("resnet18", "ResNet-18"), ("ain", "AIN"), ("ed", "ED")]
METHODS = [("Base", ""), ("+ CRM", " + CRM"), ("+ LATTICE", " + LATTICE"),
           ("+ CRM + LATTICE", " + LATTICE + CRM")]
key_of = {r["run_dir"].rstrip("/"): (r["family"], r["method"], r["dataset"], r["seed"])
          for r in csv.DictReader(open(os.path.join(RD, "runs.csv"))) if r["run_dir"]}
rows = []
for f in glob.glob(os.path.join(RD, "rho_*.csv")):
    for r in csv.DictReader(open(f)):
        k = key_of.get(r["run_dir"].rstrip("/"))
        if k and r["rho_all"]:
            rows.append((k, r))

def cells(ds=None):
    c = defaultdict(list)
    for (fam, met, d, _), r in rows:
        if ds and d != ds:
            continue
        c[(fam, met, "all")].append(float(r["rho_all"]))
        for j in range(2, 7):
            if r[f"rho_j{j}"]:
                c[(fam, met, j)].append(float(r[f"rho_j{j}"]))
    return c

def fmt(v):
    return f"{np.mean(v):.3f} ({np.std(v, ddof=1):.3f})" if len(v) > 1 else f"{np.mean(v):.3f}"

def table(ds, cols, caption, label):
    c = cells(ds)
    best = {}
    for col in cols:
        vals = {(f, m): np.mean(c[(f, m, col)]) for f, _ in FAMS for m, _ in METHODS if c[(f, m, col)]}
        if vals:
            best[col] = min(vals, key=vals.get)
    head = " & ".join(f"$j = {col}$" if col != "all" else "All $j$" for col in cols)
    out = ["\\begin{table}[tb]", f"\\caption{{{caption}}}", f"  \\label{{{label}}}",
           "\\centering", "\\small", "\\setlength{\\tabcolsep}{4pt}",
           "\\renewcommand{\\arraystretch}{1.1}", "\\resizebox{\\textwidth}{!}{%",
           "\\begin{tabular}{l" + "c" * len(cols) + "}", "\\toprule",
           "\\textbf{Model} & " + " & ".join("\\textbf{" + h + "}" for h in head.split(" & ")) + " \\\\",
           "\\midrule"]
    for fi, (fam, name) in enumerate(FAMS):
        if fi:
            out.append("\\midrule")
        for met, suffix in METHODS:
            vals = []
            for col in cols:
                v = c[(fam, met, col)]
                if not v:
                    vals.append("--")
                else:
                    s = fmt(v)
                    vals.append("\\textbf{" + s + "}" if best.get(col) == (fam, met) else s)
            out.append(f"{name}{suffix} & " + " & ".join(vals) + " \\\\")
    out += ["\\bottomrule", "\\end{tabular}}", "\\end{table}"]
    return "\n".join(out)

CAP1 = ("Normalized imagination gap $\\rho_{(I)}$ by the number $j$ of out-of-support attribute values, "
        "averaged over the six benchmarks and 5 seeds, with standard deviation in parentheses. "
        "For a test configuration $y$, $\\rho_{(I)}$ is the residual of predicting $\\phi(y)$ by linear "
        "analogy from training configurations, normalized by the residual of the anchor mean: $0$ means the "
        "unseen combination is exactly the superposition of its parts, $1$ means imagining it is no better than "
        "the centre of the training support, and $>1$ means the linear analogy points away from the true "
        "representation. Lower is better; best per column in bold. Columns do not average over the same "
        "benchmarks, since the largest $j$ is set by the number of split attributes: $j = 2, 3$ cover all six "
        "($n = 30$), $j = 4$ four of them ($n = 20$), $j = 5$ MPI3D and Shapes3D ($n = 10$), and $j = 6$ MPI3D "
        "alone ($n = 5$). "
        "CRM and CRM$_{\\mathrm{loss}}$ share the trained encoder and differ only in the "
        "decision rule, so they have identical $\\rho_{(I)}$ and appear as one row. Anchors are enumerated "
        "exhaustively except for $3\\%$ of the MPI3D and $0.4\\%$ of the Shapes3D test configurations, where "
        "$K = 256$ anchors are sampled; halving $K$ changes $\\rho_{(I)}$ by at most $0.01\\%$.")
CAP2 = ("Normalized imagination gap $\\rho_{(I)}$ on MPI3D, averaged over 5 seeds, with standard deviation in "
        "parentheses. MPI3D is the only benchmark with $j = 2 \\dots 6$, so here the columns are comparable to "
        "each other and isolate the effect of $j$ from the choice of benchmark. Lower is better; best per column "
        "in bold. Every row grows monotonically in $j$ except ED + LATTICE, which flattens between $j = 5$ and "
        "$j = 6$ ($0.305 \\to 0.300$).")

print(table(None, [2, 3, 4, 5, 6, "all"], CAP1, "tab:rho_imagination_by_j"))
print()
print(table("mpi3d", [2, 3, 4, 5, 6, "all"], CAP2, "tab:rho_imagination_mpi3d"))
