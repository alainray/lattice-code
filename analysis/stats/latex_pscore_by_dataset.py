"""Tablas de P-Score por dataset (val y test) en LaTeX, con el formato del paper.

Usa el mismo conjunto de corridas que la Tabla 1 y que la tabla agregada de
P-Score (directorios crm/ y metrics/, no las re-corridas _rep).
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paired_metrics as PM

DSLAB = [("cars3d", "Cars3D"), ("dsprites", "dSprites"), ("mpi3d", "MPI3D"),
         ("clevr", "CLEVR"), ("iraven", "I-RAVEN"), ("shapes3d", "Shapes3D")]
ROWS = [("ResNet-18", "ResNet-18"), ("ResNet-18 + CRM", "ResNet-18 + CRM"),
        ("ResNet-18 + LATTICE", "ResNet-18 + LATTICE"),
        ("ResNet-18 + LATTICE + CRM", "ResNet-18 + LATTICE + CRM"),
        ("AIN", "AIN"), ("AIN + CRM", "AIN + CRM"),
        ("AIN + LATTICE", "AIN + LATTICE"), ("AIN + LATTICE + CRM", "AIN + LATTICE + CRM"),
        ("ED", "ED"), ("ED + CRM", "ED + CRM"),
        ("ED + LATTICE", "ED + LATTICE"), ("ED + LATTICE + CRM", "ED + LATTICE + CRM")]

def table(key, split_label, label):
    vals = PM.pscore(key)
    cell, avg, ns = {}, {}, set()
    for row, _ in ROWS:
        per = []
        for ds, _ in DSLAB:
            v = [vals[(row, ds, s)] for s in PM.SEEDS if (row, ds, s) in vals]
            cell[(row, ds)] = (np.mean(v), np.std(v, ddof=1)) if len(v) > 1 else None
            ns.add(len(v))
            per += v
        avg[row] = np.mean(per) if per else float("nan")
    best = {ds: max((r for r, _ in ROWS), key=lambda r: cell[(r, ds)][0] if cell[(r, ds)] else -9)
            for ds, _ in DSLAB}
    best_avg = max(avg, key=lambda r: avg[r])
    out = [r"\begin{table}[tb]",
           r"\caption{Parallelism Score (P-Score) on " + split_label + r" representations, per benchmark, "
           r"averaged over 5 seeds with standard deviation in parentheses. Higher is better; the best value "
           r"per column is in bold. All values are computed on the checkpoint selected by validation accuracy, "
           r"on the same runs as Table~\ref{tab:test_acc_id}. CRM and CRM$_{\mathrm{loss}}$ share the trained "
           r"encoder and differ only in the decision rule, so they have identical P-Score and appear as one row.}",
           r"  \label{" + label + "}", r"\centering", r"\small",
           r"\setlength{\tabcolsep}{4pt}", r"\renewcommand{\arraystretch}{1.1}",
           r"\resizebox{\textwidth}{!}{%", r"\begin{tabular}{lccccccc}", r"\toprule",
           r"\textbf{Model} & " + " & ".join(r"\textbf{" + n + "}" for _, n in DSLAB) + r" & \textbf{Avg} \\",
           r"\midrule"]
    for i, (row, lab) in enumerate(ROWS):
        if i and i % 4 == 0:
            out.append(r"\midrule")
        cells = []
        for ds, _ in DSLAB:
            c = cell[(row, ds)]
            s = f"{c[0]:.3f} ({c[1]:.3f})" if c else "--"
            cells.append(r"\textbf{" + s + "}" if best[ds] == row else s)
        a = f"{avg[row]:.3f}"
        cells.append(r"\textbf{" + a + "}" if best_avg == row else a)
        out.append(f"{lab} & " + " & ".join(cells) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    return "\n".join(out), ns

t1, n1 = table("val_4cases_pscore_mean", "compositional validation", "tab:pscore_val_by_dataset")
t2, n2 = table("testing_pscore_mean", "held-out test", "tab:pscore_test_by_dataset")
print(t1); print(); print(t2)
sys.stderr.write(f"\n[n por celda: val {sorted(n1)}, test {sorted(n2)}]\n")
