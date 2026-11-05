"""LaTeX de los tests pareados sobre P-Score y rho_(I)."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paired import wilcoxon, boot_ci, holm
import paired_metrics as PM

CONTR = PM.CONTRASTS
NAME = {"": "Base", " + CRM": "+ CRM", " + LATTICE": "+ LATTICE",
        " + LATTICE + CRM": "+ LATTICE + CRM"}
fmt_p = lambda p: r"$<$0.0001" if p < 0.0001 else f"{p:.4f}"

def cells(vals, fam):
    rows, ps = [], []
    for a, b in CONTR:
        d, g = [], []
        for ds in PM.DS:
            for s in PM.SEEDS:
                ka, kb = (fam + a, ds, s), (fam + b, ds, s)
                if ka in vals and kb in vals:
                    d.append(vals[kb] - vals[ka]); g.append(ds)
        d, g = np.array(d), np.array(g)
        _, pw = wilcoxon(d)
        lo, hi = boot_ci(d, groups=g, seed=0)
        rows.append((NAME[a] + r" $\rightarrow$ " + NAME[b], d.mean(), lo, hi, int((d > 0).sum()), len(d)))
        ps.append(pw)
    return rows, holm(ps)

VALS = [("P-Score (val)", PM.pscore("val_4cases_pscore_mean"), +1),
        ("P-Score (test)", PM.pscore("testing_pscore_mean"), +1),
        (r"$\rho_{(I)}$", PM.rho(), -1)]

print(r"""\begin{table}[tb]
\caption{Paired comparisons on the representation metrics, over the same (benchmark, seed) pairs as Table~\ref{tab:paired_accuracy}. $\Delta$ is the mean paired difference; the 95\% CI is a cluster bootstrap over benchmarks; $p$ is a two-sided Wilcoxon signed-rank test, Holm-corrected over the five contrasts within each backbone and metric. Bold marks $p < 0.05$ after correction. Higher is better for P-Score, lower for $\rho_{(I)}$, so a favourable effect is $\Delta > 0$ for the first two blocks and $\Delta < 0$ for the third.}
  \label{tab:paired_metrics}
\centering
\small
\setlength{\tabcolsep}{4pt}
\renewcommand{\arraystretch}{1.1}
\resizebox{\textwidth}{!}{%
\begin{tabular}{llcccc}
\toprule
\textbf{Metric} & \textbf{Contrast} & \textbf{$\Delta$} & \textbf{95\% CI} & \textbf{$\Delta > 0$} & \textbf{$p$ (Holm)} \\
\midrule""")
for mi, (mname, vals, sign) in enumerate(VALS):
    if mi:
        print(r"\midrule")
    for fi, fam in enumerate(["ResNet-18", "AIN", "ED"]):
        rows, adj = cells(vals, fam)
        for ri, ((name, m, lo, hi, w, n), pa) in enumerate(zip(rows, adj)):
            first = mname if (fi == 0 and ri == 0) else ""
            lab = f"\\textit{{{fam}}}: {name}" if ri == 0 else name
            b, e = (r"\textbf{", "}") if pa < 0.05 else ("", "")
            print(f"{first} & {lab} & {b}{m:+.3f}{e} & [{lo:+.3f}, {hi:+.3f}] & {w}/{n} & {b}{fmt_p(pa)}{e} \\\\")
print(r"""\bottomrule
\end{tabular}}
\end{table}""")
