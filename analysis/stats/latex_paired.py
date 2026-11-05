"""LaTeX de los tests pareados: accuracy (9 contrastes x backbone) y metricas."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paired import wilcoxon, boot_ci, holm
import paired_metrics as PM

HERE = os.path.dirname(os.path.abspath(__file__))
acc = {tuple(k.split("|")): v for k, v in json.load(open(os.path.join(HERE, "acc_paper.json"))).items()}
DS, SEEDS = PM.DS, PM.SEEDS
CONTR = [("", " + LATTICE"), ("", " + CRM"), ("", " + CRM_loss"),
         ("", " + LATTICE + CRM"), ("", " + LATTICE + CRM_loss"),
         (" + CRM", " + LATTICE + CRM"), (" + CRM_loss", " + LATTICE + CRM_loss"),
         (" + LATTICE", " + LATTICE + CRM"), (" + LATTICE", " + LATTICE + CRM_loss")]
NAME = {"": "Base", " + CRM": "+ CRM", " + CRM_loss": r"+ CRM$_{\mathrm{loss}}$",
        " + LATTICE": "+ LATTICE", " + LATTICE + CRM": "+ LATTICE + CRM",
        " + LATTICE + CRM_loss": r"+ LATTICE + CRM$_{\mathrm{loss}}$"}

def fmt_p(p):
    if p < 0.0001:
        return r"$<$0.0001"
    return f"{p:.4f}"

def block(fam):
    rows, ps = [], []
    for a, b in CONTR:
        d, g = [], []
        for ds in DS:
            for s in SEEDS:
                ka, kb = (fam + a, ds, s), (fam + b, ds, s)
                if ka in acc and kb in acc:
                    d.append(acc[kb] - acc[ka]); g.append(ds)
        d, g = np.array(d), np.array(g)
        _, pw = wilcoxon(d)
        lo, hi = boot_ci(d, groups=g, seed=0)
        per_ds = np.array([d[g == ds].mean() for ds in DS])
        _, p6 = wilcoxon(per_ds)
        rows.append((NAME[a] + r" $\rightarrow$ " + NAME[b], d.mean(), lo, hi,
                     int((d > 0).sum()), len(d), pw, int((per_ds > 0).sum()), p6))
        ps.append(pw)
    out = []
    for (name, m, lo, hi, w, n, pw, w6, p6), pa in zip(rows, holm(ps)):
        bold = r"\textbf{" if pa < 0.05 else ""
        end = "}" if pa < 0.05 else ""
        out.append(f"{name} & {bold}{m:+.2f}{end} & [{lo:+.2f}, {hi:+.2f}] & {w}/{n} & "
                   f"{bold}{fmt_p(pa)}{end} & {w6}/6 & {fmt_p(p6)} \\\\")
    return out

print(r"""\begin{table}[tb]
\caption{Paired comparisons on test accuracy. Each pair is one (benchmark, seed): the same seed fixes the same train/validation split and the same held-out test set for both methods, so the difference removes the variation shared by them. $\Delta$ is the mean paired difference in accuracy points; the 95\% CI is a percentile bootstrap that resamples \emph{benchmarks} (cluster bootstrap), which is the level the claim generalises over. $p$ is a two-sided Wilcoxon signed-rank test, Holm-corrected over the nine contrasts within each backbone; bold marks $p < 0.05$ after correction. The last two columns repeat the test with one pair per benchmark (seeds averaged, $n = 6$), where the smallest attainable two-sided $p$ is 0.0312.}
  \label{tab:paired_accuracy}
\centering
\small
\setlength{\tabcolsep}{4pt}
\renewcommand{\arraystretch}{1.1}
\resizebox{\textwidth}{!}{%
\begin{tabular}{lcccccc}
\toprule
& \multicolumn{4}{c}{\textbf{Per (benchmark, seed)}, $n = 30$} & \multicolumn{2}{c}{\textbf{Per benchmark}, $n = 6$} \\
\cmidrule(lr){2-5}\cmidrule(lr){6-7}
\textbf{Contrast} & \textbf{$\Delta$ (pp)} & \textbf{95\% CI} & \textbf{Wins} & \textbf{$p$ (Holm)} & \textbf{Wins} & \textbf{$p$} \\
\midrule""")
for i, fam in enumerate(["ResNet-18", "AIN", "ED"]):
    if i:
        print(r"\midrule")
    print(r"\multicolumn{7}{l}{\textit{" + fam + r"}} \\")
    for line in block(fam):
        print(line)
print(r"""\bottomrule
\end{tabular}}
\end{table}""")
