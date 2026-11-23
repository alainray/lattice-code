"""Recompute the published accuracy and representation tables from the per-run CSV files.

For every row of Tables 1-3 and of the appendix accuracy tables, this script averages the five
seeds of each benchmark, averages the six benchmarks, and compares the result with the value
printed in the paper (rounded as in the paper). It uses only the Python standard library.

    python verify_tables.py
"""
import csv
import os
import statistics

HERE = os.path.dirname(os.path.abspath(__file__))
DATASETS = ['cars3d', 'dsprites', 'mpi3d', 'clevr', 'iraven', 'shapes3d']

# Values printed in the paper: per-benchmark means and the average over benchmarks.
PUBLISHED = {
 "table1": {
  "ResNet-18": [
   33.12,
   21.3,
   43.65,
   20.4,
   11.49,
   84.78,
   35.79
  ],
  "ResNet-18 + CRM": [
   52.95,
   10.76,
   30.48,
   64.03,
   12.54,
   32.06,
   33.8
  ],
  "ResNet-18 + CRM_loss": [
   51.84,
   16.55,
   55.65,
   56.4,
   19.22,
   79.64,
   46.55
  ],
  "ResNet-18 + LATTICE": [
   53.99,
   22.78,
   43.57,
   37.89,
   23.44,
   87.82,
   44.91
  ],
  "ResNet-18 + CRM + LATTICE": [
   73.04,
   11.12,
   59.06,
   59.95,
   27.87,
   75.49,
   51.09
  ],
  "ResNet-18 + CRM_loss + LATTICE": [
   59.36,
   24.67,
   54.75,
   60.09,
   32.23,
   92.07,
   53.86
  ],
  "AIN": [
   46.82,
   55.76,
   59.55,
   47.24,
   59.2,
   84.07,
   58.77
  ],
  "AIN + CRM": [
   57.08,
   27.4,
   53.26,
   69.58,
   49.33,
   51.2,
   51.31
  ],
  "AIN + CRM_loss": [
   61.62,
   54.06,
   66.71,
   67.54,
   68.65,
   91.45,
   68.34
  ],
  "AIN + LATTICE": [
   56.72,
   67.25,
   57.92,
   60.5,
   76.49,
   92.89,
   68.63
  ],
  "AIN + CRM + LATTICE": [
   71.14,
   49.29,
   69.61,
   68.7,
   68.82,
   80.55,
   68.02
  ],
  "AIN + CRM_loss + LATTICE": [
   60.9,
   63.26,
   68.69,
   60.22,
   80.63,
   92.7,
   71.07
  ],
  "ED": [
   49.22,
   69.99,
   68.92,
   50.33,
   70.49,
   90.17,
   66.52
  ],
  "ED + CRM": [
   61.67,
   28.68,
   54.86,
   67.43,
   62.09,
   49.84,
   54.1
  ],
  "ED + CRM_loss": [
   64.83,
   59.49,
   71.34,
   69.41,
   73.78,
   91.17,
   71.67
  ],
  "ED + LATTICE": [
   38.0,
   64.51,
   64.2,
   58.52,
   77.02,
   89.9,
   65.36
  ],
  "ED + CRM + LATTICE": [
   71.88,
   58.83,
   72.3,
   67.5,
   70.68,
   83.49,
   70.78
  ],
  "ED + CRM_loss + LATTICE": [
   61.79,
   63.24,
   78.59,
   59.1,
   78.77,
   91.21,
   72.12
  ]
 },
 "table2_components": {
  "AIN + LATTICE": [
   56.72,
   67.25,
   57.92,
   60.5,
   76.49,
   92.89,
   68.63
  ],
  "- sampling": [
   47.92,
   61.6,
   54.65,
   50.78,
   67.75,
   94.11,
   62.8
  ],
  "- imagination": [
   49.02,
   61.9,
   55.19,
   55.76,
   57.41,
   89.28,
   61.43
  ],
  "- both (AIN)": [
   46.82,
   55.76,
   59.55,
   47.24,
   59.2,
   84.07,
   58.77
  ]
 },
 "table2_lambda": {
  "0": [
   49.02,
   61.9,
   55.19,
   55.76,
   57.41,
   89.28,
   61.43
  ],
  "0.5": [
   53.18,
   67.68,
   56.56,
   61.2,
   70.02,
   90.54,
   66.53
  ],
  "0.75": [
   55.56,
   68.77,
   61.97,
   58.39,
   73.78,
   92.46,
   68.49
  ],
  "1": [
   54.33,
   67.36,
   57.46,
   59.76,
   69.71,
   90.79,
   66.57
  ],
  "1.5": [
   56.72,
   67.25,
   57.92,
   60.5,
   76.49,
   92.89,
   68.63
  ],
  "2": [
   55.72,
   63.85,
   59.66,
   61.2,
   71.34,
   92.68,
   67.41
  ]
 },
 "learned_rule": {
  "ResNet-18": [
   33.12,
   21.3,
   43.65,
   20.4,
   11.49,
   84.78,
   35.79
  ],
  "ResNet-18 + LATTICE": [
   53.99,
   22.78,
   43.57,
   37.89,
   23.44,
   87.82,
   44.91
  ],
  "ResNet-18 + learned d=64": [
   38.33,
   17.5,
   44.38,
   36.14,
   12.63,
   83.5,
   38.74
  ],
  "ResNet-18 + learned d=128": [
   40.66,
   22.13,
   44.97,
   36.7,
   15.07,
   84.66,
   40.7
  ],
  "ResNet-18 + learned d=256": [
   41.39,
   21.74,
   45.14,
   41.03,
   17.3,
   85.78,
   42.06
  ],
  "AIN": [
   46.82,
   55.76,
   59.55,
   47.24,
   59.2,
   84.07,
   58.77
  ],
  "AIN + LATTICE": [
   56.72,
   67.25,
   57.92,
   60.5,
   76.49,
   92.89,
   68.63
  ],
  "AIN + learned d=64": [
   51.36,
   58.01,
   55.27,
   59.55,
   69.32,
   90.07,
   63.93
  ],
  "AIN + learned d=128": [
   49.83,
   60.33,
   55.3,
   57.05,
   70.62,
   90.92,
   64.01
  ],
  "AIN + learned d=256": [
   50.8,
   57.25,
   56.3,
   53.12,
   71.52,
   91.75,
   63.46
  ],
  "ED": [
   49.22,
   69.99,
   68.92,
   50.33,
   70.49,
   90.17,
   66.52
  ],
  "ED + LATTICE": [
   38.0,
   64.51,
   64.2,
   58.52,
   77.02,
   89.9,
   65.36
  ],
  "ED + learned d=64": [
   48.08,
   48.83,
   57.5,
   58.77,
   74.21,
   96.72,
   64.02
  ],
  "ED + learned d=128": [
   49.34,
   47.6,
   59.92,
   58.48,
   72.57,
   94.42,
   63.72
  ],
  "ED + learned d=256": [
   49.07,
   47.31,
   53.26,
   59.62,
   69.63,
   93.08,
   61.99
  ]
 },
 "table3": {
  "ResNet-18": [
   0.732,
   0.721,
   0.634
  ],
  "AIN": [
   0.784,
   0.786,
   0.508
  ],
  "ED": [
   0.945,
   0.834,
   0.454
  ],
  "ResNet-18 + CRM": [
   0.677,
   0.692,
   0.638
  ],
  "AIN + CRM": [
   0.816,
   0.809,
   0.44
  ],
  "ED + CRM": [
   0.917,
   0.848,
   0.37
  ],
  "ResNet-18 + LATTICE": [
   0.988,
   0.81,
   0.497
  ],
  "AIN + LATTICE": [
   0.991,
   0.878,
   0.281
  ],
  "ED + LATTICE": [
   0.973,
   0.872,
   0.242
  ],
  "ResNet-18 + CRM + LATTICE": [
   0.985,
   0.84,
   0.466
  ],
  "AIN + CRM + LATTICE": [
   0.987,
   0.899,
   0.251
  ],
  "ED + CRM + LATTICE": [
   0.993,
   0.91,
   0.212
  ]
 }
}


def load(name):
    with open(os.path.join(HERE, name), encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))


def per_benchmark_means(rows, key, value):
    groups = {}
    for r in rows:
        groups.setdefault(r[key], {}).setdefault(r['dataset'], []).append(float(r[value]))
    out = {}
    for row, by_ds in groups.items():
        assert all(len(by_ds[d]) == 5 for d in DATASETS), (row, {d: len(v) for d, v in by_ds.items()})
        means = [statistics.mean(by_ds[d]) for d in DATASETS]
        out[row] = means + [statistics.mean(means)]
    return out


def check(title, computed, published, decimals):
    ok = bad = 0
    for row, pub in published.items():
        got = computed[str(row)]
        for i, (g, p) in enumerate(zip(got, pub)):
            if round(g, decimals) == round(p, decimals) or abs(g - p) <= 0.5 * 10 ** -decimals + 1e-9:
                ok += 1
            else:
                bad += 1
                print('  MISMATCH %s | %s | column %d: data %.4f, paper %.*f' % (title, row, i, g, decimals, p))
    print('%-38s %3d values match, %d differ' % (title, ok, bad))
    return bad


def main():
    bad = 0
    t1 = per_benchmark_means(load('table1_test_accuracy_runs.csv'), 'table_row', 'test_accuracy_pct')
    bad += check('Table 1 (test accuracy)', t1, PUBLISHED['table1'], 2)

    comp = per_benchmark_means(load('table2_ablation_components_runs.csv'), 'table_row', 'test_accuracy_pct')
    bad += check('Table 2 / D.1 (component ablation)', comp, PUBLISHED['table2_components'], 2)

    lam = per_benchmark_means(load('table2_lambda_runs.csv'), 'lambda', 'test_accuracy_pct')
    bad += check('Table 2 / D.1 (lambda sweep)', lam, PUBLISHED['table2_lambda'], 2)

    lr = per_benchmark_means(load('appendix_learned_rule_runs.csv'), 'table_row', 'test_accuracy_pct')
    bad += check('Appendix D (learned rule)', lr, PUBLISHED['learned_rule'], 2)

    t3_rows = load('table3_pscore_rho_runs.csv')
    t3 = {}
    for metric_index, column in enumerate(['pscore_val', 'pscore_test', 'rho_imagination']):
        for row, vals in per_benchmark_means(t3_rows, 'table_row', column).items():
            t3.setdefault(row, [None] * 3)[metric_index] = vals[-1]
    bad += check('Table 3 (P-score val/test, rho)', t3, PUBLISHED['table3'], 3)

    print('\nAll published values reproduced.' if bad == 0 else '\n%d values differ.' % bad)
    return bad


if __name__ == '__main__':
    raise SystemExit(1 if main() else 0)
