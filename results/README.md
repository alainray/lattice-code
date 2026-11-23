# Data behind the tables of the paper

Per-run results that support every table of *LATTICE: Latent Imagination for Compositional
Embeddings*. Each file is plain CSV. Rows are individual training runs (one configuration, one
benchmark, one seed) unless the table itself is an aggregate, such as a statistical test.

To check the published numbers, run

```
python verify_tables.py
```

It needs only the Python standard library. It recomputes, from the per-run files, every
per-benchmark mean and every average of Tables 1–3 and of the appendix accuracy tables, and compares
them with the values printed in the paper.

## Setting

Six benchmarks (`cars3d`, `dsprites`, `mpi3d`, `clevr`, `iraven`, `shapes3d`), orthotopic split with
c = 1, five seeds (1–5) per configuration, model selection on in-distribution validation accuracy.
Accuracies are exact-match test accuracies in percent. Means over seeds are reported with the
standard error of the mean.

## Files

| File | Paper | Content |
|---|---|---|
| `table1_test_accuracy_runs.csv` | Table 1 | test accuracy per run, 18 rows × 6 benchmarks × 5 seeds |
| `table2_ablation_components_runs.csv` | Table 2 (left), Appendix D | component ablation of LATTICE on AIN |
| `table2_lambda_runs.csv` | Table 2 (right), Appendix D | imagination weight λ on AIN; λ = 0 is the "− imagination" row |
| `table3_pscore_rho_runs.csv` | Table 3, Appendix F | P-score on validation and test, and normalized imagination gap ρ(I), per run |
| `appendix_train_coverage.csv` | Appendix C | training images and coverage by compositional sampling |
| `appendix_epochs.csv` | Appendix C | number of epochs and λ, read from the configuration of each run |
| `appendix_parameters.csv` | Appendix C | parameter counts; `pct_reported_trainable_plus_b_hat` is the reported value |
| `appendix_compute_budget.csv` | Appendix C | GPU-hours by group of runs |
| `appendix_running_times.csv`, `appendix_runtimes_per_run.csv` | Appendix C | median and per-run training time and GPU |
| `appendix_learned_rule_runs.csv` | Appendix D | learned imagination rule against the parameter-free analogy |
| `appendix_paired_accuracy.csv`, `appendix_paired_metrics.csv` | Appendix D | paired tests (mean difference, cluster-bootstrap 95% CI, wins, Wilcoxon p, Holm-corrected p) |
| `appendix_pscore_val_by_dataset.csv`, `appendix_pscore_test_by_dataset.csv` | Appendix F | P-score per benchmark |
| `appendix_rho_by_j.csv`, `appendix_rho_mpi3d.csv` | Appendix F | ρ(I) by number of hard values j, over all benchmarks and on MPI3D |
| `appendix_spearman.csv` | Appendix F | Spearman correlation between representation metrics and test accuracy |

## Columns worth knowing

- `source_file`: path of the result file each value comes from, relative to the results directory
  of the original runs. It makes every value traceable to its run.
- `source_key` (Table 1): `test_acc` for rows without CRM, `final_test_crm_acc` for `+ CRM`
  (joint argmax with the extrapolated bias), `final_test_baseline_acc` for `+ CRM_loss`
  (per-attribute argmax). Rows with CRM and with CRM_loss come from the same training runs.
- `provenance`: `results.json`, or `wandb history (val_acc, >=)` for the few runs whose result file
  was lost. Those values are recovered from the training history with the same
  checkpoint-selection rule as the trainer, and they reproduce the published means.
- `rho_from_a_different_run` (Table 3): 1 when ρ(I) was computed on a re-run of the same configuration
  that stored per-epoch representations, rather than on the run that gave the P-score.
- Running times: a few runs were restarted once, so they have two launches in the experiment
  tracker. `runtimes_per_run.csv` gives both readings (longest launch and first launch).

The paired tests are computed with the code in `analysis/stats/` of this repository.
