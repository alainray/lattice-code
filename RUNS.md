# Commands for the runs reported in the paper

Twelve rows: {ResNet-18, AIN, ED} x {base, + CRM, + LATTICE, + LATTICE + CRM}.
Each row is 6 benchmarks x 5 seeds = 30 runs. Two generic launchers cover all of
them; both take the benchmark as the first argument and read the split
definition (orthotopic, `c = 1`, per-attribute thresholds) from a table inside.

`<dataset>` is one of `cars3d`, `dsprites`, `mpi3d`, `clevr`, `iraven`, `shapes3d`.

## Rows without CRM

```
./pending_runner.sh <dataset> metrics <model> 1,2,3,4,5 <sampling> <num_workers>
```

| Row | `<model>` | `<sampling>` | `<num_workers>` |
|---|---|---|---|
| ResNet-18 | `resnet18` | `plain` | 0 |
| ResNet-18 + LATTICE | `lattice_resnet18_alg_l_1.5` | `non_iid` | 0 |
| AIN | `split_lin` | `plain` | 0 |
| AIN + LATTICE | `lattice_ain_alg_l_1.5` | `non_iid` | 0 |
| ED | `split_4_lin` | `plain` | 0 |
| ED + LATTICE | `split_resnet_mixer_alg_s4` | `non_iid` | 4 |

LATTICE needs the analogical term, which is only defined on 4-view batches, so
those rows use the `non_iid` dataset configs (`configs/datasets/<dataset>_non_iid.yml`);
the base rows use i.i.d. sampling. `num_workers` is the only setting that is not
uniform across rows: the ED + LATTICE runs were launched with 4 instead of 0.
It changes the dataloader worker seeding, not the split.

## Rows with CRM

```
./crm_runner.sh <dataset> crm <model> 1,2,3,4,5
```

| Row | `<model>` |
|---|---|
| ResNet-18 + CRM | `crm_resnet18` |
| ResNet-18 + LATTICE + CRM | `crm_resnet18_mixer_l15` |
| AIN + CRM | `crm_split_resnet` |
| AIN + LATTICE + CRM | `crm_split_resnet_mixer_l15` |
| ED + CRM | `crm_split_resnet_s4` |
| ED + LATTICE + CRM | `crm_split_resnet_mixer_s4_l15` |

`crm_runner.sh` picks the dataset config from the model name: the `*_mixer_*`
models get the `non_iid` one, the rest the i.i.d. one, and it forces
`training.trainer=crm`. Every CRM run reports both decision rules from the same
trained encoder: `final_test_crm_acc` is the joint argmax with the extrapolated
bias `B*` (row "+ CRM") and `final_test_baseline_acc` is the per-attribute argmax
(row "+ CRM_loss").

## On a cluster

`run_pending.sh` and `run_crm.sh` are the SLURM wrappers around the two runners:

```
sbatch --time=08:00:00 --export=ALL,ds=mpi3d,model=split_lin,seeds=1,flavor=plain,nw=0 run_pending.sh
sbatch --time=20:00:00 --export=ALL,ds=mpi3d,model=crm_split_resnet_mixer_l15,seeds=1 run_crm.sh
```

Their `#SBATCH` headers carry placeholders (`gpu-partition`, `cluster-a`,
`ACCOUNT`) that have to be replaced with your own. Results are written under
`path.base`, which defaults to `$HOME/results` and can be overridden with
`OUT_BASE`.

## Post-hoc analysis, from saved checkpoints

```
python recompute_pscore.py <run_dir>          # P-Score and the other representation metrics
python rho_imagination.py <run_dir> --out rho.csv   # the imagination gap
python cobertura_cuartetos.py                 # coverage of the training set by the sampled quartets
```

`analysis/` turns the per-run outputs into the tables: `analysis/rho/` for the
imagination gap and `analysis/stats/` for the paired significance tests and the
LaTeX tables. `analysis/stats/test_paired.py` checks the statistics code against
values with a known answer and is worth running first.
