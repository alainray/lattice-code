# Compositional generalization: LATTICE and CRM — code

Code and per-run results for *LATTICE: Latent Imagination for Compositional
Embeddings*. The repository contains no datasets and no checkpoints; `results/`
holds the per-run numbers behind every table of the paper.

## Layout

| path | what it is |
|---|---|
| `main.py` | entry point: builds the dataloaders, the model and the trainer from a config |
| `visgen/datasets/` | the six benchmarks, the orthotopic split (`base.py`) and the non-i.i.d. quartet sampler (`non_iid.py`) |
| `visgen/models/` | ResNet-18, AIN (`ain.py`, with the per-attribute linear readouts), ED, the LATTICE mixer and the CRM wrapper (`crm.py`) |
| `visgen/trainers/` | training loops, the CRM trainer and the representation metrics (`representation_metrics.py`, P-Score and the rest) |
| `configs/` | one file per dataset, model and experiment; the runs in the paper are these configs plus CLI overrides |
| `rho_imagination.py` | the normalized imagination gap $\rho_{(I)}$, computed from a saved checkpoint |
| `recompute_pscore.py` | recomputes the representation metrics from a saved checkpoint, without retraining |
| `cobertura_cuartetos.py` | coverage of the training set by the sampled LATTICE quartets |
| `analysis/` | aggregation of per-run results into the tables of the paper: $\rho_{(I)}$, the paired significance tests and the LaTeX tables |
| `RUNS.md` | the exact command for each of the twelve rows of the paper |
| `pending_runner.sh`, `crm_runner.sh` | the two launchers that cover every reported run, with `run_pending.sh` / `run_crm.sh` as their SLURM wrappers |
| `tests/`, `notebooks/` | unit tests and exploratory notebooks (outputs stripped) |
| `results/` | per-run results behind every table of the paper, and `verify_tables.py`, which recomputes the published values from them |

## Running

```bash
python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt
python main.py --config configs/experiments/<experiment>.yml
```

`RUNS.md` lists the command behind every row of the tables, for example AIN +
LATTICE on one benchmark, all five seeds:

```bash
./pending_runner.sh mpi3d metrics lattice_ain_alg_l_1.5 1,2,3,4,5 non_iid 0
```

## Results

`results/README.md` describes the CSV files, one per table of the paper. To
check the published numbers against the per-run data:

```bash
cd results && python verify_tables.py
```

## Cluster scripts

The SLURM headers are generic: partition, node and account names are
placeholders (`gpu-partition`, `cluster-a`, `ACCOUNT`) and have to be replaced
with the ones of your own cluster. Paths use `$HOME` and `/scratch/USER`.

## Datasets

Not included. `data/README.md` in the original repository lists the sources for
Cars3D, dSprites, MPI3D, CLEVR, I-RAVEN and Shapes3D; the config files expect
them under `data/<dataset>/`.

## Credits

This code builds on the public repository of *Scalable Evaluation and Neural
Models for Compositional Generalization* (Camposampiero, Barbiero, Hersche,
Wattenhofer and Rahimi, NeurIPS 2025, arXiv:2511.02667), whose MIT license is
kept in `LICENSE`. The compositional splits, the AIN and ED backbones and the
evaluation harness come from there; LATTICE, CRM, the linear readouts, the
$\rho_{(I)}$ metric and the analysis code are ours.
