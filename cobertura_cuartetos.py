"""Cobertura de los cuartetos de LATTICE sobre el split de entrenamiento.

Dos numeros por dataset:
  - estructural: que fraccion de los ejemplos de train PUEDE ser esquina de algun
    patron 2x2 (subset_with_four_case_support, lo mismo que se calcula para
    val_4cases).
  - por epoca: que fraccion toca efectivamente una epoca de muestreo. El wrapper
    ignora el indice y sortea un cuarteto nuevo en cada __getitem__, asi que una
    epoca son len(train)/4 cuartetos = len(train) imagenes, pero con repeticion.

No carga imagenes: se reemplaza el dataset subyacente por un proxy que devuelve
un tensor ficticio y anota el indice, despues de que el wrapper ya construyo sus
indices internos.
"""
import argparse, glob, os, sys
from collections import Counter

import numpy as np
import torch

from visgen.datasets import get_dataloaders, _unwrap_subset
from visgen.datasets.non_iid import NonIIDWrapper, subset_with_four_case_support
from visgen.utils.general import fix_random
from omegaconf import OmegaConf

OUT = os.path.expanduser("~/results")
LATTICE = {"cars3d": "metrics/{ds}/general_composition_1/lattice_ain_alg_l_1.5",
           "dsprites": "metrics/{ds}/general_composition_1/lattice_ain_alg_l_1.5",
           "mpi3d": "metrics/{ds}/general_composition_1/lattice_ain_alg_l_1.5",
           "clevr": "metrics/{ds}/general_composition_1/lattice_ain_alg_l_1.5",
           "iraven": "metrics/{ds}/general_composition_1/lattice_ain_alg_l_1.5",
           "shapes3d": "metrics/{ds}/general_composition_1/lattice_ain_alg_l_1.5"}


class _Rec:
    """Proxy del dataset: no lee imagenes, solo anota que indice se pidio.

    Tiene que reportar el mismo len() que el dataset que reemplaza -- el wrapper
    lo usa para su propio len(), y con otro valor la epoca simulada tendria mas
    cuartetos que la real.
    """
    def __init__(self, n, seen):
        self.n, self.seen = n, seen

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        self.seen[int(idx)] += 1
        return torch.zeros(1), np.zeros(1, dtype=np.int64)


class _NullWriter:
    def write(self, *a, **k):
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("datasets", nargs="*", default=list(LATTICE))
    ap.add_argument("--epochs", type=int, default=1)
    a = ap.parse_args()
    print(f"{'dataset':10s}{'train':>10s}{'estructural':>14s}{'por epoca':>12s}{'unicos':>10s}"
          f"{'cuartetos':>11s}{'p50 visitas':>12s}")
    for ds in (a.datasets or list(LATTICE)):
        run = sorted(glob.glob(os.path.join(OUT, LATTICE[ds].format(ds=ds), "*", "1")))
        if not run:
            print(f"{ds:10s} sin corrida LATTICE")
            continue
        cfg = OmegaConf.load(os.path.join(run[0], "cfg.yml"))
        order = [k for k in ("training", "testing") if k in cfg.data]
        order += [k for k in cfg.data if k not in order]
        data_cfg = OmegaConf.create({k: cfg.data[k] for k in order})
        fix_random(cfg.seed)
        dl = get_dataloaders(data_cfg, _NullWriter(), seed=cfg.seed)
        wrapper = dl["training"].dataset
        if not isinstance(wrapper, NonIIDWrapper):
            print(f"{ds:10s} el loader de train no esta envuelto (sampling iid?)")
            continue
        inner = wrapper.dataset                      # Subset del soporte de train
        n = len(inner)
        # estructural: cuantos ejemplos pueden ser esquina de un 2x2
        allowed = list(data_cfg["training"]["targets"].split("_")) \
            if isinstance(data_cfg["training"]["targets"], str) else list(data_cfg["training"]["targets"])
        sub = subset_with_four_case_support(inner, allowed_attributes=allowed,
                                            shared_other_attributes=True)
        seen = Counter()
        wrapper.dataset = _Rec(n, seen)
        n_groups = len(wrapper) * a.epochs
        for i in range(n_groups):
            wrapper[i % len(wrapper)]
        wrapper.dataset = inner
        uniq = len(seen)
        visits = np.array(sorted(seen.values()))
        print(f"{ds:10s}{n:>10d}{100 * len(sub) / n:>13.1f}%{100 * uniq / n:>11.1f}%"
              f"{uniq:>10d}{n_groups:>11d}{np.median(visits):>12.0f}")
        del dl, wrapper


if __name__ == "__main__":
    main()
