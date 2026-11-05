"""Recalcula las metricas de representacion sobre model_best, sin reentrenar.

Reconstruye los dataloaders desde el cfg.yml *de la propia corrida*, llamando
fix_random(cfg.seed) antes de construirlos, que es el orden exacto de main.py
(fix_random en la 173, get_dataloaders en la 176). Asi los splits de val_4cases
y testing son bit a bit los que ese checkpoint vio: verificado antes en esta
rama, con fix_random los hashes del dataset y del train coinciden entre
procesos, y el split de test es identico incluso sin sembrar.

Uso:
    python recompute_pscore.py <ruta/al/run> [...]
    python recompute_pscore.py --glob 'out/crm/*/general_composition_1/crm_*/*/*'

Escribe las claves <split>_pscore_mean y demas en checkpoints/results.json,
preservando lo que ya estaba.
"""
import argparse, gc, glob, json, os, sys
import torch
from omegaconf import OmegaConf

from visgen.datasets import get_dataloaders
from visgen.models import get_model
from visgen.trainers import get_trainer
from visgen.utils.general import fix_random, load_checkpoint
from visgen.trainers.representation_metrics import (
    compute_representation_metrics_on_loader,
)


def _install_load_cache():
    """Memoiza BaseDataset._load_data entre corridas del mismo proceso.

    Reconstruir el dataset domina el costo: leer y descomprimir el npz/h5 se
    repite identico en las 5 semillas, porque _load_data depende solo de la
    ruta y del subset, no de la semilla. Lo que si depende de la semilla -- el
    shuffle y el random_split -- ocurre despues y produce arreglos nuevos por
    indexado, nunca in place, asi que compartir el resultado crudo es seguro.
    """
    from visgen.datasets.base import BaseDataset
    store = {}
    original = BaseDataset._load_data

    def cached(self, path, dataset_subset):
        key = (type(self).__name__, str(path), str(dataset_subset))
        if key not in store:
            store[key] = original(self, path, dataset_subset)
        return store[key]

    for cls in (BaseDataset, *BaseDataset.__subclasses__()):
        if "_load_data" in cls.__dict__:
            orig = cls.__dict__["_load_data"]

            def make(orig_fn):
                def wrapper(self, path, dataset_subset):
                    key = (type(self).__name__, str(path), str(dataset_subset))
                    if key not in store:
                        store[key] = orig_fn(self, path, dataset_subset)
                    return store[key]
                return wrapper

            setattr(cls, "_load_data", make(orig))


class _NullWriter:
    """get_dataloaders registra estadisticas del split; aqui no queremos logging."""
    def write(self, *a, **k):
        pass


def run_dir_metrics(run_dir, device, splits=None, dry_run=False,
                    max_samples=5000, pairwise_max_samples=None):
    cfg_path = os.path.join(run_dir, "cfg.yml")
    ckpt = os.path.join(run_dir, "checkpoints", "model_best.pth.tar")
    res_path = os.path.join(run_dir, "checkpoints", "results.json")
    for p in (cfg_path, ckpt):
        if not os.path.exists(p):
            return None, f"falta {os.path.basename(p)}"

    cfg = OmegaConf.load(cfg_path)

    # get_dataloaders siembra cada split con seed + su indice de enumeracion:
    #     for data_split_idx, (key, cfg) in enumerate(data_cfg.items()):
    #         base_seed = int(seed) + data_split_idx
    # y ese generador es el que hace el random_split train/val. El cfg.yml que
    # se guarda en disco sale con las claves ordenadas alfabeticamente
    # (testing, training), asi que releerlo tal cual le daria a training el
    # indice 1 en vez del 0 y reconstruiria OTRO split de validacion. Hay que
    # devolver el orden del config fuente antes de construir nada.
    order = [k for k in ("training", "testing") if k in cfg.data]
    order += [k for k in cfg.data if k not in order]
    data_cfg = OmegaConf.create({k: cfg.data[k] for k in order})

    # mismo orden que main.py: sembrar y recien despues construir los loaders
    fix_random(cfg.seed)
    d_dataloaders = get_dataloaders(data_cfg, _NullWriter(), seed=cfg.seed)

    rm = cfg.training.get("representation_metrics", {}) or {}
    if splits is None:
        s = rm.get("split", ["val_4cases", "testing"])
        splits = [s] if isinstance(s, str) else list(s)

    # Mismo camino que main.py + BaseTrainer.train: construir el modelo, dejar
    # que el trainer lo envuelva (CRMTrainer devuelve un CRMWrapper, y es sobre
    # ese objeto que se guardo model_best) y recien ahi cargar los pesos.
    model = get_model(cfg).to(device)
    trainer = get_trainer(cfg)
    model = trainer.build_model(model, d_dataloaders, writer=None)
    model = model.to(device)
    model, _, _, _ = load_checkpoint(ckpt, model, None, device)
    model.eval()

    out = {}
    for split in splits:
        loader = d_dataloaders.get(f"{split}_raw", d_dataloaders.get(split))
        if loader is None:
            out[f"{split}__missing"] = True
            continue
        # OJO: no heredar max_samples del cfg de la corrida. Las corridas que
        # tenian las metricas desactivadas quedaron con max_samples: null, lo
        # que evalua el split entero, y otras con 20000; mezclarlos daria una
        # misma columna de la tabla con tres tamanos de muestra distintos.
        # Se fijan aqui para que todas las filas sean comparables.
        m = compute_representation_metrics_on_loader(
            model=model, loader=loader, device=device,
            max_samples=max_samples,
            pairwise_max_samples=pairwise_max_samples,
            sampling_seed=rm.get("sampling_seed", 0),
            variance_threshold=float(rm.get("variance_threshold", 0.9)),
            observed_metric=rm.get("observed_metric", "cosine"),
        )
        if m:
            out.update({f"{split}_{k}": v for k, v in m.items()})

    if not dry_run and out:
        old = json.load(open(res_path)) if os.path.exists(res_path) else {}
        old.update(out)
        json.dump(old, open(res_path, "w"), indent=1, sort_keys=True)

    # Cada corrida reconstruye el dataset entero; sin soltarlo explicitamente el
    # proceso acumula un dataset por corrida y termina cayendo por segfault al
    # pasar de un dataset chico a uno grande.
    del model, d_dataloaders, trainer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="*")
    ap.add_argument("--glob", action="append", default=[])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--splits", nargs="*", default=None)
    ap.add_argument("--dry-run", action="store_true",
                    help="calcula y muestra, pero no escribe results.json")
    ap.add_argument("--max-samples", type=int, default=5000)
    ap.add_argument("--pairwise-max-samples", type=int, default=None)
    ap.add_argument("--no-cache", action="store_true",
                    help="no memoizar la carga del dataset entre corridas")
    ap.add_argument("--check", action="store_true",
                    help="compara contra el valor ya guardado en vez de escribir")
    a = ap.parse_args()

    runs = list(a.runs)
    for g in a.glob:
        runs += sorted(glob.glob(g))
    if not runs:
        sys.exit("sin corridas")

    if not a.no_cache:
        _install_load_cache()

    for r in runs:
        try:
            out, err = run_dir_metrics(r, a.device, a.splits,
                                       dry_run=a.dry_run or a.check,
                                       max_samples=a.max_samples,
                                       pairwise_max_samples=a.pairwise_max_samples)
        except Exception as e:
            print(f"ERROR {r}: {type(e).__name__}: {e}", flush=True)
            continue
        if err:
            print(f"SALTADO {r}: {err}", flush=True)
            continue
        keys = [k for k in out if k.endswith("_pscore_mean")]
        if a.check:
            old = json.load(open(os.path.join(r, "checkpoints", "results.json")))
            msg = []
            for k in keys:
                if k in old:
                    d = out[k] - old[k]
                    msg.append(f"{k}: guardado={old[k]:.4f} recalc={out[k]:.4f} d={d:+.5f}")
                else:
                    msg.append(f"{k}: nuevo={out[k]:.4f}")
            print(f"{r}\n    " + "\n    ".join(msg), flush=True)
        else:
            print(f"{r}  " + "  ".join(f"{k}={out[k]:.4f}" for k in keys), flush=True)


main()
