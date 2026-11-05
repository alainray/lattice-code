"""rho_(I): brecha de imaginacion normalizada, sobre checkpoints ya entrenados.

Para cada configuracion de test y (con j(y) >= 2 valores dificiles) compara su
representacion real contra la que se obtiene imaginandola por analogia lineal
desde configuraciones del soporte de entrenamiento:

    Lambda phi(y) = E_o[ phi(o) + sum_{m in T(y)} ( phi(o[m -> y_m]) - phi(o) ) ]
    base(y)       = E_o[ phi(o) ]
    rho_(I)       = mean_y ||phi(y) - Lambda phi(y)||  /  mean_y ||phi(y) - base(y)||

T(y) son las coordenadas dificiles de y y el ancla o es y con cada una de ellas
reemplazada por un valor facil, de modo que o y cada o[m -> y_m] tienen a lo mas
un valor dificil y por lo tanto estan en el soporte de entrenamiento (c = 1).

Es un cociente de medias, no una media de cocientes, y es invariante a
transformaciones afines de phi (escalar y desplazar). rho ~ 0 significa que la
representacion de la configuracion nueva es exactamente la superposicion de sus
partes; rho ~ 1 significa que imaginar no aporta nada sobre el promedio del
soporte; rho > 1, que la analogia lineal empeora la prediccion.

No entrena ni escribe nada de los modelos: solo lee model_best.pth.tar y extrae
representaciones, igual que recompute_pscore.py.

Uso:
    python rho_imagination.py --glob 'out/crm/*/general_composition_1/crm_*/*/*' --out rho.csv
"""
import argparse, csv, gc, glob, itertools, os, sys
from collections import defaultdict

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset, Subset

from visgen.datasets import get_dataloaders, _unwrap_subset
from visgen.datasets.non_iid import NonIIDWrapper
from visgen.models import get_model
from visgen.trainers import get_trainer
from visgen.utils.general import fix_random, load_checkpoint


# ------------------------------------------------------------------ metrica ---
def config_means(reps, labels):
    """tupla de configuracion -> representacion media de sus imagenes."""
    acc, cnt = {}, defaultdict(int)
    for z, y in zip(reps, labels):
        key = tuple(int(v) for v in y)
        acc[key] = z if key not in acc else acc[key] + z
        cnt[key] += 1
    return {k: (acc[k] / cnt[k]).astype(np.float64) for k in acc}


def anchors_for(y, T, easy, K, rng, max_exact):
    """Valores faciles para las coordenadas de T: enumeracion exacta o K muestras."""
    sizes = [len(easy[m]) for m in T]
    if min(sizes) == 0:
        return [], False
    if int(np.prod(sizes)) <= max_exact:
        return list(itertools.product(*[easy[m] for m in T])), False
    return [tuple(rng.choice(easy[m]) for m in T) for _ in range(K)], True


def held_out_residual(phi_in, phi_test, difficult, values, K=256, max_exact=4096, seed=0):
    """Numeradores y denominadores por configuracion de test."""
    rng = np.random.default_rng(seed)
    easy = [[v for v in values[m] if v not in difficult[m]] for m in range(len(values))]
    num, den, js, coverage = [], [], [], []
    sampled_any = False
    for y, z_y in phi_test.items():
        T = [m for m, v in enumerate(y) if v in difficult[m]]
        if len(T) < 2:
            continue
        anchors, sampled = anchors_for(y, T, easy, K, rng, max_exact)
        sampled_any = sampled_any or sampled
        lam_sum, base_sum, n_ok, n_tot = 0.0, 0.0, 0, 0
        for e in anchors:
            n_tot += 1
            o = list(y)
            for m, v in zip(T, e):
                o[m] = v
            o = tuple(o)
            singles = []
            for m in T:
                s = list(o)
                s[m] = y[m]
                singles.append(tuple(s))
            if o not in phi_in or any(s not in phi_in for s in singles):
                continue
            phi_o = phi_in[o]
            lam_sum = lam_sum + phi_o + sum(phi_in[s] - phi_o for s in singles)
            base_sum = base_sum + phi_o
            n_ok += 1
        if n_ok == 0:
            continue
        num.append(float(np.linalg.norm(z_y - lam_sum / n_ok)))
        den.append(float(np.linalg.norm(z_y - base_sum / n_ok)))
        js.append(len(T))
        coverage.append(n_ok / max(n_tot, 1))
    return (np.array(num), np.array(den), np.array(js), np.array(coverage), sampled_any)


def rho_I(num, den, js=None, by_j=False):
    if len(num) == 0 or den.mean() == 0:
        return {"all": float("nan"), "n": int(len(num))}
    out = {"all": float(num.mean() / den.mean()), "n": int(len(num))}
    if by_j and js is not None:
        for j in np.unique(js):
            k = js == j
            out[f"j={int(j)}"] = float(num[k].mean() / den[k].mean())
    return out


# --------------------------------------------------------------- extraccion ---
class _NullWriter:
    def write(self, *a, **k):
        pass


def _install_load_cache():
    """Memoiza BaseDataset._load_data entre corridas: leer el npz/h5 domina el costo
    y es identico para las 5 semillas del mismo dataset (igual que recompute_pscore)."""
    from visgen.datasets.base import BaseDataset
    store = {}
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


def _plain_dataset(loader):
    """Saca el wrapper non-iid: queremos imagenes sueltas, no grupos de 4. El
    wrapper ademas tiene len() = len/4 y muestrea, asi que iterarlo no recorre
    el split completo."""
    ds = loader.dataset
    while isinstance(ds, NonIIDWrapper):
        ds = ds.dataset
    return ds


def _split_geometry(ds):
    """Columnas de la etiqueta que son atributos de split, sus valores y los
    dificiles. split_thresholds se calcula sobre _split_attributes (ordenados por
    indice original) y _split_attribute_indices son las columnas correspondientes
    de la etiqueta, tambien ordenadas, asi que ambos calzan posicion a posicion."""
    thr = np.asarray(ds.split_thresholds, dtype=float)
    cols = [int(c) for c in ds._split_attribute_indices]
    if len(thr) != len(cols):
        raise RuntimeError(f"umbrales ({len(thr)}) != columnas de split ({len(cols)})")
    vals = [sorted({int(v) for v in ds._attribute_values[c]}) for c in cols]
    difficult = [[v for v in vals[k] if v >= thr[k]] for k in range(len(cols))]
    return cols, vals, difficult


def _resolve_subset(ds):
    """(dataset base, indices en el base). random_split se aplica sobre el Subset
    que devuelve ood_validation_split, asi que los Subset vienen anidados y sus
    .indices apuntan al Subset de adentro, no al dataset base."""
    if isinstance(ds, Subset):
        base, idx = _resolve_subset(ds.dataset)
        return base, [idx[int(i)] for i in ds.indices]
    return ds, list(range(len(ds)))


def _targets_matrix(base):
    t = np.asarray(base._dataset_targets)
    if t.ndim > 2:
        t = t[:, -1, :]
    return t


def _pick_indices(loader_or_ds, cols, max_per_config, rng, keep_keys=None):
    """Indices a evaluar: hasta max_per_config imagenes por configuracion. Se eligen
    desde las etiquetas, sin pasar imagenes por la red, que es lo caro."""
    ds = loader_or_ds if isinstance(loader_or_ds, Dataset) else _plain_dataset(loader_or_ds)
    base, pool = _resolve_subset(ds)
    tg = _targets_matrix(base)
    per = defaultdict(list)
    for i in pool:
        key = tuple(int(v) for v in tg[i, cols])
        if keep_keys is not None and key not in keep_keys:
            continue
        if max_per_config and len(per[key]) >= max_per_config:
            continue
        per[key].append(int(i))
    sel = [i for v in per.values() for i in v]
    return base, sel, len(per)


@torch.no_grad()
def _extract_means(model, base, sel, cols, device, batch_size, num_workers):
    """configuracion -> representacion media, acumulando por lote.

    No se guardan las representaciones de todas las imagenes: en mpi3d serian
    ~560k x 3072 floats (14 GB) contra 282k configuraciones del acumulador.
    """
    loader = DataLoader(Subset(base, sel), batch_size=batch_size,
                        num_workers=num_workers, pin_memory=True)
    acc, cnt = {}, defaultdict(int)
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        if x.dim() == 5:          # grupos/secuencias: misma convencion que el P-Score
            x = x[:, -1]
        if y.dim() > 2:
            y = y[:, -1, :]
        z = model.extract_representation(x)
        if z.dim() > 2:
            z = torch.flatten(z, 1)
        z = z.float().cpu().numpy()
        yy = y[:, cols].cpu().numpy()
        for zi, yi in zip(z, yy):
            key = tuple(int(v) for v in yi)
            if key in acc:
                acc[key] += zi
            else:
                acc[key] = zi.copy()
            cnt[key] += 1
    return {k: (acc[k] / cnt[k]) for k in acc}


def run_dir_rho(run_dir, device, max_per_config=4, K=256, max_exact=4096,
                max_test_configs=0, batch_size=256, num_workers=4, khalf=False,
                seed_anchor=0):
    cfg_path = os.path.join(run_dir, "cfg.yml")
    ckpt = os.path.join(run_dir, "checkpoints", "model_best.pth.tar")
    for p in (cfg_path, ckpt):
        if not os.path.exists(p):
            return None, f"falta {os.path.basename(p)}"
    cfg = OmegaConf.load(cfg_path)
    # El cfg.yml en disco sale alfabetizado (testing antes de training) y
    # get_dataloaders siembra cada split con seed + su indice, asi que hay que
    # devolver el orden fuente o se reconstruye OTRO split de validacion.
    order = [k for k in ("training", "testing") if k in cfg.data]
    order += [k for k in cfg.data if k not in order]
    data_cfg = OmegaConf.create({k: cfg.data[k] for k in order})
    fix_random(cfg.seed)
    dl = get_dataloaders(data_cfg, _NullWriter(), seed=cfg.seed)

    # Mismo camino que main.py: construir, dejar que el trainer envuelva (CRM
    # devuelve un CRMWrapper, que es el objeto sobre el que se guardo el best) y
    # recien entonces cargar los pesos.
    model = get_model(cfg).to(device)
    trainer = get_trainer(cfg)
    model = trainer.build_model(model, dl, writer=None).to(device)
    model, _, _, _ = load_checkpoint(ckpt, model, None, device)
    model.eval()

    ds_train = _unwrap_subset(_plain_dataset(dl["training"]))
    cols, vals, difficult = _split_geometry(ds_train)
    rng = np.random.default_rng(seed_anchor)

    base_te, sel_te, n_cfg_te = _pick_indices(dl["testing"], cols, max_per_config, rng)
    if max_test_configs:
        # Submuestrear configuraciones de test (no imagenes) para acotar memoria.
        tg = _targets_matrix(base_te)
        keys = sorted({tuple(int(v) for v in tg[i, cols]) for i in sel_te})
        if len(keys) > max_test_configs:
            keep = {keys[i] for i in rng.choice(len(keys), max_test_configs, replace=False)}
            sel_te = [i for i in sel_te if tuple(int(v) for v in tg[i, cols]) in keep]
            n_cfg_te = len(keep)
    phi_te = _extract_means(model, base_te, sel_te, cols, device, batch_size, num_workers)

    base_in, sel_in, n_cfg_in = _pick_indices(ds_train, cols, max_per_config, rng)
    phi_in = _extract_means(model, base_in, sel_in, cols, device, batch_size, num_workers)

    num, den, js, cov, sampled = held_out_residual(
        phi_in, phi_te, difficult, vals, K, max_exact, seed=seed_anchor)
    r = rho_I(num, den, js, by_j=True)
    r["coverage_mean"] = float(cov.mean()) if len(cov) else float("nan")
    r["sampled"] = bool(sampled)
    r["n_configs_in"] = len(phi_in)
    r["n_configs_test"] = len(phi_te)
    r["split_c"] = int(getattr(ds_train, "split_c", -1))
    r["dim"] = int(next(iter(phi_te.values())).shape[0]) if phi_te else 0
    if khalf:
        n2, d2, _, _, _ = held_out_residual(phi_in, phi_te, difficult, vals,
                                            max(K // 2, 1), max_exact, seed=seed_anchor + 1)
        r["rho_all_Khalf"] = rho_I(n2, d2)["all"]
    del model, dl, trainer, phi_in, phi_te, ds_train
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return r, None


FIELDS = ["run_dir", "dataset", "model", "seed", "rho_all", "rho_j2", "rho_j3", "rho_j4",
          "rho_j5", "rho_j6", "n_U", "coverage_mean", "sampled", "rho_all_Khalf", "dim",
          "n_configs_in", "n_configs_test", "max_per_config"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="*")
    ap.add_argument("--glob", action="append", default=[])
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--max-per-config", type=int, default=4)
    ap.add_argument("--max-test-configs", type=int, default=0)
    ap.add_argument("--K", type=int, default=256)
    ap.add_argument("--max-exact", type=int, default=4096)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--khalf", action="store_true", help="repetir con K/2 para ver estabilidad")
    ap.add_argument("--no-cache", action="store_true")
    a = ap.parse_args()
    runs = list(a.runs)
    for g in a.glob:
        runs += sorted(glob.glob(g))
    if not runs:
        sys.exit("sin corridas")
    if not a.no_cache:
        _install_load_cache()

    done = set()
    new = not os.path.exists(a.out)
    if not new:
        with open(a.out) as fh:
            done = {row["run_dir"] for row in csv.DictReader(fh)}
    fh = open(a.out, "a", newline="")
    w = csv.DictWriter(fh, fieldnames=FIELDS)
    if new:
        w.writeheader()
        fh.flush()
    for rd in runs:
        rd = rd.rstrip("/")
        if rd in done:
            print(f"YA ESTA {rd}", flush=True)
            continue
        p = rd.split("/")
        try:
            r, err = run_dir_rho(rd, a.device, a.max_per_config, a.K, a.max_exact,
                                 a.max_test_configs, a.batch_size, a.num_workers, a.khalf)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"ERROR {rd}: {type(e).__name__}: {e}", flush=True)
            continue
        if err:
            print(f"SALTADO {rd}: {err}", flush=True)
            continue
        row = {"run_dir": rd, "dataset": p[-5], "model": p[-3], "seed": p[-1],
               "rho_all": f"{r['all']:.6f}", "n_U": r["n"],
               "coverage_mean": f"{r['coverage_mean']:.4f}", "sampled": int(r["sampled"]),
               "rho_all_Khalf": f"{r['rho_all_Khalf']:.6f}" if "rho_all_Khalf" in r else "",
               "dim": r["dim"], "n_configs_in": r["n_configs_in"],
               "n_configs_test": r["n_configs_test"], "max_per_config": a.max_per_config}
        for j in range(2, 7):
            row[f"rho_j{j}"] = f"{r[f'j={j}']:.6f}" if f"j={j}" in r else ""
        w.writerow(row)
        fh.flush()
        print(f"{rd}  rho={r['all']:.4f}  n_U={r['n']}  cov={r['coverage_mean']:.3f}"
              f"  sampled={int(r['sampled'])}  dim={r['dim']}", flush=True)
    fh.close()


if __name__ == "__main__":
    main()
