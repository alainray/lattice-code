"""Matriz de accuracy de test por (metodo, dataset, semilla), desde results.json."""
import csv, json, os
import numpy as np
RD = os.path.expanduser("~/results/rho")
DS = ["cars3d", "dsprites", "mpi3d", "clevr", "iraven", "shapes3d"]
# (familia, metodo del runs.csv, clave de accuracy) -> fila de la tabla del paper
ROWS = [("resnet18", "Base", "test_acc", "ResNet-18"),
        ("resnet18", "+ CRM", "final_test_crm_acc", "ResNet-18 + CRM"),
        ("resnet18", "+ CRM", "final_test_baseline_acc", "ResNet-18 + CRM_loss"),
        ("resnet18", "+ LATTICE", "test_acc", "ResNet-18 + LATTICE"),
        ("resnet18", "+ CRM + LATTICE", "final_test_crm_acc", "ResNet-18 + LATTICE + CRM"),
        ("resnet18", "+ CRM + LATTICE", "final_test_baseline_acc", "ResNet-18 + LATTICE + CRM_loss"),
        ("ain", "Base", "test_acc", "AIN"),
        ("ain", "+ CRM", "final_test_crm_acc", "AIN + CRM"),
        ("ain", "+ CRM", "final_test_baseline_acc", "AIN + CRM_loss"),
        ("ain", "+ LATTICE", "test_acc", "AIN + LATTICE"),
        ("ain", "+ CRM + LATTICE", "final_test_crm_acc", "AIN + LATTICE + CRM"),
        ("ain", "+ CRM + LATTICE", "final_test_baseline_acc", "AIN + LATTICE + CRM_loss"),
        ("ed", "Base", "test_acc", "ED"),
        ("ed", "+ CRM", "final_test_crm_acc", "ED + CRM"),
        ("ed", "+ CRM", "final_test_baseline_acc", "ED + CRM_loss"),
        ("ed", "+ LATTICE", "test_acc", "ED + LATTICE"),
        ("ed", "+ CRM + LATTICE", "final_test_crm_acc", "ED + LATTICE + CRM"),
        ("ed", "+ CRM + LATTICE", "final_test_baseline_acc", "ED + LATTICE + CRM_loss")]
runs = defaultdict = {}
for r in csv.DictReader(open(os.path.join(RD, "runs.csv"))):
    if r["run_dir"]:
        runs[(r["family"], r["method"], r["dataset"], r["seed"])] = r["run_dir"]

acc = {}
for fam, met, key, label in ROWS:
    for ds in DS:
        for s in map(str, range(1, 6)):
            rd = runs.get((fam, met, ds, s))
            if not rd:
                continue
            p = os.path.join(rd, "checkpoints", "results.json")
            if not os.path.exists(p):
                continue
            d = json.load(open(p))
            v = d.get(key)
            if v is None and key == "test_acc":
                v = d.get("final_test_acc")
            if v is not None:
                acc[(label, ds, s)] = float(v) * (100.0 if float(v) <= 1.0 else 1.0)

if __name__ == "__main__":
    print(f"{'fila':<34s}" + "".join(f"{d:>16s}" for d in DS))
    for *_, label in ROWS:
        line = f"{label:<34s}"
        for ds in DS:
            v = [acc[(label, ds, s)] for s in map(str, range(1, 6)) if (label, ds, s) in acc]
            line += f"{np.mean(v):>10.2f} ({np.std(v, ddof=1):.2f})" if len(v) > 1 else f"{'--':>16s}"
        print(line)
