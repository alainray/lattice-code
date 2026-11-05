"""Validacion de las implementaciones contra casos con respuesta conocida."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paired import wilcoxon, sign_flip, boot_ci, holm

ok = True
def check(c, msg):
    global ok
    print(("PASA  " if c else "FALLA ") + msg); ok = ok and c

# 1. Caso de libro: n=10, p exacto conocido de tablas de Wilcoxon.
#    d = [1..10] todos positivos -> W=55, p exacto = 2/2^10 = 0.001953
W, p = wilcoxon(list(range(1, 11)))
check(abs(W - 55) < 1e-9 and abs(p - 2 / 1024) < 1e-9, f"exacto, todos positivos: W={W} p={p:.6f} (esperado 0.001953)")

# 2. Simetrico perfecto -> p = 1
W, p = wilcoxon([1, -1, 2, -2, 3, -3])
check(abs(p - 1.0) < 1e-9, f"simetrico: p={p:.4f} (esperado 1)")

# 3. Un caso chico verificable a mano: d=[1,2,-3] -> rangos 1,2,3; W+=3
#    |W+ - 3| = 0 -> todas las asignaciones cuentan -> p=1
W, p = wilcoxon([1, 2, -3])
check(abs(W - 3) < 1e-9 and abs(p - 1.0) < 1e-9, f"n=3 a mano: W={W} p={p:.4f}")

# 4. Aproximacion normal contra exacta en n=20 (el limite del exacto)
rng = np.random.default_rng(0)
d = rng.normal(0.4, 1.0, 20)
_, p_ex = wilcoxon(d)
import paired
n = len(d); r = paired._ranks(np.abs(d)); Wp = float(r[np.asarray(d) > 0].sum())
mu = n * (n + 1) / 4.0; var = n * (n + 1) * (2 * n + 1) / 24.0
from math import erf, sqrt
z = (abs(Wp - mu) - 0.5) / sqrt(var); p_nm = 2 * (1 - 0.5 * (1 + erf(z / sqrt(2))))
check(abs(p_ex - p_nm) < 0.02, f"exacta vs normal en n=20: {p_ex:.4f} vs {p_nm:.4f}")

# 5. Calibracion bajo H0: con datos simetricos, el p del sign-flip debe ser ~uniforme
ps = [sign_flip(np.random.default_rng(s).normal(0, 1, 30), n_perm=2000, seed=s) for s in range(300)]
frac = np.mean(np.array(ps) < 0.05)
check(0.02 < frac < 0.09, f"calibracion H0 del sign-flip: {100*frac:.1f}% de rechazos al 5% (esperado ~5%)")

# 6. Potencia: efecto claro -> p chico en ambos tests
d = np.random.default_rng(1).normal(1.0, 1.0, 30)
_, pw = wilcoxon(d); pf = sign_flip(d, seed=1)
check(pw < 0.001 and pf < 0.001, f"efecto grande: wilcoxon={pw:.1e} sign-flip={pf:.1e}")

# 7. IC bootstrap cubre la media real
lo, hi = boot_ci(d, seed=0)
check(lo < d.mean() < hi, f"IC bootstrap contiene la media: [{lo:.3f}, {hi:.3f}] vs {d.mean():.3f}")

# 8. Holm: monotono y >= p original
adj = holm([0.001, 0.02, 0.04, 0.5])
check(np.all(adj >= [0.001, 0.02, 0.04, 0.5]) and np.all(np.diff(adj) >= -1e-12), f"Holm: {np.round(adj, 4)}")
print("\nTODO OK" if ok else "\nHAY FALLAS")
sys.exit(0 if ok else 1)
