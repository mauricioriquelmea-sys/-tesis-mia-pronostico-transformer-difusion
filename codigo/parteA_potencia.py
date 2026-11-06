"""
parteA_potencia.py  -  Parte A del barrido: tamano y potencia de Diebold-Mariano
================================================================================

Implementa resultados/A.1/PREREGISTRO_BARRIDO_ParteA.md (fijado en 19af5f2).
No entrena ninguna red. No lee el panel real.

QUE HACE
--------
Para cada replica r = 1..R:
  1. Genera un panel sintetico con bloques_h1._panel_sintetico (sin modificar).
  2. Construye en cada origen la predictiva EXACTA a h pasos del VAR(1) que
     genera el panel (pronosticador B, "oraculo") y la misma predictiva con su
     dispersion inflada por k_delta (pronosticador A, "degradado").
  3. Evalua A y B con el protocolo de la tesis: CRPS conjunto estandarizado
     (h1_harness.joint_crps, mismas formulas) y Diebold-Mariano unilateral con
     HAC de 11 rezagos y ajuste HLN (h1_harness.diebold_mariano, sin modificar).
Luego cuenta rechazos por celda (delta, O) y aplica las tres reglas fijadas.

CALIBRACION DE k_delta (analitica, antes de ejecutar)
-----------------------------------------------------
Si y ~ N(mu, s^2) y la predictiva es N(mu, (k s)^2):
    E[CRPS] = s * g(k),   g(k) = sqrt(2/pi) * sqrt(1 + k^2) - k / sqrt(pi)
g es lineal en s, asi que la reduccion relativa de B (k=1) frente a A (k) es la
misma en cada serie y horizonte:  delta = 1 - g(1)/g(k).  Se resuelve k_delta
por biseccion. La simulacion verifica que la reduccion realizada sea ~delta.

DETALLES DE IMPLEMENTACION (declarados en la seccion 7 del prerregistro)
----------------------------------------------------------------------
- Paneles anidados entre O: cada replica genera un panel de T0+450+12 meses y
  las celdas con O < 450 usan sus primeros O origenes. Cada celda sigue teniendo
  R paneles independientes; las celdas de distinto O de una misma replica no lo
  son entre si (numeros aleatorios comunes, como A y B).
- CRPS: el estimador insesgado de h1_harness se calcula con la identidad de
  muestras ordenadas, sum_ij |x_i - x_j| = 2 sum_i (2i - M - 1) x_(i), que da
  el MISMO valor sin construir el tensor (O,H,N,M,M). Se verifica al inicio.
- Ley 1/raiz(O): la pendiente se estima para cada delta; la regla se cumple
  si la pendiente cae en la banda para todos los delta.
- Factor 1,51: O50 y O80 se interpolan en log O. Si la potencia no cruza 50 %
  u 80 % dentro de la grilla para un delta, ese delta se reporta como no
  identificable y no entra en la regla. Como lectura descriptiva (fuera de la
  regla) se reportan delta80/delta50 interpolado en delta y delta80 / efecto
  critico mediano, para cada O.
- Celda delta = 0: si A y B fueran el mismo oraculo, el diferencial seria
  identicamente nulo y la prueba nunca rechazaria. Bajo la nula se comparan dos
  versiones del oraculo con IGUAL CRPS esperado: B subdispersa (k = 0,8) y A
  sobredispersa (k tal que g(k) = g(0,8)). El diferencial depende de la magnitud
  de los errores, que se solapan entre horizontes, de modo que el HAC se pone
  a prueba con autocorrelacion real.

USO (desde la raiz del repositorio)
-----------------------------------
    python codigo/parteA_potencia.py --rapido     # prueba: R=40, ~1 min
    python codigo/parteA_potencia.py              # completo: R=2000
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
from datetime import datetime

import numpy as np
from scipy import stats

from bloques_h1 import _panel_sintetico, H_FUT
from h1_harness import crps_ensemble, diebold_mariano

# --------------------------------------------------------------------------
# Grilla y reglas fijadas en el prerregistro (no se mueven)
# --------------------------------------------------------------------------
DELTAS = [0.00, 0.04, 0.06, 0.08, 0.10]
ORIGENES = [109, 150, 200, 300, 450]
R_COMPLETO = 2000
T0 = 144
M = 200
ALPHA = 0.05
BANDA_TAMANO = (0.035, 0.065)
BANDA_PENDIENTE = (-0.60, -0.40)
BANDA_RAZON = (1.9, 2.7)

# --------------------------------------------------------------------------
# El VAR(1) de _panel_sintetico, escrito en forma matricial
#   z_t = A z_{t-1} + e_t,  e_t ~ N(0, S)
#   panel_t = c + D z_t
# --------------------------------------------------------------------------
A_VAR = np.array([[0.85, 0.00, 0.00],
                  [-0.50, 0.80, 0.00],
                  [0.00, 0.40, 0.90]])
S_VAR = np.diag([1.0 ** 2, 0.5 ** 2, 0.3 ** 2])
C_OBS = np.array([3.5, 850.0, 5.0])
D_OBS = np.array([0.4, 40.0, 2.0])


def momentos_oraculo(H: int):
    """Potencias A^h y desviaciones marginales a h pasos en escala del panel."""
    pot, sd = [], []
    Ah = np.eye(3)
    Sig = np.zeros((3, 3))
    for _ in range(H):
        Sig = A_VAR @ Sig @ A_VAR.T + S_VAR          # Var(z_{o+h} | z_o)
        Ah = A_VAR @ Ah                              # A^h
        pot.append(Ah.copy())
        sd.append(D_OBS * np.sqrt(np.diag(Sig)))
    return np.stack(pot), np.stack(sd)               # (H,3,3), (H,3)


def g_crps(k: float) -> float:
    return math.sqrt(2 / math.pi) * math.sqrt(1 + k * k) - k / math.sqrt(math.pi)


K_NULO_B = 0.8      # con delta = 0: B subdisperso; A sobredisperso con igual CRPS esperado


def k_para_delta(delta: float) -> float:
    """Factor de A. Con delta = 0, el k > 1 que iguala g(K_NULO_B)."""
    objetivo = g_crps(K_NULO_B) if delta == 0.0 else g_crps(1.0) / (1.0 - delta)
    lo, hi = 1.0, 50.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if g_crps(mid) < objetivo:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# --------------------------------------------------------------------------
# CRPS insesgado con muestras ordenadas: identico a h1_harness.crps_ensemble
# --------------------------------------------------------------------------
def termino2_ordenado(ens_ordenado: np.ndarray) -> np.ndarray:
    Mloc = ens_ordenado.shape[-1]
    i = np.arange(1, Mloc + 1)
    suma = 2.0 * ((2 * i - Mloc - 1) * ens_ordenado).sum(-1)    # sum_ij |xi-xj|
    return suma / (2.0 * Mloc * (Mloc - 1))


def verificar_crps():
    rng = np.random.default_rng(7)
    ens = rng.normal(size=(5, 4, 3, 50)); y = rng.normal(size=(5, 4, 3))
    ref = crps_ensemble(ens, y)
    t1 = np.abs(ens - y[..., None]).mean(-1)
    rapido = t1 - termino2_ordenado(np.sort(ens, -1))
    err = np.abs(ref - rapido).max()
    if err > 1e-10:
        raise AssertionError(f"CRPS ordenado difiere de h1_harness: {err:.2e}")
    return float(err)


# --------------------------------------------------------------------------
# Una replica: perdidas por origen de A (por delta) y de B
# --------------------------------------------------------------------------
def replica(r: int, pot, sdh, ks, Omax: int):
    T = T0 + Omax + H_FUT
    panel = _panel_sintetico(T=T, seed=100_000 + r)              # [3, T]
    z = (panel - C_OBS[:, None]) / D_OBS[:, None]                # estado latente
    cambios = np.diff(panel, axis=1)
    rng = np.random.default_rng(900_000 + r)

    orig = T0 + np.arange(Omax)                                  # indices de origen
    zo = z[:, orig].T                                            # (O,3)
    mu = C_OBS + D_OBS * np.einsum("hij,oj->ohi", pot, zo)        # (O,H,3)
    y = np.stack([panel[:, o + 1: o + 1 + H_FUT].T for o in orig])   # (O,H,3)
    # escala: desviacion de los CAMBIOS hasta cada origen (como bloques_h1)
    scale = np.stack([cambios[:, :o].std(axis=1) for o in orig])     # (O,3)

    eps = np.sort(rng.standard_normal((Omax, H_FUT, 3, M)), axis=-1)  # comun a A y B
    dev = sdh[None, :, :, None] * eps                            # desviaciones de B
    t2_b = termino2_ordenado(dev)                                # (O,H,3), invariante a mu

    def perdida(k):
        ens = mu[..., None] + k * dev
        t1 = np.abs(ens - y[..., None]).mean(-1)
        crps = t1 - k * t2_b                                     # orden de dev se conserva con k>0
        return (crps / scale[:, None, :]).mean(-1)               # joint_crps estandarizado (O,H)

    jb = perdida(1.0)                          # oraculo
    jb0 = perdida(K_NULO_B)                    # B bajo la nula (delta = 0)
    ja = {d: perdida(k) for d, k in ks.items()}
    return ja, jb, jb0


def critico_relativo(la, lb, h):
    """Efecto critico t_0.95 * se_HLN / CRPS_A, con el mismo HAC que h1_harness."""
    d = la - lb
    n = d.size
    u = d - d.mean()
    L = h - 1
    var = (u @ u) / n
    for k in range(1, L + 1):
        var += 2.0 * (1.0 - k / (L + 1.0)) * ((u[k:] @ u[:-k]) / n)
    if var <= 0:
        var = (u @ u) / n
    corr = math.sqrt(max((n + 1 - 2 * h + h * (h - 1) / n) / n, 1e-12))
    se_eff = math.sqrt(var / n) / corr
    return stats.t.ppf(1 - ALPHA, n - 1) * se_eff / la.mean()


# --------------------------------------------------------------------------
# Reglas
# --------------------------------------------------------------------------
def interpolar_cruce(Os, pot, nivel):
    lo = np.log(np.asarray(Os, float))
    for i in range(len(Os) - 1):
        a, b = pot[i], pot[i + 1]
        if a < nivel <= b:
            return float(np.exp(lo[i] + (nivel - a) / (b - a) * (lo[i + 1] - lo[i])))
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rapido", action="store_true", help="R=40, solo para probar")
    ap.add_argument("--replicas", type=int, default=None)
    args = ap.parse_args()
    R = args.replicas or (40 if args.rapido else R_COMPLETO)

    print("=" * 72)
    print(f"PARTE A  -  tamano y potencia de DM-HLN   R = {R}"
          + ("   (RAPIDO: no reportar)" if R < R_COMPLETO else ""))
    print("=" * 72)
    err = verificar_crps()
    print(f"  CRPS ordenado = h1_harness.crps_ensemble   (error max {err:.1e})")

    pot, sdh = momentos_oraculo(H_FUT)
    ks = {d: k_para_delta(d) for d in DELTAS}
    for d, k in ks.items():
        print(f"  delta = {d:4.0%}   k_delta = {k:.6f}")

    Omax = max(ORIGENES)
    rech = {(d, O): 0 for d in DELTAS for O in ORIGENES}
    crit = {(d, O): [] for d in DELTAS for O in ORIGENES}
    red = {(d, O): [] for d in DELTAS for O in ORIGENES}
    t0 = time.time()

    for r in range(R):
        ja, jb, jb0 = replica(r, pot, sdh, ks, Omax)
        for O in ORIGENES:
            for d in DELTAS:
                lb = (jb0 if d == 0.0 else jb)[:O].mean(axis=1)
                la = ja[d][:O].mean(axis=1)
                dm = diebold_mariano(la, lb, h=H_FUT)          # h1_harness, sin modificar
                rech[(d, O)] += dm["p_one_sided"] < ALPHA
                crit[(d, O)].append(critico_relativo(la, lb, H_FUT))
                red[(d, O)].append((la.mean() - lb.mean()) / la.mean())
        if (r + 1) % max(1, R // 10) == 0:
            print(f"  replica {r + 1:5d}/{R}   {time.time() - t0:7.1f} s")

    celdas = []
    for d in DELTAS:
        for O in ORIGENES:
            p = rech[(d, O)] / R
            celdas.append({"delta": d, "O": O, "tasa_rechazo": p,
                           "error_mc": math.sqrt(p * (1 - p) / R),
                           "efecto_critico_mediano": float(np.median(crit[(d, O)])),
                           "reduccion_realizada_media": float(np.mean(red[(d, O)]))})

    def celda(d, O):
        return next(c for c in celdas if c["delta"] == d and c["O"] == O)

    # Regla 1: tamano
    tam = {O: celda(0.0, O)["tasa_rechazo"] for O in ORIGENES}
    r1 = all(BANDA_TAMANO[0] <= v <= BANDA_TAMANO[1] for v in tam.values())

    # Regla 2: pendiente de log(efecto critico mediano) sobre log O, por delta
    pend = {}
    for d in DELTAS:
        x = np.log(ORIGENES); yv = np.log([celda(d, O)["efecto_critico_mediano"] for O in ORIGENES])
        pend[d] = float(np.polyfit(x, yv, 1)[0])
    r2 = all(BANDA_PENDIENTE[0] <= v <= BANDA_PENDIENTE[1] for v in pend.values())

    # Regla 3: razon O80/O50 por delta > 0
    razones = {}
    for d in DELTAS[1:]:
        pw = [celda(d, O)["tasa_rechazo"] for O in ORIGENES]
        o50, o80 = interpolar_cruce(ORIGENES, pw, 0.5), interpolar_cruce(ORIGENES, pw, 0.8)
        razones[d] = {"O50": o50, "O80": o80,
                      "razon": (o80 / o50) if (o50 and o80) else None}
    ident = [v["razon"] for v in razones.values() if v["razon"] is not None]
    r3 = (all(BANDA_RAZON[0] <= v <= BANDA_RAZON[1] for v in ident) if ident else None)

    # Descriptivo: delta80/delta50 interpolado en delta, para cada O
    razon_delta = {}
    for O in ORIGENES:
        dd = DELTAS[1:]; pw = [celda(d, O)["tasa_rechazo"] for d in dd]
        def cruce(nivel):
            for i in range(len(dd) - 1):
                if pw[i] < nivel <= pw[i + 1]:
                    return dd[i] + (nivel - pw[i]) / (pw[i + 1] - pw[i]) * (dd[i + 1] - dd[i])
            return None
        d50, d80 = cruce(0.5), cruce(0.8)
        # el efecto critico mediano es, por definicion, el delta con potencia ~50 %
        crit_med = float(np.median([celda(d, O)["efecto_critico_mediano"] for d in dd]))
        razon_delta[O] = {"delta50": d50, "delta80": d80, "efecto_critico_mediano": crit_med,
                          "razon": (d80 / d50) if (d50 and d80) else None,
                          "razon_delta80_critico": (d80 / crit_med) if d80 else None}

    print("\n  Tasa de rechazo (filas delta, columnas O)")
    print("          " + "".join(f"{O:>9d}" for O in ORIGENES))
    for d in DELTAS:
        print(f"  {d:6.0%}  " + "".join(f"{celda(d, O)['tasa_rechazo']:9.3f}" for O in ORIGENES))
    print("\n  Reduccion realizada media (verificacion: debe ser ~delta)")
    for d in DELTAS:
        print(f"  {d:6.0%}  " + "".join(f"{celda(d, O)['reduccion_realizada_media']:9.4f}" for O in ORIGENES))
    print(f"\n  REGLA 1 tamano     {tam}  -> {'SE SOSTIENE' if r1 else 'NO SE SOSTIENE'}")
    print(f"  REGLA 2 pendiente  { {k: round(v, 3) for k, v in pend.items()} }  -> {'SE SOSTIENE' if r2 else 'NO SE SOSTIENE'}")
    print(f"  REGLA 3 razon      { {k: (round(v['razon'], 3) if v['razon'] else None) for k, v in razones.items()} }"
          f"  -> {'SE SOSTIENE' if r3 else ('NO IDENTIFICABLE' if r3 is None else 'NO SE SOSTIENE')}")
    print(f"  Descriptivo delta80/delta50 (teorico 1,51): "
          f"{ {O: (round(v['razon'], 3) if v['razon'] else None) for O, v in razon_delta.items()} }")
    print(f"  Descriptivo delta80/efecto critico (teorico 1,51): "
          f"{ {O: (round(v['razon_delta80_critico'], 3) if v['razon_delta80_critico'] else None) for O, v in razon_delta.items()} }")

    salida = {
        "prerregistro": "resultados/A.1/PREREGISTRO_BARRIDO_ParteA.md (commit 19af5f2)",
        "completo": R >= R_COMPLETO, "R": R, "T0": T0, "M": M, "H": H_FUT,
        "deltas": DELTAS, "origenes": ORIGENES, "k_delta": ks,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "entorno": {"python": sys.version.split()[0], "numpy": np.__version__,
                    "plataforma": platform.platform()},
        "celdas": celdas,
        "reglas": {
            "tamano": {"banda": BANDA_TAMANO, "valores": tam, "se_sostiene": r1},
            "pendiente": {"banda": BANDA_PENDIENTE, "por_delta": pend, "se_sostiene": r2},
            "razon_O80_O50": {"banda": BANDA_RAZON, "por_delta": razones,
                              "se_sostiene": r3},
        },
        "descriptivo_delta80_delta50": {"teorico": 1.51, "por_O": razon_delta},
        "k_nulo_B": K_NULO_B,
        "segundos": round(time.time() - t0, 1),
    }
    nombre = f"parteA_potencia_{'full' if R >= R_COMPLETO else 'prueba'}_{datetime.now():%Y-%m-%d}.json"
    with open(nombre, "w", encoding="utf-8") as f:
        json.dump(salida, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n  -> {nombre}")


if __name__ == "__main__":
    main()
