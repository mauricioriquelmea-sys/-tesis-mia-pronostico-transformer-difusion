"""
diag_calibracion.py  --  por que C3 falla en AMBAS cabezas y tambien en R0-G
=============================================================================

C3 exige cobertura al 90 % dentro de [0.80, 0.95]. En la corrida del 12 de
octubre la referencia R0-G dio 0.311 y la difusion en y dio 0.995. Una
sub-cubre por un factor grande, la otra sobre-cubre casi hasta el tope. Con
la referencia fuera de rango, el criterio no distingue "la difusion esta mal
calibrada" de "ninguna cabeza se calibra con este objetivo y este tamano
muestral". Esto lo resuelve.

Tres mediciones, en el orden del proyecto: dimensiones -> magnitudes ->
estabilidad.

  D1  Cobertura desagregada por horizonte y por serie, y razon de nitidez
      (desviacion de las muestras / desviacion de los realizados), POR SERIE
      --- la razon agregada es inservible porque las tres series difieren en
      un orden de magnitud.

  D2  Barrido de pasos de entrenamiento. Si la sub-cobertura de R0-G viene
      de sobreajuste (118 ventanas, objetivo de 36 dimensiones), la
      cobertura debe MEJORAR al reducir los pasos y empeorar al aumentarlos.

  D3  Cota inferior del CRPS: que CRPS alcanza un oraculo gaussiano ajustado
      sobre los residuos VERDADEROS del generador. Fija contra que se compara.

AISLAMIENTO: solo el generador sintetico. No se toca el panel real.
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(1)

from bloques_h1 import _panel_sintetico, L, H_FUT, N, D
from cabeza_difusion import CabezaDifusion
from cabeza_r0 import CabezaR0
from diag_difusion import entrenar, _evaluar, cobertura
from h1_harness import joint_crps

T_PANEL, T0, NO, M, REFIT = 300, 144, 36, 200, 12
NIVELES = (0.50, 0.80, 0.90)


def cobertura_desagregada(ens, y, nivel=0.90):
    """Devuelve (global, por_horizonte [H], por_serie [N])."""
    a = (1 - nivel) / 2
    lo = np.quantile(ens, a, axis=-1)
    hi = np.quantile(ens, 1 - a, axis=-1)
    dentro = (y >= lo) & (y <= hi)                     # [O, H, N]
    return (float(dentro.mean()),
            dentro.mean(axis=(0, 2)).tolist(),
            dentro.mean(axis=(0, 1)).tolist())


def nitidez_por_serie(ens, y):
    """sd de las muestras alrededor de su media / sd de los realizados,
    por serie. Se calcula POR SERIE porque las escalas difieren."""
    sd_m = ens.std(axis=-1).mean(axis=(0, 1))          # [N]
    # dispersion de los realizados alrededor del pronostico central
    err = np.abs(y - ens.mean(axis=-1))
    return sd_m.tolist(), err.mean(axis=(0, 1)).tolist()


def fabricas(T_dif=100):
    return {
        "R0-G": lambda: CabezaR0(latent_dim=D, rank=5, dist_type="gaussian",
                                 v_condicional=False),
        "R0-T": lambda: CabezaR0(latent_dim=D, rank=5, dist_type="student-t",
                                 v_condicional=False),
        "dif-y": lambda: CabezaDifusion(latent_dim=D, espacio="y", T_dif=T_dif),
        "dif-lat": lambda: CabezaDifusion(latent_dim=D, espacio="latente",
                                          T_dif=T_dif),
    }


def main():
    ts = datetime.now().isoformat(timespec="seconds")
    panel = _panel_sintetico(T=T_PANEL, seed=12345)
    out = {"timestamp": ts, "torch": torch.__version__, "D1": {}, "D2": {}, "D3": {}}
    cfg = dict(T0=T0, N_ORIGENES=NO, M=M, REFIT=REFIT, STEPS=400, T_DIF=100,
               PASOS_MUESTREO=None)

    print("=" * 78)
    print("DIAGNOSTICO DE CALIBRACION  -  panel sintetico")
    print("=" * 78)
    print(f"  {ts}   T0={T0}  origenes={NO}  M={M}  pasos=400\n")

    # ---------------------------------------------------------------- D3
    # Cota inferior: el generador tiene ruido gaussiano conocido. Un oraculo
    # que conoce la dinamica exacta y propaga la varianza correctamente da el
    # CRPS minimo alcanzable. Sin esa cifra, "CRPS = 2.92" no significa nada.
    print("-" * 78)
    print("D3  Cota inferior del CRPS: oraculo que conoce el generador")
    print("-" * 78)
    rng = np.random.default_rng(999)
    cambios = np.diff(panel, axis=1)
    ens_o = np.empty((NO, H_FUT, N, M)); y_o = np.empty((NO, H_FUT, N))
    sc_o = np.empty((NO, N))
    # el generador es un VAR(1) triangular conocido sobre (cu, fx, tpm)
    A = np.array([[.85, 0, 0], [-.50, .80, 0], [0, .40, .90]])
    s_e = np.array([1.0, .50, .30])
    esc = np.array([.4, 40., 2.]); ctr = np.array([3.5, 850., 5.])
    for o_i in range(NO):
        origen = T0 + o_i
        sc_o[o_i] = cambios[:, :origen].std(axis=1)
        u = (panel[:, origen] - ctr) / esc                   # estado latente
        tray = np.empty((M, H_FUT, N))
        for m_ in range(M):
            v = u.copy()
            for h in range(H_FUT):
                v = A @ v + rng.normal(0, s_e)
                tray[m_, h] = v * esc + ctr
        ens_o[o_i] = tray.transpose(1, 2, 0)
        y_o[o_i] = panel[:, origen + 1: origen + 1 + H_FUT].T
    crps_o = float(joint_crps(ens_o, y_o, sc_o, standardize=True).mean())
    cob_o, cob_h_o, cob_n_o = cobertura_desagregada(ens_o, y_o)
    print(f"  CRPS conjunto estandarizado del ORACULO = {crps_o:.4f}")
    print(f"  cobertura 90 % del oraculo = {cob_o:.3f}  "
          f"(por serie: {[round(x,3) for x in cob_n_o]})")
    print("  Esta es la cota: ninguna cabeza puede bajar de aqui de forma")
    print("  sistematica, y el oraculo SI debe estar calibrado.\n")
    out["D3"] = {"crps_oraculo": crps_o, "cobertura_oraculo": cob_o,
                 "cobertura_por_serie": cob_n_o, "cobertura_por_horizonte": cob_h_o}

    # ---------------------------------------------------------------- D1
    print("-" * 78)
    print("D1  Cobertura desagregada y nitidez por serie")
    print("-" * 78)
    guardado = {}
    for nom, fab in fabricas().items():
        t0 = time.time()
        ens, y, sc = _evaluar(panel, fab, 0, cfg)
        crps = float(joint_crps(ens, y, sc, standardize=True).mean())
        guardado[nom] = (ens, y, sc)
        print(f"\n  {nom}   CRPS = {crps:.4f}   "
              f"({100*(crps/crps_o-1):+.1f} % sobre el oraculo)   "
              f"({time.time()-t0:.0f}s)")
        fila = {"crps": crps, "exceso_sobre_oraculo": crps / crps_o - 1.0}
        for niv in NIVELES:
            g, ch, cn = cobertura_desagregada(ens, y, niv)
            print(f"    cobertura {int(niv*100)} %: global {g:.3f}   "
                  f"por serie {[round(x,3) for x in cn]}")
            fila[f"cob_{int(niv*100)}"] = {"global": g, "por_serie": cn,
                                           "por_horizonte": ch}
        g, ch, cn = cobertura_desagregada(ens, y, 0.90)
        print(f"    cobertura 90 % por horizonte h=1..12: "
              f"{[round(x,2) for x in ch]}")
        sd_m, err_m = nitidez_por_serie(ens, y)
        raz = [s / e for s, e in zip(sd_m, err_m)]
        print(f"    sd muestras / error medio, por serie: "
              f"{[round(r,2) for r in raz]}")
        print(f"      (1.25 es lo esperado si esta calibrado: E|Z| = 0.80 sd)")
        fila["nitidez"] = {"sd_muestras": sd_m, "error_medio": err_m,
                           "razon": raz}
        out["D1"][nom] = fila

    # ---------------------------------------------------------------- D2
    print("\n" + "-" * 78)
    print("D2  Barrido de pasos de entrenamiento  (hipotesis: sobreajuste)")
    print("-" * 78)
    print("  Si la sub-cobertura de R0 viene de sobreajuste, debe MEJORAR al")
    print("  reducir los pasos. Si no se mueve, la causa es otra.\n")
    print(f"  {'cabeza':<10}{'pasos':>7}{'CRPS':>10}{'cob 90%':>10}{'cob 50%':>10}")
    print("  " + "-" * 45)
    for nom in ("R0-G", "dif-y"):
        for st in (50, 100, 200, 400, 800):
            c2 = dict(cfg); c2["STEPS"] = st
            ens, y, sc = _evaluar(panel, fabricas()[nom], 0, c2)
            crps = float(joint_crps(ens, y, sc, standardize=True).mean())
            c90, _, _ = cobertura_desagregada(ens, y, 0.90)
            c50, _, _ = cobertura_desagregada(ens, y, 0.50)
            print(f"  {nom:<10}{st:>7}{crps:>10.4f}{c90:>10.3f}{c50:>10.3f}")
            out["D2"].setdefault(nom, []).append(
                {"pasos": st, "crps": crps, "cob_90": c90, "cob_50": c50})
    print()

    p = Path(f"diag_calibracion_{ts[:10]}.json")
    p.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"  guardado en {p}")


if __name__ == "__main__":
    main()
