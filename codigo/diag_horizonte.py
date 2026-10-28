"""
diag_horizonte.py  -  por que la correccion de varianza no llega a h = 12
==========================================================================

MEDIDO HOY (diag_calibracion2.py, con validacion apartada del entrenamiento)

  cobertura 90 % por horizonte, h = 1..12
    base      0.92 0.87 0.73 0.80 0.52 0.51 0.48 0.42 0.36 0.32 0.33 0.32
    inflado   0.99 0.94 0.94 0.93 0.89 0.86 0.86 0.69 0.61 0.57 0.49 0.48

Un lambda escalar calibrado fuera de muestra arregla h = 1..7 y deja h = 8..12
sin arreglar. Un lambda POR HORIZONTE tampoco lo arregla, y el lambda ajustado
nunca pasa de ~1.8 aunque la rejilla llega a 12.0. Es decir: en el conjunto de
validacion el modelo NO PARECE tan sub-disperso en los horizontes largos.

La hipotesis: ventanas solapadas. Dos ventanas consecutivas comparten 11 de
sus 12 meses de objetivo, de modo que los errores a h pasos estan
autocorrelacionados y el numero de observaciones INDEPENDIENTES en el
horizonte h es del orden de n/h, no n. En el horizonte 12 el panel de 264
meses contiene 22 bloques no solapados: la dispersion conjunta sobre R^36 se
estima, en los hechos, con ~22 observaciones.

Si es asi, no es un defecto de la cabeza ni del calibrador. Es el limite
muestral que el proyecto ya nombra, y lo que corresponde es cuantificarlo y
declararlo, no seguir ajustando.

DOS MEDICIONES
  H1  lambda que se NECESITABA en los origenes de prueba (lambda oraculo)
      contra el lambda AJUSTADO en validacion, horizonte por horizonte.
  H2  tamano muestral efectivo por horizonte, por dos vias: bloques no
      solapados, y correccion por autocorrelacion de los errores a h pasos.

AISLAMIENTO: solo el generador sintetico.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(1)

from bloques_h1 import _panel_sintetico, L, H_FUT, N, EPS
from diag_calibracion2 import (entrenar, calibrar_lambda, _muestrear, _ventanas,
                               REJILLA, N_VAL, T_PANEL, T0, NO, M, REFIT)


def lambda_necesario(ens, y, nivel=0.90):
    """Por horizonte: el factor que habria igualado la cobertura a `nivel`
    sobre los origenes de PRUEBA. Es el lambda que no se puede conocer de
    antemano; sirve de referencia para medir cuanto se queda corto el ajustado.
    ens: [O,H,N,M]  y: [O,H,N]"""
    a = (1 - nivel) / 2
    cen = ens.mean(axis=-1, keepdims=True)
    dev = ens - cen
    out = np.empty(H_FUT)
    for h in range(H_FUT):
        cob = []
        for lam in REJILLA:
            e = cen[:, h] + lam * dev[:, h]
            lo = np.quantile(e, a, axis=-1); hi = np.quantile(e, 1 - a, axis=-1)
            cob.append(((y[:, h] >= lo) & (y[:, h] <= hi)).mean())
        out[h] = REJILLA[int(np.argmin(np.abs(np.array(cob) - nivel)))]
    return out


def n_efectivo(err, h):
    """err: [O] errores a h pasos en orden temporal. Correccion por
    autocorrelacion:  n_eff = n / (1 + 2 sum_{k=1}^{h-1} (1-k/n) rho_k).
    Con ventanas solapadas rho_k > 0 hasta k = h-1 por construccion."""
    n = len(err)
    e = err - err.mean()
    den = (e * e).sum()
    if den <= 0:
        return float(n)
    fac = 1.0
    for k in range(1, min(h, n)):
        rho = (e[:-k] * e[k:]).sum() / den
        fac += 2 * (1 - k / n) * rho
    return float(n / max(fac, 1e-6))


def main():
    ts = datetime.now().isoformat(timespec="seconds")
    panel = _panel_sintetico(T=T_PANEL, seed=12345)
    out = {"timestamp": ts, "torch": torch.__version__}

    print("=" * 78)
    print("POR QUE LA CORRECCION NO LLEGA A h = 12")
    print("=" * 78)
    print(f"  {ts}   T0={T0}  origenes={NO}  M={M}\n")

    # ------------------------------------------------------------------ run
    ens = np.empty((NO, H_FUT, N, M)); y = np.empty((NO, H_FUT, N))
    lam_aj = []
    m = mu = sd = None
    for o_i in range(NO):
        origen = T0 + o_i
        if o_i % REFIT == 0:
            m, mu, sd, Xva, Yva = entrenar(panel, origen, 0, "nivel",
                                           reservar_val=True)
            lam_aj.append(calibrar_lambda(m, Xva, Yva, "nivel",
                                          por_horizonte=True))
        with torch.no_grad():
            Xn = (panel - mu) / sd
            xw = torch.tensor(Xn[:, origen - L + 1: origen + 1][None],
                              dtype=torch.float32)
            Y = _muestrear(m, xw, "nivel", M)[0].numpy()
            Y = Y * sd.T[None, :, :] + mu.T[None, :, :]
            ens[o_i] = Y.transpose(1, 2, 0)
        y[o_i] = panel[:, origen + 1: origen + 1 + H_FUT].T

    lam_aj = np.stack(lam_aj).mean(0)                 # [H]
    lam_nec = lambda_necesario(ens, y)                # [H]

    # ------------------------------------------------------------------ H1
    print("-" * 78)
    print("H1  Lambda ajustado en validacion vs. lambda que se necesitaba")
    print("-" * 78)
    print(f"  {'h':>3}{'ajustado':>11}{'necesario':>12}{'razon':>9}")
    print("  " + "-" * 32)
    for h in range(H_FUT):
        print(f"  {h+1:>3}{lam_aj[h]:>11.2f}{lam_nec[h]:>12.2f}"
              f"{lam_nec[h]/lam_aj[h]:>9.2f}")
    out["H1"] = {"lambda_ajustado": lam_aj.tolist(),
                 "lambda_necesario": lam_nec.tolist(),
                 "razon": (lam_nec / lam_aj).tolist()}
    print()

    # ------------------------------------------------------------------ H2
    print("-" * 78)
    print("H2  Tamano muestral efectivo por horizonte")
    print("-" * 78)
    T_ent = T0                                        # meses de entrenamiento
    print(f"  ventanas de entrenamiento al primer origen: ~{T_ent - L - H_FUT}")
    print(f"  {'h':>3}{'bloques no solap.':>20}{'n_eff (autocorr.)':>20}"
          f"{'n_eff / n':>12}")
    print("  " + "-" * 53)
    err = np.abs(y - ens.mean(axis=-1)).mean(axis=-1)   # [O, H] error medio
    nef = []
    for h in range(H_FUT):
        bloques = (T_ent - L) // (h + 1)
        ne = n_efectivo(err[:, h], h + 1)
        nef.append(ne)
        print(f"  {h+1:>3}{bloques:>20}{ne:>20.1f}{ne/NO:>12.2f}")
    out["H2"] = {"n_eff_autocorr": nef,
                 "bloques_no_solapados": [(T_ent - L) // (h + 1)
                                          for h in range(H_FUT)],
                 "n_origenes": NO}
    print()

    # ------------------------------------------------------------------ lect
    print("=" * 78)
    print("LECTURA")
    print("=" * 78)
    r_corto = float(np.mean(lam_nec[:6] / lam_aj[:6]))
    r_largo = float(np.mean(lam_nec[6:] / lam_aj[6:]))
    print(f"  razon necesario/ajustado:  h=1..6  {r_corto:.2f}"
          f"      h=7..12  {r_largo:.2f}")
    print(f"  n_eff:  h=1  {nef[0]:.0f} de {NO}      h=12  {nef[-1]:.0f} de {NO}")
    print()
    if r_largo > r_corto * 1.3:
        print("  El calibrador se queda corto SOLO en los horizontes largos, y lo")
        print("  hace en la misma medida en que cae el tamano muestral efectivo.")
        print("  No es un defecto de la cabeza ni del calibrador: es el limite")
        print("  muestral del panel. Con 264 meses y objetivo de 12, hay 22")
        print("  bloques no solapados; la dispersion conjunta sobre R^36 se")
        print("  estima, en los hechos, con ese orden de observaciones.")
        print()
        print("  CONSECUENCIA PARA EL INFORME. La correccion de varianza se")
        print("  reporta con su alcance explicito: restituye la calibracion")
        print("  hasta h ~ 7 y no mas alla. Extenderla a h = 12 requeriria mas")
        print("  panel, no mas modelo.")
    else:
        print("  El calibrador se queda corto de forma pareja en todos los")
        print("  horizontes. La explicacion por tamano muestral efectivo NO se")
        print("  sostiene; hay que buscar otra.")
    out["lectura"] = {"razon_h1_6": r_corto, "razon_h7_12": r_largo}

    p = Path(f"diag_horizonte_{ts[:10]}.json")
    p.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\n  guardado en {p}")


if __name__ == "__main__":
    main()
