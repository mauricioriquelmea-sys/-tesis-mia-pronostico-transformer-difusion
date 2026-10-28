"""
medir_frac_real.py  -  que fraccion estructural tuvo el contraste de H1
========================================================================

Veinte segundos. No entrena: da un solo paso en cada origen de reajuste y lee
L_simple y R_G del primer lote, que es donde el prerregistro (A.1) define la
calibracion de eta:

    eta * R_G / (L_simple + eta * R_G) = 0,20

Responde una sola pregunta: con eta = 5, el valor que el informe reporta, que
fraccion alcanzo efectivamente el contraste sobre el PANEL REAL.

Sobre el panel sintetico en T0 = 144 esa fraccion vale 0,011. Si sobre el real
da el mismo orden, la desviacion entre lo declarado y lo ejecutado afecta a los
dos escenarios del contraste de H1 y no solo al sintetico.

USO
    py -3.12 medir_frac_real.py --panel panel_2003_2024.csv

AISLAMIENTO: lee el panel para entrenar un paso y no lo guarda ni lo imprime.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from bloques_h1 import entrenar_hasta

T0 = 143
REFIT = 12
N_ORIGENES = 109
ETA_CONTRASTE = 5.0
FRAC_PRERREGISTRO = 0.20


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", required=True)
    ap.add_argument("--columnas", nargs="*", default=None)
    ap.add_argument("--semillas", nargs="*", type=int, default=[0, 1, 2])
    a = ap.parse_args()

    from h1_real import cargar_panel
    panel = cargar_panel(a.panel, a.columnas)

    origenes = list(range(T0, T0 + N_ORIGENES, REFIT))

    print("=" * 74)
    print("FRACCION ESTRUCTURAL EFECTIVA DEL CONTRASTE DE H1  -  PANEL REAL")
    print("=" * 74)
    print(f"  {datetime.now().isoformat(timespec='seconds')}")
    print(f"  panel: {panel.shape[1]} meses   reajustes en {len(origenes)} origenes")
    print(f"  eta del contraste: {ETA_CONTRASTE}")
    print(f"  fraccion declarada en el prerregistro A.1: {FRAC_PRERREGISTRO}\n")

    print(f"  {'origen':>8}{'semilla':>9}{'L_simple':>12}{'R_G':>10}"
          f"{'frac (eta=5)':>14}{'eta para 0,20':>15}")
    print("  " + "-" * 68)

    filas = []
    for o in origenes:
        for s in a.semillas:
            _, _, _, reg = entrenar_hasta(panel, o, "B", s, eta=ETA_CONTRASTE,
                                          steps=1)
            p0 = reg["pasos"][0]
            L, R = abs(p0["L_simple"]), p0["R_G"]
            frac = ETA_CONTRASTE * R / (L + ETA_CONTRASTE * R) if R > 0 else 0.0
            eta20 = (FRAC_PRERREGISTRO * L / (R * (1 - FRAC_PRERREGISTRO))
                     if R > 0 else float("nan"))
            print(f"  {o:>8}{s:>9}{p0['L_simple']:>12.4f}{R:>10.5f}"
                  f"{frac:>14.4f}{eta20:>15.2f}")
            filas.append({"origen": o, "semilla": s, "L_simple": p0["L_simple"],
                          "R_G": R, "frac_eta5": frac, "eta_para_020": eta20})

    fr = np.array([f["frac_eta5"] for f in filas])
    et = np.array([f["eta_para_020"] for f in filas])

    print("\n" + "=" * 74)
    print("LECTURA")
    print("=" * 74)
    print(f"  fraccion con eta = 5 : mediana {np.median(fr):.4f}"
          f"   rango [{fr.min():.4f}, {fr.max():.4f}]")
    print(f"  eta para 0,20        : mediana {np.median(et):.1f}"
          f"   rango [{et.min():.1f}, {et.max():.1f}]")
    print(f"  razon declarado/ejecutado: {FRAC_PRERREGISTRO/np.median(fr):.1f}x\n")

    if np.median(fr) < 0.5 * FRAC_PRERREGISTRO:
        print("  La fraccion efectiva queda MUY por debajo de la declarada. La")
        print("  desviacion entre el prerregistro y la implementacion afecta")
        print("  tambien al escenario del panel real, no solo al sintetico, y")
        print("  debe declararse para los dos.")
    elif np.median(fr) > 1.5 * FRAC_PRERREGISTRO:
        print("  La fraccion efectiva SUPERA a la declarada. Tambien es una")
        print("  desviacion y tambien debe declararse.")
    else:
        print("  La fraccion efectiva es del orden de la declarada sobre el panel")
        print("  real. La desviacion quedaria acotada al escenario sintetico, lo")
        print("  que reduce su alcance pero no la elimina: hay que verificar por")
        print("  que difieren los dos escenarios.")

    p = Path(f"frac_real_{datetime.now().date()}.json")
    p.write_text(json.dumps(
        {"T0": T0, "eta_contraste": ETA_CONTRASTE,
         "frac_prerregistro": FRAC_PRERREGISTRO,
         "frac_mediana": float(np.median(fr)),
         "eta_para_020_mediana": float(np.median(et)),
         "filas": filas}, indent=2, ensure_ascii=False, default=float))
    print(f"\n  guardado en {p}")


if __name__ == "__main__":
    main()
