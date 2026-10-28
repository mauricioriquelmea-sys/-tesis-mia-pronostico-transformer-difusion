"""
h1_real.py  -  contraste de H1 sobre el panel macro-financiero real
====================================================================

Mismo codigo que h1_sintetico.py; cambia unicamente el array de entrada.
Las reglas son las de PREREGISTRO_H1.md y NO se tocan.

AISLAMIENTO: este script corre en TU maquina. El panel nunca sale de ella.
Lo que se comparte para interpretar son los agregados del JSON, no los datos.

----------------------------------------------------------------------------
INDEXACION  -  leelo antes de correr
----------------------------------------------------------------------------
El protocolo dice "T0 = 144, 109 origenes, h = 1..12" sobre 264 meses.
Con indices base 0 eso obliga a una convencion y solo una cierra:

    meses de entrenamiento inicial : indices 0 .. 143      (144 meses)
    primer origen                  : indice 143
    ultimo origen                  : indice 251
    numero de origenes             : 251 - 143 + 1 = 109   OK
    ultimo objetivo                : 251 + 12 = 263        = ultimo mes  OK

Si se tomara el primer origen en el indice 144, el ultimo seria 252 y su
objetivo llegaria al indice 264, que no existe en un panel de 264 meses.
Por eso ORIGEN_INICIAL = 143 y no 144. El script lo verifica y aborta si
la aritmetica no cierra.
"""

from __future__ import annotations
import argparse
import json
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(1)

from bloques_h1 import evaluar_ventana_expansiva, H_FUT
from h1_harness import (test_h1, crps_ensemble, assert_shapes,
                        numerical_guards, H1_MIN_REDUCTION)

# --------------------------------------------------------------------------
ORIGEN_INICIAL = 143        # indice base 0; equivale a 144 meses de entrenamiento
N_ORIGENES = 109
M_SAMPLES = 200
REFIT_CADA = 12
STEPS = 400
SEMILLAS = (0, 1, 2)
ETA = 5.0
SERIES = ("cobre", "USD/CLP", "TPM")


def cargar_panel(ruta: str, columnas: list[str] | None = None) -> np.ndarray:
    """
    Acepta .npy o .csv y devuelve [3, T] en NIVELES.

    El .csv que produce MIA_Construccion_Panel (`panel_2003_2024.csv`) trae
    encabezado y una columna de fecha como indice, asi que se lee con pandas
    y no con np.loadtxt. `columnas` fija el ORDEN de las series; si se omite
    se usa el orden del archivo y se imprime cual es, para que quede claro
    a que corresponde cada fila.
    """
    p = Path(ruta)
    if not p.exists():
        raise FileNotFoundError(f"no existe: {p}")

    if p.suffix == ".npy":
        X = np.asarray(np.load(p), dtype=float)
    else:
        import pandas as pd
        df = pd.read_csv(p, index_col=0)
        df = df.select_dtypes(include=[np.number])     # descarta fecha/texto
        if columnas is not None:
            faltan = [c for c in columnas if c not in df.columns]
            if faltan:
                raise ValueError(f"columnas no encontradas: {faltan}; "
                                 f"el archivo tiene {list(df.columns)}")
            df = df[columnas]
        print(f"  columnas en orden: {list(df.columns)}")
        print(f"  filas: {len(df)}   faltantes: {int(df.isna().sum().sum())}")
        if df.isna().any().any():
            raise ValueError("el panel tiene valores faltantes")
        X = df.to_numpy(dtype=float).T                 # [series, meses]
    if X.ndim != 2:
        raise ValueError(f"se esperaba un arreglo 2D; recibido {X.shape}")
    if X.shape[0] != 3:
        if X.shape[1] == 3:
            X = X.T
        else:
            raise ValueError(f"ninguna dimension es 3 (las series); forma {X.shape}")
    if not np.isfinite(X).all():
        n = int((~np.isfinite(X)).sum())
        raise ValueError(f"el panel tiene {n} valores no finitos")
    return X


def verificar_aritmetica(T: int) -> None:
    ultimo_origen = ORIGEN_INICIAL + N_ORIGENES - 1
    ultimo_objetivo = ultimo_origen + H_FUT
    print(f"  meses del panel            : {T}  (indices 0..{T-1})")
    print(f"  entrenamiento inicial      : 0..{ORIGEN_INICIAL}  "
          f"({ORIGEN_INICIAL+1} meses)")
    print(f"  origenes                   : {ORIGEN_INICIAL}..{ultimo_origen}  "
          f"({N_ORIGENES})")
    print(f"  ultimo objetivo            : indice {ultimo_objetivo}")
    if ultimo_objetivo > T - 1:
        raise ValueError(
            f"la aritmetica no cierra: el ultimo objetivo cae en el indice "
            f"{ultimo_objetivo} y el panel llega hasta {T-1}. "
            f"Con {T} meses el maximo de origenes es "
            f"{T - 1 - H_FUT - ORIGEN_INICIAL + 1}.")
    sobran = T - 1 - ultimo_objetivo
    print(f"  meses sin usar al final    : {sobran}"
          + ("  (el panel se consume exacto)" if sobran == 0 else ""))


def crps_por_serie(ens, y, scale):
    """Descriptivo (preregistro §5): NO redefine el exito de H1."""
    c = crps_ensemble(ens, y) / scale[:, None, :]      # (O,H,N)
    return c.mean(axis=(0, 1))                          # (N,)


def main(ruta_panel: str, rapido: bool, columnas=None):
    semillas = (0,) if rapido else SEMILLAS
    steps = 150 if rapido else STEPS
    n_orig = 30 if rapido else N_ORIGENES
    m = 60 if rapido else M_SAMPLES

    ts = datetime.now().isoformat(timespec="seconds")
    env = {"python": sys.version.split()[0], "torch": torch.__version__,
           "numpy": np.__version__, "plataforma": platform.platform(),
           "hilos": torch.get_num_threads()}

    print("=" * 78)
    print(f"CONTRASTE DE H1  -  PANEL REAL{'  [RAPIDO]' if rapido else ''}")
    print("=" * 78)
    print(f"  {ts}")
    print(f"  torch {env['torch']} | numpy {env['numpy']} | "
          f"hilos {env['hilos']}\n")

    panel = cargar_panel(ruta_panel, columnas)
    print(f"  panel cargado: {panel.shape}\n")
    if not rapido:
        verificar_aritmetica(panel.shape[1])
    print(f"\n  M={m}  reajuste cada {REFIT_CADA}  {steps} pasos  eta={ETA}")
    print(f"  umbral H1: reduccion >= {H1_MIN_REDUCTION*100:.0f} % y p < 0.05\n")

    salida = {"timestamp": ts, "entorno": env, "panel": "real",
              "panel_forma": list(panel.shape),
              "config": {"origen_inicial": ORIGEN_INICIAL, "n_origenes": n_orig,
                         "M": m, "refit_cada": REFIT_CADA, "steps": steps,
                         "eta": ETA, "semillas": list(semillas)},
              "semillas": {}}

    t0 = time.time()
    for s in semillas:
        print("-" * 78)
        print(f"SEMILLA {s}")
        print("-" * 78)

        ta = time.time()
        ens_a, y, sc, reg_a = evaluar_ventana_expansiva(
            panel, "A", s, 0.0, T0=ORIGEN_INICIAL, n_origenes=n_orig,
            M=m, refit_cada=REFIT_CADA, steps=steps)
        print(f"  Version A  ({time.time()-ta:.0f}s)")

        tb = time.time()
        ens_b, y_b, sc_b, reg_b = evaluar_ventana_expansiva(
            panel, "B", s, ETA, T0=ORIGEN_INICIAL, n_origenes=n_orig,
            M=m, refit_cada=REFIT_CADA, steps=steps)
        print(f"  Version B  ({time.time()-tb:.0f}s)")

        assert_shapes(ens_a, y, sc)
        numerical_guards(ens_a, ens_b, y, sc)
        if not np.allclose(y, y_b):
            raise RuntimeError("A y B no comparten realizados")

        m_proh = float(np.mean([r["masa_prohibida"] for r in reg_b]))
        ent_a = float(np.mean([r["entropia_normalizada"] for r in reg_a]))
        ent_b = float(np.mean([r["entropia_normalizada"] for r in reg_b]))
        print(f"  masa prohibida (B): {m_proh:.4f}   "
              f"entropia A/B: {ent_a:.4f} / {ent_b:.4f}")

        r_std = test_h1(ens_a, y, sc, ens_b, y_b, sc_b, standardize=True)
        r_raw = test_h1(ens_a, y, sc, ens_b, y_b, sc_b, standardize=False,
                        verbose=False)

        ca, cb = crps_por_serie(ens_a, y, sc), crps_por_serie(ens_b, y_b, sc_b)
        print(f"\n  descriptivo por serie (NO redefine H1):")
        print(f"    {'serie':<10}{'CRPS A':>10}{'CRPS B':>10}{'cambio':>10}")
        for i, nm in enumerate(SERIES):
            d = (ca[i] - cb[i]) / ca[i]
            print(f"    {nm:<10}{ca[i]:>10.4f}{cb[i]:>10.4f}{d*100:>9.2f}%")

        salida["semillas"][str(s)] = {
            "principal_estandarizado": {
                "crps_a": r_std["crps_a"], "crps_b": r_std["crps_b"],
                "reduccion": r_std["reduction"], "dm_hln": r_std["dm"]["dm_hln"],
                "p": r_std["dm"]["p_one_sided"], "pasa": bool(r_std["passes"])},
            "robustez_suma_cruda": {
                "reduccion": r_raw["reduction"], "p": r_raw["dm"]["p_one_sided"],
                "pasa": bool(r_raw["passes"])},
            "atencion": {"masa_prohibida_B": m_proh,
                         "entropia_A": ent_a, "entropia_B": ent_b},
            "descriptivo_por_serie": {
                nm: {"crps_a": float(ca[i]), "crps_b": float(cb[i]),
                     "cambio_relativo": float((ca[i] - cb[i]) / ca[i])}
                for i, nm in enumerate(SERIES)}}

    # ----------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("RESUMEN  -  PANEL REAL  -  resultado principal = estandarizado")
    print("=" * 78)
    print(f"  {'semilla':<9}{'reduccion':>12}{'p (DM)':>10}{'H1':>6}"
          f"{'masa proh':>12}{'entropia B':>12}")
    print("  " + "-" * 61)
    reds = []
    for s, v in salida["semillas"].items():
        p_, a_ = v["principal_estandarizado"], v["atencion"]
        reds.append(p_["reduccion"])
        print(f"  {s:<9}{p_['reduccion']*100:>11.2f}%{p_['p']:>10.4f}"
              f"{'SI' if p_['pasa'] else 'NO':>6}"
              f"{a_['masa_prohibida_B']:>12.4f}{a_['entropia_B']:>12.4f}")
    print("  " + "-" * 61)

    med, dev = float(np.mean(reds)), float(np.std(reds))
    n_pasa = sum(v["principal_estandarizado"]["pasa"]
                 for v in salida["semillas"].values())
    print(f"  media {med*100:+.2f} %   desviacion {dev*100:.2f} pp   "
          f"rango [{min(reds)*100:+.2f}, {max(reds)*100:+.2f}]")
    print(f"  semillas que sostienen H1: {n_pasa}/{len(semillas)}")

    salida["agregado"] = {"reduccion_media": med, "reduccion_desv": dev,
                          "reduccion_min": float(min(reds)),
                          "reduccion_max": float(max(reds)),
                          "semillas_que_pasan": n_pasa,
                          "n_semillas": len(semillas)}

    print(f"\n  tiempo total: {(time.time()-t0)/60:.1f} min")
    p = Path(f"h1_real_{'rapido' if rapido else 'full'}_{ts[:10]}.json")
    p.write_text(json.dumps(salida, indent=2, ensure_ascii=False))
    print(f"  guardado en {p}")

    if not rapido:
        print("\n  El JSON no contiene datos del panel: solo metricas agregadas.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("panel", help="ruta al panel [3,T] en .npy o .csv")
    ap.add_argument("--rapido", action="store_true",
                    help="30 origenes, 1 semilla: verifica antes de la corrida larga")
    ap.add_argument("--columnas", nargs=3, default=None, metavar=("CU", "FX", "TPM"),
                    help="nombres de las 3 columnas, EN ORDEN cobre USD/CLP TPM")
    a = ap.parse_args()
    main(a.panel, a.rapido, a.columnas)
