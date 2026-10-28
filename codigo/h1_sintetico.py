"""
h1_sintetico.py  -  contraste de H1 sobre el panel SINTETICO completo
======================================================================

Contrasta H1 en el MEJOR CASO POSIBLE: un panel donde la estructura causal
Cu -> FX -> TPM esta plantada por construccion, de modo que Pi(G) no es una
postulacion de dominio sino la verdad del generador.

Si H1 no se sostiene aqui, el mecanismo no funciona ni con el prior correcto.
Si se sostiene, el paso al panel real queda reducido a una pregunta sobre si
Pi(G) describe bien el sistema chileno.

Reglas: las de PREREGISTRO_H1.md. No se tocan.

Uso:
    python h1_sintetico.py              # configuracion completa
    python h1_sintetico.py --rapido     # version corta para verificar
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

from bloques_h1 import evaluar_ventana_expansiva, _panel_sintetico
from h1_harness import test_h1, assert_shapes, numerical_guards, H1_MIN_REDUCTION

# --------------------------------------------------------------------------
# Configuracion  (preregistro §5)
# --------------------------------------------------------------------------

T_PANEL = 300          # meses del panel sintetico
T0 = 144               # inicio de la ventana expansiva
N_ORIGENES = 109
M_SAMPLES = 200
REFIT_CADA = 12
STEPS = 400
SEMILLAS = (0, 1, 2)
ETA = 5.0              # escala de R_G_MSE del cuaderno; aporta ~20 % al inicio

RAPIDO = dict(T_PANEL=250, T0=120, N_ORIGENES=40, M_SAMPLES=80,
              REFIT_CADA=20, STEPS=200, SEMILLAS=(0,))


def entorno() -> dict:
    """Se registra en el JSON: el flujo del PRNG cambia entre versiones de
    torch, asi que resultados de entornos distintos NO son comparables."""
    return {"python": sys.version.split()[0], "torch": torch.__version__,
            "numpy": np.__version__, "plataforma": platform.platform(),
            "cuda": torch.cuda.is_available()}


def main(rapido: bool):
    cfg = dict(T_PANEL=T_PANEL, T0=T0, N_ORIGENES=N_ORIGENES,
               M_SAMPLES=M_SAMPLES, REFIT_CADA=REFIT_CADA, STEPS=STEPS,
               SEMILLAS=SEMILLAS, ETA=ETA)
    if rapido:
        cfg.update(RAPIDO)

    ts = datetime.now().isoformat(timespec="seconds")
    env = entorno()

    print("=" * 78)
    print(f"CONTRASTE DE H1  -  PANEL SINTETICO{'  [RAPIDO]' if rapido else ''}")
    print("=" * 78)
    print(f"  {ts}")
    print(f"  torch {env['torch']} | numpy {env['numpy']} | python {env['python']}")
    print(f"  ATENCION: resultados de otra version de torch NO son comparables\n")
    print(f"  panel: {cfg['T_PANEL']} meses, causalidad Cu->FX->TPM PLANTADA")
    print(f"  T0={cfg['T0']}  origenes={cfg['N_ORIGENES']}  M={cfg['M_SAMPLES']}")
    print(f"  reajuste cada {cfg['REFIT_CADA']} origenes, {cfg['STEPS']} pasos")
    print(f"  eta={cfg['ETA']}  semillas={cfg['SEMILLAS']}")
    print(f"  umbral H1: reduccion >= {H1_MIN_REDUCTION*100:.0f} % y p < 0.05\n")

    panel = _panel_sintetico(T=cfg["T_PANEL"], seed=12345)   # panel FIJO
    salida = {"timestamp": ts, "entorno": env, "config": {
        k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg.items()},
        "panel": "sintetico_causalidad_plantada", "semillas": {}}

    t0 = time.time()
    for s in cfg["SEMILLAS"]:
        print("-" * 78)
        print(f"SEMILLA {s}")
        print("-" * 78)

        ta = time.time()
        ens_a, y, sc, reg_a = evaluar_ventana_expansiva(
            panel, "A", s, 0.0, T0=cfg["T0"], n_origenes=cfg["N_ORIGENES"],
            M=cfg["M_SAMPLES"], refit_cada=cfg["REFIT_CADA"], steps=cfg["STEPS"])
        print(f"  Version A entrenada y evaluada  ({time.time()-ta:.0f}s)")

        tb = time.time()
        ens_b, y_b, sc_b, reg_b = evaluar_ventana_expansiva(
            panel, "B", s, cfg["ETA"], T0=cfg["T0"], n_origenes=cfg["N_ORIGENES"],
            M=cfg["M_SAMPLES"], refit_cada=cfg["REFIT_CADA"], steps=cfg["STEPS"])
        print(f"  Version B entrenada y evaluada  ({time.time()-tb:.0f}s)")

        assert_shapes(ens_a, y, sc)
        numerical_guards(ens_a, ens_b, y, sc)
        if not np.allclose(y, y_b):
            raise RuntimeError("A y B no comparten realizados")

        m_proh = float(np.mean([r["masa_prohibida"] for r in reg_b]))
        ent_a = float(np.mean([r["entropia_normalizada"] for r in reg_a]))
        ent_b = float(np.mean([r["entropia_normalizada"] for r in reg_b]))
        frac = float(np.mean([r["pasos"][0]["frac_estructural"] for r in reg_b]))
        print(f"  masa prohibida (B): {m_proh:.4f}   "
              f"entropia A/B: {ent_a:.4f} / {ent_b:.4f}   "
              f"frac.estructural inicial: {100*frac:.1f}%")

        r_std = test_h1(ens_a, y, sc, ens_b, y_b, sc_b, standardize=True)
        r_raw = test_h1(ens_a, y, sc, ens_b, y_b, sc_b, standardize=False,
                        verbose=False)

        salida["semillas"][str(s)] = {
            "principal_estandarizado": {
                "crps_a": r_std["crps_a"], "crps_b": r_std["crps_b"],
                "reduccion": r_std["reduction"], "dm_hln": r_std["dm"]["dm_hln"],
                "p": r_std["dm"]["p_one_sided"], "pasa": bool(r_std["passes"])},
            "robustez_suma_cruda": {
                "reduccion": r_raw["reduction"], "p": r_raw["dm"]["p_one_sided"],
                "pasa": bool(r_raw["passes"])},
            "atencion": {"masa_prohibida_B": m_proh, "entropia_A": ent_a,
                         "entropia_B": ent_b, "frac_estructural": frac}}

    # ----------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("RESUMEN  -  resultado PRINCIPAL = columna estandarizada")
    print("=" * 78)
    print(f"  {'semilla':<9}{'reduccion':>12}{'p (DM)':>10}{'H1':>6}"
          f"{'masa proh':>12}{'entropia B':>12}")
    print("  " + "-" * 61)
    reds, ps = [], []
    for s, v in salida["semillas"].items():
        p_ = v["principal_estandarizado"]; a_ = v["atencion"]
        reds.append(p_["reduccion"]); ps.append(p_["p"])
        print(f"  {s:<9}{p_['reduccion']*100:>11.2f}%{p_['p']:>10.4f}"
              f"{'SI' if p_['pasa'] else 'NO':>6}"
              f"{a_['masa_prohibida_B']:>12.4f}{a_['entropia_B']:>12.4f}")
    print("  " + "-" * 61)

    med, dev = float(np.mean(reds)), float(np.std(reds))
    n_pasa = sum(v["principal_estandarizado"]["pasa"]
                 for v in salida["semillas"].values())
    print(f"  media {med*100:+.2f} %   desviacion {dev*100:.2f} pp   "
          f"rango [{min(reds)*100:+.2f}, {max(reds)*100:+.2f}]")
    print(f"  semillas que sostienen H1: {n_pasa}/{len(cfg['SEMILLAS'])}")

    salida["agregado"] = {"reduccion_media": med, "reduccion_desv": dev,
                          "reduccion_min": float(min(reds)),
                          "reduccion_max": float(max(reds)),
                          "p_max": float(max(ps)),
                          "semillas_que_pasan": n_pasa,
                          "n_semillas": len(cfg["SEMILLAS"])}

    print("\n  LECTURA")
    print("  " + "-" * 61)
    if dev * 100 > abs(med) * 100 / 2:
        print("  La dispersion entre semillas es comparable al efecto medio:")
        print("  el resultado NO distingue senal de ruido de inicializacion.")
    if n_pasa == len(cfg["SEMILLAS"]):
        print("  H1 se sostiene en las tres semillas, con Pi(G) correcta por")
        print("  construccion. El paso a datos reales queda como pregunta sobre")
        print("  si Pi(G) describe el sistema chileno.")
    elif n_pasa == 0:
        print("  H1 NO se sostiene en ninguna semilla, y esto es con Pi(G)")
        print("  siendo la verdad del generador. Es el resultado mas informativo")
        print("  posible: acota lo que cabe esperar del panel real y se reporta")
        print("  como tal. NO se ajustan umbrales ni reglas de agregacion.")
    else:
        print(f"  H1 se sostiene en {n_pasa} de {len(cfg['SEMILLAS'])} semillas.")
        print("  Resultado mixto: se reportan las tres, sin seleccionar.")

    print(f"\n  tiempo total: {(time.time()-t0)/60:.1f} min")

    p = Path(f"h1_sintetico_{'rapido' if rapido else 'full'}_{ts[:10]}.json")
    p.write_text(json.dumps(salida, indent=2, ensure_ascii=False))
    print(f"  guardado en {p}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rapido", action="store_true",
                    help="configuracion corta para verificar antes de la corrida larga")
    main(ap.parse_args().rapido)
