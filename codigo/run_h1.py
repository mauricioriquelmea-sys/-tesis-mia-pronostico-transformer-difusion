"""
run_h1.py  -  plantilla de ejecucion del contraste de H1
=========================================================

Conecta los tres modulos ya auditados:

    rg_frobenius.RegularizadorEstructural   -> R_G sobre la atencion
    h1_harness.test_h1                      -> CRPS conjunto + Diebold-Mariano

Las partes que tocan el panel real estan marcadas  # >>> TUYO <<<  y son las
UNICAS que debes completar. El resto esta verificado sobre tensores dummy.

Ejecucion de prueba (sin datos reales, verifica el cableado):
    python run_h1.py --dry-run

Ejecucion real:
    python run_h1.py
"""

from __future__ import annotations
import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from rg_frobenius import RegularizadorEstructural, entropia_normalizada, causal_mask
from h1_harness import test_h1, assert_shapes, numerical_guards, report_magnitudes

# --------------------------------------------------------------------------
# CONFIGURACION  -  debe coincidir con PREREGISTRO_H1.md
# --------------------------------------------------------------------------

N, L = 3, 15
H_HEADS, N_LAYERS = 4, 3
T = N * L

T0 = 144                    # inicio de la ventana expansiva
H_MAX = 12                  # horizontes
N_ORIGINS = 109
M_SAMPLES = 200             # tamano del ensemble

PI_G = torch.tensor([[1.0, 0.0, 0.0],
                     [0.5, 0.5, 0.0],
                     [0.0, 0.5, 0.5]])

FRACCION_ETA = 0.20         # preregistro §4
SEMILLAS = (0, 1, 2)        # preregistro §5
LAYER_REDUCE = "mean"       # preregistro §3

SERIES = ("cobre", "USD/CLP", "TPM")


# --------------------------------------------------------------------------
# 1. ENTRENAMIENTO
# --------------------------------------------------------------------------

def entrenar(version: str, semilla: int, dry_run: bool = False):
    """
    version: 'A' (sin R_G) o 'B' (con R_G).

    Devuelve el modelo entrenado y un registro de diagnosticos.
    """
    torch.manual_seed(semilla)
    reg = None
    if version == "B":
        reg = RegularizadorEstructural(PI_G, N, L, layer_reduce=LAYER_REDUCE)

    if dry_run:
        return _entrenar_dummy(version, semilla, reg)

    # >>> TUYO <<< ---------------------------------------------------------
    # modelo = TuEncoder(..., cabeza=R0)
    # opt    = torch.optim.AdamW(modelo.parameters(), lr=...)
    # loader = tu_dataloader_hasta(T0 + paso)
    #
    # El bucle debe seguir EXACTAMENTE esta forma:
    #
    #   eta_fijado = False
    #   for epoca in range(n_epocas):
    #       for lote in loader:
    #           salida, attn = modelo(lote, devolver_atencion=True)
    #           #    attn: lista de N_LAYERS tensores (B, H, T, T) POST-SOFTMAX
    #           #    y CON gradiente. No pasar un buffer de logging.
    #           loss_simple = modelo.perdida(salida, lote)
    #
    #           if version == "B":
    #               if not eta_fijado:                 # una sola vez
    #                   reg.calibrar_eta(attn, loss_simple.item(), FRACCION_ETA)
    #                   eta_fijado = True
    #               rg, diag = reg(attn)
    #               loss = loss_simple + reg.eta * rg
    #           else:
    #               loss = loss_simple
    #
    #           opt.zero_grad(); loss.backward(); opt.step()
    #
    # return modelo, registro
    # ----------------------------------------------------------------------
    raise NotImplementedError(
        "Completar el bloque marcado >>> TUYO <<< en entrenar(). "
        "Usar --dry-run para verificar el cableado sin datos reales.")


def _entrenar_dummy(version, semilla, reg):
    """Simula el bucle para verificar el cableado de R_G y la calibracion."""
    import torch.nn.functional as F
    g = torch.Generator().manual_seed(semilla)
    mask = causal_mask(T, L)
    registro = {"version": version, "semilla": semilla, "pasos": []}

    eta_fijado = False
    for paso in range(3):
        logits = [(torch.randn(8, H_HEADS, T, T, generator=g) * 0.01)
                  .requires_grad_(True) for _ in range(N_LAYERS)]
        attn = [F.softmax(x + mask, dim=-1) for x in logits]
        loss_simple = torch.tensor(1.0, requires_grad=True)

        if version == "B":
            if not eta_fijado:
                reg.calibrar_eta(attn, loss_simple.item(), FRACCION_ETA)
                eta_fijado = True
            rg, diag = reg(attn)
            loss = loss_simple + reg.eta * rg
            registro["pasos"].append({
                "paso": paso, "L_simple": loss_simple.item(),
                "R_G": rg.item(), "eta": reg.eta,
                "frac_estructural": reg.eta * rg.item() / loss.item(),
                "masa_prohibida": diag["masa_prohibida"].item()})
        else:
            loss = loss_simple
            registro["pasos"].append({"paso": paso, "L_simple": loss_simple.item()})

        loss.backward()

    registro["entropia_normalizada"] = float(
        entropia_normalizada(attn[0].detach(), mask).mean())
    return None, registro


# --------------------------------------------------------------------------
# 2. EVALUACION
# --------------------------------------------------------------------------

def _realizados_dummy(semilla: int):
    """
    Realizados y escalas compartidos por A y B. test_h1 exige que ambas
    versiones se evaluen sobre los MISMOS y: si cada una se genera con su
    propia realizacion, la 'reduccion' mide ruido entre realizaciones y no
    diferencia entre modelos.
    """
    r = np.random.default_rng(semilla)
    sc = np.tile(np.array([29.0, 37.0, 1.7]), (N_ORIGINS, 1))
    y = r.normal(0, 1, (N_ORIGINS, H_MAX, N)) * sc[:, None, :]
    return y, sc


def evaluar(modelo, dry_run: bool = False, semilla: int = 0,
            realizados=None, nitidez=1.0):
    """
    Devuelve (ens, y, scale) con las formas que exige h1_harness:

        ens    (O, H, N, M)
        y      (O, H, N)
        scale  (O, N)   desviacion de los CAMBIOS, estimada SOLO hasta el
                        origen o  (preregistro §2; decision M9)

    En dry-run, `realizados` fuerza que A y B compartan y, y `nitidez` emula
    una version B mejor calibrada (valor < 1).
    """
    if dry_run:
        y, sc = realizados if realizados is not None else _realizados_dummy(semilla)
        r = np.random.default_rng(semilla + 7777)
        ens = y[..., None] + r.normal(0, 1, (N_ORIGINS, H_MAX, N, M_SAMPLES)) \
            * sc[:, None, :, None] * nitidez
        return ens, y, sc

    # >>> TUYO <<< ---------------------------------------------------------
    # ens   = np.empty((N_ORIGINS, H_MAX, N, M_SAMPLES))
    # y     = np.empty((N_ORIGINS, H_MAX, N))
    # scale = np.empty((N_ORIGINS, N))
    #
    # for o, origen in enumerate(range(T0, T0 + N_ORIGINS)):
    #     # escala: SOLO informacion hasta `origen`, nunca despues
    #     scale[o] = cambios[:origen].std(axis=0)
    #     ens[o]   = modelo.muestrear(origen, H_MAX, M_SAMPLES)
    #     y[o]     = panel[origen + 1 : origen + 1 + H_MAX]
    # return ens, y, scale
    # ----------------------------------------------------------------------
    raise NotImplementedError(
        "Completar el bloque marcado >>> TUYO <<< en evaluar().")


# --------------------------------------------------------------------------
# 3. ORQUESTACION
# --------------------------------------------------------------------------

def main(dry_run: bool):
    ts = datetime.now().isoformat(timespec="seconds")
    salida = {"timestamp": ts, "dry_run": dry_run,
              "preregistro": {"fraccion_eta": FRACCION_ETA,
                              "layer_reduce": LAYER_REDUCE,
                              "semillas": list(SEMILLAS),
                              "estandarizado": True,
                              "umbral_reduccion": 0.05, "alpha": 0.05},
              "semillas": {}}

    print("=" * 74)
    print(f"CONTRASTE DE H1   {ts}" + ("   [DRY RUN]" if dry_run else ""))
    print("=" * 74)
    print(f"  reglas fijadas en PREREGISTRO_H1.md")
    print(f"  agregacion: CRPS conjunto ESTANDARIZADO")
    print(f"  eta: calibrado a fraccion {FRACCION_ETA}, fijo")
    print(f"  semillas: {SEMILLAS}\n")

    for s in SEMILLAS:
        print("-" * 74)
        print(f"SEMILLA {s}")
        print("-" * 74)

        mod_a, reg_a = entrenar("A", s, dry_run)
        mod_b, reg_b = entrenar("B", s, dry_run)

        if reg_b["pasos"] and "eta" in reg_b["pasos"][0]:
            p = reg_b["pasos"][0]
            print(f"  calibracion: R_G={p['R_G']:.5f}  eta={p['eta']:.4f}  "
                  f"aporte estructural={100*p['frac_estructural']:.1f}%")
            print(f"  masa en posiciones prohibidas: {p['masa_prohibida']:.4f}")
            print(f"  entropia normalizada: {reg_b['entropia_normalizada']:.4f}")

        if dry_run:
            # A y B comparten realizados; B es 10 % mas nitida, para que el
            # dry-run muestre como se ve un cableado correcto (~+10 %).
            rz = _realizados_dummy(s)
            ens_a, y, sc = evaluar(mod_a, True, s, realizados=rz, nitidez=1.00)
            ens_b, y_b, sc_b = evaluar(mod_b, True, s + 500, realizados=rz,
                                       nitidez=0.90)
        else:
            ens_a, y, sc = evaluar(mod_a, False)
            ens_b, y_b, sc_b = evaluar(mod_b, False)

        assert_shapes(ens_a, y, sc)
        numerical_guards(ens_a, ens_b, y, sc)
        report_magnitudes(ens_a, y, sc, f"Version A (semilla {s})")

        r_std = test_h1(ens_a, y, sc, ens_b, y_b, sc_b, standardize=True)
        r_raw = test_h1(ens_a, y, sc, ens_b, y_b, sc_b, standardize=False,
                        verbose=False)

        salida["semillas"][str(s)] = {
            "principal_estandarizado": {
                "crps_a": r_std["crps_a"], "crps_b": r_std["crps_b"],
                "reduccion": r_std["reduction"],
                "dm_hln": r_std["dm"]["dm_hln"],
                "p": r_std["dm"]["p_one_sided"], "pasa": bool(r_std["passes"])},
            "robustez_suma_cruda": {
                "reduccion": r_raw["reduction"],
                "p": r_raw["dm"]["p_one_sided"], "pasa": bool(r_raw["passes"])},
            "calibracion": reg_b["pasos"][0] if reg_b["pasos"] else None}

    print("\n" + "=" * 74)
    print("RESUMEN  -  resultado PRINCIPAL = columna estandarizada")
    print("=" * 74)
    print(f"  {'semilla':<10}{'reduccion':>12}{'p (DM)':>10}{'H1':>8}"
          f"{'  |  robustez cruda':>22}")
    print("  " + "-" * 62)
    reds = []
    for s, v in salida["semillas"].items():
        p_ = v["principal_estandarizado"]
        r_ = v["robustez_suma_cruda"]
        reds.append(p_["reduccion"])
        print(f"  {s:<10}{p_['reduccion']*100:>11.2f}%{p_['p']:>10.4f}"
              f"{'SI' if p_['pasa'] else 'NO':>8}"
              f"{r_['reduccion']*100:>14.2f}%{'SI' if r_['pasa'] else 'NO':>8}")
    print("  " + "-" * 62)
    print(f"  media entre semillas: {np.mean(reds)*100:+.2f} %   "
          f"desv: {np.std(reds)*100:.2f} pp")
    n_pasa = sum(v["principal_estandarizado"]["pasa"]
                 for v in salida["semillas"].values())
    print(f"  semillas que sostienen H1: {n_pasa}/{len(SEMILLAS)}")
    print("\n  Las tres semillas se reportan. No se selecciona la mejor.")

    p = Path(f"h1_resultados_{'dry' if dry_run else 'real'}_{ts[:10]}.json")
    p.write_text(json.dumps(salida, indent=2, ensure_ascii=False))
    print(f"\n  guardado en {p}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="verifica el cableado con tensores dummy")
    main(ap.parse_args().dry_run)
