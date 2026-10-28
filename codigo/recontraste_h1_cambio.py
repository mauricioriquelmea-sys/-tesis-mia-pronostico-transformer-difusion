"""
recontraste_h1_cambio.py  -  el re-contraste prerregistrado en el anexo A.4
===========================================================================

Contrasta H1 ---que R_G reduce el CRPS conjunto respecto de la Version A---
bajo la especificacion del objetivo en CAMBIOS, que el analisis del objetivo
de entrenamiento establece como superior sobre el panel real.

Es un experimento NUEVO. No sustituye ni corrige el rechazo de H1 reportado
bajo la especificacion en niveles.

TRES COSAS QUE ESTE GUION HACE Y EL CONTRASTE ORIGINAL NO HIZO
--------------------------------------------------------------
1. CALIBRA eta. El prerregistro A.1 no fija eta: fija la fraccion estructural

       eta * R_G / (|L_simple| + eta * R_G) = 0,20

   calibrada sobre el primer lote. El contraste original uso eta = 5 fijo, lo
   que resulto ser una fraccion del 1,1 %: dieciocho veces menos que lo
   declarado. Aqui se calibra, que es lo que A.1 siempre exigio.

2. DERIVA EL UMBRAL DE LA POTENCIA, ANTES DE MIRAR EL RESULTADO. A.4 compromete
   un 6 % y ordena elevarlo si la mejora minima detectable del diseño resulta
   superior. La MDE se calcula sobre las series de perdida por origen de ESTE
   experimento:

       MDE = t_crit * sqrt(var_HAC / n) / corr_HLN      (en unidades de perdida)

   El guion la calcula, fija el umbral y lo ESCRIBE EN DISCO en una FASE 1
   separada, antes de evaluar ningun veredicto. La FASE 2 solo entonces compara.
   El orden es estructural, no una convencion: si la FASE 1 falla, no hay FASE 2.

3. AGREGA SOBRE LA MEDIA ENTRE SEMILLAS, segun la regla fijada en el anexo A.5
   antes de leer el barrido de sensibilidad. El conteo de semillas que superan
   el umbral individualmente se reporta como DESCRIPTIVO y no redefine el exito.

USO
    py -3.12 recontraste_h1_cambio.py --panel panel_2003_2024.csv
    py -3.12 recontraste_h1_cambio.py --panel panel_2003_2024.csv --rapido
    py -3.12 recontraste_h1_cambio.py --cabeza gaussian --panel ...

AISLAMIENTO: lee el panel para entrenar y evaluar. No lo guarda ni lo imprime.
Ningun valor del panel entra en el JSON de salida.
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
from scipy import stats

from bloques_h1 import (evaluar_ventana_expansiva, entrenar_hasta,
                        _panel_sintetico, H_FUT)
from h1_harness import joint_crps, diebold_mariano

# --------------------------------------------------------------------------
# Constantes del prerregistro.  A.4: "Que se mantiene sin cambios".
# --------------------------------------------------------------------------
PANEL_SEED = 12345
T0_SINT = 144
T0_REAL = 143
N_ORIGENES = 109
M = 200
REFIT = 12
STEPS = 400
SEMILLAS = (0, 1, 2)

OBJETIVO = "cambio"          # LA variacion respecto del contraste original
FRAC_PRERREGISTRO = 0.20     # A.1: la fraccion, no eta
UMBRAL_COMPROMETIDO = 0.06   # A.4: el 6 %, piso; se eleva si la MDE lo supera
ALPHA = 0.05
CABEZA_ANTICIPADA = "student-t"   # A.4 anticipa R0-T; el congelamiento manda


def entorno() -> dict:
    return {"python": sys.version.split()[0], "torch": torch.__version__,
            "numpy": np.__version__, "plataforma": platform.platform(),
            "hilos": torch.get_num_threads()}


def eta_para_fraccion(panel, t0, semilla, objetivo_frac, cabeza):
    """eta * R / (|L| + eta * R) = f   ->   eta = f*|L| / (R*(1-f)).

    Un solo paso de entrenamiento. Es la calibracion que A.1 exige y que el
    contraste original no aplico.
    """
    _, _, _, reg = entrenar_hasta(panel, t0, "B", semilla, eta=1.0, steps=1,
                                  dist_type=cabeza, objetivo=OBJETIVO)
    p0 = reg["pasos"][0]
    L, R = abs(p0["L_simple"]), p0["R_G"]
    if not np.isfinite(L) or not np.isfinite(R):
        raise RuntimeError(f"calibracion no finita: L={L}, R={R}")
    if R <= 0:
        raise RuntimeError("R_G nulo en el primer lote: no se puede calibrar")
    return float(objetivo_frac * L / (R * (1.0 - objetivo_frac))), L, R


def mde_relativa(loss_a: np.ndarray, loss_b: np.ndarray, h: int,
                 alpha: float = ALPHA) -> dict:
    """Mejora minima detectable, en fraccion de la perdida de la Version A.

    Replica exactamente la varianza HAC de h1_harness.diebold_mariano y
    despeja el diferencial medio mas pequeño que habria alcanzado
    significancia:

        dm_hln = (dbar / sqrt(var/n)) * corr  >=  t_crit
        dbar_min = t_crit * sqrt(var/n) / corr
    """
    d = np.asarray(loss_a, float) - np.asarray(loss_b, float)
    n = d.size
    if n < 8:
        raise ValueError(f"n={n} insuficiente para inferencia HAC")

    u = d - d.mean()
    L = max(0, min(h - 1, n - 1))
    gamma0 = (u @ u) / n
    var = gamma0
    for k in range(1, L + 1):                       # Newey-West, identico
        var += 2.0 * (1.0 - k / (L + 1.0)) * ((u[k:] @ u[:-k]) / n)
    if var <= 0:
        var = gamma0 if gamma0 > 0 else np.finfo(float).eps

    corr = np.sqrt(max((n + 1 - 2 * h + h * (h - 1) / n) / n, 1e-12))
    t_crit = stats.t.ppf(1.0 - alpha, df=n - 1)
    dbar_min = t_crit * np.sqrt(var / n) / corr

    base = float(np.mean(loss_a))
    return {"n": n, "nw_lag": L, "var_hac": float(var), "corr_hln": float(corr),
            "t_crit": float(t_crit), "dbar_min": float(dbar_min),
            "crps_a": base, "mde": float(dbar_min / base)}


def cargar(a):
    if a.panel:
        from h1_real import cargar_panel
        return cargar_panel(a.panel, a.columnas), T0_REAL, True
    t0 = T0_SINT
    n = (24 if a.rapido else a.origenes)
    return _panel_sintetico(T=t0 + n + H_FUT, seed=PANEL_SEED), t0, False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default=None)
    ap.add_argument("--columnas", nargs="*", default=None)
    ap.add_argument("--cabeza", default=CABEZA_ANTICIPADA,
                    choices=["student-t", "gaussian"],
                    help="la cabeza CONGELADA; A.4 anticipa student-t (R0-T)")
    ap.add_argument("--rapido", action="store_true")
    ap.add_argument("--semillas", nargs="*", type=int, default=None)
    ap.add_argument("--pasos", type=int, default=STEPS)
    ap.add_argument("--muestras", type=int, default=M)
    ap.add_argument("--origenes", type=int, default=N_ORIGENES)
    a = ap.parse_args()

    semillas = a.semillas or ([0] if a.rapido else list(SEMILLAS))
    n_orig = 24 if a.rapido else a.origenes
    pasos = 60 if a.rapido else a.pasos
    muestras = 60 if a.rapido else a.muestras
    reducida = (pasos != STEPS or muestras != M or n_orig != N_ORIGENES
                or len(semillas) != len(SEMILLAS))

    panel, t0, real = cargar(a)
    ts = datetime.now().isoformat(timespec="seconds")
    env = entorno()

    print("=" * 78)
    print("RE-CONTRASTE DE H1 BAJO EL OBJETIVO EN CAMBIOS  -  anexo A.4")
    print("=" * 78)
    print(f"  {ts}")
    print(f"  torch {env['torch']} | numpy {env['numpy']} | hilos {env['hilos']}")
    print(f"  panel: {'real' if real else 'sintetico'}, {panel.shape[1]} meses")
    print(f"  T0={t0}  origenes={n_orig}  M={muestras}  pasos={pasos}  "
          f"semillas={tuple(semillas)}")
    print(f"  objetivo: {OBJETIVO}   cabeza: {a.cabeza}")
    print(f"  eta: CALIBRADO a fraccion estructural {FRAC_PRERREGISTRO} (A.1)")
    if reducida:
        print("\n  AVISO: configuracion reducida. Sirve para verificar que el")
        print("  guion corre; sus cifras NO son el re-contraste.")
    print()

    salida = {"timestamp": ts, "entorno": env,
              "panel": "real" if real else "sintetico",
              "configuracion_reducida": reducida,
              "T0": t0, "n_origenes": n_orig, "M": muestras, "pasos": pasos,
              "semillas": list(semillas), "objetivo": OBJETIVO,
              "cabeza": a.cabeza, "frac_objetivo": FRAC_PRERREGISTRO,
              "umbral_comprometido": UMBRAL_COMPROMETIDO,
              "nota": ("re-contraste prerregistrado en el anexo A.4; experimento "
                       "NUEVO, no sustituye al contraste bajo objetivo en niveles"),
              "semilla_detalle": {}}

    # ====================================================================
    # Calibracion de eta  (A.1)
    # ====================================================================
    print("-" * 78)
    print("  CALIBRACION DE ETA  (sobre el primer lote de cada semilla)")
    print("-" * 78)
    print(f"  {'semilla':>9}{'L_simple':>12}{'R_G':>10}{'eta':>12}")
    etas = {}
    for s in semillas:
        e, L, R = eta_para_fraccion(panel, t0, s, FRAC_PRERREGISTRO, a.cabeza)
        etas[s] = e
        print(f"  {s:>9}{L:>12.4f}{R:>10.5f}{e:>12.2f}")
    salida["etas"] = {str(s): v for s, v in etas.items()}
    print()

    # ====================================================================
    # Corridas.  Se guardan las series de perdida; NO se evalua nada aun.
    # ====================================================================
    print("-" * 78)
    print("  ENTRENAMIENTO Y EVALUACION")
    print("-" * 78)
    series = {}
    for s in semillas:
        t = time.time()
        ens_a, y_a, sc_a, _ = evaluar_ventana_expansiva(
            panel, "A", s, 0.0, T0=t0, n_origenes=n_orig, M=muestras,
            refit_cada=REFIT, steps=pasos, dist_type=a.cabeza, objetivo=OBJETIVO)
        ens_b, y_b, sc_b, reg_b = evaluar_ventana_expansiva(
            panel, "B", s, etas[s], T0=t0, n_origenes=n_orig, M=muestras,
            refit_cada=REFIT, steps=pasos, dist_type=a.cabeza, objetivo=OBJETIVO)

        if not np.allclose(y_a, y_b):
            raise RuntimeError("A y B no comparten realizados")
        for nom, e in (("A", ens_a), ("B", ens_b)):
            if not np.isfinite(e).all():
                raise RuntimeError(f"ensemble no finito en Version {nom}, semilla {s}")

        la = joint_crps(ens_a, y_a, sc_a, standardize=True).mean(axis=1)
        lb = joint_crps(ens_b, y_b, sc_b, standardize=True).mean(axis=1)
        frac = float(np.mean([p["frac_estructural"] for r in reg_b
                              for p in r["pasos"] if p.get("paso") == 0]))
        series[s] = (la, lb, frac)
        print(f"  semilla {s}: CRPS A = {la.mean():.4f}   CRPS B = {lb.mean():.4f}"
              f"   frac.estr. = {100*frac:.1f}%   ({time.time()-t:.0f} s)")
    print()

    # ====================================================================
    # FASE 1.  Potencia.  El umbral queda fijado AQUI, y se escribe en disco
    #          antes de que exista ningun veredicto.
    # ====================================================================
    print("=" * 78)
    print("FASE 1  -  POTENCIA DEL DISEÑO  (antes de mirar ningun veredicto)")
    print("=" * 78)
    print(f"  {'semilla':>9}{'CRPS A':>10}{'MDE %':>9}{'dbar_min':>11}{'lag NW':>8}")
    mdes = {}
    for s in semillas:
        la, lb, _ = series[s]
        m = mde_relativa(la, lb, h=H_FUT)
        mdes[s] = m
        print(f"  {s:>9}{m['crps_a']:>10.4f}{100*m['mde']:>9.2f}"
              f"{m['dbar_min']:>11.4f}{m['nw_lag']:>8}")

    mde_med = float(np.median([m["mde"] for m in mdes.values()]))
    umbral = max(UMBRAL_COMPROMETIDO, mde_med)
    elevado = umbral > UMBRAL_COMPROMETIDO + 1e-12

    print(f"\n  MDE mediana: {100*mde_med:.2f} %")
    print(f"  umbral comprometido en A.4: {100*UMBRAL_COMPROMETIDO:.2f} %")
    if elevado:
        print(f"  -> La MDE SUPERA el umbral comprometido. La clausula de A.4 se")
        print(f"     activa: el umbral se ELEVA a {100*umbral:.2f} %.")
    else:
        print(f"  -> La MDE no supera el umbral comprometido. El umbral queda en")
        print(f"     {100*umbral:.2f} %, sin modificar.")
    print(f"\n  UMBRAL FIJADO: {100*umbral:.2f} %   significancia: p < {ALPHA}")
    print("  Regla de agregacion (A.5): la MEDIA entre semillas decide; el")
    print("  conteo individual es descriptivo.")

    salida["fase1_potencia"] = {
        "mde_por_semilla": {str(s): m for s, m in mdes.items()},
        "mde_mediana": mde_med,
        "umbral_comprometido": UMBRAL_COMPROMETIDO,
        "umbral_fijado": umbral,
        "umbral_elevado": bool(elevado),
        "regla_agregacion": "media entre semillas; conteo descriptivo (anexo A.5)"}

    p1 = Path(f"recontraste_h1_cambio_FASE1_{datetime.now().date()}.json")
    p1.write_text(json.dumps(salida, indent=2, ensure_ascii=False, default=float))
    print(f"\n  FASE 1 escrita en {p1}")
    print("  El umbral queda registrado en disco antes de calcularse el veredicto.")
    print()

    # ====================================================================
    # FASE 2.  Veredicto, contra el umbral que la FASE 1 ya dejo fijado.
    # ====================================================================
    print("=" * 78)
    print(f"FASE 2  -  VEREDICTO  (contra el umbral de {100*umbral:.2f} %)")
    print("=" * 78)
    print(f"  {'semilla':>9}{'CRPS A':>10}{'CRPS B':>10}{'reduccion':>12}"
          f"{'DM (HLN)':>11}{'p':>9}{'indiv.':>8}")
    reducciones, n_pasa = [], 0
    for s in semillas:
        la, lb, frac = series[s]
        red = float((la.mean() - lb.mean()) / la.mean())
        dm = diebold_mariano(la, lb, h=H_FUT)
        indiv = (red >= umbral) and (dm["p_one_sided"] < ALPHA)
        n_pasa += int(indiv)
        reducciones.append(red)
        print(f"  {s:>9}{la.mean():>10.4f}{lb.mean():>10.4f}{100*red:>11.2f}%"
              f"{dm['dm_hln']:>11.3f}{dm['p_one_sided']:>9.4f}"
              f"{'SI' if indiv else 'no':>8}")
        salida["semilla_detalle"][str(s)] = {
            "eta": etas[s], "crps_a": float(la.mean()), "crps_b": float(lb.mean()),
            "reduccion": red, "dm": dm, "frac_observada": frac,
            "supera_individual": bool(indiv)}

    red_media = float(np.mean(reducciones))
    red_desv = float(np.std(reducciones))
    p_max = max(salida["semilla_detalle"][str(s)]["dm"]["p_one_sided"]
                for s in semillas)
    p_min = min(salida["semilla_detalle"][str(s)]["dm"]["p_one_sided"]
                for s in semillas)

    print("\n" + "=" * 78)
    print("LECTURA")
    print("=" * 78)
    print(f"  reduccion MEDIA: {100*red_media:.2f} %   (desv. {100*red_desv:.2f})")
    print(f"  umbral fijado en la FASE 1: {100*umbral:.2f} %")
    print(f"  p entre semillas: [{p_min:.4f}, {p_max:.4f}]")
    print(f"  conteo individual (descriptivo): {n_pasa}/{len(semillas)}")

    cumple_magnitud = red_media >= umbral
    # El criterio de significancia sobre la media no esta definido por un unico
    # p: A.5 fijo la media para la MAGNITUD y dejo la significancia sin regla
    # de agregacion, imprecision que el propio anexo consigna. Se adopta la
    # lectura conservadora: mayoria de semillas con p < alpha.
    #
    # Se cuenta sobre p EXCLUSIVAMENTE. `n_pasa` exige ademas superar el umbral
    # de magnitud individualmente, de modo que usarlo aqui mezclaria los dos
    # criterios y haria que la magnitud pesara dos veces.
    n_signif = sum(1 for s in semillas
                   if salida["semilla_detalle"][str(s)]["dm"]["p_one_sided"] < ALPHA)
    cumple_signif = n_signif * 2 > len(semillas)
    verifica = cumple_magnitud and cumple_signif
    print(f"  semillas con p < {ALPHA} (criterio de significancia): "
          f"{n_signif}/{len(semillas)}")

    print()
    if verifica:
        print("  H1 SE VERIFICA bajo el objetivo en cambios.")
        print("  La reduccion media supera el umbral derivado de la potencia y la")
        print("  mayoria de las semillas alcanza significancia. El re-contraste es")
        print("  un experimento NUEVO: no revierte el rechazo bajo el objetivo en")
        print("  niveles, que queda como esta reportado.")
    elif cumple_magnitud and not cumple_signif:
        print("  INCONCLUYENTE: magnitud suficiente, significancia insuficiente.")
        print("  A.4 ordena reportarlo como inconcluyente y NO como confirmacion.")
    elif cumple_signif and not cumple_magnitud:
        print("  INCONCLUYENTE: significancia suficiente, magnitud insuficiente.")
        print("  A.4 ordena reportarlo como inconcluyente y NO como confirmacion.")
    else:
        print("  H1 SE RECHAZA tambien bajo el objetivo en cambios, con eta")
        print("  calibrado a la fraccion declarada y con el umbral derivado de la")
        print("  potencia del diseño. Es el rechazo mas informativo del trabajo:")
        print("  no es atribuible ni a la especificacion del objetivo, ni a la")
        print("  fuerza del prior, ni a un umbral mal fijado.")

    if reducida:
        print("\n  RECORDATORIO: configuracion reducida. Esto NO es el re-contraste.")

    salida["veredicto"] = {
        "reduccion_media": red_media, "reduccion_desv": red_desv,
        "umbral_fijado": umbral, "n_pasa_individual": n_pasa,
        "cumple_magnitud": bool(cumple_magnitud),
        "cumple_significancia": bool(cumple_signif),
        "n_semillas_significativas": n_signif,
        "verifica_h1": bool(verifica)}

    p2 = Path(f"recontraste_h1_cambio_full_{datetime.now().date()}.json")
    p2.write_text(json.dumps(salida, indent=2, ensure_ascii=False, default=float))
    print(f"\n  guardado en {p2}")


if __name__ == "__main__":
    main()
