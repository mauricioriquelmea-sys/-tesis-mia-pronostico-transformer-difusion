"""
eta_sensibilidad.py  -  el rechazo de H1, frente a la fuerza del prior
=======================================================================

POR QUE EXISTE ESTE GUION
-----------------------------------------------------------------------
El anexo de prerregistro (A.1) declara:

    "Calibracion de eta. Se calibra una sola vez, sobre el primer lote, de
     modo que eta*R_G/(L_simple + eta*R_G) = 0,20, y queda fijo durante todo
     el entrenamiento."

La implementacion que produjo las cifras del contraste ---`entrenar_hasta` en
bloques_h1.py--- recibe eta como parametro y NO calibra. El informe reporta
eta = 5. Medido sobre el primer lote del panel sintetico en T0 = 144:

    eta = 5   ->  L_simple 50,91   R_G 0,1141   fraccion estructural 0,011
    eta = 114 ->  L_simple 50,91   R_G 0,1141   fraccion estructural 0,204

Es decir, el contraste se ejecuto con el prior a una fraccion del 1,1 %, no
del 20 % comprometido. Hay una desviacion entre lo declarado y lo ejecutado.

Eso NO invalida el calculo: el experimento se hizo y dio lo que dio. Cambia
QUE se rechazo. Y deja una pregunta que hay que contestar antes del cierre:

    el rechazo de H1, se sostiene cuando el prior se aplica con la fuerza
    que el prerregistro declara?

Este guion la contesta. Fija T0 = 144, 109 origenes y todo lo demas en la
configuracion del contraste, y varia UNICAMENTE eta.

QUE SE REPORTA, Y COMO LEERLO
-----------------------------------------------------------------------
Para cada eta: la reduccion del CRPS conjunto estandarizado de B respecto de
A, su valor-p de Diebold-Mariano con correccion HLN, y la fraccion estructural
efectivamente alcanzada en el primer lote. El punto eta = 5 debe reproducir el
contraste reportado y sirve de ancla.

La lectura tiene tres desenlaces posibles y los tres son reportables:

  (a) El rechazo se sostiene en toda la rejilla. Entonces la desviacion de
      calibracion no altera el veredicto, se declara en el informe como
      desviacion documentada y el resultado queda MAS firme que antes.

  (b) El rechazo se revierte con eta calibrado. Entonces el contraste
      reportado midio un prior a una vigesima parte de su fuerza declarada y
      hay que decirlo con todas sus letras. No se reemplaza el veredicto
      prerregistrado ---que se obtuvo bajo la configuracion ejecutada--- pero
      el informe no puede presentarlo sin esta informacion.

  (c) El efecto es no monotono o indistinguible de cero en toda la rejilla.
      Tambien es un resultado: dice que el desempeno no es sensible a eta en
      el rango explorado, lo que refuerza la disociacion ya documentada.

En ninguno de los tres casos este guion re-contrasta H1. H1 se evaluo bajo
reglas prerregistradas y su veredicto queda como esta.

AISLAMIENTO. Sin --panel corre sobre el generador sintetico y no toca dato
real alguno. Con --panel corre en la maquina del tesista.

USO
-----------------------------------------------------------------------
    python eta_sensibilidad.py --rapido
    python eta_sensibilidad.py                       # sintetico, completo
    python eta_sensibilidad.py --panel panel.csv     # panel real
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

from bloques_h1 import (evaluar_ventana_expansiva, entrenar_hasta,
                        _panel_sintetico, H_FUT)
from h1_harness import joint_crps, diebold_mariano

PANEL_SEED = 12345
T0_SINT = 144
T0_REAL = 143               # indexacion del contraste sobre el panel real
N_ORIGENES = 109
M = 200
REFIT = 12
STEPS = 400
SEMILLAS = (0, 1, 2)

ETA_CONTRASTE = 5.0         # el valor efectivamente usado
FRAC_PRERREGISTRO = 0.20    # el valor declarado en A.1
REJILLA_FRAC = [0.011, 0.05, 0.11, 0.20]   # 0,011 = el del contraste

H1_UMBRAL = 0.05
H1_ALPHA = 0.05


def eta_para_fraccion(panel, t0, semilla, objetivo):
    """eta*R/(|L| + eta*R) = f   ->   eta = f*|L| / (R*(1-f)).
    Cuesta un solo paso de entrenamiento."""
    _, _, _, reg = entrenar_hasta(panel, t0, "B", semilla, eta=1.0, steps=1)
    p0 = reg["pasos"][0]
    L, R = abs(p0["L_simple"]), p0["R_G"]
    if R <= 0:
        raise RuntimeError("R_G nulo en el primer lote: no se puede calibrar")
    return float(objetivo * L / (R * (1.0 - objetivo))), L, R


def corrida(panel, t0, version, semilla, eta, n_orig, pasos, muestras):
    ens, y, sc, reg = evaluar_ventana_expansiva(
        panel, version, semilla, eta, T0=t0, n_origenes=n_orig,
        M=muestras, refit_cada=REFIT, steps=pasos)
    return ens, y, sc, reg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default=None)
    ap.add_argument("--columnas", nargs="*", default=None)
    ap.add_argument("--rapido", action="store_true")
    ap.add_argument("--semillas", nargs="*", type=int, default=None)
    ap.add_argument("--pasos", type=int, default=STEPS)
    ap.add_argument("--muestras", type=int, default=M)
    ap.add_argument("--origenes", type=int, default=N_ORIGENES)
    a = ap.parse_args()

    real = a.panel is not None
    semillas = a.semillas or ([0] if a.rapido else list(SEMILLAS))
    n_orig = 24 if a.rapido else a.origenes
    pasos = 60 if a.rapido else a.pasos
    muestras = 60 if a.rapido else a.muestras
    fracs = [0.011, 0.20] if a.rapido else REJILLA_FRAC
    reducida = a.rapido or (pasos != STEPS or muestras != M
                            or n_orig != N_ORIGENES)

    ts = datetime.now().isoformat(timespec="seconds")
    env = {"python": sys.version.split()[0], "torch": torch.__version__,
           "numpy": np.__version__, "plataforma": platform.platform(),
           "hilos": torch.get_num_threads()}

    print("=" * 78)
    print(f"SENSIBILIDAD DE H1 A LA FUERZA DEL PRIOR  -  "
          f"{'PANEL REAL' if real else 'PANEL SINTETICO'}"
          f"{'  [RAPIDO]' if a.rapido else ''}")
    print("=" * 78)
    print(f"  {ts}")
    print(f"  torch {env['torch']} | numpy {env['numpy']} | hilos {env['hilos']}")
    print(f"  objetivo en NIVELES, el del contraste de H1")
    print(f"  T0 y origenes fijos: lo UNICO que varia es eta\n")
    if reducida:
        print("  AVISO: configuracion reducida. Verifica el cableado; sus cifras")
        print("  NO son comparables con las del informe.\n")

    if real:
        from h1_real import cargar_panel
        panel = cargar_panel(a.panel, a.columnas)
        t0 = T0_REAL
    else:
        t0 = T0_SINT
        panel = _panel_sintetico(T=t0 + n_orig + H_FUT, seed=PANEL_SEED)

    T = panel.shape[1]
    if t0 + n_orig + H_FUT > T:
        raise SystemExit(f"panel de {T} meses insuficiente para T0={t0}, "
                         f"{n_orig} origenes y H={H_FUT}")
    print(f"  panel: {T} meses   T0={t0}   origenes={n_orig}   "
          f"M={muestras}   pasos={pasos}   semillas={tuple(semillas)}\n")

    # ------------------------------------------------- calibracion de la rejilla
    print("-" * 78)
    print("  REJILLA DE ETA  (calibrada sobre el primer lote de cada semilla)")
    print("-" * 78)
    etas = {}
    for s in semillas:
        etas[s] = {}
        for f in fracs:
            e, L, R = eta_para_fraccion(panel, t0, s, f)
            etas[s][f] = e
        if s == semillas[0]:
            print(f"  L_simple = {L:.4f}   R_G = {R:.5f}   (semilla {s})")
            print(f"  {'fraccion':>10}{'eta':>12}{'nota':>34}")
            for f in fracs:
                nota = ""
                if abs(f - 0.011) < 1e-6:
                    nota = "la del contraste (eta~5)"
                elif abs(f - FRAC_PRERREGISTRO) < 1e-6:
                    nota = "la declarada en el prerregistro"
                print(f"  {f:>10.3f}{etas[s][f]:>12.2f}{nota:>34}")
    print()

    salida = {"timestamp": ts, "entorno": env,
              "panel": "real" if real else "sintetico",
              "configuracion_reducida": reducida,
              "T0": t0, "n_origenes": n_orig, "M": muestras, "pasos": pasos,
              "semillas": list(semillas), "objetivo": "nivel",
              "eta_contraste": ETA_CONTRASTE,
              "frac_prerregistro": FRAC_PRERREGISTRO,
              "etas": {str(s): {str(f): v for f, v in d.items()}
                       for s, d in etas.items()},
              "nota": ("exploratorio; NO re-contrasta H1, que queda como esta "
                       "reportada bajo las reglas prerregistradas"),
              "resultados": {}}

    # ------------------------------------------------------------- Version A
    print("-" * 78)
    print("  VERSION A  (sin R_G; comun a toda la rejilla)")
    print("-" * 78)
    base = {}
    for s in semillas:
        t = time.time()
        ens, y, sc, _ = corrida(panel, t0, "A", s, 0.0, n_orig, pasos, muestras)
        ja = joint_crps(ens, y, sc, standardize=True)
        base[s] = (ja, y)
        print(f"  semilla {s}:  CRPS conjunto = {ja.mean():.4f}"
              f"   ({time.time()-t:.0f} s)")
    print()

    # ------------------------------------------------------------- Version B
    print("-" * 78)
    print(f"  {'frac':>7}{'eta':>10}{'semilla':>9}{'CRPS B':>10}"
          f"{'reduccion':>12}{'DM (HLN)':>11}{'p':>9}{'H1':>5}{'seg':>7}")
    print("  " + "-" * 73)
    t_ini = time.time()
    for f in fracs:
        salida["resultados"][str(f)] = {"semillas": {}}
        for s in semillas:
            t = time.time()
            e = etas[s][f]
            ens, y, sc, regb = corrida(panel, t0, "B", s, e, n_orig,
                                       pasos, muestras)
            ja, y_a = base[s]
            if not np.allclose(y, y_a):
                raise RuntimeError("A y B no comparten realizados")
            jb = joint_crps(ens, y, sc, standardize=True)
            red = float((ja.mean() - jb.mean()) / ja.mean())
            dm = diebold_mariano(ja.mean(axis=1), jb.mean(axis=1), h=H_FUT)
            pasa = (red >= H1_UMBRAL) and (dm["p_one_sided"] < H1_ALPHA)
            frac_obs = float(np.mean([p["frac_estructural"] for r in regb
                                      for p in r["pasos"] if p.get("paso") == 0]))
            print(f"  {f:>7.3f}{e:>10.2f}{s:>9}{jb.mean():>10.4f}"
                  f"{100*red:>11.2f}%{dm['dm_hln']:>11.3f}"
                  f"{dm['p_one_sided']:>9.4f}{'SI' if pasa else 'no':>5}"
                  f"{time.time()-t:>7.0f}")
            salida["resultados"][str(f)]["semillas"][str(s)] = {
                "eta": e, "crps_a": float(ja.mean()), "crps_b": float(jb.mean()),
                "reduccion": red, "dm": dm, "pasa": bool(pasa),
                "frac_observada": frac_obs}
        vals = [v["reduccion"] for v in
                salida["resultados"][str(f)]["semillas"].values()]
        salida["resultados"][str(f)]["reduccion_media"] = float(np.mean(vals))
        salida["resultados"][str(f)]["reduccion_desv"] = float(np.std(vals))
        salida["resultados"][str(f)]["n_pasa"] = sum(
            v["pasa"] for v in salida["resultados"][str(f)]["semillas"].values())
        print("  " + "-" * 73)

    print(f"\n  tiempo total: {(time.time()-t_ini)/60:.1f} min\n")

    # ----------------------------------------------------------------- lectura
    print("=" * 78)
    print("LECTURA")
    print("=" * 78)
    print(f"  {'fraccion':>10}{'eta':>10}{'reduccion':>12}{'desv':>8}"
          f"{'semillas que sostienen H1':>28}")
    print("  " + "-" * 68)
    medias = []
    for f in fracs:
        d = salida["resultados"][str(f)]
        e = np.mean([v["eta"] for v in d["semillas"].values()])
        marca = ""
        if abs(f - 0.011) < 1e-6:
            marca = "  <- el contraste"
        elif abs(f - FRAC_PRERREGISTRO) < 1e-6:
            marca = "  <- el prerregistro"
        print(f"  {f:>10.3f}{e:>10.2f}{100*d['reduccion_media']:>11.2f}%"
              f"{100*d['reduccion_desv']:>7.2f}"
              f"{d['n_pasa']:>12}/{len(semillas)}{marca}")
        medias.append((f, d["reduccion_media"], d["n_pasa"]))
    print(f"\n  umbral de H1: reduccion >= {100*H1_UMBRAL:.0f} % Y p < {H1_ALPHA}")
    print()

    n_total = sum(n for _, _, n in medias)
    sostiene_en_prerregistro = next(
        (n for f, _, n in medias if abs(f - FRAC_PRERREGISTRO) < 1e-6), 0)
    sostiene_en_contraste = next(
        (n for f, _, n in medias if abs(f - 0.011) < 1e-6), 0)

    if reducida:
        print("  Configuracion reducida: NO sacar conclusiones. Repetir con la")
        print("  configuracion del contraste antes de interpretar nada.")
    elif n_total == 0:
        print("  (a) El rechazo de H1 SE SOSTIENE en toda la rejilla de eta,")
        print("      incluida la fraccion del 20 % que declara el prerregistro.")
        print("      La desviacion de calibracion no altera el veredicto. Debe")
        print("      declararse igualmente en el informe como desviacion")
        print("      documentada entre lo prerregistrado y lo ejecutado, y el")
        print("      resultado queda mas firme que antes: el rechazo no depende")
        print("      de la fuerza del prior en el rango explorado.")
    elif sostiene_en_prerregistro > 0 and sostiene_en_contraste == 0:
        print("  (b) El rechazo SE REVIERTE al aplicar el prior con la fuerza")
        print("      declarada. El contraste reportado midio un prior a una")
        print("      fraccion del 1,1 % frente al 20 % comprometido.")
        print()
        print("      El veredicto prerregistrado NO se reemplaza: se obtuvo bajo")
        print("      la configuracion efectivamente ejecutada y asi queda. Pero el")
        print("      informe no puede presentarlo sin esta informacion, y la")
        print("      desviacion debe quedar en el anexo con fecha.")
        print()
        print("      Corresponde ademas un re-contraste con prerregistro propio,")
        print("      declarado como experimento nuevo, igual que el del objetivo.")
    else:
        print("  (c) El resultado no es monotono en eta, o la rejilla mezcla")
        print("      desenlaces. Revisar la dispersion entre semillas antes de")
        print("      concluir: con esta potencia, diferencias menores a la")
        print("      desviacion entre semillas no son interpretables.")

    print()
    print("  ALCANCE. Este guion NO re-contrasta H1. H1 se evaluo bajo reglas")
    print("  prerregistradas y su veredicto queda como esta reportado. Lo que")
    print("  aqui se mide es si ese veredicto es sensible a un parametro que el")
    print("  prerregistro fijaba de un modo y la implementacion de otro.")

    modo = "rapido" if a.rapido else "full"
    p = Path(f"eta_sensibilidad_{'real' if real else 'sintetico'}"
             f"_{modo}_{ts[:10]}.json")
    p.write_text(json.dumps(salida, indent=2, ensure_ascii=False, default=float))
    print(f"\n  guardado en {p}")


if __name__ == "__main__":
    main()
