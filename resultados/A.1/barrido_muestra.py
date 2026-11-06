"""
barrido_muestra.py  -  el efecto de R_G frente al tamano de muestra
====================================================================

LA PREGUNTA
--------------------------------------------------------------------------
El informe deja consignada una regularidad que no explica (§5.2.3):

    panel sintetico, Pi(G) VERDADERA por construccion   ->  R_G rinde -1,92 %
    panel chileno,   Pi(G) parcialmente contradicha     ->  R_G rinde +3,80 %

El prior rinde mas donde su contenido es menos exacto. La conjetura ofrecida,
compatible con Fortuin (2022), es que los dos escenarios difieren no solo en la
veracidad del prior sino en el REGIMEN DE INFORMACION: con senal abundante la
Version A aprende la estructura sin asistencia y R_G solo sustrae flexibilidad;
con muestra escasa el prior pasa a operar como regularizador, donde reduce
varianza aunque introduzca sesgo.

Si la conjetura es correcta, sobre el MISMO generador sintetico el efecto de
R_G debe cambiar de signo al reducir la muestra de entrenamiento: negativo con
panel largo, positivo con panel corto.

Este guion lo contrasta.

EL DISENO, Y POR QUE ESTE Y NO OTRO
--------------------------------------------------------------------------
Varia UNA sola cosa: el corte inicial de entrenamiento T0. Todo lo demas queda
fijo, y eso importa por dos confusiones que arruinarian el experimento:

  1. El numero de origenes de evaluacion se mantiene en 109, identico al del
     contraste reportado. Si se dejara crecer con el panel, la potencia del
     Diebold-Mariano crecieria junto con la muestra y no se podria distinguir
     "el efecto cambio" de "ahora puedo detectarlo mejor".

  2. El panel se genera con una UNICA semilla, de modo que las realizaciones
     son anidadas: los primeros 48 meses del panel mas largo son exactamente
     los del mas corto. La manipulacion es entonces "mas datos", no "otros
     datos".

El generador es estacionario por construccion (VAR(1) triangular, §5.2.4), de
modo que los origenes de evaluacion son intercambiables en el tiempo y situarlos
en posiciones distintas del calendario no introduce sesgo.

ANCLA DE VERIFICACION. El punto T0=144 reproduce el contraste sintetico de H1
tal como esta reportado. Si no devuelve una reduccion media cercana a -1,92 %,
algo cambio respecto de aquella corrida y nada de lo demas es interpretable;
el guion lo contrasta explicitamente y lo dice.

CONTROL DE LA FUERZA DEL PRIOR. eta se mantiene en 5, el valor del contraste.
Pero eta fijo no garantiza una penalizacion de fuerza comparable si la escala
de L_simple cambia con el tamano de muestra. El guion reporta la fraccion
estructural eta*R_G/(L_simple + eta*R_G) en cada tamano: si deriva, el barrido
estaria confundiendo "mas datos" con "prior mas debil", y hay que decirlo.

ESTATUS. Analisis exploratorio. No redefine el exito de H1 ni lo re-contrasta:
corre sobre el panel SINTETICO y su resultado responde a la pregunta abierta de
§5.2.3, no a la hipotesis.

AISLAMIENTO. No toca el panel empirico. El generador es el de bloques_h1.py.

USO
--------------------------------------------------------------------------
    python barrido_muestra.py --rapido      # 3 tamanos, 1 semilla, ~10 min
    python barrido_muestra.py               # 6 tamanos, 3 semillas, ~90 min
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

from bloques_h1 import evaluar_ventana_expansiva, _panel_sintetico, H_FUT
from h1_harness import joint_crps, diebold_mariano

# --------------------------------------------------------------------------
PANEL_SEED = 12345          # fijo: realizaciones anidadas
N_ORIGENES = 109            # fijo: identico al contraste reportado
ETA = 5.0                   # fijo: el valor del contraste
M = 200
REFIT = 12
STEPS = 400

TAMANOS = [48, 72, 108, 144, 243, 486]      # 144 = ancla de verificacion
TAMANOS_RAPIDO = [48, 144, 486]

ANCLA_T0 = 144
ANCLA_REDUCCION = -0.0192   # -1,92 % reportado en §5.2.3
ANCLA_TOL = 0.02            # tolerancia absoluta en la reduccion

FACTOR_HLN_MIN = 0.25


def factor_hln(n, h):
    return np.sqrt(max((n + 1 - 2 * h + h * (h - 1) / n) / n, 0.0))


# --------------------------------------------------------------------------
def auditar(tamanos, n_orig, verbose=True):
    """Antes de entrenar nada: que los paneles alcancen, que las ventanas de
    entrenamiento no sean degeneradas y que la prueba sea aplicable."""
    from bloques_h1 import L
    filas = []
    for t0 in tamanos:
        T = t0 + n_orig + H_FUT
        n_vent = max(t0 - (L - 1) - H_FUT, 0)      # len(range(L-1, t0-H))
        filas.append((t0, T, n_vent))
    f = factor_hln(n_orig, H_FUT)

    if verbose:
        print("-" * 78)
        print("  AUDITORIA PREVIA")
        print("-" * 78)
        print(f"  {'T0':>6}{'panel (T)':>12}{'ventanas de entrenamiento':>28}")
        for t0, T, nv in filas:
            aviso = "  <-- pocas" if nv < 30 else ""
            print(f"  {t0:>6}{T:>12}{nv:>28}{aviso}")
        print()
        print(f"  origenes de evaluacion : {n_orig}  (fijo en todos los tamanos)")
        print(f"  factor HLN             : {f:.3f}  "
              f"({'aplicable' if f >= FACTOR_HLN_MIN else 'NO APLICABLE'})")
        print(f"  razon de muestra max/min: {max(tamanos)/min(tamanos):.1f}x")
        print()
        degeneradas = [t0 for t0, _, nv in filas if nv < 20]
        if degeneradas:
            print(f"  AVISO: con T0 en {degeneradas} hay menos de 20 ventanas de")
            print(f"  entrenamiento. El resultado en esos puntos describe un regimen")
            print(f"  de muestra practicamente vacia y debe leerse como tal.\n")
    if f < FACTOR_HLN_MIN:
        raise SystemExit("factor HLN insuficiente: la prueba no seria aplicable")
    return {str(t0): {"panel": T, "ventanas": nv} for t0, T, nv in filas}


def calibrar_eta(t0, semilla, n_orig, objetivo=0.20):
    """Despeja el eta que hace que la fraccion estructural del PRIMER lote
    valga `objetivo`, que es la regla del prerregistro.

        eta*R / (|L| + eta*R) = f    ->    eta = f*|L| / (R*(1-f))

    Cuesta un solo paso de entrenamiento. Devuelve (eta, diagnostico)."""
    from bloques_h1 import entrenar_hasta
    T = t0 + n_orig + H_FUT
    panel = _panel_sintetico(T=T, seed=PANEL_SEED)
    _, _, _, reg = entrenar_hasta(panel, t0, "B", semilla, eta=1.0, steps=1)
    p0 = reg["pasos"][0]
    L, R = abs(p0["L_simple"]), p0["R_G"]
    if R <= 0:
        return ETA, {"error": "R_G nulo en el primer lote", "eta": ETA}
    eta = objetivo * L / (R * (1.0 - objetivo))
    return float(eta), {"L_simple": L, "R_G": R, "eta": float(eta),
                        "objetivo": objetivo}


def una_corrida(t0, semilla, n_orig, pasos=STEPS, muestras=M, eta=ETA):
    """Contraste A vs B para un tamano y una semilla. Devuelve el diccionario
    de resultados y la serie de perdidas por origen de cada version."""
    T = t0 + n_orig + H_FUT
    panel = _panel_sintetico(T=T, seed=PANEL_SEED)

    ens_a, y_a, sc_a, reg_a = evaluar_ventana_expansiva(
        panel, "A", semilla, 0.0, T0=t0, n_origenes=n_orig,
        M=muestras, refit_cada=REFIT, steps=pasos)
    ens_b, y_b, sc_b, reg_b = evaluar_ventana_expansiva(
        panel, "B", semilla, eta, T0=t0, n_origenes=n_orig,
        M=muestras, refit_cada=REFIT, steps=pasos)

    if not np.allclose(y_a, y_b):
        raise RuntimeError("A y B no comparten realizados: el panel difiere")

    ja = joint_crps(ens_a, y_a, sc_a, standardize=True)      # [O, H]
    jb = joint_crps(ens_b, y_b, sc_b, standardize=True)
    la, lb = ja.mean(axis=1), jb.mean(axis=1)                # [O]

    crps_a, crps_b = float(ja.mean()), float(jb.mean())
    reduccion = (crps_a - crps_b) / crps_a
    dm = diebold_mariano(la, lb, h=H_FUT)

    # control de la fuerza efectiva del prior. Se mide en el PRIMER lote de
    # cada reajuste, que es donde el prerregistro define la calibracion de eta;
    # promediarlo con el paso final mezclaria la fuerza inicial con el hecho de
    # que R_G converge a cero durante el entrenamiento.
    fracs = [p["frac_estructural"] for r in reg_b for p in r["pasos"]
             if p.get("paso") == 0 and p.get("frac_estructural", 0) > 0]
    frac = float(np.mean(fracs)) if fracs else float("nan")
    rg_final = float(np.mean([r["pasos"][-1]["R_G"] for r in reg_b]))

    return ({"t0": t0, "semilla": semilla, "eta": eta,
             "crps_a": crps_a, "crps_b": crps_b,
             "reduccion": reduccion, "dm": dm,
             "n_ventanas": reg_a[0]["n_ventanas"],
             "frac_estructural": frac, "R_G_final": rg_final}, la, lb)


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rapido", action="store_true")
    ap.add_argument("--tamanos", nargs="*", type=int, default=None)
    ap.add_argument("--semillas", nargs="*", type=int, default=None)
    ap.add_argument("--origenes", type=int, default=N_ORIGENES)
    ap.add_argument("--pasos", type=int, default=STEPS,
                    help="pasos de entrenamiento por reajuste; solo para probar")
    ap.add_argument("--muestras", type=int, default=M,
                    help="muestras predictivas por origen; solo para probar")
    ap.add_argument("--calibrar-eta", dest="calibrar", action="store_true",
                    help="ajusta eta en cada tamano para igualar la fuerza "
                         "inicial del prior (frac. estructural = 0,20), en vez "
                         "de mantener eta=5. Aisla el tamano de muestra.")
    a = ap.parse_args()

    if a.pasos != STEPS or a.muestras != M or a.origenes != N_ORIGENES:
        print("\n  AVISO: configuracion reducida respecto de la del contraste")
        print(f"  (pasos {a.pasos} vs {STEPS}, muestras {a.muestras} vs {M},")
        print(f"  origenes {a.origenes} vs {N_ORIGENES}). Sirve para verificar el")
        print("  cableado; sus cifras NO son comparables con las del informe y el")
        print("  ancla de verificacion no aplica.\n")

    tamanos = a.tamanos or (TAMANOS_RAPIDO if a.rapido else TAMANOS)
    semillas = a.semillas or ([0] if a.rapido else [0, 1, 2])
    n_orig = a.origenes

    ts = datetime.now().isoformat(timespec="seconds")
    env = {"python": sys.version.split()[0], "torch": torch.__version__,
           "numpy": np.__version__, "plataforma": platform.platform(),
           "hilos": torch.get_num_threads()}

    print("=" * 78)
    print(f"BARRIDO DE TAMANO MUESTRAL{'  [RAPIDO]' if a.rapido else ''}")
    print("=" * 78)
    print(f"  {ts}")
    print(f"  torch {env['torch']} | numpy {env['numpy']} | hilos {env['hilos']}")
    print(f"  generador sintetico VAR(1) triangular, semilla de panel {PANEL_SEED}")
    print(f"  eta = {'calibrado por tamano' if a.calibrar else ETA}"
          f"   objetivo en NIVELES (el del contraste de H1)")
    print(f"  Pi(G) es VERDADERA por construccion en todos los tamanos\n")

    auditoria = auditar(tamanos, n_orig)

    salida = {"configuracion_reducida": bool(
                  a.pasos != STEPS or a.muestras != M or n_orig != N_ORIGENES),
              "pasos": a.pasos, "muestras": a.muestras,
              "timestamp": ts, "entorno": env, "panel": "sintetico",
              "panel_seed": PANEL_SEED,
              "eta": "calibrado" if a.calibrar else ETA, "objetivo": "nivel",
              "n_origenes": n_orig, "M": M, "refit": REFIT, "steps": STEPS,
              "semillas": semillas, "auditoria": auditoria,
              "nota": ("exploratorio; responde la pregunta abierta de la "
                       "regularidad contraintuitiva, no re-contrasta H1"),
              "tamanos": {}}

    print("-" * 78)
    print(f"  {'T0':>6}{'vent.':>7}{'semilla':>9}{'CRPS A':>10}{'CRPS B':>10}"
          f"{'reduccion':>12}{'p (DM)':>9}{'frac':>7}{'seg':>7}")
    print("  " + "-" * 73)

    t_ini = time.time()
    perdidas = {}
    for t0 in tamanos:
        salida["tamanos"][str(t0)] = {"semillas": {}}
        perdidas[t0] = {}
        eta_t0 = ETA
        if a.calibrar:
            eta_t0, diag = calibrar_eta(t0, semillas[0], n_orig)
            salida["tamanos"][str(t0)]["calibracion_eta"] = diag
            print(f"  {t0:>6}{'':>7}{'eta':>9}{eta_t0:>10.3f}"
                  f"   (calibrado a frac. inicial 0,20)")
        salida["tamanos"][str(t0)]["eta"] = eta_t0
        for s in semillas:
            t_run = time.time()
            try:
                r, la, lb = una_corrida(t0, s, n_orig, a.pasos,
                                        a.muestras, eta_t0)
            except Exception as e:
                print(f"  {t0:>6}{'':>7}{s:>9}   FALLA: {type(e).__name__}: {e}")
                salida["tamanos"][str(t0)]["semillas"][str(s)] = {"error": str(e)}
                continue
            perdidas[t0][s] = (la, lb)
            print(f"  {t0:>6}{r['n_ventanas']:>7}{s:>9}{r['crps_a']:>10.4f}"
                  f"{r['crps_b']:>10.4f}{100*r['reduccion']:>11.2f}%"
                  f"{r['dm']['p_one_sided']:>9.4f}{r['frac_estructural']:>7.2f}"
                  f"{time.time()-t_run:>7.0f}")
            salida["tamanos"][str(t0)]["semillas"][str(s)] = r

        vals = [v["reduccion"] for v in
                salida["tamanos"][str(t0)]["semillas"].values()
                if "reduccion" in v]
        if vals:
            salida["tamanos"][str(t0)]["reduccion_media"] = float(np.mean(vals))
            salida["tamanos"][str(t0)]["reduccion_desv"] = float(np.std(vals))
            fr = [v["frac_estructural"] for v in
                  salida["tamanos"][str(t0)]["semillas"].values()
                  if "frac_estructural" in v]
            salida["tamanos"][str(t0)]["frac_estructural_media"] = float(np.mean(fr))
        # DM sobre las perdidas promediadas entre semillas
        if len(perdidas[t0]) > 1:
            la = np.mean([v[0] for v in perdidas[t0].values()], axis=0)
            lb = np.mean([v[1] for v in perdidas[t0].values()], axis=0)
            salida["tamanos"][str(t0)]["dm_promedio"] = diebold_mariano(
                la, lb, h=H_FUT)
        print("  " + "-" * 73)

    print(f"\n  tiempo total: {(time.time()-t_ini)/60:.1f} min\n")

    # --------------------------------------------------- ancla de verificacion
    print("=" * 78)
    print("ANCLA DE VERIFICACION")
    print("=" * 78)
    reducida = (a.pasos != STEPS or a.muestras != M or n_orig != N_ORIGENES)
    anc = salida["tamanos"].get(str(ANCLA_T0), {}).get("reduccion_media")
    if reducida:
        print("  Configuracion reducida: el ancla no aplica y no se contrasta.\n")
        salida["ancla"] = {"corrida": False, "motivo": "configuracion reducida"}
    elif anc is None:
        print(f"  T0={ANCLA_T0} no se corrio: el barrido queda sin ancla y sus")
        print(f"  cifras no son comparables con las reportadas en el informe.\n")
        salida["ancla"] = {"corrida": False}
    else:
        d = abs(anc - ANCLA_REDUCCION)
        ok = d <= ANCLA_TOL
        print(f"  T0={ANCLA_T0}: reduccion media {100*anc:+.2f} %")
        print(f"  reportado en el informe   : {100*ANCLA_REDUCCION:+.2f} %")
        print(f"  discrepancia              : {100*d:.2f} puntos porcentuales")
        if ok:
            print(f"\n  Coincide dentro de la tolerancia. El barrido reproduce el")
            print(f"  contraste reportado y lo extiende; sus cifras son comparables.\n")
        else:
            print(f"\n  DISCREPANCIA. No interpretar el barrido: algo difiere de la")
            print(f"  corrida reportada (entorno, version del codigo, semilla del")
            print(f"  panel). Resolverlo antes de leer nada de lo que sigue.\n")
        salida["ancla"] = {"corrida": True, "reduccion": anc,
                           "reportada": ANCLA_REDUCCION, "discrepancia": d,
                           "coincide": bool(ok)}

    # --------------------------------------------------- control del prior
    print("=" * 78)
    print("CONTROL: la fuerza efectiva del prior se mantuvo comparable?")
    print("=" * 78)
    fr = [(t0, salida["tamanos"][str(t0)].get("frac_estructural_media"))
          for t0 in tamanos if salida["tamanos"][str(t0)].get("frac_estructural_media")]
    if fr:
        vals = [v for _, v in fr]
        razon = max(vals) / min(vals) if min(vals) > 0 else float("inf")
        print(f"  {'T0':>6}{'frac. estructural':>20}")
        for t0, v in fr:
            print(f"  {t0:>6}{v:>20.3f}")
        print(f"\n  razon max/min: {razon:.2f}")
        if razon > 1.5:
            print("  AVISO: la fraccion estructural varia por mas de 1,5x entre")
            print("  tamanos. El barrido esta confundiendo 'mas datos' con 'prior")
            print("  de fuerza distinta'. Hay que recalibrar eta por tamano antes")
            print("  de concluir nada sobre el signo del efecto.")
        else:
            print("  La fuerza efectiva del prior se mantiene comparable entre")
            print("  tamanos, de modo que lo que varia es la muestra y no eta.")
        salida["control_prior"] = {"razon_max_min": razon,
                                   "comparable": bool(razon <= 1.5)}
    print()

    # --------------------------------------------------------------- lectura
    print("=" * 78)
    print("LECTURA")
    print("=" * 78)
    print(f"  {'T0':>6}{'ventanas':>10}{'reduccion':>12}{'desv':>8}"
          f"{'p (DM prom.)':>14}")
    print("  " + "-" * 50)
    pares = []
    for t0 in tamanos:
        d = salida["tamanos"][str(t0)]
        if "reduccion_media" not in d:
            continue
        p = d.get("dm_promedio", {}).get("p_one_sided", float("nan"))
        print(f"  {t0:>6}{auditoria[str(t0)]['ventanas']:>10}"
              f"{100*d['reduccion_media']:>11.2f}%{100*d['reduccion_desv']:>7.2f}"
              f"{p:>14.4f}")
        pares.append((t0, d["reduccion_media"]))

    if len(pares) >= 3:
        xs = np.log([t for t, _ in pares])
        ys = np.array([r for _, r in pares])
        pend = float(np.polyfit(xs, ys, 1)[0])
        signos = [np.sign(r) for _, r in pares]
        cambia = len(set(signos)) > 1
        print(f"\n  pendiente de la reduccion frente a log(T0): {pend:+.4f}")
        print(f"  (negativa = el prior rinde MENOS a medida que crece la muestra)")
        salida["tendencia"] = {"pendiente_log": pend, "cambia_de_signo": bool(cambia)}

        print()
        if cambia and pend < 0:
            print("  EL SIGNO SE INVIERTE y la tendencia es decreciente: el efecto")
            print("  de R_G pasa de positivo con muestra escasa a negativo con")
            print("  muestra abundante, sobre un generador donde Pi(G) es verdadera")
            print("  en todos los puntos. Es la prediccion de la conjetura del")
            print("  informe, y la respalda: lo que cambia el signo es el regimen de")
            print("  informacion y no la veracidad del prior.")
        elif pend < 0 and not cambia:
            print("  La tendencia es decreciente pero el signo no se invierte en el")
            print("  rango barrido. Es compatible con la conjetura y no la establece:")
            print("  habria que extender el barrido hacia muestras mas pequenas, o")
            print("  bien el cambio de signo entre paneles tiene otra causa.")
        elif not cambia:
            print("  Ni el signo cambia ni la tendencia es decreciente. La conjetura")
            print("  NO se sostiene sobre este generador: la diferencia entre el")
            print("  panel sintetico y el chileno debe explicarse por algo distinto")
            print("  del tamano de muestra. Es un resultado y debe reportarse.")
        else:
            print("  El signo cambia pero la tendencia no es monotona decreciente.")
            print("  Revisar la dispersion entre semillas antes de concluir: puede")
            print("  ser ruido y no efecto.")

        print()
        print("  ALCANCE. Esto responde la pregunta abierta de por que el prior")
        print("  rinde mas donde su contenido es menos exacto. NO re-contrasta H1,")
        print("  que se evaluo sobre el panel real y cuyo veredicto queda como esta.")
    else:
        print("\n  menos de tres tamanos validos: no se estima tendencia.")

    modo = "rapido" if a.rapido else "full"
    p = Path(f"barrido_muestra_{modo}_{ts[:10]}.json")
    p.write_text(json.dumps(salida, indent=2, ensure_ascii=False, default=float))
    print(f"\n  guardado en {p}")


if __name__ == "__main__":
    main()
