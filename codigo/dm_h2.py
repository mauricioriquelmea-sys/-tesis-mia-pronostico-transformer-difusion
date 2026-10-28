"""
dm_h2.py  -  contraste completo de H2 contra la mejor linea base
=================================================================

H2, tal como esta prerregistrada en §3 del informe de avance:

  "El modelo hibrido Transformer--Difusion mejora en al menos un 10 % el
   Energy Score de la mejor linea base, con significancia al 5 % en la prueba
   de Diebold--Mariano. Al cierre de este avance la mejor linea base es
   ARIMA/GARCH, con un Energy Score de 1,100, de modo que el umbral
   corresponde a un valor no superior a 0,990."

Son DOS condiciones y este guion contrasta las dos:

  (a) ES del modelo <= 0,990
  (b) DM contra la serie de perdidas por origen de ARIMA/GARCH, p < 0,05

Ninguna por separado basta. Un ES de 0,95 sin significancia no sostiene H2, y
un p pequeno con ES de 1,05 tampoco.

-------------------------------------------------------------------------------
REQUISITO DE COMPARABILIDAD
-------------------------------------------------------------------------------
Las dos series de Energy Score deben estar en la MISMA estandarizacion. El
1,100 de la linea base se calculo dividiendo por la desviacion de los NIVELES
de la historia hasta cada origen; `h1_harness.energy_score` divide por la de
los CAMBIOS, lo que infla la cifra por un factor del orden de 2 (verificado en
es_escala.py). Por eso `h2_cabezas.py` reporta la columna `ES niv`, que es la
unica que este guion lee.

-------------------------------------------------------------------------------
USO
-------------------------------------------------------------------------------
    1. Ejecutar la celda de exportar_es_lineabase.py en el cuaderno de lineas
       base. Deja es_lineabase_por_origen.npz
    2. Ejecutar h2_cabezas.py sobre el panel real.
    3. python dm_h2.py h2_real_cambio_full_AAAA-MM-DD.json \
                       es_lineabase_por_origen.npz

AISLAMIENTO: lee solo archivos de resultados; ninguno contiene el panel.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from h1_harness import diebold_mariano

H_FUT = 12
ES_UMBRAL = 0.990
FACTOR_HLN_MIN = 0.25


def factor_hln(n, h):
    return np.sqrt(max((n + 1 - 2 * h + h * (h - 1) / n) / n, 0.0))


def main():
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    res = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    base = np.load(sys.argv[2], allow_pickle=True)

    if res.get("panel") != "real":
        print("  AVISO: el JSON no corresponde al panel real. H2 se contrasta")
        print("  contra lineas base ajustadas al panel real; seguir no tiene")
        print("  sentido metodologico.\n")

    es_base = np.asarray(base["arima_garch"], dtype=float)
    es_rw = np.asarray(base["random_walk"], dtype=float)

    print("=" * 78)
    print("CONTRASTE DE H2  -  modelo contra la mejor linea base")
    print("=" * 78)
    print(f"  objetivo de entrenamiento : {res.get('objetivo')}")
    print(f"  correccion de varianza    : {'si' if res.get('inflar') else 'no'}")
    print(f"  origenes (linea base)     : {len(es_base)}")
    print(f"  ARIMA/GARCH  ES = {es_base.mean():.4f}"
          f"     random walk  ES = {es_rw.mean():.4f}")
    print(f"  umbral de H2: ES <= {ES_UMBRAL:.3f}  Y  p < 0,05 en DM\n")

    # --- la perdida por origen del modelo esta en CRPS, no en ES ----------
    # h2_cabezas.py guarda `perdida_por_origen` en CRPS conjunto. Para el DM
    # de H2 hace falta la serie de ES por origen, que hoy no se guarda.
    tiene_es_origen = "es_niveles_por_origen" in res
    if not tiene_es_origen:
        print("-" * 78)
        print("  FALTA UN DATO")
        print("-" * 78)
        print("  El JSON del modelo guarda la perdida por origen en CRPS, que")
        print("  es lo que H1 necesita. H2 se contrasta sobre ENERGY SCORE, de")
        print("  modo que hace falta la serie de ES por origen del modelo.")
        print()
        print("  Se contrasta aqui la condicion (a) --umbral-- y queda pendiente")
        print("  la (b) --significancia--. Para habilitarla, h2_cabezas.py debe")
        print("  guardar tambien es_niveles_por_origen; es una linea.")
        print()

    print("-" * 78)
    print("Condicion (a): umbral del Energy Score")
    print("-" * 78)
    print(f"  {'cabeza':<10}{'ES (niveles)':>14}{'vs base':>10}{'umbral':>18}")
    print("  " + "-" * 50)
    salida = {"fuente_modelo": sys.argv[1], "fuente_base": sys.argv[2],
              "es_lineabase": float(es_base.mean()), "umbral": ES_UMBRAL,
              "cabezas": {}}
    for nom, d in res.get("cabezas", {}).items():
        e = d.get("es_niveles_medio")
        if e is None:
            continue
        mej = 1.0 - e / es_base.mean()
        cumple = e <= ES_UMBRAL
        print(f"  {nom:<10}{e:>14.4f}{100*mej:>9.1f}%"
              f"{'CUMPLE' if cumple else 'no alcanza':>18}")
        salida["cabezas"][nom] = {"es_niveles": e, "mejora": mej,
                                  "cumple_umbral": bool(cumple)}

    # --- condicion (b), si el dato existe ---------------------------------
    if tiene_es_origen:
        print("\n" + "-" * 78)
        print("Condicion (b): Diebold-Mariano contra ARIMA/GARCH")
        print("-" * 78)
        print("  H0: no hay diferencia   H1: el modelo es mejor que la base\n")
        print(f"  {'cabeza':<10}{'semilla':>9}{'dif. media':>13}"
              f"{'DM (HLN)':>11}{'p':>9}")
        print("  " + "-" * 52)
        for nom, series in res["es_niveles_por_origen"].items():
            for s, lb in series.items():
                lb = np.asarray(lb, dtype=float)
                if len(lb) != len(es_base):
                    print(f"  {nom:<10}{s:>9}   longitudes distintas "
                          f"({len(lb)} vs {len(es_base)}), se omite")
                    continue
                f = factor_hln(len(lb), H_FUT)
                if f < FACTOR_HLN_MIN:
                    print(f"  {nom:<10}{s:>9}   DM no aplicable (factor HLN "
                          f"{f:.3f})")
                    continue
                r = diebold_mariano(es_base, lb, h=H_FUT)
                print(f"  {nom:<10}{s:>9}{r['mean_diff']:>13.4f}"
                      f"{r['dm_hln']:>11.3f}{r['p_one_sided']:>9.4f}")
                salida["cabezas"].setdefault(nom, {}).setdefault(
                    "dm_por_semilla", {})[s] = r

    # --- veredicto --------------------------------------------------------
    print("\n" + "=" * 78)
    print("VEREDICTO")
    print("=" * 78)
    cumplen = [n for n, d in salida["cabezas"].items() if d.get("cumple_umbral")]
    if not cumplen:
        print("  Ninguna cabeza alcanza el umbral de 0,990. H2 se rechaza y se")
        print("  reporta como tal: el criterio se fijo antes de evaluar y se")
        print("  cumplio. El informe ya compromete que un efecto inferior al")
        print("  umbral se reporta como hallazgo valido y no como fracaso.")
    elif not tiene_es_origen:
        print(f"  Alcanzan el umbral: {', '.join(cumplen)}")
        print("  Pero H2 exige ADEMAS significancia, que aqui no se pudo")
        print("  contrastar. NO declarar H2 sostenida con la mitad del criterio.")
    else:
        sig = [n for n in cumplen
               if any(v["p_one_sided"] < 0.05 for v in
                      salida["cabezas"][n].get("dm_por_semilla", {}).values())]
        if sig:
            print(f"  Cumplen AMBAS condiciones: {', '.join(sig)}")
            print("  H2 se sostiene para esas cabezas. Verificar que la cabeza")
            print("  en cuestion sea la efectivamente congelada; si H2 se")
            print("  sostiene con la cabeza parametrica y no con la de difusion,")
            print("  el alcance del enunciado cambia y debe decirse.")
        else:
            print(f"  Alcanzan el umbral ({', '.join(cumplen)}) pero ninguna")
            print("  logra significancia al 5 %. H2 no se sostiene: el criterio")
            print("  prerregistrado exige las dos condiciones.")

    p = Path("dm_h2_resultado.json")
    p.write_text(json.dumps(salida, indent=2, ensure_ascii=False))
    print(f"\n  guardado en {p}")


if __name__ == "__main__":
    main()
