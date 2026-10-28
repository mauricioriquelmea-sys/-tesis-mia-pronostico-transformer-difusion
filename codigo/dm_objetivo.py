"""
dm_objetivo.py  -  Diebold-Mariano entre objetivo en NIVEL y en CAMBIO
=======================================================================

Es la prueba principal del hallazgo del 13 de octubre: entrenar sobre el
cambio desde el origen en vez de sobre el nivel reduce el CRPS conjunto un
52 % en el panel real. Una diferencia de medias no es evidencia; esto la
contrasta.

La comparacion es PAREADA en el sentido fuerte: mismos origenes, mismas
semillas, mismo panel, misma cabeza, misma via de condicionamiento. Lo unico
que cambia entre los dos JSON es la definicion del objetivo.

  H0: E[L_nivel - L_cambio] = 0
  H1: E[L_nivel - L_cambio] > 0     (es decir, el objetivo en cambio es mejor)

Rezago de Newey-West h-1 = 11 y correccion Harvey-Leybourne-Newbold, igual
que en el contraste de H1.

AISLAMIENTO: lee solo los JSON de resultados, que contienen perdidas por
origen y no el panel.

Uso:
    python dm_objetivo.py condicionante_real_nivel_full_AAAA-MM-DD.json \
                          condicionante_real_cambio_full_AAAA-MM-DD.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from h1_harness import diebold_mariano

H_FUT = 12


def cargar(ruta):
    d = json.loads(Path(ruta).read_text(encoding="utf-8"))
    if "perdida_por_origen" not in d:
        raise SystemExit(
            f"{ruta} no trae 'perdida_por_origen'. Fue generado con una version\n"
            f"anterior del script; hay que repetir esa corrida.")
    return d


def main():
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    a, b = cargar(sys.argv[1]), cargar(sys.argv[2])

    if a.get("objetivo") != "nivel" or b.get("objetivo") != "cambio":
        print(f"  AVISO: objetivos leidos = {a.get('objetivo')} / "
              f"{b.get('objetivo')}; se esperaba nivel / cambio.\n")

    print("=" * 78)
    print("DIEBOLD-MARIANO  -  objetivo en NIVEL contra objetivo en CAMBIO")
    print("=" * 78)
    for k in ("entorno", "config"):
        if a[k] != b[k]:
            print(f"  AVISO: '{k}' difiere entre los dos JSON. Las corridas no")
            print(f"  son estrictamente pareadas; interpretar con cuidado.")
    print(f"  panel: {a['panel_meses']} meses   origenes: "
          f"{a['config']['N_ORIGENES']}   semillas: {a['config']['SEMILLAS']}")
    print(f"  torch {a['entorno']['torch']}\n")
    print("  H0: no hay diferencia    H1: el objetivo en CAMBIO es mejor\n")
    print(f"  {'via':<6}{'semilla':>9}{'CRPS nivel':>12}{'CRPS cambio':>13}"
          f"{'reduccion':>11}{'DM (HLN)':>11}{'p':>9}")
    print("  " + "-" * 71)

    salida = {"fuente_nivel": sys.argv[1], "fuente_cambio": sys.argv[2],
              "resultados": {}}
    for via in ("enc", "lin", "ult"):
        if via not in a["perdida_por_origen"] or via not in b["perdida_por_origen"]:
            continue
        salida["resultados"][via] = {"por_semilla": {}}
        semillas = sorted(set(a["perdida_por_origen"][via])
                          & set(b["perdida_por_origen"][via]))
        for s in semillas:
            la = np.array(a["perdida_por_origen"][via][s], float)
            lb = np.array(b["perdida_por_origen"][via][s], float)
            if la.shape != lb.shape:
                print(f"  {via:<6}{s:>9}   formas distintas, se omite")
                continue
            r = diebold_mariano(la, lb, h=H_FUT)
            red = lb.mean() / la.mean() - 1.0
            print(f"  {via:<6}{s:>9}{la.mean():>12.4f}{lb.mean():>13.4f}"
                  f"{100*red:>10.1f}%{r['dm_hln']:>11.3f}{r['p_one_sided']:>9.4f}")
            salida["resultados"][via]["por_semilla"][s] = {
                "crps_nivel": float(la.mean()), "crps_cambio": float(lb.mean()),
                "reduccion": float(red), **r}
        if len(semillas) > 1:
            la = np.mean([a["perdida_por_origen"][via][s] for s in semillas], 0)
            lb = np.mean([b["perdida_por_origen"][via][s] for s in semillas], 0)
            r = diebold_mariano(la, lb, h=H_FUT)
            red = lb.mean() / la.mean() - 1.0
            print(f"  {via:<6}{'promedio':>9}{la.mean():>12.4f}{lb.mean():>13.4f}"
                  f"{100*red:>10.1f}%{r['dm_hln']:>11.3f}{r['p_one_sided']:>9.4f}")
            salida["resultados"][via]["promedio_semillas"] = {
                "crps_nivel": float(la.mean()), "crps_cambio": float(lb.mean()),
                "reduccion": float(red), **r}
        print("  " + "-" * 71)

    print("\n  LECTURA")
    print("  " + "-" * 71)
    ps = [v["promedio_semillas"]["p_one_sided"]
          for v in salida["resultados"].values() if "promedio_semillas" in v]
    if ps and max(ps) < 0.05:
        print("  El objetivo en cambio vence al objetivo en nivel en LAS TRES vias,")
        print("  con p < 0.05 en todas. El hallazgo queda establecido y la")
        print("  especificacion del objetivo pasa a ser el factor dominante")
        print("  documentado para el panel real.")
    elif ps and min(ps) < 0.05:
        print("  El resultado es mixto entre vias: se reportan las tres sin")
        print("  seleccionar, y la afirmacion se acota a las vias donde p < 0.05.")
    else:
        print("  Ninguna via alcanza p < 0.05. Pese a la diferencia de medias, el")
        print("  efecto NO se distingue del ruido una vez contada la")
        print("  autocorrelacion de los pronosticos solapados. Debe reportarse")
        print("  como no concluyente y NO como un hallazgo.")

    p = Path("dm_objetivo_resultado.json")
    p.write_text(json.dumps(salida, indent=2, ensure_ascii=False))
    print(f"\n  guardado en {p}")


if __name__ == "__main__":
    main()
