"""
verificar_psi_lineabase.py  -  direccion del sesgo de dispersion en ARIMA/GARCH  (v3: registra convergencia)
==============================================================================

El paso 7 del cuaderno de lineas base construye la dispersion predictiva como

    sendas = media + cumsum(z * sd)

La media es exacta: `arima.forecast(steps=12)` devuelve la media condicional a
doce pasos calculada por el espacio de estados. La dispersion, en cambio,
acumula innovaciones con peso uno, que es la estructura del error de un PASEO
ALEATORIO. La del ARIMA ajustado es

    var(h) = sigma^2 * sum_{j<h} Psi_j^2,   Psi = cumsum(psi),

donde psi son los pesos MA(inf) del ARMA sobre las diferencias. Solo cuando
Psi_j = 1 para todo j --paseo aleatorio puro-- ambas coinciden.

Este guion NO re-evalua nada. Reajusta el ARIMA por serie con el mismo criterio
BIC y la misma rejilla de ordenes del cuaderno, y reporta UNICAMENTE la razon
entre el ancho que produce el cumsum y el que implica el ajuste. Sirve para
declarar la direccion del sesgo en el informe.

    > 1  los intervalos de la linea base son MAS ANCHOS de lo que implica su
         propio ajuste (sobredispersion)
    < 1  son MAS ESTRECHOS de lo que implica su propio ajuste (sobrenitidez)

NOTA SOBRE LA LECTURA. El CRPS y el Energy Score son reglas de puntuacion
estrictamente propias: se minimizan cuando la dispersion predictiva coincide
con la verdadera, y penalizan la desviacion EN AMBOS SENTIDOS. Una razon
distinta de 1 no dice si la linea base salio favorecida o perjudicada; dice
solo que su dispersion no era la de su propio modelo. El efecto neto sobre
el Energy Score se obtiene unicamente re-evaluando con la acumulacion
corregida (recalcular_es_lineabase.py). Una version anterior de este guion
afirmaba que razon < 1 implicaba ventaja para la linea base porque "el CRPS
premia la nitidez"; eso es falso para una regla propia, y la re-evaluacion
lo contradijo (ES 1,0999 -> 1,0572, la linea base mejora al corregirse).

USO
    py -3.12 verificar_psi_lineabase.py --panel panel_2003_2024.csv

AISLAMIENTO: lee el panel para ajustar. No imprime ni guarda ningun valor del
panel: la salida son ordenes ARIMA y razones adimensionales.
"""
from __future__ import annotations

import argparse
import json
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima_process import arma2ma

LOG = {"cobre", "usdclp"}
ORDENES = [(1, 1, 0), (0, 1, 1), (1, 1, 1), (2, 1, 1), (1, 1, 2)]
H_MAX = 12
T0 = 144
REFIT = 12
N_ORIGENES = 109


def ajusta(y):
    """Identico al _ajusta_serie del cuaderno, sin la parte GARCH."""
    mejor, mejor_bic, mejor_ord = None, np.inf, None
    for orden in ORDENES:
        try:
            r = ARIMA(y, order=orden, enforce_stationarity=False,
                      enforce_invertibility=False).fit()
            if np.isfinite(r.bic) and r.bic < mejor_bic:
                mejor, mejor_bic, mejor_ord = r, r.bic, orden
        except Exception:
            continue
    if mejor is None:
        mejor = ARIMA(y, order=(0, 1, 0)).fit()
        mejor_ord = (0, 1, 0)
    return mejor, mejor_ord


def razon_ancho(res, orden, h_max=H_MAX):
    """ancho_cumsum / ancho_implicado, por horizonte."""
    p, _, q = orden
    ar = np.r_[1.0, -np.asarray(res.arparams)] if p else np.r_[1.0]
    ma = np.r_[1.0, np.asarray(res.maparams)] if q else np.r_[1.0]
    psi = arma2ma(ar, ma, lags=h_max)
    Psi = np.cumsum(psi)
    var_implicada = np.cumsum(Psi ** 2)
    var_cumsum = np.arange(1, h_max + 1, dtype=float)
    return np.sqrt(var_cumsum / var_implicada)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", required=True)
    ap.add_argument("--columnas", nargs="*", default=None)
    a = ap.parse_args()

    import pandas as pd
    panel = pd.read_csv(a.panel, index_col=0, parse_dates=True)
    if a.columnas:
        panel = panel[a.columnas]
    series = list(panel.columns)

    origenes = list(range(T0, T0 + N_ORIGENES, REFIT))

    print("=" * 78)
    print("SESGO DE DISPERSION DE LA LINEA BASE ARIMA/GARCH")
    print("=" * 78)
    print(f"  {datetime.now().isoformat(timespec='seconds')}")
    print(f"  {len(origenes)} origenes de reajuste, {len(series)} series")
    print("  razon = ancho(cumsum) / ancho(implicado por el ARIMA ajustado)")
    print("  >1 sobredispersa   <1 sobrenitida   (ninguna implica ventaja: ver NOTA)\n")

    salida = {"origenes": origenes, "series": {}}

    for j, s in enumerate(series):
        y_full = (np.log(panel[s].to_numpy(float)) if s in LOG
                  else panel[s].to_numpy(float))
        filas, ordenes_vistos, convergio = [], {}, []
        for o in origenes:
            res, orden = ajusta(y_full[:o])
            ordenes_vistos[str(orden)] = ordenes_vistos.get(str(orden), 0) + 1
            filas.append(razon_ancho(res, orden))
            # Convergencia del ajuste SELECCIONADO (los avisos de statsmodels
            # pueden venir de ordenes descartados por el BIC).
            convergio.append(bool(getattr(res, "mle_retvals", {}).get("converged", True)))
        R = np.array(filas)                       # [origenes, H]
        n_conv = sum(convergio)

        print(f"  serie {j} ({'log' if s in LOG else 'nivel'}):")
        print(f"    ordenes BIC seleccionados: " +
              ", ".join(f"{k}x{v}" for k, v in sorted(ordenes_vistos.items(),
                                                      key=lambda t: -t[1])))
        print(f"    {'h':>4}{'mediana':>10}{'min':>9}{'max':>9}")
        for h in (1, 3, 6, 12):
            col = R[:, h - 1]
            print(f"    {h:>4}{np.median(col):>10.3f}{col.min():>9.3f}{col.max():>9.3f}")
        med12 = float(np.median(R[:, -1]))
        veredicto = ("SOBREDISPERSA (intervalos mas anchos de lo que implica el ajuste)"
                     if med12 > 1.05 else
                     "SOBRENITIDA (intervalos mas estrechos de lo que implica el ajuste)"
                     if med12 < 0.95 else
                     "discrepancia despreciable (<5 % en h=12)")
        print(f"    ajustes seleccionados convergidos: {n_conv}/{len(origenes)}"
              + ("" if n_conv == len(origenes) else
                 "   <-- origenes sin converger: "
                 + ", ".join(str(o) for o, c in zip(origenes, convergio) if not c)))
        print(f"    -> {veredicto}\n")
        salida["series"][str(j)] = {
            "transformacion": "log" if s in LOG else "nivel",
            "ordenes": ordenes_vistos,
            "convergio_por_origen": convergio,
            "n_convergidos": n_conv,
            "razon_mediana_por_h": [float(np.median(R[:, h])) for h in range(H_MAX)],
            "razon_mediana_h12": med12,
            "veredicto": veredicto}

    meds = [v["razon_mediana_h12"] for v in salida["series"].values()]
    print("=" * 78)
    print("LECTURA")
    print("=" * 78)
    print(f"  razon en h=12, por serie: " + ", ".join(f"{m:.3f}" for m in meds))
    if all(m < 0.95 for m in meds):
        print("\n  Las tres series quedan SOBRENITIDAS: la linea base produjo")
        print("  intervalos mas estrechos de lo que su propio ajuste implica.")
    elif all(m > 1.05 for m in meds):
        print("\n  Las tres series quedan SOBREDISPERSAS: la linea base produjo")
        print("  intervalos mas anchos de lo que su propio ajuste implica.")
    else:
        print("\n  El signo difiere entre series. Se reporta por serie.")
    print("\n  Esto establece que la dispersion evaluada NO era la del modelo")
    print("  ajustado. No establece por si solo si eso favorecio o perjudico a")
    print("  la linea base: CRPS y Energy Score son reglas propias y penalizan")
    print("  la desviacion en ambos sentidos. El efecto neto sobre H2 se mide")
    print("  con recalcular_es_lineabase.py (resultado: 1,0999 -> 1,0572).")

    p = Path(f"psi_lineabase_{datetime.now().date()}.json")
    p.write_text(json.dumps(salida, indent=2, ensure_ascii=False, default=float))
    print(f"\n  guardado en {p}")


if __name__ == "__main__":
    main()
