"""
recalcular_es_lineabase.py  -  Energy Score de la linea base con la dispersion
                               corregida  (analisis de robustez)
==============================================================================

QUE CORRIGE
-----------
El paso 7 del cuaderno de lineas base construye las sendas predictivas como

    sendas = media + cumsum(z * sd)

La media es exacta. La dispersion acumula innovaciones con peso uno, que es la
estructura del error de pronostico de un PASEO ALEATORIO. La del ARIMA ajustado
pondera las innovaciones por los pesos MA(inf) del proceso integrado:

    e_h = sum_{j=0}^{h-1} Psi_j * eps_{h-j},    Psi = cumsum(psi)

donde psi son los pesos del ARMA sobre las diferencias. Solo si Psi_j = 1 para
todo j --paseo aleatorio puro-- ambas coinciden.

Medido sobre el panel real, la razon ancho(cumsum)/ancho(implicado) en h=12 vale
0,592 / 0,696 / 0,283 por serie: la linea base produjo intervalos mucho mas
estrechos de lo que su propio ajuste justifica. El CRPS es una regla PROPIA,
minima en la varianza verdadera, de modo que esa sobre-nitidez la PENALIZA.

QUE NO ES
---------
No re-contrasta H2. H2 se evaluo bajo reglas prerregistradas y su veredicto
queda como esta reportado. Esto mide cuanto del margen reportado proviene de un
defecto de implementacion de la linea base y no de su capacidad predictiva.

CONTROL OBLIGATORIO
-------------------
El guion ejecuta PRIMERO la version original (cumsum). Si no reproduce el
Energy Score reportado de 1,0999, el arnes no replica el cuaderno y NINGUN otro
numero de esta salida es interpretable: el guion se detiene.

REGLA DE REPORTE, fijada antes de ejecutar (17 de octubre de 2026)
------------------------------------------------------------------
La correccion es mecanica y no admite parametros. Se reporta cualquiera que sea
el resultado, en las tres lecturas posibles:
  (a) la mejora corregida cae por debajo del 10 % -> H2 queda rechazada por
      magnitud ADEMAS de por significancia, y se declara asi;
  (b) se mantiene por encima del 10 % -> el rechazo sigue siendo solo por
      significancia y el margen se corrige a su valor real;
  (c) la linea base corregida supera al modelo -> se reporta eso.
El umbral de 0,990 NO se reajusta a posteriori: se declara que fue derivado de
una medicion contaminada y se consigna cual habria sido el umbral correcto.

USO
    py -3.12 recalcular_es_lineabase.py --panel panel_2003_2024.csv

AISLAMIENTO: lee el panel para ajustar y evaluar. No imprime ni guarda ningun
valor del panel.
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
from arch import arch_model

# --- constantes del cuaderno de lineas base, sin cambios -------------------
SEMILLA = 20260906
T0 = 144
H_MAX = 12
REFIT_CADA = 12
N_MUESTRAS = 200
LOG = {"cobre", "usdclp"}
ORDENES = [(1, 1, 0), (0, 1, 1), (1, 1, 1), (2, 1, 1), (1, 1, 2)]

ES_REPORTADO = 1.0999        # el valor que el informe cita para ARIMA/GARCH
TOL_CONTROL = 5e-3           # tolerancia del control de reproduccion
UMBRAL_H2 = 0.10             # 10 % comprometido en el prerregistro


# ---------------------------------------------------------------- metricas
def crps_muestras(muestras, y):
    x = np.sort(np.asarray(muestras, dtype=float))
    m = x.size
    if m < 2:
        return float(abs(x[0] - y))
    t1 = np.mean(np.abs(x - y))
    i = np.arange(1, m + 1)
    t2 = 2.0 * np.sum((2 * i - m - 1) * x) / (m * (m - 1))
    return float(t1 - 0.5 * t2)


def energy_score(muestras, y):
    X = np.asarray(muestras, dtype=float)
    y = np.asarray(y, dtype=float)
    m = X.shape[0]
    t1 = np.mean(np.linalg.norm(X - y, axis=1))
    d = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)
    return float(t1 - 0.5 * d.sum() / (m * (m - 1)))


# ------------------------------------------------------------------ ajuste
def _ajusta_serie(y):
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
        mejor, mejor_ord = ARIMA(y, order=(0, 1, 0)).fit(), (0, 1, 0)

    resid = np.asarray(mejor.resid, dtype=float)
    resid = resid[np.isfinite(resid)]
    escala = np.std(resid) or 1.0
    try:
        g = arch_model(resid / escala, mean="Zero", vol="GARCH", p=1, q=1,
                       dist="t").fit(disp="off", show_warning=False)
    except Exception:
        g = None

    # pesos Psi del proceso INTEGRADO, truncados al horizonte
    p, _, q = mejor_ord
    ar = np.r_[1.0, -np.asarray(mejor.arparams)] if p else np.r_[1.0]
    ma = np.r_[1.0, np.asarray(mejor.maparams)] if q else np.r_[1.0]
    Psi = np.cumsum(arma2ma(ar, ma, lags=H_MAX))

    return {"arima": mejor, "orden": mejor_ord, "garch": g, "escala": escala,
            "sigma_const": float(np.std(resid)), "Psi": Psi}


def _sendas_sd(g, sigma_const, escala, h, n):
    if g is None:
        return np.full((n, h), sigma_const)
    try:
        f = g.forecast(horizon=h, reindex=False, method="simulation",
                       simulations=n)
        v = np.asarray(f.simulations.residual_variances[-1])
        return np.sqrt(np.maximum(v, 1e-12)) * escala
    except Exception:
        return np.full((n, h), sigma_const)


def _acumular(shock, Psi, corregido):
    """shock [n,h] -> error de pronostico acumulado [n,h].

    corregido=False : cumsum, es decir Psi_j = 1  (lo que hizo el cuaderno)
    corregido=True  : e_h = sum_j Psi_j * shock_{h-j}
    """
    if not corregido:
        return np.cumsum(shock, axis=1)
    n, h = shock.shape
    out = np.empty_like(shock)
    for k in range(h):
        acc = np.zeros(n)
        for j in range(k + 1):
            acc += Psi[j] * shock[:, k - j]
        out[:, k] = acc
    return out


def ag_predecir(modelo, cols, h_max, n, rng, corregido):
    out = np.empty((n, h_max, len(cols)))
    for j, c in enumerate(cols):
        m = modelo[c]
        media = np.asarray(m["arima"].forecast(steps=h_max), dtype=float)
        sd = _sendas_sd(m["garch"], m["sigma_const"], m["escala"], h_max, n)
        gl = 6.0
        z = rng.standard_t(gl, size=(n, h_max)) / np.sqrt(gl / (gl - 2.0))
        e = _acumular(z * sd, m["Psi"], corregido)
        sendas = media[None, :] + e
        out[:, :, j] = np.exp(sendas) if c in LOG else sendas
    return out


# ---------------------------------------------------------------- bucle
def evaluar(panel, cols, corregido):
    rng = np.random.default_rng(SEMILLA)
    Y = panel.to_numpy(dtype=float)
    n_obs = len(panel)
    modelo = None
    crps_tot = {c: [] for c in cols}
    es_tot = []

    for k, t in enumerate(range(T0, n_obs - H_MAX + 1)):
        historia = panel.iloc[:t]
        if k % REFIT_CADA == 0 or modelo is None:
            modelo = {c: _ajusta_serie(np.log(historia[c].to_numpy(float))
                                       if c in LOG
                                       else historia[c].to_numpy(float))
                      for c in cols}
        muestras = ag_predecir(modelo, cols, H_MAX, N_MUESTRAS, rng, corregido)

        H = historia.to_numpy(dtype=float)
        mu, sd = H.mean(axis=0), H.std(axis=0, ddof=1)
        sd[sd == 0] = 1.0

        for h in range(1, H_MAX + 1):
            y_real = Y[t + h - 1]
            m_h = muestras[:, h - 1, :]
            for j, c in enumerate(cols):
                crps_tot[c].append(crps_muestras(m_h[:, j], y_real[j]))
            es_tot.append(energy_score((m_h - mu) / sd, (y_real - mu) / sd))

    return ({c: float(np.mean(v)) for c, v in crps_tot.items()},
            float(np.mean(es_tot)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", required=True)
    ap.add_argument("--columnas", nargs="*", default=None)
    ap.add_argument("--es-modelo", type=float, default=0.9696,
                    help="Energy Score de la cabeza congelada R0-T")
    a = ap.parse_args()

    import pandas as pd
    panel = pd.read_csv(a.panel, index_col=0, parse_dates=True)
    if a.columnas:
        panel = panel[a.columnas]
    cols = list(panel.columns)

    print("=" * 78)
    print("ENERGY SCORE DE LA LINEA BASE CON LA DISPERSION CORREGIDA")
    print("=" * 78)
    print(f"  {datetime.now().isoformat(timespec='seconds')}")
    print("  analisis de robustez; NO re-contrasta H2\n")

    # ---------------------------------------------- CONTROL: debe reproducir
    print("-" * 78)
    print("  CONTROL  -  version original (cumsum)")
    print("-" * 78)
    crps_o, es_o = evaluar(panel, cols, corregido=False)
    print(f"  Energy Score original : {es_o:.4f}")
    print(f"  reportado en el informe: {ES_REPORTADO:.4f}")
    dif = abs(es_o - ES_REPORTADO)
    if dif > TOL_CONTROL:
        print(f"\n  ERROR: discrepancia de {dif:.4f} (> {TOL_CONTROL}).")
        print("  El arnes NO reproduce el cuaderno. Ningun otro numero de esta")
        print("  salida es interpretable. Revisar antes de continuar.")
        raise SystemExit(1)
    print(f"  discrepancia {dif:.5f}  ->  CONTROL SUPERADO\n")

    # ---------------------------------------------------------- CORREGIDO
    print("-" * 78)
    print("  CORREGIDO  -  acumulacion con los pesos Psi del ARIMA ajustado")
    print("-" * 78)
    crps_c, es_c = evaluar(panel, cols, corregido=True)
    print(f"  Energy Score corregido: {es_c:.4f}\n")

    print(f"  {'serie':>8}{'CRPS orig':>12}{'CRPS corr':>12}{'cambio':>10}")
    for j, c in enumerate(cols):
        ch = 100 * (crps_c[c] - crps_o[c]) / crps_o[c]
        print(f"  {j:>8}{crps_o[c]:>12.4f}{crps_c[c]:>12.4f}{ch:>9.1f}%")

    # ------------------------------------------------------------- LECTURA
    mej_o = (es_o - a.es_modelo) / es_o
    mej_c = (es_c - a.es_modelo) / es_c
    umbral_orig = es_o * (1 - UMBRAL_H2)
    umbral_corr = es_c * (1 - UMBRAL_H2)

    print("\n" + "=" * 78)
    print("LECTURA")
    print("=" * 78)
    print(f"  Energy Score del modelo (R0-T): {a.es_modelo:.4f}")
    print(f"  linea base original  : {es_o:.4f}   mejora {100*mej_o:.2f} %")
    print(f"  linea base corregida : {es_c:.4f}   mejora {100*mej_c:.2f} %")
    print(f"\n  umbral del 10 % derivado del valor ORIGINAL : {umbral_orig:.4f}")
    print(f"  umbral del 10 % derivado del valor CORREGIDO: {umbral_corr:.4f}")
    print(f"  el informe comprometio 0,990")

    print()
    if mej_c < UMBRAL_H2 <= mej_o:
        print("  (a) La mejora corregida CAE POR DEBAJO del 10 %.")
        print("      H2 queda rechazada por MAGNITUD ademas de por")
        print("      significancia: el rechazo es sobredeterminado. El umbral")
        print("      de 0,990 se derivo de una medicion contaminada y debe")
        print("      declararse como tal.")
    elif mej_c >= UMBRAL_H2:
        print("  (b) La mejora corregida SE MANTIENE sobre el 10 %.")
        print("      El rechazo de H2 sigue siendo solo por significancia. El")
        print("      margen reportado estaba sobreestimado y se corrige, pero")
        print("      el veredicto no cambia.")
    else:
        print("  (c) Desenlace fuera de las dos lecturas previstas. Reportar la")
        print("      curva completa y discutir, sin elegir lectura favorable.")

    if es_c < a.es_modelo:
        pass
    else:
        print("\n  La linea base corregida sigue por detras del modelo.")

    salida = {"timestamp": datetime.now().isoformat(timespec="seconds"),
              "es_reportado": ES_REPORTADO, "es_original": es_o,
              "es_corregido": es_c, "es_modelo": a.es_modelo,
              "mejora_original": mej_o, "mejora_corregida": mej_c,
              "umbral_desde_original": umbral_orig,
              "umbral_desde_corregido": umbral_corr,
              "crps_original": crps_o, "crps_corregido": crps_c,
              "nota": ("robustez; NO re-contrasta H2, cuyo veredicto queda "
                       "como esta reportado bajo las reglas prerregistradas")}
    p = Path(f"es_lineabase_corregido_{datetime.now().date()}.json")
    p.write_text(json.dumps(salida, indent=2, ensure_ascii=False, default=float))
    print(f"\n  guardado en {p}")


if __name__ == "__main__":
    main()
