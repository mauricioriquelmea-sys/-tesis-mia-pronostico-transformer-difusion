"""
Linea base clasica ARIMA/GARCH
Proyecto: Forecasting probabilistico multivariado con Transformer-Difusion (INF4980)

Es la "referencia sin IA" en los terminos del seminario: como se abordaria el
problema con el instrumental econometrico habitual. ARIMA modela la media
condicional y GARCH(1,1) la varianza condicional de los residuos.

DECISION DE DISENO, RELEVANTE PARA LA INTERPRETACION
  El modelo se ajusta serie por serie, de forma univariada. No captura
  dependencia cruzada entre las tres series, y por construccion sus muestras
  predictivas son independientes entre variables. Esa es precisamente la
  limitacion que esta linea base debe exhibir: entrega distribuciones
  MARGINALES bien calibradas pero no una distribucion CONJUNTA. Se espera, en
  consecuencia, que compita en CRPS y quede en desventaja en Energy Score.
  Acoplar artificialmente los residuos la convertiria en otra cosa (un VAR con
  errores multivariados) y desdibujaria el contraste que motiva la tesis.

Transformaciones
  cobre, usdclp -> logaritmo; el ARIMA opera sobre log-niveles y las muestras
                   se devuelven a la escala original por exponenciacion
  tpm           -> nivel; ya es una tasa en puntos porcentuales

Entradas
  panel_2003_2024.csv, evaluacion.py

Salidas
  resultados_arima_garch.csv
  tabla_baselines.csv          resumen comparativo con el random walk
  tabla_dm_baselines.csv       pruebas de Diebold-Mariano
"""

from pathlib import Path
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model

from evaluacion import (evaluar, resumen, diebold_mariano, origenes,
                        rw_ajustar, rw_predecir, H_MAX, N_MUESTRAS, T0)

AQUI = Path(__file__).resolve().parent

# Series que se modelan en logaritmos
LOG = {"cobre", "usdclp"}

# Ordenes candidatos (p, d, q). Rejilla deliberadamente pequena: el objetivo
# es una linea base solida y reproducible, no la busqueda del mejor ARIMA.
ORDENES = [(1, 1, 0), (0, 1, 1), (1, 1, 1), (2, 1, 1), (1, 1, 2)]


def _ajusta_serie(y: np.ndarray):
    """Selecciona el orden ARIMA por BIC y ajusta GARCH(1,1) a los residuos."""
    mejor, mejor_bic = None, np.inf
    for orden in ORDENES:
        try:
            res = ARIMA(y, order=orden,
                        enforce_stationarity=False,
                        enforce_invertibility=False).fit()
            if np.isfinite(res.bic) and res.bic < mejor_bic:
                mejor, mejor_bic = res, res.bic
        except Exception:
            continue
    if mejor is None:
        mejor = ARIMA(y, order=(0, 1, 0)).fit()

    resid = np.asarray(mejor.resid, dtype=float)
    resid = resid[np.isfinite(resid)]

    # El GARCH se ajusta sobre residuos reescalados: arch advierte y pierde
    # precision numerica cuando la serie esta muy lejos de O(1).
    escala = np.std(resid)
    escala = escala if escala > 0 else 1.0
    garch = None
    try:
        garch = arch_model(resid / escala, mean="Zero", vol="GARCH",
                           p=1, q=1, dist="t").fit(disp="off", show_warning=False)
    except Exception:
        garch = None

    return {"arima": mejor, "garch": garch, "escala": escala,
            "sigma_const": float(np.std(resid))}


def ag_ajustar(historia: pd.DataFrame):
    modelos = {}
    for col in historia.columns:
        y = historia[col].to_numpy(dtype=float)
        if col in LOG:
            y = np.log(y)
        modelos[col] = _ajusta_serie(y)
    modelos["_cols"] = list(historia.columns)
    return modelos


def _sendas_varianza(garch, sigma_const, escala, h, n, rng):
    """Trayectorias de desviacion estandar condicional para h pasos.

    Si el GARCH no converge se usa volatilidad constante: es el fallback
    honesto, equivalente a un ARIMA con errores homocedasticos.
    """
    if garch is None:
        return np.full((n, h), sigma_const)
    try:
        f = garch.forecast(horizon=h, reindex=False, method="simulation",
                           simulations=n)
        var = np.asarray(f.simulations.residual_variances[-1])  # (n, h)
        return np.sqrt(np.maximum(var, 1e-12)) * escala
    except Exception:
        try:
            f = garch.forecast(horizon=h, reindex=False)
            var = np.asarray(f.variance.values[-1])             # (h,)
            return np.tile(np.sqrt(np.maximum(var, 1e-12)) * escala, (n, 1))
        except Exception:
            return np.full((n, h), sigma_const)


def ag_predecir(modelo, historia, h_max, n_muestras, rng):
    cols = modelo["_cols"]
    out = np.empty((n_muestras, h_max, len(cols)))

    for j, col in enumerate(cols):
        m = modelo[col]
        media = np.asarray(m["arima"].forecast(steps=h_max), dtype=float)

        sd = _sendas_varianza(m["garch"], m["sigma_const"], m["escala"],
                              h_max, n_muestras, rng)

        # Distribucion t de Student estandarizada: colas mas gruesas que la
        # normal, coherente con el ajuste GARCH con dist="t".
        gl = 6.0
        z = rng.standard_t(gl, size=(n_muestras, h_max))
        z /= np.sqrt(gl / (gl - 2.0))

        # La incertidumbre de un ARIMA integrado se acumula con el horizonte.
        shocks = np.cumsum(z * sd, axis=1)
        sendas = media[None, :] + shocks

        out[:, :, j] = np.exp(sendas) if col in LOG else sendas

    return out


# ===============================================================
if __name__ == "__main__":
    panel = pd.read_csv(AQUI / "panel_2003_2024.csv",
                        index_col=0, parse_dates=True)
    series = list(panel.columns)
    orgs = origenes(len(panel), panel.index)

    print("=" * 70)
    print("Linea base ARIMA/GARCH")
    print("=" * 70)
    print(f"  Origenes: {len(orgs)}   Horizontes: 1-{H_MAX}   "
          f"Muestras: {N_MUESTRAS}")
    print("  Ajustando... (10 reajustes x 3 series)")

    res_ag = evaluar(panel, ag_ajustar, ag_predecir, "arima_garch")
    res_ag.to_csv(AQUI / "resultados_arima_garch.csv", index=False)

    # Random walk como piso de referencia
    res_rw = evaluar(panel, rw_ajustar, rw_predecir, "random_walk")

    tabla = pd.DataFrame({
        "random_walk": resumen(res_rw, series),
        "arima_garch": resumen(res_ag, series),
    }).T
    tabla.to_csv(AQUI / "tabla_baselines.csv")

    print()
    print("=" * 70)
    print("Resumen (menor es mejor en CRPS y Energy Score)")
    print("=" * 70)
    print(tabla.round(4).to_string())

    # -----------------------------------------------------------
    # Diebold-Mariano por serie y horizonte
    # -----------------------------------------------------------
    print()
    print("=" * 70)
    print("Diebold-Mariano: ARIMA/GARCH frente a random walk")
    print("  d = perdida(ARIMA/GARCH) - perdida(RW);  d<0 favorece a ARIMA")
    print("=" * 70)

    filas = []
    for s in series:
        for h in (1, 3, 6, 12):
            a = res_ag[res_ag["h"] == h].set_index("origen")[f"crps_{s}"]
            b = res_rw[res_rw["h"] == h].set_index("origen")[f"crps_{s}"]
            d = (a - b).dropna().to_numpy()
            r = diebold_mariano(d, h)
            filas.append({"serie": s, "h": h,
                          "dif_media": round(float(np.mean(d)), 4),
                          "DM": round(r["DM"], 3) if np.isfinite(r["DM"]) else np.nan,
                          "p": round(r["p"], 4) if np.isfinite(r["p"]) else np.nan,
                          "mejor": ("ARIMA/GARCH" if np.mean(d) < 0 else "random walk")
                                   if np.isfinite(r["p"]) and r["p"] < 0.05 else "sin dif."})
    dm = pd.DataFrame(filas)
    dm.to_csv(AQUI / "tabla_dm_baselines.csv", index=False)
    print(dm.to_string(index=False))

    print()
    print("CRPS por horizonte (ARIMA/GARCH):")
    print(res_ag.groupby("h")[[f"crps_{s}" for s in series]]
          .mean().round(3).to_string())

    print()
    print("Cobertura al 90 % (nominal 0.90):")
    for s in series:
        print(f"  {s:8s} RW={res_rw[f'cob90_{s}'].mean():.3f}   "
              f"AG={res_ag[f'cob90_{s}'].mean():.3f}")
