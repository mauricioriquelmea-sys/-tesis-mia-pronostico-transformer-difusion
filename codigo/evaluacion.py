"""
Modulo de evaluacion probabilistica
Proyecto: Forecasting probabilistico multivariado con Transformer-Difusion (INF4980)

Especificacion del protocolo (Seccion 4.4 del informe)
  Ventana expansiva, corte inicial T0 = 144 meses (2003-2014)
  Horizontes h = 1..12
  Reajuste del modelo cada 12 meses; los origenes intermedios reutilizan
  el modelo vigente, que nunca vio datos posteriores a su origen
  109 origenes de pronostico (2015-2024)

Metricas
  CRPS         calidad marginal por serie, estimador insesgado por muestras
  Energy Score calidad de la distribucion CONJUNTA
  Cobertura    fraccion de realizaciones dentro de intervalos al 50 y 90 %
  Diebold-Mariano con correccion HAC de Newey-West y ajuste de muestra
               pequena de Harvey-Leybourne-Newbold

Todos los modelos deben producir un tensor de muestras predictivas de forma
(n_muestras, h, n_series). El modulo es agnostico al modelo: cualquier cosa
que entregue ese tensor es evaluable con las mismas reglas.
"""

from dataclasses import dataclass
from typing import Callable, Dict, List
import numpy as np
import pandas as pd

# ===============================================================
# Configuracion del protocolo
# ===============================================================
T0 = 144            # corte inicial: 2003-01 a 2014-12
H_MAX = 12          # horizonte maximo
REFIT_CADA = 12     # meses entre reajustes del modelo
N_MUESTRAS = 200    # muestras predictivas por origen


# ===============================================================
# Reglas de puntuacion
# ===============================================================
def crps_muestras(muestras: np.ndarray, y: float) -> float:
    """CRPS estimado a partir de muestras (Gneiting y Raftery, 2007).

        CRPS = E|X - y| - (1/2) E|X - X'|

    Se usa el estimador insesgado, que divide el segundo termino por
    m(m-1) y no por m^2. Con m moderado la diferencia no es despreciable:
    el estimador sesgado premia artificialmente a las distribuciones
    demasiado concentradas.
    """
    x = np.sort(np.asarray(muestras, dtype=float))
    m = x.size
    if m < 2:
        return float(abs(x[0] - y))
    termino1 = np.mean(np.abs(x - y))
    # Suma de diferencias absolutas por pares, en O(m log m) usando el orden
    i = np.arange(1, m + 1)
    suma_pares = 2.0 * np.sum((2 * i - m - 1) * x)
    termino2 = suma_pares / (m * (m - 1))
    return float(termino1 - 0.5 * termino2)


def energy_score(muestras: np.ndarray, y: np.ndarray, beta: float = 1.0) -> float:
    """Energy Score multivariado, estimador insesgado por muestras.

        ES = E||X - y|| - (1/2) E||X - X'||

    muestras: (m, d)   y: (d,)

    ADVERTENCIA DE ESCALA: la norma euclidiana no es invariante a las
    unidades. Sobre datos en escala original, la serie de mayor magnitud
    domina el score. Estandarizar ANTES de llamar a esta funcion.
    """
    X = np.asarray(muestras, dtype=float)
    y = np.asarray(y, dtype=float)
    m = X.shape[0]
    t1 = np.mean(np.linalg.norm(X - y, axis=1) ** beta)
    dif = X[:, None, :] - X[None, :, :]
    d = np.linalg.norm(dif, axis=2) ** beta
    t2 = d.sum() / (m * (m - 1))
    return float(t1 - 0.5 * t2)


def cobertura(muestras: np.ndarray, y: float, nivel: float) -> bool:
    """Indicador de si y cae en el intervalo predictivo central."""
    alfa = (1.0 - nivel) / 2.0
    lo, hi = np.quantile(muestras, [alfa, 1.0 - alfa])
    return bool(lo <= y <= hi)


# ===============================================================
# Diebold-Mariano
# ===============================================================
def diebold_mariano(d: np.ndarray, h: int) -> Dict[str, float]:
    """Prueba de Diebold-Mariano (1995) con varianza HAC.

    d: serie de diferenciales de perdida (perdida_A - perdida_B).
       Negativo significa que A es mejor.
    h: horizonte de pronostico.

    Con origenes mensuales y horizonte h, los errores se solapan y d queda
    autocorrelacionado hasta el rezago h-1. Ignorarlo subestima la varianza
    y hace que el test rechace de mas. Se usa Newey-West con h-1 rezagos.

    Se aplica ademas el ajuste de muestra pequena de Harvey, Leybourne y
    Newbold (1997), pertinente aqui: 109 origenes no es asintotico.
    """
    d = np.asarray(d, dtype=float)
    d = d[np.isfinite(d)]
    n = d.size
    if n < 8:
        return {"DM": np.nan, "p": np.nan, "n": n}

    dbar = d.mean()
    e = d - dbar
    gamma0 = np.dot(e, e) / n
    var = gamma0
    for k in range(1, max(h, 1)):
        if k >= n:
            break
        gk = np.dot(e[k:], e[:-k]) / n
        peso = 1.0 - k / float(h)          # nucleo de Bartlett
        var += 2.0 * peso * gk
    if var <= 0:
        return {"DM": np.nan, "p": np.nan, "n": n}

    dm = dbar / np.sqrt(var / n)

    # Ajuste HLN de muestra pequena
    ajuste = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    dm *= ajuste

    from scipy import stats
    p = 2.0 * (1.0 - stats.t.cdf(abs(dm), df=n - 1))
    return {"DM": float(dm), "p": float(p), "n": n}


# ===============================================================
# Ventana expansiva
# ===============================================================
@dataclass
class Origen:
    idx: int                # indice del ultimo mes observado
    fecha: pd.Timestamp
    reajustar: bool         # si corresponde reentrenar en este origen


def origenes(n_obs: int, indice: pd.DatetimeIndex,
             t0: int = T0, h_max: int = H_MAX,
             refit_cada: int = REFIT_CADA) -> List[Origen]:
    """Genera los origenes de pronostico de la ventana expansiva.

    El ultimo origen valido es n_obs - h_max: mas alla no existen las
    12 realizaciones futuras necesarias para evaluar el horizonte completo.
    """
    out = []
    for k, t in enumerate(range(t0, n_obs - h_max + 1)):
        out.append(Origen(idx=t, fecha=indice[t - 1],
                          reajustar=(k % refit_cada == 0)))
    return out


# ===============================================================
# Bucle de evaluacion
# ===============================================================
def evaluar(panel: pd.DataFrame,
            ajustar: Callable,
            predecir: Callable,
            nombre: str,
            n_muestras: int = N_MUESTRAS,
            semilla: int = 20260906) -> pd.DataFrame:
    """Evalua un modelo sobre todos los origenes de la ventana expansiva.

    ajustar(historia: DataFrame) -> objeto de modelo
    predecir(modelo, historia, h_max, n_muestras, rng) -> (n_muestras, h_max, n_series)

    Devuelve un DataFrame con una fila por (origen, horizonte).
    """
    rng = np.random.default_rng(semilla)
    Y = panel.to_numpy(dtype=float)
    series = list(panel.columns)
    filas = []
    modelo = None

    for org in origenes(len(panel), panel.index):
        historia = panel.iloc[:org.idx]
        if org.reajustar or modelo is None:
            modelo = ajustar(historia)

        muestras = predecir(modelo, historia, H_MAX, n_muestras, rng)

        # Estandarizacion para el Energy Score: parametros calculados SOLO
        # con la historia disponible en este origen, nunca con el futuro.
        mu = historia.to_numpy(dtype=float).mean(axis=0)
        sd = historia.to_numpy(dtype=float).std(axis=0, ddof=1)
        sd[sd == 0] = 1.0

        for h in range(1, H_MAX + 1):
            y_real = Y[org.idx + h - 1]
            m_h = muestras[:, h - 1, :]

            fila = {"modelo": nombre, "origen": org.fecha, "h": h}
            for j, s in enumerate(series):
                fila[f"crps_{s}"] = crps_muestras(m_h[:, j], y_real[j])
                fila[f"cob50_{s}"] = cobertura(m_h[:, j], y_real[j], 0.50)
                fila[f"cob90_{s}"] = cobertura(m_h[:, j], y_real[j], 0.90)
            fila["es"] = energy_score((m_h - mu) / sd, (y_real - mu) / sd)
            filas.append(fila)

    return pd.DataFrame(filas)


def resumen(res: pd.DataFrame, series: List[str]) -> pd.Series:
    """Agrega los resultados de un modelo sobre todos los origenes."""
    out = {"n_origenes": res["origen"].nunique()}
    for s in series:
        out[f"CRPS_{s}"] = res[f"crps_{s}"].mean()
        out[f"cob90_{s}"] = res[f"cob90_{s}"].mean()
    out["EnergyScore"] = res["es"].mean()
    return pd.Series(out)


# ===============================================================
# Linea base: random walk
# ===============================================================
def rw_ajustar(historia: pd.DataFrame):
    """Random walk: la escala de la innovacion se estima de la historia.

    Es el piso de referencia. En series financieras mensuales resulta
    sorprendentemente competitivo a horizontes cortos, y por eso se
    incluye: superar a ARIMA sin superar al random walk no es un logro.
    """
    dif = historia.diff().dropna().to_numpy(dtype=float)
    return {"ultimo": historia.to_numpy(dtype=float)[-1],
            "sigma": dif.std(axis=0, ddof=1)}


def rw_predecir(modelo, historia, h_max, n_muestras, rng):
    """Trayectorias del random walk: la varianza crece linealmente con h."""
    d = len(modelo["sigma"])
    pasos = rng.normal(0.0, 1.0, size=(n_muestras, h_max, d)) * modelo["sigma"]
    return modelo["ultimo"] + np.cumsum(pasos, axis=1)


# ===============================================================
# Ejecucion de referencia
# ===============================================================
if __name__ == "__main__":
    from pathlib import Path

    AQUI = Path(__file__).resolve().parent
    panel = pd.read_csv(AQUI / "panel_2003_2024.csv",
                        index_col=0, parse_dates=True)
    series = list(panel.columns)

    orgs = origenes(len(panel), panel.index)
    print("=" * 68)
    print("Protocolo de evaluacion")
    print("=" * 68)
    print(f"  Observaciones      : {len(panel)}")
    print(f"  Corte inicial T0   : {T0}  ({panel.index[T0-1]:%Y-%m})")
    print(f"  Horizontes         : 1 a {H_MAX}")
    print(f"  Origenes           : {len(orgs)}")
    print(f"  Primer origen      : {orgs[0].fecha:%Y-%m}")
    print(f"  Ultimo origen      : {orgs[-1].fecha:%Y-%m}")
    print(f"  Reajustes          : {sum(o.reajustar for o in orgs)}"
          f"  (cada {REFIT_CADA} meses)")
    print(f"  Muestras por origen: {N_MUESTRAS}")

    print("\nEvaluando linea base random walk...")
    res_rw = evaluar(panel, rw_ajustar, rw_predecir, "random_walk")
    res_rw.to_csv(AQUI / "resultados_random_walk.csv", index=False)

    print("\n" + "=" * 68)
    print("Resumen (menor es mejor en CRPS y Energy Score)")
    print("=" * 68)
    print(resumen(res_rw, series).round(4).to_string())

    print("\nCRPS por horizonte:")
    porh = res_rw.groupby("h")[[f"crps_{s}" for s in series]].mean().round(3)
    print(porh.to_string())

    print("\nCobertura al 90 % por serie (nominal: 0.90):")
    for s in series:
        print(f"  {s:8s} {res_rw[f'cob90_{s}'].mean():.3f}")

    print("\nPrueba DM de referencia (random walk contra si mismo, h=1):")
    d = np.zeros(len(orgs))
    print(f"  {diebold_mariano(d, 1)}")
