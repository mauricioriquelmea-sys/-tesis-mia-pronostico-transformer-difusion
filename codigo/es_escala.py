"""
es_escala.py  -  el Energy Score del modelo NO es comparable con el de las lineas base
=======================================================================================

HALLAZGO. H2 esta prerregistrada contra un valor concreto: la mejor linea base
es ARIMA/GARCH con Energy Score 1,100, y el umbral del 10 % lo traduce en
ES <= 0,990. Para que ese contraste tenga sentido, el ES del modelo debe
calcularse EXACTAMENTE como se calculo el de la linea base. No se calcula asi.

  MIA_Evaluacion_LineasBase.ipynb        (produjo el 1,100)
      mu, sd = H.mean(axis=0), H.std(axis=0, ddof=1)     # H = NIVELES
      es = energy_score((m_h - mu)/sd, (y_real - mu)/sd)

  h1_harness.energy_score                (lo que usan las corridas del modelo)
      s = scale                                          # sd de los CAMBIOS
      e = ens / s

Dos diferencias:
  1. CENTRADO. Irrelevante: el ES depende solo de X - y y X - X', de modo que
     restar una constante a ambos se cancela. Verificado abajo.
  2. DIVISOR. Decisivo: desviacion de los NIVELES contra desviacion de los
     CAMBIOS. En una serie con tendencia la primera es varias veces la
     segunda, y el ES escala inversamente con ella.

CUAL ES LA CORRECTA. No es cuestion de preferencia. El umbral de H2 esta
anclado a un numero producido con la convencion de NIVELES; el ES del modelo
debe usar esa misma convencion o el contraste compara cosas distintas. La
convencion de CAMBIOS sigue siendo la correcta para el CRPS conjunto de H1,
donde el prerregistro la fija explicitamente y donde ambos brazos la comparten.

AISLAMIENTO: solo el generador sintetico.
"""
from __future__ import annotations

import numpy as np
import torch

torch.set_num_threads(1)

from bloques_h1 import _panel_sintetico, L, H_FUT, N
from h1_harness import energy_score


def energy_score_lineabase(ens, y, mu_h, sd_h):
    """Convencion de MIA_Evaluacion_LineasBase.ipynb: estandariza por media y
    desviacion de los NIVELES de la historia hasta cada origen.

    ens (O,H,N,M), y (O,H,N), mu_h (O,N), sd_h (O,N)  ->  (O,H)
    """
    s = sd_h[:, None, :, None]
    m = mu_h[:, None, :, None]
    e = (ens - m) / s
    t = ((y - mu_h[:, None, :]) / sd_h[:, None, :])[..., None]
    M = e.shape[-1]
    term1 = np.linalg.norm(e - t, axis=2).mean(axis=-1)
    d = e[..., :, None] - e[..., None, :]
    term2 = np.linalg.norm(d, axis=2).sum(axis=(-2, -1)) / (2.0 * M * (M - 1))
    return term1 - term2


def main():
    rng = np.random.default_rng(7)
    panel = _panel_sintetico(T=300, seed=12345)
    T0, NO, M = 144, 36, 200

    print("=" * 78)
    print("ESCALA DEL ENERGY SCORE  -  modelo contra lineas base")
    print("=" * 78)
    print("  panel sintetico; el factor sobre el panel real sera distinto pero")
    print("  el mecanismo es el mismo.\n")

    # --- razon entre los dos divisores, por serie y por origen -------------
    cambios = np.diff(panel, axis=1)
    sd_cam = np.empty((NO, N)); sd_niv = np.empty((NO, N)); mu_niv = np.empty((NO, N))
    for i in range(NO):
        o = T0 + i
        sd_cam[i] = cambios[:, :o].std(axis=1)
        sd_niv[i] = panel[:, :o].std(axis=1, ddof=1)
        mu_niv[i] = panel[:, :o].mean(axis=1)

    print("-" * 78)
    print("1  Razon entre divisores:  sd(niveles) / sd(cambios), por serie")
    print("-" * 78)
    r = sd_niv / sd_cam
    for j, nom in enumerate(("serie 1", "serie 2", "serie 3")):
        print(f"  {nom}:  media {r[:, j].mean():6.2f}   "
              f"rango [{r[:, j].min():.2f}, {r[:, j].max():.2f}]")
    print(f"\n  El ES escala como 1/divisor, de modo que usar la desviacion de")
    print(f"  los CAMBIOS infla el ES por un factor del orden de {r.mean():.1f}")
    print(f"  respecto de la convencion de las lineas base.\n")

    # --- verificacion de que el centrado se cancela ------------------------
    print("-" * 78)
    print("2  El centrado NO afecta al ES  (verificacion)")
    print("-" * 78)
    ens = rng.normal(size=(NO, H_FUT, N, M)) * 2.0 + 1.5
    y = rng.normal(size=(NO, H_FUT, N)) * 2.0 + 1.5
    a = energy_score_lineabase(ens, y, mu_niv, sd_niv)
    b = energy_score_lineabase(ens, y, np.zeros_like(mu_niv), sd_niv)
    err = np.abs(a - b).max() / np.abs(a).max()
    print(f"  con centrado vs sin centrado: error relativo max = {err:.2e}")
    print(f"  -> se cancela, como debe ser. Lo unico que importa es el divisor.\n")

    # --- impacto sobre un conjunto de muestras cualquiera ------------------
    print("-" * 78)
    print("3  Impacto sobre el mismo conjunto de muestras")
    print("-" * 78)
    es_cam = energy_score(ens, y, sd_cam).mean()
    es_niv = energy_score_lineabase(ens, y, mu_niv, sd_niv).mean()
    print(f"  ES con convencion de CAMBIOS   (h1_harness)      : {es_cam:.4f}")
    print(f"  ES con convencion de NIVELES   (lineas base)     : {es_niv:.4f}")
    print(f"  razon                                            : {es_cam/es_niv:.2f}x")
    print()
    print("  CONSECUENCIA. Comparar el ES del modelo, calculado con la primera")
    print("  convencion, contra el umbral de 0,990 derivado de la segunda, es")
    print("  comparar magnitudes distintas. H2 no es contrastable hasta que")
    print("  ambas se calculen igual.")
    print()
    print("  QUE HACER. Recalcular el ES del modelo con `energy_score_lineabase`")
    print("  (esta funcion) sobre el panel real. No hay que volver a entrenar:")
    print("  basta guardar mu y sd de los NIVELES por origen junto al ensemble.")
    print("  El CRPS conjunto de H1 NO cambia: su regla de agregacion esta")
    print("  prerregistrada sobre la desviacion de los cambios y ambos brazos")
    print("  la comparten.")


if __name__ == "__main__":
    main()
