"""
diag_calibracion2.py  -  Bloque 2 del martes 13: el problema independiente
===========================================================================

LO QUE QUEDO ABIERTO EL 12 DE OCTUBRE

La cobertura al 90 % esta entre 0.25 y 0.55 en TODAS las configuraciones
probadas --- cuatro cabezas generativas, cuatro agregadores, cinco duraciones
de entrenamiento --- contra 0.921 del oraculo del generador. Ningun cambio en
la cabeza ni en el agregador la movio. Es un tercer problema, independiente de
la informacion y de la capacidad.

Que NO es, segun lo ya medido:
  - no es la familia de la cabeza: gaussiana, Student-t y difusion fallan igual
  - no es el agregador: las cuatro variantes dan 0.31-0.37
  - no es solo sobreajuste: reducir de 800 a 50 pasos lleva la cobertura de
    0.291 a 0.534, mejora real pero muy lejos de 0.90

CUATRO INTERVENCIONES, todas sobre la mejor linea base conocida
(condicionante `ult` + cabeza R0-G, CRPS 1.635, cobertura 0.549):

  base       sin cambios
  inflado    factor de varianza escalar lambda, calibrado FUERA DE MUESTRA
             sobre las ventanas mas recientes anteriores al origen
  inflado_h  un lambda por horizonte (12 parametros en vez de 1)
  cambios    objetivo = cambio respecto del origen en vez de nivel

`inflado` es agnostico a la cabeza: reescala las desviaciones de las muestras
respecto de su media, de modo que sirve igual para R0-G, R0-T o difusion.
Eso importa: si funciona, se aplica tambien a la cabeza que se congele el 15.

PRUEBA DE CORDURA. Si el modelo estuviera bien calibrado, inflar la varianza
EMPEORARIA el CRPS. Si el CRPS mejora al inflar, la sub-dispersion es real y
grande. Esa es la verificacion de que no se esta comprando cobertura con
pronostico.

AISLAMIENTO: solo el generador sintetico.
"""
from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(1)

from bloques_h1 import ventanas_multi, _panel_sintetico, L, H_FUT, N, D, EPS
from cabeza_r0 import CabezaR0
from h1_harness import joint_crps

T_PANEL, T0, NO, M, REFIT, STEPS = 300, 144, 36, 200, 12, 400
N_VAL = 24                      # ventanas de validacion para calibrar lambda
REJILLA = np.concatenate([np.linspace(1.0, 4.0, 31), np.linspace(4.2, 12.0, 40)])
CRPS_ORACULO, COB_ORACULO = 1.1140, 0.921


class CondUltimo(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(N, D)

    def forward(self, X):
        return self.lin(X[:, :, -1])


class Modelo(nn.Module):
    def __init__(self):
        super().__init__()
        self.cond = CondUltimo()
        self.cabeza = CabezaR0(latent_dim=D, rank=5, dist_type="gaussian",
                               v_condicional=False)


# ===========================================================================
def _ventanas(Xn, t_fin, objetivo):
    """objetivo='nivel'  -> Y = Xn[o+1 : o+1+H]          (lo actual)
       objetivo='cambio' -> Y = Xn[o+1 : o+1+H] - Xn[o]  (cambio desde el origen)

    El cambio recentra el objetivo en cero y libera a mu de tener que seguir
    el nivel; con una gaussiana de rango bajo y ~120 ventanas, eso puede ser
    la diferencia entre estimar la varianza del proceso y estimar la del nivel.
    """
    X, Y = ventanas_multi(Xn, L, H_FUT, t_fin)
    if objetivo == "cambio":
        Y = Y - X[:, :, -1].unsqueeze(1)                 # [B,H,N] - [B,1,N]
    return X, Y


def entrenar(panel, t_fin, semilla, objetivo, steps=STEPS, batch=64, lr=1e-3,
             reservar_val=False):
    """reservar_val=True aparta las ultimas N_VAL ventanas y NO entrena con
    ellas.

    Esto no es un detalle. Calibrar lambda sobre ventanas que el modelo ya
    vio da un lambda sistematicamente pequeno ---la sub-dispersion dentro de
    muestra es mucho menor que fuera--- y la correccion no transfiere. En la
    primera version de este script la validacion salia de X[-N_VAL:] sin
    apartarla del entrenamiento: lambda llegaba a lo sumo a 1.8 con la
    rejilla abierta hasta 12.0, y la cobertura en h = 12 se quedaba en 0.45.
    """
    torch.manual_seed(semilla); np.random.seed(semilla)
    mu = panel[:, :t_fin].mean(axis=1, keepdims=True)
    sd = panel[:, :t_fin].std(axis=1, keepdims=True) + EPS
    Xn = (panel - mu) / sd
    X, Y = _ventanas(Xn, t_fin, objetivo)
    if reservar_val:
        Xtr, Ytr = X[:-N_VAL], Y[:-N_VAL]
        Xva, Yva = X[-N_VAL:], Y[-N_VAL:]
    else:
        Xtr, Ytr, Xva, Yva = X, Y, X[-N_VAL:], Y[-N_VAL:]
    m = Modelo()
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    m.train()
    for _ in range(steps):
        i = torch.randint(0, len(Xtr), (min(batch, len(Xtr)),))
        loss = m.cabeza.nll(m.cond(Xtr[i]), Ytr[i])
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
    m.eval()
    return m, mu, sd, Xva, Yva


@torch.no_grad()
def _muestrear(m, X, objetivo, S):
    """[B, S, H, N] en la escala NORMALIZADA, ya reintegrado si objetivo=cambio."""
    z = m.cond(X)
    Y = m.cabeza.predecir(z, S=S)
    if Y.dim() == 3:
        Y = Y.unsqueeze(0)                               # B = 1
    if objetivo == "cambio":
        Y = Y + X[:, :, -1].unsqueeze(1).unsqueeze(1)    # + nivel del origen
    return Y


def calibrar_lambda(m, Xv, Yv, objetivo, por_horizonte, S=200, nivel=0.90):
    """lambda tal que la cobertura sobre las N_VAL ventanas APARTADAS del
    entrenamiento iguale `nivel`. Causal: son anteriores al origen, y el
    modelo no las vio.

    Reescalado agnostico a la cabeza:  y_s <- mu_hat + lambda (y_s - mu_hat).
    """
    if objetivo == "cambio":
        Yv = Yv + Xv[:, :, -1].unsqueeze(1)              # a nivel normalizado
    ens = _muestrear(m, Xv, objetivo, S).numpy()         # [B,S,H,N]
    obs = Yv.numpy()                                     # [B,H,N]
    cen = ens.mean(axis=1, keepdims=True)
    dev = ens - cen
    a = (1 - nivel) / 2

    def cobertura(lam):
        e = cen + lam * dev
        lo = np.quantile(e, a, axis=1); hi = np.quantile(e, 1 - a, axis=1)
        d = (obs >= lo) & (obs <= hi)
        return d.mean(axis=(0, 2)) if por_horizonte else d.mean()

    if not por_horizonte:
        cob = np.array([cobertura(l) for l in REJILLA])
        return float(REJILLA[np.argmin(np.abs(cob - nivel))])
    cobs = np.stack([cobertura(l) for l in REJILLA])      # [n_lam, H]
    return REJILLA[np.argmin(np.abs(cobs - nivel), axis=0)]   # [H]


def evaluar(panel, semilla, variante):
    objetivo = "cambio" if variante == "cambios" else "nivel"
    inflar = variante.startswith("inflado")
    por_h = variante == "inflado_h"

    ens = np.empty((NO, H_FUT, N, M)); y = np.empty((NO, H_FUT, N))
    sc = np.empty((NO, N)); cam = np.diff(panel, axis=1)
    m = mu = sd = None; lam = 1.0; lams = []

    for o_i in range(NO):
        origen = T0 + o_i
        if o_i % REFIT == 0:
            m, mu, sd, Xva, Yva = entrenar(panel, origen, semilla, objetivo,
                                           reservar_val=inflar)
            if inflar:
                lam = calibrar_lambda(m, Xva, Yva, objetivo, por_h)
                lams.append(np.atleast_1d(lam).tolist())
        sc[o_i] = cam[:, :origen].std(axis=1)
        with torch.no_grad():
            Xn = (panel - mu) / sd
            xw = torch.tensor(Xn[:, origen - L + 1: origen + 1][None],
                              dtype=torch.float32)
            Y = _muestrear(m, xw, objetivo, M)[0].numpy()        # [S,H,N]
            if inflar:
                c = Y.mean(axis=0, keepdims=True)
                f = (np.asarray(lam).reshape(1, -1, 1) if por_h else lam)
                Y = c + f * (Y - c)
            Y = Y * sd.T[None, :, :] + mu.T[None, :, :]
            ens[o_i] = Y.transpose(1, 2, 0)
        y[o_i] = panel[:, origen + 1: origen + 1 + H_FUT].T
    return ens, y, sc, lams


def cobertura(ens, y, nivel=0.90):
    a = (1 - nivel) / 2
    lo = np.quantile(ens, a, axis=-1); hi = np.quantile(ens, 1 - a, axis=-1)
    d = (y >= lo) & (y <= hi)
    return float(d.mean()), d.mean(axis=(0, 2)).tolist()


def main():
    ts = datetime.now().isoformat(timespec="seconds")
    panel = _panel_sintetico(T=T_PANEL, seed=12345)
    out = {"timestamp": ts, "torch": torch.__version__,
           "cota_oraculo": {"crps": CRPS_ORACULO, "cob_90": COB_ORACULO},
           "variantes": {}}

    print("=" * 78)
    print("CALIBRACION  -  el problema independiente")
    print("=" * 78)
    print(f"  {ts}   T0={T0}  origenes={NO}  M={M}  pasos={STEPS}")
    print(f"  base: condicionante `ult` + cabeza R0-G")
    print(f"  cota del oraculo: CRPS {CRPS_ORACULO:.4f}, cobertura "
          f"{COB_ORACULO:.3f}\n")
    print(f"  {'variante':<11}{'CRPS':>9}{'vs orac':>10}{'cob 90%':>9}"
          f"{'cob 80%':>9}{'cob 50%':>9}{'lambda':>9}{'seg':>6}")
    print("  " + "-" * 64)

    for var in ("base", "inflado", "inflado_h", "cambios"):
        t0 = time.time()
        ens, y, sc, lams = evaluar(panel, 0, var)
        crps = float(joint_crps(ens, y, sc, standardize=True).mean())
        c90, c90h = cobertura(ens, y, .90)
        c80, _ = cobertura(ens, y, .80)
        c50, _ = cobertura(ens, y, .50)
        lam_txt = ("--" if not lams else
                   f"{np.mean([np.mean(l) for l in lams]):.2f}")
        print(f"  {var:<11}{crps:>9.4f}{100*(crps/CRPS_ORACULO-1):>9.1f}%"
              f"{c90:>9.3f}{c80:>9.3f}{c50:>9.3f}{lam_txt:>9}"
              f"{time.time()-t0:>6.0f}")
        out["variantes"][var] = {
            "crps": crps, "exceso_oraculo": crps / CRPS_ORACULO - 1,
            "cob_90": c90, "cob_80": c80, "cob_50": c50,
            "cob_90_por_horizonte": c90h, "lambdas": lams}

    print()
    b, i = out["variantes"]["base"], out["variantes"]["inflado"]
    print("  PRUEBA DE CORDURA")
    print("  " + "-" * 64)
    if i["crps"] < b["crps"]:
        print(f"  Inflar la varianza MEJORA el CRPS ({b['crps']:.4f} ->"
              f" {i['crps']:.4f}, {100*(i['crps']/b['crps']-1):+.1f} %).")
        print("  La sub-dispersion es real y grande: no se esta comprando")
        print("  cobertura a costa del pronostico.")
    else:
        print(f"  Inflar la varianza EMPEORA el CRPS ({b['crps']:.4f} ->"
              f" {i['crps']:.4f}, {100*(i['crps']/b['crps']-1):+.1f} %).")
        print("  Hay un intercambio: la cobertura se compra con nitidez. Debe")
        print("  reportarse como tal y NO presentarse como mejora neta.")
    print(f"\n  cobertura 90 % por horizonte (base):    "
          f"{[round(x,2) for x in b['cob_90_por_horizonte']]}")
    print(f"  cobertura 90 % por horizonte (inflado): "
          f"{[round(x,2) for x in i['cob_90_por_horizonte']]}")

    p = Path(f"diag_calibracion2_{ts[:10]}.json")
    p.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\n  guardado en {p}")


if __name__ == "__main__":
    main()
