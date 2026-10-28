"""
diag_encoder.py  --  es la cabeza o es el condicionante?
=========================================================

HALLAZGO QUE MOTIVA ESTO (12-10-2026, diag_calibracion.py)

  oraculo que conoce el generador : CRPS 1.114   cobertura 90 % 0.921
  R0-G sobre z_enc                : CRPS 2.919   cobertura 90 % 0.311
  difusion en y sobre z_enc       : CRPS 2.925   cobertura 90 % 0.995

Dos cabezas de familias distintas, con calibraciones opuestas, dan
practicamente el MISMO CRPS, y ambas quedan 162 % por sobre una cota que es
alcanzable y que esta bien calibrada. Cuando dos modelos muy distintos
convergen al mismo error, el error no esta en ellos: esta en lo que
comparten. Lo que comparten es z_enc.

Y hay un antecedente en el propio informe: el Diagnostico de Colapso de
Atencion (DCA) mide entropia normalizada de la atencion entre 0.97 y 0.99,
es decir atencion practicamente uniforme. Con atencion uniforme y z_enc =
media sobre los 45 tokens, z_enc tiende a un promedio global de la ventana,
que es casi independiente del estado actual.

LO QUE SE MIDE
--------------
E1  Contenido informativo de z_enc. Se regresa linealmente el estado
    verdadero del generador (cu_t, fx_t, tpm_t) sobre z_enc y se reporta R^2
    fuera de muestra. Si z_enc contiene el estado, R^2 ~ 1.

E2  Misma cabeza R0-G, cuatro condicionantes distintos:
      enc    z_enc del transformer (lo actual)
      lin    la ventana cruda [3 x 15] aplanada -> Linear(45, 32)
      ult    solo el ultimo valor de cada serie -> Linear(3, 32)
      ident  el estado verdadero del generador  -> Linear(3, 32)  [techo]
    Si `lin` o `ult` superan a `enc`, el transformer destruye informacion.
    `ident` acota cuanto se puede ganar por la via del condicionante.

AISLAMIENTO: solo el generador sintetico. No se toca el panel real.
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

from bloques_h1 import (EncoderR0, Phase1Tokenizer, Block, causal_mask_modL,
                        ventanas_multi, _panel_sintetico, L, H_FUT, N, D, EPS,
                        NL)
from cabeza_r0 import CabezaR0
from diag_difusion import cobertura
from h1_harness import joint_crps

T_PANEL, T0, NO, M, REFIT, STEPS = 300, 144, 36, 200, 12, 400

# parametros del generador de _panel_sintetico, usados SOLO para construir el
# condicionante `ident` (techo) y la regresion de E1. No entran en ninguna
# cabeza que se vaya a usar sobre datos reales.
ESC = np.array([.4, 40., 2.])
CTR = np.array([3.5, 850., 5.])


# ===========================================================================
# Condicionantes intercambiables
# ===========================================================================
class CondEnc(nn.Module):
    """El transformer completo. z = media sobre los 45 tokens."""

    def __init__(self):
        super().__init__()
        self.tok = Phase1Tokenizer()
        self.blocks = nn.ModuleList([Block() for _ in range(NL)])
        self.register_buffer("mask", causal_mask_modL(N, L))

    def forward(self, X):
        h = self.tok(X)
        for b in self.blocks:
            h, _ = b(h, self.mask)
        return h.mean(1)


class CondLineal(nn.Module):
    """La ventana cruda aplanada. Un solo Linear(45, 32): sin atencion, sin
    capas, sin no linealidad. Es deliberadamente lo mas pobre posible."""

    def __init__(self, d_in=N * L):
        super().__init__()
        self.lin = nn.Linear(d_in, D)

    def forward(self, X):
        return self.lin(X.reshape(X.shape[0], -1))


class CondUltimo(nn.Module):
    """Solo el ultimo mes observado de cada serie: 3 numeros."""

    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(N, D)

    def forward(self, X):
        return self.lin(X[:, :, -1])


class Modelo(nn.Module):
    def __init__(self, cond, cabeza):
        super().__init__()
        self.cond, self.cabeza = cond, cabeza

    def forward(self, X):
        return self.cond(X)


def _cabeza_r0():
    return CabezaR0(latent_dim=D, rank=5, dist_type="gaussian",
                    v_condicional=False)


FABRICAS_COND = {"enc": CondEnc, "lin": CondLineal, "ult": CondUltimo}


# ===========================================================================
def _datos(panel, t_fin):
    mu = panel[:, :t_fin].mean(axis=1, keepdims=True)
    sd = panel[:, :t_fin].std(axis=1, keepdims=True) + EPS
    Xn = (panel - mu) / sd
    X, Y = ventanas_multi(Xn, L, H_FUT, t_fin)
    return X, Y, mu, sd, Xn


def entrenar_cond(panel, t_fin, nombre_cond, semilla, steps=STEPS, batch=64,
                  lr=1e-3):
    torch.manual_seed(semilla)
    np.random.seed(semilla)
    X, Y, mu, sd, _ = _datos(panel, t_fin)
    m = Modelo(FABRICAS_COND[nombre_cond](), _cabeza_r0())
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    m.train()
    p0 = p1 = None
    for st in range(steps):
        i = torch.randint(0, len(X), (min(batch, len(X)),))
        z = m.cond(X[i])
        loss = m.cabeza.nll(z, Y[i])
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        if st == 0:
            p0 = float(loss)
        p1 = float(loss)
    m.eval()
    return m, mu, sd, {"nll_inicial": p0, "nll_final": p1,
                       "params": sum(p.numel() for p in m.parameters()),
                       "n_ventanas": int(len(X))}


def evaluar_cond(panel, nombre_cond, semilla):
    ens = np.empty((NO, H_FUT, N, M)); y = np.empty((NO, H_FUT, N))
    scale = np.empty((NO, N))
    cambios = np.diff(panel, axis=1)
    m = mu = sd = None
    for o_i in range(NO):
        origen = T0 + o_i
        if o_i % REFIT == 0:
            m, mu, sd, _ = entrenar_cond(panel, origen, nombre_cond, semilla)
        scale[o_i] = cambios[:, :origen].std(axis=1)
        with torch.no_grad():
            Xn = (panel - mu) / sd
            xw = torch.tensor(Xn[:, origen - L + 1: origen + 1][None],
                              dtype=torch.float32)
            z = m.cond(xw)
            Y = m.cabeza.predecir(z, S=M).numpy()
            Y = Y * sd.T[None, :, :] + mu.T[None, :, :]
            ens[o_i] = Y.transpose(1, 2, 0)
        y[o_i] = panel[:, origen + 1: origen + 1 + H_FUT].T
    return ens, y, scale


# ===========================================================================
# E1: cuanta informacion sobre el estado hay en z_enc
# ===========================================================================
def e1(panel, semilla=0):
    """R^2 fuera de muestra de la regresion del estado verdadero sobre el
    condicionante. Entrena el modelo hasta T0, extrae z para todas las
    ventanas, parte en 70/30 y regresa."""
    res = {}
    estado = (panel - CTR[:, None]) / ESC[:, None]          # [3, T] latente
    print(f"  {'cond':<6}{'dim':>5}{'R2 cu':>9}{'R2 fx':>9}{'R2 tpm':>9}"
          f"{'R2 medio':>11}")
    print("  " + "-" * 52)
    for nom in ("enc", "lin", "ult"):
        m, mu, sd, _ = entrenar_cond(panel, T0, nom, semilla)
        X, Y, *_ = _datos(panel, T0)
        with torch.no_grad():
            Z = m.cond(X).numpy()                            # [B, 32]
        # el origen de la ventana b es o = L-1+b
        orig = np.arange(L - 1, L - 1 + len(X))
        U = estado[:, orig].T                                # [B, 3]
        n = len(Z); k = int(.7 * n)
        Ztr = np.c_[Z[:k], np.ones(k)]; Zte = np.c_[Z[k:], np.ones(n - k)]
        coef, *_ = np.linalg.lstsq(Ztr, U[:k], rcond=None)
        pred = Zte @ coef
        ss_res = ((U[k:] - pred) ** 2).sum(0)
        ss_tot = ((U[k:] - U[:k].mean(0)) ** 2).sum(0)
        r2 = 1 - ss_res / np.maximum(ss_tot, 1e-12)
        res[nom] = {"r2_por_serie": r2.tolist(), "r2_medio": float(r2.mean())}
        print(f"  {nom:<6}{Z.shape[1]:>5}{r2[0]:>9.3f}{r2[1]:>9.3f}"
              f"{r2[2]:>9.3f}{r2.mean():>11.3f}")
    return res


# ===========================================================================
def main():
    ts = datetime.now().isoformat(timespec="seconds")
    panel = _panel_sintetico(T=T_PANEL, seed=12345)
    out = {"timestamp": ts, "torch": torch.__version__}

    print("=" * 78)
    print("ES LA CABEZA O ES EL CONDICIONANTE?")
    print("=" * 78)
    print(f"  {ts}   T0={T0}  origenes={NO}  M={M}  pasos={STEPS}\n")

    print("-" * 78)
    print("E1  Informacion sobre el estado verdadero contenida en z")
    print("-" * 78)
    print("  R^2 fuera de muestra al regresar (cu_t, fx_t, tpm_t) sobre z.")
    print("  Si z contiene el estado, R^2 -> 1.\n")
    out["E1"] = e1(panel)
    print()

    print("-" * 78)
    print("E2  Misma cabeza R0-G, condicionantes distintos")
    print("-" * 78)
    print(f"  {'cond':<6}{'CRPS':>10}{'cob 90%':>10}{'cob 50%':>10}"
          f"{'params':>10}{'seg':>7}")
    print("  " + "-" * 53)
    out["E2"] = {}
    for nom in ("enc", "lin", "ult"):
        t0 = time.time()
        ens, y, sc = evaluar_cond(panel, nom, 0)
        crps = float(joint_crps(ens, y, sc, standardize=True).mean())
        c90, c50 = cobertura(ens, y, .90), cobertura(ens, y, .50)
        _, _, _, reg = entrenar_cond(panel, T0, nom, 0, steps=1)
        print(f"  {nom:<6}{crps:>10.4f}{c90:>10.3f}{c50:>10.3f}"
              f"{reg['params']:>10}{time.time()-t0:>7.0f}")
        out["E2"][nom] = {"crps": crps, "cob_90": c90, "cob_50": c50,
                          "params": reg["params"]}
    print()

    p = Path(f"diag_encoder_{ts[:10]}.json")
    p.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"  guardado en {p}")


if __name__ == "__main__":
    main()
