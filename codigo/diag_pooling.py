"""
diag_pooling.py  --  el agregador de tokens, no el transformer
===============================================================

MEDIDO EL 12-10-2026 (diag_encoder.py)

  condicionante                       R^2 estado   CRPS   cob 90 %
  enc  = transformer + h.mean(1)         -5.188    3.036    0.347
  lin  = Linear(45, 32) de la ventana     0.995    2.568    0.454
  ult  = Linear(3, 32) del ultimo mes     1.000    1.635    0.549
  oraculo del generador                       --    1.114    0.921

Condicionar en TRES NUMEROS vence al transformer de 3 capas y 4 cabezas por
46 % de CRPS. La hipotesis que esto contrasta es que la culpa no es del
transformer sino del AGREGADOR: z_enc = h.mean(1) promedia los 45 tokens, y
con la atencion cuasi-uniforme que el DCA ya habia medido (entropia
normalizada 0.97-0.99) ese promedio tiende al promedio de la ventana, que en
una serie con reversion a la media es casi ortogonal al valor ACTUAL.

Si la hipotesis es correcta, basta cambiar el agregador --- sin tocar
bloques, cabezas, mascara ni R_G --- para recuperar la informacion.

CUATRO AGREGADORES
  media      h.mean(1)                                        [lo actual]
  ultimos    los 3 tokens con t mod L = L-1, concatenados
  residual   h.mean(1) + Linear(45, 32) de la ventana cruda
  consulta   pooling por atencion con una consulta aprendida

ADVERTENCIA METODOLOGICA. Esto es analisis EXPLORATORIO POSTERIOR al
contraste prerregistrado de H1. El rechazo de H1 queda como esta reportado:
se fijo antes de ejecutar y se cumplio la regla. Lo que esto agrega es el
MECANISMO del rechazo. Cualquier re-contraste de H1 sobre una arquitectura
corregida es un experimento NUEVO, con prerregistro propio y fecha propia, y
debe declararse como tal. No se reescribe el resultado de ayer con el
diagnostico de hoy.

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

from bloques_h1 import (Phase1Tokenizer, Block, causal_mask_modL,
                        ventanas_multi, _panel_sintetico, L, H_FUT, N, D, EPS,
                        NL, AggregateWatt, PI_G, R_G)
from cabeza_r0 import CabezaR0
from diag_difusion import cobertura
from h1_harness import joint_crps

T_PANEL, T0, NO, M, REFIT, STEPS = 300, 144, 36, 200, 12, 400
ESC = np.array([.4, 40., 2.])
CTR = np.array([3.5, 850., 5.])


class EncoderPool(nn.Module):
    """El transformer de los cuadernos, con el agregador como argumento.
    Bloques, mascara mod-L y proyeccion de atencion intactos."""

    def __init__(self, agregador="media"):
        super().__init__()
        self.agregador = agregador
        self.tok = Phase1Tokenizer()
        self.blocks = nn.ModuleList([Block() for _ in range(NL)])
        self.register_buffer("mask", causal_mask_modL(N, L))
        # indices de los tokens del ultimo rezago: t mod L = L-1
        idx = torch.arange(N * L)
        self.register_buffer("idx_ult", idx[(idx % L) == (L - 1)])   # [N]
        if agregador == "ultimos":
            self.proy = nn.Linear(N * D, D)
        elif agregador == "residual":
            self.salto = nn.Linear(N * L, D)
        elif agregador == "consulta":
            self.q = nn.Parameter(torch.randn(D) / D ** 0.5)
        elif agregador != "media":
            raise ValueError(agregador)

    def forward(self, X, ret_attn=False):
        h = self.tok(X)
        atts = []
        for b in self.blocks:
            h, a = b(h, self.mask)
            atts.append(a)
        if self.agregador == "media":
            z = h.mean(1)
        elif self.agregador == "ultimos":
            z = self.proy(h[:, self.idx_ult].reshape(h.shape[0], -1))
        elif self.agregador == "residual":
            z = h.mean(1) + self.salto(X.reshape(X.shape[0], -1))
        else:
            w = (h @ self.q / D ** 0.5).softmax(-1)
            z = (w.unsqueeze(-1) * h).sum(1)
        return z, (torch.stack(atts) if ret_attn else None)


class Modelo(nn.Module):
    def __init__(self, agregador):
        super().__init__()
        self.cond = EncoderPool(agregador)
        self.cabeza = CabezaR0(latent_dim=D, rank=5, dist_type="gaussian",
                               v_condicional=False)


def _datos(panel, t_fin):
    mu = panel[:, :t_fin].mean(axis=1, keepdims=True)
    sd = panel[:, :t_fin].std(axis=1, keepdims=True) + EPS
    X, Y = ventanas_multi((panel - mu) / sd, L, H_FUT, t_fin)
    return X, Y, mu, sd


def entrenar(panel, t_fin, agregador, semilla, eta=0.0, steps=STEPS,
             batch=64, lr=1e-3):
    torch.manual_seed(semilla); np.random.seed(semilla)
    X, Y, mu, sd = _datos(panel, t_fin)
    m = Modelo(agregador)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    m.train()
    for st in range(steps):
        i = torch.randint(0, len(X), (min(batch, len(X)),))
        z, att = m.cond(X[i], ret_attn=(eta > 0))
        loss = m.cabeza.nll(z, Y[i])
        if eta > 0:
            rg, _ = R_G(att, PI_G)
            loss = loss + eta * rg
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
    m.eval()
    with torch.no_grad():
        _, att = m.cond(X[:min(128, len(X))], ret_attn=True)
        W = AggregateWatt(att).mean(0)
        A = att.mean((0, 1, 2))
        nv = torch.isfinite(causal_mask_modL(N, L)).sum(-1).clamp_min(2).float()
        ent = float((-(A * A.clamp_min(1e-12).log()).sum(-1) / nv.log()).mean())
        masa = float(W[PI_G == 0].sum())
    return m, mu, sd, {"entropia": ent, "masa_prohibida": masa,
                       "params": sum(p.numel() for p in m.parameters()),
                       "n_ventanas": int(len(X))}


def r2_estado(panel, agregador, semilla=0):
    """R^2 fuera de muestra al regresar el estado verdadero sobre z."""
    m, *_ , reg = entrenar(panel, T0, agregador, semilla)
    X, Y, *_ = _datos(panel, T0)
    with torch.no_grad():
        Z = m.cond(X)[0].numpy()
    estado = ((panel - CTR[:, None]) / ESC[:, None])
    U = estado[:, np.arange(L - 1, L - 1 + len(X))].T
    n = len(Z); k = int(.7 * n)
    Ztr = np.c_[Z[:k], np.ones(k)]; Zte = np.c_[Z[k:], np.ones(n - k)]
    coef, *_ = np.linalg.lstsq(Ztr, U[:k], rcond=None)
    ss_res = ((U[k:] - Zte @ coef) ** 2).sum(0)
    ss_tot = ((U[k:] - U[:k].mean(0)) ** 2).sum(0)
    return (1 - ss_res / np.maximum(ss_tot, 1e-12)), reg


def evaluar(panel, agregador, semilla, eta=0.0):
    ens = np.empty((NO, H_FUT, N, M)); y = np.empty((NO, H_FUT, N))
    sc = np.empty((NO, N)); cam = np.diff(panel, axis=1)
    m = mu = sd = None
    for o_i in range(NO):
        origen = T0 + o_i
        if o_i % REFIT == 0:
            m, mu, sd, _ = entrenar(panel, origen, agregador, semilla, eta=eta)
        sc[o_i] = cam[:, :origen].std(axis=1)
        with torch.no_grad():
            Xn = (panel - mu) / sd
            xw = torch.tensor(Xn[:, origen - L + 1: origen + 1][None],
                              dtype=torch.float32)
            z, _ = m.cond(xw)
            Y = m.cabeza.predecir(z, S=M).numpy() * sd.T[None] + mu.T[None]
            ens[o_i] = Y.transpose(1, 2, 0)
        y[o_i] = panel[:, origen + 1: origen + 1 + H_FUT].T
    return ens, y, sc


def main():
    ts = datetime.now().isoformat(timespec="seconds")
    panel = _panel_sintetico(T=T_PANEL, seed=12345)
    out = {"timestamp": ts, "torch": torch.__version__, "P1": {}, "P2": {}}
    CRPS_ORACULO = 1.1140

    print("=" * 78)
    print("EL AGREGADOR DE TOKENS")
    print("=" * 78)
    print(f"  {ts}   T0={T0}  origenes={NO}  M={M}  pasos={STEPS}")
    print(f"  cota del oraculo: CRPS {CRPS_ORACULO:.4f}, cobertura 0.921\n")

    print("-" * 78)
    print("P1  Informacion del estado recuperada por cada agregador")
    print("-" * 78)
    print(f"  {'agregador':<11}{'R2 cu':>9}{'R2 fx':>9}{'R2 tpm':>9}"
          f"{'R2 medio':>11}{'entropia':>10}")
    print("  " + "-" * 59)
    for ag in ("media", "ultimos", "residual", "consulta"):
        r2, reg = r2_estado(panel, ag)
        print(f"  {ag:<11}{r2[0]:>9.3f}{r2[1]:>9.3f}{r2[2]:>9.3f}"
              f"{r2.mean():>11.3f}{reg['entropia']:>10.4f}")
        out["P1"][ag] = {"r2": r2.tolist(), "r2_medio": float(r2.mean()),
                         "entropia": reg["entropia"],
                         "masa_prohibida": reg["masa_prohibida"],
                         "params": reg["params"]}
    print()

    print("-" * 78)
    print("P2  Desempeno predictivo con la misma cabeza R0-G")
    print("-" * 78)
    print(f"  {'agregador':<11}{'CRPS':>10}{'vs oraculo':>12}"
          f"{'cob 90%':>10}{'cob 50%':>10}{'seg':>7}")
    print("  " + "-" * 60)
    for ag in ("media", "ultimos", "residual", "consulta"):
        t0 = time.time()
        ens, y, sc = evaluar(panel, ag, 0)
        crps = float(joint_crps(ens, y, sc, standardize=True).mean())
        c90, c50 = cobertura(ens, y, .90), cobertura(ens, y, .50)
        print(f"  {ag:<11}{crps:>10.4f}{100*(crps/CRPS_ORACULO-1):>11.1f}%"
              f"{c90:>10.3f}{c50:>10.3f}{time.time()-t0:>7.0f}")
        out["P2"][ag] = {"crps": crps, "exceso_oraculo": crps/CRPS_ORACULO-1,
                         "cob_90": c90, "cob_50": c50}
    print()

    p = Path(f"diag_pooling_{ts[:10]}.json")
    p.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"  guardado en {p}")


if __name__ == "__main__":
    main()
