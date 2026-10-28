"""
bloques_h1.py  -  los dos bloques  >>> TUYO <<<  de run_h1.py, completos
=========================================================================

Escrito contra el codigo real:
  - encoder:  MIA_VersionA_Diagnostico / MIA_VersionB_RegularizacionRG
  - cabeza:   cabeza_r0.CabezaR0  (multi-horizonte directa, [S,12,3])
  - R_G:      AggregateWatt del cuaderno, con la escala de R_G_MSE

AISLAMIENTO: este modulo no contiene datos. Recibe el panel ya cargado como
un array [N, T] y nunca lo imprime ni lo transmite.

TRES CAMBIOS RESPECTO DE LOS CUADERNOS, TODOS NECESARIOS
---------------------------------------------------------
1. Desaparece la cabeza latente VAE. z_enc va DIRECTO a CabezaR0, asi que
   L_simple = cabeza.nll(z_enc, y) y ya no hay termino beta*KL. La perdida
   cambia de definicion: los valores de L_simple NO son comparables con los
   de los cuadernos de junio.

2. El objetivo pasa de 1 paso a 12. `windows()` devolvia Y = X[:, s+L],
   un solo mes. R0 modela la conjunta sobre 12 horizontes x 3 series, de
   modo que Y debe ser [B, 12, 3].

3. Ventana expansiva con reajuste cada 12 origenes (preregistro §5). El
   modelo se reentrena ~10 veces por version y por semilla.

CONVENCION DE APLANADO
----------------------
cabeza_r0 usa indice = 3*h + n (horizonte-mayor), y `predecir` ya deshace
el aplanado y devuelve [S, 12, 3] = [M, H, N]. h1_harness espera (H, N, M),
asi que la unica transformacion es una transposicion de ejes.
"""

from __future__ import annotations
import math
import numpy as np
import torch
import torch.nn as nn

from cabeza_r0 import CabezaR0

# --------------------------------------------------------------------------
N, L = 3, 15
D, DFF, NL, H_HEADS = 32, 128, 3, 4
EPS = 1e-6
T_TOK = N * L
H_FUT = 12

PI_G = torch.tensor([[1.0, 0.0, 0.0], [0.5, 0.5, 0.0], [0.0, 0.5, 0.5]])


# --------------------------------------------------------------------------
# Encoder: identico a los cuadernos, pero expone z_enc en vez de (ym, ylv)
# --------------------------------------------------------------------------

def NormalizeX(X):
    return (X - X.mean(-1, keepdim=True)) / (X.std(-1, keepdim=True) + EPS)


def PE_temporal(L, D):
    pe = torch.zeros(L, D)
    pos = torch.arange(L).unsqueeze(1).float()
    div = 10000.0 ** (torch.arange(0, D, 2).float() / D)
    pe[:, 0::2] = torch.sin(pos / div)
    pe[:, 1::2] = torch.cos(pos / div)
    return pe


def causal_mask_modL(N, L):
    idx = torch.arange(N * L)
    ti = idx % L
    return torch.where(ti[None, :] <= ti[:, None], 0.0, float("-inf"))


class Phase1Tokenizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.InputProjection = nn.Linear(1, D)
        self.PE_variable = nn.Embedding(N, D)
        idx_n = torch.arange(N).repeat_interleave(L)
        idx_t = torch.arange(L).repeat(N)
        self.register_buffer("PE_t", PE_temporal(L, D)[idx_t])
        self.register_buffer("idx_n", idx_n)

    def forward(self, X, normalize=False):
        if normalize:
            X = NormalizeX(X)
        T = X.reshape(X.shape[0], N * L, 1)
        return self.InputProjection(T) + (self.PE_t + self.PE_variable(self.idx_n)).unsqueeze(0)


class MHA(nn.Module):
    def __init__(self):
        super().__init__()
        self.q, self.k, self.v, self.o = (nn.Linear(D, D) for _ in range(4))
        self.dk = D // H_HEADS

    def forward(self, x, mask):
        B, S, _ = x.shape
        sp = lambda t: t.view(B, S, H_HEADS, self.dk).transpose(1, 2)
        q, k, v = sp(self.q(x)), sp(self.k(x)), sp(self.v(x))
        att = (q @ k.transpose(-2, -1) / math.sqrt(self.dk) + mask).softmax(-1)
        return self.o((att @ v).transpose(1, 2).reshape(B, S, D)), att


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.mha = MHA()
        self.ln1, self.ln2 = nn.LayerNorm(D), nn.LayerNorm(D)
        self.ff = nn.Sequential(nn.Linear(D, DFF), nn.ReLU(), nn.Linear(DFF, D))

    def forward(self, x, m):
        a, att = self.mha(x, m)
        x = self.ln1(x + a)
        return self.ln2(x + self.ff(x)), att


class EncoderR0(nn.Module):
    """forward -> (z_enc [B,32], att [NL,B,H,45,45] en el grafo)."""

    def __init__(self, dist_type="gaussian", rank=5, v_condicional=False):
        super().__init__()
        self.tok = Phase1Tokenizer()
        self.blocks = nn.ModuleList([Block() for _ in range(NL)])
        self.register_buffer("mask", causal_mask_modL(N, L))
        # v_condicional=False por defecto: con ~100-240 ventanas, un V global
        # ahorra 128*36*r parametros. Es la recomendacion del propio cuaderno R0.
        self.cabeza = CabezaR0(latent_dim=D, rank=rank, dist_type=dist_type,
                               v_condicional=v_condicional)

    def forward(self, X, ret_attn=False):
        h = self.tok(X)
        atts = []
        for b in self.blocks:
            h, a = b(h, self.mask)
            atts.append(a)
        z_enc = h.mean(1)
        return z_enc, (torch.stack(atts) if ret_attn else None)


# --------------------------------------------------------------------------
# R_G con la escala y la proyeccion del cuaderno
# --------------------------------------------------------------------------

def AggregateWatt(att):
    """[NL,B,H,T,T] -> [B,N,N], filas que suman 1 (identico al cuaderno)."""
    A = att.mean((0, 2)).view(-1, N, L, N, L).mean((2, 4))
    return A / A.sum(-1, keepdim=True)


def R_G(att, PiG, verificar=False):
    """mean_{ij} (W_agg - Pi(G))^2  -- la escala de tu R_G_MSE."""
    if verificar:
        filas = att.sum(-1)
        if not torch.allclose(filas, torch.ones_like(filas), atol=1e-4):
            raise ValueError(f"atencion no post-softmax (filas suman {filas.mean():.4f})")
        if torch.is_grad_enabled() and not att.requires_grad:
            raise RuntimeError(
                "atencion desprendida del grafo: R_G sin gradiente, la Version B "
                "entrenaria igual que la A sin lanzar error")
    W = AggregateWatt(att).mean(0)
    return ((W - PiG) ** 2).mean(), W


# --------------------------------------------------------------------------
# Datos: ventanas con objetivo de 12 horizontes
# --------------------------------------------------------------------------

def ventanas_multi(Xn: np.ndarray, L: int, H: int, t_fin: int):
    """
    Xn: [N, T] YA normalizado con estadisticos hasta t_fin (sin fuga).
    Devuelve X [B,N,L], Y [B,H,N] con todas las ventanas cuyo objetivo
    completo cabe dentro de [0, t_fin].

    El origen de la ventana b es o = L-1+b; su entrada son los meses
    [o-L+1 .. o] y su objetivo [o+1 .. o+H].
    """
    xs, ys = [], []
    for o in range(L - 1, t_fin - H):
        xs.append(Xn[:, o - L + 1: o + 1])
        ys.append(Xn[:, o + 1: o + 1 + H].T)          # [H, N]
    if not xs:
        raise ValueError(f"sin ventanas con t_fin={t_fin}, L={L}, H={H}")
    return (torch.tensor(np.stack(xs), dtype=torch.float32),
            torch.tensor(np.stack(ys), dtype=torch.float32))


# --------------------------------------------------------------------------
# BLOQUE 1  ->  reemplaza el  >>> TUYO <<<  de entrenar()
# --------------------------------------------------------------------------

def entrenar_hasta(panel: np.ndarray, t_fin: int, version: str, semilla: int,
                   eta: float, steps=400, batch=64, lr=1e-3, dist_type="gaussian",
                   objetivo="nivel"):
    """
    panel : [N, T] en NIVELES. Nada se normaliza con informacion posterior a t_fin.
    t_fin : ultimo mes observable (exclusivo para objetivos).
    version: 'A' (eta ignorado) o 'B'.
    objetivo: 'nivel'  -> Y = x~_{o+h}          (el del contraste original)
              'cambio' -> Y = x~_{o+h} - x~_o   (el del re-contraste, anexo A.4)

    El valor por defecto 'nivel' deja esta funcion BIT A BIT identica a su
    version anterior: el parametro no toca el flujo del PRNG ni el grafo. Las
    cifras ya reportadas siguen siendo reproducibles sin cambiar una linea de
    quien la llama.

    Devuelve (modelo, mu, sd, registro). mu y sd son los estadisticos de
    normalizacion calculados SOLO hasta t_fin; hay que reutilizarlos al evaluar.
    """
    if objetivo not in ("nivel", "cambio"):
        raise ValueError(f"objetivo desconocido: {objetivo!r}")

    torch.manual_seed(semilla)
    np.random.seed(semilla)

    mu = panel[:, :t_fin].mean(axis=1, keepdims=True)
    sd = panel[:, :t_fin].std(axis=1, keepdims=True) + EPS
    Xn = (panel - mu) / sd                                  # normaliza todo con stats de t_fin

    Xtr, Ytr = ventanas_multi(Xn, L, H_FUT, t_fin)
    if objetivo == "cambio":
        # Xtr [B,N,L] -> ultimo mes observado de cada ventana: Xtr[:,:,-1] [B,N].
        # Ytr es [B,H,N], asi que el origen entra con unsqueeze(1) y se difunde
        # sobre los H horizontes. Identico a h2_cabezas._datos.
        Ytr = Ytr - Xtr[:, :, -1].unsqueeze(1)
    m = EncoderR0(dist_type=dist_type)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    PiG = PI_G
    registro = {"version": version, "semilla": semilla, "eta": eta,
                "t_fin": t_fin, "n_ventanas": len(Xtr), "pasos": []}

    m.train()
    for st in range(steps):
        idx = torch.randint(0, len(Xtr), (min(batch, len(Xtr)),))
        X, Y = Xtr[idx], Ytr[idx]

        z, att = m(X, ret_attn=(version == "B"))
        ls = m.cabeza.nll(z, Y)                              # L_simple (sin KL)

        if version == "B":
            rg, _ = R_G(att, PiG, verificar=(st == 0))
            loss = ls + eta * rg
        else:
            rg = torch.tensor(0.0)
            loss = ls

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()

        if st == 0 or st == steps - 1:
            c = eta * rg.item()
            registro["pasos"].append({
                "paso": st, "L_simple": ls.item(), "R_G": rg.item(),
                "frac_estructural": c / (abs(ls.item()) + c) if c > 0 else 0.0})

    with torch.no_grad():
        m.eval()
        _, att = m(Xtr[:min(128, len(Xtr))], ret_attn=True)
        W = AggregateWatt(att).mean(0)
        A = att.mean((0, 1, 2))
        nv = torch.isfinite(causal_mask_modL(N, L)).sum(-1).clamp_min(2).float()
        registro["masa_prohibida"] = W[PI_G == 0].sum().item()
        registro["entropia_normalizada"] = float(
            (-(A * A.clamp_min(1e-12).log()).sum(-1) / nv.log()).mean())
        registro["W_agg"] = W.tolist()

    return m, mu, sd, registro


# --------------------------------------------------------------------------
# BLOQUE 2  ->  reemplaza el  >>> TUYO <<<  de evaluar()
# --------------------------------------------------------------------------

def evaluar_ventana_expansiva(panel: np.ndarray, version: str, semilla: int,
                              eta: float, T0=144, n_origenes=109, M=200,
                              refit_cada=12, steps=400, dist_type="gaussian",
                              objetivo="nivel"):
    """
    Devuelve (ens, y, scale) con las formas que exige h1_harness:
        ens   (O, H, N, M)
        y     (O, H, N)      en NIVELES, igual que las muestras
        scale (O, N)         desviacion de los CAMBIOS hasta cada origen

    El modelo se reentrena cada `refit_cada` origenes con datos hasta ese
    origen. Las muestras se devuelven DESNORMALIZADAS para que el CRPS se
    calcule en la misma escala que las lineas base.
    """
    Np, T = panel.shape
    if T < T0 + n_origenes + H_FUT:
        raise ValueError(f"panel de {T} meses insuficiente para T0={T0}, "
                         f"{n_origenes} origenes y H={H_FUT}")

    ens = np.empty((n_origenes, H_FUT, N, M))
    y = np.empty((n_origenes, H_FUT, N))
    scale = np.empty((n_origenes, N))

    cambios = np.diff(panel, axis=1)                         # [N, T-1]
    m = mu = sd = None
    registros = []

    for o_i in range(n_origenes):
        origen = T0 + o_i

        if o_i % refit_cada == 0:
            m, mu, sd, reg = entrenar_hasta(panel, origen, version, semilla,
                                            eta, steps=steps, dist_type=dist_type,
                                            objetivo=objetivo)
            registros.append(reg)
            m.eval()

        # escala: SOLO cambios hasta el origen. Nunca posteriores.
        scale[o_i] = cambios[:, :origen].std(axis=1)

        # El muestreo va bajo no_grad; el reentrenamiento de arriba NO, porque
        # envolver todo el bucle deja a loss.backward() sin grafo.
        with torch.no_grad():
            Xn = (panel - mu) / sd
            x_win = torch.tensor(Xn[:, origen - L + 1: origen + 1][None],
                                 dtype=torch.float32)        # [1, N, L]
            z, _ = m(x_win, ret_attn=False)

            Y = m.cabeza.predecir(z, S=M)                    # [M, H, N] normalizado
            if objetivo == "cambio":
                # La cabeza predijo CAMBIOS respecto del origen; hay que
                # devolver el nivel antes de desnormalizar. x_win[0,:,-1] es
                # [N] y se difunde sobre [M,H,N]. Inverso exacto de la resta
                # aplicada a Ytr en entrenar_hasta().
                Y = Y + x_win[0, :, -1]
            Y = Y.numpy() * sd.T[None, :, :] + mu.T[None, :, :]   # -> niveles
            ens[o_i] = Y.transpose(1, 2, 0)                  # [H, N, M]

        y[o_i] = panel[:, origen + 1: origen + 1 + H_FUT].T  # [H, N]

    return ens, y, scale, registros


# --------------------------------------------------------------------------
# Autotest: sobre el generador SINTETICO, nunca sobre el panel real
# --------------------------------------------------------------------------

def _panel_sintetico(T=300, seed=0):
    """Generador del cuaderno: causalidad Cu->FX->TPM PLANTADA."""
    rng = np.random.default_rng(seed)
    cu, fx, tpm = np.zeros(T), np.zeros(T), np.zeros(T)
    for t in range(1, T):
        cu[t] = .85 * cu[t - 1] + rng.normal(0, 1.0)
        fx[t] = .80 * fx[t - 1] - .50 * cu[t - 1] + rng.normal(0, .5)
        tpm[t] = .90 * tpm[t - 1] + .40 * fx[t - 1] + rng.normal(0, .3)
    return np.stack([3.5 + .4 * cu, 850. + 40. * fx, 5. + 2. * tpm], 0)


def _autotest():
    from h1_harness import test_h1, assert_shapes, numerical_guards

    print("=" * 76)
    print("AUTOTEST  -  panel SINTETICO (causalidad plantada), T=300")
    print("=" * 76)
    panel = _panel_sintetico(T=300)
    T0, NO, M = 180, 24, 60      # reducido para que el autotest corra rapido

    print(f"  T0={T0}  origenes={NO}  M={M}  (reducido; el real usa 144/109/200)\n")

    m, mu, sd, reg = entrenar_hasta(panel, T0, "B", 0, eta=5.0, steps=150)
    p0, p1 = reg["pasos"][0], reg["pasos"][-1]
    print(f"  ventanas de entrenamiento: {reg['n_ventanas']}")
    print(f"  L_simple  {p0['L_simple']:8.3f} -> {p1['L_simple']:8.3f}")
    print(f"  R_G       {p0['R_G']:8.5f} -> {p1['R_G']:8.5f}")
    print(f"  frac.estructural inicial: {100*p0['frac_estructural']:.1f}%")
    print(f"  masa prohibida: {reg['masa_prohibida']:.4f}   "
          f"entropia: {reg['entropia_normalizada']:.4f}")

    print("\n  evaluando A y B sobre los mismos origenes...")
    ens_a, y_a, sc_a, _ = evaluar_ventana_expansiva(
        panel, "A", 0, 0.0, T0=T0, n_origenes=NO, M=M, steps=150)
    ens_b, y_b, sc_b, _ = evaluar_ventana_expansiva(
        panel, "B", 0, 5.0, T0=T0, n_origenes=NO, M=M, steps=150)

    assert_shapes(ens_a, y_a, sc_a)
    numerical_guards(ens_a, ens_b, y_a, sc_a)
    assert np.allclose(y_a, y_b), "A y B deben compartir realizados"
    print(f"  formas OK: ens {ens_a.shape}  y {y_a.shape}  scale {sc_a.shape}")

    r = test_h1(ens_a, y_a, sc_a, ens_b, y_b, sc_b, standardize=True)
    print("\n  NOTA: 24 origenes y 150 pasos no bastan para concluir nada.")
    print("  El autotest verifica el cableado, no contrasta H1.")


if __name__ == "__main__":
    _autotest()
