"""
R_G con distancia de Frobenius  -  modulo listo para enchufar en la Version B
=============================================================================

AISLAMIENTO: el autotest usa solo tensores dummy. El modulo no lee datos.

QUE CAMBIA RESPECTO DE LA IMPLEMENTACION ACTUAL
-----------------------------------------------
1. D(.,.) pasa de KL(W||Pi) suavizada con eps a  ||W_agg - Pi(G)||_F^2.
   Motivo: Pi(G) tiene 4 ceros de 9 entradas; KL(W||Pi) no es finita sin
   suavizar, y el valor suavizado depende de eps (x3.7 entre 1e-3 y 1e-9),
   de modo que eta no tiene escala interpretable. Frobenius no necesita eps,
   su gradiente es lineal y acotado, y separa atencion colapsada de
   estructurada por un factor ~550 contra ~34 de las variantes de KL.

2. La proyeccion T x T -> N x N queda explicita y auditada:
   suma sobre CLAVES, promedio sobre CONSULTAS. Es la unica de las tres
   reglas razonables que deja filas que suman 1, como las de Pi(G).

3. eta se CALIBRA para que eta*R_G aporte una fraccion objetivo de L_total
   al inicio del entrenamiento, en vez de fijarse a mano.

ADVERTENCIA DE COMPARABILIDAD
-----------------------------
Una Version B reentrenada con este modulo NO es comparable con los resultados
previos (KL, eta=5). Si se adopta, el contraste de H1 debe rehacerse completo
y reportarse asi. Los umbrales de H1 NO se tocan.
"""

from __future__ import annotations
import torch
import torch.nn as nn


# --------------------------------------------------------------------------

def causal_mask(T: int, L: int, device=None) -> torch.Tensor:
    """A_{i,j} = 0 si (j mod L) <= (i mod L), si no -inf."""
    idx = torch.arange(T, device=device)
    lag = idx % L
    return torch.where(lag[None, :] <= lag[:, None], 0.0, float("-inf"))


def aggregate_blocks(W: torch.Tensor, L: int, N: int) -> torch.Tensor:
    """
    Proyecta (..., T, T) -> (..., N, N) con T = N*L.

    suma sobre claves   : cada fila de W suma 1 sobre las T claves; sumarlas
                          dentro del bloque de cada serie deja filas que suman
                          1 sobre las N series, igual que Pi(G).
    promedio sobre consultas : las L consultas de una serie son repeticiones
                          del mismo rol estructural; sumarlas multiplica por L.
    """
    *lead, Ti, Tj = W.shape
    if Ti != N * L or Tj != N * L:
        raise ValueError(f"se esperaba (...,{N*L},{N*L}); recibido {(Ti, Tj)}")
    Wb = W.reshape(*lead, N, L, N, L)
    return Wb.sum(dim=-1).mean(dim=-2)


# --------------------------------------------------------------------------

class RegularizadorEstructural(nn.Module):
    """
    R_G = || W_agg - Pi(G) ||_F^2  promediado por fila.

    Uso en el bucle de entrenamiento:

        reg = RegularizadorEstructural(PI_G, N=3, L=15)
        ...
        loss_simple = difusion(...)            # o cabeza gaussiana / Student-t
        rg, diag = reg(attn_weights)           # lista por capa, o un tensor
        loss = loss_simple + reg.eta * rg
        loss.backward()

    `attn_weights` debe venir del grafo de computo (requires_grad), NO de un
    buffer guardado para logging: regularizar un tensor desprendido deja R_G
    sin gradiente y la Version B entrena igual que la A sin dar error.
    """

    def __init__(self, pi_g: torch.Tensor, N: int, L: int,
                 eta: float = 1.0, layer_reduce: str = "mean"):
        super().__init__()
        pi_g = torch.as_tensor(pi_g, dtype=torch.float32)
        if pi_g.shape != (N, N):
            raise ValueError(f"Pi(G) debe ser ({N},{N}); recibido {tuple(pi_g.shape)}")
        if not torch.allclose(pi_g.sum(-1), torch.ones(N), atol=1e-6):
            raise ValueError(f"las filas de Pi(G) deben sumar 1; suman {pi_g.sum(-1)}")
        self.register_buffer("pi_g", pi_g)
        self.N, self.L = N, L
        self.eta = float(eta)
        if layer_reduce not in ("mean", "last"):
            raise ValueError("layer_reduce debe ser 'mean' o 'last'")
        self.layer_reduce = layer_reduce

    # ----------------------------------------------------------------

    def _collapse(self, attn):
        """
        attn: tensor (B,H,T,T) o lista de tensores por capa.

        layer_reduce='mean' promedia las capas DESPUES de proyectar cada una.
        Proyectar primero y promediar despues no es lo mismo que promediar los
        mapas crudos y proyectar una vez, porque la proyeccion es lineal pero
        el softmax que la precede no lo es; promediar mapas de capas distintas
        mezcla distribuciones sobre soportes distintos.

        layer_reduce='last' regulariza solo la ultima capa. Es la alternativa
        defendible si se quiere que las capas tempranas queden libres.
        """
        if torch.is_tensor(attn):
            attn = [attn]
        aggs = []
        for W in attn:
            if W.dim() != 4:
                raise ValueError(f"cada mapa debe ser (B,H,T,T); recibido {tuple(W.shape)}")
            rows = W.sum(-1)
            if not torch.allclose(rows, torch.ones_like(rows), atol=1e-4):
                raise ValueError(
                    f"las filas de la atencion no suman 1 (media {rows.mean():.6f}); "
                    "se esperan pesos post-softmax, no logits")
            aggs.append(aggregate_blocks(W, self.L, self.N).mean(dim=(0, 1)))
        return aggs[-1] if self.layer_reduce == "last" else torch.stack(aggs).mean(0)

    def forward(self, attn):
        W_agg = self._collapse(attn)
        rg = ((W_agg - self.pi_g) ** 2).sum(-1).mean()

        if self.training and not rg.requires_grad:
            raise RuntimeError(
                "R_G no tiene gradiente: los mapas de atencion llegan desprendidos "
                "del grafo. Entregue los pesos del forward, no un buffer de logging.")

        diag = {"W_agg": W_agg.detach(),
                "desvio_max": (W_agg - self.pi_g).abs().max().detach(),
                "masa_prohibida": W_agg[self.pi_g == 0].sum().detach()}
        return rg, diag

    # ----------------------------------------------------------------

    @torch.no_grad()
    def calibrar_eta(self, attn, loss_simple, fraccion=0.20) -> float:
        """
        Fija eta tal que  eta*R_G / (L_simple + eta*R_G) = fraccion
        en el PRIMER lote, es decir:

            eta = fraccion * L_simple / ((1 - fraccion) * R_G)

        fraccion=0.20 deja el termino estructural como una contribucion
        sustantiva pero no dominante. Con KL y eta=5 esa fraccion era 0.964:
        el modelo optimizaba la forma de la atencion, no el pronostico.

        Se llama UNA vez, antes de entrenar, y el eta resultante queda fijo:
        recalibrarlo durante el entrenamiento haria que la perdida cambie de
        definicion sobre la marcha.
        """
        if not 0.0 < fraccion < 1.0:
            raise ValueError("fraccion debe estar en (0,1)")
        rg = ((self._collapse(attn) - self.pi_g) ** 2).sum(-1).mean().item()
        if rg < 1e-12:
            raise ValueError("R_G ~ 0 en el primer lote: nada que calibrar")
        ls = float(loss_simple)
        self.eta = fraccion * ls / ((1.0 - fraccion) * rg)
        return self.eta


# --------------------------------------------------------------------------

@torch.no_grad()
def entropia_normalizada(W: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Entropia de cada fila / log(n_validas DE ESA FILA).

    Normalizar por el numero de claves validas es obligatorio: la mascara
    causal deja entre 3 y N*L claves permitidas segun la fila. Dividir por
    log(T) fijo subestima la entropia de las filas tempranas y hace parecer
    menos uniforme un colapso real.
    """
    n_valid = torch.isfinite(mask).sum(-1).clamp_min(2).float()
    return (-(W * W.clamp_min(1e-12).log()).sum(-1) / n_valid.log())


# --------------------------------------------------------------------------

def _autotest():
    import torch.nn.functional as F
    torch.manual_seed(0)
    N, L, H, B = 3, 15, 4, 8
    T, NL = N * L, 3
    PI = torch.tensor([[1.0, 0.0, 0.0], [0.5, 0.5, 0.0], [0.0, 0.5, 0.5]])
    mask = causal_mask(T, L)
    reg = RegularizadorEstructural(PI, N, L)

    def mapas(kind, seed):
        g = torch.Generator().manual_seed(seed)
        out = []
        for _ in range(NL):
            if kind == "estructurada":
                v = torch.arange(T) // L
                bias = PI[v][:, v]
                lg = torch.randn(B, H, T, T, generator=g) * 0.1 + bias * 6.0
            else:
                lg = torch.randn(B, H, T, T, generator=g) * 0.01
            lg.requires_grad_(True)
            out.append(F.softmax(lg + mask, dim=-1))
        return out

    print("=" * 72)
    print("AUTOTEST  -  R_G Frobenius")
    print("=" * 72)

    for kind in ("colapsada", "estructurada"):
        rg, d = reg(mapas(kind, 1))
        print(f"\n  {kind}")
        print(f"    R_G              = {rg.item():.6f}")
        print(f"    desvio maximo    = {d['desvio_max'].item():.4f}")
        print(f"    masa en posiciones prohibidas = {d['masa_prohibida'].item():.4f}")
        print(f"    W_agg =\n{d['W_agg'].numpy().round(4)}")

    print("\n" + "-" * 72)
    print("  calibracion de eta  (L_simple = 1.0)")
    print("-" * 72)
    a = mapas("colapsada", 1)
    for f in (0.10, 0.20, 0.30):
        e = reg.calibrar_eta(a, 1.0, f)
        rg, _ = reg(a)
        c = e * rg.item()
        print(f"    fraccion={f:.2f} -> eta={e:7.4f}   eta*R_G={c:7.4f}   "
              f"{100*c/(1.0+c):5.1f}% de L_total")

    print("\n" + "-" * 72)
    print("  guardas")
    print("-" * 72)
    try:
        reg(torch.randn(B, H, T, T))
    except ValueError as e:
        print(f"    logits en vez de pesos      -> {str(e)[:58]}...")
    try:
        reg(mapas("colapsada", 1)[0].detach())
    except RuntimeError as e:
        print(f"    atencion desprendida        -> {str(e)[:58]}...")
    try:
        RegularizadorEstructural(torch.tensor([[1., 1.], [0., 1.]]), 2, L)
    except ValueError as e:
        print(f"    Pi(G) con filas != 1        -> {str(e)[:58]}...")

    print("\n" + "-" * 72)
    print("  gradiente y DCA")
    print("-" * 72)
    lg = []
    g = torch.Generator().manual_seed(1)
    for _ in range(NL):
        x = (torch.randn(B, H, T, T, generator=g) * 0.01).requires_grad_(True)
        lg.append(x)
    rg, _ = reg([F.softmax(x + mask, dim=-1) for x in lg])
    rg.backward()
    gs = torch.cat([x.grad.flatten() for x in lg])
    print(f"    R_G                   = {rg.item():.6f}")
    print(f"    gradientes finitos    = {torch.isfinite(gs).all().item()}")
    print(f"    |grad|_max            = {gs.abs().max().item():.3e}")
    print(f"    |grad| no nulo        = {(gs.abs() > 0).any().item()}")
    print(f"    entropia normalizada colapsada    = "
          f"{entropia_normalizada(mapas('colapsada', 1)[0], mask).mean():.4f}")
    print(f"    entropia normalizada estructurada = "
          f"{entropia_normalizada(mapas('estructurada', 1)[0], mask).mean():.4f}")


if __name__ == "__main__":
    _autotest()
