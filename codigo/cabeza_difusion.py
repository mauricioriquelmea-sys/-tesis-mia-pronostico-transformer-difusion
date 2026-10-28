"""
cabeza_difusion.py -- Cabeza generativa de difusion con la interfaz de R0
==========================================================================

OBJETO. Entregar una cabeza de difusion que el modulo de evaluacion acepte
sin modificaciones, exponiendo EXACTAMENTE los dos metodos de CabezaR0:

    nll(z_enc, y)              # y: [B, 12, 3]  ->  escalar
    predecir(z_enc, S=200)     #                ->  [S, 12, 3]   (B = 1)
                               #                ->  [B, S, 12, 3] (B > 1)

Con eso, `bloques_h1.py` cambia una sola linea (la construccion de
`self.cabeza` en EncoderR0) y H1 y H2 corren sin trabajo adicional.

AISLAMIENTO. Este modulo no contiene ni carga datos. Opera sobre tensores
que recibe; el autotest usa `torch.randn` y el generador sintetico.

----------------------------------------------------------------------------
ADVERTENCIA SOBRE `nll`  --  LEER ANTES DE TABULAR CUALQUIER VALOR
----------------------------------------------------------------------------
Un modelo de difusion NO entrega log p(y) exacta. `nll` devuelve aqui la
perdida de denoising (MSE sobre epsilon), que es un **proxy** del limite
variacional con pesos distintos de los de la ELBO. Por tanto:

  * sirve para ENTRENAR (es lo que se minimiza),
  * NO es comparable en valor con la NLL exacta de CabezaR0,
  * NO debe aparecer en una tabla junto a la NLL de R0.

El atributo `nll_es_exacta = False` existe para que el codigo de evaluacion
pueda negarse a tabularla. Las unicas metricas comparables entre cabezas son
las del prerregistro: CRPS, Energy Score y Diebold-Mariano. Esta es la misma
limitacion que §5.2.1 del informe senala para el proxy NLL.

----------------------------------------------------------------------------
DOS ESPACIOS DE DIFUSION  (es lo que decide C3)
----------------------------------------------------------------------------
espacio="y"        Difusion directa sobre la ventana futura y in R^36.
                   Corresponde al escalon E2 del plan de contingencia. No
                   hay autoencoder, de modo que no hay error de
                   reconstruccion que acote por abajo el CRPS.

espacio="latente"  Difusion sobre un latente de y (d_lat dimensiones) con
                   autoencoder deterministico, condicionada por z_enc.
                   Es la arquitectura descrita en §3.2 del informe.
                   Dos decisiones que deben quedar explicitas porque son
                   justamente donde este diseno se rompe en silencio:
                     (a) el objetivo de difusion es enc_y(y).detach(): el
                         gradiente de difusion NO entra al autoencoder, lo
                         que evita el colapso del latente a una escala que
                         minimice la MSE de epsilon;
                     (b) el latente se estandariza con media y desviacion
                         EMA (buffers). El schedule de ruido supone datos de
                         varianza ~1; sin esa estandarizacion, alpha_barra
                         deja de significar lo que la formula supone y el
                         muestreo degrada sin lanzar ningun error.

----------------------------------------------------------------------------
CONVENCION DE APLANADO
----------------------------------------------------------------------------
Identica a cabeza_r0: y[B,12,3] -> y.reshape(B,36), indice = 3*h + n,
h = 0..11 (horizonte), n = 0..2 (Cu, USD/CLP, TPM).
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn

H_FUT, N_SER = 12, 3
D_Y = H_FUT * N_SER                      # 36


# ===========================================================================
# Schedule de ruido
# ===========================================================================
def schedule_coseno(T: int, s: float = 0.008, beta_max: float = 0.999,
                    dtype=torch.float32):
    """Nichol & Dhariwal (2021).  f(t) = cos((t/T + s)/(1+s) * pi/2)^2,
    alpha_barra_t = f(t)/f(0),  beta_t = 1 - ab_t/ab_{t-1}  recortado a
    beta_max, y **alpha_barra RECOMPUTADA como cumprod(1 - beta)**.

    Devuelve (ab, beta, alpha) con ab de T+1 entradas y ab[0] = 1 exacto.
    El indice 0 reservado no es cosmetica: hace que el ultimo paso del
    muestreo aterrice sobre y_0 sin residuo de ruido.

    POR QUE SE RECOMPUTA  (falla detectada por C1, 12-10-2026)
    -----------------------------------------------------------
    En t = T el coseno vale cos(pi/2)^2 = 0, de modo que el schedule crudo
    da ab_T = 3.8e-33 y la inversa y_0 = (y_t - sqrt(1-ab_t) eps)/sqrt(ab_t)
    divide por 6e-17: numero de condicion 1.6e16, y ninguna aritmetica de
    punto flotante sobrevive a eso. El recorte de beta a 0.999 (que Nichol &
    Dhariwal aplican y que aqui faltaba propagar) deja ab_T = 2.4e-7 para
    T = 100, con condicion 2.0e3. Si se recorta beta pero se conserva la ab
    cruda, las dos quedan INCONSISTENTES: los coeficientes del posterior
    mezclan dos schedules distintos y la cadena no regresa a y_0 ni con un
    oraculo perfecto (error relativo 6.4e-1 medido en C1.4). El recorte toca
    un solo paso de T, asi que el schedule sigue siendo el del coseno.
    """
    t = torch.arange(T + 1, dtype=torch.float64)
    f = torch.cos(((t / T + s) / (1.0 + s)) * math.pi / 2.0) ** 2
    ab_crudo = f / f[0]
    ab_crudo[0] = 1.0
    beta = (1.0 - ab_crudo[1:] / ab_crudo[:-1]).clamp(1e-8, beta_max)
    alpha = 1.0 - beta
    ab = torch.cat([torch.ones(1, dtype=torch.float64), alpha.cumprod(0)])
    return ab.to(dtype), beta.to(dtype), alpha.to(dtype)


class Difusor(nn.Module):
    """Guarda el schedule y la aritmetica de q y p. Sin red neuronal dentro:
    la red se le pasa como argumento, para poder sustituirla por un ORACULO
    en C1."""

    def __init__(self, T: int = 100, s: float = 0.008, beta_max: float = 0.999,
                 clip_y0: float | None = 8.0):
        super().__init__()
        self.T, self.clip_y0 = T, clip_y0
        ab, beta, alpha = schedule_coseno(T, s, beta_max, dtype=torch.float64)
        self.register_buffer("ab", ab.float())                   # [T+1]
        self.register_buffer("beta", beta.float())               # [T]
        self.register_buffer("alpha", alpha.float())             # [T]

    @property
    def condicion(self) -> float:
        """max_t 1/sqrt(ab_t): cuanto amplifica la inversa y0_desde_eps el
        error de eps_hat. Se reporta; no se oculta."""
        return float((1.0 / self.ab[1:].sqrt()).max())

    # -- q(y_t | y_0) -------------------------------------------------------
    def ruidar(self, y0, t, eps):
        """y_t = sqrt(ab_t) y_0 + sqrt(1 - ab_t) eps.   t: [B] en 1..T."""
        a = self.ab[t].sqrt().unsqueeze(-1)
        b = (1.0 - self.ab[t]).clamp_min(0.0).sqrt().unsqueeze(-1)
        return a * y0 + b * eps

    def y0_desde_eps(self, yt, t, eps):
        """Inversa algebraica: y_0 = (y_t - sqrt(1-ab_t) eps) / sqrt(ab_t)."""
        a = self.ab[t].sqrt().unsqueeze(-1)
        b = (1.0 - self.ab[t]).clamp_min(0.0).sqrt().unsqueeze(-1)
        return (yt - b * eps) / a

    def eps_desde_y0(self, yt, t, y0):
        a = self.ab[t].sqrt().unsqueeze(-1)
        b = (1.0 - self.ab[t]).clamp_min(0.0).sqrt().unsqueeze(-1)
        return (yt - a * y0) / b

    def _y0_recortado(self, yt, t, eps_hat, recortar=True):
        """y_0_hat con recorte. El recorte NO es cosmetico: en t = T la
        inversa amplifica el error de eps_hat por 1/sqrt(ab_T) ~ 2e3, de modo
        que un eps_hat imperfecto produce un y_0_hat de magnitud absurda que
        luego el coeficiente del posterior vuelve a multiplicar por ~0. El
        producto 2e3 * 5e-4 es formalmente inocuo pero en fp32 pierde todos
        los digitos. Con datos estandarizados, |y_0| <= 8 es holgado."""
        y0 = self.y0_desde_eps(yt, t, eps_hat)
        if recortar and self.clip_y0 is not None:
            y0 = y0.clamp(-self.clip_y0, self.clip_y0)
        return y0

    # -- p(y_{t-1} | y_t) ---------------------------------------------------
    def paso_ddpm(self, yt, t, eps_hat, generador=None, recortar=True):
        """Media exacta de q(y_{t-1} | y_t, y_0_hat) + ruido beta_tilde."""
        y0 = self._y0_recortado(yt, t, eps_hat, recortar)
        ab_t = self.ab[t].unsqueeze(-1)
        ab_p = self.ab[t - 1].unsqueeze(-1)
        bt = self.beta[t - 1].unsqueeze(-1)
        al = self.alpha[t - 1].unsqueeze(-1)
        um = (1.0 - ab_t).clamp_min(1e-12)
        mu = (ab_p.sqrt() * bt / um) * y0 + (al.sqrt() * (1.0 - ab_p) / um) * yt
        var = bt * (1.0 - ab_p) / um
        z = torch.randn(yt.shape, dtype=yt.dtype, device=yt.device,
                        generator=generador)
        # en t = 1 la varianza posterior es 0: se devuelve la media
        activo = (t > 1).float().unsqueeze(-1)
        return mu + activo * var.clamp_min(0.0).sqrt() * z, y0

    def paso_ddim(self, yt, t, eps_hat, t_prev=None, recortar=True):
        """Variante deterministica (eta = 0). La usa C1: con un oraculo que
        devuelve el epsilon verdadero, la cadena DEBE regresar a y_0."""
        y0 = self._y0_recortado(yt, t, eps_hat, recortar)
        tp = (t - 1) if t_prev is None else t_prev
        ab_p = self.ab[tp].unsqueeze(-1)
        return ab_p.sqrt() * y0 + (1.0 - ab_p).clamp_min(0.0).sqrt() * eps_hat, y0


# ===========================================================================
# ScoreNet: epsilon_hat(y_t, t, z_enc)
# ===========================================================================
def emb_sinusoidal(t, d):
    """t: [B] entero -> [B, d]."""
    half = d // 2
    frec = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device,
                                                       dtype=torch.float32) / half)
    arg = t.float().unsqueeze(-1) * frec.unsqueeze(0)
    return torch.cat([arg.sin(), arg.cos()], dim=-1)


class BloqueResidual(nn.Module):
    """h <- h + MLP(LN(h) + t_emb + c_emb).  Conexion residual y
    normalizacion previas: con 3 bloques y ~120 ventanas, sin residual la
    senal del condicionante se pierde antes de la salida."""

    def __init__(self, hidden):
        super().__init__()
        self.ln = nn.LayerNorm(hidden)
        self.mlp = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(),
                                 nn.Linear(hidden, hidden))

    def forward(self, h, cond):
        return h + self.mlp(self.ln(h) + cond)


class ScoreNet(nn.Module):
    def __init__(self, d_x, d_cond=32, hidden=96, n_bloques=2, d_t=32):
        super().__init__()
        self.d_t = d_t
        self.ent = nn.Linear(d_x, hidden)
        self.t_mlp = nn.Sequential(nn.Linear(d_t, hidden), nn.GELU(),
                                   nn.Linear(hidden, hidden))
        self.c_mlp = nn.Sequential(nn.Linear(d_cond, hidden), nn.GELU(),
                                   nn.Linear(hidden, hidden))
        self.bloques = nn.ModuleList([BloqueResidual(hidden)
                                      for _ in range(n_bloques)])
        self.ln_out = nn.LayerNorm(hidden)
        self.sal = nn.Linear(hidden, d_x)
        # salida a cero: el modelo arranca prediciendo eps_hat = 0, de modo
        # que la perdida inicial es E||eps||^2 = d_x y no un valor arbitrario
        nn.init.zeros_(self.sal.weight)
        nn.init.zeros_(self.sal.bias)

    def forward(self, x_t, t, z):
        cond = self.t_mlp(emb_sinusoidal(t, self.d_t)) + self.c_mlp(z)
        h = self.ent(x_t)
        for b in self.bloques:
            h = b(h, cond)
        return self.sal(self.ln_out(h))


# ===========================================================================
# Cabeza de difusion
# ===========================================================================
class CabezaDifusion(nn.Module):
    """Interfaz identica a CabezaR0.

    espacio : "y" (E2, difusion sobre R^36) o "latente" (§3.2 del informe).
    T_dif   : pasos de difusion del entrenamiento.
    pasos_muestreo : subsecuencia usada al muestrear. None = los T_dif pasos.
    """

    nll_es_exacta = False        # bandera para el modulo de evaluacion

    def __init__(self, latent_dim=32, output_dim=D_Y, espacio="y",
                 T_dif=100, hidden=96, n_bloques=2, d_lat=8,
                 lambda_rec=1.0, ema=0.99, pasos_muestreo=None,
                 muestreador="ddpm"):
        super().__init__()
        if espacio not in ("y", "latente"):
            raise ValueError(f"espacio desconocido: {espacio}")
        if muestreador not in ("ddpm", "ddim"):
            raise ValueError(f"muestreador desconocido: {muestreador}")
        self.espacio, self.output_dim = espacio, output_dim
        self.lambda_rec, self.ema = lambda_rec, ema
        self.pasos_muestreo, self.muestreador = pasos_muestreo, muestreador

        self.dif = Difusor(T=T_dif)
        d_x = output_dim if espacio == "y" else d_lat
        self.d_x = d_x
        self.red = ScoreNet(d_x, d_cond=latent_dim, hidden=hidden,
                            n_bloques=n_bloques)

        if espacio == "latente":
            self.enc_y = nn.Sequential(nn.Linear(output_dim, 64), nn.GELU(),
                                       nn.Linear(64, d_lat))
            self.dec_y = nn.Sequential(nn.Linear(d_lat, 64), nn.GELU(),
                                       nn.Linear(64, output_dim))
            self.register_buffer("lat_mu", torch.zeros(d_lat))
            self.register_buffer("lat_sd", torch.ones(d_lat))
            self.register_buffer("ema_listo", torch.zeros(()))

    # ---------------------------------------------------------------- utils
    @staticmethod
    def _aplanar(y):
        return y.reshape(y.shape[0], -1) if y.dim() > 2 else y

    def _normalizar_lat(self, u, actualizar):
        if actualizar and self.training:
            with torch.no_grad():
                mu, sd = u.mean(0), u.std(0).clamp_min(1e-4)
                if self.ema_listo.item() == 0:
                    self.lat_mu.copy_(mu)
                    self.lat_sd.copy_(sd)
                    self.ema_listo.fill_(1.0)
                else:
                    self.lat_mu.mul_(self.ema).add_((1 - self.ema) * mu)
                    self.lat_sd.mul_(self.ema).add_((1 - self.ema) * sd)
        return (u - self.lat_mu) / self.lat_sd

    def _desnormalizar_lat(self, x):
        return x * self.lat_sd + self.lat_mu

    # ----------------------------------------------------------------- nll
    def nll(self, z_enc, y):
        """PROXY de entrenamiento, no NLL exacta. Ver cabecera del modulo.

        espacio="y":        E_t,eps || eps - eps_hat(y_t, t, z) ||^2
        espacio="latente":  lo anterior sobre enc_y(y).detach(), mas
                            lambda_rec * MSE(dec_y(enc_y(y)), y)
        """
        y = self._aplanar(y)
        B = y.shape[0]

        if self.espacio == "y":
            x0 = y
            rec = torch.zeros((), dtype=y.dtype, device=y.device)
        else:
            u = self.enc_y(y)
            y_rec = self.dec_y(u)
            rec = ((y_rec - y) ** 2).mean()
            x0 = self._normalizar_lat(u.detach(), actualizar=True)

        t = torch.randint(1, self.dif.T + 1, (B,), device=y.device)
        eps = torch.randn_like(x0)
        xt = self.dif.ruidar(x0, t, eps)
        eps_hat = self.red(xt, t, z_enc)
        perdida_dif = ((eps - eps_hat) ** 2).mean()
        return perdida_dif + self.lambda_rec * rec

    # ------------------------------------------------------------ muestreo
    def _secuencia(self):
        T = self.dif.T
        if self.pasos_muestreo is None or self.pasos_muestreo >= T:
            return list(range(T, 0, -1))
        k = int(self.pasos_muestreo)
        idx = torch.linspace(T, 1, k).round().long().tolist()
        vistos, sec = set(), []
        for i in idx:                       # sin repetidos, orden decreciente
            if i not in vistos:
                vistos.add(i)
                sec.append(i)
        return sec

    @torch.no_grad()
    def _muestrear_x(self, z_cond, generador=None):
        """z_cond: [M, d_cond] -> x_0 [M, d_x]."""
        M = z_cond.shape[0]
        x = torch.randn((M, self.d_x), dtype=z_cond.dtype,
                        device=z_cond.device, generator=generador)
        sec = self._secuencia()
        for j, ti in enumerate(sec):
            t = torch.full((M,), ti, dtype=torch.long, device=z_cond.device)
            eps_hat = self.red(x, t, z_cond)
            if self.muestreador == "ddim":
                tp = sec[j + 1] if j + 1 < len(sec) else 0
                t_prev = torch.full((M,), tp, dtype=torch.long,
                                    device=z_cond.device)
                x, _ = self.dif.paso_ddim(x, t, eps_hat, t_prev=t_prev)
            else:
                x, _ = self.dif.paso_ddpm(x, t, eps_hat, generador=generador)
        return x

    @torch.no_grad()
    def predecir(self, z_enc, S=200):
        """[S, 12, 3] si B = 1;  [B, S, 12, 3] si B > 1.
        Misma firma y misma forma que CabezaR0.predecir."""
        B = z_enc.shape[0]
        z_rep = z_enc.repeat_interleave(S, dim=0)              # [B*S, d_cond]
        x = self._muestrear_x(z_rep)
        y = x if self.espacio == "y" else self.dec_y(self._desnormalizar_lat(x))
        y = y.reshape(B, S, H_FUT, N_SER)
        return y[0] if B == 1 else y

    # ------------------------------------------------------- compatibilidad
    def forward(self, z_enc):
        raise NotImplementedError(
            "La difusion no expone una distribucion con log_prob. Usar "
            "nll() para entrenar y predecir() para muestrear.")


def contar_parametros(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
