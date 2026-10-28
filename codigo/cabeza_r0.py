"""
cabeza_r0.py -- Cabeza parametrica de respaldo R0 (Plan de contingencia v2, seccion 4).

Recibe z_enc in R^32 (el mismo resumen del encoder que usa la ScoreNet) y entrega
una distribucion conjunta sobre y in R^36 = 12 horizontes x 3 series.

  R0-G : y ~ N(mu, Sigma)
  R0-T : y ~ t_nu(mu, Sigma)   (Student-t multivariante; Cov = nu/(nu-2) Sigma)
  con Sigma = V V^T + diag(d), V in R^{36 x r}.

Interfaz unica para ambas variantes (la seleccion es un solo argumento):
  dist = cabeza(z_enc)            -> objeto con .log_prob(y), .rsample(S) / .sample(S)
  nll  = cabeza.nll(z_enc, y)     -> perdida de entrenamiento (escalar)
  Y    = cabeza.predecir(z_enc, S)-> tensor [S, 12, 3] por origen (B = 1) o [B, S, 12, 3]

Convencion de aplanado (debe coincidir con la ScoreNet y con el pipeline de datos):
  y[B, 12, 3] -> y.reshape(B, 36), orden horizonte-mayor: indice = 3*h + n,
  h = 0..11 (horizonte), n = 0..2 (Cu, USD/CLP, TPM).

Origen: propuesta del autor (Cabeza_RO.py), corregida en cuatro puntos:
  1. La variante Student-t no tenia log_prob, de modo que no podia entrenarse.
  2. Las dos variantes devolvian tipos distintos (distribucion vs tupla): el cambio
     de cabeza no era "un solo flag".
  3. El muestreo devolvia [B, 36, S]; la evaluacion espera [S, 12, 3].
  4. El muestreo t construia Sigma completa y su Cholesky; aqui se usa la
     estructura de factores, sin matrices de 36 x 36.
"""
import math
import torch
import torch.nn as nn
import torch.distributions as D

H_FUT, N_SER = 12, 3
D_Y = H_FUT * N_SER


# ---------------------------------------------------------------------------
# Student-t multivariante con escala de rango bajo
# ---------------------------------------------------------------------------
class LowRankMultivariateStudentT:
    """t_nu(mu, Sigma), Sigma = V V^T + diag(d).

    log p(y) = lgamma((nu+p)/2) - lgamma(nu/2) - (p/2) log(nu*pi)
               - (1/2) log|Sigma| - ((nu+p)/2) log(1 + delta/nu),
    delta = (y-mu)^T Sigma^{-1} (y-mu).

    Sigma^{-1} por Woodbury y log|Sigma| por el lema del determinante, ambos a
    traves de la matriz de capacitancia K = I_r + V^T D^{-1} V (r x r):
      delta      = a^T D^{-1} a - b^T K^{-1} b,   b = V^T D^{-1} a,  a = y - mu
      log|Sigma| = log|K| + sum log d
    """

    def __init__(self, loc, cov_factor, cov_diag, df):
        self.loc, self.V, self.d, self.df = loc, cov_factor, cov_diag, df
        self.p = loc.shape[-1]
        r = cov_factor.shape[-1]
        Vt_Dinv = cov_factor.transpose(-1, -2) / cov_diag.unsqueeze(-2)          # [B, r, p]
        K = torch.eye(r, dtype=loc.dtype, device=loc.device) + Vt_Dinv @ cov_factor
        self._Vt_Dinv = Vt_Dinv
        self._LK = torch.linalg.cholesky(K)                                    # [B, r, r]
        self._logdet = 2 * torch.log(torch.diagonal(self._LK, dim1=-2, dim2=-1)).sum(-1) \
            + torch.log(cov_diag).sum(-1)

    def mahalanobis(self, y):
        a = y - self.loc                                                       # [..., B, p]
        q1 = (a * a / self.d).sum(-1)
        b = (self._Vt_Dinv @ a.unsqueeze(-1))                                  # [..., B, r, 1]
        # b^T K^{-1} b = ||L_K^{-1} b||^2. Se usa solve_triangular y no cholesky_solve:
        # algunas compilaciones de torch para CUDA no traen el nucleo de cholesky_solve
        # (CUDA error: no kernel image is available, Colab T4, 06-10-2026).
        c = torch.linalg.solve_triangular(self._LK.expand(b.shape[:-1] + (b.shape[-2],)), b, upper=False)
        q2 = (c * c).sum((-1, -2))
        return q1 - q2

    def log_prob(self, y):
        nu, p = self.df, self.p
        delta = self.mahalanobis(y)
        return (torch.lgamma((nu + p) / 2) - torch.lgamma(nu / 2)
                - 0.5 * p * torch.log(nu * math.pi) - 0.5 * self._logdet
                - 0.5 * (nu + p) * torch.log1p(delta / nu))

    def sample(self, sample_shape=torch.Size()):
        """y = mu + (V g_r + sqrt(d) * g_p) / sqrt(w / nu),  w ~ chi2(nu).
        V g_r + sqrt(d) g_p ~ N(0, Sigma) sin factorizar Sigma."""
        S = torch.Size(sample_shape)
        forma = S + self.loc.shape
        g_r = torch.randn(S + self.V.shape[:-2] + (self.V.shape[-1],),
                          dtype=self.loc.dtype, device=self.loc.device)
        g_p = torch.randn(forma, dtype=self.loc.dtype, device=self.loc.device)
        normal = torch.einsum("...pr,...r->...p", self.V, g_r) + self.d.sqrt() * g_p
        w = D.Chi2(self.df).sample(S)                                          # [S..., B]
        return self.loc + normal / (w / self.df).sqrt().unsqueeze(-1)

    @property
    def covariance_matrix(self):
        Sigma = self.V @ self.V.transpose(-1, -2) + torch.diag_embed(self.d)
        return (self.df / (self.df - 2)).unsqueeze(-1).unsqueeze(-1) * Sigma


# ---------------------------------------------------------------------------
# Cabeza R0
# ---------------------------------------------------------------------------
class CabezaR0(nn.Module):
    """dist_type: 'gaussian' (R0-G) o 'student-t' (R0-T).
    rank: r de Sigma = V V^T + diag(d).
    v_condicional: si False, V es un parametro global (no depende de z_enc);
        reduce 128*36*r parametros, util con ~118 ventanas de entrenamiento.
    nu_min: cota inferior de nu; con 2 la covarianza existe, con 4 tambien la curtosis.
    """

    def __init__(self, latent_dim=32, output_dim=D_Y, rank=5, dist_type="gaussian",
                 v_condicional=True, nu_min=2.0, d_min=1e-4, hidden=128):
        super().__init__()
        if dist_type not in ("gaussian", "student-t"):
            raise ValueError(f"dist_type desconocido: {dist_type}")
        self.dist_type, self.output_dim, self.rank = dist_type, output_dim, rank
        self.v_condicional, self.nu_min, self.d_min = v_condicional, nu_min, d_min

        self.shared_net = nn.Sequential(nn.Linear(latent_dim, 64), nn.GELU(),
                                        nn.Linear(64, hidden), nn.GELU())
        self.mu_head = nn.Linear(hidden, output_dim)
        self.d_head = nn.Linear(hidden, output_dim)
        if v_condicional:
            self.v_head = nn.Linear(hidden, output_dim * rank)
            nn.init.normal_(self.v_head.weight, std=0.01)       # V pequeno al inicio:
            nn.init.normal_(self.v_head.bias, std=0.1)          # Sigma ~ diag(d) + perturbacion
        else:
            self.V_global = nn.Parameter(0.1 * torch.randn(output_dim, rank))
        if dist_type == "student-t":
            self.nu_head = nn.Linear(hidden, 1)
            nn.init.zeros_(self.nu_head.weight)
            nn.init.constant_(self.nu_head.bias, 2.0)           # nu inicial ~ nu_min + 2.1

    def parametros(self, z_enc):
        h = self.shared_net(z_enc)
        mu = self.mu_head(h)
        d = nn.functional.softplus(self.d_head(h)) + self.d_min
        if self.v_condicional:
            V = self.v_head(h).view(-1, self.output_dim, self.rank)
        else:
            V = self.V_global.expand(z_enc.shape[0], -1, -1)
        nu = None
        if self.dist_type == "student-t":
            nu = nn.functional.softplus(self.nu_head(h)).squeeze(-1) + self.nu_min
        return mu, V, d, nu

    def forward(self, z_enc):
        mu, V, d, nu = self.parametros(z_enc)
        if self.dist_type == "gaussian":
            return D.LowRankMultivariateNormal(loc=mu, cov_factor=V, cov_diag=d)
        return LowRankMultivariateStudentT(mu, V, d, nu)

    def nll(self, z_enc, y):
        """Perdida de entrenamiento: -log p(y | z_enc), promedio del lote.
        y: [B, 12, 3] o [B, 36] en la convencion horizonte-mayor."""
        return -self(z_enc).log_prob(y.reshape(y.shape[0], -1)).mean()

    @torch.no_grad()
    def predecir(self, z_enc, S=200):
        """Trayectorias en la forma que espera el modulo de evaluacion.
        B = 1 -> [S, 12, 3];  B > 1 -> [B, S, 12, 3]."""
        y = self(z_enc).sample((S,))                                           # [S, B, 36]
        y = y.reshape(S, -1, H_FUT, N_SER).transpose(0, 1)                     # [B, S, 12, 3]
        return y[0] if y.shape[0] == 1 else y


# Nombre de la propuesta original, conservado por compatibilidad
RespaldoParametricoRO = CabezaR0


def contar_parametros(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
