"""
Banco de pruebas abstracto para el contraste de H1  (Version A  vs  Version B / R_G)
====================================================================================

AISLAMIENTO: este modulo NO lee datos reales. Todo corre sobre tensores dummy
generados con torch.randn / numpy.random. Las funciones son agnosticas al origen
de los datos: para usarlas con el panel real basta reemplazar `make_dummy_run`
por el volcado del pipeline, respetando las formas auditadas abajo.

ORDEN DE AUDITORIA (segun la disciplina de memoria de calculo):
    1. dimensiones   -> assert_shapes()
    2. magnitudes    -> report_magnitudes()
    3. estabilidad   -> numerical_guards()

CONVENCION DE FORMAS
--------------------
    ens      (O, H, N, M)   ensemble predictivo
    y        (O, H, N)      realizado
    scale    (O, N)         escala por origen y serie (sigma de los cambios,
                            estimada SOLO con informacion hasta el origen o)

    O = numero de origenes de pronostico   (protocolo: 109)
    H = horizontes                         (protocolo: 12)
    N = series                             (cobre, USD/CLP, TPM -> 3)
    M = tamano del ensemble
"""

from __future__ import annotations

import numpy as np

# --------------------------------------------------------------------------
# 0. Umbrales preregistrados.  NO SE MUEVEN SEGUN EL RESULTADO.
# --------------------------------------------------------------------------

H1_MIN_REDUCTION = 0.05   # R_G reduce el CRPS conjunto en al menos 5 %
H1_ALPHA         = 0.05   # significancia Diebold-Mariano


# --------------------------------------------------------------------------
# 1. AUDITORIA DE DIMENSIONES
# --------------------------------------------------------------------------

def assert_shapes(ens: np.ndarray, y: np.ndarray, scale: np.ndarray) -> tuple:
    """Verifica la coherencia dimensional antes de cualquier aritmetica."""
    if ens.ndim != 4:
        raise ValueError(f"ens debe ser (O,H,N,M); recibido ndim={ens.ndim}")
    O, H, N, M = ens.shape

    if y.shape != (O, H, N):
        raise ValueError(f"y debe ser {(O, H, N)}; recibido {y.shape}")
    if scale.shape != (O, N):
        raise ValueError(f"scale debe ser {(O, N)}; recibido {scale.shape}")
    if M < 2:
        raise ValueError("el estimador insesgado de CRPS requiere M >= 2")

    return O, H, N, M


def numerical_guards(*arrays: np.ndarray) -> None:
    """Estabilidad: ningun NaN/Inf entra al calculo de metricas."""
    for i, a in enumerate(arrays):
        if not np.isfinite(a).all():
            n_bad = int((~np.isfinite(a)).sum())
            raise FloatingPointError(f"array {i}: {n_bad} valores no finitos")


def report_magnitudes(ens, y, scale, label=""):
    """Magnitudes: detecta dominancia de escala ANTES de agregar entre series."""
    O, H, N, M = ens.shape
    print(f"\n--- magnitudes {label} ---")
    for n in range(N):
        err = np.abs(ens[:, :, n, :].mean(axis=-1) - y[:, :, n])
        print(f"  serie {n}: |error| medio = {err.mean():10.4f}   "
              f"escala mediana = {np.median(scale[:, n]):10.4f}")


# --------------------------------------------------------------------------
# 2. METRICAS
# --------------------------------------------------------------------------

def crps_ensemble(ens: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    CRPS con el estimador INSESGADO (fair):

        CRPS = (1/M) sum_i |x_i - y|  -  1/(2 M (M-1)) sum_i sum_j |x_i - x_j|

    El segundo termino usa M(M-1) y no M^2: con M^2 el estimador queda sesgado
    a favor de ensembles pequenos, lo que premiaria artificialmente a la cabeza
    con menos muestras.

    ens (O,H,N,M), y (O,H,N)  ->  (O,H,N)
    """
    M = ens.shape[-1]
    term1 = np.abs(ens - y[..., None]).mean(axis=-1)
    diff = np.abs(ens[..., :, None] - ens[..., None, :])      # (O,H,N,M,M)
    term2 = diff.sum(axis=(-2, -1)) / (2.0 * M * (M - 1))
    return term1 - term2


def energy_score(ens: np.ndarray, y: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """
    Energy Score multivariado sobre series ESTANDARIZADAS.
    La norma se toma sobre el eje de series, por eso la estandarizacion
    previa es obligatoria: sin ella la norma la fija la serie de mayor escala.

    ens (O,H,N,M), y (O,H,N), scale (O,N)  ->  (O,H)
    """
    s = scale[:, None, :, None]                               # (O,1,N,1)
    e = ens / s
    t = (y / scale[:, None, :])[..., None]                    # (O,H,N,1)

    M = e.shape[-1]
    term1 = np.linalg.norm(e - t, axis=2).mean(axis=-1)       # (O,H)
    d = e[..., :, None] - e[..., None, :]                     # (O,H,N,M,M)
    term2 = np.linalg.norm(d, axis=2).sum(axis=(-2, -1)) / (2.0 * M * (M - 1))
    return term1 - term2


def joint_crps(ens, y, scale, standardize: bool = True) -> np.ndarray:
    """
    CRPS conjunto por (origen, horizonte).

    standardize=True  -> cada serie se divide por su escala antes de promediar.
                         Las tres series pesan igual. ESTE ES EL DEFECTO.
    standardize=False -> suma cruda. Se conserva solo para mostrar en el informe
                         cuanto cambia la conclusion segun la agregacion.

    -> (O,H)
    """
    c = crps_ensemble(ens, y)                                 # (O,H,N)
    if standardize:
        c = c / scale[:, None, :]
        return c.mean(axis=-1)
    return c.sum(axis=-1)


# --------------------------------------------------------------------------
# 3. DIEBOLD-MARIANO con HAC (Newey-West) y correccion HLN
# --------------------------------------------------------------------------

def diebold_mariano(loss_a: np.ndarray, loss_b: np.ndarray, h: int,
                    lag: int | None = None) -> dict:
    """
    Contrasta H0: E[L_A - L_B] = 0  contra  H1: E[L_A - L_B] > 0
    (es decir, B mejor que A).

    loss_a, loss_b : (O,)  perdida por origen, MISMO origen y MISMO dato.
    h              : horizonte. Pronosticos solapados -> autocorrelacion de
                     orden h-1 como minimo; por eso el HAC no es opcional.
    """
    from scipy import stats

    d = np.asarray(loss_a, float) - np.asarray(loss_b, float)
    n = d.size
    if n < 8:
        raise ValueError(f"n={n} insuficiente para inferencia HAC")

    dbar = d.mean()
    u = d - dbar
    L = (h - 1) if lag is None else lag
    L = max(0, min(L, n - 1))

    gamma0 = (u @ u) / n
    var = gamma0
    for k in range(1, L + 1):                                 # Newey-West
        gk = (u[k:] @ u[:-k]) / n
        var += 2.0 * (1.0 - k / (L + 1.0)) * gk

    if var <= 0:                                              # salvaguarda
        var = gamma0 if gamma0 > 0 else np.finfo(float).eps

    dm = dbar / np.sqrt(var / n)

    # Harvey-Leybourne-Newbold: corrige el sesgo de muestra pequena
    corr = np.sqrt(max((n + 1 - 2 * h + h * (h - 1) / n) / n, 1e-12))
    dm_hln = dm * corr
    p_one = 1.0 - stats.t.cdf(dm_hln, df=n - 1)

    return {"n": n, "mean_diff": dbar, "dm": dm, "dm_hln": dm_hln,
            "p_one_sided": p_one, "nw_lag": L}


# --------------------------------------------------------------------------
# 4. DECISION SOBRE H1
# --------------------------------------------------------------------------

def test_h1(ens_a, y_a, scale_a, ens_b, y_b, scale_b,
            standardize: bool = True, verbose: bool = True) -> dict:
    """
    H1 es UNA hipotesis, no 36. Se contrasta sobre el CRPS conjunto agregado
    en horizontes, con un unico test. Los cortes por serie y por horizonte son
    descriptivos y NO redefinen el exito.
    """
    assert_shapes(ens_a, y_a, scale_a)
    assert_shapes(ens_b, y_b, scale_b)
    numerical_guards(ens_a, y_a, scale_a, ens_b, y_b, scale_b)

    if not np.allclose(y_a, y_b):
        raise ValueError("A y B deben evaluarse sobre los MISMOS realizados")

    ja = joint_crps(ens_a, y_a, scale_a, standardize)          # (O,H)
    jb = joint_crps(ens_b, y_b, scale_b, standardize)

    crps_a, crps_b = ja.mean(), jb.mean()
    reduction = (crps_a - crps_b) / crps_a

    H = ja.shape[1]
    dm = diebold_mariano(ja.mean(axis=1), jb.mean(axis=1), h=H)

    passes = (reduction >= H1_MIN_REDUCTION) and (dm["p_one_sided"] < H1_ALPHA)

    if verbose:
        agg = "estandarizado" if standardize else "suma cruda"
        print(f"\n=== H1  ({agg}) ===")
        print(f"  CRPS conjunto   A = {crps_a:.6f}")
        print(f"  CRPS conjunto   B = {crps_b:.6f}")
        print(f"  reduccion         = {reduction*100:+.2f} %   "
              f"(umbral {H1_MIN_REDUCTION*100:.0f} %)")
        print(f"  DM (HLN)          = {dm['dm_hln']:.3f}   "
              f"p = {dm['p_one_sided']:.4f}   lag NW = {dm['nw_lag']}")
        print(f"  VEREDICTO         : H1 {'SE SOSTIENE' if passes else 'NO SE SOSTIENE'}")

    return {"crps_a": crps_a, "crps_b": crps_b, "reduction": reduction,
            "dm": dm, "passes": passes}


# --------------------------------------------------------------------------
# 5. DATOS DUMMY  (reemplazar por el volcado real conservando las formas)
# --------------------------------------------------------------------------

def make_dummy_run(O=109, H=12, N=3, M=200, seed=0,
                   scales=(29.0, 37.0, 1.7), sharper=1.0):
    """
    Genera un run sintetico con las escalas DISPARES del panel real, para que
    el efecto de la agregacion sea visible. `sharper` < 1 emula una cabeza
    mejor calibrada solo en la serie 0 (cobre).
    """
    rng = np.random.default_rng(seed)
    sc = np.tile(np.asarray(scales, float), (O, 1))            # (O,N)

    y = rng.normal(0.0, 1.0, size=(O, H, N)) * sc[:, None, :]

    spread = np.ones(N)
    spread[0] = sharper                                        # solo cobre
    ens = (y[..., None]
           + rng.normal(0, 1, size=(O, H, N, M))
           * (sc[:, None, :, None] * spread[None, None, :, None]))
    return ens, y, sc


if __name__ == "__main__":
    print("=" * 70)
    print("AUDITORIA 1 - DIMENSIONES")
    print("=" * 70)
    ens_a, y, sc = make_dummy_run(seed=1, sharper=1.00)
    ens_b, _, _ = make_dummy_run(seed=1, sharper=0.70)          # B mejora cobre
    print(f"  ens {ens_a.shape}   y {y.shape}   scale {sc.shape}")
    print(f"  assert_shapes -> {assert_shapes(ens_a, y, sc)}")

    print("\n" + "=" * 70)
    print("AUDITORIA 2 - MAGNITUDES")
    print("=" * 70)
    report_magnitudes(ens_a, y, sc, "Version A")

    print("\n" + "=" * 70)
    print("AUDITORIA 3 - ESTABILIDAD")
    print("=" * 70)
    numerical_guards(ens_a, ens_b, y, sc)
    print("  sin NaN ni Inf")
    bad = ens_a.copy(); bad[0, 0, 0, 0] = np.nan
    try:
        numerical_guards(bad)
    except FloatingPointError as e:
        print(f"  guarda activa: {e}")

    print("\n" + "=" * 70)
    print("CONTRASTE - la agregacion cambia la conclusion")
    print("=" * 70)
    r_std = test_h1(ens_a, y, sc, ens_b, y, sc, standardize=True)
    r_raw = test_h1(ens_a, y, sc, ens_b, y, sc, standardize=False)

    print("\n" + "-" * 70)
    print(f"  estandarizado : {r_std['reduction']*100:+.2f} %  "
          f"-> H1 {'sostiene' if r_std['passes'] else 'no sostiene'}")
    print(f"  suma cruda    : {r_raw['reduction']*100:+.2f} %  "
          f"-> H1 {'sostiene' if r_raw['passes'] else 'no sostiene'}")
    print("-" * 70)
