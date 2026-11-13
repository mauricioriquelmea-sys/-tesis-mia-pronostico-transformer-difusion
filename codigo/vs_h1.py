"""
vs_h1.py  -  Variogram Score de la Version A frente a la Version B
===================================================================

Analisis complementario posterior al borrador. Mide si R_G cambia la
DEPENDENCIA entre series que el pronostico conjunto reproduce, algo que el
CRPS marginal no observa y que el Energy Score detecta con poca sensibilidad
(Pinson y Tastu, 2013; Scheuerer y Hamill, 2015).

No redefine el veredicto de ninguna hipotesis. Las reglas estan fijadas en
resultados/A.1/PREREGISTRO_VS.md antes de la primera ejecucion completa.

DEFINICION (Scheuerer y Hamill, 2015, ec. 4)
--------------------------------------------
Para un vector realizado y en R^d y un ensemble X^(1..M):

    VS_p = sum_{i<j} w_ij ( |y_i - y_j|^p  -  (1/M) sum_m |X_i^m - X_j^m|^p )^2

con p = 0,5 y pesos w_ij = 1. Las series se dividen antes por su escala
(desviacion de los cambios hasta cada origen), igual que en el Energy Score
del harness, para que ningun par quede dominado por el USD/CLP.

Se calculan dos versiones:
    VS_series      d = 3   (las tres series en un mismo horizonte), por (o, h)
                           y promediado en h. Es la metrica principal.
    VS_trayectoria d = 36  (3 series x 12 horizontes), por origen. Incluye la
                           dependencia entre horizontes. Descriptiva.

CONFIGURACIONES
---------------
    --config recontraste  (por defecto)  objetivo en cambios, cabeza t-Student,
                          eta calibrado a fraccion estructural 0,20. Es la
                          configuracion del re-contraste de H1 (anexo A.4).
    --config h1           objetivo en niveles, cabeza gaussiana, eta = 5.
                          Es la configuracion del contraste original de H1.

Como control de reproduccion, el guion recalcula el CRPS conjunto de A y B y
lo imprime junto al valor publicado. Si no coinciden, el entorno difiere y
las cifras de VS no deben compararse con las del informe.

USO (desde la raiz del repositorio)
    py -3.12 codigo/vs_h1.py --panel cuadernos/panel_2003_2024.csv
    py -3.12 codigo/vs_h1.py --panel cuadernos/panel_2003_2024.csv --config h1
    py -3.12 codigo/vs_h1.py --rapido          # panel sintetico, prueba de humo

AISLAMIENTO: lee el panel para entrenar y evaluar. No lo guarda ni lo imprime.
El JSON contiene solo metricas agregadas y perdidas por origen.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

P_VS = 0.5                      # exponente fijado en el prerregistro
H_DM = 12                       # rezago Newey-West = h - 1 = 11

# CRPS conjunto publicado, para el control de reproduccion (media de semilla)
PUBLICADO = {
    "recontraste": {0: (2.5508, 2.3976), 1: (2.5218, 2.3412), 2: (2.5974, 2.3054)},
    "h1": {},                   # el informe publica reducciones, no niveles
}


# --------------------------------------------------------------------------
# 1. Metricas
# --------------------------------------------------------------------------

def variogram_score(ens: np.ndarray, y: np.ndarray, p: float = P_VS) -> np.ndarray:
    """
    Variogram Score con pesos unitarios sobre el ultimo eje de dimension.

    ens (..., d, M)   y (..., d)   ->   (...)

    Usa solo pares i<j. La esperanza bajo el pronostico se estima con la media
    del ensemble, que es insesgada para E|X_i - X_j|^p.
    """
    if ens.shape[:-1] != y.shape:
        raise ValueError(f"formas incompatibles: ens {ens.shape}, y {y.shape}")
    d = y.shape[-1]
    iu, ju = np.triu_indices(d, k=1)
    vy = np.abs(y[..., iu] - y[..., ju]) ** p                       # (..., P)
    vx = (np.abs(ens[..., iu, :] - ens[..., ju, :]) ** p).mean(-1)  # (..., P)
    return ((vy - vx) ** 2).sum(-1)


def estandarizar(ens, y, scale):
    """ens (O,H,N,M), y (O,H,N), scale (O,N) -> versiones sin unidades."""
    return ens / scale[:, None, :, None], y / scale[:, None, :]


def vs_series(ens, y, scale):
    """VS sobre las N series en cada (o, h).   -> (O, H)"""
    e, t = estandarizar(ens, y, scale)
    return variogram_score(e, t)


def vs_trayectoria(ens, y, scale):
    """VS sobre el vector (H*N) completo de cada origen.   -> (O,)"""
    e, t = estandarizar(ens, y, scale)
    O, H, N, M = e.shape
    return variogram_score(e.reshape(O, H * N, M), t.reshape(O, H * N))


# --------------------------------------------------------------------------
# 2. Verificaciones de la metrica (sin datos)
# --------------------------------------------------------------------------

def autoverificacion(seed: int = 0) -> dict:
    """
    Tres pruebas sobre tensores dummy:
      a) VS = 0 si todas las muestras y el realizado son iguales.
      b) Un ensemble con la correlacion verdadera puntua mejor que uno con
         correlacion invertida y las mismas marginales.
      c) El CRPS marginal NO distingue esos dos ensembles (por eso se usa VS).
    """
    from h1_harness import crps_ensemble
    rng = np.random.default_rng(seed)
    O, H, N, M = 300, 1, 3, 200

    y0 = rng.normal(size=(O, H, N))
    a = variogram_score(np.repeat(y0[..., None], M, -1), y0)
    ok_a = bool(np.allclose(a, 0.0))

    rho = 0.8
    C = np.full((N, N), rho) + (1 - rho) * np.eye(N)
    Cm = C.copy(); Cm[0, 1:] = Cm[1:, 0] = -rho / 2       # dependencia distinta
    L, Lm = np.linalg.cholesky(C), np.linalg.cholesky(Cm)
    y = (L @ rng.normal(size=(O * H, N, 1)))[..., 0].reshape(O, H, N)
    ok = (L @ rng.normal(size=(O * H, N, M))).reshape(O, H, N, M)
    mal = (Lm @ rng.normal(size=(O * H, N, M))).reshape(O, H, N, M)
    vs_ok, vs_mal = variogram_score(ok, y).mean(), variogram_score(mal, y).mean()
    cr_ok, cr_mal = crps_ensemble(ok, y).mean(), crps_ensemble(mal, y).mean()

    res = {"a_cero_si_identicos": ok_a,
           "b_vs_correcta": float(vs_ok), "b_vs_invertida": float(vs_mal),
           "b_distingue": bool(vs_ok < vs_mal),
           "c_crps_correcta": float(cr_ok), "c_crps_invertida": float(cr_mal),
           "c_crps_rel_dif": float(abs(cr_ok - cr_mal) / cr_ok)}
    if not (ok_a and res["b_distingue"]):
        raise AssertionError(f"autoverificacion fallida: {res}")
    return res


# --------------------------------------------------------------------------
# 3. Ejecucion
# --------------------------------------------------------------------------

def correr(a):
    import torch
    from bloques_h1 import evaluar_ventana_expansiva, _panel_sintetico, H_FUT
    from h1_harness import (joint_crps, energy_score, diebold_mariano,
                            assert_shapes, numerical_guards)
    import recontraste_h1_cambio as rc

    torch.set_num_threads(1)
    if a.config == "recontraste":
        objetivo, cabeza, eta_fijo = "cambio", "student-t", None
    else:
        objetivo, cabeza, eta_fijo = "nivel", "gaussian", 5.0

    semillas = a.semillas or ([0] if a.rapido else [0, 1, 2])
    n_orig = 24 if a.rapido else 109
    pasos = 60 if a.rapido else 400
    M = 60 if a.rapido else 200

    if a.panel:
        from h1_real import cargar_panel
        panel, t0, real = cargar_panel(a.panel, None), 143, True
    else:
        t0, real = 144, False
        panel = _panel_sintetico(T=t0 + n_orig + H_FUT, seed=12345)

    env = rc.entorno()
    ts = datetime.now().isoformat(timespec="seconds")
    print("=" * 78)
    print(f"VARIOGRAM SCORE  A vs B   config={a.config}   p={P_VS}")
    print("=" * 78)
    print(f"  {ts}   torch {env['torch']}   numpy {env['numpy']}   hilos {env['hilos']}")
    print(f"  panel {'real' if real else 'sintetico'}  T0={t0}  origenes={n_orig}"
          f"  M={M}  pasos={pasos}  semillas={tuple(semillas)}")
    print(f"  objetivo={objetivo}  cabeza={cabeza}  "
          f"eta={'calibrado a 0,20' if eta_fijo is None else eta_fijo}")
    reducida = a.rapido
    if reducida:
        print("  AVISO: configuracion reducida, solo prueba de humo.")

    print("\n  Autoverificacion de la metrica (tensores dummy):")
    av = autoverificacion()
    print(f"    VS=0 con ensemble degenerado: {av['a_cero_si_identicos']}")
    print(f"    VS correcta {av['b_vs_correcta']:.4f} < invertida "
          f"{av['b_vs_invertida']:.4f}: {av['b_distingue']}")
    print(f"    CRPS apenas cambia: {100*av['c_crps_rel_dif']:.2f} %")

    salida = {"timestamp": ts, "entorno": env, "config": a.config,
              "panel": "real" if real else "sintetico",
              "configuracion_reducida": reducida, "p": P_VS, "pesos": "unitarios",
              "objetivo": objetivo, "cabeza": cabeza, "T0": t0,
              "n_origenes": n_orig, "M": M, "pasos": pasos,
              "semillas": list(semillas), "autoverificacion": av,
              "prerregistro": "resultados/A.1/PREREGISTRO_VS.md",
              "por_semilla": {}}

    for s in semillas:
        t = time.time()
        eta = (eta_fijo if eta_fijo is not None else
               rc.eta_para_fraccion(panel, t0, s, 0.20, cabeza)[0])
        kw = dict(T0=t0, n_origenes=n_orig, M=M, refit_cada=12, steps=pasos,
                  dist_type=cabeza, objetivo=objetivo)
        ens_a, y, sc, _ = evaluar_ventana_expansiva(panel, "A", s, 0.0, **kw)
        ens_b, y_b, sc_b, _ = evaluar_ventana_expansiva(panel, "B", s, eta, **kw)
        assert_shapes(ens_a, y, sc); assert_shapes(ens_b, y_b, sc_b)
        numerical_guards(ens_a, ens_b, y, sc)
        if not (np.allclose(y, y_b) and np.allclose(sc, sc_b)):
            raise RuntimeError("A y B no comparten realizados o escalas")

        crps_a = joint_crps(ens_a, y, sc).mean(1)
        crps_b = joint_crps(ens_b, y, sc).mean(1)
        es_a, es_b = energy_score(ens_a, y, sc).mean(1), energy_score(ens_b, y, sc).mean(1)
        vs_a, vs_b = vs_series(ens_a, y, sc).mean(1), vs_series(ens_b, y, sc).mean(1)
        vt_a, vt_b = vs_trayectoria(ens_a, y, sc), vs_trayectoria(ens_b, y, sc)

        def resumen(la, lb):
            dm = diebold_mariano(la, lb, h=H_DM)
            return {"A": float(la.mean()), "B": float(lb.mean()),
                    "reduccion": float((la.mean() - lb.mean()) / la.mean()),
                    "dm_hln": float(dm["dm_hln"]), "p": float(dm["p_one_sided"]),
                    "p_reciproco": float(diebold_mariano(lb, la, h=H_DM)["p_one_sided"])}

        r = {"eta": float(eta), "crps": resumen(crps_a, crps_b),
             "es": resumen(es_a, es_b), "vs_series": resumen(vs_a, vs_b),
             "vs_trayectoria": resumen(vt_a, vt_b),
             "perdidas_por_origen": {"vs_series_A": vs_a.tolist(),
                                     "vs_series_B": vs_b.tolist()},
             "segundos": round(time.time() - t, 1)}
        pub = PUBLICADO.get(a.config, {}).get(s)
        if pub and not reducida:
            r["control_reproduccion"] = {
                "crps_A_publicado": pub[0], "crps_B_publicado": pub[1],
                "coincide": bool(abs(r["crps"]["A"] - pub[0]) < 5e-4
                                 and abs(r["crps"]["B"] - pub[1]) < 5e-4)}
        salida["por_semilla"][str(s)] = r

        print(f"\n  semilla {s}  (eta = {eta:.2f}, {r['segundos']:.0f} s)")
        print(f"    {'metrica':<16}{'A':>10}{'B':>10}{'reduccion':>11}{'DM-HLN':>9}{'p':>8}")
        for k in ("crps", "es", "vs_series", "vs_trayectoria"):
            m = r[k]
            print(f"    {k:<16}{m['A']:>10.4f}{m['B']:>10.4f}"
                  f"{100*m['reduccion']:>10.2f}%{m['dm_hln']:>9.3f}{m['p']:>8.4f}")
        if "control_reproduccion" in r:
            c = r["control_reproduccion"]
            print(f"    control: CRPS publicado A={c['crps_A_publicado']} "
                  f"B={c['crps_B_publicado']}  -> coincide: {c['coincide']}")

    red = [salida["por_semilla"][str(s)]["vs_series"]["reduccion"] for s in semillas]
    salida["vs_series_reduccion_media"] = float(np.mean(red))
    print(f"\n  VS_series: reduccion media {100*np.mean(red):+.2f} %   "
          f"(descriptivo; no redefine H1)")

    tipo = "prueba" if reducida else "full"
    out = Path(f"vs_h1_{a.config}_{tipo}_{datetime.now().date()}.json")
    out.write_text(json.dumps(salida, indent=2, ensure_ascii=False))
    print(f"  -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default=None)
    ap.add_argument("--config", choices=("recontraste", "h1"), default="recontraste")
    ap.add_argument("--semillas", nargs="*", type=int, default=None)
    ap.add_argument("--rapido", action="store_true")
    ap.add_argument("--solo-autoverificacion", action="store_true")
    a = ap.parse_args()
    if a.solo_autoverificacion:
        print(json.dumps(autoverificacion(), indent=2))
        return
    correr(a)


if __name__ == "__main__":
    main()
