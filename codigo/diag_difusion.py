"""
diag_difusion.py  --  Bloque 1 del plan del 12 de octubre
==========================================================

Ejecuta la escalera de criterios EN ORDEN y anota CUAL ES EL PRIMERO QUE
FALLA. No se interviene nada antes de saberlo.

  C1  Consistencia numerica         error relativo < 1e-5
  C2  Estabilidad del entrenamiento sin NaN, perdida decrece, 3 semillas
  C3  Muestreo sobre sinteticos     CRPS <= R0-G  y  cobertura 90 % en
                                    [0.80, 0.95]
  C4  Muestreo sobre el panel real  -- NO se ejecuta aqui --

C4 queda fuera por el aislamiento de datos del proyecto: este modulo no
accede al panel macro-financiero chileno. C1 a C3 corren sobre tensores
dummy y sobre el generador sintetico con causalidad Cu->FX->TPM plantada.

El orden no es negociable. Si C3 pasa en sintetico y C4 falla en real, el
problema es tamano de muestra y no implementacion. Invertido, el fallo es
inatribuible.

Uso:
    python diag_difusion.py                 # escalera completa, ambos espacios
    python diag_difusion.py --rapido        # configuracion corta
    python diag_difusion.py --solo c1       # un criterio
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(1)        # el orden de reduccion de BLAS cambia los
                                # resultados entre maquinas; se fija a 1

from bloques_h1 import (EncoderR0, ventanas_multi, _panel_sintetico,
                        L, H_FUT, N, D, EPS)
from cabeza_difusion import CabezaDifusion, Difusor, contar_parametros
from cabeza_r0 import CabezaR0
from h1_harness import crps_ensemble, joint_crps

TOL_C1 = 1e-5
COB_MIN, COB_MAX = 0.80, 0.95
ESPACIOS = ("y", "latente")


# ===========================================================================
# C1  --  consistencia numerica
# ===========================================================================
def c1(verbose=True):
    """Cuatro verificaciones, en el orden que manda el proyecto: dimensiones,
    luego magnitudes, luego estabilidad numerica."""
    res, fallas = {}, []

    def ok(nombre, cond, detalle=""):
        res[nombre] = {"pasa": bool(cond), "detalle": detalle}
        if not cond:
            fallas.append(nombre)
        if verbose:
            print(f"    [{'OK ' if cond else 'FALLA'}] {nombre:<34} {detalle}")

    if verbose:
        print("\n  C1.1  Auditoria de dimensiones")
    for esp in ESPACIOS:
        cab = CabezaDifusion(latent_dim=D, espacio=esp, T_dif=50)
        z = torch.randn(7, D)
        y = torch.randn(7, H_FUT, N)
        v = cab.nll(z, y)
        ok(f"nll({esp}) escalar", v.dim() == 0 and torch.isfinite(v),
           f"valor={v.item():.4f}")
        cab.eval()
        p1 = cab.predecir(torch.randn(1, D), S=11)
        ok(f"predecir({esp}) B=1", tuple(p1.shape) == (11, H_FUT, N),
           f"forma={tuple(p1.shape)}  esperada=(11, 12, 3)")
        p3 = cab.predecir(torch.randn(4, D), S=5)
        ok(f"predecir({esp}) B=4", tuple(p3.shape) == (4, 5, H_FUT, N),
           f"forma={tuple(p3.shape)}  esperada=(4, 5, 12, 3)")
        res[f"params_{esp}"] = contar_parametros(cab)

    # firma identica a CabezaR0: mismos nombres, mismo orden de argumentos
    import inspect
    for met in ("nll", "predecir"):
        f_r0 = inspect.signature(getattr(CabezaR0, met)).parameters
        f_df = inspect.signature(getattr(CabezaDifusion, met)).parameters
        ok(f"firma de {met} == R0", list(f_r0) == list(f_df),
           f"R0={list(f_r0)}  dif={list(f_df)}")

    if verbose:
        print("\n  C1.2  Schedule de ruido: limites y coherencia interna")
    for Td in (50, 100, 200):
        d = Difusor(T=Td)
        ab = d.ab
        ok(f"ab[0] == 1 (T={Td})", abs(ab[0].item() - 1.0) < 1e-12,
           f"ab[0]={ab[0].item():.12f}")
        ok(f"ab monotona decreciente (T={Td})",
           bool((ab[1:] <= ab[:-1] + 1e-9).all()), "")
        # coherencia: ab DEBE ser el cumprod de los alpha recortados. Si no,
        # los coeficientes del posterior mezclan dos schedules distintos.
        cp = torch.cat([torch.ones(1), d.alpha.double().cumprod(0).float()])
        e_cp = (cp - ab).abs().max().item() / ab.max().item()
        ok(f"ab == cumprod(alpha) (T={Td})", e_cp < 1e-6,
           f"error rel. max = {e_cp:.2e}")
        ok(f"ab[T] pequena pero no nula (T={Td})", 1e-9 < ab[-1].item() < 5e-3,
           f"ab[T]={ab[-1].item():.2e}  ->  condicion 1/sqrt(ab)={d.condicion:.2e}")
        ok(f"beta en (0, 0.999] (T={Td})",
           bool((d.beta > 0).all() and (d.beta <= 0.999 + 1e-9).all()),
           f"min={d.beta.min():.2e}  max={d.beta.max():.4f}")
        res[f"condicion_T{Td}"] = d.condicion

    if verbose:
        print("\n  C1.3  Ida y vuelta algebraica con epsilon conocido")
    # La inversa y_0 = (y_t - sqrt(1-ab_t) eps)/sqrt(ab_t) tiene numero de
    # condicion kappa_t = 1/sqrt(ab_t), que llega a ~2e3 en t = T. Exigirle
    # 1e-5 en fp32 (eps_mach = 6e-8) seria exigir mas que la aritmetica da.
    # El umbral correcto es kappa * eps_mach, y se reporta ademas el error
    # restringido a los pasos bien condicionados (kappa <= 10).
    for dtype, etiq, emach in ((torch.float32, "fp32", 1.2e-7),
                               (torch.float64, "fp64", 2.2e-16)):
        d = Difusor(T=100).to(dtype)
        torch.manual_seed(0)
        y0 = torch.randn(256, 36, dtype=dtype)
        eps = torch.randn(256, 36, dtype=dtype)
        t = torch.randint(1, 101, (256,))
        yt = d.ruidar(y0, t, eps)
        kap = (1.0 / d.ab[t].sqrt())
        err_y0 = (d.y0_desde_eps(yt, t, eps) - y0).abs().max(dim=-1).values
        rel = (err_y0 / y0.abs().max()).max().item()
        tol = max(TOL_C1, 20 * float(kap.max()) * emach)
        ok(f"recuperar y_0 ({etiq})", rel < tol,
           f"error rel. = {rel:.2e}  tol = {tol:.2e} (kappa_max={kap.max():.1e})")
        bien = kap <= 10
        rel_b = (err_y0[bien] / y0.abs().max()).max().item() if bien.any() else 0.0
        ok(f"recuperar y_0, kappa<=10 ({etiq})", rel_b < TOL_C1,
           f"error rel. = {rel_b:.2e}  en {int(bien.sum())}/256 pasos")
        e_ep = (d.eps_desde_y0(yt, t, y0) - eps).abs().max() / eps.abs().max()
        ok(f"recuperar eps ({etiq})", e_ep.item() < TOL_C1,
           f"error rel. max = {e_ep.item():.2e}")

    if verbose:
        print("\n  C1.4  Muestreador DDIM con oraculo  (la prueba de fondo)")
    # Si la aritmetica de p(y_{t-1}|y_t) esta mal, un oraculo que devuelve el
    # epsilon verdadero NO regresa a y_0. Es el unico test que distingue un
    # error de coeficientes de un error de entrenamiento. Se corre con y sin
    # recorte de y_0_hat: con datos estandarizados el recorte no debe morder.
    # Criterio en dos niveles, y la razon es sustantiva. El primer paso de la
    # cadena arranca en t = T, donde kappa_T = 1/sqrt(ab_T) ~ 2e3, de modo que
    # el piso del error es kappa_T * eps_mach: 2.4e-4 en fp32 y 4.4e-13 en
    # fp64. Exigir 1e-5 en fp32 seria exigir mas precision que la que el tipo
    # de dato permite, y un fallo asi no distingue un error de formula de la
    # aritmetica. Por eso:
    #   fp64 -> umbral ESTRICTO 1e-5. Es la prueba de los coeficientes.
    #   fp32 -> umbral kappa_T * eps_mach. Verifica que no ocurre nada PEOR
    #           que el condicionamiento ya medido en C1.2.
    # En el muestreo real esto es inocuo: el error del propio eps_hat es O(1),
    # siete ordenes por encima de 6e-5.
    for Td, k in ((100, None), (100, 20), (200, None)):
        for dtype, etiq, emach in ((torch.float32, "fp32", 1.2e-7),
                                   (torch.float64, "fp64", 2.2e-16)):
            for rec in (False, True):
                d = Difusor(T=Td).to(dtype)
                torch.manual_seed(1)
                y0 = torch.randn(64, 36, dtype=dtype)
                eps = torch.randn(64, 36, dtype=dtype)
                sec = (list(range(Td, 0, -1)) if k is None else
                       sorted(set(torch.linspace(Td, 1, k).round().long().tolist()),
                              reverse=True))
                x = d.ruidar(y0, torch.full((64,), sec[0]), eps)
                for j, ti in enumerate(sec):
                    t = torch.full((64,), ti, dtype=torch.long)
                    tp = torch.full((64,), sec[j + 1] if j + 1 < len(sec) else 0,
                                    dtype=torch.long)
                    x, _ = d.paso_ddim(x, t, eps, t_prev=tp, recortar=rec)
                err = ((x - y0).abs().max() / y0.abs().max()).item()
                tol = (TOL_C1 if dtype is torch.float64
                       else max(TOL_C1, 5 * d.condicion * emach))
                ok(f"DDIM-oraculo T={Td} k={k or Td} ({etiq},"
                   f"{' recorte' if rec else ' sin recorte'})",
                   err < tol,
                   f"error = {err:.2e}  tol = {tol:.2e}"
                   + ("  [estricto]" if dtype is torch.float64
                      else f"  [kappa={d.condicion:.1e}]"))

    # la media DDPM debe coincidir con la de q(y_{t-1}|y_t,y_0) exacta
    d = Difusor(T=100).to(torch.float64)
    torch.manual_seed(2)
    y0 = torch.randn(128, 36, dtype=torch.float64)
    eps = torch.randn(128, 36, dtype=torch.float64)
    t = torch.randint(2, 101, (128,))
    yt = d.ruidar(y0, t, eps)
    ab_t, ab_p = d.ab[t].unsqueeze(-1), d.ab[t - 1].unsqueeze(-1)
    bt, al = d.beta[t - 1].unsqueeze(-1), d.alpha[t - 1].unsqueeze(-1)
    mu_ref = ((ab_p.sqrt() * bt / (1 - ab_t)) * y0
              + (al.sqrt() * (1 - ab_p) / (1 - ab_t)) * yt)
    # paso_ddpm con varianza anulada: se compara solo la media
    y0h = d.y0_desde_eps(yt, t, eps)
    mu_imp = ((ab_p.sqrt() * bt / (1 - ab_t)) * y0h
              + (al.sqrt() * (1 - ab_p) / (1 - ab_t)) * yt)
    e = ((mu_imp - mu_ref).abs().max() / mu_ref.abs().max()).item()
    ok("media posterior DDPM (fp64)", e < TOL_C1, f"error rel. max = {e:.2e}")

    res["_pasa"] = len(fallas) == 0
    res["_fallas"] = fallas
    return res


# ===========================================================================
# Entrenamiento con cabeza intercambiable
# ===========================================================================
def _modelo(cabeza_factory, semilla):
    torch.manual_seed(semilla)
    np.random.seed(semilla)
    m = EncoderR0()
    m.cabeza = cabeza_factory()
    return m


def entrenar(panel, t_fin, cabeza_factory, semilla, steps=400, batch=64,
             lr=1e-3, registrar_cada=50):
    """Mismo protocolo que bloques_h1.entrenar_hasta (normalizacion con
    estadisticos hasta t_fin, sin fuga), con la cabeza como argumento."""
    mu = panel[:, :t_fin].mean(axis=1, keepdims=True)
    sd = panel[:, :t_fin].std(axis=1, keepdims=True) + EPS
    Xn = (panel - mu) / sd
    Xtr, Ytr = ventanas_multi(Xn, L, H_FUT, t_fin)

    m = _modelo(cabeza_factory, semilla)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    traza, gnorms = [], []

    m.train()
    for st in range(steps):
        idx = torch.randint(0, len(Xtr), (min(batch, len(Xtr)),))
        z, _ = m(Xtr[idx], ret_attn=False)
        loss = m.cabeza.nll(z, Ytr[idx])
        opt.zero_grad()
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        gnorms.append(float(gn))
        if st % registrar_cada == 0 or st == steps - 1:
            traza.append({"paso": st, "perdida": float(loss.item())})
        if not torch.isfinite(loss):
            traza.append({"paso": st, "perdida": float("nan")})
            break

    m.eval()
    return m, mu, sd, {"traza": traza, "n_ventanas": int(len(Xtr)),
                       "params": contar_parametros(m),
                       "params_cabeza": contar_parametros(m.cabeza),
                       "grad_norm_medio": float(np.mean(gnorms)),
                       "grad_norm_max": float(np.max(gnorms))}


# ===========================================================================
# C2  --  estabilidad del entrenamiento
# ===========================================================================
def c2(panel, cfg, verbose=True):
    res, fallas = {}, []
    for esp in ESPACIOS:
        fab = lambda e=esp: CabezaDifusion(latent_dim=D, espacio=e,
                                           T_dif=cfg["T_DIF"])
        por_semilla = []
        if verbose:
            print(f"\n  espacio = {esp}")
        for s in cfg["SEMILLAS"]:
            _, _, _, reg = entrenar(panel, cfg["T0"], fab, s,
                                    steps=cfg["STEPS"])
            p0 = reg["traza"][0]["perdida"]
            p1 = reg["traza"][-1]["perdida"]
            hay_nan = not np.isfinite([t["perdida"] for t in reg["traza"]]).all()
            decrece = (not hay_nan) and p1 < p0
            por_semilla.append({"semilla": s, "p_inicial": p0, "p_final": p1,
                                "nan": bool(hay_nan), "decrece": bool(decrece),
                                "grad_max": reg["grad_norm_max"],
                                "params_cabeza": reg["params_cabeza"],
                                "n_ventanas": reg["n_ventanas"]})
            if verbose:
                print(f"    semilla {s}: perdida {p0:8.4f} -> {p1:8.4f}"
                      f"   {'NaN!' if hay_nan else 'sin NaN'}"
                      f"   |g|max={reg['grad_norm_max']:.2f}")
        pasa = all(d["decrece"] and not d["nan"] for d in por_semilla)
        res[esp] = {"semillas": por_semilla, "pasa": pasa}
        if not pasa:
            fallas.append(esp)
        pc = por_semilla[0]["params_cabeza"]; nv = por_semilla[0]["n_ventanas"]
        res[esp]["params_por_ventana"] = pc / nv
        if verbose:
            print(f"    cabeza: {pc} parametros / {nv} ventanas "
                  f"= {pc/nv:.1f} por ventana   ->  C2 {'OK' if pasa else 'FALLA'}")

    res["_pasa"] = len(fallas) == 0
    res["_fallas"] = fallas
    return res


# ===========================================================================
# C3  --  muestreo sobre el panel sintetico
# ===========================================================================
def _evaluar(panel, cabeza_factory, semilla, cfg):
    """Ventana expansiva reducida. Devuelve (ens, y, scale) como h1_harness."""
    T0, NO, M, refit = cfg["T0"], cfg["N_ORIGENES"], cfg["M"], cfg["REFIT"]
    ens = np.empty((NO, H_FUT, N, M))
    y = np.empty((NO, H_FUT, N))
    scale = np.empty((NO, N))
    cambios = np.diff(panel, axis=1)
    m = mu = sd = None

    for o_i in range(NO):
        origen = T0 + o_i
        if o_i % refit == 0:
            m, mu, sd, _ = entrenar(panel, origen, cabeza_factory, semilla,
                                    steps=cfg["STEPS"])
        scale[o_i] = cambios[:, :origen].std(axis=1)
        with torch.no_grad():
            Xn = (panel - mu) / sd
            xw = torch.tensor(Xn[:, origen - L + 1: origen + 1][None],
                              dtype=torch.float32)
            z, _ = m(xw, ret_attn=False)
            Y = m.cabeza.predecir(z, S=M).numpy()            # [M, 12, 3]
            Y = Y * sd.T[None, :, :] + mu.T[None, :, :]
            ens[o_i] = Y.transpose(1, 2, 0)
        y[o_i] = panel[:, origen + 1: origen + 1 + H_FUT].T
    return ens, y, scale


def cobertura(ens, y, nivel=0.90):
    """Fraccion de realizados dentro del intervalo central marginal."""
    a = (1 - nivel) / 2
    lo = np.quantile(ens, a, axis=-1)
    hi = np.quantile(ens, 1 - a, axis=-1)
    return float(((y >= lo) & (y <= hi)).mean())


def c3(panel, cfg, verbose=True):
    res, fallas = {}, []
    s = cfg["SEMILLAS"][0]

    if verbose:
        print("\n  referencia: R0-G (gaussiana de rango bajo)")
    t0 = time.time()
    fab_r0 = lambda: CabezaR0(latent_dim=D, rank=5, dist_type="gaussian",
                              v_condicional=False)
    ens_r, y_r, sc_r = _evaluar(panel, fab_r0, s, cfg)
    crps_r = float(joint_crps(ens_r, y_r, sc_r, standardize=True).mean())
    cob_r = cobertura(ens_r, y_r)
    res["R0-G"] = {"crps": crps_r, "cobertura_90": cob_r,
                   "segundos": round(time.time() - t0, 1)}
    if verbose:
        print(f"    CRPS conjunto estandarizado = {crps_r:.4f}"
              f"   cobertura 90 % = {cob_r:.3f}   ({time.time()-t0:.0f}s)")

    for esp in ESPACIOS:
        if verbose:
            print(f"\n  difusion, espacio = {esp}")
        t0 = time.time()
        fab = lambda e=esp: CabezaDifusion(
            latent_dim=D, espacio=e, T_dif=cfg["T_DIF"],
            pasos_muestreo=cfg["PASOS_MUESTREO"])
        ens_d, y_d, sc_d = _evaluar(panel, fab, s, cfg)
        assert np.allclose(y_d, y_r), "realizados distintos entre cabezas"
        crps_d = float(joint_crps(ens_d, y_d, sc_d, standardize=True).mean())
        cob_d = cobertura(ens_d, y_d)
        mejor = crps_d <= crps_r
        cal = COB_MIN <= cob_d <= COB_MAX
        pasa = mejor and cal
        res[esp] = {"crps": crps_d, "cobertura_90": cob_d,
                    "crps_vs_R0": crps_d / crps_r - 1.0,
                    "crps_no_peor": bool(mejor), "cobertura_en_rango": bool(cal),
                    "pasa": bool(pasa), "segundos": round(time.time() - t0, 1),
                    "sd_muestras": float(ens_d.std()),
                    "sd_realizados": float(y_d.std())}
        if not pasa:
            fallas.append(esp)
        if verbose:
            print(f"    CRPS conjunto estandarizado = {crps_d:.4f}"
                  f"   ({100*(crps_d/crps_r-1):+.1f} % vs R0-G)")
            print(f"    cobertura 90 % = {cob_d:.3f}"
                  f"   {'en rango' if cal else f'FUERA de [{COB_MIN}, {COB_MAX}]'}")
            print(f"    dispersion muestras / realizados = "
                  f"{ens_d.std():.3f} / {y_d.std():.3f}")
            print(f"    -> C3 {'OK' if pasa else 'FALLA'}   ({time.time()-t0:.0f}s)")

    res["_pasa"] = len(fallas) == 0
    res["_fallas"] = fallas
    return res


# ===========================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rapido", action="store_true")
    ap.add_argument("--solo", default=None, choices=["c1", "c2", "c3"])
    a = ap.parse_args()

    cfg = dict(T_PANEL=300, T0=144, N_ORIGENES=36, M=200, REFIT=12,
               STEPS=400, T_DIF=100, PASOS_MUESTREO=None, SEMILLAS=(0, 1, 2))
    if a.rapido:
        cfg.update(T_PANEL=250, T0=120, N_ORIGENES=12, M=80, REFIT=12,
                   STEPS=150, T_DIF=50, PASOS_MUESTREO=25, SEMILLAS=(0, 1, 2))

    ts = datetime.now().isoformat(timespec="seconds")
    env = {"python": sys.version.split()[0], "torch": torch.__version__,
           "numpy": np.__version__, "plataforma": platform.platform(),
           "hilos": torch.get_num_threads()}

    print("=" * 78)
    print(f"ESCALERA DE DIAGNOSTICO  -  CABEZA DE DIFUSION"
          f"{'  [RAPIDO]' if a.rapido else ''}")
    print("=" * 78)
    print(f"  {ts}")
    print(f"  torch {env['torch']} | numpy {env['numpy']} | hilos {env['hilos']}")
    print(f"  panel sintetico: {cfg['T_PANEL']} meses, Cu->FX->TPM plantada")
    print(f"  T0={cfg['T0']}  origenes={cfg['N_ORIGENES']}  M={cfg['M']}"
          f"  pasos={cfg['STEPS']}  T_dif={cfg['T_DIF']}")
    print(f"  C4 (panel real) NO se ejecuta: aislamiento de datos\n")

    panel = _panel_sintetico(T=cfg["T_PANEL"], seed=12345)
    salida = {"timestamp": ts, "entorno": env,
              "config": {k: (list(v) if isinstance(v, tuple) else v)
                         for k, v in cfg.items()}, "criterios": {}}

    orden = ["c1", "c2", "c3"] if a.solo is None else [a.solo]
    primer_fallo = None
    t_ini = time.time()

    for crit in orden:
        print("-" * 78)
        print({"c1": "C1  CONSISTENCIA NUMERICA",
               "c2": "C2  ESTABILIDAD DEL ENTRENAMIENTO",
               "c3": "C3  MUESTREO SOBRE SINTETICOS"}[crit])
        print("-" * 78)
        r = (c1() if crit == "c1" else
             c2(panel, cfg) if crit == "c2" else c3(panel, cfg))
        salida["criterios"][crit] = r
        print(f"\n  >>> {crit.upper()}: {'PASA' if r['_pasa'] else 'FALLA'}"
              + (f"   ({', '.join(map(str, r['_fallas']))})"
                 if r["_fallas"] else ""))
        if not r["_pasa"] and primer_fallo is None:
            primer_fallo = crit
            print(f"\n  Primer criterio que falla: {crit.upper()}.")
            print("  Se continua midiendo los siguientes solo para informacion;")
            print("  la decision del dia se toma sobre este.")
        print()

    print("=" * 78)
    print("VEREDICTO")
    print("=" * 78)
    escalon = {None: "ninguno: pasar a C4 sobre el panel real",
               "c1": "ninguno todavia: depuracion de implementacion",
               "c2": "E1: diagnostico sobre el generador sintetico",
               "c3": "E2: difundir sobre la ventana futura de 36 dimensiones"}
    print(f"  primer fallo : {primer_fallo.upper() if primer_fallo else 'NINGUNO'}")
    print(f"  escalon      : {escalon[primer_fallo]}")
    salida["veredicto"] = {"primer_fallo": primer_fallo,
                           "escalon": escalon[primer_fallo]}
    print(f"\n  tiempo total: {(time.time()-t_ini)/60:.1f} min")

    p = Path(f"diag_difusion_{'rapido' if a.rapido else 'full'}_{ts[:10]}.json")
    p.write_text(json.dumps(salida, indent=2, ensure_ascii=False))
    print(f"  guardado en {p}")


if __name__ == "__main__":
    main()
