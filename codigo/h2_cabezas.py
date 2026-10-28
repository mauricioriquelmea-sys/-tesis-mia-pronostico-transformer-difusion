"""
h2_cabezas.py  -  contraste de H2: la cabeza generativa
========================================================

PREGUNTA. ¿Aporta la cabeza de difusion sobre la cabeza parametrica de
respaldo R0? Es H2, y hasta ahora no se habia contrastado sobre ningun panel.

POR QUE RECIEN AHORA. El 13 de octubre quedo establecido que la
especificacion del objetivo domina el desempeno sobre el panel real: entrenar
sobre el CAMBIO desde el origen en vez de sobre el NIVEL reduce el CRPS
conjunto un 52,3 % (DM-HLN 4,73, p < 0,001). Contrastar H2 bajo la
especificacion peor habria medido las cabezas a traves de un termino de sesgo
comun a todas. Por eso el objetivo por defecto aqui es `cambio`.

CUATRO CABEZAS, un solo condicionante
  R0-G       gaussiana de rango bajo         (la de respaldo; es la REFERENCIA)
  R0-T       Student-t de rango bajo
  dif-y      difusion sobre y in R^36        (escalon E2)
  dif-lat    difusion latente                (la arquitectura de §3.2)

El condicionante es el codificador completo en las cuatro: lo unico que varia
es la cabeza. Sobre el panel real el codificador es la mejor de las tres vias
examinadas, de modo que es tambien la eleccion correcta como base.

-------------------------------------------------------------------------------
EL UMBRAL DE EXITO NO LO FIJA ESTE GUION
-------------------------------------------------------------------------------
Este guion reporta CRPS conjunto estandarizado, Energy Score, cobertura y
Diebold-Mariano contra R0-G. El umbral que define el exito de H2 es el
comprometido en el informe de avance y debe leerse de ahi, no inventarse al
ver los numeros. Si no hay umbral prerregistrado para H2, eso mismo debe
declararse y el resultado reportarse como descriptivo.

-------------------------------------------------------------------------------
ESCALERA
-------------------------------------------------------------------------------
C3 (sinteticos) antes que C4 (panel real), y el orden no es negociable: si
pasa en sintetico y falla en real, el problema es tamano de muestra y no
implementacion; invertido, el fallo es inatribuible.

    python h2_cabezas.py                      # C3: panel sintetico
    python h2_cabezas.py --panel panel.csv    # C4: panel real (tu maquina)
    python h2_cabezas.py --rapido             # verificacion de cableado

    --objetivo nivel      para comparar con la configuracion de los cuadernos
    --inflar              aplica la correccion de varianza (agnostica a la cabeza)

AISLAMIENTO: sin --panel no toca dato real alguno. Con --panel corre en la
maquina del autor y el JSON de salida no contiene el panel.
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
import torch.nn as nn

torch.set_num_threads(1)

from bloques_h1 import (Phase1Tokenizer, Block, causal_mask_modL,
                        ventanas_multi, _panel_sintetico, L, H_FUT, N, D, EPS,
                        NL)
from cabeza_r0 import CabezaR0
from cabeza_difusion import CabezaDifusion
from h1_harness import joint_crps, energy_score, diebold_mariano
from es_escala import energy_score_lineabase

T_SINT = 300
N_VAL = 24
# La rejilla DEBE abarcar lambda < 1. Una cabeza sobre-dispersa necesita
# ESTRECHAR, no ensanchar: dif-y da cobertura 0.99 al 90 %, de modo que una
# rejilla acotada por abajo en 1.0 la deja sin correccion posible mientras
# corrige a las sub-dispersas. Eso no es una calibracion, es una ventaja
# concedida a una familia de cabezas sobre otra.
REJILLA = np.concatenate([np.linspace(0.30, 0.95, 14),
                          np.linspace(1.0, 4.0, 31),
                          np.linspace(4.2, 12.0, 40)])

CFG_SINT = dict(T0=144, N_ORIGENES=36, M=200, REFIT=12, STEPS=400, T_DIF=100)
CFG_REAL = dict(T0=143, N_ORIGENES=109, M=200, REFIT=12, STEPS=400, T_DIF=100)
CFG_RAPIDO = dict(N_ORIGENES=12, M=80, REFIT=12, STEPS=150, T_DIF=50)
SEMILLAS = (0, 1, 2)
REFERENCIA = "R0-G"

# Valores comprometidos en el informe de avance (§3): la mejor linea base es
# ARIMA/GARCH con Energy Score 1,100, y el umbral del 10 % de H2 lo traduce en
# un objetivo de 0,990 o menos, con significancia al 5 % en Diebold-Mariano.
# AMBOS numeros estan en la convencion de NIVELES del cuaderno de lineas base;
# compararlos con el ES en convencion de cambios infla la cifra del modelo por
# un factor del orden de 2 y rechazaria H2 por construccion (ver es_escala.py).
ES_LINEABASE = 1.100
ES_UMBRAL_H2 = 0.990


# ===========================================================================
class CondEnc(nn.Module):
    """El codificador de los cuadernos: tokenizador, NL bloques, mascara
    mod-L, z = media sobre los tokens."""

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


def fabricas(T_dif):
    return {
        "R0-G": lambda: CabezaR0(latent_dim=D, rank=5, dist_type="gaussian",
                                 v_condicional=False),
        "R0-T": lambda: CabezaR0(latent_dim=D, rank=5, dist_type="student-t",
                                 v_condicional=False),
        "dif-y": lambda: CabezaDifusion(latent_dim=D, espacio="y", T_dif=T_dif),
        "dif-lat": lambda: CabezaDifusion(latent_dim=D, espacio="latente",
                                          T_dif=T_dif),
    }


class Modelo(nn.Module):
    def __init__(self, fab_cabeza):
        super().__init__()
        self.cond = CondEnc()
        self.cabeza = fab_cabeza()


# ===========================================================================
def _datos(panel, t_fin, objetivo):
    mu = panel[:, :t_fin].mean(axis=1, keepdims=True)
    sd = panel[:, :t_fin].std(axis=1, keepdims=True) + EPS
    X, Y = ventanas_multi((panel - mu) / sd, L, H_FUT, t_fin)
    if objetivo == "cambio":
        Y = Y - X[:, :, -1].unsqueeze(1)
    return X, Y, mu, sd


def entrenar(panel, t_fin, nombre, semilla, cfg, objetivo, reservar_val=False,
             batch=64, lr=1e-3):
    torch.manual_seed(semilla); np.random.seed(semilla)
    X, Y, mu, sd = _datos(panel, t_fin, objetivo)
    if reservar_val:
        Xtr, Ytr, Xva, Yva = X[:-N_VAL], Y[:-N_VAL], X[-N_VAL:], Y[-N_VAL:]
    else:
        Xtr, Ytr, Xva, Yva = X, Y, X[-N_VAL:], Y[-N_VAL:]
    m = Modelo(fabricas(cfg["T_DIF"])[nombre])
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    m.train()
    p0 = p1 = float("nan")
    for st in range(cfg["STEPS"]):
        i = torch.randint(0, len(Xtr), (min(batch, len(Xtr)),))
        loss = m.cabeza.nll(m.cond(Xtr[i]), Ytr[i])
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        if st == 0:
            p0 = float(loss.detach())
        p1 = float(loss.detach())
        if not np.isfinite(p1):
            break
    m.eval()
    return m, mu, sd, Xva, Yva, {"perdida_inicial": p0, "perdida_final": p1,
                                 "n_ventanas": int(len(Xtr)),
                                 "params": sum(p.numel() for p in m.parameters())}


@torch.no_grad()
def _muestrear(m, X, objetivo, S):
    Y = m.cabeza.predecir(m.cond(X), S=S)
    if Y.dim() == 3:
        Y = Y.unsqueeze(0)                                   # B = 1
    if objetivo == "cambio":
        Y = Y + X[:, :, -1].unsqueeze(1).unsqueeze(1)
    return Y


def calibrar_lambda(m, Xv, Yv, objetivo, S=200, nivel=0.90):
    """Factor escalar que iguala la cobertura fuera de muestra a `nivel`.
    Agnostico a la cabeza: reescala las desviaciones de las muestras respecto
    de su media, asi que vale igual para R0 y para difusion."""
    if objetivo == "cambio":
        Yv = Yv + Xv[:, :, -1].unsqueeze(1)
    ens = _muestrear(m, Xv, objetivo, S).numpy()
    obs = Yv.numpy()
    cen = ens.mean(axis=1, keepdims=True)
    dev = ens - cen
    a = (1 - nivel) / 2
    cob = []
    for lam in REJILLA:
        e = cen + lam * dev
        lo = np.quantile(e, a, axis=1); hi = np.quantile(e, 1 - a, axis=1)
        cob.append(((obs >= lo) & (obs <= hi)).mean())
    return float(REJILLA[int(np.argmin(np.abs(np.array(cob) - nivel)))])


def evaluar(panel, nombre, semilla, cfg, objetivo, inflar):
    NO, M, refit = cfg["N_ORIGENES"], cfg["M"], cfg["REFIT"]
    ens = np.empty((NO, H_FUT, N, M)); y = np.empty((NO, H_FUT, N))
    sc = np.empty((NO, N)); cam = np.diff(panel, axis=1)
    # estadisticos de NIVELES por origen: los exige la convencion de ES de las
    # lineas base, que es la unica comparable con el umbral de H2
    mu_niv = np.empty((NO, N)); sd_niv = np.empty((NO, N))
    m = mu = sd = None; lam = 1.0; lams = []; regs = []

    for o_i in range(NO):
        origen = cfg["T0"] + o_i
        if o_i % refit == 0:
            m, mu, sd, Xva, Yva, reg = entrenar(panel, origen, nombre, semilla,
                                                cfg, objetivo,
                                                reservar_val=inflar)
            regs.append(reg)
            if inflar:
                lam = calibrar_lambda(m, Xva, Yva, objetivo, S=M)
                lams.append(lam)
        sc[o_i] = cam[:, :origen].std(axis=1)
        mu_niv[o_i] = panel[:, :origen].mean(axis=1)
        sd_niv[o_i] = panel[:, :origen].std(axis=1, ddof=1)
        with torch.no_grad():
            Xn = (panel - mu) / sd
            xw = torch.tensor(Xn[:, origen - L + 1: origen + 1][None],
                              dtype=torch.float32)
            Y = _muestrear(m, xw, objetivo, M)[0].numpy()     # [S,H,N]
            if inflar:
                c = Y.mean(axis=0, keepdims=True)
                Y = c + lam * (Y - c)
            Y = Y * sd.T[None, :, :] + mu.T[None, :, :]
            ens[o_i] = Y.transpose(1, 2, 0)
        y[o_i] = panel[:, origen + 1: origen + 1 + H_FUT].T
    return ens, y, sc, lams, regs, mu_niv, sd_niv


def factor_hln(n, h):
    """Factor de Harvey-Leybourne-Newbold.  Se anula cuando
    n + 1 - 2h + h(h-1)/n = 0, lo que ocurre exactamente en n = h = 12. Con
    ese valor dm_hln sale 0 y p sale 0.5 para CUALQUIER diferencia: un
    resultado aparentemente valido y completamente vacio. Por eso se verifica
    antes de llamar a la prueba y no despues."""
    return np.sqrt(max((n + 1 - 2 * h + h * (h - 1) / n) / n, 0.0))


def dm_seguro(la, lb, h, factor_min=0.25):
    """Diebold-Mariano con verificacion de aplicabilidad. Devuelve None si el
    numero de origenes es insuficiente frente al horizonte."""
    n = len(la)
    f = factor_hln(n, h)
    if n < 8 or f < factor_min:
        return None, (f"n={n} frente a h={h}: factor HLN {f:.3f} < {factor_min}. "
                      f"La prueba no es aplicable; hacen falta mas origenes.")
    return diebold_mariano(la, lb, h=h), None


def cobertura(ens, y, nivel=0.90):
    a = (1 - nivel) / 2
    lo = np.quantile(ens, a, axis=-1); hi = np.quantile(ens, 1 - a, axis=-1)
    d = (y >= lo) & (y <= hi)
    return float(d.mean()), d.mean(axis=(0, 2)).tolist(), d.mean(axis=(0, 1)).tolist()


# ===========================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default=None,
                    help="omitir para el panel sintetico (C3)")
    ap.add_argument("--columnas", nargs="*", default=None)
    ap.add_argument("--objetivo", default="cambio", choices=["nivel", "cambio"])
    ap.add_argument("--inflar", action="store_true",
                    help="correccion de varianza calibrada fuera de muestra")
    ap.add_argument("--rapido", action="store_true")
    ap.add_argument("--cabezas", nargs="*", default=None,
                    help="subconjunto, p.ej. R0-G dif-y")
    a = ap.parse_args()

    real = a.panel is not None
    cfg = dict(CFG_REAL if real else CFG_SINT)
    if a.rapido:
        cfg.update(CFG_RAPIDO)
    semillas = (0,) if a.rapido else SEMILLAS
    nombres = a.cabezas or list(fabricas(cfg["T_DIF"]))
    if REFERENCIA not in nombres:
        nombres = [REFERENCIA] + nombres

    ts = datetime.now().isoformat(timespec="seconds")
    env = {"python": sys.version.split()[0], "torch": torch.__version__,
           "numpy": np.__version__, "plataforma": platform.platform(),
           "hilos": torch.get_num_threads()}

    print("=" * 78)
    print(f"CONTRASTE DE H2  -  {'PANEL REAL (C4)' if real else 'PANEL SINTETICO (C3)'}"
          f"{'  [RAPIDO]' if a.rapido else ''}")
    print("=" * 78)
    print(f"  {ts}")
    print(f"  torch {env['torch']} | numpy {env['numpy']} | hilos {env['hilos']}")
    print(f"  objetivo: {a.objetivo}   correccion de varianza: "
          f"{'si' if a.inflar else 'no'}")
    print(f"  condicionante: codificador completo en TODAS las cabezas\n")

    if real:
        from h1_real import cargar_panel
        panel = cargar_panel(a.panel, a.columnas)
    else:
        panel = _panel_sintetico(T=T_SINT, seed=12345)
    T = panel.shape[1]
    if cfg["T0"] + cfg["N_ORIGENES"] + H_FUT > T:
        raise SystemExit(f"panel de {T} meses insuficiente para T0={cfg['T0']}, "
                         f"{cfg['N_ORIGENES']} origenes y H={H_FUT}")
    print(f"  panel: {T} meses   T0={cfg['T0']}   origenes={cfg['N_ORIGENES']}"
          f"   M={cfg['M']}   pasos={cfg['STEPS']}   semillas={semillas}\n")

    salida = {"timestamp": ts, "entorno": env, "panel": "real" if real else
              "sintetico", "panel_meses": int(T), "objetivo": a.objetivo,
              "inflar": a.inflar, "referencia": REFERENCIA,
              "config": {k: v for k, v in cfg.items()},
              "semillas": list(semillas),
              "nota": "exploratorio; el umbral de exito de H2 se lee del informe",
              "cabezas": {}, "perdida_por_origen": {}}

    print("-" * 78)
    print(f"  {'cabeza':<9}{'semilla':>8}{'CRPS':>9}{'ES cam':>9}{'ES niv':>9}"
          f"{'cob90':>8}{'cob50':>8}{'lambda':>8}{'seg':>7}")
    print("  " + "-" * 73)
    print(f"  {'':<9}{'':>8}{'':>9}{'(H1)':>9}{'(H2)':>9}\n")
    t_ini = time.time()
    perdidas = {}
    es_origen = {}          # cabeza -> semilla -> ES por origen (conv. niveles)
    for nom in nombres:
        salida["cabezas"][nom] = {"semillas": {}}
        perdidas[nom] = {}
        for s in semillas:
            t0 = time.time()
            try:
                ens, y, sc, lams, regs, mu_niv, sd_niv = evaluar(
                    panel, nom, s, cfg, a.objetivo, a.inflar)
            except Exception as e:
                print(f"  {nom:<9}{s:>8}   FALLA: {type(e).__name__}: {e}")
                salida["cabezas"][nom]["semillas"][str(s)] = {"error": str(e)}
                continue
            c_oh = joint_crps(ens, y, sc, standardize=True)      # [O,H]
            crps = float(c_oh.mean())
            # serie de ES por origen en convencion de NIVELES: la exige el
            # Diebold-Mariano de H2 contra la linea base, que se contrasta
            # sobre Energy Score y no sobre CRPS
            es_oh = energy_score_lineabase(ens, y, mu_niv, sd_niv)   # [O,H]
            es = float(energy_score(ens, y, sc).mean())           # conv. cambios
            es_lb = float(energy_score_lineabase(ens, y, mu_niv, sd_niv).mean())
            c90, c90h, c90n = cobertura(ens, y, .90)
            c50, _, _ = cobertura(ens, y, .50)
            perdidas[nom][s] = c_oh.mean(axis=1)
            es_origen.setdefault(nom, {})[str(s)] = es_oh.mean(axis=1).tolist()
            lam_txt = f"{np.mean(lams):.2f}" if lams else "--"
            print(f"  {nom:<9}{s:>8}{crps:>9.4f}{es:>9.4f}{es_lb:>9.4f}"
                  f"{c90:>8.3f}{c50:>8.3f}{lam_txt:>8}{time.time()-t0:>7.0f}")
            salida["cabezas"][nom]["semillas"][str(s)] = {
                "crps": crps, "energy_score_cambios": es,
                "energy_score_niveles": es_lb, "cob_90": c90, "cob_50": c50,
                "cob_90_por_horizonte": c90h, "cob_90_por_serie": c90n,
                "lambdas": lams, "params": regs[0]["params"],
                "perdida_final": regs[0]["perdida_final"]}
        vals = [v["crps"] for v in salida["cabezas"][nom]["semillas"].values()
                if "crps" in v]
        if vals:
            salida["cabezas"][nom]["crps_medio"] = float(np.mean(vals))
            salida["cabezas"][nom]["crps_desv"] = float(np.std(vals))
        vals_es = [v["energy_score_niveles"]
                   for v in salida["cabezas"][nom]["semillas"].values()
                   if "energy_score_niveles" in v]
        if vals_es:
            salida["cabezas"][nom]["es_niveles_medio"] = float(np.mean(vals_es))
        print("  " + "-" * 73)

    # --------------------------------------------------------------- DM
    print(f"\n  Diebold-Mariano contra {REFERENCIA}  (HAC NW h-1=11, HLN)")
    print(f"  H0: no hay diferencia    H1: la cabeza es mejor que {REFERENCIA}\n")
    print(f"  {'cabeza':<9}{'semilla':>9}{'dif. media':>13}{'DM (HLN)':>11}{'p':>9}")
    print("  " + "-" * 51)
    salida["diebold_mariano"] = {}
    for nom in nombres:
        if nom == REFERENCIA or not perdidas.get(nom):
            continue
        salida["diebold_mariano"][nom] = {"por_semilla": {}}
        comunes = sorted(set(perdidas[REFERENCIA]) & set(perdidas[nom]))
        for s in comunes:
            r, aviso = dm_seguro(perdidas[REFERENCIA][s], perdidas[nom][s], H_FUT)
            if r is None:
                print(f"  {nom:<9}{s:>9}   DM NO APLICABLE: {aviso}")
                salida["diebold_mariano"][nom]["no_aplicable"] = aviso
                continue
            print(f"  {nom:<9}{s:>9}{r['mean_diff']:>13.4f}"
                  f"{r['dm_hln']:>11.3f}{r['p_one_sided']:>9.4f}")
            salida["diebold_mariano"][nom]["por_semilla"][str(s)] = r
        if len(comunes) > 1:
            la = np.mean([perdidas[REFERENCIA][s] for s in comunes], axis=0)
            lb = np.mean([perdidas[nom][s] for s in comunes], axis=0)
            r, aviso = dm_seguro(la, lb, H_FUT)
            if r is None:
                print(f"  {nom:<9}{'promedio':>9}   DM NO APLICABLE: {aviso}")
            else:
                print(f"  {nom:<9}{'promedio':>9}{r['mean_diff']:>13.4f}"
                      f"{r['dm_hln']:>11.3f}{r['p_one_sided']:>9.4f}")
                salida["diebold_mariano"][nom]["promedio_semillas"] = r
        print("  " + "-" * 51)

    salida["perdida_por_origen"] = {
        n: {str(s): v.tolist() for s, v in d.items()} for n, d in perdidas.items()}
    salida["es_niveles_por_origen"] = es_origen

    # ---------------------------------------------------------- lectura
    print("\n" + "=" * 78)
    print("LECTURA")
    print("=" * 78)
    ordenadas = sorted(((n, salida["cabezas"][n].get("crps_medio", np.inf))
                        for n in nombres), key=lambda t: t[1])
    print("  orden por CRPS: " + "  <  ".join(f"{n} ({c:.4f})"
                                              for n, c in ordenadas))
    ref = salida["cabezas"][REFERENCIA].get("crps_medio", np.nan)
    mejores = [n for n in salida.get("diebold_mariano", {})
               if salida["diebold_mariano"][n].get(
                   "promedio_semillas", {}).get("p_one_sided", 1) < 0.05]
    print()
    if mejores:
        print(f"  Vencen a {REFERENCIA} con p < 0.05: {', '.join(mejores)}")
        print("  Verificar contra el umbral de H2 comprometido en el informe")
        print("  ANTES de declarar exito; este guion no lo fija.")
    else:
        print(f"  Ninguna cabeza vence a {REFERENCIA} con p < 0.05.")
        if not real:
            print("  C3 no se supera: NO corresponde pasar al panel real.")
            print("  Congelar con R0 es la decision que la escalera indica.")
        else:
            print("  Se reporta como tal: la degradacion controlada a la cabeza")
            print("  parametrica queda justificada por evidencia, no por falta")
            print("  de tiempo.")
    # --------------------------------------------- veredicto de H2
    if real:
        print("\n" + "-" * 78)
        print("H2 contra la mejor linea base  (convencion de NIVELES)")
        print("-" * 78)
        print(f"  ARIMA/GARCH: ES = {ES_LINEABASE:.3f}    "
              f"umbral de H2 (-10 %): ES <= {ES_UMBRAL_H2:.3f}\n")
        for nom in nombres:
            e = salida["cabezas"][nom].get("es_niveles_medio")
            if e is None:
                continue
            mej = 1.0 - e / ES_LINEABASE
            cumple = e <= ES_UMBRAL_H2
            print(f"  {nom:<9}ES = {e:7.4f}   {100*mej:+6.1f} % vs linea base   "
                  f"{'CUMPLE el umbral' if cumple else 'no alcanza el umbral'}")
            salida["cabezas"][nom]["h2"] = {
                "es_niveles": e, "mejora_vs_lineabase": mej,
                "cumple_umbral": bool(cumple)}
        print("\n  El umbral es condicion NECESARIA y no suficiente: H2 exige")
        print("  ademas significancia al 5 % en Diebold-Mariano contra la serie")
        print("  de perdidas por origen de ARIMA/GARCH, que este guion no tiene.")
        print("  Hay que exportarla desde MIA_Evaluacion_LineasBase.ipynb.")

    print(f"\n  dispersion entre semillas: " + "  ".join(
        f"{n} {salida['cabezas'][n].get('crps_desv', float('nan')):.4f}"
        for n in nombres))
    print(f"  piso de ruido referencial: ~4 %")
    print(f"\n  tiempo total: {(time.time()-t_ini)/60:.1f} min")

    modo = "rapido" if a.rapido else "full"
    p = Path(f"h2_{'real' if real else 'sintetico'}_{a.objetivo}"
             f"{'_inflado' if a.inflar else ''}_{modo}_{ts[:10]}.json")
    p.write_text(json.dumps(salida, indent=2, ensure_ascii=False))
    print(f"  guardado en {p}")


if __name__ == "__main__":
    main()
