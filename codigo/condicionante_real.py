"""
condicionante_real.py  -  Bloque 1 del martes 13: replicar sobre el panel real
===============================================================================

QUE CONTRASTA

El 12 de octubre, sobre el panel sintetico con causalidad plantada, se midio:

  condicionante                        R^2 estado   CRPS   cob 90 %
  enc = transformer(3 capas) + h.mean()   -5.188    3.036    0.347
  lin = Linear(45, 32) de la ventana       0.995    2.568    0.454
  ult = Linear(3, 32) del ultimo mes       1.000    1.635    0.549
  oraculo del generador                        --   1.114    0.921

Es decir: condicionar en TRES NUMEROS vence al transformer completo por 46 %
de CRPS, y z_enc tiene R^2 NEGATIVO para recuperar el estado del sistema.

Este script responde una sola pregunta: ¿pasa lo mismo en el panel real?

  - Si `ult` tambien vence a `enc`, el diagnostico es general y entra al
    informe como mecanismo del rechazo de H1.
  - Si NO lo vence, es un artefacto del generador sintetico y se reporta
    como tal. Ese resultado tambien sirve: acota el alcance del hallazgo.

-------------------------------------------------------------------------------
ESTO NO ES EL CONTRASTE PRERREGISTRADO
-------------------------------------------------------------------------------
Es analisis EXPLORATORIO POSTERIOR. El rechazo de H1 queda exactamente como
esta reportado: su regla se fijo antes de ejecutar y se cumplio. Esto agrega
el mecanismo, no reescribe el resultado. Cualquier re-contraste de H1 sobre
una arquitectura corregida seria un experimento NUEVO, con prerregistro y
fecha propios.

-------------------------------------------------------------------------------
AISLAMIENTO
-------------------------------------------------------------------------------
Corre en TU maquina. El panel nunca sale de ella. Lo que se comparte para
interpretar son los agregados del JSON, no los datos.

-------------------------------------------------------------------------------
ADAPTACION AL PANEL REAL
-------------------------------------------------------------------------------
En el panel sintetico el R^2 se mide contra el estado VERDADERO del generador,
que aqui no existe. El sustituto es exacto en espiritu: se regresa el ULTIMO
MES OBSERVADO de cada serie (normalizado) sobre z. Es la informacion que el
condicionante tiene literalmente en la entrada; si no puede reproducirla, el
diagnostico es el mismo. Para `ult` el R^2 sera ~1 por construccion --- es la
prueba de cordura del test, no un resultado.

-------------------------------------------------------------------------------
USO
-------------------------------------------------------------------------------
    python condicionante_real.py --panel panel_2003_2024.csv
    python condicionante_real.py --panel panel.csv --rapido     # verificar
    python condicionante_real.py --panel panel.csv \
        --columnas cobre usdclp tpm                             # fijar orden

Requiere en el mismo directorio: bloques_h1.py, cabeza_r0.py, h1_harness.py,
h1_real.py.  Tiempo estimado: ~25 min en configuracion completa (3 vias x
109 origenes x 10 reajustes), ~3 min con --rapido.
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

torch.set_num_threads(1)     # el orden de reduccion de BLAS cambia resultados

from bloques_h1 import (Phase1Tokenizer, Block, causal_mask_modL,
                        ventanas_multi, L, H_FUT, N, D, EPS, NL)
from cabeza_r0 import CabezaR0
from h1_harness import joint_crps, diebold_mariano
from h1_real import cargar_panel, verificar_aritmetica, ORIGEN_INICIAL

N_ORIGENES = 109
M_SAMPLES = 200
REFIT_CADA = 12
STEPS = 400
SEMILLAS = (0, 1, 2)
SERIES = ("cobre", "USD/CLP", "TPM")

RAPIDO = dict(N_ORIGENES=24, M_SAMPLES=80, REFIT_CADA=12, STEPS=150,
              SEMILLAS=(0,))


# ===========================================================================
# Las tres vias de condicionamiento
# ===========================================================================
class CondEnc(nn.Module):
    """El transformer de los cuadernos: tokenizador, NL bloques, mascara
    mod-L, y z = media sobre los 45 tokens. Es lo que hay hoy."""

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
    """La ventana cruda aplanada -> un solo Linear(45, 32). Sin atencion,
    sin capas, sin no linealidad. Deliberadamente lo mas pobre posible."""

    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(N * L, D)

    def forward(self, X):
        return self.lin(X.reshape(X.shape[0], -1))


class CondUltimo(nn.Module):
    """Solo el ultimo mes observado de cada serie: 3 numeros."""

    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(N, D)

    def forward(self, X):
        return self.lin(X[:, :, -1])


FABRICAS = {"enc": CondEnc, "lin": CondLineal, "ult": CondUltimo}
ETIQUETA = {"enc": "transformer + media de tokens",
            "lin": "Linear(45,32) de la ventana",
            "ult": "Linear(3,32) del ultimo mes"}


class Modelo(nn.Module):
    def __init__(self, via):
        super().__init__()
        self.cond = FABRICAS[via]()
        # misma cabeza en las tres vias: lo unico que cambia es el condicionante
        self.cabeza = CabezaR0(latent_dim=D, rank=5, dist_type="gaussian",
                               v_condicional=False)


# ===========================================================================
def _datos(panel, t_fin, objetivo="nivel"):
    """Normaliza con estadisticos hasta t_fin UNICAMENTE. Sin fuga.

    objetivo='nivel'   Y = Xn[o+1 : o+1+H]            (lo original)
    objetivo='cambio'  Y = Xn[o+1 : o+1+H] - Xn[o]    (cambio desde el origen)

    El modo 'cambio' existe por lo observado el 13-10-2026 en el panel real:
    la cobertura del USD/CLP dio 0.000 en las TRES vias de condicionamiento,
    es decir desplazamiento sistematico y no sub-dispersion. Entrenar sobre
    niveles normalizados con media hasta el origen deja a esa media siempre
    por detras en una serie con tendencia, y el pronostico queda corrido.
    Trabajar sobre el cambio desde el origen elimina esa dependencia del
    nivel. Es la prueba directa de esa hipotesis.
    """
    mu = panel[:, :t_fin].mean(axis=1, keepdims=True)
    sd = panel[:, :t_fin].std(axis=1, keepdims=True) + EPS
    X, Y = ventanas_multi((panel - mu) / sd, L, H_FUT, t_fin)
    if objetivo == "cambio":
        Y = Y - X[:, :, -1].unsqueeze(1)             # [B,H,N] - [B,1,N]
    return X, Y, mu, sd


def entrenar(panel, t_fin, via, semilla, steps, batch=64, lr=1e-3,
             objetivo="nivel"):
    torch.manual_seed(semilla)
    np.random.seed(semilla)
    X, Y, mu, sd = _datos(panel, t_fin, objetivo)
    m = Modelo(via)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    m.train()
    p0 = p1 = None
    for st in range(steps):
        i = torch.randint(0, len(X), (min(batch, len(X)),))
        loss = m.cabeza.nll(m.cond(X[i]), Y[i])
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        if st == 0:
            p0 = float(loss.detach())
        p1 = float(loss.detach())
    m.eval()
    return m, mu, sd, {"nll_inicial": p0, "nll_final": p1,
                       "n_ventanas": int(len(X)),
                       "params": sum(p.numel() for p in m.parameters())}


def r2_ultimo_mes(panel, via, semilla, steps, objetivo="nivel"):
    """R^2 fuera de muestra al regresar linealmente el ultimo mes observado
    (normalizado) sobre z. Particion 70/30 por orden temporal.

    El objetivo IMPORTA y en la primera version no se pasaba: el R^2 salia
    siempre medido sobre un modelo entrenado en niveles, de modo que las dos
    corridas daban cifras identicas y la del modo 'cambio' no significaba lo
    que decia. Corregido el 13-10-2026.
    """
    m, mu, sd, reg = entrenar(panel, ORIGEN_INICIAL, via, semilla, steps,
                              objetivo=objetivo)
    X, Y, *_ = _datos(panel, ORIGEN_INICIAL, objetivo)
    with torch.no_grad():
        Z = m.cond(X).numpy()
    U = X[:, :, -1].numpy()                       # [B, 3] ultimo mes de cada serie
    n = len(Z); k = int(.7 * n)
    Ztr = np.c_[Z[:k], np.ones(k)]
    Zte = np.c_[Z[k:], np.ones(n - k)]
    coef, *_ = np.linalg.lstsq(Ztr, U[:k], rcond=None)
    ss_res = ((U[k:] - Zte @ coef) ** 2).sum(0)
    ss_tot = ((U[k:] - U[:k].mean(0)) ** 2).sum(0)
    return 1 - ss_res / np.maximum(ss_tot, 1e-12), reg


def cobertura(ens, y, nivel=0.90):
    a = (1 - nivel) / 2
    lo = np.quantile(ens, a, axis=-1)
    hi = np.quantile(ens, 1 - a, axis=-1)
    dentro = (y >= lo) & (y <= hi)
    return float(dentro.mean()), dentro.mean(axis=(0, 2)).tolist(), \
        dentro.mean(axis=(0, 1)).tolist()


def evaluar(panel, via, semilla, cfg, objetivo="nivel"):
    """Ventana expansiva del prerregistro §5, con la via como argumento."""
    NO, M, refit, steps = (cfg["N_ORIGENES"], cfg["M_SAMPLES"],
                           cfg["REFIT_CADA"], cfg["STEPS"])
    ens = np.empty((NO, H_FUT, N, M))
    y = np.empty((NO, H_FUT, N))
    scale = np.empty((NO, N))
    cambios = np.diff(panel, axis=1)
    m = mu = sd = None
    for o_i in range(NO):
        origen = ORIGEN_INICIAL + o_i
        if o_i % refit == 0:
            m, mu, sd, _ = entrenar(panel, origen, via, semilla, steps,
                                    objetivo=objetivo)
        scale[o_i] = cambios[:, :origen].std(axis=1)
        with torch.no_grad():
            Xn = (panel - mu) / sd
            xw = torch.tensor(Xn[:, origen - L + 1: origen + 1][None],
                              dtype=torch.float32)
            Y = m.cabeza.predecir(m.cond(xw), S=M).numpy()    # [M,H,N]
            if objetivo == "cambio":
                Y = Y + Xn[:, origen][None, None, :]          # + nivel normalizado
            Y = Y * sd.T[None, :, :] + mu.T[None, :, :]       # -> niveles
            ens[o_i] = Y.transpose(1, 2, 0)
        y[o_i] = panel[:, origen + 1: origen + 1 + H_FUT].T
    return ens, y, scale


def sesgo_relativo(ens, y, scale):
    """(mediana del pronostico - realizado) / escala de los cambios, por serie.
    Separa DESPLAZAMIENTO de SUB-DISPERSION: con cobertura baja por intervalos
    estrechos este numero ronda cero; con cobertura cero es grande y de signo
    constante."""
    med = np.median(ens, axis=-1)                              # [O,H,N]
    return ((med - y) / scale[:, None, :]).mean(axis=(0, 1))   # [N]


# ===========================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", required=True, help=".csv o .npy con el panel")
    ap.add_argument("--columnas", nargs="*", default=None,
                    help="orden de las series, p.ej. cobre usdclp tpm")
    ap.add_argument("--rapido", action="store_true")
    ap.add_argument("--objetivo", default="nivel", choices=["nivel", "cambio"],
                    help="'nivel' replica la configuracion de los cuadernos; "
                         "'cambio' entrena sobre el cambio desde el origen")
    a = ap.parse_args()

    cfg = dict(N_ORIGENES=N_ORIGENES, M_SAMPLES=M_SAMPLES,
               REFIT_CADA=REFIT_CADA, STEPS=STEPS, SEMILLAS=SEMILLAS)
    if a.rapido:
        cfg.update(RAPIDO)

    ts = datetime.now().isoformat(timespec="seconds")
    env = {"python": sys.version.split()[0], "torch": torch.__version__,
           "numpy": np.__version__, "plataforma": platform.platform(),
           "hilos": torch.get_num_threads()}

    print("=" * 78)
    print(f"VIA DE CONDICIONAMIENTO  -  PANEL REAL"
          f"{'  [RAPIDO]' if a.rapido else ''}")
    print("=" * 78)
    print(f"  {ts}")
    print(f"  torch {env['torch']} | numpy {env['numpy']} | hilos {env['hilos']}")
    print("  ATENCION: resultados de otra version de torch NO son comparables")
    print("  Analisis EXPLORATORIO POSTERIOR; no es el contraste prerregistrado")
    print(f"  objetivo de entrenamiento: {a.objetivo}\n")

    panel = cargar_panel(a.panel, a.columnas)
    T = panel.shape[1]
    if not a.rapido:
        verificar_aritmetica(T)
    else:
        print(f"  meses del panel: {T}   (modo rapido: {cfg['N_ORIGENES']} origenes)")
    if cfg["N_ORIGENES"] + ORIGEN_INICIAL + H_FUT > T:
        raise ValueError("no caben los origenes pedidos en este panel")
    print(f"\n  origenes={cfg['N_ORIGENES']}  M={cfg['M_SAMPLES']}  "
          f"reajuste cada {cfg['REFIT_CADA']}  pasos={cfg['STEPS']}  "
          f"semillas={cfg['SEMILLAS']}\n")

    salida = {"timestamp": ts, "entorno": env, "panel_meses": int(T),
              "objetivo": a.objetivo,
              "config": {k: (list(v) if isinstance(v, tuple) else v)
                         for k, v in cfg.items()},
              "nota": "exploratorio posterior; no es el contraste prerregistrado",
              "R2": {}, "desempeno": {}}

    # ----------------------------------------------------------- parte 1
    print("-" * 78)
    print("1  Informacion del ultimo mes observado contenida en z")
    print("-" * 78)
    print("  R^2 fuera de muestra al regresar el ultimo mes (normalizado)")
    print("  sobre z. `ult` debe dar ~1: es la prueba de cordura del test.\n")
    print(f"  {'via':<6}{'cobre':>9}{'USD/CLP':>10}{'TPM':>9}{'medio':>10}"
          f"{'parametros':>12}")
    print("  " + "-" * 56)
    for via in ("enc", "lin", "ult"):
        r2, reg = r2_ultimo_mes(panel, via, cfg["SEMILLAS"][0], cfg["STEPS"],
                                objetivo=a.objetivo)
        print(f"  {via:<6}{r2[0]:>9.3f}{r2[1]:>10.3f}{r2[2]:>9.3f}"
              f"{r2.mean():>10.3f}{reg['params']:>12}")
        salida["R2"][via] = {"por_serie": r2.tolist(),
                             "medio": float(r2.mean()),
                             "params": reg["params"],
                             "n_ventanas": reg["n_ventanas"]}
    print()

    # ----------------------------------------------------------- parte 2
    print("-" * 78)
    print("2  Desempeno predictivo, misma cabeza R0-G")
    print("-" * 78)
    print(f"  {'via':<6}{'semilla':>8}{'CRPS':>10}{'cob 90%':>9}"
          f"{'cob 80%':>9}{'cob 50%':>9}{'seg':>7}")
    print("  " + "-" * 58)
    t_ini = time.time()
    alarmas = []
    perdidas = {}          # via -> semilla -> perdida por origen, para DM
    for via in ("enc", "lin", "ult"):
        salida["desempeno"][via] = {"etiqueta": ETIQUETA[via], "semillas": {}}
        for s in cfg["SEMILLAS"]:
            t0 = time.time()
            ens, y, sc = evaluar(panel, via, s, cfg, objetivo=a.objetivo)
            crps = float(joint_crps(ens, y, sc, standardize=True).mean())
            c90, c90h, c90n = cobertura(ens, y, .90)
            c80, _, _ = cobertura(ens, y, .80)
            c50, _, _ = cobertura(ens, y, .50)
            c_oh = joint_crps(ens, y, sc, standardize=True)   # [O, H]
            crps_n = c_oh.mean(axis=0)
            perdidas.setdefault(via, {})[s] = c_oh.mean(axis=1)  # [O] por origen
            ses = sesgo_relativo(ens, y, sc)
            print(f"  {via:<6}{s:>8}{crps:>10.4f}{c90:>9.3f}{c80:>9.3f}"
                  f"{c50:>9.3f}{time.time()-t0:>7.0f}")
            print(f"  {'':<6}{'':>8}  cob 90 % por serie: "
                  + "  ".join(f"{n} {v:.3f}" for n, v in zip(SERIES, c90n)))
            print(f"  {'':<6}{'':>8}  sesgo (med-real)/esc: "
                  + "  ".join(f"{n} {v:+.2f}" for n, v in zip(SERIES, ses)))
            for n, v in zip(SERIES, c90n):
                if v < 0.05:
                    alarmas.append((via, s, n, v))
            salida["desempeno"][via]["semillas"][str(s)] = {
                "crps": crps, "cob_90": c90, "cob_80": c80, "cob_50": c50,
                "cob_90_por_horizonte": c90h, "cob_90_por_serie": c90n,
                "sesgo_relativo_por_serie": ses.tolist(),
                "crps_por_horizonte": crps_n.tolist()}
        cs = [v["crps"] for v in salida["desempeno"][via]["semillas"].values()]
        salida["desempeno"][via]["crps_medio"] = float(np.mean(cs))
        salida["desempeno"][via]["crps_desv"] = float(np.std(cs))
    print()

    if alarmas:
        print("  " + "!" * 62)
        print("  COBERTURA PRACTICAMENTE NULA en:")
        for via, s, n, v in alarmas:
            print(f"    via {via}, semilla {s}, serie {n}: {v:.4f}")
        print("  Una cobertura de 0.00 no es sub-dispersion ---con intervalos")
        print("  estrechos se esperaria 0.2 o 0.3--- sino DESPLAZAMIENTO: la")
        print("  distribucion predictiva cae fuera de donde estan los")
        print("  realizados. Mirar la fila de sesgo: si es grande y de signo")
        print("  constante, el modelo esta corrido, no mal calibrado.")
        print("  Contrastar con  --objetivo cambio.")
        print("  " + "!" * 62 + "\n")
        salida["alarmas_cobertura_nula"] = [
            {"via": v, "semilla": s, "serie": n, "cobertura": c}
            for v, s, n, c in alarmas]

    # ------------------------------------------------- Diebold-Mariano
    # Una diferencia porcentual entre medias NO es evidencia. El protocolo
    # del proyecto exige DM con HAC, y aqui ademas corresponde porque las
    # tres vias comparten los MISMOS origenes: la comparacion pareada es
    # mucho mas potente que comparar medias entre semillas.
    # H0: E[L_enc - L_B] = 0   contra   H1: > 0  (es decir, B mejor que enc)
    print("-" * 78)
    print("3  Diebold-Mariano contra `enc`  (HAC Newey-West, correccion HLN)")
    print("-" * 78)
    print("  H0: no hay diferencia   H1: la via es mejor que `enc`")
    print(f"  {'via':<6}{'semilla':>9}{'dif. media':>13}{'DM (HLN)':>11}{'p':>9}")
    print("  " + "-" * 48)
    salida["diebold_mariano"] = {}
    for via in ("lin", "ult"):
        salida["diebold_mariano"][via] = {"por_semilla": {}}
        for s in cfg["SEMILLAS"]:
            try:
                r = diebold_mariano(perdidas["enc"][s], perdidas[via][s], h=H_FUT)
            except Exception as e:                       # n<8 en modo rapido
                print(f"  {via:<6}{s:>9}   DM no aplicable: {e}")
                continue
            print(f"  {via:<6}{s:>9}{r['mean_diff']:>13.4f}"
                  f"{r['dm_hln']:>11.3f}{r['p_one_sided']:>9.4f}")
            salida["diebold_mariano"][via]["por_semilla"][str(s)] = r
        # pareado sobre la perdida promediada entre semillas
        if len(cfg["SEMILLAS"]) > 1:
            la = np.mean([perdidas["enc"][s] for s in cfg["SEMILLAS"]], axis=0)
            lb = np.mean([perdidas[via][s] for s in cfg["SEMILLAS"]], axis=0)
            try:
                r = diebold_mariano(la, lb, h=H_FUT)
                print(f"  {via:<6}{'promedio':>9}{r['mean_diff']:>13.4f}"
                      f"{r['dm_hln']:>11.3f}{r['p_one_sided']:>9.4f}")
                salida["diebold_mariano"][via]["promedio_semillas"] = r
            except Exception as e:
                print(f"  {via:<6}{'promedio':>9}   DM no aplicable: {e}")
    print("\n  Una p < 0.05 sostiene que la via vence a `enc`. Una p alta con")
    print("  diferencia de medias grande significa que la diferencia no se")
    print("  distingue del ruido una vez contada la autocorrelacion de los")
    print("  pronosticos solapados, y debe reportarse como no concluyente.\n")
    salida["perdida_por_origen"] = {
        v: {str(s): perdidas[v][s].tolist() for s in cfg["SEMILLAS"]}
        for v in perdidas}

    # ----------------------------------------------------------- lectura
    print("=" * 78)
    print("LECTURA")
    print("=" * 78)
    c_enc = salida["desempeno"]["enc"]["crps_medio"]
    c_lin = salida["desempeno"]["lin"]["crps_medio"]
    c_ult = salida["desempeno"]["ult"]["crps_medio"]
    dev = max(salida["desempeno"][v]["crps_desv"] for v in ("enc", "lin", "ult"))
    print(f"  CRPS medio   enc {c_enc:.4f}   lin {c_lin:.4f}   ult {c_ult:.4f}")
    print(f"  ult vs enc: {100*(c_ult/c_enc-1):+.1f} %      "
          f"lin vs enc: {100*(c_lin/c_enc-1):+.1f} %")
    print(f"  dispersion maxima entre semillas: {dev:.4f}")
    print(f"  piso de ruido referencial de estas mediciones: ~4 %\n")
    r2_enc = salida["R2"]["enc"]["medio"]

    # La version anterior de este bloque juzgaba SOLO `ult` contra `enc`, y
    # con eso declaraba "no replica" en un caso donde `lin` le ganaba a `enc`
    # por 20 %. El criterio correcto es el MEJOR condicionante minimo contra
    # el codificador completo; cual de los dos gana es informacion adicional,
    # no el criterio. Corregido el 13-10-2026.
    c_min = min(c_lin, c_ult)
    cual = "lin" if c_lin <= c_ult else "ult"
    orden = " < ".join(k for k, _ in sorted(
        (("enc", c_enc), ("lin", c_lin), ("ult", c_ult)), key=lambda t: t[1]))
    print(f"  orden de desempeno: {orden}")
    print(f"  mejor condicionante minimo: `{cual}` ({c_min:.4f}), "
          f"{100*(c_min/c_enc-1):+.1f} % vs enc\n")

    if c_min < c_enc * 0.96 and r2_enc < 0.5:
        ver = "replica"
        print(f"  REPLICA. En el panel real un condicionante minimo (`{cual}`)")
        print("  vence al codificador completo, y z tampoco retiene el ultimo")
        print("  mes observado. El diagnostico es general y entra al informe")
        print("  como mecanismo del rechazo de H1.")
    elif c_min < c_enc * 0.96:
        ver = "replica_parcial"
        print(f"  REPLICA PARCIAL. `{cual}` vence a `enc`, pero z SI retiene el")
        print(f"  ultimo mes (R^2 = {r2_enc:.3f}). La via de lectura no es el")
        print("  problema aqui: queda el efecto de capacidad (parametros por")
        print("  ventana). Se reporta asi, sin extender al panel real el")
        print("  diagnostico del agregador.")
    else:
        ver = "no_replica"
        print("  NO REPLICA. En el panel real el codificador no queda por debajo")
        print("  de ningun condicionante minimo. El hallazgo del 12 de octubre")
        print("  es especifico del generador sintetico y debe reportarse con ese")
        print("  alcance. Resultado util: acota la generalidad del diagnostico.")

    if cual == "lin" and c_ult > c_enc:
        print()
        print("  OJO AL ORDEN. `ult` queda PEOR que `enc`, de modo que el")
        print("  resultado NO es 'tres numeros bastan': es que la ventana")
        print("  completa importa y el codificador no la aprovecha. Mirar la")
        print("  fila de sesgo de `ult`: sin la ventana no detecta el arrastre.")

    salida["veredicto"] = {"tipo": ver, "crps_enc": c_enc, "crps_lin": c_lin,
                           "crps_ult": c_ult, "r2_enc": r2_enc,
                           "mejor_minimo": cual, "orden": orden,
                           "objetivo": a.objetivo}

    print(f"\n  tiempo total: {(time.time()-t_ini)/60:.1f} min")
    modo = "rapido" if a.rapido else "full"
    p = Path(f"condicionante_real_{a.objetivo}_{modo}_{ts[:10]}.json")
    p.write_text(json.dumps(salida, indent=2, ensure_ascii=False))
    print(f"  guardado en {p}")
    print("\n  Para interpretar juntos basta el JSON: no contiene datos del panel.")


if __name__ == "__main__":
    main()
