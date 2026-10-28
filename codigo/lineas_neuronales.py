"""
lineas_neuronales.py  -  lineas base neuronales por sustitucion de codificador
==============================================================================

QUE RESUELVE. El avance anterior comprometio lineas base neuronales (LSTM, TFT)
y quedaron pendientes. Este guion las entrega en la unica forma que las hace
comparables: sustituyendo el CODIFICADOR y dejando intacto todo lo demas.

POR QUE ASI, Y NO CON UNA BIBLIOTECA EXTERNA
------------------------------------------------------------------------------
Una linea base solo es valida si corre bajo el MISMO protocolo. Una
implementacion de TFT tomada de pytorch-forecasting traeria su propio ciclo de
entrenamiento, su propia estandarizacion, su propia especificacion del objetivo
y su propia forma de producir cuantiles. Comparar su Energy Score contra el
nuestro seria exactamente el error de escala que ya costo un diagnostico
completo en es_escala.py: dos cifras correctas por separado y no comparables
entre si.

Aqui, en cambio, se reemplaza una sola clase ---el codificador--- y se reutiliza
literalmente el mismo codigo de h2_cabezas para todo el resto: las ventanas, la
estandarizacion por origen, el objetivo, la cabeza generativa, el muestreo, la
ventana expansiva de 109 origenes, el reajuste cada 12, las tres semillas, el
CRPS conjunto, las dos convenciones de Energy Score, la cobertura y el
Diebold-Mariano con correccion HLN. No hay una segunda ruta de evaluacion que
pueda divergir, porque no hay una segunda ruta.

La sustitucion opera asignando h2_cabezas.CondEnc, que `Modelo.__init__`
resuelve en tiempo de llamada. Por eso el codificador `transformer` de este
guion debe reproducir EXACTAMENTE las cifras de h2_cabezas.py con la misma
semilla y el mismo objetivo. Esa coincidencia es la verificacion de que la
sustitucion es fiel, y el guion la contrasta de forma explicita si se le pasa
el JSON de la corrida original con --control.

QUE SE CONTRASTA
------------------------------------------------------------------------------
    transformer   el codificador de la tesis (CONTROL, no es linea base)
    lstm          recurrente, 2 capas; la linea base neuronal comprometida
    gru           recurrente con menos parametros; variante de robustez
    mlp           sin estructura secuencial alguna; el PISO

Todos sobre la misma cabeza generativa (por defecto R0-T, la de mejor
desempeno) y con numero de parametros deliberadamente emparejado, que el guion
imprime. Un codificador con el triple de parametros no seria una ablacion sino
otro experimento.

La pregunta que responde es la que un corrector hara: *la auto-atencion, con
su costo y su complejidad, gana algo frente a una recurrente?* Es una pregunta
distinta de H1 y de H2, y se reporta como tal.

USO
------------------------------------------------------------------------------
    python lineas_neuronales.py --rapido
    python lineas_neuronales.py --panel panel_2003_2024.csv --objetivo cambio
    python lineas_neuronales.py --panel panel_2003_2024.csv --objetivo cambio \
           --control h2_real_cambio_full_2026-10-13.json

AISLAMIENTO: sin --panel no toca dato real alguno. Con --panel corre en la
maquina del tesista; el JSON de salida guarda metricas y no datos.
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

import h2_cabezas as h2c
from bloques_h1 import N, L, D, H_FUT


# ===========================================================================
# Codificadores. Contrato: forward(X: [B, N, L]) -> z: [B, D]
#
# DOS DECISIONES DE EQUIDAD, ambas verificables en la auditoria de contrato:
#
# 1. LayerNorm final. El Transformer produce z a traves de bloques que ya
#    normalizan, de modo que su z sale con sd ~0,7 al inicializar; una
#    recurrente o un MLP salen con sd ~0,08. Un factor de 8 en la escala del
#    vector que recibe la cabeza generativa es una desventaja de partida que
#    nada tiene que ver con la arquitectura. La LayerNorm final iguala la
#    escala sin anadir capacidad (2D parametros) y hace que la comparacion
#    mida lo que dice medir.
#
# 2. Paridad de parametros. Los tamanos ocultos estan elegidos para igualar el
#    recuento del Transformer (38.272) dentro de un 1 %, no por costumbre.
# ===========================================================================
class CondLSTM(nn.Module):
    """Recurrente sobre el eje temporal. X llega como [B, N, L]: se transpone
    a [B, L, N], es decir L pasos de N variables, que es la lectura natural de
    una ventana multivariada. Se toma el estado oculto de la ultima capa y se
    proyecta a D para que la cabeza generativa reciba exactamente la misma
    dimension que recibe del Transformer."""

    def __init__(self, oculto=54, capas=2):
        super().__init__()
        self.rnn = nn.LSTM(input_size=N, hidden_size=oculto,
                           num_layers=capas, batch_first=True)
        self.proj = nn.Linear(oculto, D)
        self.norm = nn.LayerNorm(D)

    def forward(self, X):
        _, (h, _) = self.rnn(X.transpose(1, 2))
        return self.norm(self.proj(h[-1]))


class CondGRU(nn.Module):
    """Variante con una compuerta menos. Util como control de robustez: si
    LSTM y GRU coinciden, la conclusion no depende de la celda concreta."""

    def __init__(self, oculto=62, capas=2):
        super().__init__()
        self.rnn = nn.GRU(input_size=N, hidden_size=oculto,
                          num_layers=capas, batch_first=True)
        self.proj = nn.Linear(oculto, D)
        self.norm = nn.LayerNorm(D)

    def forward(self, X):
        _, h = self.rnn(X.transpose(1, 2))
        return self.norm(self.proj(h[-1]))


class CondMLP(nn.Module):
    """Sin estructura secuencial ni atencion: aplana la ventana completa. Es
    el PISO de la comparacion. Si el Transformer no lo supera, el problema no
    esta en la eleccion de arquitectura secuencial sino mas atras."""

    def __init__(self, h1=192, h2=128):
        super().__init__()
        self.red = nn.Sequential(
            nn.Linear(N * L, h1), nn.GELU(),
            nn.Linear(h1, h2), nn.GELU(),
            nn.Linear(h2, D), nn.LayerNorm(D))

    def forward(self, X):
        return self.red(X.reshape(X.shape[0], N * L))


# El codificador original se captura AQUI, al importar, y no dentro de la
# lambda. Resolverlo en tiempo de llamada haria que, una vez sustituido
# h2c.CondEnc, el control se invocase a si mismo: RecursionError.
_COND_ORIGINAL = h2c.CondEnc

CODIFICADORES = {
    "transformer": lambda: _COND_ORIGINAL(),  # CONTROL
    "lstm":        lambda: CondLSTM(),
    "gru":         lambda: CondGRU(),
    "mlp":         lambda: CondMLP(),
}
CONTROL = "transformer"


def n_params(mod):
    return sum(p.numel() for p in mod.parameters() if p.requires_grad)


# ===========================================================================
def auditar_contrato(verbose=True):
    """Antes de evaluar nada: toda sustitucion debe producir [B, D] a partir
    de [B, N, L]. Un codificador que devuelva otra forma rompe la cabeza mucho
    mas adelante, con un error que no apunta al origen."""
    X = torch.randn(7, N, L)
    filas = []
    for nom, fab in CODIFICADORES.items():
        m = fab()
        z = m(X)
        ok = tuple(z.shape) == (7, D)
        filas.append((nom, n_params(m), tuple(z.shape), ok,
                      float(z.detach().std()), bool(torch.isfinite(z).all())))
        if not ok:
            raise SystemExit(f"  {nom}: devuelve {tuple(z.shape)}, se esperaba "
                             f"(7, {D}). Sustitucion invalida.")
    if verbose:
        print("-" * 78)
        print("  AUDITORIA DE CONTRATO  (dummy [7, %d, %d] -> z)" % (N, L))
        print("-" * 78)
        print(f"  {'codificador':<14}{'params':>9}{'salida':>12}"
              f"{'sd(z)':>9}{'finito':>9}")
        ref = filas[0][1]
        for nom, p, sh, ok, sd, fin in filas:
            raz = p / ref if ref else float("nan")
            print(f"  {nom:<14}{p:>9,}{str(sh):>12}{sd:>9.3f}"
                  f"{'si' if fin else 'NO':>9}   ({raz:.2f}x)")
        print()
        peor = max(f[1] for f in filas) / min(f[1] for f in filas)
        if peor > 1.6:
            print(f"  AVISO: la razon entre el mayor y el menor numero de")
            print(f"  parametros es {peor:.2f}. Por encima de ~1,6 la")
            print(f"  comparacion deja de aislar la arquitectura y empieza a")
            print(f"  medir capacidad. Ajustar `oculto` o `h1/h2`.\n")
    return {n: p for n, p, _, _, _, _ in filas}


def verificar_control(salida, ruta_control, objetivo):
    """El codificador `transformer` de este guion debe reproducir las cifras de
    h2_cabezas.py. Si no lo hace, la sustitucion introdujo una diferencia y
    ninguna de las comparaciones siguientes es interpretable."""
    try:
        orig = json.loads(Path(ruta_control).read_text(encoding="utf-8"))
    except Exception as e:
        print(f"  no se pudo leer {ruta_control}: {e}\n")
        return
    if orig.get("objetivo") != objetivo:
        print(f"  AVISO: el control es de objetivo '{orig.get('objetivo')}' y")
        print(f"  esta corrida es '{objetivo}'. No son comparables.\n")
        return
    cab = salida["codificadores"].get(CONTROL, {}).get("semillas", {})
    ref = orig.get("cabezas", {}).get(salida["cabeza"], {}).get("semillas", {})
    if not ref:
        print(f"  el control no contiene la cabeza {salida['cabeza']}.\n")
        return
    print("-" * 78)
    print("  VERIFICACION DE FIDELIDAD DE LA SUSTITUCION")
    print("-" * 78)
    print(f"  {'semilla':>9}{'CRPS aqui':>12}{'CRPS orig':>12}{'dif rel':>11}")
    peor = 0.0
    for s in sorted(set(cab) & set(ref)):
        a = cab[s].get("crps"); b = ref[s].get("crps")
        if a is None or b is None:
            continue
        d = abs(a - b) / max(abs(b), 1e-12)
        peor = max(peor, d)
        print(f"  {s:>9}{a:>12.4f}{b:>12.4f}{d:>10.2e}")
    if peor < 1e-9:
        print("\n  Coincidencia exacta. La sustitucion de codificador no altera")
        print("  ninguna otra parte del protocolo.\n")
    elif peor < 1e-4:
        print(f"\n  Coincidencia hasta {peor:.1e}, atribuible a coma flotante.\n")
    else:
        print(f"\n  DISCREPANCIA de {peor:.1e}. NO interpretar las comparaciones")
        print("  de abajo: el control deberia ser identico y no lo es.\n")
    salida["verificacion_control"] = {"discrepancia_relativa_max": peor,
                                      "fuente": str(ruta_control)}


# ===========================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default=None)
    ap.add_argument("--columnas", nargs="*", default=None)
    ap.add_argument("--objetivo", default="cambio", choices=["nivel", "cambio"])
    ap.add_argument("--cabeza", default="R0-T",
                    help="cabeza generativa comun a todos los codificadores")
    ap.add_argument("--codificadores", nargs="*", default=None,
                    help=f"por defecto {list(CODIFICADORES)}")
    ap.add_argument("--inflar", action="store_true")
    ap.add_argument("--rapido", action="store_true")
    ap.add_argument("--control", default=None,
                    help="JSON de h2_cabezas.py para verificar fidelidad")
    a = ap.parse_args()

    real = a.panel is not None
    cfg = dict(h2c.CFG_REAL if real else h2c.CFG_SINT)
    if a.rapido:
        cfg.update(h2c.CFG_RAPIDO)
    semillas = (0,) if a.rapido else h2c.SEMILLAS

    nombres = a.codificadores or list(CODIFICADORES)
    desconocidos = [n for n in nombres if n not in CODIFICADORES]
    if desconocidos:
        raise SystemExit(f"codificadores desconocidos: {desconocidos}")
    if CONTROL not in nombres:
        nombres = [CONTROL] + nombres

    ts = datetime.now().isoformat(timespec="seconds")
    env = {"python": sys.version.split()[0], "torch": torch.__version__,
           "numpy": np.__version__, "plataforma": platform.platform(),
           "hilos": torch.get_num_threads()}

    print("=" * 78)
    print(f"LINEAS BASE NEURONALES  -  {'PANEL REAL' if real else 'PANEL SINTETICO'}"
          f"{'  [RAPIDO]' if a.rapido else ''}")
    print("=" * 78)
    print(f"  {ts}")
    print(f"  torch {env['torch']} | numpy {env['numpy']} | hilos {env['hilos']}")
    print(f"  objetivo: {a.objetivo}   cabeza comun: {a.cabeza}")
    print(f"  se sustituye UNICAMENTE el codificador; todo lo demas es el")
    print(f"  mismo codigo de h2_cabezas.py\n")

    params = auditar_contrato()

    if real:
        from h1_real import cargar_panel
        panel = cargar_panel(a.panel, a.columnas)
    else:
        panel = h2c._panel_sintetico(T=h2c.T_SINT, seed=12345)
    T = panel.shape[1]
    if cfg["T0"] + cfg["N_ORIGENES"] + H_FUT > T:
        raise SystemExit(f"panel de {T} meses insuficiente")
    print(f"  panel: {T} meses   T0={cfg['T0']}   origenes={cfg['N_ORIGENES']}"
          f"   M={cfg['M']}   pasos={cfg['STEPS']}   semillas={semillas}\n")

    salida = {"timestamp": ts, "entorno": env,
              "panel": "real" if real else "sintetico", "panel_meses": int(T),
              "objetivo": a.objetivo, "inflar": a.inflar,
              "cabeza": a.cabeza, "control": CONTROL,
              "config": {k: v for k, v in cfg.items()},
              "semillas": list(semillas), "params": params,
              "nota": ("lineas base neuronales por sustitucion de codificador; "
                       "unico cambio respecto de h2_cabezas.py"),
              "codificadores": {}}

    print("-" * 78)
    print(f"  {'codificador':<13}{'semilla':>8}{'CRPS':>9}{'ES cam':>9}"
          f"{'ES niv':>9}{'cob90':>8}{'cob50':>8}{'seg':>7}")
    print("  " + "-" * 70)

    t_ini = time.time()
    perdidas = {}
    es_origen = {}
    cond_original = _COND_ORIGINAL           # se restaura al terminar

    try:
        for nom in nombres:
            h2c.CondEnc = CODIFICADORES[nom]          # <-- la sustitucion
            salida["codificadores"][nom] = {"semillas": {}, "params": params[nom]}
            perdidas[nom] = {}
            for s in semillas:
                t0 = time.time()
                try:
                    ens, y, sc, lams, regs, mu_niv, sd_niv = h2c.evaluar(
                        panel, a.cabeza, s, cfg, a.objetivo, a.inflar)
                except Exception as e:
                    print(f"  {nom:<13}{s:>8}   FALLA: {type(e).__name__}: {e}")
                    salida["codificadores"][nom]["semillas"][str(s)] = {
                        "error": str(e)}
                    continue
                c_oh = h2c.joint_crps(ens, y, sc, standardize=True)
                crps = float(c_oh.mean())
                es_oh = h2c.energy_score_lineabase(ens, y, mu_niv, sd_niv)
                es = float(h2c.energy_score(ens, y, sc).mean())
                es_lb = float(es_oh.mean())
                c90, c90h, c90n = h2c.cobertura(ens, y, .90)
                c50, _, _ = h2c.cobertura(ens, y, .50)
                perdidas[nom][s] = c_oh.mean(axis=1)
                es_origen.setdefault(nom, {})[str(s)] = es_oh.mean(axis=1).tolist()
                print(f"  {nom:<13}{s:>8}{crps:>9.4f}{es:>9.4f}{es_lb:>9.4f}"
                      f"{c90:>8.3f}{c50:>8.3f}{time.time()-t0:>7.0f}")
                salida["codificadores"][nom]["semillas"][str(s)] = {
                    "crps": crps, "energy_score_cambios": es,
                    "energy_score_niveles": es_lb, "cob_90": c90, "cob_50": c50,
                    "cob_90_por_serie": c90n,
                    "params_totales": regs[0]["params"]}
            vals = [v["crps"] for v in
                    salida["codificadores"][nom]["semillas"].values()
                    if "crps" in v]
            if vals:
                salida["codificadores"][nom]["crps_medio"] = float(np.mean(vals))
                salida["codificadores"][nom]["crps_desv"] = float(np.std(vals))
            vals_es = [v["energy_score_niveles"] for v in
                       salida["codificadores"][nom]["semillas"].values()
                       if "energy_score_niveles" in v]
            if vals_es:
                salida["codificadores"][nom]["es_niveles_medio"] = \
                    float(np.mean(vals_es))
            print("  " + "-" * 70)
    finally:
        h2c.CondEnc = cond_original

    print(f"\n  tiempo total: {(time.time()-t_ini)/60:.1f} min\n")

    # ----------------------------------------------------- DM contra control
    print("=" * 78)
    print(f"  Diebold-Mariano contra {CONTROL}  (HAC NW h-1=11, correccion HLN)")
    print("=" * 78)
    print(f"  H0: no hay diferencia")
    print(f"  H1: el codificador alternativo es MEJOR que el Transformer\n")
    print(f"  {'codificador':<13}{'semilla':>9}{'dif. media':>13}"
          f"{'DM (HLN)':>11}{'p':>9}")
    print("  " + "-" * 55)
    salida["diebold_mariano"] = {}
    if not perdidas.get(CONTROL):
        print(f"  El control ({CONTROL}) no produjo resultados, de modo que no")
        print("  hay contra que contrastar. Revisar el error de arriba antes de")
        print("  leer cualquier cifra: sin control no se puede saber si una")
        print("  diferencia viene de la arquitectura o de la sustitucion.\n")
    for nom in nombres:
        if nom == CONTROL or not perdidas.get(nom):
            continue
        salida["diebold_mariano"][nom] = {"por_semilla": {}}
        comunes = sorted(set(perdidas[CONTROL]) & set(perdidas[nom]))
        for s in comunes:
            r, aviso = h2c.dm_seguro(perdidas[CONTROL][s], perdidas[nom][s],
                                     H_FUT)
            if r is None:
                print(f"  {nom:<13}{s:>9}   DM NO APLICABLE: {aviso}")
                salida["diebold_mariano"][nom]["no_aplicable"] = aviso
                continue
            print(f"  {nom:<13}{s:>9}{r['mean_diff']:>13.4f}"
                  f"{r['dm_hln']:>11.3f}{r['p_one_sided']:>9.4f}")
            salida["diebold_mariano"][nom]["por_semilla"][str(s)] = r
        if len(comunes) > 1:
            la = np.mean([perdidas[CONTROL][s] for s in comunes], axis=0)
            lb = np.mean([perdidas[nom][s] for s in comunes], axis=0)
            r, aviso = h2c.dm_seguro(la, lb, H_FUT)
            if r is not None:
                print(f"  {nom:<13}{'promedio':>9}{r['mean_diff']:>13.4f}"
                      f"{r['dm_hln']:>11.3f}{r['p_one_sided']:>9.4f}")
                salida["diebold_mariano"][nom]["promedio_semillas"] = r
        print("  " + "-" * 55)

    salida["perdida_por_origen"] = {
        n: {str(s): v.tolist() for s, v in d.items()} for n, d in perdidas.items()}
    salida["es_niveles_por_origen"] = es_origen

    if a.control:
        print()
        verificar_control(salida, a.control, a.objetivo)

    # ------------------------------------------------------------- lectura
    print("=" * 78)
    print("LECTURA")
    print("=" * 78)
    orden = sorted(((n, d.get("crps_medio")) for n, d in
                    salida["codificadores"].items() if d.get("crps_medio")),
                   key=lambda t: t[1])
    if orden:
        print(f"  {'codificador':<13}{'CRPS':>9}{'ES niv':>9}{'params':>9}"
              f"{'vs Transformer':>17}")
        base = salida["codificadores"].get(CONTROL, {}).get("crps_medio")
        for n, c in orden:
            d = salida["codificadores"][n]
            e = d.get("es_niveles_medio", float("nan"))
            if n == CONTROL or base is None:
                rel = ""
            else:
                rel = f"{100*(c/base - 1):+.1f} %"
            print(f"  {n:<13}{c:>9.4f}{e:>9.4f}{d['params']:>9,}{rel:>17}")
        print()
        mejor = orden[0][0]
        if base is None:
            print("  SIN CONTROL: el Transformer no corrio. Las cifras de arriba")
            print("  no son interpretables como comparacion arquitectonica.")
        elif mejor == CONTROL:
            print("  El codificador Transformer obtiene el mejor CRPS conjunto.")
            print("  Verificar en la tabla de arriba si la ventaja alcanza")
            print("  significancia: una diferencia de punto sin p < 0,05 no")
            print("  sostiene la eleccion arquitectonica, solo la hace")
            print("  compatible con los datos.")
        else:
            print(f"  El codificador {mejor} supera al Transformer en CRPS")
            print("  conjunto. Esto NO refuta H1 ni H2, que se contrastan sobre")
            print("  otras comparaciones, pero es un resultado que debe")
            print("  reportarse: la auto-atencion no estaria ganando su costo")
            print("  en este panel. Verificar antes la significancia y la")
            print("  paridad de parametros.")
        print()
        print("  ALCANCE. Esta comparacion responde si la auto-atencion aporta")
        print("  frente a alternativas mas simples sobre el MISMO protocolo.")
        print("  No sustituye a las lineas base clasicas: H2 se contrasta")
        print("  contra ARIMA/GARCH, que sigue siendo la mejor linea base.")

    modo = "rapido" if a.rapido else "full"
    p = Path(f"neuronales_{'real' if real else 'sintetico'}_{a.objetivo}"
             f"_{modo}_{ts[:10]}.json")
    p.write_text(json.dumps(salida, indent=2, ensure_ascii=False))
    print(f"\n  guardado en {p}")


if __name__ == "__main__":
    main()
