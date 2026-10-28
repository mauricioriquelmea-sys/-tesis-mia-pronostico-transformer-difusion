# Forecasting probabilístico multivariado con Transformer–Difusión y regularización estructural informada por dominio

Tesis del Magíster en Inteligencia Artificial, Pontificia Universidad Católica de Chile (INF4980).
Autor: Mauricio Riquelme Alvarado · Profesor supervisor: Marcelo Mendoza.

Este repositorio respalda el informe final: contiene el código, los datos públicos y los archivos de resultados de cada tabla. Cada cifra del informe se puede rastrear hasta el guion que la produjo y el archivo donde quedó registrada.

---

## Estructura

```
/cuadernos/            cuadernos Jupyter con sus datos de entrada y registros de trazabilidad
/codigo/               todos los guiones y módulos de Python (una sola copia de cada uno)
/resultados/<sección>/ tablas y archivos de resultados, ordenados por la sección del informe que los reporta
```

Los cuadernos y sus datos de entrada comparten la carpeta `cuadernos/` porque los cuadernos buscan sus archivos en el directorio de trabajo (`Path.cwd()`). Hay que abrirlos o ejecutarlos desde esa carpeta. Las tablas que producen están en `resultados/`; si se vuelven a ejecutar, los cuadernos las escriben de nuevo en `cuadernos/`. Los guiones están juntos en `codigo/` porque se importan entre sí. Para ejecutarlos, se corren desde la raíz con `python codigo/<guion>.py` y se les pasa la ruta del panel (`cuadernos/panel_2003_2024.csv`) como argumento.

### `cuadernos/`: cuadernos y datos de entrada

| Archivo | Sección | Contenido |
|---|---|---|
| `MIA_Construccion_Panel.ipynb` | §4 | Descarga, verificación de la convención de agregación, pruebas ADF, correlaciones, Granger y figuras. |
| `Precios-del-Cobre-Refinado-Mensual.xlsx` | §4.1 | Cochilco, archivo histórico. |
| `panel_bcch_2003_2024.csv` | §4.1 | Banco Central de Chile, API de la Base de Datos Estadísticos. |
| `verificacion_agregacion.csv` | §4.1 | Serie mensual oficial frente a la agregación de la serie diaria. |
| `panel_2003_2024.csv` | §4.3 | Panel final: 264 meses × 3 series. |
| `cobertura_bcch.txt`, `registro_panel.txt` | §4.5 | Registros de trazabilidad. |
| `MIA_Evaluacion_LineasBase.ipynb` | §5.2.2, §5.3 | Ventana expansiva, CRPS, Energy Score y Diebold–Mariano de las líneas base. Su última celda exporta `es_lineabase_por_origen.npz`. |
| `MIA_Parte1_GeneradorSistemaEconomico.ipynb` | §5.2.1 | Generador sintético GARCH + ARIMA, el primero de los dos. |
| `MIA_VersionA_Diagnostico.ipynb`, `MIA_VersionB_RegularizacionRG.ipynb` | §5.2.1 (Tabla 9) | **Cuadernos de desarrollo**, con término variacional y η = 5. Sus valores de L_simple no son comparables con los de §5 (ver §3.2.2). |
| `CRPS_GARCH.ipynb` | — | Desarrollo del CRPS por muestras. |

### Índice por sección del informe

| Sección | Guiones (`codigo/`) | Resultados (`resultados/`) |
|---|---|---|
| **§4.3** Estadísticas del panel (Tablas 3–6) | `cuadernos/MIA_Construccion_Panel.ipynb` | `4.3/tabla_descriptivos.csv`, `4.3/tabla_estacionariedad.csv`, `4.3/tabla_correlaciones.csv`, `4.3/tabla_granger.csv` |
| **§5.1–5.3** Protocolo y líneas base (Tablas 10 y 19) | `cuadernos/MIA_Evaluacion_LineasBase.ipynb`, `evaluacion.py`, `arima_garch.py` | `5.2.2/resultados_random_walk.csv`, `5.2.2/resultados_arima_garch.csv`, `5.2.2/tabla_baselines.csv`, `5.2.2/tabla_dm_baselines.csv` |
| **§5.2.3** Contraste de H1 (Tablas 11–13) | `h1_sintetico.py`, `h1_real.py`, `bloques_h1.py`, `cabeza_r0.py`, `h1_harness.py`, `rg_frobenius.py`, `run_h1.py` | `5.2.3/h1_sintetico_full_2026-10-11.json`, `5.2.3/h1_real_full_2026-10-11.json` |
| **§5.2.4** Objetivo de entrenamiento (Tablas 14–15) | `condicionante_real.py`, `dm_objetivo.py`, `diag_difusion.py`, `diag_calibracion.py`, `diag_calibracion2.py`, `diag_encoder.py`, `diag_pooling.py`, `diag_horizonte.py`, `cabeza_difusion.py` | `5.2.4/*` |
| **§5.2.5** H2 y auditoría de la línea base (Tablas 16–17) | `h2_cabezas.py`, `dm_h2.py`, `es_escala.py`, `exportar_es_lineabase.py`, `verificar_psi_lineabase.py`, `recalcular_es_lineabase.py` | `5.2.5/*` |
| **§5.2.6 y §A.4** Re-contraste de H1 (Tabla 18) | `recontraste_h1_cambio.py` | `5.2.6/recontraste_h1_cambio_FASE1_2026-10-14.json`, `5.2.6/recontraste_h1_cambio_full_2026-10-14.json` |
| **§5.4** Codificadores (Tabla 20) | `lineas_neuronales.py` | `5.4/neuronales_real_cambio_full_2026-10-13.json` |
| **§6.2.2** Escalera de verificación de la difusión | `diag_difusion.py` (criterio C1) y `cabeza_difusion.py` (`schedule_coseno`) | `5.2.4/diag_difusion_full_2026-10-12.json` |
| **§A.1** Prerregistro | — | `A.1/PREREGISTRO_H1.md` |
| **§A.3** Fuerza del prior | `medir_frac_real.py`, `eta_sensibilidad.py` | `A.3/eta_sensibilidad_real_full_2026-10-14.json`, `A.3/eta_sensibilidad_sintetico_full_2026-10-14.json` |

## Ejecución

```
pip install -r requirements.txt

# §4 y §5.2.2–5.3: abrir y ejecutar los cuadernos de cuadernos/, en orden y desde esa carpeta
#   1. MIA_Construccion_Panel.ipynb   (reutiliza los archivos descargados; no requiere credenciales)
#   2. MIA_Evaluacion_LineasBase.ipynb

# §5.2.3
python codigo/h1_sintetico.py
python codigo/h1_real.py cuadernos/panel_2003_2024.csv

# §A.3
python codigo/medir_frac_real.py --panel cuadernos/panel_2003_2024.csv
python codigo/eta_sensibilidad.py
python codigo/eta_sensibilidad.py --panel cuadernos/panel_2003_2024.csv

# §5.2.4
python codigo/condicionante_real.py --panel cuadernos/panel_2003_2024.csv --objetivo nivel
python codigo/condicionante_real.py --panel cuadernos/panel_2003_2024.csv --objetivo cambio
python codigo/dm_objetivo.py <salida_nivel>.json <salida_cambio>.json
python codigo/diag_difusion.py            # y los demás diag_*.py, sobre el generador sintético

# §5.2.5
python codigo/h2_cabezas.py --panel cuadernos/panel_2003_2024.csv
python codigo/dm_h2.py resultados/5.2.5/h2_real.json resultados/5.2.5/es_lineabase_por_origen.npz
python codigo/verificar_psi_lineabase.py --panel cuadernos/panel_2003_2024.csv
python codigo/recalcular_es_lineabase.py --panel cuadernos/panel_2003_2024.csv

# §5.2.6 y §A.4
python codigo/recontraste_h1_cambio.py --panel cuadernos/panel_2003_2024.csv

# §5.4
python codigo/lineas_neuronales.py --panel cuadernos/panel_2003_2024.csv
```

Los guiones escriben sus resultados en la carpeta desde la que se ejecutan. Los de `resultados/` son los que produjeron las cifras del informe.

Entorno de los contrastes: Python 3.12.0, PyTorch 2.14.1+cpu, NumPy 2.2.5, Windows 10 y `torch.set_num_threads(1)`. El barrido de §A.3 se ejecutó con 4 hilos. Los resultados obtenidos en entornos distintos no se combinan entre sí (§5.2.3).

## Notas de trazabilidad

- **R_G ejecutado frente al prerregistrado.** Los contrastes usan la función `R_G` de `bloques_h1.py`, no `rg_frobenius.py`. La proyección T×T → N×N es idéntica. La escala difiere en un factor constante: el R_G de `rg_frobenius.py` es exactamente 3 veces el de `bloques_h1.py`, porque uno promedia sobre las 9 entradas y el otro suma por fila y promedia sobre filas. Los valores de η del informe están en la escala de `bloques_h1.py`. Las fracciones estructurales no dependen de ese factor.
- **`run_h1.py`** es la plantilla con calibración de η que, según documenta §A.3, nunca se ejecutó. Se publica tal cual, como evidencia de la desviación.
- **Re-contraste en dos fases.** `recontraste_h1_cambio.py` escribe el archivo de la FASE 1, con el umbral fijado en 7,26 %, antes de comparar ningún CRPS. Ambos archivos llevan en el campo `timestamp` la hora de inicio de la ejecución. El orden entre las fases lo garantiza la estructura del guion: la FASE 2 no corre si la FASE 1 falla.
- **Fechas de los prerregistros.** Los prerregistros se fijaron en las fechas indicadas en el Anexo A. Las fechas de los commits de este repositorio son las de publicación y no prueban la fecha de fijación.

## Fuentes de datos

- **Comisión Chilena del Cobre (Cochilco).** Precio del cobre refinado, Bolsa de Metales de Londres, nominal, en centavos de dólar por libra.
- **Banco Central de Chile.** Base de Datos Estadísticos: dólar observado (`F073.TCO.PRE.HIST.M`) y Tasa de Política Monetaria (`F022.TPM.TIN.D001.NO.Z.M`).

Los datos son públicos y pertenecen a sus instituciones. Se incluyen solo para permitir la reproducción de los resultados.

## Uso

El código se publica con fines académicos, como respaldo de la tesis.
