# Prerregistro del Variogram Score sobre H1

Fijado el 11 de noviembre de 2026, antes de la primera ejecución completa sobre el panel real. El hash del commit que lo fija se anota en el README.

## 1. Pregunta

¿La regularización estructural $R_G$ cambia la dependencia entre series que reproduce el pronóstico conjunto? El CRPS conjunto del informe es un promedio de CRPS marginales y no ve la dependencia. El Energy Score la incluye, pero con poca sensibilidad. El Variogram Score (Scheuerer y Hamill, 2015) responde a la estructura de correlación y es la métrica que §7.6 deja como trabajo futuro.

## 2. Carácter

Es un análisis **descriptivo y complementario**. No redefine el veredicto de H1 (rechazada en niveles, inconcluyente en el re-contraste) ni el de H2. Ningún resultado de este análisis se usa para elegir configuraciones.

## 3. Métrica

$$\mathrm{VS}_p = \sum_{i<j} w_{ij}\left(|y_i-y_j|^p - \tfrac{1}{M}\textstyle\sum_m |X_i^{(m)}-X_j^{(m)}|^p\right)^2$$

- Exponente $p = 0{,}5$, el valor que recomiendan Scheuerer y Hamill.
- Pesos $w_{ij}=1$.
- Antes de calcular, cada serie se divide por su escala (desviación de los cambios hasta cada origen), igual que en el Energy Score del informe.
- **Métrica principal: `VS_series`.** Usa las tres series en cada par (origen, horizonte), con $d=3$ y tres pares, y se promedia en los 12 horizontes para obtener una pérdida por origen.
- **Métrica descriptiva: `VS_trayectoria`.** Usa el vector completo de 36 componentes (3 series por 12 horizontes) de cada origen. Incluye la dependencia entre horizontes.

## 4. Configuraciones

| Nombre | Objetivo | Cabeza | $\eta$ | Rol |
|---|---|---|---|---|
| `recontraste` | cambios | R0-T | calibrado a fracción 0,20 | principal |
| `h1` | niveles | R0-G | 5 | secundaria |

El resto se mantiene idéntico al re-contraste y al contraste original: $T_0=143$, 109 orígenes, $M=200$, 400 pasos, reajuste cada 12 orígenes, semillas 0, 1 y 2, y `torch.set_num_threads(1)`.

## 5. Prueba

- Diebold–Mariano unilateral (B mejor que A) con corrección HLN y rezago de Newey–West $h-1=11$, sobre la pérdida por origen. Es la misma función `diebold_mariano` de `h1_harness.py`.
- Se informan la reducción relativa y el valor $p$ por semilla, junto con el $p$ del contraste recíproco (A mejor que B).
- No hay umbral de magnitud.
- La Parte A mostró que la prueba rechaza de más con 109 orígenes (tamaño real cercano al 10 % para un 5 % nominal). Por eso solo se lee como evidencia un $p < 0{,}025$, y un $p$ entre 0,025 y 0,05 se informa como no concluyente.
- No se calcula una prueba agregada entre semillas, por la misma razón que en el anexo A.5.

## 6. Control de reproducción

El guion recalcula el CRPS conjunto de A y B. En la configuración `recontraste` debe coincidir con los valores publicados en la Tabla del re-contraste dentro de $5\times10^{-4}$:

| Semilla | CRPS A | CRPS B |
|---|---|---|
| 0 | 2,5508 | 2,3976 |
| 1 | 2,5218 | 2,3412 |
| 2 | 2,5974 | 2,3054 |

Si no coincide, el resultado se informa como obtenido en otro entorno y no se combina con las cifras del informe.

## 7. Lectura fijada de antemano

- **Si `VS_series` de B es menor que el de A con $p<0{,}025$ en al menos dos semillas**, se informa que $R_G$ mejora la dependencia entre series reproducida por el pronóstico. Esto es compatible con el canal que propone la tesis, aunque el CRPS no la vea.
- **Si la diferencia no es significativa**, se informa que el prior no deja huella detectable en la dependencia con este tamaño muestral, y se agrega a la discusión de potencia.
- **Si B es peor que A con $p<0{,}025$ en al menos dos semillas**, se informa que imponer $\Pi(G)$ deteriora la dependencia reproducida. Este resultado sería coherente con la evidencia sobre la exogeneidad del cobre.

## 8. Archivos

- Guion: `codigo/vs_h1.py`.
- Salidas: `resultados/parteA/vs_h1_recontraste_full_<fecha>.json` y `resultados/parteA/vs_h1_h1_full_<fecha>.json`.
- Antes de ejecutar, el guion verifica la métrica sobre tensores dummy (`--solo-autoverificacion`) y la contrasta con una implementación directa por bucles.
