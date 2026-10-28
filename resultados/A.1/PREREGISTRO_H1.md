# Prerregistro del contraste de H1

**Fecha de fijación:** 11 de octubre de 2026
**Estado:** fijado ANTES de entrenar y de ver cualquier resultado

Este documento registra las decisiones de evaluación antes de ejecutar el
experimento. Su propósito es que ninguna de estas reglas pueda haberse
elegido en función del resultado obtenido.

---

## 1. Hipótesis contrastada

**H1.** La regularización estructural R_G reduce el CRPS conjunto respecto de
la Versión A en al menos **5 %**, con significancia al **5 %** en la prueba de
Diebold-Mariano con corrección HAC.

Los umbrales son los comprometidos en §3.1, §3.4 y §5.3 del informe de avance
del bimestre 3, entregado el 21 de septiembre de 2026. **No se modifican.**

## 2. Regla de agregación del CRPS conjunto

El CRPS conjunto se calcula **estandarizando cada serie** por la desviación
estándar de sus cambios, estimada únicamente con información disponible hasta
cada origen de pronóstico, y promediando después entre las tres series.

**Justificación.** Las escalas de las tres series difieren en un orden de
magnitud (CRPS de línea base: cobre ≈ 28,9; USD/CLP ≈ 37,0; TPM ≈ 1,69). Bajo
suma cruda, el USD/CLP aporta el 55 % del total y la TPM el 2,5 %, de modo que
el veredicto de H1 dependería de en qué serie mejora el modelo y no de cuánto
mejora. Verificado sobre tensores sintéticos: una mejora del 30 % en nitidez
rinde +1,2 % bajo suma cruda si ocurre en la TPM y +16,7 % si ocurre en el
USD/CLP, mientras que estandarizado rinde ≈ +10 % en ambos casos.

La estandarización es consistente con el tratamiento ya aplicado al Energy
Score y con la decisión M9 sobre normalización por origen.

**Reporte secundario.** La suma cruda se reporta como análisis de robustez,
junto a la tabla de escenarios, sin que su resultado redefina el éxito.

## 3. Especificación de R_G

**Divergencia:** distancia de Frobenius, D = ‖W_agg − Π(G)‖²_F promediada por fila.

**Justificación.** Π(G) contiene 4 ceros de 9 entradas. KL(W‖Π) no es finita
sin suavizado, y el valor suavizado depende del ε elegido (factor 3,7 entre
ε = 1e-3 y ε = 1e-9), por lo que η carece de escala interpretable. Frobenius
no requiere ε, tiene gradiente lineal y acotado, y separa atención colapsada
de estructurada por un factor ≈ 550 frente a ≈ 34 de las variantes de KL.

**Proyección T×T → N×N:** suma sobre claves, promedio sobre consultas. Es la
única de las tres reglas razonables bajo la cual las filas agregadas suman 1,
condición necesaria para compararlas con las filas de Π(G).

**layer_reduce:** `mean` — se proyecta cada capa por separado y se promedian
las proyecciones. Π(G) se postula como estructura del sistema, por lo que
debe observarse en todas las capas y no solo en la última.

**Nota de comparabilidad.** La especificación previa (KL con ε = 1e-6, η = 5)
destinaba el 96,4 % de la pérdida total al término estructural. Los resultados
obtenidos bajo esa especificación no son comparables con los de este
experimento y se reportan como hallazgo metodológico en §5, no como brazo de
comparación.

## 4. Calibración de η

η se calibra una sola vez, sobre el primer lote, de modo que

    η · R_G / (L_simple + η · R_G) = 0,20

y queda **fijo** durante todo el entrenamiento. Recalibrar durante el
entrenamiento haría que la función de pérdida cambie de definición sobre la
marcha e invalidaría la comparación entre épocas.

La fracción 0,20 deja el término estructural como contribución sustantiva sin
que domine la optimización.

## 5. Protocolo de evaluación

- Ventana expansiva, T0 = 144, 109 orígenes de pronóstico, horizontes h = 1..12.
- Reajuste cada 12 pasos.
- Cabeza generativa: R0 (gaussiana de rango bajo) en ambas versiones.
- Semillas: 0, 1, 2. Se reportan las tres; no se selecciona la mejor.
- Estimador de CRPS: insesgado, con denominador M(M−1).
- Diebold-Mariano: una sola prueba sobre el CRPS conjunto agregado en
  horizontes, con rezago de Newey-West h−1 = 11 y corrección
  Harvey-Leybourne-Newbold. Los cortes por serie y por horizonte son
  descriptivos y **no** redefinen el éxito de H1.

## 6. Qué constituye rechazo

Si la reducción es menor al 5 %, o si p ≥ 0,05, **H1 se rechaza y se reporta
como tal**. Un rechazo con umbrales fijados de antemano es un resultado
legítimo del trabajo. No se modificará la regla de agregación, el conjunto de
horizontes, la cabeza generativa ni los umbrales para obtener un resultado
favorable.

---

*Fijado antes de la ejecución. Cualquier modificación posterior debe quedar
registrada con fecha y justificación, y declararse en el informe.*
