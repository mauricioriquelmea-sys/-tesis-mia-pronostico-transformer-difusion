# Prerregistro: tamaño y potencia de la prueba de Diebold–Mariano (Parte A)

**Fecha de fijación:** 6 de noviembre de 2026 · *commit*: ________ (completar al subirlo)

**Regla de fijación:** este documento queda fijado al subirse al repositorio con un *commit* fechado, **antes** de ejecutar cualquier simulación. Toda desviación posterior se registra con fecha en la sección 7, sin reescribir las anteriores.

**Relación con la tesis:** análisis complementario posterior al borrador (ítem 1 de §7.6). **No modifica** los veredictos de H1, del re-contraste ni de H2, ni sus umbrales.

**Aislamiento de datos:** usa solo el generador sintético. No toca el panel chileno.

---

## 1. Propósito

§5.2.6 afirma que con el efecto medio del re-contraste harían falta unos 7 años más de datos para una potencia del 80 %. Esa cifra descansa en dos supuestos no verificados:

1. que el efecto crítico escala como 1/√O, con O el número de orígenes;
2. que el factor (t₀,₉₅ + t₀,₈₀)/t₀,₉₅ ≈ 1,51 predice la potencia del 80 % bajo la prueba DM-HLN con pérdidas solapadas a 12 horizontes.

Además, la tesis supone que la prueba tiene un tamaño cercano al 5 % nominal. Esta parte verifica los tres supuestos por simulación.

## 2. Pregunta

Bajo la prueba que usa la tesis (Diebold–Mariano unilateral sobre el CRPS conjunto estandarizado por origen, HAC de Newey–West con 11 rezagos y ajuste HLN), con pérdidas autocorrelacionadas por horizontes solapados:

- ¿El tamaño empírico es cercano al 5 % nominal?
- ¿El efecto crítico escala como 1/√O?
- ¿El factor ×1,51 predice bien la potencia del 80 %?

## 3. Generador

El VAR(1) triangular del contraste sintético de H1 (§5.2.3), con los mismos coeficientes de la cadena Cu → USD/CLP → TPM y las mismas innovaciones gaussianas (`bloques_h1._panel_sintetico`). Se usa sin modificar.

## 4. Pronosticadores

No se entrena ninguna red: la parte evalúa la prueba, no el modelo.

- **B (referencia):** pronosticador oráculo, es decir, la predictiva gaussiana exacta a h pasos dados los parámetros verdaderos, con M = 200 muestras.
- **A (degradado):** el mismo oráculo con su dispersión inflada por un factor k_δ, elegido numéricamente **antes de ejecutar** para que la reducción esperada del CRPS conjunto de B respecto de A sea exactamente δ.
- A y B se evalúan **sobre las mismas realizaciones**, con la escala por origen calculada solo con información disponible hasta ese origen (regla de §A.1).

## 5. Grilla

| Factor | Valores |
|---|---|
| Efecto verdadero δ | 0 %, 4 %, 6 %, 8 %, 10 % |
| Orígenes O | 109, 150, 200, 300, 450 |
| Horizontes | h = 1, …, 12, solapados como en la tesis |
| Réplicas por celda | R = 2.000 paneles independientes |

## 6. Salidas y reglas de decisión (fijadas)

**Salidas por celda (δ, O):** tasa de rechazo al 5 % con su error de Monte Carlo, y mediana del efecto crítico t₀,₉₅ · ŝ(d̄) / CRPS_A.

**Reglas:**

| Supuesto | Criterio | Se sostiene si |
|---|---|---|
| Tamaño | Tasa de rechazo con δ = 0, en todos los O | entre **3,5 % y 6,5 %** (±3 errores de Monte Carlo con R = 2.000) |
| Ley 1/√O | Pendiente de log(efecto crítico mediano) sobre log O | entre **−0,60 y −0,40** (teórica −0,5) |
| Factor ×1,51 | Para cada δ > 0, razón O₈₀ / O₅₀ entre los orígenes que dan potencia del 80 % y del 50 % (interpolados) | entre **1,9 y 2,7** (teórica 1,51² ≈ 2,28) |

Fuera de la banda de tamaño, la prueba se reporta como sobre- o sub-rechazante.

**Consecuencia:** si alguna regla falla, el informe final corrige la cifra de años de §5.2.6 con el factor simulado y lo declara. Los veredictos no cambian, porque se basan en las pruebas ejecutadas y no en la extrapolación.

**Alcance:** la varianza del diferencial en la simulación es la del oráculo frente a su versión degradada, no la del Transformer A frente al B sobre el panel chileno. La Parte A valida la **ley de escala y el factor**, no el nivel de varianza que entra en la cifra de 7 años.

## 7. Declaraciones

**Corrida piloto previa (14 de octubre de 2026).** Antes de este prerregistro se ejecutó `barrido_muestra.py` (resultado en `barrido_muestra_full_2026-10-14.json`). Fue una versión reducida de la Parte B (no de esta Parte A): una semilla, 24 orígenes y η calibrado. Las reducciones relativas registradas fueron −6,0 % con 48 meses de entrenamiento, +14,0 % con 144 y −13,6 % con 486. El patrón no es monótono. Se declara porque es una mirada previa a resultados de la misma línea de trabajo; **no se usa como evidencia** ni influyó en la grilla, las bandas ni los pronosticadores de esta Parte A, que no entrena modelos. *(Confirmar en el JSON sobre qué panel se ejecutó y anotarlo aquí antes de fijar.)*

**Parte B (efecto de R_G en muestra escasa).** El borrador de prerregistro contemplaba una Parte B con 720 entrenamientos sobre el generador sintético. **No se ejecuta en este período** por presupuesto de tiempo, y queda como trabajo futuro (§7.6). Si se retoma, requiere un prerregistro propio con sus decisiones pendientes: configuración principal, presupuesto de cómputo y viabilidad del mínimo de 60 meses.

**Desviaciones posteriores a la fijación:** (ninguna a la fecha)

## 8. Qué se reporta siempre

- Todas las celdas de la grilla, con independencia del resultado.
- Los resultados por celda en un archivo adjunto en el repositorio.
- Cualquier desviación, con fecha y explicación, en la sección 7.

## 9. Qué no se hará

- Agregar réplicas, valores de δ u orígenes después de ver resultados.
- Cambiar la grilla o las bandas a la luz de los resultados.
- Usar estos resultados para reinterpretar el veredicto de H1, del re-contraste o de H2.
- Tocar el panel real.
