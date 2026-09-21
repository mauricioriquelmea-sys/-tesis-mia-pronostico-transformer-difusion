# Pronóstico probabilístico multivariado con Transformer–Difusión

**Extensión de regularización estructural informada por dominio**

Tesis del Magíster en Inteligencia Artificial, Pontificia Universidad Católica de Chile (INF4980).
Autor: Mauricio Riquelme Alvarado · Profesor guía: Marcelo Mendoza.

---

## Descripción

Este repositorio contiene el código y los datos del proyecto de tesis, cuyo objetivo es pronosticar la **distribución conjunta** de tres series macro-financieras chilenas —precio del cobre, tipo de cambio USD/CLP y Tasa de Política Monetaria (TPM)— con datos mensuales 2003–2024.

La propuesta es una arquitectura híbrida Transformer–Difusión que incorpora conocimiento económico de dominio como una regularización explícita sobre los mapas de atención del modelo. La hipótesis estructural es una cadena de dependencia Cu → USD/CLP → TPM, inyectada como una penalización suave en la función de pérdida, de modo que los datos puedan corregirla si resulta mal especificada.

## Estado

Corresponde al **Informe de Avance N.º 3**. Incluye el panel de datos real verificado, el protocolo de evaluación probabilística y dos líneas base evaluadas (*random walk* y ARIMA/GARCH). Las Versiones A y B del modelo propuesto aún no han sido evaluadas sobre el panel real: requieren la cabeza de difusión condicional, en desarrollo.

## Contenido

### Cuadernos

| Archivo | Contenido |
|---|---|
| `MIA_Construccion_Panel.ipynb` | Construcción y caracterización del panel: descarga, verificación de la convención de agregación, pruebas ADF, correlaciones, causalidad de Granger y figuras. Corresponde a la Sección 4 del informe. |
| `MIA_Evaluacion_LineasBase.ipynb` | Protocolo de ventana expansiva, reglas de puntuación (CRPS y *Energy Score*), prueba de Diebold–Mariano y evaluación de las líneas base. Corresponde a la Sección 5. |
| `MIA_Parte1_GeneradorSistemaEconomico.ipynb` | Generador de un sistema económico sintético con causalidad controlada, empleado para verificar el mecanismo de inyección del *prior*. |
| `MIA_VersionA_Diagnostico.ipynb` | Modelo base (Versión A): tokenización multivariada, *encoder* Transformer causal y diagnóstico de la atención. |
| `MIA_VersionB_RegularizacionRG.ipynb` | Extensión con regularización estructural (Versión B) y barrido del peso de la penalización. |
| `CRPS_GARCH.ipynb` | Cálculo del CRPS por muestras. |

### Módulos

| Archivo | Contenido |
|---|---|
| `evaluacion.py` | Implementación de referencia del protocolo de evaluación y de las métricas. |
| `arima_garch.py` | Línea base ARIMA/GARCH conectada al módulo de evaluación. |

### Datos

| Archivo | Origen |
|---|---|
| `Precios-del-Cobre-Refinado-Mensual.xlsx` | Cochilco, archivo histórico de precios de metales. |
| `panel_bcch_2003_2024.csv` | Banco Central de Chile, API de la Base de Datos Estadísticos. |
| `verificacion_agregacion.csv` | Contraste entre la serie mensual oficial y la agregación de la serie diaria. |
| `panel_2003_2024.csv` | Panel final: 264 meses × 3 series, sin valores faltantes. |

### Resultados y trazabilidad

Tablas de la Sección 4 (`tabla_descriptivos.csv`, `tabla_estacionariedad.csv`, `tabla_correlaciones.csv`, `tabla_granger.csv`), resultados de las líneas base (`resultados_random_walk.csv`, `resultados_arima_garch.csv`) y dos registros de trazabilidad:

- `cobertura_bcch.txt` documenta la descarga desde el Banco Central, los códigos de serie con sus títulos oficiales y la verificación de la convención de agregación.
- `registro_panel.txt` documenta las transformaciones aplicadas, las dimensiones del panel y las versiones exactas del entorno de cómputo.

## Ejecución

### Requisitos

```
pip install -r requirements.txt
```

Las pruebas estadísticas dependen de la implementación de las bibliotecas, en particular la selección automática de rezagos. `requirements.txt` fija las versiones empleadas para producir los resultados reportados.

### Orden

1. **`MIA_Construccion_Panel.ipynb`** produce `panel_2003_2024.csv` y las tablas de la Sección 4.
2. **`MIA_Evaluacion_LineasBase.ipynb`** consume ese panel y produce los resultados de la Sección 5.

Los cuadernos buscan sus archivos en la misma carpeta en que se encuentran. Deben ejecutarse desde la raíz del repositorio, sin reorganizar los archivos en subcarpetas.

### Credenciales y caché

La descarga desde el Banco Central requiere credenciales personales de la API, leídas desde un archivo `credenciales.txt` que **no forma parte del repositorio** (dos líneas: correo y clave). Las credenciales se obtienen registrándose en la Base de Datos Estadísticos y activando el acceso a la API desde el perfil de usuario.

No son necesarias para reproducir los resultados: el cuaderno de construcción del panel verifica primero si existen los archivos descargados y, en ese caso, los reutiliza. Con los CSV incluidos en este repositorio, todos los cuadernos se ejecutan sin acceso a la API.

## Reproducibilidad

El pipeline fue ejecutado en dos entornos independientes, con una discrepancia máxima del orden de 10⁻¹³ entre los resultados, atribuible a aritmética de punto flotante.

La convención de agregación mensual de las series no se asumió: se determinó empíricamente contrastando la serie mensual oficial con la agregación de la serie diaria. Las tres series corresponden a promedios mensuales.

## Fuentes de datos

- **Comisión Chilena del Cobre (Cochilco).** Precio del cobre refinado, Bolsa de Metales de Londres, nominal, en centavos de dólar por libra.
- **Banco Central de Chile.** Base de Datos Estadísticos. Tipo de cambio del dólar observado (`F073.TCO.PRE.HIST.M`) y Tasa de Política Monetaria (`F022.TPM.TIN.D001.NO.Z.M`).

Los datos provienen de fuentes públicas y oficiales, y pertenecen a sus respectivas instituciones. Se incluyen en este repositorio únicamente para permitir la reproducción de los resultados.

## Uso

El código se publica con fines académicos, como respaldo de la tesis. Para citarlo o reutilizarlo, contactar al autor.
