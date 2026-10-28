"""
exportar_es_lineabase.py  -  celda para MIA_Evaluacion_LineasBase.ipynb
=======================================================================

POR QUE. H2 exige dos cosas: que el Energy Score del modelo baje del umbral
de 0,990 Y que la diferencia sea significativa al 5 % en Diebold-Mariano
contra la linea base. Lo primero se resuelve con un promedio; lo segundo
necesita la SERIE DE PERDIDAS POR ORIGEN de ARIMA/GARCH, que hoy solo existe
dentro del cuaderno y se pierde al cerrarlo.

Son 109 numeros. Esta celda los guarda.

COMO USARLO. Pegar como celda nueva al final del cuaderno, despues de que
`res_arima` y `res_rw` esten calculados, y ejecutarla. Deja un unico archivo
`es_lineabase_por_origen.npz` junto al cuaderno.

NOTA SOBRE LA AGREGACION. El guion del modelo calcula su perdida por origen
como el promedio del Energy Score sobre los 12 horizontes de ese origen. Esta
celda hace exactamente lo mismo con `groupby("origen")["es"].mean()`, de modo
que ambas series son comparables termino a termino. Si se cambiara una, hay
que cambiar la otra.
"""

CELDA = r'''
# ---------------------------------------------------------------------------
# Exportacion de la serie de Energy Score por origen, para el contraste de H2
#
# H2 requiere Diebold-Mariano contra la linea base, y eso exige la serie de
# perdidas por origen y no solo su promedio. Se agrega sobre horizontes con
# la media, que es la misma agregacion que usa el guion del modelo.
# ---------------------------------------------------------------------------
import numpy as np

def _es_por_origen(res):
    """res: DataFrame con columnas origen, h, es.  ->  (origenes, es)"""
    g = res.groupby("origen")["es"].mean().sort_index()
    return g.index.to_numpy(), g.to_numpy(dtype=float)

org_ar, es_ar = _es_por_origen(res_arima)
org_rw, es_rw = _es_por_origen(res_rw)

assert (org_ar == org_rw).all(), "los dos modelos no comparten origenes"
print(f"origenes            : {len(org_ar)}")
print(f"ARIMA/GARCH  ES medio: {es_ar.mean():.4f}   (informe: 1,100)")
print(f"random walk  ES medio: {es_rw.mean():.4f}   (informe: 1,216)")

np.savez("es_lineabase_por_origen.npz",
         origenes=np.array([str(o) for o in org_ar]),
         arima_garch=es_ar,
         random_walk=es_rw,
         nota=("Energy Score por origen, promediado sobre h=1..12. "
               "Estandarizacion: media y desviacion de los NIVELES de la "
               "historia hasta cada origen (convencion del cuaderno). "
               "Comparable con energy_score_lineabase() de es_escala.py."))
print("\nguardado en es_lineabase_por_origen.npz")
# ---------------------------------------------------------------------------
'''

if __name__ == "__main__":
    print(__doc__)
    print("=" * 78)
    print("CELDA A PEGAR EN EL CUADERNO")
    print("=" * 78)
    print(CELDA)
