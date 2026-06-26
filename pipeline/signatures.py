"""
Etiquetado de perfiles/clusters por FIRMA FÍSICA del artefacto.

Reemplaza el viejo auto_label() que mapeaba (caudal, duración) a fixtures con
rangos regulatorios US (EPA WaterSense) — techos que no aplican a una instalación
gravitacional chilena y que IGNORABAN el volumen, el discriminador físico #1.

Cada artefacto tiene una firma:
  - INODORO: volumen ~fijo de cisterna (CV de volumen BAJO entre eventos) y
    duración media. El caudal NO se umbraliza duro (a baja presión cae fuera de
    tablas US). Banda 3-9 L para cubrir cisternas estándar (6 L) y eco (3-4.5 L).
  - DUCHA: volumen ALTO y duración larga (plateau sostenido).
  - GOTEO / FUGA: caudal basal bajo y persistente (clase accionable: ahorro real).
  - GRIFO / LAVAMANOS: volumen chico, corto, caudal variable.
  - SIN CLASIFICAR: si nada matchea (honesto; mejor que forzar una etiqueta falsa).

Se quitan del vocabulario los fixtures imposibles en un refugio de montaña
(lavadora, lavavajillas, riego), que antes absorbían duchas largas y otros usos.
"""

UNCLASSIFIED = "Sin clasificar"


def label_by_signature(
    median_flow: float,
    median_duration: float,
    median_volume: float,
    cv_volume: float,
) -> str:
    """Etiqueta un cluster por su firma física.

    Args:
        median_flow: caudal medio del cluster (L/min).
        median_duration: duración mediana (s).
        median_volume: volumen mediano por evento (L) — discriminador #1.
        cv_volume: coef. de variación del VOLUMEN entre los eventos del cluster
            (std/mean). Bajo (<~0.4) = volumen consistente = firma de cisterna.
    """
    f, d, v, cvv = median_flow, median_duration, median_volume, cv_volume

    # Etiqueta por CARÁCTER FÍSICO. Lo CLARO se nombra con confianza (ducha por
    # volumen alto sostenido; fuga por caudal basal persistente); el resto por su
    # firma como MEJOR ESTIMACIÓN para un baño, calibrable por el operador (que ve
    # volumen+CV por cluster). El "Inodoro (cisterna)" — volumen consistente — solo
    # se nombra así cuando hay firma de cisterna real; si el volumen es variable se
    # usa "Inodoro / descarga" (estimación) en vez de afirmar una cisterna.

    # 1. DUCHA: volumen alto y sostenido.
    if v >= 15.0 and d >= 90:
        return "Ducha"

    # 2. GOTEO / FUGA: caudal basal bajo y persistente (accionable).
    if f < 1.2 and d >= 90:
        return "Goteo / fuga"

    # 3. INODORO (cisterna): volumen ~fijo y CONSISTENTE entre eventos.
    if 3.5 <= v <= 9.0 and cvv < 0.30 and f >= 4.0:
        return "Inodoro (cisterna)"

    # 4. INODORO / descarga: caudal alto y corto. En un baño, el uso breve de alto
    # caudal más frecuente es la descarga (mejor estimación; el operador confirma).
    if f >= 4.5 and d <= 60 and v >= 1.5:
        return "Inodoro / descarga"

    # 5. GRIFO / LAVAMANOS: caudal bajo-medio, uso manual.
    if v < 6.0:
        return "Grifo / lavamanos"

    # 6. GRIFO prolongado / llenado: volumen medio sin firma clara.
    if v < 15.0:
        return "Grifo (uso prolongado)"

    return UNCLASSIFIED


# Rangos físicos de volumen por etiqueta (L) para el RECHAZO en inferencia: un
# evento atribuido a una etiqueta cuyo volumen cae MUY fuera de su rango plausible
# se manda a No Detectado en vez de contaminar el cluster.
# Rangos GENEROSOS: el labeling por firma ya ubica los eventos en clusters físicos,
# así que el rechazo solo atrapa atribuciones ABSURDAS (una "Ducha" de 0.5 L), sin
# castigar la cola natural de cada uso (preserva cobertura).
VOLUME_RANGE_BY_LABEL = {
    "Inodoro (cisterna)": (2.0, 15.0),
    "Inodoro / descarga": (0.3, 20.0),
    "Ducha": (5.0, 300.0),
    "Grifo / lavamanos": (0.01, 15.0),
    "Grifo (uso prolongado)": (1.0, 40.0),
    "Goteo / fuga": (0.0, 60.0),
}
