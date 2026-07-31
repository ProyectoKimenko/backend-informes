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

import math

UNCLASSIFIED = "Sin clasificar"

# Margen del veto de frecuencia: un cluster puede tener hasta FREQ_VETO_FACTOR veces
# más eventos/día que los usos/día declarados del artefacto antes de descartarse como
# candidato. Generoso a propósito (cubre días peak, splits de eventos y error de la
# declaración del operador); solo mata desajustes GROSEROS — p.ej. un cluster con 40
# eventos/día jamás es la tina declarada con 1 uso/día.
FREQ_VETO_FACTOR = 8.0

# Fracción mínima de eventos del cluster dentro de la ventana horaria declarada del
# artefacto para que este sea candidato. 0.5: la mitad de los eventos fuera del
# horario de la cocina => ese cluster no es el lavaplatos.
HOURS_MIN_FRACTION = 0.5


def _valid_windows(windows):
    """Filtra ventanas horarias bien formadas: pares (a, b) con a, b en [0, 24] y
    a != b. a > b significa cruce de medianoche ([22, 2] = 22:00-02:00). Lo
    malformado se ignora; si NADA es válido se trata como SIN restricción (fail-open:
    un dato corrupto en el catastro no debe vetar un artefacto las 24 horas)."""
    out = []
    for w in (windows or []):
        try:
            a, b = float(w[0]), float(w[1])
        except (TypeError, ValueError, IndexError):
            continue
        # NaN falla toda comparación => queda excluido aquí mismo.
        if not (0 <= a <= 24 and 0 <= b <= 24) or a == b:
            continue
        out.append((a, b))
    return out


def hour_in_windows(hour: float, windows) -> bool:
    """¿Cae la hora local (0-24, fraccional) dentro de alguna ventana [ini, fin]?

    [ini, fin) con ini < fin; ini > fin cruza medianoche ([22, 2] = 22:00-02:00).
    Sin ventanas declaradas (o ninguna válida) => sin restricción horaria.
    """
    vw = _valid_windows(windows)
    if not vw:
        return True
    for a, b in vw:
        if a < b:
            if a <= hour < b:
                return True
        elif hour >= a or hour < b:
            return True
    return False


def _hours_fraction(local_hours, windows) -> float:
    """Fracción de horas (array de horas locales de eventos) dentro de las ventanas."""
    if not _valid_windows(windows) or local_hours is None or len(local_hours) == 0:
        return 1.0
    inside = sum(1 for h in local_hours if hour_in_windows(float(h), windows))
    return inside / len(local_hours)


def label_by_fixtures(median_flow, median_duration, median_volume, cv_volume, fixtures,
                      events_per_day=None, local_hours=None):
    """Etiqueta un cluster por el ARTEFACTO DECLARADO más cercano del inventario del
    recinto, en vez de las bandas heurísticas de label_by_signature.

    fixtures: lista de dicts {label, flow_lmin, volume_l, count?, uses_per_day?, hours?}
    declarados por el operador. Match por caudal + volumen (log) normalizados por error
    RELATIVO — así "Ducha 9 L/min ~40 L" ancla los clusters de ese caudal/volumen a la
    firma REAL del recinto. Si nada cae dentro de tolerancia → UNCLASSIFIED (honesto).
    Devuelve None si no hay inventario, para que el caller use label_by_signature.

    Información adicional del catastro (opcional, ignorada si no se declara):
      - events_per_day: eventos/día del cluster. Veto de FRECUENCIA contra el
        `uses_per_day` declarado (usos/día esperados del TOTAL de unidades `count`):
        un cluster con muchos más eventos/día que los usos plausibles del artefacto
        no puede ser ese artefacto (p.ej. la tina, 1 uso/día).
      - local_hours: horas locales (0-24) de los eventos del cluster. Veto de HORARIO
        contra `hours` (ventanas [[ini, fin], ...] en hora local del recinto): si la
        mayoría de los eventos cae fuera del horario de operación del artefacto
        (p.ej. cocina 09-23), ese artefacto se descarta como candidato.
    """
    if not fixtures:
        return None
    f, v = median_flow, max(median_volume, 0.0)
    best_label, best_d = None, None
    for fx in fixtures:
        try:
            fx_flow = float(fx.get("flow_lmin") or 0.0)
            fx_vol = float(fx.get("volume_l") or 0.0)
        except (TypeError, ValueError):
            continue
        if fx_flow <= 0:
            continue
        # Veto de FRECUENCIA: el cluster tiene demasiados eventos/día para este artefacto.
        try:
            fx_upd = float(fx.get("uses_per_day") or 0.0)
        except (TypeError, ValueError):
            fx_upd = 0.0
        if events_per_day is not None and fx_upd > 0 and events_per_day > FREQ_VETO_FACTOR * fx_upd:
            continue
        # Veto de HORARIO: la mayoría de los eventos del cluster cae fuera de la
        # ventana de operación declarada del artefacto.
        if _hours_fraction(local_hours, fx.get("hours")) < HOURS_MIN_FRACTION:
            continue
        # error relativo de caudal (primario) + de volumen en log (secundario, cola larga)
        df = (f - fx_flow) / max(fx_flow, 1.0)
        dv = (math.log1p(v) - math.log1p(fx_vol)) / max(math.log1p(fx_vol), 0.5) if fx_vol > 0 else 0.0
        d = math.sqrt(df * df + 0.5 * dv * dv)
        if best_d is None or d < best_d:
            best_d, best_label = d, fx.get("label")
    # tolerancia ~ 60% de error combinado; más allá, no se parece a ningún artefacto real.
    if best_label and best_d is not None and best_d <= 0.6:
        return best_label
    return UNCLASSIFIED


def hours_by_label(fixtures) -> dict:
    """Mapa label -> ventanas horarias declaradas, con semántica de UNIÓN.

    Varias unidades pueden compartir label: la restricción del label es la unión de
    sus ventanas (una ducha disponible 6-10 y otra 18-23 => el label opera en ambas).
    Si ALGUNA unidad del label no declara horario (o no tiene ventanas válidas), el
    label queda SIN restricción — hay una unidad siempre disponible. Sin esto, el
    último fixture pisaba a los demás y el resultado dependía del orden declarado.
    """
    out: dict = {}
    unrestricted = set()
    for fx in (fixtures or []):
        lab = fx.get("label")
        if not lab or lab in unrestricted:
            continue
        vw = _valid_windows(fx.get("hours"))
        if vw:
            out.setdefault(lab, []).extend([a, b] for a, b in vw)
        else:
            unrestricted.add(lab)
            out.pop(lab, None)
    return out

# Evento que es SUPERPOSICIÓN de >=2 fixtures concurrentes (p.ej. ducha + grifo a la
# vez). Con un solo sensor de caudal NO se puede separar de forma fiable, así que se
# marca honestamente con esta categoría en vez de mal-atribuirlo a un fixture (lo que
# inflaba su volumen/caudal y envenenaba el clustering). El operador ve cuánta agua fue
# uso concurrente no separable.
COMPOSITE = "Uso simultáneo"


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
    # Techo bajado de 300 L a 150 L: a ~7 L/min (caudal real medido en el
    # Refugio) 150 L son >20 min de ducha continua, ya generoso. Con 300 L se
    # colaban como "Ducha" llenados de estanque y fugas largas — se observaron
    # eventos de 895 L/92 min y 527 L/7 h etiquetados como ducha.
    "Ducha": (5.0, 150.0),
    "Grifo / lavamanos": (0.01, 15.0),
    "Grifo (uso prolongado)": (1.0, 40.0),
    "Goteo / fuga": (0.0, 60.0),
    # Artefactos declarados en el inventario del recinto (label_by_fixtures).
    "Urinario": (0.3, 6.0),
    "Tina": (40.0, 300.0),
    "Lavaplatos / cocina": (2.0, 60.0),
    # Llenado de acumulador/estanque: caudal sostenido y volumen muy grande.
    # Es un uso REAL del recinto (suministro intermitente por congelamiento),
    # no un artefacto sanitario: separarlo evita inflar la ducha.
    "Llenado de estanque": (150.0, 2000.0),
}
