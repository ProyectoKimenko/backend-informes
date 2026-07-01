"""
Preprocesado y segmentación de eventos de la señal de caudal de agua.

Módulo de utilidades compartido por el desagregador. Detecta eventos de consumo
(tramos activos de caudal) y extrae sus features. La lógica de modelo
(clustering, asignación) vive en pipeline/disaggregator_simple.py.
"""
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd


def preprocess_signal(df: pd.DataFrame) -> pd.DataFrame:
    """Indexa por timestamp, normaliza la columna de caudal y suaviza."""
    df = df.copy()

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df = df.dropna(subset=["timestamp"]).set_index("timestamp")

    df = df.sort_index()

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be DatetimeIndex")

    if "flow" not in df.columns and "flow_rate" in df.columns:
        df.rename(columns={"flow_rate": "flow"}, inplace=True)
    elif "flow" not in df.columns:
        raise KeyError("Debe existir columna 'flow' o 'flow_rate'")

    df["flow_smooth"] = (
        df["flow"]
        .rolling(window=5, center=True, min_periods=1)
        .median()
        .fillna(df["flow"])
    )
    df["delta"] = df["flow_smooth"].diff().fillna(0.0)

    return df


# Features extraídas por evento. El VOLUMEN integrado y el CV intra-evento son los
# discriminadores físicos clave: el inodoro tiene VOLUMEN fijo de cisterna (CV bajo),
# la ducha volumen alto sostenido, el grifo volumen chico. Antes solo se usaban
# mean_flow+duration (rangos US) e ignoraban el volumen.
EVENT_FEATURES = [
    "mean_flow",
    "duration_s",
    "volume_liters",
    "peak_flow",
    "cv_flow",
    "hour_sin",
    "hour_cos",
]


def _split_long_segment(
    seg: pd.Series,
    delta_threshold: float = 1.0,
    min_subsegment_seconds: int = 5,
) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    """Parte un segmento activo en subsegmentos si hay cambios internos fuertes."""
    if len(seg) <= 3:
        return [(seg.index[0], seg.index[-1])]

    # Umbral de corte ADAPTATIVO (relativo al caudal del segmento), no absoluto. Un
    # umbral fijo de 1.0 L/min parte una ducha (~9 L/min con ruido ±1) en trozos
    # espurios (~18% de las "duchas" quedaban fragmentadas, con duraciones irreales).
    # Escalarlo con la mediana del caudal hace que solo las transiciones REALES
    # (cambio de artefacto, que pasa por caudal bajo) corten, no el ruido interno.
    # Piso en delta_threshold para no perder transiciones en caudales bajos.
    eff_threshold = max(delta_threshold, 0.4 * float(seg.median()))

    local_delta = seg.diff().fillna(0.0).abs()
    cut_points = seg.index[local_delta > eff_threshold]

    if len(cut_points) == 0:
        return [(seg.index[0], seg.index[-1])]

    boundaries = [seg.index[0]] + list(cut_points) + [seg.index[-1]]
    subsegments = []

    for i in range(len(boundaries) - 1):
        s = boundaries[i]
        e = boundaries[i + 1]
        duration = (e - s).total_seconds()
        if duration >= min_subsegment_seconds:
            subsegments.append((s, e))

    if not subsegments:
        return [(seg.index[0], seg.index[-1])]

    return subsegments


def detect_composite(
    seg: "pd.Series",
    min_flow: float = 0.5,
    min_hold_s: float = 8.0,
    step_abs: float = 1.5,
    step_frac: float = 0.35,
) -> bool:
    """¿Es este tramo activo un USO CONCURRENTE (superposición de >=2 fixtures)?

    Firma de concurrencia (step-pairing / level-counting; US10838434B2, Pastor-Jabaloyes
    et al., Water 2018): sobre un caudal base sostenido aparece una JOROBA — el caudal
    sube a una meseta más alta, se mantiene, y vuelve a bajar a un nivel sostenido
    parecido al base (un segundo artefacto se abrió y cerró mientras el primero seguía).
    Esa meseta interior MÁS ALTA que sus dos vecinas (ambas por encima del basal) no se
    explica como un único fixture (que sube, mantiene UN nivel y baja), sino como
    superposición.

    Distingue concurrencia de eventos SECUENCIALES: dos usos pegados pero NO simultáneos
    bajan hacia el basal entre medio (valle bajo) y la derivada ya los separa en
    ``_split_long_segment``; aquí el "valle" entre mesetas se mantiene ALTO.

    Conservador a propósito (alta precisión): exige >=3 mesetas SOSTENIDAS
    (>= ``min_hold_s``) con escalones significativos (>= max(step_abs, step_frac·nivel)).
    La variabilidad normal de una ducha (que ``cv_flow`` ya capta) NO lo dispara, ni el
    ajuste de caudal de una ducha (sube/baja monótono, sin joroba). LÍMITE conocido: dos
    fixtures que arrancan SIMULTÁNEAMENTE (sin hombro) son indistinguibles de un único
    fixture de caudal alto con un solo sensor — no se detectan, y es el techo físico.
    """
    if seg is None or len(seg) < 6:
        return False
    t = seg.index
    v0 = np.asarray(seg.to_numpy(dtype=float), dtype=float)
    n = len(v0)
    # Denoise ligero (no ancho: un suavizado ancho convierte el escalón en rampa).
    try:
        v = seg.rolling(window=3, center=True, min_periods=1).median().to_numpy(dtype=float)
    except Exception:
        v = v0
    half = max(3, int(round(min_hold_s / 2.0)))
    if n < 2 * half + 2:
        return False

    # Fronteras de ESCALÓN por diferencia de medianas pre/post (robusto a rampas y a
    # ruido: una rampa marca su frontera en el medio; el ruido se promedia). Tras una
    # frontera saltamos `half` para no marcar el mismo flanco varias veces.
    bnds = [0]
    i = half
    while i < n - half:
        pre = float(np.median(v[i - half:i]))
        post = float(np.median(v[i:i + half]))
        thr = max(step_abs, step_frac * max(min(pre, post), 1e-9))
        if abs(post - pre) > thr:
            bnds.append(i)
            i += half
        else:
            i += 1
    bnds.append(n)

    # Mesetas SOSTENIDAS entre fronteras (duración real >= min_hold_s).
    sustained: List[Tuple[float, int, int]] = []
    for k in range(len(bnds) - 1):
        a, b = bnds[k], bnds[k + 1] - 1
        if b < a:
            continue
        if (t[b] - t[a]).total_seconds() >= min_hold_s:
            sustained.append((float(np.median(v[a:b + 1])), a, b))
    if len(sustained) < 3:
        return False

    # ¿Hay una meseta INTERIOR sostenida más alta que sus dos vecinas (joroba)?
    for k in range(1, len(sustained) - 1):
        lvl_prev = sustained[k - 1][0]
        lvl_cur = sustained[k][0]
        lvl_next = sustained[k + 1][0]
        up = lvl_cur - lvl_prev
        down = lvl_cur - lvl_next
        thr_up = max(step_abs, step_frac * max(min(lvl_cur, lvl_prev), 1e-9))
        thr_dn = max(step_abs, step_frac * max(min(lvl_cur, lvl_next), 1e-9))
        # Joroba real + vecinos que son uso genuino (no caudal basal): superposición.
        if up >= thr_up and down >= thr_dn and min(lvl_prev, lvl_next) >= 1.5 * min_flow:
            return True
    return False


def _split_by_gaps(seg: "pd.Series", gap_threshold_s: float = 45.0) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    """Corta un tramo activo donde hay un HUECO de datos (Δt entre muestras
    consecutivas > gap_threshold_s). Si el scraper se cae, un uso activo antes y otro
    después del hueco quedan unidos en UN evento cuya duración incluye el hueco (p.ej.
    +2 h) → feature corrupta → mal clasificado ("Riego"/"Goteo") y persistido para
    siempre. La cadencia real es ~1 Hz (p99 ~1.7 s), así que 45 s no corta datos
    legítimos y sí atrapa caídas reales. El volumen ya no se infla (gap_cap en
    integrate_volume), pero la duración sí: por eso se corta aquí."""
    n = len(seg)
    if n == 0:
        return []
    if n < 2:
        return [(seg.index[0], seg.index[-1])]
    idx = seg.index
    dt = idx.to_series().diff().dt.total_seconds().to_numpy()
    cuts = [0] + [i for i in range(1, n) if dt[i] > gap_threshold_s] + [n]
    ranges: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
    for k in range(len(cuts) - 1):
        a, b = cuts[k], cuts[k + 1] - 1
        if b > a:
            ranges.append((idx[a], idx[b]))
    return ranges


def segment_events(
    df_proc: pd.DataFrame,
    min_flow: float = 0.5,
    split_internal_changes: bool = True,
    delta_split_threshold: float = 1.0,
    gap_threshold_s: float = 45.0,
) -> pd.DataFrame:
    """Segmenta la señal en eventos y extrae features por evento."""
    flow = df_proc["flow_smooth"]
    is_active = flow > min_flow
    changes = is_active.astype(int).diff()

    starts = df_proc.index[changes == 1]
    ends = df_proc.index[changes == -1]

    if len(is_active) > 0 and bool(is_active.iloc[0]):
        starts = starts.insert(0, df_proc.index[0])
    if len(ends) < len(starts):
        ends = ends.append(pd.Index([df_proc.index[-1]]))

    rows = []

    for s0, e0 in zip(starts, ends):
        # Primero cortar por HUECOS de datos: un tramo "activo" que cruza una caída del
        # scraper no es un evento continuo. Cada trozo sin hueco se procesa aparte.
        for s, e in _split_by_gaps(flow[s0:e0], gap_threshold_s=gap_threshold_s):
            base_seg = flow[s:e]
            duration = (e - s).total_seconds()

            if duration < 5 or base_seg.mean() < min_flow:
                continue

            # ¿Superposición de fixtures concurrentes? Se evalúa sobre el tramo activo
            # COMPLETO, antes de partir: el split por derivada cortaría la joroba en
            # subsegmentos con caudal combinado (igual de inflados y mal-etiquetados).
            is_comp = detect_composite(base_seg, min_flow)

            if is_comp:
                # Concurrente: se emite ENTERO y marcado (se excluye del entrenamiento y
                # va a la categoría "Uso simultáneo" en inferencia, no a un fixture).
                segment_ranges = [(s, e)]
            elif split_internal_changes:
                segment_ranges = _split_long_segment(
                    base_seg,
                    delta_threshold=delta_split_threshold,
                    min_subsegment_seconds=5,
                )
            else:
                segment_ranges = [(s, e)]

            for ss, ee in segment_ranges:
                seg = flow[ss:ee]
                duration = (ee - ss).total_seconds()

                if duration < 5 or seg.mean() < min_flow:
                    continue

                hour = ss.hour
                mflow = float(seg.mean())
                sflow = float(seg.std()) if len(seg) > 1 else 0.0
                vals = np.asarray(seg.values, dtype=float)
                peak = float(seg.max())

                # Caudal MODAL: el caudal más frecuente dentro del evento (binneado a
                # 0.5 L/min). Discrimina fixtures de caudal nominal FIJO (válvula de
                # inodoro, cabezal de ducha) mejor que la media — feature estándar de
                # Autoflow/CIWS (Nguyen/Stewart/Beal; Attallah 2021, ~98% con peak+modo).
                if len(vals):
                    binned = np.round(vals * 2.0) / 2.0
                    uq, cnt = np.unique(binned, return_counts=True)
                    modal_flow = float(uq[int(np.argmax(cnt))])
                else:
                    modal_flow = mflow

                # Gradientes de los flancos (forma rise/plateau/fall del evento).
                k = min(3, len(vals) - 1) if len(vals) > 1 else 1
                rise_grad = float((vals[k] - vals[0]) / k) if len(vals) > 1 else 0.0
                fall_grad = float((vals[-1] - vals[-1 - k]) / k) if len(vals) > 1 else 0.0

                rows.append({
                    "mean_flow": mflow,
                    "duration_s": float(duration),
                    # VOLUMEN integrado por Δt real (litros): el discriminador físico #1.
                    "volume_liters": integrate_volume(seg),
                    "peak_flow": peak,
                    "modal_flow": modal_flow,              # caudal nominal (Autoflow/CIWS)
                    "peak_to_mean": (peak / mflow) if mflow > 0 else 1.0,
                    "rise_grad": rise_grad,
                    "fall_grad": fall_grad,
                    # CV intra-evento: bajo = caudal estable (fixture automático/cisterna),
                    # alto = caudal variable (uso manual de grifo).
                    "cv_flow": (sflow / mflow) if mflow > 0 else 0.0,
                    "std_flow": sflow,
                    "hour_sin": float(np.sin(2 * np.pi * hour / 24)),
                    "hour_cos": float(np.cos(2 * np.pi * hour / 24)),
                    # True = superposición de fixtures concurrentes (no separable con un
                    # solo sensor): se excluye del entrenamiento y se marca "Uso simultáneo".
                    "is_composite": bool(is_comp),
                    "start_time": ss,
                    "end_time": ee,
                })

    return pd.DataFrame(rows)


# Umbrales mínimos para descartar ruido al construir eventos finales.
# Solo se mantiene un mínimo absoluto de 5s y 20mL para descartar spikes.
_EVENT_FILTER_RANGES: List[Dict] = [
    {"max_flow": 0.5,          "min_duration_s": 0,   "min_volume_l": 0.0},   # ruido puro
    {"max_flow": 2.0,          "min_duration_s": 30,  "min_volume_l": 0.02},  # fugas/goteo
    {"max_flow": float("inf"), "min_duration_s": 5,   "min_volume_l": 0.02},  # resto
]


def is_valid_event(mean_flow: float, duration_s: float, volume_l: float) -> bool:
    for bucket in _EVENT_FILTER_RANGES:
        if mean_flow < bucket["max_flow"]:
            return (
                duration_s >= bucket["min_duration_s"] and
                volume_l >= bucket["min_volume_l"]
            )
    return True


def integrate_volume(series: "pd.Series", gap_cap_s: float = 5.0) -> float:
    """Litros de un tramo de caudal integrando con el Δt REAL entre muestras.

    volumen = Σ flow_i(L/min) · Δt_i(s) / 60. Reemplaza el viejo ``serie.sum()/60``
    que asumía cadencia fija de 1 muestra/min y sesgaba el volumen (~+6.8% en la
    señal realtime ~1 Hz; muy errado en la legacy a 30 s). El Δt se deriva del
    índice temporal; se acota a ``max(gap_cap_s, 3·mediana)`` para que un HUECO de
    datos (no una cadencia legítima) no integre un volumen espurio. Adaptativo:
    a 1 Hz el cap efectivo es 5 s, a 30 s es ~90 s.
    """
    if series is None or len(series) == 0:
        return 0.0
    vals = np.asarray(series.to_numpy(dtype=float))
    try:
        # .copy() porque to_numpy() puede devolver una vista de solo-lectura.
        dt = np.array(series.index.to_series().diff().dt.total_seconds().to_numpy(), dtype=float)
    except (AttributeError, TypeError):
        # Índice no temporal: fallback conservador a cadencia unitaria (1 s).
        return float(np.nansum(vals) / 60.0)
    med = np.nanmedian(dt[1:]) if len(dt) > 1 else np.nan
    if not np.isfinite(med) or med <= 0:
        med = 1.0
    dt[0] = med  # la primera muestra no tiene Δt previo → usa la mediana
    cap = max(gap_cap_s, 3.0 * med)
    dt = np.clip(np.nan_to_num(dt, nan=med), 0.0, cap)
    return float(np.nansum(vals * dt) / 60.0)
