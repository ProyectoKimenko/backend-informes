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


# Features extraídas por evento (las usa el clustering; el desagregador 2D usa
# mean_flow y duration_s).
EVENT_FEATURES = [
    "mean_flow",
    "duration_s",
    "std_flow",
    "peak_flow",
    "delta_rise",
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

    local_delta = seg.diff().fillna(0.0).abs()
    cut_points = seg.index[local_delta > delta_threshold]

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


def segment_events(
    df_proc: pd.DataFrame,
    min_flow: float = 0.5,
    split_internal_changes: bool = True,
    delta_split_threshold: float = 1.0,
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

    for s, e in zip(starts, ends):
        base_seg = flow[s:e]
        duration = (e - s).total_seconds()

        if duration < 5 or base_seg.mean() < min_flow:
            continue

        if split_internal_changes:
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

            rise = float(
                seg.iloc[min(3, len(seg) - 1)] - seg.iloc[0]
            ) if len(seg) > 1 else 0.0

            hour = ss.hour

            rows.append({
                "mean_flow": float(seg.mean()),
                "duration_s": float(duration),
                "std_flow": float(seg.std()) if len(seg) > 1 else 0.0,
                "peak_flow": float(seg.max()),
                "delta_rise": rise,
                "hour_sin": float(np.sin(2 * np.pi * hour / 24)),
                "hour_cos": float(np.cos(2 * np.pi * hour / 24)),
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
