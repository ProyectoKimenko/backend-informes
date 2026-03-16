import numpy as np
import pandas as pd
from typing import Tuple, Dict, List
from itertools import combinations
from sklearn.cluster import KMeans


# -----------------------------------------------------------------------------
# 1) HEURÍSTICA DE NOMBRES
# -----------------------------------------------------------------------------
def get_device_category(flow_rate: float) -> str:
    if flow_rate < 0.5:
        return "Ruido"
    if flow_rate < 2.0:
        return "Fuga / Goteo"
    if flow_rate < 5.0:
        return "Grifo Baño"
    if flow_rate < 10.0:
        return "Ducha / Grifo Cocina"
    if flow_rate < 18.0:
        return "Inodoro / Lavadora"
    return "Riego / Consumo Alto"


# -----------------------------------------------------------------------------
# 2) PREPROCESAMIENTO
# -----------------------------------------------------------------------------
def preprocess_signal(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.set_index("timestamp")

    df = df.sort_index()

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be DatetimeIndex")

    if "flow" not in df.columns and "flow_rate" in df.columns:
        df.rename(columns={"flow_rate": "flow"}, inplace=True)
    elif "flow" not in df.columns:
        raise KeyError("Debe existir columna flow o flow_rate")

    df["flow_smooth"] = (
        df["flow"]
        .rolling(window=5, center=True, min_periods=1)
        .median()
        .fillna(df["flow"])
    )

    df["delta"] = df["flow_smooth"].diff().fillna(0)

    return df


# -----------------------------------------------------------------------------
# 3) ENTRENAMIENTO v1.2
# -----------------------------------------------------------------------------
def train_disaggregator(df: pd.DataFrame) -> Dict[int, Dict]:
    df_proc = preprocess_signal(df)

    stable_mask = (
        (df_proc["delta"].abs() < 0.2) &
        (df_proc["flow_smooth"] > 0.5)
    )

    stable_values = df_proc.loc[stable_mask, "flow_smooth"].values.reshape(-1, 1)

    if len(stable_values) <= 10:
        return {}

    n_clusters = min(5, len(stable_values) // 50 + 1)

    kmeans = KMeans(
        n_clusters=n_clusters,
        n_init=10,
        random_state=42
    )
    kmeans.fit(stable_values)

    centers = sorted(kmeans.cluster_centers_.flatten())

    profiles: Dict[int, Dict] = {}

    for i, mean_val in enumerate(centers):
        if mean_val < 0.5:
            continue

        name = f"{get_device_category(mean_val)}_{i}"

        profiles[i] = {
            "name": name,
            "mean_flow": round(float(mean_val), 2),
            "tol": float(max(1.0, float(mean_val) * 0.30)),
        }

    return profiles


# -----------------------------------------------------------------------------
# 4) DETECCIÓN DE SEGMENTOS ACTIVOS
# -----------------------------------------------------------------------------
def detect_active_segments(
    flow_series: pd.Series,
    min_flow: float = 0.5,
    min_duration_seconds: int = 5
) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    is_active = flow_series > min_flow
    changes = is_active.astype(int).diff()

    starts = flow_series.index[changes == 1]
    ends = flow_series.index[changes == -1]

    if len(is_active) > 0 and bool(is_active.iloc[0]):
        starts = starts.insert(0, flow_series.index[0])

    if len(ends) < len(starts):
        ends = ends.append(pd.Index([flow_series.index[-1]]))

    segments = []
    for s, e in zip(starts, ends):
        duration = (e - s).total_seconds()
        if duration >= min_duration_seconds:
            segments.append((s, e))

    return segments


# -----------------------------------------------------------------------------
# 5) SELECCIÓN DE COMBINACIÓN POR SEGMENTO
# -----------------------------------------------------------------------------
def find_best_profile_combo(
    avg_flow: float,
    profiles: Dict,
    max_devices_on: int = 3
) -> List[Dict]:
    """
    Busca la mejor combinación de perfiles para explicar el flujo promedio
    de un segmento. Devuelve la lista de perfiles elegidos.
    """
    profile_list = list(profiles.values())

    if not profile_list:
        return []

    all_candidates = []

    max_r = min(max_devices_on, len(profile_list))

    for r in range(1, max_r + 1):
        for combo in combinations(profile_list, r):
            combo_sum = sum(float(p["mean_flow"]) for p in combo)
            combo_error = abs(avg_flow - combo_sum)

            # Penalización leve por complejidad para no meter perfiles de más
            score = combo_error + 0.15 * r

            all_candidates.append((score, combo))

    if not all_candidates:
        return []

    best_combo = min(all_candidates, key=lambda x: x[0])[1]
    return list(best_combo)


# -----------------------------------------------------------------------------
# 6) DESAGREGACIÓN v1.2
# -----------------------------------------------------------------------------
def run_disaggregation(
    df: pd.DataFrame,
    profiles: Dict,
    max_devices_on: int = 3,
    min_flow: float = 0.5,
    min_duration_seconds: int = 5
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    if not profiles:
        raise ValueError("No trained profiles provided.")

    df_proc = preprocess_signal(df)
    current_flow = df_proc["flow_smooth"]

    device_names = [p["name"] for p in profiles.values()]
    all_columns = device_names + ["No Detectado"]

    df_result = pd.DataFrame(
        0.0,
        index=current_flow.index,
        columns=all_columns
    )

    total_original_liters = current_flow.sum() / 60.0
    print(f"[AUDIT] Total original litros: {total_original_liters:.4f}")

    segments = detect_active_segments(
        current_flow,
        min_flow=min_flow,
        min_duration_seconds=min_duration_seconds
    )

    events_list = []

    for s, e in segments:
        segment = current_flow.loc[s:e]
        avg_flow = float(segment.mean())

        best_combo = find_best_profile_combo(
            avg_flow=avg_flow,
            profiles=profiles,
            max_devices_on=max_devices_on
        )

        if not best_combo:
            df_result.loc[s:e, "No Detectado"] = segment.values
            continue

        combo_sum = sum(float(p["mean_flow"]) for p in best_combo)

        if combo_sum <= 0:
            df_result.loc[s:e, "No Detectado"] = segment.values
            continue

        assigned_total = np.zeros(len(segment), dtype=float)

        # Reparto proporcional del segmento entre los perfiles seleccionados
        for prof in best_combo:
            name = prof["name"]
            mean_i = float(prof["mean_flow"])

            assigned_values = segment.values * (mean_i / combo_sum)
            df_result.loc[s:e, name] = assigned_values
            assigned_total += assigned_values

            # evento por dispositivo para este segmento
            duration = (e - s).total_seconds()
            volume_liters = float(assigned_values.sum() / 60.0)
            avg_assigned_flow = float(np.mean(assigned_values))

            if duration >= min_duration_seconds and volume_liters > 0:
                events_list.append({
                    "device": name,
                    "start_time": s,
                    "end_time": e,
                    "duration_seconds": float(duration),
                    "flow_rate": avg_assigned_flow,
                    "volume_liters": volume_liters,
                })

        # Residuo del segmento
        residual = segment.values - assigned_total
        residual = np.where(residual > 0.01, residual, 0.0)

        if residual.sum() > 0:
            df_result.loc[s:e, "No Detectado"] += residual

    # Todo lo que quedó fuera de segmentos activos pero tiene flujo -> No Detectado
    covered_mask = df_result.sum(axis=1) > 0
    residual_mask = (~covered_mask) & (current_flow > 0)

    df_result.loc[residual_mask, "No Detectado"] = current_flow[residual_mask]

    assigned_flow = df_result[device_names].sum().sum() / 60.0
    unassigned_flow = df_result["No Detectado"].sum() / 60.0

    print(f"[AUDIT] Assigned litros: {assigned_flow:.4f}")
    print(f"[AUDIT] Unassigned litros: {unassigned_flow:.4f}")

    df_events = pd.DataFrame(events_list)

    if not df_events.empty:
        # unir eventos duplicados exactos del mismo dispositivo/segmento
        df_events = (
            df_events
            .groupby(
                ["device", "start_time", "end_time", "duration_seconds"],
                as_index=False
            )
            .agg({
                "flow_rate": "mean",
                "volume_liters": "sum"
            })
        )
        total_events_liters = df_events["volume_liters"].sum()
    else:
        total_events_liters = 0.0

    print(f"[AUDIT] Total litros en eventos: {total_events_liters:.4f}")

    diff = total_original_liters - (total_events_liters + unassigned_flow)
    print(f"[AUDIT] Diferencia real (original - eventos - no_detectado): {diff:.4f}")

    return df_events, df_result