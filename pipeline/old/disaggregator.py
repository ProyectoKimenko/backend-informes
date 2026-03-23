import numpy as np
import pandas as pd
from typing import Tuple, Dict
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


# -----------------------------------------------------------------------------
# 3) ENTRENAMIENTO v1.0 COMPATIBLE CON save_official_profiles
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

    labels = kmeans.labels_
    centers = sorted(kmeans.cluster_centers_.flatten())

    profiles: Dict[int, Dict] = {}

    for i, mean_val in enumerate(centers):
        if mean_val < 0.5:
            continue

        cluster_vals = stable_values[labels == i].flatten()

        std_val = float(np.std(cluster_vals)) if len(cluster_vals) > 1 else 0.1
        std_val = max(std_val, 0.1)

        weight_val = float(len(cluster_vals) / len(stable_values))

        name = f"{get_device_category(mean_val)}_{i}"

        profiles[i] = {
            "name": name,
            "mean_flow": round(float(mean_val), 2),
            "st_deviation": round(std_val, 2),
            "weight": round(weight_val, 6),
            "tol": float(max(1.0, float(mean_val) * 0.30)),
        }

    print("Perfiles entrenados:")
    for p in profiles.values():
        print(
            f"Dispositivo: {p['name']} | "
            f"Media: {p['mean_flow']} L/min | "
            f"Std: {p['st_deviation']} | "
            f"Peso: {p['weight']} | "
            f"Tol: {p['tol']}"
        )

    return profiles


# -----------------------------------------------------------------------------
# 4) DESAGREGACIÓN v1.0 + NO DETECTADO
# -----------------------------------------------------------------------------
def run_disaggregation(df: pd.DataFrame, profiles: Dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
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

    assigned_mask = pd.Series(False, index=df_proc.index)

    sorted_profiles = sorted(
        profiles.values(),
        key=lambda x: x.get("mean_flow", x.get("mean", 0.0)),
        reverse=True
    )

    for prof in sorted_profiles:
        name = prof["name"]
        mean = float(prof.get("mean_flow", prof.get("mean", 0.0)))

        tol = prof.get("tol", prof.get("tolerance", None))
        if tol is None:
            tol = max(1.0, 0.30 * mean)
        tol = float(tol)

        mask = (
            (current_flow >= (mean - tol)) &
            (current_flow <= (mean + tol)) &
            (~assigned_mask)
        )

        df_result.loc[mask, name] = current_flow[mask]
        assigned_mask |= mask

    # Todo lo no asignado va a "No Detectado"
    df_result.loc[~assigned_mask, "No Detectado"] = current_flow[~assigned_mask]

    assigned_flow = df_result[device_names].sum().sum() / 60.0
    unassigned_flow = df_result["No Detectado"].sum() / 60.0

    print(f"[AUDIT] Assigned litros: {assigned_flow:.4f}")
    print(f"[AUDIT] Unassigned litros: {unassigned_flow:.4f}")

    # -----------------------------------------------------------------------------
    # EVENTOS
    # -----------------------------------------------------------------------------
    events_list = []

    # Solo generamos eventos para perfiles detectados, no para "No Detectado"
    for col in device_names:
        is_active = df_result[col] > 0
        changes = is_active.astype(int).diff()

        starts = df_result.index[changes == 1]
        ends = df_result.index[changes == -1]

        if len(is_active) > 0 and bool(is_active.iloc[0]):
            starts = starts.insert(0, df_result.index[0])

        if len(ends) < len(starts):
            ends = ends.append(pd.Index([df_result.index[-1]]))

        for s, e in zip(starts, ends):
            duration = (e - s).total_seconds()

            if duration < 5:
                continue

            segment = df_result.loc[s:e, col]
            avg_flow = float(segment.mean())
            volume_liters = float(segment.sum() / 60.0)

            events_list.append({
                "device": col,
                "start_time": s,
                "end_time": e,
                "duration_seconds": float(duration),
                "flow_rate": avg_flow,
                "volume_liters": volume_liters,
            })

    df_events = pd.DataFrame(events_list)

    if not df_events.empty:
        total_events_liters = float(df_events["volume_liters"].sum())
    else:
        total_events_liters = 0.0

    print(f"[AUDIT] Total litros en eventos: {total_events_liters:.4f}")

    diff = total_original_liters - (total_events_liters + unassigned_flow)
    print(f"[AUDIT] Diferencia real (original - eventos - no_detectado): {diff:.4f}")

    return df_events, df_result