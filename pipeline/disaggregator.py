import os
import joblib
import numpy as np
import pandas as pd
from typing import Tuple, Dict, List, Optional
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


SCALER_DIR = os.getenv("SCALER_DIR", "/tmp/nilm_scalers")
os.makedirs(SCALER_DIR, exist_ok=True)


# -----------------------------------------------------------------------------
# 1) NOMBRE NEUTRO DE DISPOSITIVO
# -----------------------------------------------------------------------------
def get_device_name(cluster_id: int, mean_flow: float) -> str:
    return f"Dispositivo_{cluster_id} ({mean_flow:.2f} L/min)"


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
# 3) FEATURES DE EVENTO
# -----------------------------------------------------------------------------
EVENT_FEATURES = [
    "mean_flow",
    "duration_s",
    "std_flow",
    "peak_flow",
    "delta_rise",
    "hour_sin",
    "hour_cos",
]


# -----------------------------------------------------------------------------
# 4) SEGMENTACIÓN REFINADA
# -----------------------------------------------------------------------------
def _split_long_segment(
    seg: pd.Series,
    delta_threshold: float = 1.0,
    min_subsegment_seconds: int = 5
) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    """
    Parte un segmento activo en subsegmentos si hay cambios internos fuertes
    de nivel. Usa la derivada interna del segmento.
    """
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
    delta_split_threshold: float = 1.0
) -> pd.DataFrame:
    """
    Segmenta la señal en eventos y extrae features.
    Si split_internal_changes=True, parte segmentos largos por cambios internos.
    """
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
                min_subsegment_seconds=5
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


# -----------------------------------------------------------------------------
# 5) k ÓPTIMO
# -----------------------------------------------------------------------------
def find_optimal_k(X: np.ndarray, k_range: range = range(2, 8)) -> int:
    best_k = k_range.start
    best_score = -1.0

    for k in k_range:
        if len(X) < k * 10:
            break

        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(X)

        if len(set(labels)) < 2:
            continue

        score = silhouette_score(X, labels)
        print(f"  [Silhouette] k={k} → score={score:.4f}")

        if score > best_score:
            best_score = score
            best_k = k

    # Mínimo 4: con 3 es frecuente que un gap real quede sin perfil propio.
    # El silhouette en espacio ND puede subestimar k si los datos son pocos.
    best_k = max(4, best_k)
    print(f"  [Silhouette] k óptimo: {best_k} (score={best_score:.4f})")
    return best_k


# -----------------------------------------------------------------------------
# 6) RADIO DEL CLUSTER
# -----------------------------------------------------------------------------
def compute_cluster_radii(
    X_norm: np.ndarray,
    labels: np.ndarray,
    centroids_norm: np.ndarray,
    percentile: float = 98.0,
) -> Dict[int, float]:
    """
    Radio por cluster = percentil 98 de distancias al centroide.
    """
    radii = {}
    for k in range(len(centroids_norm)):
        mask = labels == k
        if mask.sum() == 0:
            radii[k] = 1.0
            continue

        dists = np.linalg.norm(X_norm[mask] - centroids_norm[k], axis=1)
        radii[k] = float(np.percentile(dists, percentile))
        print(f"  [Radio cluster {k}] p{percentile:.0f} dist = {radii[k]:.4f}")

    return radii


# -----------------------------------------------------------------------------
# 7) TOLERANCIAS 1D ADAPTATIVAS (fallback)
# -----------------------------------------------------------------------------
def compute_adaptive_tolerances(profiles: Dict[int, Dict]) -> Dict[int, Dict]:
    if len(profiles) <= 1:
        for key in profiles:
            mean = profiles[key]["mean_flow"]
            profiles[key]["tol"] = float(max(0.5, mean * 0.25))
        return profiles

    sorted_items = sorted(profiles.items(), key=lambda x: x[1]["mean_flow"])

    for i, (key, prof) in enumerate(sorted_items):
        mean = prof["mean_flow"]
        if i == 0:
            tol = (sorted_items[i + 1][1]["mean_flow"] - mean) / 2 * 0.85
        elif i == len(sorted_items) - 1:
            tol = (mean - sorted_items[i - 1][1]["mean_flow"]) / 2 * 0.85
        else:
            gap_l = (mean - sorted_items[i - 1][1]["mean_flow"]) / 2
            gap_r = (sorted_items[i + 1][1]["mean_flow"] - mean) / 2
            tol = min(gap_l, gap_r) * 0.85

        profiles[key]["tol"] = float(max(0.5, tol))

    return profiles


# -----------------------------------------------------------------------------
# 8) PRIOR TEMPORAL NEUTRAL
# -----------------------------------------------------------------------------
def get_time_prior(mean_flow: float, hour: int) -> float:
    """
    Temporalmente neutral para aumentar cobertura.
    """
    return 1.0


# -----------------------------------------------------------------------------
# 9) FILTRO DE EVENTOS
# -----------------------------------------------------------------------------
# Umbrales reducidos para capturar más eventos reales.
# El clustering ya filtra ruido estructuralmente — no necesitamos umbrales altos aquí.
# Solo se mantiene un mínimo absoluto de 5s y 0.02L para descartar spikes eléctricos.
_EVENT_FILTER_RANGES: List[Dict] = [
    {"max_flow": 0.5,          "min_duration_s": 0,   "min_volume_l": 0.0},   # ruido puro
    {"max_flow": 2.0,          "min_duration_s": 30,  "min_volume_l": 0.02},  # fugas/goteo: 30s mínimo
    {"max_flow": float("inf"), "min_duration_s": 5,   "min_volume_l": 0.02},  # resto: solo 5s y 20mL
]


def is_valid_event(mean_flow: float, duration_s: float, volume_l: float) -> bool:
    for bucket in _EVENT_FILTER_RANGES:
        if mean_flow < bucket["max_flow"]:
            return (
                duration_s >= bucket["min_duration_s"] and
                volume_l >= bucket["min_volume_l"]
            )
    return True


# -----------------------------------------------------------------------------
# 10) CONSTRUCCIÓN DE EVENTOS
# -----------------------------------------------------------------------------
def _build_events(
    df_result: pd.DataFrame,
    device_names: List[str],
) -> pd.DataFrame:
    events_list = []

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
            seg = df_result.loc[s:e, col]
            duration = (e - s).total_seconds()
            avg_flow = float(seg.mean())
            volume_l = float(seg.sum() / 60.0)

            if not is_valid_event(avg_flow, duration, volume_l):
                continue

            events_list.append({
                "device": col,
                "start_time": s,
                "end_time": e,
                "duration_seconds": float(duration),
                "flow_rate": avg_flow,
                "volume_liters": volume_l,
            })

    return pd.DataFrame(events_list)


# -----------------------------------------------------------------------------
# 11A) RELLENO DE GAPS ENTRE PERFILES
# -----------------------------------------------------------------------------
def fill_profile_gaps(profiles: Dict[int, Dict], min_gap: float = 1.5) -> Dict[int, Dict]:
    """
    Detecta gaps entre perfiles adyacentes y añade un perfil sintético en el centro.

    Un gap ocurre cuando dos perfiles vecinos (ordenados por mean_flow) tienen
    una separación > min_gap L/min. En esa zona hay flujo real que ningún cluster
    captura, terminando en "No Detectado".

    Los perfiles sintéticos:
    - No tienen centroid_nd (la inferencia ND los ignora).
    - Usan solo fallback 1D con tol = gap/2 * 0.9.
    - Se marcan con label=None y weight=0 para distinguirlos de los reales.
    - Se guardan en Supabase igual que los demás perfiles.

    Args:
        profiles: Diccionario de perfiles reales generados por el clustering.
        min_gap:  Separación mínima en L/min para crear un perfil sintético.

    Returns:
        El mismo diccionario con perfiles sintéticos añadidos donde haya gaps.
    """
    if len(profiles) < 2:
        return profiles

    sorted_items = sorted(profiles.items(), key=lambda x: x[1]["mean_flow"])
    max_key = max(profiles.keys()) + 1
    synthetic_count = 0

    for i in range(len(sorted_items) - 1):
        mean_low  = sorted_items[i][1]["mean_flow"]
        mean_high = sorted_items[i + 1][1]["mean_flow"]
        gap = mean_high - mean_low

        if gap > min_gap:
            mid = round((mean_low + mean_high) / 2, 2)
            tol = round(gap / 2 * 0.9, 3)

            print(
                f"  [GapFill] Gap {mean_low:.2f}-{mean_high:.2f} L/min "
                f"({gap:.2f} L/min) → perfil sintético en {mid:.2f} L/min "
                f"tol={tol:.3f}"
            )

            profiles[max_key] = {
                "name":          get_device_name(max_key, mid),
                "label":         None,
                "mean_flow":     mid,
                "mean_duration": 30.0,
                "st_deviation":  round(gap / 4, 2),
                "weight":        0.0,
                "tol":           tol,
                "tol_nd":        0.0,        # sin centroid — solo fallback 1D
                "centroid_nd":   [],
                "scaler_path":   "",
            }
            max_key += 1
            synthetic_count += 1

    if synthetic_count:
        print(f"[Train] {synthetic_count} perfiles sintéticos añadidos para cubrir gaps.")

    return profiles


# -----------------------------------------------------------------------------
# 11) ENTRENAMIENTO
# -----------------------------------------------------------------------------
def train_disaggregator(df: pd.DataFrame) -> Dict[int, Dict]:
    df_proc = preprocess_signal(df)
    df_events = segment_events(df_proc)

    if len(df_events) < 20:
        print(f"[Train] Solo {len(df_events)} eventos — insuficiente para entrenar.")
        return {}

    print(f"[Train] {len(df_events)} eventos segmentados para clustering.")

    X_raw = df_events[EVENT_FEATURES].values

    scaler = StandardScaler()
    X_norm = scaler.fit_transform(X_raw)

    print(f"[Train] Buscando k óptimo en espacio {len(EVENT_FEATURES)}D...")
    n_clusters = find_optimal_k(X_norm)

    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    kmeans.fit(X_norm)

    labels = kmeans.labels_
    centroids_norm = kmeans.cluster_centers_
    centroids_raw = scaler.inverse_transform(centroids_norm)

    radii = compute_cluster_radii(X_norm, labels, centroids_norm, percentile=98.0)

    profiles: Dict[int, Dict] = {}

    for i in range(n_clusters):
        mean_val = float(centroids_raw[i][EVENT_FEATURES.index("mean_flow")])
        dur_val = float(centroids_raw[i][EVENT_FEATURES.index("duration_s")])

        if mean_val < 0.5:
            continue

        mask = labels == i
        cluster_flows = df_events.loc[mask, "mean_flow"].values
        std_val = float(np.std(cluster_flows)) if mask.sum() > 1 else 0.1
        std_val = max(std_val, 0.1)
        weight_val = float(mask.sum() / len(df_events))
        name = get_device_name(i, mean_val)

        profiles[i] = {
            "name": name,
            "label": None,
            "mean_flow": round(mean_val, 2),
            "mean_duration": round(dur_val, 1),
            "st_deviation": round(std_val, 2),
            "weight": round(weight_val, 6),
            "tol": float(max(1.0, mean_val * 0.30)),
            "tol_nd": round(radii[i], 4),
            "centroid_nd": centroids_norm[i].tolist(),
            "scaler_path": "",
        }

    if not profiles:
        print("[Train] Todos los clusters tienen mean_flow < 0.5.")
        return {}

    profiles = compute_adaptive_tolerances(profiles)

    # MEJORA: cubrir gaps entre perfiles con perfiles sintéticos.
    # Cuando hay un gap > 1.5 L/min entre dos perfiles adyacentes, existe flujo
    # real sin cluster propio que cae en "No Detectado". Un perfil sintético
    # centrado en el gap lo captura usando solo fallback 1D (no tiene centroid_nd).
    profiles = fill_profile_gaps(profiles, min_gap=1.5)

    place_id_hint = abs(hash(str(df_proc.index[0]))) % 100000
    scaler_path = os.path.join(SCALER_DIR, f"scaler_{place_id_hint}.joblib")
    joblib.dump(scaler, scaler_path)
    print(f"[Train] Scaler guardado en {scaler_path}")

    for key in profiles:
        profiles[key]["scaler_path"] = scaler_path

    print("\n[Train] Perfiles finales:")
    for p in profiles.values():
        print(
            f"  {p['name']} | "
            f"mean={p['mean_flow']} L/min | "
            f"dur~={p['mean_duration']}s | "
            f"tol_nd={p['tol_nd']:.4f} | "
            f"tol_1d={p['tol']:.3f}"
        )

    return profiles


# -----------------------------------------------------------------------------
# 12) LOAD SCALER
# -----------------------------------------------------------------------------
def _load_scaler(profiles: Dict) -> Optional[StandardScaler]:
    for prof in profiles.values():
        path = prof.get("scaler_path", "")
        if path and os.path.exists(path):
            return joblib.load(path)
    return None


# -----------------------------------------------------------------------------
# 13) RUN MAIN
# -----------------------------------------------------------------------------
def run_disaggregation(
    df: pd.DataFrame,
    profiles: Dict,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not profiles:
        raise ValueError("No trained profiles provided.")

    df_proc = preprocess_signal(df)
    current_flow = df_proc["flow_smooth"]
    scaler = _load_scaler(profiles)

    if scaler is None:
        print("[WARN] Scaler no encontrado — usando inferencia 1D (fallback v2).")
        return _run_disaggregation_1d(df_proc, current_flow, profiles)

    return _run_disaggregation_nd(df_proc, current_flow, profiles, scaler)


# -----------------------------------------------------------------------------
# 14) INFERENCIA ND
# -----------------------------------------------------------------------------
def _run_disaggregation_nd(
    df_proc: pd.DataFrame,
    current_flow: pd.Series,
    profiles: Dict,
    scaler: StandardScaler,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    device_names = [p.get("label") or p["name"] for p in profiles.values()]
    df_result = pd.DataFrame(0.0, index=current_flow.index, columns=device_names + ["No Detectado"])

    total_original_liters = current_flow.sum() / 60.0
    print(f"[AUDIT] Total original litros: {total_original_liters:.4f}")

    residual = current_flow.copy()
    df_seg = segment_events(df_proc)

    if df_seg.empty:
        print("[DEBUG] No se segmentaron eventos en inferencia.")
        df_result["No Detectado"] = residual
        return pd.DataFrame(), df_result

    print(f"[DEBUG] Eventos segmentados en inferencia: {len(df_seg)}")

    X_inf_raw = df_seg[EVENT_FEATURES].values
    X_inf_norm = scaler.transform(X_inf_raw)

    # Separar perfiles con centroid_nd real (del clustering) de los sintéticos
    # (fill_profile_gaps). Los sintéticos no tienen centroid_nd y solo participan
    # en el fallback 1D.
    all_profiles_sorted = sorted(
        profiles.values(),
        key=lambda x: x.get("mean_flow", 0.0),
        reverse=True,
    )

    nd_profiles   = [p for p in all_profiles_sorted if p.get("centroid_nd")]
    fb_profiles   = [p for p in all_profiles_sorted if not p.get("centroid_nd")]

    centroids = np.array([p["centroid_nd"] for p in nd_profiles])
    tols_nd   = np.array([p["tol_nd"]      for p in nd_profiles])
    means_nd  = np.array([p.get("mean_flow", 0.0) for p in nd_profiles])
    names_nd  = [p.get("label") or p["name"] for p in nd_profiles]

    # Datos para fallback 1D (perfiles reales + sintéticos)
    means_1d  = np.array([p.get("mean_flow", 0.0) for p in all_profiles_sorted])
    tols_1d   = np.array([p.get("tol", max(1.0, 0.30 * p.get("mean_flow", 1.0))) for p in all_profiles_sorted])
    names_1d  = [p.get("label") or p["name"] for p in all_profiles_sorted]

    accepted_nd = 0
    accepted_fb = 0
    rejected_events = 0
    event_debug_rows = []

    for ev_idx, ev_row in df_seg.iterrows():
        x_norm   = X_inf_norm[ev_idx]
        ev_mean  = float(ev_row["mean_flow"])
        hour     = ev_row["start_time"].hour

        accepted  = False
        reason    = "rejected"
        best_name = "No Detectado"
        best_mean = ev_mean

        # ── Paso 1: matching ND sobre perfiles con centroid ─────────────────
        if len(centroids) > 0:
            dists    = np.linalg.norm(centroids - x_norm, axis=1)
            best_idx = int(np.argmin(dists))
            best_dist = float(dists[best_idx])
            best_tol  = float(tols_nd[best_idx])

            # Soft assignment: aceptar si está dentro del radio,
            # o si es el candidato claro (margen al segundo > 30% de la distancia)
            dists_sorted = np.sort(dists)
            margin = dists_sorted[1] - dists_sorted[0] if len(dists_sorted) > 1 else best_dist
            tol_eff = best_tol * 1.8  # radio generoso para cobertura

            if best_dist <= tol_eff or (margin > 0.3 * best_dist and best_dist < tol_eff * 2.5):
                best_name = names_nd[best_idx]
                best_mean = float(means_nd[best_idx])
                accepted  = True
                reason    = "nd"
                accepted_nd += 1

        # ── Paso 2: fallback 1D si ND no asignó ─────────────────────────────
        if not accepted:
            diffs_1d = np.abs(means_1d - ev_mean)
            fb_idx   = int(np.argmin(diffs_1d))
            fb_diff  = float(diffs_1d[fb_idx])
            fb_tol   = float(tols_1d[fb_idx]) * 1.5

            if fb_diff <= fb_tol:
                best_name = names_1d[fb_idx]
                best_mean = float(means_1d[fb_idx])
                accepted  = True
                reason    = "fallback_1d"
                accepted_fb += 1

        if not accepted:
            rejected_events += 1

        event_debug_rows.append({
            "event_id":        int(ev_idx),
            "mean_flow_event": round(ev_mean, 3),
            "duration_s":      float(ev_row["duration_s"]),
            "best_profile":    best_name,
            "best_mean":       round(best_mean, 3),
            "accepted":        accepted,
            "reason":          reason,
        })

        if not accepted:
            continue

        s = ev_row["start_time"]
        e = ev_row["end_time"]

        seg_residual  = residual[s:e]
        assigned_vals = np.minimum(seg_residual.values, best_mean)

        df_result.loc[s:e, best_name] += assigned_vals
        residual[s:e] = (seg_residual - assigned_vals).clip(lower=0)

    df_result["No Detectado"] = residual

    print(f"[DEBUG] Eventos aceptados ND:          {accepted_nd}")
    print(f"[DEBUG] Eventos aceptados fallback 1D: {accepted_fb}")
    print(f"[DEBUG] Eventos rechazados:            {rejected_events}")

    debug_df = pd.DataFrame(event_debug_rows)
    if not debug_df.empty and rejected_events > 0:
        rejected_df = debug_df[~debug_df["accepted"]]
        print("[DEBUG] Eventos rechazados:")
        for _, row in rejected_df.head(10).iterrows():
            print(
                f"  event={row['event_id']} "
                f"mean={row['mean_flow_event']:.2f} L/min "
                f"dur={row['duration_s']:.1f}s "
                f"best_profile={row['best_profile']}"
            )

    assigned_flow = df_result[device_names].sum().sum() / 60.0
    unassigned_flow = df_result["No Detectado"].sum() / 60.0
    reconstructed = assigned_flow + unassigned_flow

    print(f"[AUDIT] Assigned litros:    {assigned_flow:.4f}")
    print(f"[AUDIT] Unassigned litros:  {unassigned_flow:.4f}")
    print(f"[AUDIT] Total reconstruido: {reconstructed:.4f}")

    if total_original_liters > 0:
        disc_pct = abs(total_original_liters - reconstructed) / total_original_liters * 100
        status = "OK" if disc_pct <= 1.0 else "REVISAR"
        print(f"[AUDIT] Conservacion de masa: {status} ({disc_pct:.3f}%)")

    df_events = _build_events(df_result, device_names)

    total_events_liters = float(df_events["volume_liters"].sum()) if not df_events.empty else 0.0
    diff = total_original_liters - (total_events_liters + unassigned_flow)
    print(f"[AUDIT] Total litros en eventos: {total_events_liters:.4f}")
    print(f"[AUDIT] Diferencia (original - eventos - no_detectado): {diff:.4f}")

    return df_events, df_result


# -----------------------------------------------------------------------------
# 15) FALLBACK 1D
# -----------------------------------------------------------------------------
def _run_disaggregation_1d(
    df_proc: pd.DataFrame,
    current_flow: pd.Series,
    profiles: Dict,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    print("[FALLBACK] Usando inferencia 1D.")
    device_names = [p.get("label") or p["name"] for p in profiles.values()]
    df_result = pd.DataFrame(0.0, index=current_flow.index, columns=device_names + ["No Detectado"])
    residual = current_flow.copy()

    sorted_profiles = sorted(
        profiles.values(),
        key=lambda x: x.get("mean_flow", 0.0),
        reverse=True,
    )

    for prof in sorted_profiles:
        name = prof.get("label") or prof["name"]
        mean = float(prof.get("mean_flow", 0.0))
        tol = float(prof.get("tol", max(1.0, 0.30 * mean)))

        active = (residual >= mean - tol) & (residual <= mean + tol)
        assigned = pd.Series(0.0, index=residual.index)
        assigned[active] = np.minimum(residual[active], mean)
        df_result[name] = assigned
        residual = (residual - assigned).clip(lower=0)

    df_result["No Detectado"] = residual
    return _build_events(df_result, device_names), df_result