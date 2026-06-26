from typing import Optional, Dict, Any, List
from supabase import create_client, Client
import pandas as pd
import os

from pipeline.segmentation import integrate_volume

_supabase: Optional[Client] = None


def get_supabase() -> Client:
    """Cliente singleton de Supabase para el BACKEND.

    Prefiere SUPABASE_SERVICE_ROLE_KEY (bypassa RLS y permite escribir
    perfiles/eventos/readings) y cae a SUPABASE_KEY (anon) solo si no está, para
    compatibilidad. Con esta credencial el backend sigue operando aunque se
    revoquen los grants de anon y se active RLS en Supabase (cierre del agujero
    por el que la anon key pública podía DELETE/TRUNCATE las tablas).
    """
    global _supabase
    if _supabase is None:
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_KEY")

        if not url or not key:
            raise RuntimeError("Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY/SUPABASE_KEY")

        _supabase = create_client(url, key)

    return _supabase


def fetch_measurements(place_id: int, start_time: str, end_time: str, batch_size: int = 10000) -> pd.DataFrame:
    """
    Obtiene mediciones usando paginación por offset (más robusta que por cursor).

    La paginación anterior usaba el último timestamp como cursor, lo que causaba
    cortes prematuros cuando el filtro de deduplicación reducía el batch por debajo
    de batch_size — el loop interpretaba eso como "no hay más datos" aunque hubiera
    decenas de miles de filas restantes.

    La paginación por offset (range) no tiene ese problema: itera en bloques fijos
    independientemente del contenido, y termina solo cuando Supabase devuelve 0 filas.

    Args:
        place_id: ID del lugar
        start_time: Timestamp inicial (ISO format)
        end_time: Timestamp final (ISO format)
        batch_size: Tamaño de lote por request

    Returns:
        DataFrame con columnas [timestamp, flow_rate, place_id, ...]
        con timestamps únicos, ordenado ascendente.
    """
    sb = get_supabase()

    all_rows = []
    offset = 0

    while True:
        response = (
            sb.table("measurements_realtime")
            .select("*")
            .eq("place_id", place_id)
            .gte("timestamp", start_time)
            .lte("timestamp", end_time)
            .order("timestamp")
            .range(offset, offset + batch_size - 1)
            .execute()
        )

        rows = response.data

        if not rows:
            break

        all_rows.extend(rows)
        print(f"[fetch] place_id={place_id} offset={offset} filas={len(rows)} acumulado={len(all_rows)}")

        # Si devolvió menos del batch completo no hay más páginas
        if len(rows) < batch_size:
            break

        offset += batch_size

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"])
    df = df.drop_duplicates(subset=["timestamp", "place_id"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    print(f"[fetch] place_id={place_id} total={len(df)} filas únicas | {df['timestamp'].min()} → {df['timestamp'].max()}")

    return df


def get_official_profiles(place_id: int) -> List[Dict[str, Any]]:
    """
    Obtiene solo los perfiles marcados como oficiales para congelar la detección.
    
    Args:
        place_id: ID del lugar
        
    Returns:
        Lista de diccionarios con perfiles oficiales
    """
    sb = get_supabase()
    
    try:
        res = sb.table("disaggregation_profiles")\
            .select("*")\
            .eq("place_id", place_id)\
            .eq("is_official", True)\
            .execute()
        
        return res.data if res.data else []
    
    except Exception as e:
        print(f"[ERROR] Failed to fetch profiles for place_id={place_id}: {e}")
        return []


def save_official_profiles(place_id: int, profiles: Dict[int, Dict]) -> None:
    """
    Guarda perfiles marcándolos como oficiales.

    Compatible con perfiles v1, v2 y v3. Siempre incluye tol (NOT NULL en DB).
    Los campos v3 (tol_nd, centroid_nd, scaler_path, mean_duration, label)
    se escriben solo si están presentes en el perfil.

    Args:
        place_id: ID del lugar
        profiles: Diccionario {id: {name, mean_flow, st_deviation, weight, tol?, ...}}
    """
    sb = get_supabase()

    if not profiles:
        raise ValueError("No profiles to save")

    records = []
    for _, info in profiles.items():
        required_fields = ["name", "mean_flow", "st_deviation", "weight"]
        if not all(k in info for k in required_fields):
            raise ValueError(f"Invalid profile format. Missing fields in: {info}")

        mean_flow = float(info["mean_flow"])

        # tol siempre presente — fallback a 30% del mean_flow si no viene calculado
        tol = float(info["tol"]) if info.get("tol") is not None else float(max(1.0, mean_flow * 0.30))

        record: Dict[str, Any] = {
            "place_id":     place_id,
            "name":         info["name"],
            "label":        info.get("label") or None,
            "mean_flow":    mean_flow,
            "st_deviation": float(info["st_deviation"]),
            "weight":       float(info["weight"]),
            "tol":          tol,
            "is_official":  True,
        }

        # Campos v3 — solo si están presentes
        if info.get("mean_duration") is not None:
            record["mean_duration"] = float(info["mean_duration"])
        if info.get("tol_nd") is not None:
            record["tol_nd"] = float(info["tol_nd"])
        if info.get("centroid_nd"):
            record["centroid_nd"] = [float(v) for v in info["centroid_nd"]]
        if info.get("scaler_path"):
            record["scaler_path"] = str(info["scaler_path"])

        records.append(record)

    try:
        # Retirar los oficiales actuales marcándolos inactivos (UPDATE) en lugar de
        # DELETE: get_official_profiles filtra is_official=True, así que los viejos
        # dejan de usarse sin borrarlos. Esto vuelve al backend libre de DELETE en
        # TODO su flujo, lo que permite REVOCARLE DELETE/TRUNCATE al rol anon (que el
        # backend y el scraper comparten) — cerrando el riesgo de que la anon key
        # pública (va en el frontend) destruya datos, sin necesitar la service_role.
        retired = sb.table("disaggregation_profiles") \
            .update({"is_official": False}) \
            .eq("place_id", place_id) \
            .eq("is_official", True) \
            .execute()

        n_retired = len(retired.data) if retired.data else 0
        print(f"[Profiles] Retired {n_retired} old profiles (is_official=False) for place_id={place_id}")

        sb.table("disaggregation_profiles").insert(records).execute()

        labeled = sum(1 for r in records if r.get("label"))
        has_nd  = sum(1 for r in records if r.get("tol_nd") is not None)
        print(
            f"[Profiles] place_id={place_id} — saved {len(records)} profiles "
            f"({labeled} labeled, {has_nd} with ND clustering)"
        )

    except Exception as e:
        print(f"[ERROR] Failed to save profiles for place_id={place_id}: {e}")
        raise


def _build_no_detected_events(
    place_id: int,
    df_result: pd.DataFrame,
    min_volume_l: float = 0.02,
    min_duration_s: float = 5.0,
) -> List[Dict]:
    """
    Construye eventos de No Detectado desde el residual de df_result.
    """
    if "No Detectado" not in df_result.columns:
        return []

    no_det    = df_result["No Detectado"]
    is_active = no_det > 0.05
    changes   = is_active.astype(int).diff()

    starts = df_result.index[changes == 1]
    ends   = df_result.index[changes == -1]

    if len(is_active) > 0 and bool(is_active.iloc[0]):
        starts = starts.insert(0, df_result.index[0])
    if len(ends) < len(starts):
        ends = ends.append(pd.Index([df_result.index[-1]]))

    records = []
    for s, e in zip(starts, ends):
        seg      = no_det[s:e]
        duration = (e - s).total_seconds()
        avg_flow = float(seg.mean())
        volume_l = integrate_volume(seg)   # litros con Δt real (antes seg.sum()/60)

        if duration < min_duration_s or volume_l < min_volume_l:
            continue

        records.append({
            "place_id":      place_id,
            "device_name":   "No Detectado",
            "start_time":    s.isoformat(),
            "end_time":      e.isoformat(),
            "duration_s":    float(duration),
            "flow_rate":     avg_flow,
            "volume_liters": round(volume_l, 6),
        })

    return records


def save_disaggregation_result(
    place_id: int,
    df_events: pd.DataFrame,
    df_result: pd.DataFrame,
) -> None:
    """
    Guarda eventos de desagregacion incluyendo No Detectado.

    Combina los eventos de dispositivos identificados (df_events) con los
    periodos de flujo no asignado (df_result[No Detectado]) en una sola
    insercion a disaggregation_events. Esto permite que backfill_disaggregated_readings
    incluya No Detectado automaticamente en disaggregated_readings.
    """
    sb = get_supabase()

    device_records = []

    if not df_events.empty:
        if "volume_liters" not in df_events.columns:
            df_events["volume_liters"] = (
                df_events["flow_rate"] * (df_events["duration_seconds"] / 60)
            )

        for _, row in df_events.iterrows():
            device_records.append({
                "place_id":      place_id,
                "device_name":   row["device"],
                "start_time":    row["start_time"].isoformat(),
                "end_time":      row["end_time"].isoformat(),
                "duration_s":    float(row["duration_seconds"]),
                "flow_rate":     float(row["flow_rate"]),
                "volume_liters": float(row["volume_liters"]),
            })

    no_det_records = _build_no_detected_events(place_id, df_result)

    all_records = device_records + no_det_records

    if not all_records:
        print(f"[Disaggregation] place_id={place_id} No events to save.")
        return

    try:
        sb.table("disaggregation_events").upsert(
            all_records,
            on_conflict="place_id,device_name,start_time,end_time"
        ).execute()

        vol_devices = sum(r["volume_liters"] for r in device_records)
        vol_no_det  = sum(r["volume_liters"] for r in no_det_records)

        print(
            f"[Disaggregation] place_id={place_id} — "
            f"saved {len(device_records)} device events ({vol_devices:.3f}L) + "
            f"{len(no_det_records)} no-detectado events ({vol_no_det:.3f}L)"
        )

    except Exception as e:
        print(f"[ERROR] Failed to save events for place_id={place_id}: {e}")
        raise


def get_all_places() -> List[Dict[str, Any]]:
    """
    Obtiene todos los lugares registrados.
    
    Returns:
        Lista de diccionarios con información de lugares
    """
    sb = get_supabase()
    
    try:
        result = sb.table("places").select("*").execute()
        return result.data or []
    
    except Exception as e:
        print(f"[ERROR] Failed to fetch places: {e}")
        return []


def bulk_insert_dissagregated(data: list) -> None:
    """
    Inserta datos desagregados en lotes con manejo robusto de errores.
    
    Mejoras:
    - Conteo de registros insertados exitosamente
    - Identificación de lotes fallidos
    - Propagación de errores
    
    Args:
        data: Lista de diccionarios con {place_id, time_bucket, category, volume_liters}
    """
    if not data:
        print("[Bulk Insert] No data to insert")
        return
    
    sb = get_supabase()
    batch_size = 1000
    total_inserted = 0
    failed_batches = []
    
    try:
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            
            try:
                sb.table("disaggregated_readings").upsert(
                    batch, 
                    on_conflict="place_id,time_bucket,category"
                ).execute()
                
                total_inserted += len(batch)
                
            except Exception as batch_error:
                print(f"[WARNING] Batch {i}-{i+batch_size} failed: {batch_error}")
                failed_batches.append(i)
        
        # ✅ Reportar resultado
        if failed_batches:
            raise Exception(
                f"Bulk insert partially failed. "
                f"Inserted {total_inserted}/{len(data)} rows. "
                f"Failed batches: {failed_batches}"
            )
        
        print(f"[Bulk Insert] Successfully inserted {total_inserted} rows")
        
    except Exception as e:
        print(f"[ERROR] Bulk insert failed: {e}")
        raise  # ✅ Propagar para que Celery lo maneje


def get_stackplot_data(
    place_id: int | str, 
    start_time: Optional[str] = None, 
    end_time: Optional[str] = None, 
    granularity: str = "day",
    max_points: int = 100000  # ✅ NUEVO: Límite de seguridad
) -> pd.DataFrame:
    """
    Obtiene datos para stackplot con protección contra OOM.
    
    Mejoras:
    - Paginación para conjuntos grandes
    - Límite de seguridad (max_points)
    - Validación de timestamps
    
    Args:
        place_id: ID del lugar
        start_time: Timestamp inicial (opcional)
        end_time: Timestamp final (opcional)
        granularity: Granularidad temporal ("hour", "day", "week", "month")
        max_points: Máximo número de puntos a retornar (protección OOM)
        
    Returns:
        DataFrame con índice temporal y columnas por categoría
    """
    sb = get_supabase()

    query = (
        sb.table("disaggregated_readings")
        .select("time_bucket, category, volume_liters")
        .eq("place_id", place_id)
    )

    if start_time:
        query = query.gte("time_bucket", start_time)
    
    if end_time:
        query = query.lte("time_bucket", end_time)

    # ✅ Paginación para conjuntos grandes
    all_data = []
    offset = 0
    page_size = 10000
    
    while True:
        try:
            response = query.order("time_bucket").range(offset, offset + page_size - 1).execute()
        except Exception as e:
            print(f"[ERROR] Failed to fetch stackplot data at offset {offset}: {e}")
            break
        
        if not response.data:
            break
        
        all_data.extend(response.data)
        
        # ✅ Protección contra OOM
        if len(all_data) >= max_points:
            print(f"[WARNING] Reached max_points limit ({max_points}). Consider narrowing time range.")
            break
        
        if len(response.data) < page_size:
            break
        
        offset += page_size

    if not all_data:
        return pd.DataFrame()

    df = pd.DataFrame(all_data)
    
    # ✅ Conversión robusta de timestamps
    df["time_bucket"] = pd.to_datetime(df["time_bucket"], utc=True, errors='coerce')
    df = df.dropna(subset=['time_bucket'])  # Eliminar timestamps inválidos

    # Pivot
    df_pivot = (
        df.pivot_table(
            index="time_bucket", 
            columns="category", 
            values="volume_liters", 
            aggfunc="sum"
        )
        .fillna(0)
    )

    # Resample
    freq_map = {
        "hour": "h",
        "day": "D",
        "week": "W",
        "month": "ME"
    }

    freq = freq_map.get(granularity, "D")
    df_resampled = df_pivot.resample(freq).sum()

    # Ordenar columnas por total (mayor a menor)
    total_by_device = df_resampled.sum().sort_values(ascending=False)
    df_resampled = df_resampled[total_by_device.index]

    return df_resampled


def audit_volume_conservation(place_id: int, start_time: str, end_time: str) -> Dict[str, float]:
    """
    Audita conservación de volumen en todo el pipeline.
    
    Compara:
    1. Volumen original de measurements
    2. Volumen en disaggregation_events
    3. Volumen en disaggregated_readings
    
    Args:
        place_id: ID del lugar
        start_time: Timestamp inicial
        end_time: Timestamp final
        
    Returns:
        Diccionario con métricas de auditoría
    """
    sb = get_supabase()
    
    try:
        # 1. Volumen original de measurements
        df_measurements = fetch_measurements(place_id, start_time, end_time)
        
        if df_measurements.empty:
            print(f"[AUDIT] No measurements found for place_id={place_id}")
            return {
                "original_volume": 0,
                "events_volume": 0,
                "readings_volume": 0,
                "events_discrepancy_pct": 0,
                "readings_discrepancy_pct": 0
            }
        
        # Detectar columna de flujo e integrar con Δt real (antes sum()/60 asumía
        # cadencia fija de 1/min y sesgaba el volumen "original" de referencia).
        flow_col = 'flow_rate' if 'flow_rate' in df_measurements.columns else 'flow'
        _ts = pd.to_datetime(df_measurements['timestamp'], utc=True, errors='coerce')
        _flow_series = (
            pd.Series(df_measurements[flow_col].astype(float).values, index=_ts)
            .dropna()
            .sort_index()
        )
        original_volume = integrate_volume(_flow_series)
        
        # 2. Volumen en eventos
        events = sb.table("disaggregation_events")\
            .select("volume_liters")\
            .eq("place_id", place_id)\
            .gte("start_time", start_time)\
            .lte("end_time", end_time)\
            .execute()
        
        events_volume = sum(e['volume_liters'] for e in events.data) if events.data else 0
        
        # 3. Volumen en disaggregated_readings
        readings = sb.table("disaggregated_readings")\
            .select("volume_liters")\
            .eq("place_id", place_id)\
            .gte("time_bucket", start_time)\
            .lte("time_bucket", end_time)\
            .execute()
        
        readings_volume = sum(r['volume_liters'] for r in readings.data) if readings.data else 0
        
        # Calcular discrepancias
        events_discrepancy = abs(original_volume - events_volume) / original_volume * 100 if original_volume > 0 else 0
        readings_discrepancy = abs(original_volume - readings_volume) / original_volume * 100 if original_volume > 0 else 0
        
        # Logging
        print(f"\n{'='*60}")
        print(f"[VOLUME AUDIT] place_id={place_id}")
        print(f"  Time range: {start_time} to {end_time}")
        print(f"  Original (measurements): {original_volume:.4f} L")
        print(f"  Events: {events_volume:.4f} L (diff: {events_discrepancy:.2f}%)")
        print(f"  Readings: {readings_volume:.4f} L (diff: {readings_discrepancy:.2f}%)")
        print(f"{'='*60}\n")
        
        return {
            "original_volume": float(original_volume),
            "events_volume": float(events_volume),
            "readings_volume": float(readings_volume),
            "events_discrepancy_pct": float(events_discrepancy),
            "readings_discrepancy_pct": float(readings_discrepancy)
        }
    
    except Exception as e:
        print(f"[ERROR] Volume audit failed for place_id={place_id}: {e}")
        raise