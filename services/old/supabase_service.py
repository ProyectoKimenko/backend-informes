from typing import Optional, Dict, Any, List
from supabase import create_client, Client
import pandas as pd
import os

_supabase: Optional[Client] = None


def get_supabase() -> Client:
    """Obtiene o crea una instancia singleton del cliente de Supabase."""
    global _supabase
    if _supabase is None:
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_KEY")

        if not url or not key:
            raise RuntimeError("Missing SUPABASE_URL or SUPABASE_KEY")

        _supabase = create_client(url, key)

    return _supabase


def fetch_measurements(place_id: int, start_time: str, end_time: str, batch_size: int = 10000) -> pd.DataFrame:
    """
    Obtiene mediciones con paginación robusta que evita duplicados.
    
    Correcciones:
    - Exclusión explícita del último timestamp procesado
    - Validación de timestamps
    - Eliminación de duplicados
    
    Args:
        place_id: ID del lugar
        start_time: Timestamp inicial (ISO format)
        end_time: Timestamp final (ISO format)
        batch_size: Tamaño de lote para paginación
        
    Returns:
        DataFrame con columnas [timestamp, flow_rate, place_id, ...]
    """
    sb = get_supabase()

    all_rows = []
    current_start = start_time
    last_timestamp = None

    while True:
        query = (
            sb.table("measurements_realtime")
            .select("*")
            .eq("place_id", place_id)
            .gte("timestamp", current_start)
            .lte("timestamp", end_time)
            .order("timestamp")
            .limit(batch_size)
        )

        response = query.execute()
        rows = response.data

        if not rows:
            break

        # ✅ CORRECCIÓN: Filtrar duplicados del límite anterior
        if last_timestamp:
            rows = [r for r in rows if r["timestamp"] > last_timestamp]
        
        if not rows:
            break

        all_rows.extend(rows)

        # ✅ Si obtuvimos menos del batch_size, terminamos
        if len(rows) < batch_size:
            break

        # ✅ Actualizar para siguiente iteración
        last_timestamp = rows[-1]["timestamp"]
        current_start = last_timestamp

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    
    # ✅ Conversión robusta de timestamps
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors='coerce')
    
    # ✅ Eliminar registros con timestamps inválidos
    df = df.dropna(subset=['timestamp'])
    
    # ✅ Eliminar duplicados explícitos (por si acaso)
    df = df.drop_duplicates(subset=['timestamp', 'place_id'])
    
    df = df.sort_values("timestamp")

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
    Guarda perfiles marcándolos como oficiales (Training Mode).
    
    Mejoras:
    - Validación previa de datos
    - Manejo robusto de errores
    - Logging detallado
    
    Args:
        place_id: ID del lugar
        profiles: Diccionario de perfiles {id: {name, mean_flow, st_deviation, weight}}
    """
    sb = get_supabase()
    
    # ✅ Validar perfiles antes de tocar la DB
    if not profiles:
        raise ValueError("No profiles to save")
    
    records = []
    for _, info in profiles.items():
        # Validación de campos requeridos
        required_fields = ["name", "mean_flow", "st_deviation", "weight"]
        if not all(k in info for k in required_fields):
            raise ValueError(f"Invalid profile format. Missing fields in: {info}")
        
        records.append({
            "place_id": place_id,
            "name": info["name"],
            "mean_flow": float(info["mean_flow"]),
            "st_deviation": float(info["st_deviation"]),
            "weight": float(info["weight"]),
            "is_official": True
        })
    
    try:
        # ✅ Borrar oficiales anteriores
        delete_response = sb.table("disaggregation_profiles")\
            .delete()\
            .eq("place_id", place_id)\
            .eq("is_official", True)\
            .execute()
        
        print(f"[Profiles] Deleted {len(delete_response.data) if delete_response.data else 0} old profiles")
        
        # ✅ Insertar nuevos
        insert_response = sb.table("disaggregation_profiles").insert(records).execute()
        
        print(f"[Profiles] place_id={place_id} Saved {len(records)} official profiles")
        
    except Exception as e:
        print(f"[ERROR] Failed to save profiles for place_id={place_id}: {e}")
        raise


def save_disaggregation_result(place_id: int, df_events: pd.DataFrame, df_result: pd.DataFrame) -> None:
    """
    Guarda eventos de desagregación con validación y volumen.
    
    Correcciones:
    - Ahora guarda volume_liters (CRÍTICO)
    - Manejo de errores con propagación
    - Validación de datos
    
    Args:
        place_id: ID del lugar
        df_events: DataFrame con eventos [device, start_time, end_time, duration_seconds, flow_rate, volume_liters]
        df_result: DataFrame con series temporales (no usado actualmente)
    """
    sb = get_supabase()

    if df_events.empty:
        print(f"[Disaggregation] place_id={place_id} No events to save.")
        return

    # ✅ Validar que volume_liters existe
    if 'volume_liters' not in df_events.columns:
        print(f"[WARNING] volume_liters not found in df_events. This should not happen.")
        # Calcular volume_liters si no existe (fallback)
        df_events['volume_liters'] = df_events['flow_rate'] * (df_events['duration_seconds'] / 60)

    event_records = []

    for _, row in df_events.iterrows():
        event_records.append({
            "place_id": place_id,
            "device_name": row["device"],
            "start_time": row["start_time"].isoformat(),
            "end_time": row["end_time"].isoformat(),
            "duration_s": float(row["duration_seconds"]),
            "flow_rate": float(row["flow_rate"]),
            "volume_liters": float(row["volume_liters"])  # ✅ AÑADIDO
        })

    # ✅ Manejo de errores con logging
    try:
        sb.table("disaggregation_events").upsert(
            event_records, 
            on_conflict="place_id,device_name,start_time,end_time"
        ).execute()
        
        # Calcular volumen total para logging
        total_volume = sum(r['volume_liters'] for r in event_records)
        print(f"[Disaggregation] place_id={place_id} Saved events={len(event_records)}, total_volume={total_volume:.4f}L")
        
    except Exception as e:
        print(f"[ERROR] Failed to save events for place_id={place_id}: {e}")
        raise  # ✅ Propagar error para que Celery lo maneje


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
        
        # Detectar columna de flujo
        flow_col = 'flow_rate' if 'flow_rate' in df_measurements.columns else 'flow'
        original_volume = df_measurements[flow_col].sum() / 60
        
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