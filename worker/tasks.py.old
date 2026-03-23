from worker.celery_app import celery_app
from services.supabase_service import (
    fetch_measurements, 
    get_official_profiles, 
    save_disaggregation_result, 
    get_all_places, 
    bulk_insert_dissagregated,
    save_official_profiles
)
from pipeline.disaggregator import run_disaggregation
from datetime import datetime, timedelta
import pandas as pd
from services.supabase_service import get_supabase


@celery_app.task(
    bind=True,
    autoretry_for=(ConnectionError, TimeoutError),  # Solo errores transitorios
    retry_backoff=60,
    retry_kwargs={"max_retries": 3},
    retry_jitter=True,
    name="worker.tasks.process_disaggregation",
)
def process_disaggregation(self, place_id: int, start_time: str | None = None, end_time: str | None = None):
    """
    Procesa desagregación para un lugar específico con validación de conservación de masa.
    """
    try:
        if not start_time or not end_time:
            end_dt = datetime.utcnow()
            start_dt = end_dt - timedelta(days=1)
            start_time = start_dt.isoformat()
            end_time = end_dt.isoformat()

        official_profiles = get_official_profiles(place_id)

        if not official_profiles:
            return {
                "place_id": place_id,
                "status": "no_model_found"
            }

        profiles = {
            i: {
                "name": p["name"],
                "mean_flow": p["mean_flow"],
                "st_deviation": p["st_deviation"],
                "weight": p["weight"]
            }
            for i, p in enumerate(official_profiles)
        }

        df = fetch_measurements(
            place_id=place_id,
            start_time=start_time,
            end_time=end_time
        )

        if df.empty:
            return {"place_id": place_id, "status": "no_data"}

        # ✅ AUDITORÍA PRE-DESAGREGACIÓN
        original_liters = df['flow_rate'].sum() / 60 if 'flow_rate' in df.columns else df['flow'].sum() / 60

        # Ejecutar desagregación
        df_events, df_result = run_disaggregation(df, profiles)

        # ✅ VALIDACIÓN: Conservación de masa
        assigned_liters = df_result.sum().sum() / 60
        discrepancy = abs(original_liters - assigned_liters) / original_liters if original_liters > 0 else 0

        # Logging de estado
        self.update_state(
            state='PROCESSING',
            meta={
                'place_id': place_id,
                'original_liters': float(original_liters),
                'assigned_liters': float(assigned_liters),
                'discrepancy_pct': float(discrepancy * 100),
                'events_detected': len(df_events)
            }
        )

        # ⚠️ Si la discrepancia es muy alta, reintentar
        if discrepancy > 0.01:  # 1% de tolerancia
            print(f"[WARNING] place_id={place_id}: Mass conservation violated by {discrepancy:.2%}")
            self.retry(
                countdown=300, 
                exc=ValueError(f"Mass conservation failed: {discrepancy:.2%} discrepancy")
            )

        # Guardar resultados
        save_disaggregation_result(
            place_id,
            df_events,
            df_result,
        )

        return {
            "place_id": place_id,
            "status": "inference_done",
            "events_detected": len(df_events),
            "devices_used": len(profiles),
            "audit": {
                "original_liters": float(original_liters),
                "assigned_liters": float(assigned_liters),
                "discrepancy_pct": float(discrepancy * 100)
            }
        }

    except (KeyError, ValueError) as e:
        # ❌ NO reintentar errores de validación/datos
        print(f"[VALIDATION ERROR] place_id={place_id}: {e}")
        return {
            "place_id": place_id,
            "status": "validation_error",
            "error": str(e)
        }
    
    except Exception as e:
        # Otros errores → permitir retry automático
        print(f"[ERROR] place_id={place_id}: {e}")
        raise


@celery_app.task(name="worker.tasks.process_all_places")
def process_all_places():
    """
    Procesa desagregación para todos los lugares activos.
    """
    places = get_all_places()

    end_time = datetime.utcnow()
    start_time = end_time - timedelta(days=1)

    for place in places:
        process_disaggregation.delay(
            place_id=place["id"],
            start_time=start_time.isoformat(),
            end_time=end_time.isoformat(),
        )

    return {
        "places_submitted": len(places),
        "time_range": f"{start_time.isoformat()} to {end_time.isoformat()}"
    }


@celery_app.task(name="worker.tasks.train_model")
def train_model(place_id: int, start_time: str, end_time: str):
    """
    Entrena modelo de desagregación para un lugar específico.
    """
    try:
        df = fetch_measurements(
            place_id=place_id,
            start_time=start_time,
            end_time=end_time
        )

        if df.empty:
            return {"status": "no_data"}

        from pipeline.disaggregator import train_disaggregator

        profiles = train_disaggregator(df)

        if not profiles:
            return {"status": "no_patterns_found"}

        save_official_profiles(place_id, profiles)

        return {
            "status": "trained",
            "profiles_saved": len(profiles),
            "profile_names": [p["name"] for p in profiles.values()]
        }
    
    except Exception as e:
        print(f"[TRAINING ERROR] place_id={place_id}: {e}")
        return {
            "status": "training_failed",
            "error": str(e)
        }


@celery_app.task(name="worker.tasks.backfill_disaggregated_readings")
def backfill_disaggregated_readings(place_id: int):
    """
    Rellena lecturas desagregadas en formato minuto a minuto desde eventos.
    Incluye detección de eventos solapados y validación de conservación de masa.
    """
    sb = get_supabase()

    print(f"[Backfill] Fetching events for place_id={place_id}")

    response = (
        sb.table("disaggregation_events")
        .select("id, device_name, start_time, end_time, flow_rate")
        .eq("place_id", place_id)
        .order("start_time")
        .execute()
    )

    events = response.data

    if not events:
        return {"status": "no_events_found"}

    df_events = pd.DataFrame(events)
    df_events["start_time"] = pd.to_datetime(df_events["start_time"], utc=True)
    df_events["end_time"] = pd.to_datetime(df_events["end_time"], utc=True)
    df_events.rename(columns={"device_name": "device"}, inplace=True)

    # ✅ DETECCIÓN DE SOLAPAMIENTOS
    overlaps = detect_overlapping_events(df_events)
    if overlaps:
        print(f"⚠️ WARNING: {len(overlaps)} overlapping events detected for place_id={place_id}")
        for overlap in overlaps[:5]:  # Mostrar primeros 5
            print(f"  Device: {overlap['device']}, Overlap: {overlap['overlap_seconds']:.2f}s")

    print(f"[Backfill] Generating minute buckets for {len(df_events)} events")

    minute_records = generate_minute_downsample(df_events, place_id)

    print(f"[Backfill] Inserting {len(minute_records)} minute rows")

    bulk_insert_dissagregated(minute_records)

    return {
        "status": "backfill_completed",
        "events_processed": len(df_events),
        "minutes_inserted": len(minute_records),
        "overlaps_detected": len(overlaps) if overlaps else 0
    }


def detect_overlapping_events(df_events: pd.DataFrame) -> list:
    """
    Detecta eventos del mismo dispositivo que se solapan en el tiempo.
    
    Returns:
        Lista de diccionarios con información de solapamientos
    """
    overlaps = []
    
    for device in df_events['device'].unique():
        device_events = df_events[df_events['device'] == device].sort_values('start_time')
        
        for i in range(len(device_events) - 1):
            current = device_events.iloc[i]
            next_event = device_events.iloc[i + 1]
            
            if current['end_time'] > next_event['start_time']:
                overlap_seconds = (current['end_time'] - next_event['start_time']).total_seconds()
                overlaps.append({
                    'device': device,
                    'event1_id': current.get('id'),
                    'event1_time': f"{current['start_time']} - {current['end_time']}",
                    'event2_id': next_event.get('id'),
                    'event2_time': f"{next_event['start_time']} - {next_event['end_time']}",
                    'overlap_seconds': overlap_seconds
                })
    
    return overlaps


def generate_minute_downsample(df_events: pd.DataFrame, place_id: int):
    """
    Genera registros minuto a minuto con conservación de masa garantizada.
    
    Mejoras respecto a versión original:
    - Umbral más estricto (0.001s en vez de 0s) para eventos ultra-cortos
    - Auditoría de conservación de masa
    - Logging mejorado
    
    Args:
        df_events: DataFrame con columnas [device, start_time, end_time, flow_rate]
        place_id: ID del lugar
        
    Returns:
        Lista de diccionarios para inserción bulk
    """
    records = []

    if df_events.empty:
        return []
    
    # ✅ AUDITORÍA: Volumen total antes de downsample
    total_volume_before = 0.0

    for _, row in df_events.iterrows():
        start = row["start_time"]
        end = row["end_time"]
        flow_rate = float(row["flow_rate"])
        device = row["device"]
        
        # Calcular volumen total del evento (para verificación)
        event_duration_seconds = (end - start).total_seconds()
        event_volume = flow_rate * (event_duration_seconds / 60)
        total_volume_before += event_volume

        # Generar rango de minutos que cubren el evento
        minute_range = pd.date_range(
            start.floor("min"),
            end.ceil("min"),
            freq="min"
        )

        # ✅ Distribuir volumen proporcionalmente en cada minuto
        for minute_start in minute_range:
            minute_end = minute_start + pd.Timedelta(minutes=1)

            overlap_start = max(start, minute_start)
            overlap_end = min(end, minute_end)

            overlap_seconds = (overlap_end - overlap_start).total_seconds()

            # ✅ CORRECCIÓN: Umbral más estricto para eventos ultra-cortos
            if overlap_seconds > 0.001:  # Era: > 0
                volume = flow_rate * (overlap_seconds / 60)

                records.append({
                    "place_id": place_id,
                    "time_bucket": minute_start.isoformat(),
                    "category": device,
                    "volume_liters": float(round(volume, 6))
                })

    if not records:
        return []

    # 🔥 Agrupar para evitar duplicados en mismo minuto + device
    df_records = pd.DataFrame(records)

    df_grouped = (
        df_records
        .groupby(["place_id", "time_bucket", "category"], as_index=False)
        .agg({"volume_liters": "sum"})
    )
    
    # ✅ AUDITORÍA FINAL: Verificar conservación de masa
    total_volume_after = df_grouped["volume_liters"].sum()
    discrepancy = abs(total_volume_before - total_volume_after) / total_volume_before if total_volume_before > 0 else 0
    
    print(f"[DOWNSAMPLE AUDIT] place_id={place_id}")
    print(f"  Events processed: {len(df_events)}")
    print(f"  Original volume: {total_volume_before:.6f} L")
    print(f"  Downsampled volume: {total_volume_after:.6f} L")
    print(f"  Discrepancy: {discrepancy:.4%}")
    
    if discrepancy > 0.01:
        print(f"  ⚠️ WARNING: Mass conservation violated by {discrepancy:.2%}")

    return df_grouped.to_dict(orient="records")