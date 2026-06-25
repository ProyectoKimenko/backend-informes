from datetime import datetime, timedelta
from typing import Any

import pandas as pd

from services.supabase_service import (
    fetch_measurements,
    get_official_profiles,
    save_disaggregation_result,
    get_all_places,
    bulk_insert_dissagregated,
    save_official_profiles,
    get_supabase,
)
from pipeline.disaggregator_simple import run_disaggregation
from pipeline.segmentation import integrate_volume


def _parse_iso(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _month_keys_between(start_dt: datetime, end_dt: datetime) -> list[tuple[int, int]]:
    months: list[tuple[int, int]] = []
    current = datetime(start_dt.year, start_dt.month, 1, tzinfo=start_dt.tzinfo)
    end_month = datetime(end_dt.year, end_dt.month, 1, tzinfo=end_dt.tzinfo)

    while current <= end_month:
        months.append((current.year, current.month))
        if current.month == 12:
            current = datetime(current.year + 1, 1, 1, tzinfo=current.tzinfo)
        else:
            current = datetime(current.year, current.month + 1, 1, tzinfo=current.tzinfo)

    return months


def _count_available_days(place_id: int, start_time: str, end_time: str) -> int:
    sb = get_supabase()
    start_dt = _parse_iso(start_time)
    end_dt = _parse_iso(end_time)

    total = 0

    for year, month in _month_keys_between(start_dt, end_dt):
        res = sb.rpc(
            "get_available_days",
            {
                "p_place_id": place_id,
                "p_year": year,
                "p_month": month,
            },
        ).execute()

        raw_days: list[int] = []
        for row in (res.data or []):
            value = row.get("day")
            if isinstance(value, int):
                raw_days.append(value)
            elif isinstance(value, str):
                if value.isdigit():
                    raw_days.append(int(value))
                elif "-" in value:
                    try:
                        raw_days.append(int(value.split("-")[-1]))
                    except ValueError:
                        pass
            elif hasattr(value, "day"):
                raw_days.append(int(value.day))

        days = sorted(set(raw_days))

        for day in days:
            candidate = datetime(year, month, day, tzinfo=start_dt.tzinfo)
            if start_dt.date() <= candidate.date() <= end_dt.date():
                total += 1

    return total


def process_disaggregation(place_id: int, start_time: str | None = None, end_time: str | None = None, progress_cb=None):
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
                "status": "no_model_found",
            }

        profiles = {i: p for i, p in enumerate(official_profiles)}

        df = fetch_measurements(
            place_id=place_id,
            start_time=start_time,
            end_time=end_time,
        )

        if df.empty:
            return {"place_id": place_id, "status": "no_data"}

        df_events, df_result = run_disaggregation(df, profiles)

        # Litros con integral por Δt real (antes sum()/60 asumía cadencia fija).
        flow_col = "flow_rate" if "flow_rate" in df.columns else "flow"
        _ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce") if "timestamp" in df.columns else df.index
        _raw = pd.Series(df[flow_col].astype(float).values, index=_ts).dropna().sort_index()
        raw_liters = integrate_volume(_raw)

        device_cols = [c for c in df_result.columns if c != "No Detectado"]
        assigned_liters = integrate_volume(df_result[device_cols].sum(axis=1)) if device_cols else 0.0
        nd_liters = integrate_volume(df_result["No Detectado"]) if "No Detectado" in df_result.columns else 0.0
        total_liters = assigned_liters + nd_liters
        # smoothing_loss: raw vs total procesado (≈0, solo el suavizado). discrepancy:
        # % NO atribuido a fixtures (= cuota de No Detectado), la métrica real de cobertura.
        smoothing_loss_pct = abs(raw_liters - total_liters) / raw_liters * 100 if raw_liters > 0 else 0.0
        discrepancy_pct = round(nd_liters / total_liters * 100, 3) if total_liters > 0 else 0.0

        if progress_cb:
            progress_cb({
                "stage": "inference",
                "progress": 65,
                "place_id": place_id,
                "raw_liters": raw_liters,
                "assigned_liters": float(assigned_liters),
                "smoothing_loss_pct": round(smoothing_loss_pct, 3),
                "events_detected": len(df_events),
            })

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
                "raw_liters": raw_liters,
                "assigned_liters": float(assigned_liters),
                "no_detectado_liters": float(nd_liters),
                "smoothing_loss_pct": round(smoothing_loss_pct, 3),
                "discrepancy_pct": discrepancy_pct,
            },
        }

    except (KeyError, ValueError) as e:
        print(f"[VALIDATION ERROR] place_id={place_id}: {e}")
        return {
            "place_id": place_id,
            "status": "validation_error",
            "error": str(e),
        }

    except Exception as e:
        print(f"[ERROR] place_id={place_id}: {e}")
        raise


def infer_and_refresh(place_id: int, start_time: str | None = None, end_time: str | None = None, progress_cb=None):
    """Inferencia + backfill: lo que un disparo manual o el cron necesitan para
    refrescar disaggregated_readings (lo que consume el stackplot del frontend).
    Degrada gracioso si no hay datos/modelo (no_data / no_model_found)."""
    res = process_disaggregation(place_id, start_time, end_time, progress_cb=progress_cb)
    if isinstance(res, dict) and res.get("status") == "inference_done":
        if progress_cb:
            progress_cb({"stage": "backfill", "progress": 85, "place_id": place_id})
        try:
            res["backfill"] = backfill_disaggregated_readings(place_id, start_time, end_time)
        except Exception as e:
            res["backfill_error"] = str(e)
    return res


def process_all_places():
    """Job horario (lo llama el scheduler interno de app.py): por cada place,
    desagrega las últimas 24h y refresca el backfill. Síncrono y secuencial — a la
    escala actual (pocos places) es robusto y suficiente, sin cola distribuida."""
    places = get_all_places()

    end_time = datetime.utcnow()
    start_time = end_time - timedelta(days=1)
    start_iso, end_iso = start_time.isoformat(), end_time.isoformat()

    processed = 0
    for place in places:
        pid = place["id"]
        try:
            res = infer_and_refresh(pid, start_iso, end_iso)
            if isinstance(res, dict) and res.get("status") == "inference_done":
                processed += 1
        except Exception as e:
            print(f"[CRON] place_id={pid} falló: {e}")

    return {
        "places_processed": processed,
        "places_total": len(places),
        "time_range": f"{start_iso} to {end_iso}",
    }


def train_model(place_id: int, start_time: str, end_time: str):
    try:
        df = fetch_measurements(
            place_id=place_id,
            start_time=start_time,
            end_time=end_time,
        )

        if df.empty:
            return {"status": "no_data"}

        from pipeline.disaggregator_simple import train_disaggregator

        profiles = train_disaggregator(df)

        if not profiles:
            return {"status": "no_patterns_found"}

        save_official_profiles(place_id, profiles)

        return {
            "status": "trained",
            "profiles_saved": len(profiles),
            "profile_names": [p["name"] for p in profiles.values()],
        }

    except Exception as e:
        print(f"[TRAINING ERROR] place_id={place_id}: {e}")
        return {
            "status": "training_failed",
            "error": str(e),
        }


def backfill_disaggregated_readings(place_id: int, start_time: str | None = None, end_time: str | None = None):
    sb = get_supabase()

    print(f"[Backfill] Fetching events for place_id={place_id}")

    query = (
        sb.table("disaggregation_events")
        .select("id, device_name, start_time, end_time, flow_rate, volume_liters")
        .eq("place_id", place_id)
    )

    if start_time:
        query = query.gte("start_time", start_time)
    if end_time:
        query = query.lte("end_time", end_time)

    response = query.order("start_time").execute()
    events = response.data

    if not events:
        return {"status": "no_events_found"}

    df_events = pd.DataFrame(events)
    df_events["start_time"] = pd.to_datetime(df_events["start_time"], utc=True)
    df_events["end_time"] = pd.to_datetime(df_events["end_time"], utc=True)
    df_events.rename(columns={"device_name": "device"}, inplace=True)

    overlaps = detect_overlapping_events(df_events)
    if overlaps:
        print(f"⚠️ WARNING: {len(overlaps)} overlapping events detected for place_id={place_id}")
        for overlap in overlaps[:5]:
            print(f"  Device: {overlap['device']}, Overlap: {overlap['overlap_seconds']:.2f}s")

    print(f"[Backfill] Generating minute buckets for {len(df_events)} events")

    minute_records = generate_minute_downsample(df_events, place_id)

    print(f"[Backfill] Inserting {len(minute_records)} minute rows")

    bulk_insert_dissagregated(minute_records)

    return {
        "status": "backfill_completed",
        "events_processed": len(df_events),
        "minutes_inserted": len(minute_records),
        "overlaps_detected": len(overlaps) if overlaps else 0,
    }


def train_and_refresh_disaggregation(place_id: int, start_time: str, end_time: str, progress_cb=None):
    try:
        start_dt = _parse_iso(start_time)
        end_dt = _parse_iso(end_time)

        if end_dt <= start_dt:
            return {
                "place_id": place_id,
                "status": "invalid_range",
                "error": "end_time must be greater than start_time",
            }

        available_days = _count_available_days(place_id, start_time, end_time)
        if available_days < 21:
            return {
                "place_id": place_id,
                "status": "not_enough_days",
                "available_days": available_days,
                "required_days": 21,
            }

        if progress_cb:
            progress_cb({"stage": "training", "progress": 10, "place_id": place_id, "available_days": available_days})

        training_result = train_model(place_id, start_time, end_time)
        if training_result.get("status") != "trained":
            return {
                "place_id": place_id,
                "status": "training_failed",
                "training_result": training_result,
            }

        if progress_cb:
            progress_cb({"stage": "disaggregation", "progress": 45, "place_id": place_id,
                         "available_days": available_days, "training_result": training_result})

        inference_result = process_disaggregation(
            place_id=place_id,
            start_time=start_time,
            end_time=end_time,
        )

        if inference_result.get("status") not in {"inference_done"}:
            return {
                "place_id": place_id,
                "status": "inference_failed",
                "training_result": training_result,
                "inference_result": inference_result,
            }

        if progress_cb:
            progress_cb({"stage": "backfill", "progress": 80, "place_id": place_id,
                         "available_days": available_days, "training_result": training_result,
                         "inference_result": inference_result})

        backfill_result = backfill_disaggregated_readings(
            place_id=place_id,
            start_time=start_time,
            end_time=end_time,
        )

        return {
            "place_id": place_id,
            "status": "completed",
            "available_days": available_days,
            "range": {
                "start_time": start_time,
                "end_time": end_time,
            },
            "training_result": training_result,
            "inference_result": inference_result,
            "backfill_result": backfill_result,
        }

    except Exception as e:
        print(f"[TRAIN+REFRESH ERROR] place_id={place_id}: {e}")
        raise


def detect_overlapping_events(df_events: pd.DataFrame) -> list[dict[str, Any]]:
    overlaps = []

    for device in df_events["device"].unique():
        device_events = df_events[df_events["device"] == device].sort_values("start_time")

        for i in range(len(device_events) - 1):
            current = device_events.iloc[i]
            next_event = device_events.iloc[i + 1]

            if current["end_time"] > next_event["start_time"]:
                overlap_seconds = (current["end_time"] - next_event["start_time"]).total_seconds()
                overlaps.append({
                    "device": device,
                    "event1_id": current.get("id"),
                    "event1_time": f"{current['start_time']} - {current['end_time']}",
                    "event2_id": next_event.get("id"),
                    "event2_time": f"{next_event['start_time']} - {next_event['end_time']}",
                    "overlap_seconds": overlap_seconds,
                })

    return overlaps


def generate_minute_downsample(df_events: pd.DataFrame, place_id: int):
    records = []

    if df_events.empty:
        return []

    total_volume_before = 0.0

    for _, row in df_events.iterrows():
        start = row["start_time"]
        end = row["end_time"]
        flow_rate = float(row["flow_rate"])
        device = row["device"]

        event_duration_seconds = (end - start).total_seconds()
        # Usar el volumen YA INTEGRADO del evento (Δt real) y repartirlo
        # proporcional al tiempo; fallback a caudal·duración si no viniera.
        _stored_vol = row.get("volume_liters")
        event_volume = (
            float(_stored_vol) if _stored_vol is not None and pd.notna(_stored_vol)
            else flow_rate * (event_duration_seconds / 60)
        )
        total_volume_before += event_volume

        minute_range = pd.date_range(
            start.floor("min"),
            end.ceil("min"),
            freq="min",
        )

        for minute_start in minute_range:
            minute_end = minute_start + pd.Timedelta(minutes=1)

            overlap_start = max(start, minute_start)
            overlap_end = min(end, minute_end)

            overlap_seconds = (overlap_end - overlap_start).total_seconds()

            if overlap_seconds > 0.001:
                frac = overlap_seconds / event_duration_seconds if event_duration_seconds > 0 else 0.0
                volume = event_volume * frac

                records.append({
                    "place_id": place_id,
                    "time_bucket": minute_start.isoformat(),
                    "category": device,
                    "volume_liters": float(round(volume, 6)),
                })

    if not records:
        return []

    df_records = pd.DataFrame(records)

    df_grouped = (
        df_records
        .groupby(["place_id", "time_bucket", "category"], as_index=False)
        .agg({"volume_liters": "sum"})
    )

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