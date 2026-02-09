from typing import Optional, Dict
from supabase import create_client, Client
import pandas as pd
import os

_supabase: Optional[Client] = None

def get_supabase() -> Client:
    global _supabase
    if _supabase is None:
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_KEY")

        if not url or not key:
            raise RuntimeError("Missing SUPABASE_URL or SUPABASE_KEY")

        _supabase = create_client(url, key)

    return _supabase

def fetch_measurements(
        place_id: Optional[str] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
) -> pd.DataFrame:

    sb = get_supabase()
    query = sb.table("measurements_realtime").select("*")

    if place_id:
        query = query.eq("place_id", place_id)

    if start_time:
        query = query.gte("timestamp", start_time)

    if end_time:
        query = query.lte("timestamp", end_time)

    response = query.order("timestamp").execute()

    data = response.data or []
    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data)

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp").sort_index()

    if "flow_rate" not in df.columns:
        raise ValueError("Missing 'flow_rate' column in measurements")
    return df[["flow_rate"]].rename(columns={"flow_rate": "flow"})

def save_disaggregation_result(
    place_id: int | str,
    df_events: pd.DataFrame,
    df_result: pd.DataFrame,
    profiles: Dict[str, Dict],
) -> None:
    supabase: Client = get_supabase()

    # 1. Guardar Perfiles (Detectados por el algoritmo)
    if profiles:
        # Borrar perfiles viejos para este lugar (opcional, para evitar duplicados)
        supabase.table("disaggregation_profiles").delete().eq("place_id", place_id).execute()
        
        profile_records = []
        for dev_id, info in profiles.items():
            profile_records.append({
                "place_id": place_id,
                "device_id": int(dev_id),
                "name": info["name"],
                "mean_flow": float(info["mean"]),
                "tolerance": float(info["tol"])
            })
        supabase.table("disaggregation_profiles").insert(profile_records).execute()

    # 2. Guardar Eventos (Como ya lo tenías, mapeando flow_rate)
    if not df_events.empty:
        event_records = []
        for _, row in df_events.iterrows():
            event_records.append({
                "place_id": place_id,
                "device_name": row["Device"],
                "start_time": row["Start"].isoformat(),
                "end_time": row["End"].isoformat(),
                "duration_s": float(row["Duration_s"]),
                "flow_rate": float(row["Flow_L_min"]),
            })
        supabase.table("disaggregation_events").insert(event_records).execute()


    print(f"[Disaggregation] place_id={place_id} Saved: Events={len(df_events)}, Profiles={len(profiles)}")

def get_all_places():
    sb = get_supabase()
    result = sb.table("places").select("*").execute()
    return result.data or []