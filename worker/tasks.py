from worker.celery_app import celery_app
from services.supabase_service import fetch_measurements, save_disaggregation_result, get_all_places
from pipeline.disaggregator import run_disaggregation
from datetime import datetime, timedelta

@celery_app.task(
    bind=True,
    autoretry_for=(Exception,),
    retry_backoff=60,
    retry_kwargs={"max_retries": 3},
    retry_jitter=True,
    name="worker.tasks.process_disaggregation",
)

def process_disaggregation(self, place_id: str, start_time: str | None = None, end_time: str | None = None):
    df = fetch_measurements(place_id=place_id, start_time=start_time, end_time=end_time)

    if df.empty:
        return {
            "place_id": place_id,
            "status": "no_data",
        }
    df_events, df_result, profiles = run_disaggregation(df)
    save_disaggregation_result(place_id, df_events, df_result, profiles)
    return {
        "place_id": place_id,
        "events": len(df_events),
        "devices": profiles,
    }


@celery_app.task(name="worker.tasks.process_all_places")
def process_all_places():
    places = get_all_places()
    end_time = datetime.now()
    start_time = end_time - timedelta(hours=1)

    results = []
    for place in places:
        result = process_disaggregation.delay(
            place_id=place["id"],
            start_time=start_time.isoformat(),
            end_time=end_time.isoformat(),
        )
        results.append(result)
    return {
        'total_places': len(places),
        'tasks_submitted': len(results),
    }
