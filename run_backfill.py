# run_backfill.py
from worker.tasks import process_disaggregation
from datetime import datetime, timedelta

# Configura tu rango histórico
start_date = datetime(2026, 1, 27)
end_date = datetime(2026, 2, 14) # Un mes de datos
place_id = 1

current = start_date
while current < end_date:
    next_day = current + timedelta(days=1)
    
    print(f"Encolando día: {current.date()}...")
    
    # Lanzamos la tarea a Celery para este día específico
    process_disaggregation.delay(
        place_id=place_id,
        start_time=current.isoformat(),
        end_time=next_day.isoformat()
    )
    
    current = next_day

print("¡Todo encolado! Celery procesará los días uno por uno.")