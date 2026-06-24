from celery import Celery
from celery.schedules import crontab
import os

REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")

celery_app = Celery(
    "disaggregation",
    broker=REDIS_URL,
    backend=REDIS_URL,
    include=["worker.tasks"]
)

celery_app.conf.update(
    task_track_started=True,
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    # Expirar los resultados de tareas (claves celery-task-meta-*) para que no se
    # acumulen indefinidamente en Redis (el cron horario encola N tareas/hora).
    result_expires=86400,        # 24h
)

celery_app.conf.beat_schedule = {
    "disaggregate-all-every-hour": {
        "task": "worker.tasks.process_all_places",
        "schedule": crontab(minute=0),
    },
}