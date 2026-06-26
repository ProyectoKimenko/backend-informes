from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from fastapi.requests import Request
from fastapi import Body
from fastapi.staticfiles import StaticFiles
from starlette.background import BackgroundTask
from fastapi.templating import Jinja2Templates
from jinja2 import Environment, FileSystemLoader
from weasyprint import HTML
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timedelta, timezone
import colorsys
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import os
import pandas as pd
import tempfile
import requests
import time
from src.analysis import analyze_data, analyze_data_from_df
from src.report import Report
from src.report_sections import WeekdaySection, WeekendSection, ComparisonSection
from src.logger_config import setup_logger, log_request, log_error, log_data_operation, log_startup
from pydantic import BaseModel, field_validator
from worker.tasks import (
    train_and_refresh_disaggregation,
    infer_and_refresh,
    process_all_places,
)
from services.redis_client import redis_client as redis
from services.supabase_service import get_stackplot_data
import json
import uuid
import asyncio
import threading
from services.supabase_service import get_supabase

# Configuración de logging centralizada
logger = setup_logger(__name__)

# Initialize FastAPI app and templates
app = FastAPI()

# CORS configurable por entorno. En producción (Cloudflare/Dokploy) define
# ALLOWED_ORIGINS con los orígenes del frontend separados por coma, p.ej.:
#   ALLOWED_ORIGINS=https://app.kimenko.cl,https://kimenko.cl
# Nota: allow_origins=["*"] + allow_credentials=True es INVÁLIDO (los navegadores
# lo rechazan), así que las credenciales solo se habilitan con orígenes explícitos.
_origins_env = os.getenv("ALLOWED_ORIGINS", "*").strip()
ALLOWED_ORIGINS = [o.strip() for o in _origins_env.split(",") if o.strip()] or ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=ALLOWED_ORIGINS != ["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------------------------------------
# Jobs en memoria + scheduler interno (reemplazan Celery + Redis para la
# desagregación). A la escala actual (1 operador, jobs ocasionales) esto es
# robusto y suficiente: sin worker/beat/redis separados. El estado de job es
# efímero (se pierde al reiniciar el server), aceptable porque la inferencia es
# idempotente y reentrenar es manual.
# -----------------------------------------------------------------------------
JOBS: dict = {}
_JOBS_LOCK = threading.Lock()
_JOBS_MAX = 200


def _job_set(task_id: str, **fields):
    with _JOBS_LOCK:
        job = JOBS.get(task_id, {"task_id": task_id})
        job.update(fields)
        JOBS[task_id] = job
        if len(JOBS) > _JOBS_MAX:                      # cota de memoria
            for k in list(JOBS.keys())[: len(JOBS) - _JOBS_MAX]:
                JOBS.pop(k, None)


def _run_job(task_id: str, fn, *args, **kwargs):
    """Corre fn(...) en background (threadpool de BackgroundTasks) trackeando estado."""
    _job_set(task_id, status="processing", progress=5)

    def cb(meta: dict):
        _job_set(task_id, status="processing", **meta)

    try:
        result = fn(*args, progress_cb=cb, **kwargs)
        st = result.get("status") if isinstance(result, dict) else None
        terminal_ok = {"completed", "inference_done", "trained", "backfill_completed", None}
        _job_set(
            task_id,
            status="completed" if st in terminal_ok else st,
            progress=100,
            result=result,
        )
    except Exception as e:
        logger.exception(f"[JOB {task_id}] error")
        _job_set(task_id, status="failed", error=str(e))


def _new_task_id() -> str:
    return uuid.uuid4().hex


async def _hourly_scheduler():
    """Dispara process_all_places al inicio de cada hora (reemplaza Celery beat)."""
    while True:
        now = datetime.utcnow()
        nxt = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
        await asyncio.sleep(max(1.0, (nxt - now).total_seconds()))
        try:
            res = await asyncio.to_thread(process_all_places)
            logger.info(f"[scheduler] process_all_places -> {res}")
        except Exception:
            logger.exception("[scheduler] process_all_places falló")


@app.on_event("startup")
async def _start_scheduler():
    asyncio.create_task(_hourly_scheduler())


class DisaggregationRequest(BaseModel):
    place_id: int
    start_time: str | None = None
    end_time: str | None = None
class TrainingWorkflowRequest(BaseModel):
    place_id: int
    start_time: str
    end_time: str

    @field_validator("start_time", "end_time")
    @classmethod
    def validate_iso_datetime(cls, value: str) -> str:
        try:
            datetime.fromisoformat(value.replace("Z", "+00:00"))
        except Exception as exc:
            raise ValueError("start_time/end_time deben estar en formato ISO-8601") from exc
        return value
class UpdateDisaggregationProfileLabelRequest(BaseModel):
    label: Optional[str] = None

DATASETS = {}
templates = Jinja2Templates(directory="templates")

# Directorio acotado para imágenes/PDF generados (NO se sirve públicamente).
# Antes /static montaba tempfile.gettempdir() entero -> exponía TODO /tmp a
# internet (cualquiera podía leer archivos temporales del host). Ahora /static
# sirve solo los assets propios de la app (logo, fondos) y los temporales viven
# en un subdir dedicado que se barre por antigüedad.
GEN_DIR = os.path.join(tempfile.gettempdir(), "kimenko_generated")
os.makedirs(GEN_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")


def _sweep_generated(max_age_s: int = 3600) -> None:
    """Borra de GEN_DIR los archivos más viejos que max_age_s (evita fuga de disco)."""
    try:
        now = time.time()
        for fn in os.listdir(GEN_DIR):
            fp = os.path.join(GEN_DIR, fn)
            try:
                if os.path.isfile(fp) and (now - os.path.getmtime(fp)) > max_age_s:
                    os.remove(fp)
            except OSError:
                pass
    except OSError:
        pass

# Middleware para logging de requests
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    
    # Procesar request
    response = await call_next(request)
    
    # Calcular duración
    duration_ms = (time.time() - start_time) * 1000
    
    # Log solo requests importantes (no incluir healthcheck, static files, etc.)
    if not request.url.path.startswith(("/static", "/docs", "/redoc", "/openapi.json")):
        log_request(logger, request.method, request.url.path, duration_ms)
        
        # Log errores HTTP
        if response.status_code >= 400:
            logger.warning(f"{request.method} {request.url.path} - HTTP {response.status_code}")
    
    return response

# Log de inicio de aplicación
log_startup(logger)

# Constants
WEEK_COLORS = {
    'Semana 1': '#1fa9c9',
    'Semana 2': '#34495e', 
    'Semana 3': '#2ecc71',
    'Semana 4': '#e67e22',
    'Semana 5': '#9b59b6'
}

# Helper Functions
def adjust_color_brightness(hex_color, factor):
    """Adjust the brightness of a hex color"""
    hex_color = hex_color.lstrip('#')
    rgb = tuple(int(hex_color[i:i+2], 16)/255 for i in (0, 2, 4))
    hsv = colorsys.rgb_to_hsv(*rgb)
    hsv = (hsv[0], hsv[1], min(1, hsv[2] * factor))
    rgb = colorsys.hsv_to_rgb(*hsv)
    return '#{:02x}{:02x}{:02x}'.format(int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255))

def split_at_gaps(index, values, mask):
    time_diffs = np.diff(index[mask])
    gaps = np.where(time_diffs > pd.Timedelta(minutes=1))[0]
    segments = []
    start_idx = 0

    for gap_idx in gaps:
        end_idx = np.where(mask)[0][gap_idx + 1]
        if end_idx > start_idx:
            segment_mask = mask.copy()
            segment_mask[end_idx:] = False
            segment_mask[:start_idx] = False
            if np.sum(segment_mask) > 1:
                segments.append((index[segment_mask], values[segment_mask]))
        start_idx = end_idx

    segment_mask = mask.copy()
    segment_mask[:start_idx] = False
    if np.sum(segment_mask) > 1:
        segments.append((index[segment_mask], values[segment_mask]))

    return segments

def get_local_timezone_offset():
    """Get the local timezone offset from UTC as a string (e.g., 'UTC-4', 'UTC+2')"""
    local_now = datetime.now()
    utc_now = datetime.utcnow()
    
    # Calculate offset in hours
    offset_seconds = (local_now - utc_now).total_seconds()
    offset_hours = int(offset_seconds / 3600)
    
    if offset_hours >= 0:
        return f"UTC+{offset_hours}"
    else:
        return f"UTC{offset_hours}"  # offset_hours already has the minus sign

def create_plot(data, title_prefix, date_str):
    required_columns = ['flow_rate', 'RollingMin']
    for col in required_columns:
        if col not in data.columns:
            logger.error(f"Missing required column '{col}' in data for plotting.")
            raise ValueError(f"Missing required column '{col}' in data")

    has_data = len(data) > 0 and data['flow_rate'].sum() > 0
    
    fig, ax = plt.subplots(figsize=(14, 8))
    plt.style.use('default')
    
    # Get local timezone info
    tz_offset = get_local_timezone_offset()
    
    if not has_data:
        ax.axhline(y=0, color='lightgray', linestyle='-', linewidth=2)
        ax.text(0.5, 0.5, 'Sin consumo registrado en este período\n(100% de eficiencia)', 
                horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes, fontsize=16, color='gray',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="gray", alpha=0.8))
        ax.set_ylim(-0.5, 5)
        ax.set_xlim(0, 1)
        ax.set_xticks([])
    else:
        flow_mask = ~np.isnan(data['flow_rate'])
        min_mask = ~np.isnan(data['RollingMin'])

        flow_segments = split_at_gaps(data.index, data['flow_rate'], flow_mask)
        for i, (x, y) in enumerate(flow_segments):
            if len(x) > 0:
                ax.plot(x, y, color='#1f77b4', linewidth=2, 
                       label='Flujo total' if i == 0 else "")

        min_segments = split_at_gaps(data.index, data['RollingMin'], min_mask)
        for i, (x, y) in enumerate(min_segments):
            if len(x) > 1:
                ax.plot(x, y, color='#d62728', linewidth=2, 
                       label='Límite de pérdida' if i == 0 else "")
                ax.fill_between(x, 0, y, color='#d62728', alpha=0.3,
                               label='Pérdida' if i == 0 else "")
        
        max_val = max(data['flow_rate'].max(), 1)
        ax.set_ylim(0, max_val * 1.1)
        
        # Improved time axis formatting
        if len(data) > 0:
            # Format x-axis with dates and times
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m\n%H:%M'))
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))
            ax.xaxis.set_minor_locator(mdates.HourLocator(interval=2))
            
            # Rotate labels for better readability
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
            
            # Add grid for better readability
            ax.grid(True, alpha=0.3, which='major')
            ax.grid(True, alpha=0.1, which='minor')
        
        ax.legend(frameon=True, fancybox=True, shadow=True, fontsize=10)

    # Enhanced title and labels with local timezone information
    ax.set_title(f'{title_prefix} - {date_str}\n(Hora {tz_offset})', pad=20, fontsize=14, weight='bold')
    ax.set_xlabel(f'Fecha y Hora ({tz_offset})', fontsize=12, weight='bold')
    ax.set_ylabel('Flujo (litros/min)', fontsize=12, weight='bold')
    
    # Improve layout
    plt.tight_layout()

    with tempfile.NamedTemporaryFile(delete=False, suffix='.png', dir=GEN_DIR) as temp_file:
        plt.savefig(temp_file.name, dpi=100, bbox_inches='tight', facecolor='white')
        plt.close()
    return temp_file.name

def create_weekly_trend_plot(weeks_data):
    required_keys = [
        'weekday_consumption', 'weekend_consumption', 'weekday_efficiency', 
        'weekend_efficiency', 'weekday_wasted', 'weekend_wasted', 'color'
    ]
    for i, w in enumerate(weeks_data, start=1):
        for k in required_keys:
            if k not in w:
                logger.error(f"Missing key '{k}' in week data for Semana {i}.")
                raise ValueError(f"Missing key '{k}' in week data")

    weeks = [i.get('title') for i in weeks_data]
    weekday_consumptions = [week['weekday_consumption'] for week in weeks_data]
    weekend_consumptions = [week['weekend_consumption'] for week in weeks_data]
    weekday_efficiencies = [week['weekday_efficiency'] for week in weeks_data]
    weekend_efficiencies = [week['weekend_efficiency'] for week in weeks_data]
    weekday_losses = [week['weekday_wasted'] for week in weeks_data]
    weekend_losses = [week['weekend_wasted'] for week in weeks_data]
    colors = [week['color'] for week in weeks_data]

    all_zero_consumption = all(c == 0 for c in weekday_consumptions + weekend_consumptions)
    
    # Get local timezone info
    tz_offset = get_local_timezone_offset()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), height_ratios=[1, 1])
    fig.suptitle(f'Tendencias Semanales de Consumo de Agua\n(Todos los datos en hora {tz_offset})',
                 fontsize=20, y=0.95, weight='bold')

    width = 0.35
    x = np.arange(len(weeks))

    if all_zero_consumption:
        ax1.text(0.5, 0.5, 'Sin consumo registrado en las semanas analizadas', 
                horizontalalignment='center', verticalalignment='center',
                transform=ax1.transAxes, fontsize=16, color='gray')
        ax1.set_ylim(0, 1)
    else:
        weekday_net_consumption = [max(0, c - w) for c, w in zip(weekday_consumptions, weekday_losses)]
        weekend_net_consumption = [max(0, c - w) for c, w in zip(weekend_consumptions, weekend_losses)]
        
        ax1.bar(x - width / 2, weekday_net_consumption, width, 
                label='Consumo Laboral', color=colors)
        ax1.bar(x + width / 2, weekend_net_consumption, width, 
                label='Consumo Fin de Semana', color=[adjust_color_brightness(c, 0.7) for c in colors])
        
        if any(loss > 0 for loss in weekday_losses):
            ax1.bar(x - width / 2, weekday_losses, width, 
                    bottom=weekday_net_consumption,
                    label='Pérdidas Laborales', color=[adjust_color_brightness(c, 0.5) for c in colors])
        
        if any(loss > 0 for loss in weekend_losses):
            ax1.bar(x + width / 2, weekend_losses, width, 
                    bottom=weekend_net_consumption,
                    label='Pérdidas Fin de Semana', color=[adjust_color_brightness(c, 0.3) for c in colors])
        
        max_y = max(max(weekday_consumptions), max(weekend_consumptions), 1) * 1.1
        ax1.set_ylim(0, max_y)

    ax1.set_ylabel('Litros', fontsize=14, weight='bold')
    ax1.set_title('Consumo Total y Pérdidas por Semana', fontsize=16, pad=20, weight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(weeks, fontsize=12)
    ax1.tick_params(axis='y', labelsize=12)
    ax1.legend(fontsize=12, bbox_to_anchor=(1.05, 1), loc='upper left')

    ax2.plot(weeks, weekday_efficiencies, marker='o', linewidth=3, markersize=12,
             color='#1fa9c9', label='Eficiencia Laboral')
    ax2.plot(weeks, weekend_efficiencies, marker='s', linewidth=3, markersize=12,
             color='#34495e', label='Eficiencia Fin de Semana')

    for i, (weekday_eff, weekend_eff) in enumerate(zip(weekday_efficiencies, weekend_efficiencies)):
        ax2.annotate(f'{weekday_eff}%', (weeks[i], weekday_eff),
                     textcoords="offset points", xytext=(0, 15),
                     ha='center', fontsize=11, weight='bold')
        ax2.annotate(f'{weekend_eff}%', (weeks[i], weekend_eff),
                     textcoords="offset points", xytext=(0, -20),
                     ha='center', fontsize=11, weight='bold')

    ax2.set_ylabel('Porcentaje de Eficiencia', fontsize=14, weight='bold')
    ax2.set_title('Eficiencia por Semana', fontsize=16, pad=20, weight='bold')
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.set_ylim(0, 105)
    ax2.tick_params(axis='both', labelsize=12)
    ax2.legend(fontsize=12, bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout(rect=[0, 0, 0.85, 0.95], h_pad=0.8)

    with tempfile.NamedTemporaryFile(delete=False, suffix='.png', dir=GEN_DIR) as temp_file:
        plt.savefig(temp_file.name, bbox_inches='tight', dpi=100, facecolor='white')
        plt.close()
    return temp_file.name

def create_total_weekly_consumption_chart(weeks_data):
    """Create a minimalist stacked bar chart showing useful consumption and losses per week"""
    weeks = [w['title'] for w in weeks_data]
    total_consumptions = [w['weekday_consumption'] + w['weekend_consumption'] for w in weeks_data]
    total_losses = [w['weekday_wasted'] + w['weekend_wasted'] for w in weeks_data]
    colors = [w['color'] for w in weeks_data]
    
    # Calculate useful consumption (total - losses)
    useful_consumptions = [max(0, total - loss) for total, loss in zip(total_consumptions, total_losses)]
    
    # Calculate loss percentages
    loss_percentages = []
    for consumption, loss in zip(total_consumptions, total_losses):
        if consumption > 0:
            loss_pct = (loss / consumption) * 100
            loss_percentages.append(loss_pct)
        else:
            loss_percentages.append(0)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Create stacked bars: useful consumption + losses
    bars_useful = ax.bar(weeks, useful_consumptions, color=colors, 
                         edgecolor='white', linewidth=2, alpha=0.85, label='Consumo Útil')
    bars_loss = ax.bar(weeks, total_losses, bottom=useful_consumptions,
                       color='#d62728', edgecolor='white', linewidth=2, 
                       alpha=0.7, label='Pérdidas')
    
    # Clean styling
    ax.set_title('Consumo Total por Semana', fontsize=16, weight='bold', pad=20, color='#2c3e50')
    ax.set_ylabel('Litros', fontsize=13, weight='bold', color='#2c3e50')
    ax.set_xlabel('Semana', fontsize=13, weight='bold', color='#2c3e50')
    
    # Minimal grid
    ax.grid(True, alpha=0.2, axis='y', linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Clean spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#cccccc')
    ax.spines['bottom'].set_color('#cccccc')
    
    # Tick styling
    ax.tick_params(axis='both', labelsize=11, colors='#2c3e50')
    
    # Legend in upper right to avoid overlap
    ax.legend(loc='upper right', frameon=False, fontsize=11)
    
    # Add extra space on top for labels - set ylim with padding
    if len(total_consumptions) > 0 and max(total_consumptions) > 0:
        ax.set_ylim(0, max(total_consumptions) * 1.12)
    
    # Add total consumption value on top
    for i, (total, loss, loss_pct) in enumerate(zip(total_consumptions, total_losses, loss_percentages)):
        if total > 0:
            # Total consumption label on top with offset
            ax.text(i, total, f'{total:,.0f} L', 
                   ha='center', va='bottom', fontsize=10, weight='bold', color='#2c3e50')
            # Loss percentage label on the red section
            if loss > 0 and loss > (total * 0.05):  # Only show if loss is > 5% to avoid cramping
                loss_y_pos = useful_consumptions[i] + (loss / 2)
                ax.text(i, loss_y_pos, f'{loss_pct:.1f}%', 
                       ha='center', va='center', fontsize=9, 
                       color='white', weight='bold')
    
    plt.tight_layout()
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.png', dir=GEN_DIR) as temp_file:
        plt.savefig(temp_file.name, dpi=100, bbox_inches='tight', facecolor='white')
        plt.close()
    return temp_file.name

def get_dates_from_week_number(year: int, week: int, num_weeks: int = 4) -> List[Tuple[datetime, datetime]]:
    """Get start and end dates for specified week number(s) in UTC, from Monday 00:00 to Sunday 23:59."""
    week_ranges = []
    if num_weeks < 1:
        raise ValueError("num_weeks must be at least 1")

    for i in range(num_weeks):
        current_week = week + i
        if current_week < 1 or current_week > 53:
            raise ValueError(f"Requested out-of-range week number: {current_week}")

        try:
            # Monday 00:00:00 UTC
            week_start = datetime.fromisocalendar(year, current_week, 1)
            week_start = week_start.replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=timezone.utc)
            
            # Sunday 23:59:59.999999 UTC
            week_end = datetime.fromisocalendar(year, current_week, 7)
            week_end = week_end.replace(hour=23, minute=59, second=59, microsecond=999999, tzinfo=timezone.utc)
            
        except ValueError:
            raise ValueError(f"No valid week {current_week} found in year {year}.")

        # Solo log en DEBUG para desarrollo
        logger.debug(f"Week {current_week}: {week_start.strftime('%Y-%m-%d')} to {week_end.strftime('%Y-%m-%d')}")
        
        week_ranges.append((week_start, week_end))

    return week_ranges

def make_scraping_request(place_id: int, start_date: str, end_date: str):
    """Background task to make the scraping request"""
    try:
        requests.get(
            "http://172.17.0.1:8001/scrape-devices",
            params={
                "place_id": place_id,
                "start_date": start_date,
                "end_date": end_date,
                "force": True
            },
            headers={"Content-Type": "application/json"},
            timeout=180
        )
        logger.info(f"Data updated for place {place_id}")
    except Exception as e:
        log_error(logger, f"data update for place {place_id}", e)

# Initialize Supabase client
supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_KEY")
supabase = get_supabase()  # singleton (prefiere service_role)

# API Endpoints
@app.get("/ping", status_code=200)
def ping():
    return {"status": "success", "message": "Pong"}

@app.get("/places", response_class=JSONResponse)
async def get_places():
    """Devuelve la lista de lugares desde la tabla 'places'."""
    try:
        response = supabase.table("places").select("*").execute()
        places = response.data if response.data else []
        return {"places": places}
    except Exception as e:
        log_error(logger, "fetching places", e)
        raise HTTPException(status_code=500, detail="Error fetching places")

@app.get("/analysis", response_class=JSONResponse)
async def analysis_json(
    window_size: int = 60,
    start_week: int = 1,
    end_week: int = 1,
    year: int = 2024,
    place_id: int = 1
):
    """Devuelve métricas y serie temporal en JSON para renderizar en el frontend."""
    try:
        if not (1 <= start_week <= 53 and 1 <= end_week <= 53):
            raise HTTPException(status_code=400, detail="Rango de semanas inválido")
        if end_week < start_week:
            raise HTTPException(status_code=400, detail="end_week debe ser >= start_week")
        num_weeks = end_week - start_week + 1
        if num_weeks > 5:
            raise HTTPException(status_code=400, detail="No se pueden seleccionar más de 5 semanas")

        week_ranges = get_dates_from_week_number(year, start_week, num_weeks)
        start_epoch = int(week_ranges[0][0].timestamp() * 1000)
        end_epoch   = int(week_ranges[-1][1].timestamp() * 1000)
        
        # Log solo el resumen del análisis
        start_date = week_ranges[0][0].strftime('%Y-%m-%d')
        end_date = week_ranges[-1][1].strftime('%Y-%m-%d')
        logger.info(f"Analysis requested: {start_date} to {end_date}")

        try:
            results = analyze_data(
                window_size=window_size,
                start_epoch=start_epoch,
                end_epoch=end_epoch,
                place_id=place_id
            )
        except Exception as e:
            logger.warning("No data available for analysis period")
            return {
                "total_water_wasted_weekdays":      None,
                "efficiency_percentage_weekdays":   None,
                "total_water_consumed_weekdays":    None,
                "total_water_wasted_weekends":      None,
                "efficiency_percentage_weekends":   None,
                "total_water_consumed_weekends":    None,
                "time_series":                      []
            }

        # Log resumen de datos procesados
        combined = pd.concat([results['weekday_data'], results['weekend_data']])
        if not combined.empty:
            logger.debug(f"Processed {len(combined)} data points from {combined.index.min().strftime('%Y-%m-%d')} to {combined.index.max().strftime('%Y-%m-%d')}")
        
        df = combined.reset_index()
        if not df.empty and 'timestamp' in df.columns:
            df['timestamp'] = df['timestamp'].astype(str)
        time_series = df[['timestamp', 'flow_rate', 'RollingMin']].to_dict(orient='records') if not df.empty else []

        return {
            "total_water_wasted_weekdays":      results['weekday_wasted'],
            "efficiency_percentage_weekdays":   results['weekday_efficiency'],
            "total_water_consumed_weekdays":    results['weekday_total'],
            "total_water_wasted_weekends":      results['weekend_wasted'],
            "efficiency_percentage_weekends":   results['weekend_efficiency'],
            "total_water_consumed_weekends":    results['weekend_total'],
            "time_series":                      time_series
        }
    except HTTPException:
        raise
    except Exception as e:
        log_error(logger, "data analysis", e)
        return {
            "total_water_wasted_weekdays":      None,
            "efficiency_percentage_weekdays":   None,
            "total_water_consumed_weekdays":    None,
            "total_water_wasted_weekends":      None,
            "efficiency_percentage_weekends":   None,
            "total_water_consumed_weekends":    None,
            "time_series":                      []
        }

@app.get("/generate_weekly_pdf", status_code=200)
async def generate_weekly_pdf(
    year: int,
    end_week: int,
    start_week: int,
    window_size: int = 60,
    place_id: int = None
):
    time_start = time.time()
    _sweep_generated()  # limpia PNGs/PDF temporales viejos (evita fuga de disco)
    try:
        response = supabase.table("places").select("*").execute()
        places = response.data if response.data else []
        place_name = next((place['name'] for place in places if place['id'] == place_id), "Unknown Location")

        if not (1 <= start_week <= 53 and 1 <= end_week <= 53):
            raise HTTPException(status_code=400, detail="Invalid week range")

        if end_week < start_week:
            raise HTTPException(status_code=400, detail=f"end_week must be >= start_week")

        num_weeks = end_week - start_week + 1
        if num_weeks > 5:
            raise HTTPException(status_code=400, detail=f"Cannot select more than 5 weeks")
        
        week_ranges = get_dates_from_week_number(year, start_week, num_weeks)
        if not week_ranges:
            raise HTTPException(status_code=400, detail=f"No valid weeks found")
        
        # Single query to get all data for all weeks
        total_start_epoch = int(week_ranges[0][0].timestamp() * 1000)
        total_end_epoch = int(week_ranges[-1][1].timestamp() * 1000)
        
        query = supabase.table("measurements").select("*")
        if place_id is not None:
            query = query.eq("place_id", place_id)
        query = query.gte("timestamp", total_start_epoch).lte("timestamp", total_end_epoch)
        
        response = query.execute()
        all_data = pd.DataFrame(response.data)

        # Fail-loud: sin datos en el período NO generamos un PDF engañoso. Antes se
        # renderizaba igual con totales en 0 y "eficiencia" sin sentido, devolviendo
        # HTTP 200 (el reporte "mentía"). NOTA: este endpoint aún lee la tabla legacy
        # 'measurements' (vacía en 2026); el repunte a measurements_realtime queda
        # pendiente — por ahora falla claro en vez de entregar un informe falso.
        if all_data.empty:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"No hay datos de consumo para el lugar {place_id} en el período "
                    f"{week_ranges[0][0].strftime('%d/%m/%Y')}–{week_ranges[-1][1].strftime('%d/%m/%Y')}."
                ),
            )

        report = Report(
            title=f"Informe de Consumo de Agua - {week_ranges[0][0].strftime('%d/%m/%Y')} al {week_ranges[-1][1].strftime('%d/%m/%Y')}",
            place_name=place_name
        )
        
        weeks_data = []
        max_consumption = 0
        max_wasted = 0

        for i, (week_start, week_end) in enumerate(week_ranges):
            week_number = start_week + i
            start_epoch = int(week_start.timestamp() * 1000)
            end_epoch = int(week_end.timestamp() * 1000)
            
            analysis_results = analyze_data_from_df(
                data_df=all_data,
                window_size=window_size,
                start_epoch=start_epoch,
                end_epoch=end_epoch
            )

            required_keys = [
                "weekday_peak", "weekend_peak", "weekday_total", "weekday_wasted", "weekday_efficiency",
                "weekend_total", "weekend_wasted", "weekend_efficiency", "weekday_data", "weekend_data"
            ]
            for rk in required_keys:
                if rk not in analysis_results:
                    raise HTTPException(status_code=500, detail=f"Missing '{rk}' in analysis results")

            weekday_peak = analysis_results.get('weekday_peak', {'day': 'N/A', 'consumption': 0})
            weekend_peak = analysis_results.get('weekend_peak', {'day': 'N/A', 'consumption': 0})

            weekday_section = WeekdaySection(f"Semana {week_number} - Días Laborales")
            weekday_section.add_data("dates", f"{week_start.strftime('%d/%m/%Y')} - {week_end.strftime('%d/%m/%Y')}")
            weekday_section.add_data("peak_day", weekday_peak.get('day', 'N/A'))
            weekday_section.add_data("peak_consumption", weekday_peak.get('consumption', 0))
            weekday_section.add_data("total_consumption", analysis_results['weekday_total'])
            weekday_section.add_data("wasted", analysis_results['weekday_wasted'])
            weekday_section.add_data("efficiency", analysis_results['weekday_efficiency'])
            
            weekend_section = WeekendSection(f"Semana {week_number} - Fin de Semana")
            weekend_section.add_data("dates", f"{week_start.strftime('%d/%m/%Y')} - {week_end.strftime('%d/%m/%Y')}")
            weekend_section.add_data("peak_day", weekend_peak.get('day', 'N/A'))
            weekend_section.add_data("peak_consumption", weekend_peak.get('consumption', 0))
            weekend_section.add_data("total_consumption", analysis_results['weekend_total'])
            weekend_section.add_data("wasted", analysis_results['weekend_wasted'])
            weekend_section.add_data("efficiency", analysis_results['weekend_efficiency'])
            
            week_data = {
                'title': f"Semana {week_number}",
                'dates': f"{week_start.strftime('%d/%m/%Y')} - {week_end.strftime('%d/%m/%Y')}",
                'weekday_consumption': analysis_results['weekday_total'],
                'weekend_consumption': analysis_results['weekend_total'],
                'weekday_efficiency': analysis_results['weekday_efficiency'],
                'weekend_efficiency': analysis_results['weekend_efficiency'],
                'weekday_wasted': analysis_results['weekday_wasted'],
                'weekend_wasted': analysis_results['weekend_wasted'],
                'color': WEEK_COLORS.get(f'Semana {week_number - start_week + 1}', '#1fa9c9')
            }
            weeks_data.append(week_data)
            
            max_consumption = max(max_consumption, 
                                  analysis_results['weekday_total'],
                                  analysis_results['weekend_total'])
            max_wasted = max(max_wasted, 
                             analysis_results['weekday_wasted'],
                             analysis_results['weekend_wasted'])
            
            if analysis_results['weekday_data'].empty and analysis_results['weekend_data'].empty:
                logger.warning(f"No data for week {week_number}")
                weekday_section.add_data("plot", None)
                weekend_section.add_data("plot", None)
            else:
                if not analysis_results['weekday_data'].empty:
                    weekday_plot = create_plot(
                        analysis_results['weekday_data'],
                        'Días Laborales',
                        week_start.strftime('%Y-%m-%d')
                    )
                    weekday_section.add_data("plot", weekday_plot)
                else:
                    weekday_section.add_data("plot", None)
                    
                if not analysis_results['weekend_data'].empty:
                    weekend_plot = create_plot(
                        analysis_results['weekend_data'],
                        'Fin de Semana',
                        week_start.strftime('%Y-%m-%d')
                    )
                    weekend_section.add_data("plot", weekend_plot)
                else:
                    weekend_section.add_data("plot", None)
            
            report.add_section(weekday_section)
            report.add_section(weekend_section)
        
        comparison_section = ComparisonSection("Comparación de Semanas")
        comparison_section.add_data('weeks', weeks_data)
        comparison_section.add_data('max_consumption', max_consumption)
        comparison_section.add_data('max_wasted', max_wasted)
        
        # Calculate total consumption across all weeks
        total_consumption_all_weeks = sum(
            w['weekday_consumption'] + w['weekend_consumption'] for w in weeks_data
        )
        
        weekly_trend_plot = create_weekly_trend_plot(weeks_data)
        comparison_section.add_data('weekly_trend_plot', weekly_trend_plot)
        
        # Add total consumption chart
        total_consumption_chart = create_total_weekly_consumption_chart(weeks_data)
        comparison_section.add_data('total_consumption_chart', total_consumption_chart)
        comparison_section.add_data('total_consumption_all_weeks', total_consumption_all_weeks)
        
        report.add_section(comparison_section)
        
        pdf_file = report.render()
        logger.info("Report generated successfully")

        time_end = time.time()
        print(f"Time taken: {time_end - time_start} seconds")
        return FileResponse(
            pdf_file,
            media_type='application/pdf',
            filename=f'water_analysis_weeks_{start_week}-{end_week}_{year}.pdf',
            background=BackgroundTask(os.remove, pdf_file),  # borrar el PDF tras enviarlo
        )
        
    except HTTPException as he:
        raise he
    except Exception as e:
        log_error(logger, "PDF generation", e)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/check_weeks", response_class=JSONResponse)
async def check_weeks(year: int):
    """Check which weeks have data in the specified year"""
    try:
        supabase_url = os.environ.get("SUPABASE_URL")
        supabase_key = os.environ.get("SUPABASE_KEY")
        if not supabase_url or not supabase_key:
            raise HTTPException(status_code=500, detail="Supabase credentials not set.")

        supabase = get_supabase()  # singleton (prefiere service_role)
        weeks_data = {}
        
        # Use UTC timezone starting at 00:00
        last_day = datetime(year, 12, 28, tzinfo=timezone.utc)
        total_weeks = last_day.isocalendar()[1]
        
        for week in range(1, total_weeks + 1):
            week_start = datetime.fromisocalendar(year, week, 1)
            week_start = week_start.replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=timezone.utc)
            week_end = datetime.fromisocalendar(year, week, 7)
            week_end = week_end.replace(hour=23, minute=59, second=59, microsecond=999999, tzinfo=timezone.utc)
            
            # Convert to milliseconds timestamp
            start_epoch = int(week_start.timestamp() * 1000)
            end_epoch = int(week_end.timestamp() * 1000)
            
            response = supabase.table("measurements") \
                .select("*") \
                .gte("timestamp", start_epoch) \
                .lte("timestamp", end_epoch) \
                .execute()
            
            data_records = response.data if response.data else []
            has_data = len(data_records) > 100
            
            weeks_data[week] = {
                'has_data': has_data,
                'start_date': week_start.strftime('%Y-%m-%d'),
                'end_date': week_end.strftime('%Y-%m-%d'),
                'records': len(data_records)
            }
        
        return {"year": year, "weeks": weeks_data}
        
    except HTTPException as he:
        raise he
    except Exception as e:
        log_error(logger, "week validation", e)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/new_place", response_class=JSONResponse)
async def new_place(name: str = Body(...), flow_reporter_id: int = Body(...)):
    """Create a new place"""
    supabase_url = os.environ.get("SUPABASE_URL")
    supabase_key = os.environ.get("SUPABASE_KEY")
    if not supabase_url or not supabase_key:
        raise HTTPException(status_code=500, detail="Supabase credentials not set.")

    supabase = get_supabase()  # singleton (prefiere service_role)
    response = supabase.table("places").insert({"name": name, "flow_reporter_id": flow_reporter_id}).execute()
    new_place = response.data[0] if response.data else None

    return {"success": True, "place": new_place}

@app.post("/scrape_place", response_class=JSONResponse)
async def scrape_place(
    background_tasks: BackgroundTasks,
    place_id: int = Body(...),
    start_date: str = Body(...),
    end_date: str = Body(...)
):
    """Scrape data for a place between start_date and end_date."""
    if place_id is None:
        raise HTTPException(status_code=400, detail="place_id is required")
    if start_date is None:
        raise HTTPException(status_code=400, detail="start_date is required")
    if end_date is None:
        raise HTTPException(status_code=400, detail="end_date is required")

    supabase_url = os.environ.get("SUPABASE_URL")
    supabase_key = os.environ.get("SUPABASE_KEY")
    if not supabase_url or not supabase_key:
        raise HTTPException(status_code=500, detail="Supabase credentials not set.")

    background_tasks.add_task(make_scraping_request, place_id, start_date, end_date)

    logger.info(f"Data sync queued for place {place_id}")
    return {"success": True, "message": "Scraping request initiated successfully"}

@app.post("/jobs/disaggregate/{place_id}")
def trigger(place_id: int, background_tasks: BackgroundTasks):
    task_id = _new_task_id()
    _job_set(task_id, status="queued", place_id=place_id)
    background_tasks.add_task(_run_job, task_id, infer_and_refresh, place_id)
    return {"task_id": task_id, "status": "queued"}
@app.post("/api/disaggregate/trigger")
async def trigger_disaggregation(request: DisaggregationRequest, background_tasks: BackgroundTasks):
    if not request.start_time or not request.end_time:
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=1)
        request.start_time = start_time.isoformat()
        request.end_time = end_time.isoformat()

    task_id = _new_task_id()
    _job_set(task_id, status="queued", place_id=request.place_id)
    background_tasks.add_task(
        _run_job, task_id, infer_and_refresh,
        request.place_id, request.start_time, request.end_time,
    )
    return {
        "task_id": task_id,
        "status": "queued",
        "place_id": request.place_id,
        "message": "Disaggregation started",
    }


@app.get("/api/disaggregate/status/{task_id}")
async def get_task_status(task_id: str):
    # Estado del job desde el dict en memoria (reemplaza el result backend de Celery).
    job = JOBS.get(task_id)
    if not job:
        return {"task_id": task_id, "status": "not_found", "progress": 0}
    return job


@app.post("/api/disaggregate/last-hour/{place_id}")
async def trigger_last_hour(place_id: int, background_tasks: BackgroundTasks):
    end_time = datetime.now()
    start_time = end_time - timedelta(hours=1)

    task_id = _new_task_id()
    _job_set(task_id, status="queued", place_id=place_id)
    background_tasks.add_task(
        _run_job, task_id, infer_and_refresh,
        place_id, start_time.isoformat(), end_time.isoformat(),
    )
    return {"task_id": task_id, "status": "queued", "place_id": place_id}

@app.get("/scrapers/{scraper_id}/status")
def scraper_status(scraper_id: str):
    liveness_key = f"scraper:{scraper_id}:liveness"
    state_key = f"scraper:{scraper_id}:state"
    metrics_success = f"scraper:metrics:{scraper_id}:success"
    metrics_errors = f"scraper:metrics:{scraper_id}:errors"

    alive = bool(redis.exists(liveness_key))
    last_heartbeat = int(redis.get(liveness_key)) if alive else None

    raw_state = redis.get(state_key)
    state = json.loads(raw_state) if raw_state else None

    success = int(redis.get(metrics_success) or 0)
    errors = int(redis.get(metrics_errors) or 0)

    if not alive:
        health = "DOWN"
    elif state and state.get("status") == "error":
        health = "ERROR"
    elif errors > 5:
        health = "DEGRADED"
    else:
        health = "RUNNING"

    return {
        "scraper_id": scraper_id,
        "alive": alive,
        "health": health,
        "last_heartbeat_ts": last_heartbeat,
        "state": state,
        "metrics": {
            "success": success,
            "errors": errors,
        },
    }

@app.get("/monitors", response_class=JSONResponse)
async def get_monitors():
    """
    Devuelve un listado unificado de Lugares + Estado de su Scraper.
    Sirve para detectar qué lugares de la BD no tienen un scraper corriendo (DOWN).
    """
    try:
        # 1. FUENTE DE VERDAD: Obtener todos los lugares registrados en Supabase
        response = supabase.table("places").select("*").order('id').execute()
        places = response.data if response.data else []
        
        monitors = []

        # 2. VERIFICACIÓN: Iterar cada lugar y buscar su rastro en Redis
        for place in places:
            place_id = place['id']
            # Construimos el ID que el scraper DEBERÍA tener si estuviera corriendo
            scraper_id = f"scraper_{place_id}"
            
            # Definimos las claves donde ese scraper debería estar escribiendo
            liveness_key = f"scraper:{scraper_id}:liveness"
            state_key = f"scraper:{scraper_id}:state"
            metrics_success_key = f"scraper:metrics:{scraper_id}:success"
            metrics_errors_key = f"scraper:metrics:{scraper_id}:errors"

            # 3. CONSULTA: ¿Existe este scraper en la memoria de Redis?
            # redis.exists devuelve > 0 si la clave existe
            alive = bool(redis.exists(liveness_key))
            
            # Obtenemos el estado detallado si existe
            raw_state = redis.get(state_key)
            state_data = json.loads(raw_state) if raw_state else None
            
            success_count = int(redis.get(metrics_success_key) or 0)
            error_count = int(redis.get(metrics_errors_key) or 0)

            # 4. DIAGNÓSTICO DE SALUD
            health = "DOWN" # Asumimos que no existe por defecto
            
            if alive:
                # Si existe la clave de vida, evaluamos la calidad
                if state_data and state_data.get("status") == "error":
                    health = "ERROR"
                elif error_count > 5: # Si hay muchos errores recientes
                    health = "DEGRADED"
                else:
                    health = "RUNNING"

            # 5. CONSTRUCCIÓN DEL DATO COMBINADO
            monitors.append({
                "id": place_id,
                "name": place['name'],
                "flow_reporter_id": place.get('flow_reporter_id'),
                "scraper_id": scraper_id,
                "health": health, # AQUÍ es donde ves si existe (RUNNING) o falta (DOWN)
                "metrics": {
                    "last_value": state_data.get('last_value') if state_data else None,
                    "uptime": state_data.get('uptime_sec') if state_data else 0,
                    "total_readings": success_count,
                    "errors": error_count
                }
            })

        return {"monitors": monitors}

    except Exception as e:
        log_error(logger, "fetching monitors", e)
        raise HTTPException(status_code=500, detail=f"Error fetching monitors: {str(e)}")

@app.get("/analysis/stackplot", response_class=JSONResponse)
async def get_stackplot(place_id: int, start_date: str, end_date: str, granularity: str = "day"):
    try:
        if len(start_date) == 10: start_date += "T00:00:00"
        if len(end_date) == 10: end_date += "T23:59:59"
        df = get_stackplot_data(place_id, start_date, end_date, granularity)
        if df.empty:
            return {
                "place_id": place_id,
                "granularity": granularity,
                "categories": [],
                "data": []
            }

        df = df.reset_index()
        df['time_bucket']= pd.to_datetime(df['time_bucket'], utc=True)
        df['time_bucket'] = df['time_bucket'].dt.strftime('%Y-%m-%d %H:%M:%S')

        data = df.to_dict(orient='records')

        categories = [c for c in df.columns if c != 'time_bucket']

        return {
            "place_id": place_id,
            "granularity": granularity,
            "categories": categories,
            "data": data
        }
    except Exception as e:
        log_error(logger, "fetching stackplot data", e)
        raise HTTPException(status_code=500, detail=f"Error fetching stackplot data: {str(e)}")
@app.post("/api/disaggregate/train")
def trigger_training(req: TrainingWorkflowRequest, background_tasks: BackgroundTasks):
    try:
        start_dt = datetime.fromisoformat(req.start_time.replace("Z", "+00:00"))
        end_dt = datetime.fromisoformat(req.end_time.replace("Z", "+00:00"))
    except ValueError:
        raise HTTPException(status_code=400, detail="Fechas inválidas")

    if end_dt <= start_dt:
        raise HTTPException(status_code=400, detail="end_time debe ser mayor que start_time")

    task_id = _new_task_id()
    _job_set(task_id, status="queued", place_id=req.place_id)
    background_tasks.add_task(
        _run_job, task_id, train_and_refresh_disaggregation,
        req.place_id, req.start_time, req.end_time,
    )
    return {
        "task_id": task_id,
        "status": "queued",
        "place_id": req.place_id,
        "message": "Training + disaggregation refresh started",
    }
@app.get("/api/places/{place_id}/available-dates")
def get_available_dates(place_id: int, year: int, month: int):
    sb = get_supabase()
    res = sb.rpc("get_available_days", {
        "p_place_id": place_id,
        "p_year": year,
        "p_month": month,
    }).execute()

    print("place_id:", place_id, "year:", year, "month:", month)
    print("rpc data:", res.data)

    return {
        "available_days": [r["day"] for r in (res.data or [])]
    }


@app.get("/api/places/{place_id}/data-range")
def get_data_range(place_id: int):
    """Rango temporal con datos en measurements_realtime para un place.
    El frontend lo usa para inicializar el selector en el último mes CON datos
    (en vez del mes actual del reloj, que suele estar vacío)."""
    sb = get_supabase()
    try:
        mx = (sb.table("measurements_realtime").select("timestamp")
              .eq("place_id", place_id).order("timestamp", desc=True).limit(1).execute())
        mn = (sb.table("measurements_realtime").select("timestamp")
              .eq("place_id", place_id).order("timestamp").limit(1).execute())
        if not mx.data or not mn.data:
            return {"has_data": False}
        return {"has_data": True, "min": mn.data[0]["timestamp"], "max": mx.data[0]["timestamp"]}
    except Exception as e:
        log_error(logger, f"data-range place {place_id}", e)
        return {"has_data": False, "error": str(e)}


@app.get("/api/places/{place_id}/water-health")
def get_water_health(place_id: int, days: int = 30):
    """Salud hídrica / detección de fuga por CAUDAL BASE NOCTURNO (02:00-04:59).

    Cuando nadie usa agua el caudal debería caer a ~0. Un PISO sostenido (p10 del
    caudal nocturno) por encima del umbral indica consumo continuo: fuga, goteo o
    inodoro corriendo. Se reporta el piso típico entre noches y los litros/día que
    representa, para que el operador lo detecte temprano (es de los hallazgos de
    mayor valor en gestión hídrica)."""
    BETA = 0.11  # fracción del caudal medio que es consumo nocturno LEGÍTIMO (MNC=β·Q)
    sb = get_supabase()
    try:
        res = sb.rpc("water_health", {"p_place_id": place_id, "p_days": days}).execute()
        nights = res.data or []
        # nights viene ordenado por día DESC (recientes primero).
        series = [(n["dia"], float(n["base_lmin"])) for n in nights if n.get("base_lmin") is not None]
        if not series:
            return {"status": "sin_datos", "nights_analyzed": 0, "nights": nights}
        bases = [b for _, b in series]

        # MNC = β·Q: consumo nocturno LEGÍTIMO esperado como fracción del caudal medio
        # (outflow), umbral ADAPTATIVO por lugar en vez de fijo (Frontiers in Water 2025).
        # La fuga = base medido - MNC. Piso mínimo para no marcar lugares de uso muy bajo.
        try:
            qres = sb.rpc("place_avg_flow", {"p_place_id": place_id, "p_days": days}).execute()
            q_avg = float(qres.data[0]) if isinstance(qres.data, list) else float(qres.data or 0.0)
        except Exception:
            q_avg = 0.0
        mnc = round(BETA * q_avg, 3)
        threshold = round(max(mnc, 0.10), 3)

        base_sostenida = sorted(bases)[len(bases) // 2]   # mediana del piso nocturno
        leak_excess = max(0.0, base_sostenida - mnc)       # caudal de fuga estimado
        nights_flagged = sum(1 for b in bases if b > threshold)

        # CHANGE-POINT: ¿el caudal base SALTÓ de forma permanente (una fuga que
        # apareció)? Compara la mediana de las noches recientes vs las anteriores
        # (evita confundir un piso constante con uno que se disparó).
        change = None
        if len(series) >= 6:
            half = len(series) // 2
            recent = sorted(b for _, b in series[:half])
            older = sorted(b for _, b in series[half:])
            mr, mo = recent[len(recent) // 2], older[len(older) // 2]
            if mr > mo + threshold and mr > threshold:
                change = {"detectado": True, "desde": series[half][0],
                          "antes_lmin": round(mo, 3), "ahora_lmin": round(mr, 3)}

        is_leak = base_sostenida > threshold
        status = ("fuga_probable" if (is_leak or change)
                  else ("revisar" if nights_flagged else "ok"))
        return {
            "status": status,
            "base_flow_lmin": round(base_sostenida, 3),
            "expected_night_lmin": mnc,                 # MNC = β·Q (consumo legítimo)
            "q_avg_lmin": round(q_avg, 3),
            "leak_flow_lmin": round(leak_excess, 3),
            "estimated_daily_waste_l": round(leak_excess * 1440.0, 1),  # solo el EXCESO
            "threshold_lmin": threshold,
            "method": "MNC=beta*Q (beta=0.11) + change-point",
            "change_point": change,
            "nights_analyzed": len(nights),
            "nights_flagged": nights_flagged,
            "nights": nights,
        }
    except Exception as e:
        log_error(logger, f"water-health place {place_id}", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/places/{place_id}/calibration-events")
def get_calibration_events(place_id: int, n: int = 4):
    """Eventos MÁS TÍPICOS de cada cluster para que el operador confirme el fixture.

    Typicality sampling (Hacohen ICML 2022): para pocas confirmaciones gana mostrar
    los eventos más REPRESENTATIVOS de cada cluster (cercanos al centroide), no los
    ambiguos. El operador los etiqueta y eso refina el modelo (semi-supervisado)."""
    sb = get_supabase()
    try:
        res = (
            sb.table("disaggregation_events")
            .select("id, device_name, flow_rate, duration_s, volume_liters, start_time")
            .eq("place_id", place_id)
            .neq("device_name", "No Detectado")
            .execute()
        )
        rows = res.data or []
        if not rows:
            return {"clusters": []}
        df = pd.DataFrame(rows)
        for c in ("flow_rate", "duration_s", "volume_liters"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=["flow_rate", "duration_s", "volume_liters"])

        import numpy as _np
        out = []
        for label, g in df.groupby("device_name"):
            med = g[["flow_rate", "duration_s", "volume_liters"]].median()
            # distancia al centroide (mediana) en espacio normalizado log para dur/vol
            d = (
                ((g["flow_rate"] - med["flow_rate"]) / (med["flow_rate"] + 1.0)) ** 2
                + (_np.log1p(g["duration_s"]) - _np.log1p(med["duration_s"])) ** 2
                + (_np.log1p(g["volume_liters"]) - _np.log1p(med["volume_liters"])) ** 2
            )
            typ = g.assign(_d=d).nsmallest(max(1, n), "_d")
            out.append({
                "label": label,
                "n_events": int(len(g)),
                "signature": {
                    "flow": round(float(med["flow_rate"]), 2),
                    "duration_s": round(float(med["duration_s"]), 0),
                    "volume_l": round(float(med["volume_liters"]), 2),
                },
                "events": [
                    {
                        "id": int(r["id"]),
                        "start_time": r["start_time"],
                        "flow": round(float(r["flow_rate"]), 2),
                        "duration_s": round(float(r["duration_s"]), 0),
                        "volume_l": round(float(r["volume_liters"]), 2),
                    }
                    for _, r in typ.iterrows()
                ],
            })
        return {"clusters": out}
    except Exception as e:
        log_error(logger, f"calibration-events place {place_id}", e)
        raise HTTPException(status_code=500, detail=str(e))


class ConfirmEventRequest(BaseModel):
    mean_flow: float
    duration_s: float
    volume_liters: float
    confirmed_label: str


@app.post("/api/places/{place_id}/confirm")
def confirm_event(place_id: int, req: ConfirmEventRequest):
    """El operador confirma el fixture real de un evento típico (semilla semi-supervisada)."""
    from services.supabase_service import save_confirmation
    try:
        rec = save_confirmation(
            place_id, req.mean_flow, req.duration_s, req.volume_liters, req.confirmed_label
        )
        return {"success": True, "confirmation": rec}
    except Exception as e:
        log_error(logger, f"confirm place {place_id}", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/places/{place_id}/disaggregation-profiles")
def get_disaggregation_profiles(place_id: int):
    sb = get_supabase()

    response = (
        sb.table("disaggregation_profiles")
        .select("id, place_id, name, label, mean_flow, mean_duration, median_volume_l, cv_volume")
        .eq("place_id", place_id)
        .eq("is_official", True)
        .order("name")
        .execute()
    )

    return {
        "profiles": response.data or []
    }
@app.patch("/api/disaggregation-profiles/{profile_id}/label")
def update_disaggregation_profile_label(
    profile_id: int,
    req: UpdateDisaggregationProfileLabelRequest,
):
    sb = get_supabase()

    label = req.label.strip() if isinstance(req.label, str) else None
    if label == "":
        label = None

    response = (
        sb.table("disaggregation_profiles")
        .update({"label": label})
        .eq("id", profile_id)
        .execute()
    )

    if not response.data:
        raise HTTPException(status_code=404, detail="Profile not found")

    return {
        "success": True,
        "profile": response.data[0],
    }