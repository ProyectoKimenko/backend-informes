from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.requests import Request
from fastapi import Body
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from jinja2 import Environment, FileSystemLoader
from weasyprint import HTML
from typing import List, Dict, Tuple
from datetime import datetime, timedelta
import calendar
import colorsys
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import tempfile
from supabase import create_client
import logging
import sys
import requests
from src.analysis import analyze_data
from src.report import Report
from src.report_sections import WeekdaySection, WeekendSection, ComparisonSection
import time

# Simplified logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app and templates
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory=tempfile.gettempdir()), name="static")

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

def create_plot(data, title_prefix, date_str):
    required_columns = ['flow_rate', 'RollingMin']
    for col in required_columns:
        if col not in data.columns:
            logger.error(f"Missing required column '{col}' in data for plotting.")
            raise ValueError(f"Missing required column '{col}' in data")

    has_data = len(data) > 0 and data['flow_rate'].sum() > 0
    
    plt.figure(figsize=(12, 6))
    plt.style.use('default')
    
    if not has_data:
        ax = plt.gca()
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
                plt.plot(x, y, color='#1f77b4', linewidth=2, 
                         label='Flujo total' if i == 0 else "")

        min_segments = split_at_gaps(data.index, data['RollingMin'], min_mask)
        for i, (x, y) in enumerate(min_segments):
            if len(x) > 1:
                plt.plot(x, y, color='#d62728', linewidth=2, 
                         label='Límite de desperdicio' if i == 0 else "")
                plt.fill_between(x, 0, y, color='#d62728', alpha=0.3,
                                 label='Flujo desperdiciado' if i == 0 else "")
        
        max_val = max(data['flow_rate'].max(), 1)
        plt.ylim(0, max_val * 1.1)
        plt.legend(frameon=True, fancybox=True, shadow=True, fontsize=10)

    plt.title(f'{title_prefix} - {date_str}', pad=20, fontsize=14)
    plt.xlabel('Tiempo', fontsize=12)
    plt.ylabel('Flujo (litros)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as temp_file:
        plt.savefig(temp_file.name, dpi=50, bbox_inches='tight')
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

    weeks = [f"Semana {i + 1}" for i in range(len(weeks_data))]
    weekday_consumptions = [week['weekday_consumption'] for week in weeks_data]
    weekend_consumptions = [week['weekend_consumption'] for week in weeks_data]
    weekday_efficiencies = [week['weekday_efficiency'] for week in weeks_data]
    weekend_efficiencies = [week['weekend_efficiency'] for week in weeks_data]
    weekday_losses = [week['weekday_wasted'] for week in weeks_data]
    weekend_losses = [week['weekend_wasted'] for week in weeks_data]
    colors = [week['color'] for week in weeks_data]

    all_zero_consumption = all(c == 0 for c in weekday_consumptions + weekend_consumptions)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), height_ratios=[1, 1])
    fig.suptitle('Tendencias Semanales de Consumo de Agua', fontsize=20, y=0.95)

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

    ax1.set_ylabel('Litros', fontsize=14)
    ax1.set_title('Consumo Total y Pérdidas por Semana', fontsize=16, pad=20)
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

    ax2.set_ylabel('Porcentaje de Eficiencia', fontsize=14)
    ax2.set_title('Eficiencia por Semana', fontsize=16, pad=20)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.set_ylim(0, 105)
    ax2.tick_params(axis='both', labelsize=12)
    ax2.legend(fontsize=12, bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout(rect=[0, 0, 0.85, 0.95], h_pad=0.8)

    with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as temp_file:
        plt.savefig(temp_file.name, bbox_inches='tight', dpi=50)
        plt.close()
    return temp_file.name

def get_dates_from_week_number(year: int, week: int, num_weeks: int = 4) -> List[Tuple[datetime, datetime]]:
    """Get start and end dates for specified week number(s)."""
    week_ranges = []
    if num_weeks < 1:
        raise ValueError("num_weeks must be at least 1")

    if week == 1:
        jan_first = datetime(year, 1, 1)
        days_to_monday = (jan_first.weekday()) % 7
        week_start = jan_first - timedelta(days=days_to_monday)
        
        if week_start.year < year and (jan_first - week_start).days > 3:
            week_start = jan_first
            
        week_end = week_start + timedelta(days=6)
        week_end = week_end.replace(hour=23, minute=59, second=59)
        week_ranges.append((week_start, week_end))
        
        for i in range(1, num_weeks):
            next_week_start = week_end + timedelta(days=1)
            next_week_start = next_week_start.replace(hour=0, minute=0, second=0)
            next_week_end = next_week_start + timedelta(days=6)
            next_week_end = next_week_end.replace(hour=23, minute=59, second=59)
            week_ranges.append((next_week_start, next_week_end))
            week_end = next_week_end
            
        return week_ranges
    
    for i in range(num_weeks):
        current_week = week + i
        if current_week < 1 or current_week > 53:
            raise ValueError(f"Requested out-of-range week number: {current_week}")

        try:
            iso_calendar = datetime.fromisocalendar(year, current_week, 1)
        except ValueError:
            raise ValueError(f"No valid week {current_week} found in year {year}.")

        week_start = datetime.fromisocalendar(iso_calendar.year, current_week, 1)
        week_end = datetime.fromisocalendar(iso_calendar.year, current_week, 7).replace(hour=23, minute=59, second=59)
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
                "end_date": end_date
            },
            headers={"Content-Type": "application/json"},
            timeout=180
        )
        logger.info(f"Scraping request completed for place {place_id}")
    except Exception as e:
        logger.error(f"Background scraping request failed for place {place_id}: {str(e)}")

# Initialize Supabase client
supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_KEY")
supabase = create_client(supabase_url, supabase_key)

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
        logger.error(f"Error fetching places: {str(e)}")
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

        try:
            results = analyze_data(
                window_size=window_size,
                start_epoch=start_epoch,
                end_epoch=end_epoch,
                place_id=place_id
            )
        except Exception as e:
            logger.warning(f"No data found for analysis: {str(e)}")
            return {
                "total_water_wasted_weekdays":      None,
                "efficiency_percentage_weekdays":   None,
                "total_water_consumed_weekdays":    None,
                "total_water_wasted_weekends":      None,
                "efficiency_percentage_weekends":   None,
                "total_water_consumed_weekends":    None,
                "time_series":                      []
            }

        combined = pd.concat([results['weekday_data'], results['weekend_data']])
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
        logger.error(f"Error in analysis_json: {str(e)}")
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
            
            analysis_results = analyze_data(
                window_size=window_size,
                start_epoch=start_epoch,
                end_epoch=end_epoch,
                place_id=place_id
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
                logger.warning(f"No data available for plotting week {week_number}")
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
        
        weekly_trend_plot = create_weekly_trend_plot(weeks_data)
        comparison_section.add_data('weekly_trend_plot', weekly_trend_plot)
        
        report.add_section(comparison_section)
        
        pdf_file = report.render()
        logger.info(f"PDF report generated successfully")

        time_end = time.time()
        print(f"Time taken: {time_end - time_start} seconds")
        return FileResponse(
            pdf_file,
            media_type='application/pdf',
            filename=f'water_analysis_weeks_{start_week}-{end_week}_{year}.pdf'
        )
        
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error generating weekly PDF: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/check_weeks", response_class=JSONResponse)
async def check_weeks(year: int):
    """Check which weeks have data in the specified year"""
    try:
        supabase_url = os.environ.get("SUPABASE_URL")
        supabase_key = os.environ.get("SUPABASE_KEY")
        if not supabase_url or not supabase_key:
            raise HTTPException(status_code=500, detail="Supabase credentials not set.")

        supabase = create_client(supabase_url, supabase_key)
        weeks_data = {}
        
        last_day = datetime(year, 12, 28)
        total_weeks = last_day.isocalendar()[1]
        
        for week in range(1, total_weeks + 1):
            week_start = datetime.fromisocalendar(year, week, 1)
            week_end = datetime.fromisocalendar(year, week, 7)
            week_end = week_end.replace(hour=23, minute=59, second=59)
            
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
        logger.error(f"Error checking weeks: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/new_place", response_class=JSONResponse)
async def new_place(name: str = Body(...), flow_reporter_id: int = Body(...)):
    """Create a new place"""
    supabase_url = os.environ.get("SUPABASE_URL")
    supabase_key = os.environ.get("SUPABASE_KEY")
    if not supabase_url or not supabase_key:
        raise HTTPException(status_code=500, detail="Supabase credentials not set.")

    supabase = create_client(supabase_url, supabase_key)
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
    
    logger.info(f"Scraping request queued for place {place_id}")
    return {"success": True, "message": "Scraping request initiated successfully"}