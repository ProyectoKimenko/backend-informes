from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.requests import Request
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

from src.analysis import analyze_data
from src.scrape import fetch_page_info
from src.report import Report
from src.report_sections import WeekdaySection, WeekendSection, ComparisonSection

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

# Create logger
logger = logging.getLogger(__name__)
logger.info("Starting application initialization")

# Initialize FastAPI app and templates
app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory=tempfile.gettempdir()), name="static")
logger.info("FastAPI app and templates initialized")

# Constants
WEEK_COLORS = {
    'Semana 1': '#1fa9c9',
    'Semana 2': '#34495e', 
    'Semana 3': '#2ecc71',
    'Semana 4': '#e67e22',
    'Semana 5': '#9b59b6'
}
logger.debug(f"Week colors configured: {WEEK_COLORS}")

# Helper Functions
def adjust_color_brightness(hex_color, factor):
    """Adjust the brightness of a hex color"""
    logger.debug(f"Adjusting color brightness for {hex_color} with factor {factor}")
    hex_color = hex_color.lstrip('#')
    rgb = tuple(int(hex_color[i:i+2], 16)/255 for i in (0, 2, 4))
    hsv = colorsys.rgb_to_hsv(*rgb)
    hsv = (hsv[0], hsv[1], min(1, hsv[2] * factor))
    rgb = colorsys.hsv_to_rgb(*hsv)
    adjusted_color = '#{:02x}{:02x}{:02x}'.format(int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255))
    logger.debug(f"Adjusted color: {adjusted_color}")
    return adjusted_color

def split_at_gaps(index, values, mask):
    logger.debug(f"Splitting data at gaps with {len(index)} points")
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
            if np.sum(segment_mask) > 1:  # Ensure more than one point
                segments.append((index[segment_mask], values[segment_mask]))
        start_idx = end_idx

    # Handle the last segment
    segment_mask = mask.copy()
    segment_mask[:start_idx] = False
    if np.sum(segment_mask) > 1:
        segments.append((index[segment_mask], values[segment_mask]))

    logger.debug(f"Created {len(segments)} segments")
    return segments

def create_plot(data, title_prefix, date_str):
    # Validate required columns
    required_columns = ['flow_rate', 'RollingMin']
    for col in required_columns:
        if col not in data.columns:
            logger.error(f"Missing required column '{col}' in data for plotting.")
            raise ValueError(f"Missing required column '{col}' in data")

    logger.info(f"Creating plot for {title_prefix} - {date_str}")
    plt.figure(figsize=(12, 6))
    plt.style.use('default')
    
    # Create masks for valid data
    flow_mask = ~np.isnan(data['flow_rate'])
    min_mask = ~np.isnan(data['RollingMin'])

    logger.debug(f"Valid flow data points: {np.sum(flow_mask)}")
    logger.debug(f"Valid min data points: {np.sum(min_mask)}")

    # Plot flow rate segments
    flow_segments = split_at_gaps(data.index, data['flow_rate'], flow_mask)
    for x, y in flow_segments:
        plt.plot(x, y, color='#1f77b4', linewidth=2, 
                 label='Flujo total' if x is flow_segments[0][0] else "")

    min_segments = split_at_gaps(data.index, data['RollingMin'], min_mask)
    for x, y in min_segments:
        if len(x) > 1:  # Ensure more than one point
            plt.plot(x, y, color='#d62728', linewidth=2, 
                     label='Límite de desperdicio' if x is min_segments[0][0] else "")
            plt.fill_between(x, 0, y, color='#d62728', alpha=0.3,
                             label='Flujo desperdiciado' if x is min_segments[0][0] else "")

    # Configure plot formatting
    plt.title(f'{title_prefix} - {date_str}', pad=20, fontsize=14)
    plt.xlabel('Tiempo', fontsize=12)
    plt.ylabel('Flujo (litros)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(frameon=True, fancybox=True, shadow=True, fontsize=10)
    plt.tight_layout()

    with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as temp_file:
        plt.savefig(temp_file.name, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Plot saved to temporary file: {temp_file.name}")
    return temp_file.name

def create_weekly_trend_plot(weeks_data):
    logger.info(f"Creating weekly trend plot with {len(weeks_data)} weeks of data")

    # Validate that weeks_data contains all required keys
    required_keys = [
        'weekday_consumption', 'weekend_consumption', 'weekday_efficiency', 
        'weekend_efficiency', 'weekday_wasted', 'weekend_wasted', 'color'
    ]
    for i, w in enumerate(weeks_data, start=1):
        for k in required_keys:
            if k not in w:
                logger.error(f"Missing key '{k}' in week data for Semana {i}.")
                raise ValueError(f"Missing key '{k}' in week data")

    # Extract data
    weeks = [f"Semana {i + 1}" for i in range(len(weeks_data))]
    weekday_consumptions = [week['weekday_consumption'] for week in weeks_data]
    weekend_consumptions = [week['weekend_consumption'] for week in weeks_data]
    weekday_efficiencies = [week['weekday_efficiency'] for week in weeks_data]
    weekend_efficiencies = [week['weekend_efficiency'] for week in weeks_data]
    weekday_losses = [week['weekday_wasted'] for week in weeks_data]
    weekend_losses = [week['weekend_wasted'] for week in weeks_data]
    colors = [week['color'] for week in weeks_data]

    logger.debug(f"Weekday consumptions: {weekday_consumptions}")
    logger.debug(f"Weekend consumptions: {weekend_consumptions}")
    logger.debug(f"Weekday efficiencies: {weekday_efficiencies}")
    logger.debug(f"Weekend efficiencies: {weekend_efficiencies}")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), height_ratios=[1, 1])
    fig.suptitle('Tendencias Semanales de Consumo de Agua', fontsize=20, y=0.95)

    width = 0.35
    x = np.arange(len(weeks))

    # Consumption and Losses Plot
    ax1.bar(x - width / 2, [c - w for c, w in zip(weekday_consumptions, weekday_losses)], width, 
            label='Consumo Laboral', color=colors)
    ax1.bar(x + width / 2, [c - w for c, w in zip(weekend_consumptions, weekend_losses)], width, 
            label='Consumo Fin de Semana', color=[adjust_color_brightness(c, 0.7) for c in colors])
    ax1.bar(x - width / 2, weekday_losses, width, 
            bottom=[c - w for c, w in zip(weekday_consumptions, weekday_losses)],
            label='Pérdidas Laborales', color=[adjust_color_brightness(c, 0.5) for c in colors])
    ax1.bar(x + width / 2, weekend_losses, width, 
            bottom=[c - w for c, w in zip(weekend_consumptions, weekend_losses)],
            label='Pérdidas Fin de Semana', color=[adjust_color_brightness(c, 0.3) for c in colors])

    ax1.set_ylabel('Litros', fontsize=14)
    ax1.set_title('Consumo Total y Pérdidas por Semana', fontsize=16, pad=20)
    ax1.set_xticks(x)
    ax1.set_xticklabels(weeks, fontsize=12)
    ax1.tick_params(axis='y', labelsize=12)
    ax1.legend(fontsize=12, bbox_to_anchor=(1.05, 1), loc='upper left')

    # Efficiency Plot
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
    ax2.set_ylim(0, 100)
    ax2.tick_params(axis='both', labelsize=12)
    ax2.legend(fontsize=12, bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout(rect=[0, 0, 0.85, 0.95], h_pad=0.8)

    with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as temp_file:
        plt.savefig(temp_file.name, bbox_inches='tight', dpi=300)
        plt.close()
        logger.info(f"Weekly trend plot saved to: {temp_file.name}")
    return temp_file.name


def get_dates_from_week_number(year: int, week: int, num_weeks: int = 4) -> List[Tuple[datetime, datetime]]:
    """
    Get start and end dates for specified week number(s).
    Weeks start on Monday and end on Sunday (ISO week numbering).
    """
    week_ranges = []
    if num_weeks < 1:
        logger.error("num_weeks must be >= 1")
        raise ValueError("num_weeks must be at least 1")

    for i in range(num_weeks):
        current_week = week + i
        # If current_week < 1 or > 53, handle or raise error
        if current_week < 1 or current_week > 53:
            error_msg = f"Requested out-of-range week number: {current_week}. ISO weeks are typically between 1 and 53."
            logger.error(error_msg)
            raise ValueError(error_msg)

        try:
            iso_calendar = datetime.fromisocalendar(year, current_week, 1)
        except ValueError:
            # If the current_week exceeds the number of weeks in the year, this might fail
            error_msg = f"No valid week {current_week} found in year {year}."
            logger.error(error_msg)
            raise ValueError(error_msg)

        week_start = datetime.fromisocalendar(iso_calendar.year, current_week, 1)
        week_end = datetime.fromisocalendar(iso_calendar.year, current_week, 7).replace(hour=23, minute=59, second=59)
        week_ranges.append((week_start, week_end))
        logger.info(f"Week {current_week} of {iso_calendar.year}: {week_start.strftime('%Y-%m-%d')} to {week_end.strftime('%Y-%m-%d')}")

    return week_ranges


# Initialize Supabase client
supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_KEY")
supabase = create_client(supabase_url, supabase_key)

# API Endpoints
@app.get("/ping", status_code=200)
def ping():
    logger.debug("Ping request received")
    return {"status": "success", "message": "Pong"}


@app.get("/view_analysis", response_class=HTMLResponse)
async def view_analysis(
    request: Request, 
    table_name: str = "measurements",
    window_size: int = 60, 
    end_week: int = 27,
    start_week: int = 27,
    year: int = 2024,
    place_id: int = 1
):
    logger.info(f"Starting view_analysis with params: table={table_name}, window={window_size}, year={year}, start_week={start_week}, end_week={end_week}, place_id={place_id}")
    try:
        # Query the "places" table
        response = supabase.table("places").select("*").execute()
        places = response.data if response.data else []

        # Validate inputs
        if year is None or end_week is None or start_week is None:
            error_msg = "Parameters 'year', 'start_week', and 'end_week' must be provided."
            logger.error(error_msg)
            raise HTTPException(status_code=400, detail=error_msg)

        if not (1 <= start_week <= 53 and 1 <= end_week <= 53):
            error_msg = "Invalid 'start_week' or 'end_week'. Both must be between 1 and 53."
            logger.error(error_msg)
            raise HTTPException(status_code=400, detail=error_msg)

        if end_week < start_week:
            error_msg = f"'end_week' ({end_week}) must be greater or equal to 'start_week' ({start_week})."
            logger.error(error_msg)
            raise HTTPException(status_code=400, detail=error_msg)

        num_weeks = end_week - start_week + 1
        if num_weeks > 4:
            error_msg = f"Cannot select more than 4 weeks. Selected: {num_weeks}."
            logger.error(error_msg)
            raise HTTPException(status_code=400, detail=error_msg)

        # Get week ranges
        week_ranges = get_dates_from_week_number(year, start_week, num_weeks)
        if not week_ranges:
            error_msg = f"No valid weeks found starting from week {start_week} of {year}."
            logger.error(error_msg)
            raise HTTPException(status_code=400, detail=error_msg)
        
        start_date = week_ranges[0][0]
        end_date = week_ranges[-1][1]
        
        start_epoch = int(start_date.timestamp() * 1000)
        end_epoch = int(end_date.timestamp() * 1000)
        logger.info(f"Calculated epochs for selected weeks: start={start_epoch}, end={end_epoch}")

        logger.info("Calling analyze_data function")
        # The analyze_data function now queries the "measurements" table internally
        analysis_results = analyze_data(
            window_size=window_size, 
            start_epoch=start_epoch,
            end_epoch=end_epoch,
            place_id=place_id
        )

        required_keys = [
            "weekday_total", "weekend_total", "weekday_wasted", "weekend_wasted", 
            "weekday_efficiency", "weekend_efficiency", "weekday_data", "weekend_data"
        ]
        for rk in required_keys:
            if rk not in analysis_results:
                error_msg = f"Missing '{rk}' in analysis results."
                logger.error(error_msg)
                raise HTTPException(status_code=500, detail=error_msg)

        if (analysis_results['weekday_total'] == 0 and 
            analysis_results['weekend_total'] == 0):
            return templates.TemplateResponse("analysis.html", {
                "request": request,
                "error_message": f"No se encontraron datos para el período seleccionado: Semanas {start_week}-{end_week}, {year}",
                "plot_url": None,
                "window_size": window_size,
                "end_week": end_week,
                "year": year,
                "total_water_wasted_weekdays": 0,
                "efficiency_percentage_weekdays": 0,
                "total_water_consumed_weekdays": 0,
                "total_water_wasted_weekends": 0,
                "efficiency_percentage_weekends": 0,
                "total_water_consumed_weekends": 0,
                "places": places,
            })
        
        logger.info("Successfully got analysis results")

        # Combine weekday and weekend data
        combined_data = pd.concat([
            analysis_results['weekday_data'],
            analysis_results['weekend_data']
        ])

        logger.info("Creating plot")
        if combined_data.empty:
            plot_url = None
            logger.warning("No data available for plotting")
        else:
            plot_file = create_plot(
                combined_data,
                f'Análisis Semanas {start_week}-{end_week} - {year}',
                f'{year}-W{end_week:02d}'
            )
            plot_url = f"/static/{os.path.basename(plot_file)}"
            logger.info(f"Plot created at: {plot_url}")

        return templates.TemplateResponse("analysis.html", {
            "request": request,
            "total_water_wasted_weekdays": analysis_results['weekday_wasted'],
            "efficiency_percentage_weekdays": analysis_results['weekday_efficiency'], 
            "total_water_consumed_weekdays": analysis_results['weekday_total'],
            "total_water_wasted_weekends": analysis_results['weekend_wasted'],
            "efficiency_percentage_weekends": analysis_results['weekend_efficiency'],
            "total_water_consumed_weekends": analysis_results['weekend_total'],
            "plot_url": plot_url,
            "window_size": window_size,
            "end_week": end_week,
            "year": year,
            "error_message": None,
            "places": places,
        })
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error in view_analysis: {str(e)}", exc_info=True)
        return templates.TemplateResponse("analysis.html", {
            "request": request,
            "error_message": f"An error occurred: {str(e)}",
            "plot_url": None,
            "window_size": window_size,
            "end_week": end_week,
            "year": year,
            "places": [],
        })


@app.get("/generate_weekly_pdf", status_code=200)
async def generate_weekly_pdf(
    year: int,
    end_week: int,
    start_week: int,
    window_size: int = 60,
    place_id: int = None
):
    logger.info(f"Starting generate_weekly_pdf with params: year={year}, start_week={start_week}, end_week={end_week}, window_size={window_size}, place_id={place_id}")
    try:
        # Fetch places data
        response = supabase.table("places").select("*").execute()
        places = response.data if response.data else []

        # Find the place name
        place_name = next((place['name'] for place in places if place['id'] == place_id), "Unknown Location")

        # Validate inputs
        if not (1 <= start_week <= 53 and 1 <= end_week <= 53):
            error_msg = "Invalid 'start_week' or 'end_week'. Both must be between 1 and 53."
            logger.error(error_msg)
            raise HTTPException(status_code=400, detail=error_msg)

        if end_week < start_week:
            error_msg = f"'end_week' ({end_week}) must be greater or equal to 'start_week' ({start_week})."
            logger.error(error_msg)
            raise HTTPException(status_code=400, detail=error_msg)

        num_weeks = end_week - start_week + 1
        if num_weeks > 4:
            error_msg = f"Cannot select more than 4 weeks. Selected: {num_weeks}."
            logger.error(error_msg)
            raise HTTPException(status_code=400, detail=error_msg)
        
        week_ranges = get_dates_from_week_number(year, start_week, num_weeks)
        if not week_ranges:
            error_msg = f"No valid weeks found starting from week {start_week} of {year}."
            logger.error(error_msg)
            raise HTTPException(status_code=400, detail=error_msg)
        
        report = Report(
            title=f"Informe de Consumo de Agua - {week_ranges[0][0].strftime('%d/%m/%Y')} al {week_ranges[-1][1].strftime('%d/%m/%Y')}",
            place_name=place_name
        )
        
        weeks_data = []
        max_consumption = 0
        max_wasted = 0

        for i, (week_start, week_end) in enumerate(week_ranges):
            week_number = start_week + i
            logger.info(f"Processing week {week_number}: {week_start} to {week_end}")
            start_epoch = int(week_start.timestamp() * 1000)
            end_epoch = int(week_end.timestamp() * 1000)
            
            analysis_results = analyze_data(
                window_size=window_size,
                start_epoch=start_epoch,
                end_epoch=end_epoch,
                place_id=place_id
            )
            logger.info(f"Got analysis results for week {week_number}")

            # Check required keys
            required_keys = [
                "weekday_peak", "weekend_peak", "weekday_total", "weekday_wasted", "weekday_efficiency",
                "weekend_total", "weekend_wasted", "weekend_efficiency", "weekday_data", "weekend_data"
            ]
            for rk in required_keys:
                if rk not in analysis_results:
                    error_msg = f"Missing '{rk}' in analysis results for week {week_number}."
                    logger.error(error_msg)
                    raise HTTPException(status_code=500, detail=error_msg)

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
            
            # Create plots for the PDF
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
        
        # Add comparison section
        comparison_section = ComparisonSection("Comparación de Semanas")
        comparison_section.add_data('weeks', weeks_data)
        comparison_section.add_data('max_consumption', max_consumption)
        comparison_section.add_data('max_wasted', max_wasted)
        
        weekly_trend_plot = create_weekly_trend_plot(weeks_data)
        comparison_section.add_data('weekly_trend_plot', weekly_trend_plot)
        
        report.add_section(comparison_section)
        
        pdf_file = report.render()
        logger.info(f"PDF report rendered successfully: {pdf_file}")
        
        return FileResponse(
            pdf_file,
            media_type='application/pdf',
            filename=f'water_analysis_weeks_{start_week}-{end_week}_{year}.pdf'
        )
        
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error generating weekly PDF: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/check_weeks", response_class=JSONResponse)
async def check_weeks(year: int):
    """Check which weeks have data in the specified year"""
    logger.info(f"Checking data availability for weeks in {year}")
    try:
        # Check Supabase credentials
        supabase_url = os.environ.get("SUPABASE_URL")
        supabase_key = os.environ.get("SUPABASE_KEY")
        if not supabase_url or not supabase_key:
            error_msg = "Supabase credentials not set."
            logger.error(error_msg)
            raise HTTPException(status_code=500, detail=error_msg)

        supabase = create_client(supabase_url, supabase_key)

        weeks_data = {}
        
        # Get the total number of ISO weeks in the year
        last_day = datetime(year, 12, 28)
        total_weeks = last_day.isocalendar()[1]
        
        # Check each week of the year
        for week in range(1, total_weeks + 1):
            week_start = datetime.fromisocalendar(year, week, 1)  # Monday
            week_end = datetime.fromisocalendar(year, week, 7)    # Sunday
            week_end = week_end.replace(hour=23, minute=59, second=59)
            
            start_epoch = int(week_start.timestamp() * 1000)
            end_epoch = int(week_end.timestamp() * 1000)
            
            # Query the new "measurements" table and "timestamp" field
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
            
            logger.debug(f"Week {week}: {has_data} ({len(data_records)} records)")
        
        return {"year": year, "weeks": weeks_data}
        
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error checking weeks: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
