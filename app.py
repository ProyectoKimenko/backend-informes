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

# Configure logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('app.log')
    ]
)

# Create logger for this module
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
    logger.info(f"Creating plot for {title_prefix} - {date_str}")
    plt.figure(figsize=(12, 6))
    plt.style.use('default')
    
    # Create masks for valid data
    flow_mask = ~np.isnan(data['Flow rate'])
    min_mask = ~np.isnan(data['RollingMin'])
    min_mask = ~np.isnan(data['RollingMin'])
    
    logger.debug(f"Valid flow data points: {np.sum(flow_mask)}")
    logger.debug(f"Valid min data points: {np.sum(min_mask)}")
    
    # Plot flow rate segments
    flow_segments = split_at_gaps(data.index, data['Flow rate'], flow_mask)
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

    # Create figure with larger size
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), height_ratios=[1, 1])
    fig.suptitle('Tendencias Semanales de Consumo de Agua', fontsize=20, y=0.95)

    width = 0.35
    x = np.arange(len(weeks))

    # Consumption and Losses Plot (now showing actual consumption and losses separately)
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

    # Efficiency Plot with larger markers and fonts
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

def generate_pdf_report(weekly_data):
    logger.info("Generating PDF report")
    env = Environment(loader=FileSystemLoader('templates'))
    template_week = env.get_template('pdf_week.html')
    
    rendered_html = ""
    
    # Create the weekly trend plot first
    weekly_trend_plot = create_weekly_trend_plot(weekly_data)
    
    for week in weekly_data:
        logger.debug(f"Rendering template for week: {week['dates']}")
        rendered_html += template_week.render(
            weekday_dates=week['dates'],
            weekend_dates=week['dates'],
            weekday_peak_day=week['weekday_peak_day'],
            weekend_peak_day=week['weekend_peak_day'],
            weekday_peak_consumption=week['weekday_peak_consumption'],
            weekend_peak_consumption=week['weekend_peak_consumption'],
            weekday_total_consumption=week['weekday_total'],
            weekend_total_consumption=week['weekend_total'],
            weekday_plot=week['weekday_plot'],
            weekend_plot=week['weekend_plot'],
            weekly_trend_plot=weekly_trend_plot  # Add the trend plot to the template context
        )
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
        pdf_file = temp_file.name
        HTML(string=rendered_html).write_pdf(pdf_file)
        logger.info(f"PDF report generated at: {pdf_file}")
    return pdf_file

def get_dates_from_week_number(year: int, week: int, num_weeks: int = 4) -> List[Tuple[datetime, datetime]]:
    """
    Get start and end dates for specified week number(s).
    Weeks start on Monday and end on Sunday based on ISO week numbering.
    
    Args:
        year (int): The year.
        week (int): The ISO week number.
        num_weeks (int): Number of consecutive weeks to retrieve.

    Returns:
        List[Tuple[datetime, datetime]]: List of tuples containing (start_date, end_date) for each week.
    """
    week_ranges = []
    for i in range(num_weeks):
        current_week = week + i
        try:
            # Handle week overflow (e.g., week 53 in some years)
            iso_calendar = datetime.fromisocalendar(year, current_week, 1)
        except ValueError:
            # If the current_week exceeds the number of weeks in the year, reset to week 1 of the next year
            iso_calendar = datetime.fromisocalendar(year + 1, 1, 1)
            current_week = 1
        
        # Start date: Monday
        week_start = datetime.fromisocalendar(iso_calendar.year, current_week, 1)
        # End date: Sunday at 23:59:59
        week_end = datetime.fromisocalendar(iso_calendar.year, current_week, 7).replace(hour=23, minute=59, second=59)
        week_ranges.append((week_start, week_end))
        logger.info(f"Week {current_week} of {iso_calendar.year}: {week_start.strftime('%Y-%m-%d')} to {week_end.strftime('%Y-%m-%d')}")
    
    return week_ranges

# Function to get table names from Supabase
def get_table_names():
    logger.info("Fetching table names from Supabase")
    supabase_url = os.environ.get("SUPABASE_URL")
    supabase_key = os.environ.get("SUPABASE_KEY")
    
    if not supabase_url or not supabase_key:
        logger.error("Missing Supabase credentials")
        raise ValueError("Missing Supabase credentials")
        
    supabase = create_client(supabase_url, supabase_key)
    
    # Directly query the information schema for table names
    query = """
    SELECT tablename
    FROM pg_catalog.pg_tables
    WHERE schemaname = 'public';
    """
    
    try:
        response = supabase.sql(query).execute()
        if response.error:
            logger.error(f"Supabase query error: {response.error.message}")
            raise Exception(response.error.message)
        
        # Extract table names
        table_names = [table['tablename'] for table in response.data]
        logger.info(f"Found {len(table_names)} tables")
        logger.debug(f"Table names: {table_names}")
        return table_names
    except Exception as e:
        logger.error(f"Error fetching table names: {str(e)}")
        raise

@app.get("/tables", response_class=JSONResponse)
async def fetch_tables():
    logger.info("Handling /tables request")
    try:
        table_names = get_table_names()
        return {"tables": table_names}
    except Exception as e:
        logger.error(f"Error in /tables endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# API Endpoints
@app.get("/ping", status_code=200)
def ping():
    logger.debug("Ping request received")
    return {"status": "success", "message": "Pong"}

@app.get("/analysis", status_code=200)
def analysis():
    logger.info("Starting analysis endpoint")
    total_water_wasted, data_resampled, total_water_consumed, efficiency_percentage = analyze_data()
    logger.info(f"Analysis completed - Efficiency: {efficiency_percentage}%")
    return {
        "status": "success", 
        "total_water_wasted": total_water_wasted,
        "total_water_consumed": total_water_consumed,
        "efficiency_percentage": efficiency_percentage
    }

@app.get("/view_analysis", response_class=HTMLResponse)
async def view_analysis(
    request: Request, 
    table_name: str = "refugioAleman",
    window_size: int = 60, 
    start_week: int = None, 
    year: int = None
):
    logger.info(f"Starting view_analysis with params: table={table_name}, window={window_size}, year={year}, start_week={start_week}")
    try:
        # If week/year not provided, use current date
        if start_week is None or year is None:
            now = datetime.now()
            year = now.year
            start_week = now.isocalendar()[1]  # Get current ISO week number
            logger.info(f"Using current date: week={start_week}, year={year}")
            
        # Get week ranges
        week_ranges = get_dates_from_week_number(year, start_week)
        if not week_ranges:
            error_msg = f"No valid weeks found starting from week {start_week} of {year}."
            logger.error(error_msg)
            raise HTTPException(status_code=400, detail=error_msg)
        
        start_date = week_ranges[0][0]  # Start of first week
        end_date = week_ranges[-1][1]   # End of last week
        
        start_epoch = int(start_date.timestamp() * 1000)
        end_epoch = int(end_date.timestamp() * 1000)
        logger.info(f"Calculated epochs: start={start_epoch}, end={end_epoch}")

        logger.info("Calling analyze_data function")
        analysis_results = analyze_data(
            window_size=window_size, 
            start_epoch=start_epoch,
            end_epoch=end_epoch
        )
        
        if (analysis_results['weekday_total'] == 0 and 
            analysis_results['weekend_total'] == 0):
            # No data found for the selected period
            return templates.TemplateResponse("analysis.html", {
                "request": request,
                "error_message": f"No se encontraron datos para el período seleccionado: Semanas {start_week}-{start_week+3}, {year}",
                "plot_url": None,
                "window_size": window_size,
                "start_week": start_week,
                "year": year,
                "total_water_wasted_weekdays": 0,
                "efficiency_percentage_weekdays": 0,
                "total_water_consumed_weekdays": 0,
                "total_water_wasted_weekends": 0,
                "efficiency_percentage_weekends": 0,
                "total_water_consumed_weekends": 0
            })
        
        logger.info("Successfully got analysis results")
                                      
        combined_data = pd.concat([
            analysis_results['weekday_data'],
            analysis_results['weekend_data']
        ])
        
        logger.info("Creating plot")
        plot_file = create_plot(
            combined_data,
            f'Análisis Semanas {start_week}-{start_week+3} - {year}',
            f'{year}-W{start_week:02d}'
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
            "start_week": start_week,
            "year": year,
            "error_message": None
        })
    except Exception as e:
        logger.error(f"Error in view_analysis: {str(e)}", exc_info=True)
        return templates.TemplateResponse("analysis.html", {
            "request": request,
            "error_message": f"An error occurred: {str(e)}",
            "plot_url": None,
            "window_size": window_size,
            "start_week": start_week,
            "year": year
        })

@app.get("/generate_weekly_pdf", status_code=200)
async def generate_weekly_pdf(
    year: int,
    start_week: int,
    window_size: int = 60
):
    logger.info(f"Starting generate_weekly_pdf with params: year={year}, start_week={start_week}, window_size={window_size}")
    try:
        # Validate inputs
        if not 1 <= start_week <= 53:
            error_msg = f"Invalid week number: {start_week}. Must be between 1 and 53."
            logger.error(error_msg)
            raise HTTPException(status_code=400, detail=error_msg)
        
        # Get 4 week ranges
        week_ranges = get_dates_from_week_number(year, start_week, num_weeks=4)
        if not week_ranges:
            error_msg = f"No valid weeks found starting from week {start_week} of {year}."
            logger.error(error_msg)
            raise HTTPException(status_code=400, detail=error_msg)
        
        report = Report(
            title=f"Informe de Consumo de Agua - Semanas {start_week}-{start_week + 3}, {year}",
            place_name="Your Location"
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
                end_epoch=end_epoch
            )
            logger.info(f"Got analysis results for week {week_number}")
            
            # Create sections
            weekday_section = WeekdaySection(f"Semana {week_number} - Días Laborales")
            weekday_section.add_data("dates", f"{week_start.strftime('%d/%m/%Y')} - {week_end.strftime('%d/%m/%Y')}")
            weekday_section.add_data("peak_day", analysis_results['weekday_peak']['day'])
            weekday_section.add_data("peak_consumption", analysis_results['weekday_peak']['consumption'])
            weekday_section.add_data("total_consumption", analysis_results['weekday_total'])
            weekday_section.add_data("wasted", analysis_results['weekday_wasted'])
            weekday_section.add_data("efficiency", analysis_results['weekday_efficiency'])
            
            weekend_section = WeekendSection(f"Semana {week_number} - Fin de Semana")
            weekend_section.add_data("dates", f"{week_start.strftime('%d/%m/%Y')} - {week_end.strftime('%d/%m/%Y')}")
            weekend_section.add_data("peak_day", analysis_results['weekend_peak']['day'])
            weekend_section.add_data("peak_consumption", analysis_results['weekend_peak']['consumption'])
            weekend_section.add_data("total_consumption", analysis_results['weekend_total'])
            weekend_section.add_data("wasted", analysis_results['weekend_wasted'])
            weekend_section.add_data("efficiency", analysis_results['weekend_efficiency'])
            
            # Add data for comparison section
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
            
            # Update maximums
            max_consumption = max(max_consumption, 
                                  analysis_results['weekday_total'],
                                  analysis_results['weekend_total'])
            max_wasted = max(max_wasted, 
                             analysis_results['weekday_wasted'],
                             analysis_results['weekend_wasted'])
            
            # Create plots
            weekday_plot = create_plot(
                analysis_results['weekday_data'],
                'Días Laborales',
                week_start.strftime('%Y-%m-%d')
            )
            weekend_plot = create_plot(
                analysis_results['weekend_data'],
                'Fin de Semana',
                week_start.strftime('%Y-%m-%d')
            )
            
            weekday_section.add_data("plot", weekday_plot)
            weekend_section.add_data("plot", weekend_plot)
            
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
            filename=f'water_analysis_weeks_{start_week}-{start_week + 3}_{year}.pdf'
        )
        
    except Exception as e:
        logger.error(f"Error generating weekly PDF: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/test_comparison_plot")
async def test_comparison_plot():
    logger.info("Starting test comparison plot generation")
    test_weeks_data = [
        {
            'title': 'Semana 1',
            'weekday_consumption': 1000,
            'weekend_consumption': 800,
            'weekday_efficiency': 85,
            'weekend_efficiency': 80,
            'weekday_wasted': 150,
            'weekend_wasted': 160,
            'color': WEEK_COLORS['Semana 1']
        },
        {
            'title': 'Semana 2',
            'weekday_consumption': 1200,
            'weekend_consumption': 900,
            'weekday_efficiency': 90,
            'weekend_efficiency': 85,
            'weekday_wasted': 120,
            'weekend_wasted': 135,
            'color': WEEK_COLORS['Semana 2']
        },
        {
            'title': 'Semana 3',
            'weekday_consumption': 950,
            'weekend_consumption': 850,
            'weekday_efficiency': 88,
            'weekend_efficiency': 82,
            'weekday_wasted': 114,
            'weekend_wasted': 153,
            'color': WEEK_COLORS['Semana 3']
        }
    ]
    
    logger.debug("Creating test comparison plot with sample data")
    plot_file = create_weekly_trend_plot(test_weeks_data)
    logger.info(f"Test comparison plot created: {plot_file}")
    
    return FileResponse(
        plot_file,
        media_type='image/png',
        filename='test_comparison_plot.png'
    )

@app.get("/check_weeks", response_class=JSONResponse)
async def check_weeks(year: int):
    """Check which weeks have data in the specified year"""
    logger.info(f"Checking data availability for weeks in {year}")
    try:
        weeks_data = {}
        
        # Get the total number of ISO weeks in the year
        last_day = datetime(year, 12, 28)  # Use December 28th to ensure we're in the last week
        total_weeks = last_day.isocalendar()[1]
        
        # Check each week of the year
        for week in range(1, total_weeks + 1):
            week_start = datetime.fromisocalendar(year, week, 1)  # Monday
            week_end = datetime.fromisocalendar(year, week, 7)    # Sunday
            week_end = week_end.replace(hour=23, minute=59, second=59)
            
            start_epoch = int(week_start.timestamp() * 1000)
            end_epoch = int(week_end.timestamp() * 1000)
            
            # Query Supabase for data in this week's range
            supabase = create_client(os.environ.get("SUPABASE_URL"), os.environ.get("SUPABASE_KEY"))
            response = supabase.table("refugioAleman") \
                .select("*") \
                .gte("Category", start_epoch) \
                .lte("Category", end_epoch) \
                .execute()
                
            has_data = len(response.data) > 100
            
            weeks_data[week] = {
                'has_data': has_data,
                'start_date': week_start.strftime('%Y-%m-%d'),
                'end_date': week_end.strftime('%Y-%m-%d'),
                'records': len(response.data)
            }
            
            logger.debug(f"Week {week}: {has_data} ({len(response.data)} records)")
        
        return {"year": year, "weeks": weeks_data}
        
    except Exception as e:
        logger.error(f"Error checking weeks: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
