from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.requests import Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from jinja2 import Environment, FileSystemLoader
from weasyprint import HTML
from typing import List, Dict
from datetime import datetime, timedelta
import calendar
import colorsys
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import tempfile
from supabase import create_client

from src.analysis import analyze_data
from src.scrape import fetch_page_info
from src.report import Report
from src.report_sections import WeekdaySection, WeekendSection, ComparisonSection

# Initialize FastAPI app and templates
app = FastAPI()
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
            if np.sum(segment_mask) > 1:  # Ensure more than one point
                segments.append((index[segment_mask], values[segment_mask]))
        start_idx = end_idx
    
    # Handle the last segment
    segment_mask = mask.copy()
    segment_mask[:start_idx] = False
    if np.sum(segment_mask) > 1:
        segments.append((index[segment_mask], values[segment_mask]))
    
    return segments

def create_plot(data, title_prefix, date_str):
    plt.figure(figsize=(12, 6))
    plt.style.use('default')
    
    # Create masks for valid data
    flow_mask = ~np.isnan(data['Flow rate'])
    min_mask = ~np.isnan(data['RollingMin'])
    min_mask = ~np.isnan(data['RollingMin'])
    
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
    return temp_file.name

def create_monthly_trend_plot(weeks_data):
    # Extract data
    weeks = [f"Semana {i + 1}" for i in range(len(weeks_data))]
    weekday_consumptions = [week['weekday_consumption'] for week in weeks_data]
    weekend_consumptions = [week['weekend_consumption'] for week in weeks_data]
    weekday_efficiencies = [week['weekday_efficiency'] for week in weeks_data]
    weekend_efficiencies = [week['weekend_efficiency'] for week in weeks_data]
    weekday_losses = [week['weekday_wasted'] for week in weeks_data]
    weekend_losses = [week['weekend_wasted'] for week in weeks_data]
    colors = [week['color'] for week in weeks_data]

    # Create figure with larger size
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), height_ratios=[1, 1])
    fig.suptitle('Tendencias Mensuales de Consumo de Agua', fontsize=20, y=0.95)

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
    return temp_file.name

def generate_pdf_report(weekly_data):
    env = Environment(loader=FileSystemLoader('templates'))
    template_week = env.get_template('pdf_week.html')
    
    rendered_html = ""
    for week in weekly_data:
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
            weekend_plot=week['weekend_plot']
        )
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
        pdf_file = temp_file.name
        HTML(string=rendered_html).write_pdf(pdf_file)
    return pdf_file

def get_weeks_of_month(year: int, month: int) -> List[tuple]:
    first_day = datetime(year, month, 1)
    _, num_days = calendar.monthrange(year, month)
    last_day = datetime(year, month, num_days)
    
    current_day = first_day
    while current_day.weekday() != 0:
        current_day += timedelta(days=1)
    
    week_ranges = []
    while current_day + timedelta(days=6) <= last_day:
        week_start = current_day
        week_end = current_day + timedelta(days=6)
        if week_start.month == month and week_end.month == month:
            week_ranges.append((week_start, week_end))
        current_day += timedelta(days=7)
    
    return week_ranges

# Function to get table names from Supabase
def get_table_names():
    supabase_url = os.environ.get("SUPABASE_URL")
    supabase_key = os.environ.get("SUPABASE_KEY")
    supabase = create_client(supabase_url, supabase_key)
    
    # Directly query the information schema for table names
    query = """
    SELECT tablename
    FROM pg_catalog.pg_tables
    WHERE schemaname = 'public';
    """
    
    response = supabase.sql(query).execute()
    if response.error:
        raise Exception(response.error.message)
    
    # Extract table names
    table_names = [table['tablename'] for table in response.data]
    return table_names

@app.get("/tables", response_class=JSONResponse)
async def fetch_tables():
    try:
        table_names = get_table_names()
        return {"tables": table_names}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# API Endpoints
@app.get("/ping", status_code=200)
def ping():
    return {"status": "success", "message": "Pong"}

@app.get("/analysis", status_code=200)
def analysis():
    total_water_wasted, data_resampled, total_water_consumed, efficiency_percentage = analyze_data()
    return {
        "status": "success", 
        "total_water_wasted": total_water_wasted,
        "total_water_consumed": total_water_consumed,
        "efficiency_percentage": efficiency_percentage
    }

@app.get("/view_analysis", response_class=HTMLResponse)
async def view_analysis(
    request: Request, 
    table_name: str = "refugioAleman",  # Default value set here
    window_size: int = 60, 
    month: int = None, 
    year: int = None
):
    try:
        # If month/year not provided, use current date
        if month is None or year is None:
            now = datetime.now()
            month = now.month
            year = now.year
            
        # Calculate epoch timestamps for start/end of month
        start_date = datetime(year, month, 1)
        _, last_day = calendar.monthrange(year, month)
        end_date = datetime(year, month, last_day, 23, 59, 59)
        
        start_epoch = int(datetime.timestamp(start_date) * 1000)
        end_epoch = int(datetime.timestamp(end_date) * 1000)

        analysis_results = analyze_data(
            table_name=table_name,
            window_size=window_size, 
            start_epoch=start_epoch,
            end_epoch=end_epoch
        )
                                      
        combined_data = pd.concat([
            analysis_results['weekday_data'],
            analysis_results['weekend_data']
        ])
        
        plot_file = create_plot(
            combined_data,
            f'Análisis {calendar.month_name[month]} {year}',
            f'{year}-{month:02d}'
        )
        plot_url = f"/static/{os.path.basename(plot_file)}"
        
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
            "month": month,
            "year": year,
            "error_message": None  # No error
        })
    except Exception as e:
        # Pass the error message to the template
        return templates.TemplateResponse("analysis.html", {
            "request": request,
            "error_message": f"An error occurred: {str(e)}",
            "plot_url": None,  # No plot to display
            "window_size": window_size,
            "month": month,
            "year": year
        })

@app.get("/generate_monthly_pdf", status_code=200)
async def generate_monthly_pdf(month: int, year: int, window_size: int = 60):
    try:
        report = Report(
            title=f"Informe Mensual de Consumo de Agua - {calendar.month_name[month]} {year}",
            place_name="Your Location"
        )
        
        week_ranges = get_weeks_of_month(year, month)
        weeks_data = []
        max_consumption = 0
        max_wasted = 0
        week_number = 1
        
        for start_date, end_date in week_ranges:
            start_epoch = int(datetime.timestamp(start_date) * 1000)
            end_epoch = int(datetime.timestamp(end_date) * 1000)
            
            analysis_results = analyze_data(
                window_size=window_size,
                start_epoch=start_epoch,
                end_epoch=end_epoch
            )
            
            # Create sections
            weekday_section = WeekdaySection(f"Semana {week_number} - Días Laborales")
            weekday_section.add_data("dates", f"{start_date.strftime('%d/%m/%Y')} - {end_date.strftime('%d/%m/%Y')}")
            weekday_section.add_data("peak_day", analysis_results['weekday_peak']['day'])
            weekday_section.add_data("peak_consumption", analysis_results['weekday_peak']['consumption'])
            weekday_section.add_data("total_consumption", analysis_results['weekday_total'])
            weekday_section.add_data("wasted", analysis_results['weekday_wasted'])
            weekday_section.add_data("efficiency", analysis_results['weekday_efficiency'])
            
            weekend_section = WeekendSection(f"Semana {week_number} - Fin de Semana")
            weekend_section.add_data("dates", f"{start_date.strftime('%d/%m/%Y')} - {end_date.strftime('%d/%m/%Y')}")
            weekend_section.add_data("peak_day", analysis_results['weekend_peak']['day'])
            weekend_section.add_data("peak_consumption", analysis_results['weekend_peak']['consumption'])
            weekend_section.add_data("total_consumption", analysis_results['weekend_total'])
            weekend_section.add_data("wasted", analysis_results['weekend_wasted'])
            weekend_section.add_data("efficiency", analysis_results['weekend_efficiency'])
            
            week_data = {
                'title': f"Semana {week_number}",
                'dates': f"{start_date.strftime('%d/%m/%Y')} - {end_date.strftime('%d/%m/%Y')}",
                'weekday_consumption': analysis_results['weekday_total'],
                'weekend_consumption': analysis_results['weekend_total'],
                'weekday_efficiency': analysis_results['weekday_efficiency'],
                'weekend_efficiency': analysis_results['weekend_efficiency'],
                'weekday_wasted': analysis_results['weekday_wasted'],
                'weekend_wasted': analysis_results['weekend_wasted'],
                'color': WEEK_COLORS.get(f'Semana {week_number}', '#1fa9c9')
            }
            weeks_data.append(week_data)
            
            max_consumption = max(max_consumption, 
                                analysis_results['weekday_total'],
                                analysis_results['weekend_total'])
            max_wasted = max(max_wasted, 
                           analysis_results['weekday_wasted'],
                           analysis_results['weekend_wasted'])
            
            weekday_plot = create_plot(
                analysis_results['weekday_data'],
                'Días Laborales',
                start_date.strftime('%Y-%m-%d')
            )
            weekend_plot = create_plot(
                analysis_results['weekend_data'],
                'Fin de Semana',
                start_date.strftime('%Y-%m-%d')
            )
            
            weekday_section.add_data("plot", weekday_plot)
            weekend_section.add_data("plot", weekend_plot)
            
            report.add_section(weekday_section)
            report.add_section(weekend_section)
            
            week_number += 1

        comparison_section = ComparisonSection("Comparación Mensual")
        comparison_section.add_data('weeks', weeks_data)
        comparison_section.add_data('max_consumption', max_consumption)
        comparison_section.add_data('max_wasted', max_wasted)
        
        monthly_trend_plot = create_monthly_trend_plot(weeks_data)
        comparison_section.add_data('monthly_trend_plot', monthly_trend_plot)
        
        report.add_section(comparison_section)
        
        pdf_file = report.render()
        
        for section in report.sections:
            if os.path.exists(section.data.get('plot', '')):
                os.remove(section.data['plot'])
        
        return FileResponse(
            pdf_file,
            media_type='application/pdf',
            filename=f'water_analysis_{year}_{month}.pdf'
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/test_comparison_plot")
async def test_comparison_plot():
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
    
    plot_file = create_monthly_trend_plot(test_weeks_data)
    
    return FileResponse(
        plot_file,
        media_type='image/png',
        filename='test_comparison_plot.png'
    )
