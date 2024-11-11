from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import matplotlib.pyplot as plt
import os
from analysis import analyze_data
from pdf_generator import generate_pdf_report
import tempfile
from fastapi.responses import HTMLResponse
from fastapi.requests import Request
from fastapi.templating import Jinja2Templates
from scrape import fetch_page_info

from typing import List, Dict

app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory=tempfile.gettempdir()), name="static")

@app.get("/ping", status_code=200)
def ping():
    return {"status": "success", "message": "Pong"}

@app.get("/analysis", status_code=200)
def analysis():
    data, data_resampled = analyze_data()
    return {"status": "success", "Total Water Wasted": data, "data_resampled": data_resampled}

@app.get("/generate_pdf", status_code=200)
async def generate_pdf(window_size: int = 60):
    try:
        total_water_wasted, data_resampled, efficiency_percentage, total_water_consumed = analyze_data(window_size)

        plot_file = create_plot(data_resampled)
        pdf_file = generate_pdf_report(plot_file, total_water_wasted, efficiency_percentage, total_water_consumed)
        os.remove(plot_file)
        
        return FileResponse(pdf_file, media_type='application/pdf', filename='water_analysis.pdf')
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

@app.get("/view_analysis", response_class=HTMLResponse)
async def view_analysis(request: Request, window_size: int = 60):
    try:
        total_water_wasted, data_resampled, total_water_consumed, efficiency_percentage = analyze_data(window_size)
        
        # Creating the plot for display
        plot_file = create_plot(data_resampled)
        
        # Generate the URL to access the plot image
        plot_url = f"/static/{os.path.basename(plot_file)}"
        
        return templates.TemplateResponse("analysis.html", {
            "request": request,
            "total_water_wasted": total_water_wasted,
            "efficiency_percentage": efficiency_percentage,
            "total_water_consumed": total_water_consumed,
            "plot_url": plot_url,
            "window_size": window_size
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# @app.get("/view_analysis", response_class=HTMLResponse)
# async def view_analysis(request: Request, window_ranges: List[Dict[str, str]] = Query(None)):
#     try:
#         if not window_ranges:
#             # Use default window size if no ranges are provided
#             window_ranges = [
#                 {'window_size': 60, 'timestamp_start': '2024-08-28 06:00:00', 'timestamp_end': '2024-08-28 18:00:00'}
#             ]

#         # Parse window_ranges to be passed into the analyze_data function
#         total_water_wasted, data_resampled, total_water_consumed, efficiency_percentage = analyze_data(window_ranges)

#         # Creating the plot for display
#         plot_file = create_plot(data_resampled)

#         # Generate the URL to access the plot image
#         plot_url = f"/static/{os.path.basename(plot_file)}"
        
#         return templates.TemplateResponse("analysis.html", {
#             "request": request,
#             "total_water_wasted": total_water_wasted,
#             "efficiency_percentage": efficiency_percentage,
#             "total_water_consumed": total_water_consumed,
#             "plot_url": plot_url,
#             "window_ranges": window_ranges
#         })
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))




@app.get("/fetch_page_info", status_code=200)
async def get_page_info(url: str):
    print(f"Fetching page info for URL: {url}")
    title = fetch_page_info(url)  # Call the internal function to fetch the title
    return {"status": "success", "title": title}


def create_plot(data):
    data_resampled_hour = data.resample('30T').mean()
    plt.figure(figsize=(10, 6))
    plt.plot(data_resampled_hour.index, data_resampled_hour['RollingMin'], label='Perdida de agua')
    plt.plot(data_resampled_hour.index, data_resampled_hour['Flow rate'], label='Flujo de agua')
    plt.title('Análisis de agua')
    plt.ylabel('Cantidad (litros)')
    plt.legend()

    with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as temp_file:
        plt.savefig(temp_file.name)
        plt.close()
    
    return temp_file.name

