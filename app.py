from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
import matplotlib.pyplot as plt
import os
from analysis import analyze_data
from pdf_generator import generate_pdf_report
import tempfile

app = FastAPI()

@app.get("/ping", status_code=200)
def ping():
    return {"status": "success", "message": "Pong"}

@app.get("/analysis", status_code=200)
def analysis():
    data, data_resampled = analyze_data()
    return {"status": "success", "Total Water Wasted": data, "data_resampled": data_resampled}

@app.get("/generate_pdf", status_code=200)
async def generate_pdf():
    try:
        total_water_wasted, data_resampled, efficiency_percentage, total_water_consumed = analyze_data()

        plot_file = create_plot(data_resampled)
        pdf_file = generate_pdf_report(plot_file, total_water_wasted, efficiency_percentage, total_water_consumed)
        os.remove(plot_file)
        
        return FileResponse(pdf_file, media_type='application/pdf', filename='water_analysis.pdf')
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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

