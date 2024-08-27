from jinja2 import Template
from weasyprint import HTML
import tempfile

def generate_pdf_report(plot_file, total_water_wasted, total_water_consumed, efficiency_percentage):
    template = Template("""
    <html>
    <head>
        <title>Reporte de Análisis de Agua</title>
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 0;
                padding: 0;
                background-color: #f4f4f4;
                color: #333;
            }
            .container {
                width: 80%;
                margin: 40px auto;
                padding: 20px;
                background-color: #fff;
                border-radius: 12px;
                box-shadow: 0 0 15px rgba(0, 0, 0, 0.1);
            }
            h1 {
                text-align: center;
                color: #00AEEF;
                font-size: 28px;
                margin-bottom: 30px;
            }
            .content {
                display: flex;
                justify-content: space-between;
                gap: 20px;
            }
            .left {
                flex: 1;
            }
            .right {
                flex: 1;
                background-color: #00AEEF;
                color: white;
                padding: 20px;
                border-radius: 10px;
                display: flex;
                flex-direction: column;
                justify-content: center;
                box-shadow: 0 0 15px rgba(0, 0, 0, 0.1);
            }
            .right h2 {
                text-align: center;
                font-size: 20px;
                margin-bottom: 20px;
            }
            .right p {
                font-size: 20px;
                line-height: 1.6;
                text-align: center;
                margin: 10px 0;
            }
            .right .note {
                font-size: 14px;
                margin-top: 20px;
                color: #f4f4f4;
            }
            img {
                display: block;
                max-width: 100%;
                height: auto;
                border-radius: 10px;
                box-shadow: 0 0 15px rgba(0, 0, 0, 0.1);
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Reporte de Análisis de Agua</h1>
            <div class="content">
                <div class="left">
                    <img src="file://{{ plot_file }}" alt="Gráfico de análisis de agua">
                </div>
                <div class="right">
                    <h2>Resumen de Pérdidas de Agua</h2>
                    <p>Total litros perdidos por ineficiencias: <strong>{{ total_water_wasted }} litros</strong></p>
                    <p>Total litros consumidos en el mismo período: <strong>{{ total_water_consumed }} litros</strong></p>
                    <p>Porcentaje promedio de pérdidas: <strong>{{ efficiency_percentage }}%</strong></p>
                    <p class="note">NOTA: Los días no incluidos en la comparación corresponden a jornadas sin pérdidas de agua detectadas.</p>
                </div>
            </div>
        </div>
    </body>
    </html>

    """)
    rendered_html = template.render(total_water_wasted=total_water_wasted, plot_file=plot_file, total_water_consumed=total_water_consumed, efficiency_percentage=efficiency_percentage)
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
        pdf_file = temp_file.name
        HTML(string=rendered_html).write_pdf(pdf_file)
    return pdf_file
