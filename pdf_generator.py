# from jinja2 import Template
# from weasyprint import HTML
# import tempfile

# def generate_pdf_report(plot_file, total_water_wasted, total_water_consumed, efficiency_percentage):
#     template = Template("""
#         <html>
#             <head>
#                 <title>Reporte de Análisis de Agua</title>
#                 <link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
#             </head>
#             <body>
#                 <div class="container mx-auto">
#                     <h1 class="text-center text-blue-500 text-2xl mb-8">Reporte de Análisis de Agua</h1>
#                     <div class="flex flex-col gap-8">
#                         <div class="flex-1">
#                             <img src="file://{{ plot_file }}" alt="Gráfico de análisis de agua" class="rounded-lg shadow-lg">
#                         </div>
#                         <div class="flex-1 bg-blue-500 text-white p-4 rounded-lg shadow-lg flex flex-col justify-center">
#                             <h2 class="text-center text-xl mb-4">Resumen de Pérdidas de Agua</h2>
#                             <p class="text-center text-lg mb-2">Total litros perdidos por ineficiencias: <strong>{{ total_water_wasted }} litros</strong></p>
#                             <p class="text-center text-lg mb-2">Total litros consumidos en el mismo período: <strong>{{ total_water_consumed }} litros</strong></p>
#                             <p class="text-center text-lg mb-2">Porcentaje promedio de pérdidas: <strong>{{ efficiency_percentage }}%</strong></p>
#                             <p class="text-center text-sm mt-4">NOTA: Los días no incluidos en la comparación corresponden a jornadas sin pérdidas de agua detectadas.</p>
#                         </div>
#                     </div>
#                 </div>
#             </body>
#         </html>
#     """)
#     rendered_html = template.render(total_water_wasted=total_water_wasted, plot_file=plot_file, total_water_consumed=total_water_consumed, efficiency_percentage=efficiency_percentage)
    
#     with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
#         pdf_file = temp_file.name
#         HTML(string=rendered_html).write_pdf(pdf_file)
#     return pdf_file

from jinja2 import Template
from weasyprint import HTML
import tempfile

def generate_pdf_report(plot_file, total_water_wasted, total_water_consumed, efficiency_percentage):
    template = Template("""
        <html>
        <head>
            <title>Reporte de Análisis de Agua</title>
            <link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
        </head>
        <body>
            <div class="container mx-auto">
                <h1 class="text-center text-blue-500 text-2xl mb-8">Reporte de Análisis de Agua</h1>
                <div class="flex flex-col gap-8">
                    <div class="flex-1">
                        <img src="file://{{ plot_file }}" alt="Gráfico de análisis de agua" class="rounded-lg shadow-lg">
                    </div>
                    <div class="flex-1 bg-blue-500 text-white p-8 rounded-lg shadow-lg flex flex-col justify-center">
                        <h2 class="text-center text-xl mb-4">Resumen de Pérdidas de Agua</h2>
                        <p class="text-center text-lg mb-2">Total litros perdidos por ineficiencias: <strong>{{ total_water_wasted }} litros</strong></p>
                        <p class="text-center text-lg mb-2">Total litros consumidos en el mismo período: <strong>{{ total_water_consumed }} litros</strong></p>
                        <p class="text-center text-lg mb-2">Porcentaje promedio de pérdidas: <strong>{{ efficiency_percentage }}%</strong></p>
                        <p class="text-center text-sm mt-4">NOTA: Los días no incluidos en la comparación corresponden a jornadas sin pérdidas de agua detectadas.</p>
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
