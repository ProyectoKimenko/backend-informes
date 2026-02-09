# 💧 Backend de Informes y Desagregación de Agua

Backend desarrollado en **FastAPI** para el análisis, desagregación automática y reporte de consumo de agua. Este sistema procesa datos de medidores en tiempo real, identifica patrones de consumo mediante Machine Learning (Gaussian Mixture Models) y genera reportes PDF detallados.

## 🚀 Características Principales

* **🤖 Desagregación Automática:** Algoritmos de ML que identifican eventos de consumo (duchas, grifos, lavadoras) y los clasifican por caudal y duración.
* **⚡ Procesamiento Asíncrono:** Arquitectura basada en **Celery + Redis** para procesar grandes volúmenes de datos en segundo plano sin bloquear la API.
* **⏱️ Scheduling Inteligente:** Tareas programadas con **Celery Beat** que analizan automáticamente todos los lugares activos cada hora.
* **📄 Reportes PDF:** Motor de renderizado (WeasyPrint) para crear informes semanales con gráficos de tendencias, eficiencia y comparativas.
* **☁️ Integración Supabase:** Conexión nativa para lectura de telemetría y almacenamiento persistente de resultados.

## 🛠️ Stack Tecnológico

* **Core:** Python 3.12, FastAPI, Uvicorn.
* **Data & ML:** Pandas, NumPy, Scikit-Learn.
* **Async & Queue:** Celery, Redis.
* **Reporting:** WeasyPrint, Jinja2, Matplotlib.
* **Infraestructura:** Docker, Docker Compose.

---

## ⚙️ Configuración e Instalación

Sigue estos pasos para levantar el proyecto en un entorno local o servidor.

### 1. Prerrequisitos

* Docker y Docker Compose instalados.
* Proyecto en Supabase con la tabla `measurements_realtime` poblada con datos de sensores.

### 2. Variables de Entorno

Crea un archivo llamado `.env` en la raíz del proyecto (`backend-informes/.env`) con el siguiente contenido:

```env
SUPABASE_URL=tu_url_de_supabase_aqui
SUPABASE_KEY=tu_service_role_key_o_anon_key_aqui
# URL de conexión a Redis (valor por defecto para Docker Compose)
REDIS_URL=redis://redis:6379/0

```

### 3. Ejecución con Docker

Levanta la orquestación completa de servicios (API, Worker, Scheduler y Redis):

```bash
docker compose up --build

```

* El flag `--build` asegura que se instalen las dependencias nuevas si hubo cambios.

---

## 🔌 Uso de la API

Una vez levantado, la documentación interactiva (Swagger UI) está disponible en:
👉 `http://localhost:8000/docs`

### Endpoints Principales

#### ⚙️ Jobs y Procesos

* **POST** `/jobs/disaggregate/{place_id}`
* Dispara manualmente el análisis de desagregación para un lugar específico.
* *Nota:* El sistema (Scheduler) también ejecuta esto automáticamente en segundo plano.



#### 📊 Análisis de Datos

* **GET** `/analysis`
* Devuelve métricas de eficiencia, consumo total y series temporales en formato JSON para visualización en frontend.
* **Parámetros:** `place_id`, `start_week`, `end_week`, `year`.


* **GET** `/check_weeks`
* Verifica qué semanas de un año específico contienen datos disponibles.



#### 📄 Generación de Reportes

* **GET** `/generate_weekly_pdf`
* Genera y descarga un informe PDF completo con análisis de días laborales vs. fines de semana.
* **Parámetros:** `place_id`, `year`, `start_week`, `end_week`.



---

## 🏗️ Arquitectura del Proyecto

Estructura de carpetas clave:

```text
backend-informes/
├── app.py                   # API Gateway (FastAPI) y definición de rutas
├── compose.yaml             # Orquestación de servicios Docker
├── worker/                  # Lógica de tareas en segundo plano
│   ├── celery_app.py        # Configuración de Celery y Beat (Cron)
│   └── tasks.py             # Tareas definidas (Desagregación, Scraping)
├── pipeline/
│   └── disaggregator.py     # Núcleo del algoritmo ML (GMM)
├── services/
│   └── supabase_service.py  # Capa de acceso a datos (Lectura/Escritura)
├── templates/               # Plantillas HTML (Jinja2) para los PDFs
└── requirements.txt         # Dependencias del proyecto

```

