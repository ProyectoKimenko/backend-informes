# syntax=docker/dockerfile:1

# Comments are provided throughout this file to help you get started.
# If you need more help, visit the Dockerfile reference guide at
# https://docs.docker.com/go/dockerfile-reference/

# Want to help us make this template better? Share your feedback here: https://forms.gle/ybq9Krt8jtBL3iCk7

ARG PYTHON_VERSION=3.12.4
FROM python:${PYTHON_VERSION}-slim

# Install system dependencies including fonts
RUN apt-get update && apt-get install -y \
    libgirepository1.0-dev \
    libcairo2-dev \
    gir1.2-gtk-3.0 \
    libpango1.0-dev \
    libwebp-dev \
    python3-cffi \
    python3-brotli \
    libpangoft2-1.0-0 \
    fonts-liberation \
    libfontconfig1 \
    libfreetype6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories with proper permissions
RUN mkdir -p /app/static/images && \
    mkdir -p /app/matplotlib_cache && \
    chmod -R 777 /app/static && \
    chmod -R 777 /app/matplotlib_cache && \
    chown -R root:root /app

# Set environment variables
ENV MPLCONFIGDIR=/app/matplotlib_cache
ENV PYTHONPATH=/app
ENV PORT=8000
ENV PYTHONUNBUFFERED=1

# Expose the port that the application listens on
EXPOSE 8000

# Healthcheck contra /ping (sin curl: usa python, siempre presente). Permite que
# el orquestador detecte un contenedor vivo-pero-no-sirviendo.
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/ping', timeout=4).status==200 else 1)"

# Exec-form para que SIGTERM llegue a uvicorn (shutdown limpio: los BackgroundTasks
# en curso terminan en vez de cortarse). Antes shell-form -> uvicorn corría como
# hijo de /bin/sh y no recibía la señal. Puerto fijo 8000 (ENV PORT).
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--timeout-graceful-shutdown", "30"]
