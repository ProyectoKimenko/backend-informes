# Configuración de Logging Simplificado

## Variables de Entorno

Para controlar el nivel de logging de tu aplicación, puedes usar estas variables de entorno:

```bash
# Nivel de logging (por defecto: INFO)
export LOG_LEVEL=INFO

# Entorno de aplicación (por defecto: development)
export ENV=development
```

## Niveles de Logging Disponibles

### `DEBUG`
- Información muy detallada para debugging
- Incluye detalles técnicos de procesamiento de datos
- **Recomendado para**: Desarrollo local cuando necesitas troubleshooting

### `INFO` (por defecto)
- Eventos importantes de la aplicación
- Requests HTTP, operaciones de datos, generación de reportes
- **Recomendado para**: Desarrollo normal y staging

### `WARNING`
- Solo advertencias y errores
- **Recomendado para**: Producción con logs mínimos

### `ERROR`
- Solo errores críticos
- **Recomendado para**: Producción en ambientes con alta carga

## Ejemplos de Configuración

### Desarrollo Local (logs detallados)
```bash
export LOG_LEVEL=DEBUG
export ENV=development
python app.py
```

### Producción (logs mínimos)
```bash
export LOG_LEVEL=WARNING
export ENV=production
python app.py
```

### Docker Compose
```yaml
environment:
  - LOG_LEVEL=INFO
  - ENV=production
```

## Qué se Redujo en los Logs

### ❌ Antes (verboso):
```
2024-01-15 10:30:45 - INFO - Input data shape: (15432, 3)
2024-01-15 10:30:45 - INFO - Timestamp range in raw data: 1705123200000 to 1705987200000
2024-01-15 10:30:45 - INFO - First few raw timestamps: [1705123200000, 1705123260000, 1705123320000]
2024-01-15 10:30:45 - INFO - After start_epoch filter (1705123200000): 15432 -> 15432 records
2024-01-15 10:30:45 - INFO - After end_epoch filter (1705987200000): 15432 -> 15432 records
2024-01-15 10:30:46 - INFO - After timestamp conversion, datetime range: 2024-01-13 00:00:00+00:00 to 2024-01-22 00:00:00+00:00
```

### ✅ Ahora (conciso):
```
10:30:45 [INFO] Analysis requested: 2024-01-13 to 2024-01-22
10:30:46 [INFO] Analysis complete: 8654 weekday, 6778 weekend records
10:30:46 [INFO] Report generated successfully
```

## Formato de Logs

El nuevo formato es más limpio:
- **Tiempo**: Solo hora:minuto:segundo (no fecha completa)
- **Nivel**: Entre corchetes [INFO], [ERROR], etc.
- **Mensaje**: Conciso y útil

## Logging Automático de HTTP Requests

Los requests HTTP ahora se loggean automáticamente:
```
10:30:45 [INFO] GET /api/analysis - 234ms
10:30:46 [INFO] POST /api/report/weekly - 1850ms
10:30:47 [WARNING] GET /api/data/invalid - HTTP 404
```