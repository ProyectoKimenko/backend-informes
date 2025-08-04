"""
Configuración centralizada de logging para la aplicación de informes.
Simplificada para incluir solo información útil.
"""
import logging
import sys
import os
from datetime import datetime

def setup_logger(name: str = None, level: str = None) -> logging.Logger:
    """
    Configura un logger simplificado con formato limpio.
    
    Args:
        name: Nombre del logger (por defecto usa el módulo que lo llama)
        level: Nivel de logging (INFO por defecto, DEBUG para desarrollo)
    
    Returns:
        Logger configurado
    """
    # Obtener nivel de logging del entorno o usar INFO por defecto
    log_level = level or os.getenv('LOG_LEVEL', 'INFO').upper()
    
    # Crear logger
    if name is None:
        name = __name__
    
    logger = logging.getLogger(name)
    
    # Evitar configurar múltiples veces
    if logger.handlers:
        return logger
    
    logger.setLevel(getattr(logging, log_level))
    
    # Formato simplificado - solo tiempo, nivel y mensaje
    formatter = logging.Formatter(
        fmt='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # Handler para consola
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Evitar propagación para no duplicar logs
    logger.propagate = False
    
    return logger

def log_request(logger: logging.Logger, method: str, endpoint: str, duration_ms: float = None):
    """Helper para loggear requests HTTP de forma consistente"""
    if duration_ms:
        logger.info(f"{method} {endpoint} - {duration_ms:.0f}ms")
    else:
        logger.info(f"{method} {endpoint}")

def log_error(logger: logging.Logger, operation: str, error: Exception):
    """Helper para loggear errores de forma consistente"""
    logger.error(f"Error in {operation}: {str(error)}")

def log_data_operation(logger: logging.Logger, operation: str, count: int, place_id: int = None):
    """Helper para loggear operaciones de datos de forma concisa"""
    place_info = f" (place {place_id})" if place_id else ""
    logger.info(f"{operation}: {count} records{place_info}")

def log_startup(logger: logging.Logger, app_name: str = "Backend Informes"):
    """Log de inicio de aplicación"""
    env = os.getenv('ENV', 'development')
    log_level = os.getenv('LOG_LEVEL', 'INFO')
    logger.info(f"{app_name} started - Environment: {env}, Log Level: {log_level}")